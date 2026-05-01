"""Deterministic pre-execution SQL validator (Phase E).

Replaces the v1 LLM-as-judge validator (``sql_validator.validate_sql``)
with a call to a local stdio MCP server that performs:

  parse → policy → scope → BigQuery dry_run → cost gate

All deterministic. No LLM. ~150 ms per validation, $0 per validation
(BigQuery dry_run is free). Catches the failure modes the v1 judge
would miss because the v1 judge only sees the SQL text and pattern-
matches against a taxonomy — the dry_run actually consults BigQuery's
type system, schema catalog, and cost estimator.

The wrapper here owns:
1. Mapping the MCP server's verdict envelope to our ``SqlValidationVerdict``.
2. The "never raise" contract — on any failure (MCP unreachable, malformed
   verdict, timeout) returns None and the orchestrator proceeds to
   ``backend.execute`` exactly as today (no regression — the kill-switch
   ``maximum_bytes_billed`` on the BigQuery job stops runaway scans).
3. The block-only-when-LOINC-is-clean rule from v1 — we never escalate
   a LIKE-fallback to a hard block; that path stays as ``warn`` so the
   critic's self-healing retry can fix it.
"""

from __future__ import annotations

import logging
from typing import Any

from src.conversational.health_evidence.mcp_client import McpClient
from src.conversational.models import SqlValidationVerdict
from src.conversational.sql_fastpath import SqlFastpathQuery

logger = logging.getLogger(__name__)


_DEFAULT_MAX_USD = 0.50
_DEFAULT_MAX_BYTES = 10 * 1024**3  # 10 GiB
_DEFAULT_TIMEOUT = 5.0  # MCP round-trip + dry_run; comfortable headroom


def validate_sql_deterministic(
    query: SqlFastpathQuery,
    *,
    mcp_client: McpClient,
    fallback_warning: str | None = None,
    resolved_itemids: list[int] | None = None,
    max_usd: float = _DEFAULT_MAX_USD,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: float = _DEFAULT_TIMEOUT,
) -> SqlValidationVerdict | None:
    """Validate a compiled SQL query against the bq-validator MCP.

    Returns ``SqlValidationVerdict`` on success, ``None`` on any failure.
    NEVER raises.

    Decision rules:
      * Server returns ``ok=True`` ⇒ verdict="pass" (with cost telemetry).
      * Server returns ``ok=False`` AND we're in a LIKE-fallback path
        (``fallback_warning`` set OR ``resolved_itemids`` is None) ⇒
        downgrade to verdict="warn". The critic's self-healing retry is
        the right escalation for those cases — blocking would remove
        that path.
      * Server returns ``ok=False`` AND LOINC grounding succeeded ⇒
        verdict="block". The user gets a no-execute short-circuit.
    """
    try:
        envelope = mcp_client.call_tool(
            "validate_sql",
            {
                "sql": query.sql,
                "params": list(query.params or []),
                "max_usd": max_usd,
                "max_bytes": max_bytes,
            },
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — defense in depth; client should never raise
        logger.warning(
            "validate_sql_deterministic: MCP call raised: %s", exc,
        )
        return None

    if envelope.get("status") != "ok":
        # Infrastructure failure (MCP unreachable, etc.). Proceed without
        # validation rather than blocking the user.
        logger.info(
            "validate_sql_deterministic: MCP unavailable (%s); proceeding",
            envelope.get("error"),
        )
        return None

    results = envelope.get("results") or []
    if not results or not isinstance(results[0], dict):
        logger.warning(
            "validate_sql_deterministic: malformed envelope: %r", envelope,
        )
        return None
    raw = results[0]

    return _to_verdict(raw, fallback_warning, resolved_itemids)


def _to_verdict(
    raw: dict[str, Any],
    fallback_warning: str | None,
    resolved_itemids: list[int] | None,
) -> SqlValidationVerdict | None:
    """Map a server verdict dict to ``SqlValidationVerdict``.

    Applies the LIKE-fallback downgrade rule before assigning the
    final verdict tier.
    """
    ok = bool(raw.get("ok"))
    stage = raw.get("stage", "unknown")
    bytes_processed = raw.get("bytes_processed")
    estimated_usd = raw.get("estimated_usd")
    error = raw.get("error")

    if ok:
        return SqlValidationVerdict(
            verdict="pass",
            concern=None,
            reference_used=f"dry_run:{stage}",
            bytes_processed=int(bytes_processed) if bytes_processed is not None else None,
            estimated_usd=float(estimated_usd) if estimated_usd is not None else None,
        )

    # LIKE-fallback downgrade rule: don't hard-block when LOINC grounding
    # didn't happen — the critic's self-healing retry should get a shot.
    is_fallback_path = bool(fallback_warning) or not resolved_itemids
    final_verdict = "warn" if is_fallback_path else "block"

    return SqlValidationVerdict(
        verdict=final_verdict,
        concern=str(error) if error else f"validator failed at stage={stage}",
        suggested_fix=_suggested_fix_for_stage(stage),
        reference_used=f"dry_run:{stage}",
        bytes_processed=int(bytes_processed) if bytes_processed is not None else None,
        estimated_usd=float(estimated_usd) if estimated_usd is not None else None,
    )


def _suggested_fix_for_stage(stage: str) -> str | None:
    return {
        "parse": "Fix SQL syntax before retry.",
        "policy": "Use SELECT/WITH/UNION only — no DML/DDL allowed.",
        "scope": "Restrict tables to the configured MIMIC dataset allowlist.",
        "dry_run": (
            "BigQuery rejected the query at dry-run — check column "
            "references and table joins."
        ),
        "bytes": (
            "Add a partition filter, narrower cohort, or LIMIT to reduce "
            "the scan size."
        ),
        "cost": (
            "Add a partition filter or narrower predicate to reduce "
            "estimated cost."
        ),
    }.get(stage)
