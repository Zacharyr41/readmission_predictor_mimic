"""Pre-execution SQL validator for the conversational pipeline.

Runs between :func:`compile_sql` and ``backend.execute`` so confidently-
broken SQL can be blocked BEFORE the (expensive) BigQuery scan is paid
for. Returns a :class:`SqlValidationVerdict` describing the outcome:

- ``verdict="pass"`` — proceed to execution as usual
- ``verdict="warn"`` — proceed, but the orchestrator threads ``concern``
  into the post-execution critic's ``fallback_warning`` so the critic
  can incorporate the pre-flight concern.
- ``verdict="block"`` — orchestrator short-circuits: no execute, no
  answerer, no critic. Returns the validator's concern to the user.

The validator NEVER raises; on any failure (API error, malformed JSON,
schema mismatch, timeout) it returns ``None`` and the orchestrator
proceeds to execute exactly as today (no regression).

v1 design choices:
- Sonnet 4.6 (judgment task — same tier as the critic).
- Prompt-cached system block (taxonomy is stable across calls).
- NO tools (instruction-based only). The four target failure modes
  (concept-pollution, agg/column mismatch, reference-without-join,
  unit-pooling-on-fallback) are all detectable from the SQL text +
  CompetencyQuestion alone. Tool round-trips would defeat the
  latency/cost purpose of pre-flight validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from src.conversational.models import (
    CompetencyQuestion,
    SqlValidationVerdict,
)
from src.conversational.prompts import SQL_VALIDATOR_SYSTEM_PROMPT
from src.conversational.sql_fastpath import SqlFastpathQuery

logger = logging.getLogger(__name__)


_VALIDATOR_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 400
_DEFAULT_TIMEOUT_SECONDS = 12.0
_RAW_RESPONSE_TRUNCATE = 800


def validate_sql(
    client: anthropic.Anthropic,
    cq: CompetencyQuestion,
    query: SqlFastpathQuery,
    *,
    fallback_warning: str | None,
    resolved_itemids: list[int] | None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> SqlValidationVerdict | None:
    """Pre-execution check on a compiled SQL query.

    Parameters
    ----------
    client:
        The shared ``anthropic.Anthropic`` client from the pipeline.
    cq:
        The CompetencyQuestion the SQL is supposed to answer.
    query:
        The compiled :class:`SqlFastpathQuery` (sql + params + columns).
    fallback_warning:
        Resolver-emitted note when LOINC grounding failed. The validator
        uses this to decide whether the unit-pooling-on-fallback warn
        case applies.
    resolved_itemids:
        MIMIC ``itemid`` list when LOINC grounding succeeded; ``None`` on
        a LIKE-fallback path. The validator consults this to decide whether
        a LIKE-on-label is the safe code-grounded path or the polluting
        fallback.
    timeout:
        Soft-cap on the API call. The SDK's ``timeout=`` enforces it.

    Returns
    -------
    SqlValidationVerdict on success; ``None`` on any failure. Never raises.
    """
    try:
        user_message = _build_user_message(
            cq, query,
            fallback_warning=fallback_warning,
            resolved_itemids=resolved_itemids,
        )
        response = client.messages.create(
            model=_VALIDATOR_MODEL,
            max_tokens=_MAX_TOKENS,
            timeout=timeout,
            system=[
                {
                    "type": "text",
                    "text": SQL_VALIDATOR_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {"role": "user", "content": user_message},
            ],
        )

        text = _extract_final_text(response.content)
        parsed = _extract_json(text)
        if parsed is None:
            logger.warning(
                "sql_validator returned non-JSON response (truncated): %s",
                text[:_RAW_RESPONSE_TRUNCATE],
            )
            return None

        try:
            verdict = SqlValidationVerdict(
                **parsed,
                raw_response=text[:_RAW_RESPONSE_TRUNCATE],
            )
        except ValidationError as exc:
            logger.warning(
                "sql_validator response failed schema validation: %s; payload=%r",
                exc, parsed,
            )
            return None
        return verdict

    except Exception as exc:  # noqa: BLE001 — never block the pipeline
        logger.warning(
            "sql_validator call failed; proceeding without verdict: %s (%s)",
            exc, type(exc).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_message(
    cq: CompetencyQuestion,
    query: SqlFastpathQuery,
    *,
    fallback_warning: str | None,
    resolved_itemids: list[int] | None,
) -> str:
    """Assemble the per-call user message.

    Sections are explicit so the validator can attribute its concern to a
    specific slot. Includes both the LOINC-grounding state (so the
    concept-pollution rule can fire correctly) and the resolver's
    fallback_warning (so the unit-pooling-on-fallback rule can fire).
    """
    parts: list[str] = []
    parts.append(f"## Original question\n{cq.original_question}")

    if cq.interpretation_summary:
        parts.append(
            f"## System interpretation\n{cq.interpretation_summary}"
        )

    if cq.clinical_concepts:
        concept = cq.clinical_concepts[0]
        loinc_repr = concept.loinc_code or "<none>"
        parts.append(
            "## Concept\n"
            f"name={concept.name!r}\n"
            f"concept_type={concept.concept_type!r}\n"
            f"loinc_code={loinc_repr}"
        )

    if resolved_itemids:
        parts.append(
            "## LOINC grounding: succeeded\n"
            f"resolved_itemids={list(resolved_itemids)}\n"
            "(SQL should filter via `itemid IN (...)`; LIKE-on-label "
            "would be an error.)"
        )
    else:
        parts.append(
            "## LOINC grounding: not used (LIKE-fallback path)\n"
            "No resolved itemids were supplied. The compiled SQL is "
            "expected to use a LIKE-on-label filter. Apply the unit-"
            "pooling-on-fallback rule (warn, never block) when relevant."
        )

    if fallback_warning:
        parts.append(
            "## Resolver fallback warning\n"
            f"{fallback_warning}"
        )

    if cq.aggregation:
        parts.append(f"## Requested aggregation\n{cq.aggregation}")

    parts.append(
        "## Compiled SQL\n```sql\n" + query.sql + "\n```"
    )
    if query.params:
        parts.append(
            "## SQL parameters (positional)\n```\n"
            + json.dumps(query.params, default=str)
            + "\n```"
        )

    parts.append(
        "Respond with the JSON verdict object only, per the system schema."
    )
    return "\n\n".join(parts)


def _extract_final_text(content_blocks) -> str:
    for block in content_blocks or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    if content_blocks:
        text = getattr(content_blocks[0], "text", None)
        if isinstance(text, str):
            return text
    return ""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM response, or None."""
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    return result
