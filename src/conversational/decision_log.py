"""Append-only decision log for query-routing classifications.

Every turn, each ``CompetencyQuestion`` is classified into exactly one
``QueryPlan`` by ``QueryPlanner`` (the choke point described in
``querytriagesystem.md``). Today that decision is invisible. This module writes
one JSON line *per sub-CQ* recording the chosen plan **and the rule that fired**,
so an operator can ``tail -f`` the routing decisions in real time:

    tail -f logs/routing_decisions.jsonl

and a developer can compute the misrouting rate (§8 "D0") and replay any single
decision through the planner from the line alone (the full ``cq`` dump is the
replay payload).

The destination is ``$NEUROGRAPH_ROUTING_LOG`` (default
``logs/routing_decisions.jsonl``), resolved at call time so an override set
*after* this module imports still takes effect. This mirrors
:mod:`src.conversational.query_log` exactly, including the most important
property: ``log_routing_decision`` is **total** — it never raises, because a
logging failure must not break the turn the user is waiting on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = "logs/routing_decisions.jsonl"
_ENV_VAR = "NEUROGRAPH_ROUTING_LOG"
# Bump when the record shape changes incompatibly. The audit tooling skips lines
# it doesn't understand instead of crashing on a stale full-CQ dump.
_SCHEMA_VERSION = "1"


def _resolve_path(log_path: str | Path | None) -> Path:
    """Explicit ``log_path`` wins; otherwise read the env var at call time."""
    if log_path is not None:
        return Path(log_path)
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_LOG_PATH))


@lru_cache(maxsize=1)
def _registry_signature() -> str | None:
    """A short fingerprint of which aggregates/axes are SQL-eligible *now*.

    A routing decision is only reproducible *given the registry* (rules 10/13 in
    §4.1 consult ``sql_fn`` / ``sql_group_by``). Logging this fingerprint lets the
    audit distinguish lines produced before vs. after a widening-contract change.
    Cached: the answer is fixed for a given code version. Best-effort — ``None``
    if the registry can't be introspected.
    """
    try:
        from src.conversational.operations import get_default_registry

        reg = get_default_registry()
        aggs = sorted(
            n
            for n in reg.supported_names("aggregate")
            if getattr(reg.require("aggregate", n), "sql_fn", None) is not None
        )
        axes = sorted(
            n
            for n in reg.supported_names("comparison_axis")
            if getattr(reg.require("comparison_axis", n), "sql_group_by", None)
            is not None
        )
        payload = "|".join(aggs) + "||" + "|".join(axes)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    except Exception:  # never let a signature failure surface
        return None


def _causal_fallthrough(cq: Any, plan_code: Any) -> bool:
    """True when a ``causal_effect`` CQ *degenerated* (|I| < 2) and dropped
    through to the legacy SQL/GRAPH dispatch instead of routing to CAUSAL.

    Invisible in the final ``reason`` (it lands on whatever later rule fires), so
    surface it as its own flag — a §6 diagnostic the audit can count.
    """
    try:
        return bool(
            getattr(cq, "scope", None) == "causal_effect"
            and len(getattr(cq, "intervention_set", None) or []) < 2
            and plan_code != "causal"
        )
    except Exception:
        return False


def _digest_cq(cq: Any) -> dict[str, Any]:
    """Denormalized, JSON-safe columns for histogramming without parsing the
    full CQ dump.

    Defensive by design (mirrors :func:`query_log._digest_answer`): read each
    attribute with ``getattr`` and tolerate ``None``/missing fields rather than
    assume a schema, so a malformed CQ degrades the digest instead of breaking
    the turn.
    """
    try:
        concepts = list(getattr(cq, "clinical_concepts", None) or [])
        temporal = list(getattr(cq, "temporal_constraints", None) or [])
        filters = list(getattr(cq, "patient_filters", None) or [])
        interventions = list(getattr(cq, "intervention_set", None) or [])
        return {
            "scope": getattr(cq, "scope", None),
            "aggregation": getattr(cq, "aggregation", None),
            "comparison_field": getattr(cq, "comparison_field", None),
            "concept_count": len(concepts),
            "concept_types": [getattr(c, "concept_type", None) for c in concepts],
            "has_temporal": bool(temporal),
            "temporal_relations": [getattr(t, "relation", None) for t in temporal],
            "n_filters": len(filters),
            "split_condition_present": getattr(cq, "split_condition", None)
            is not None,
            "intervention_count": len(interventions),
        }
    except Exception as exc:  # a malformed CQ must not break the turn
        logger.warning("routing decision CQ digest failed: %s", exc)
        return {}


def _build_record(
    cq: Any,
    decision: Any,
    *,
    turn_id: str | None,
    cq_index: int,
    n_cqs: int,
    is_multi: bool,
    question: str | None,
) -> dict[str, Any]:
    plan = getattr(decision, "plan", None)
    reason = getattr(decision, "reason", None)
    plan_code = getattr(plan, "value", plan)
    reason_code = getattr(reason, "value", reason)

    record: dict[str, Any] = {
        "kind": "routing_decision",
        "schema_version": _SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_id": turn_id,
        "cq_index": cq_index,
        "n_cqs": n_cqs,
        "is_multi": bool(is_multi),
        "question": question,
        "cq_question": getattr(cq, "original_question", None),
        "plan": plan_code,
        "reason": reason_code,
        "rule": getattr(decision, "rule", None),
        "detail": getattr(decision, "detail", ""),
        "had_causal_fallthrough": _causal_fallthrough(cq, plan_code),
        "registry_signature": _registry_signature(),
    }
    record.update(_digest_cq(cq))
    try:
        record["cq"] = cq.model_dump(mode="json")
    except Exception as exc:  # full dump is best-effort replay payload
        logger.warning("routing decision CQ dump failed: %s", exc)
        record["cq"] = None
    return record


def log_routing_decision(
    cq: Any,
    decision: Any,
    *,
    turn_id: str | None,
    cq_index: int,
    n_cqs: int,
    is_multi: bool,
    question: str | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one JSON line describing a single routing classification.

    ``decision`` is a ``RoutingDecision`` (duck-typed: ``.plan``/``.reason`` each
    expose ``.value``, plus ``.rule`` and ``.detail``). Returns the record dict
    (handy for tests and callers). Writes are best-effort: any digest or I/O
    failure is logged at WARNING and swallowed so the turn keeps working.
    """
    try:
        record = _build_record(
            cq,
            decision,
            turn_id=turn_id,
            cq_index=cq_index,
            n_cqs=n_cqs,
            is_multi=is_multi,
            question=question,
        )
    except Exception as exc:  # never break the turn on a logging failure
        logger.warning("routing decision record build failed: %s", exc)
        record = {
            "kind": "routing_decision",
            "schema_version": _SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_id": turn_id,
            "cq_index": cq_index,
        }

    # Mirror to the standard logger too, so routing also shows up wherever the
    # process's logging is wired (console/stderr) — not just the JSONL file.
    logger.info(
        "route plan=%s reason=%s rule=%s turn=%s cq=%s/%s q=%r",
        record.get("plan"),
        record.get("reason"),
        record.get("rule"),
        turn_id,
        cq_index,
        n_cqs,
        record.get("cq_question") or question,
    )

    path = _resolve_path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # never break the turn on a logging failure
        logger.warning("routing decision log write to %s failed: %s", path, exc)

    return record
