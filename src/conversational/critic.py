"""LLM-as-judge / second-pass plausibility critic.

Runs after `answerer.generate_answer` (or any other AnswerResult builder) to
review the answer for clinical plausibility. The critic catches what the
answerer's confident-by-default prose cannot self-assess:

  * Unit-pooling pollution surviving an explicit fallback warning
    (e.g. "Mean lactate 199 mg/dL" actually being LDH at U/L scale).
  * Biologically impossible aggregates (mean age 380, mortality > 1).
  * Unit mismatches between the answer narrative and the reference scale.

The single public entry point is :func:`critique`. It must NEVER raise —
on any failure (API error, timeout, malformed JSON, schema validation
error) it returns ``None`` so the answer still renders. The orchestrator
guards against missing verdicts by checking for ``None`` before attaching.

Architecture: this module is a thin client over
:class:`src.conversational.health_evidence.EvidenceAgent`. The agent owns
the tool-use loop, citation tracking, and graceful failure; this module
owns the critic-specific prompt assembly and verdict parsing.

Model: Sonnet 4.6 (judgement quality matters; Haiku rejected). Prompt
caching ON for the system block since the failure-mode taxonomy and
reference ranges are stable across turns.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import anthropic
from pydantic import ValidationError

# Public aliases kept at the module top so existing tests that monkeypatch
# ``src.conversational.critic.<tool_name>`` continue to work — the dispatch
# built below resolves each tool name through this module's globals at call
# time.
from src.conversational.health_evidence import EvidenceAgent
from src.conversational.health_evidence.tool_defs import ALL_TOOL_DEFS
from src.conversational.health_evidence.tools import (  # noqa: F401  (test monkeypatch hooks)
    code_map,
    icd_autocode,
    icd_lookup,
    loinc_reference_range,
    mimic_distribution_lookup,
    mimic_itemid_search,
    openfda_drug_label,
    pubmed_search,
    rxnorm_lookup,
    snomed_expand_ecl,
    snomed_search,
    trials_search,
)
from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    CriticVerdict,
)
from src.conversational.prompts import CRITIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Sonnet 4.6 is the right tier for clinical-judgment tasks.
_CRITIC_MODEL = "claude-sonnet-4-6"
# Bumped from 600 → 1500 in Tier-D follow-up: cohort-stratified
# reasoning (single-patient vs cohort-mean distinction, sibling-cohort
# checks, multi-tool deliberation) regularly needed > 600 tokens to
# both reason AND emit clean JSON. The 600-token ceiling caused
# verdicts to come back as None because the model spent its budget
# thinking and never reached the JSON output. Cost impact: ~2.5×
# critic tokens per call worst case; mitigated by Sonnet 4.6 prompt
# caching on the system block.
_MAX_TOKENS = 1500
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_DATA_TABLE_ROWS = 10
_RAW_RESPONSE_TRUNCATE = 500
# Hard cap on tool-use iterations per critique. Same as before refactor.
_MAX_TOOL_ITERATIONS = 3


# Tools the critic has access to. Adding a new tool is a one-line change
# here PLUS a corresponding ``from ... import <name>  # noqa: F401`` so
# tests can monkeypatch ``src.conversational.critic.<name>``.
_CRITIC_TOOLS: tuple[str, ...] = (
    "pubmed_search",
    "loinc_reference_range",
    "mimic_distribution_lookup",
    "mimic_itemid_search",
    "snomed_search",
    "snomed_expand_ecl",
    "rxnorm_lookup",
    "code_map",
    "trials_search",
    "openfda_drug_label",
    "icd_lookup",
    "icd_autocode",
)


# Tool definitions filtered from the canonical registry. The model receives
# this list in its system block; the dispatch below routes invocations.
_CRITIC_TOOL_DEFS: list[dict[str, Any]] = [
    d for d in ALL_TOOL_DEFS if d["name"] in _CRITIC_TOOLS
]


def _critic_tool_dispatch() -> dict[str, Any]:
    """Build a per-call tool_dispatch that resolves names through THIS
    module's globals.

    This preserves the existing test pattern: tests monkeypatch
    ``src.conversational.critic.<tool_name>`` to inject canned tool
    responses. Each lambda looks up its tool in the module dict at *call*
    time, so the patch takes effect even though the function object was
    imported at module load.

    The ``_n=name`` default-arg trick is the standard fix for Python's
    late-binding closure pitfall — without it, every lambda in this
    comprehension would close over the same ``name`` and dispatch to
    whichever tool happened to be last in the iteration.
    """
    module_dict = sys.modules[__name__].__dict__
    return {
        name: (lambda _n=name, **kw: module_dict[_n](**kw))
        for name in _CRITIC_TOOLS
    }


def critique(
    client: anthropic.Anthropic,
    cq: CompetencyQuestion,
    answer: AnswerResult,
    fallback_warning: str | None = None,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> CriticVerdict | None:
    """Second-pass plausibility check.

    Parameters
    ----------
    client:
        The shared ``anthropic.Anthropic`` client from the pipeline.
    cq:
        The competency question that produced this answer.
    answer:
        The AnswerResult to review. Must have ``text_summary`` populated.
    fallback_warning:
        If a prior pipeline stage emitted a structured warning (e.g. the
        LOINC-fallback note from ``_run_sql_fastpath``), pass it here so
        the critic knows to look for confident interpretations contradicting
        the warning. The warning text is included in the user message.
    timeout:
        Soft-cap on the API call. The SDK's ``timeout=`` param enforces it.

    Returns
    -------
    CriticVerdict on success; ``None`` on any failure mode (API error,
    timeout, malformed JSON, schema validation). Never raises.
    """
    try:
        user_message = _build_user_message(cq, answer, fallback_warning)
        agent = EvidenceAgent(
            client,
            model=_CRITIC_MODEL,
            max_tokens=_MAX_TOKENS,
            max_iterations=_MAX_TOOL_ITERATIONS,
            timeout=timeout,
            tools=list(_CRITIC_TOOL_DEFS),
            tool_dispatch=_critic_tool_dispatch(),
        )
        result = agent.consult(CRITIC_SYSTEM_PROMPT, user_message)

        if result.parsed_json is None:
            logger.warning(
                "critic returned non-JSON response (truncated): %s",
                (result.final_text or "")[:_RAW_RESPONSE_TRUNCATE],
            )
            return None

        parsed = dict(result.parsed_json)
        parsed["cited_sources"] = result.filter_claimed_citations(
            parsed.get("cited_sources"),
        )
        # Drop tool_calls if the model emitted one — the field is
        # populated post-construction from the agent's ground-truth
        # ``EvidenceResult.tool_calls``, not from the model's claim.
        parsed.pop("tool_calls", None)
        try:
            verdict = CriticVerdict(
                **parsed,
                raw_response=result.final_text[:_RAW_RESPONSE_TRUNCATE],
                tool_calls=[tc.model_dump() for tc in result.tool_calls] or None,
            )
        except ValidationError as exc:
            logger.warning(
                "critic response failed schema validation: %s; payload=%r",
                exc, parsed,
            )
            return None
        return verdict

    except Exception as exc:  # noqa: BLE001 — never block answer rendering
        logger.warning(
            "critic call failed; falling through with no verdict: %s (%s)",
            exc, type(exc).__name__,
        )
        return None


def _build_user_message(
    cq: CompetencyQuestion,
    answer: AnswerResult,
    fallback_warning: str | None,
) -> str:
    """Compose the per-turn user message.

    Includes: the original question, the decomposer's interpretation, the
    answer text, the first ``_MAX_DATA_TABLE_ROWS`` of the data table, the
    SPARQL/SQL trace, and any system warning. Each section is labeled so
    the critic can attribute its concern."""
    parts: list[str] = []
    parts.append(f"## Question\n{cq.original_question}")

    if cq.interpretation_summary:
        parts.append(f"## System interpretation\n{cq.interpretation_summary}")

    parts.append(f"## Answer text\n{answer.text_summary}")

    if answer.data_table:
        rows = answer.data_table[:_MAX_DATA_TABLE_ROWS]
        parts.append(
            "## Data table (first {} rows)\n```\n{}\n```".format(
                len(rows), json.dumps(rows, default=str, indent=2),
            )
        )

    if answer.sparql_queries_used:
        parts.append(
            "## SQL / SPARQL executed\n```\n{}\n```".format(
                "\n---\n".join(answer.sparql_queries_used)
            )
        )

    if fallback_warning:
        parts.append(f"## System warning\n{fallback_warning}")

    parts.append(
        "Respond with the JSON verdict object only, per the output schema."
    )
    return "\n\n".join(parts)
