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

# Public alias kept at the module top so existing tests that monkeypatch
# ``src.conversational.critic.pubmed_search`` continue to work — the
# dispatch built below resolves the tool name through this module's
# globals at call time.
from src.conversational.health_evidence import EvidenceAgent
from src.conversational.health_evidence.tool_defs import PUBMED_SEARCH_TOOL_DEF
from src.conversational.health_evidence.tools import pubmed_search  # noqa: F401  (test monkeypatch hook)
from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    CriticVerdict,
)
from src.conversational.prompts import CRITIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Sonnet 4.6 is the right tier for clinical-judgment tasks.
_CRITIC_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 600
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_DATA_TABLE_ROWS = 10
_RAW_RESPONSE_TRUNCATE = 500
# Hard cap on tool-use iterations per critique. Same as before refactor.
_MAX_TOOL_ITERATIONS = 3


def _critic_tool_dispatch() -> dict[str, Any]:
    """Build a per-call tool_dispatch that resolves names through THIS
    module's globals.

    This preserves the existing test pattern: tests monkeypatch
    ``src.conversational.critic.pubmed_search`` to inject canned tool
    responses. The lambda looks up ``pubmed_search`` in the module dict
    at *call* time, so the patch takes effect even though the function
    object was imported at module load.

    Without this indirection, the EvidenceAgent's default ``TOOL_DISPATCH``
    captures the unpatched function reference at import time and the
    monkeypatch would silently no-op.
    """
    module_dict = sys.modules[__name__].__dict__
    return {
        "pubmed_search": lambda **kw: module_dict["pubmed_search"](**kw),
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
            tools=[PUBMED_SEARCH_TOOL_DEF],
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
        try:
            verdict = CriticVerdict(
                **parsed,
                raw_response=result.final_text[:_RAW_RESPONSE_TRUNCATE],
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
