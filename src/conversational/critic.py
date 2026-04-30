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

Model: Sonnet 4.6 (judgement quality matters; Haiku rejected). Prompt
caching ON for the system block since the failure-mode taxonomy and
reference ranges are stable across turns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    CriticVerdict,
)
from src.conversational.prompts import CRITIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Sonnet 4.6 is the right tier for clinical-judgment tasks. The answerer and
# decomposer use a stale Sonnet model name (``claude-sonnet-4-20250514``);
# that's a separate follow-up — leave the critic on the current Sonnet.
_CRITIC_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 600
_DEFAULT_TIMEOUT_SECONDS = 10.0
_MAX_DATA_TABLE_ROWS = 10
_RAW_RESPONSE_TRUNCATE = 500


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
        response = client.messages.create(
            model=_CRITIC_MODEL,
            max_tokens=_MAX_TOKENS,
            timeout=timeout,
            system=[
                {
                    "type": "text",
                    "text": CRITIC_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text
        parsed = _extract_json(text)
        if parsed is None:
            logger.warning(
                "critic returned non-JSON response (truncated): %s",
                text[:_RAW_RESPONSE_TRUNCATE],
            )
            return None
        try:
            verdict = CriticVerdict(**parsed, raw_response=text[:_RAW_RESPONSE_TRUNCATE])
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


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM response.

    Mirrors the regex pattern in ``answerer._extract_json`` (lines 102-114)
    but kept local to avoid circular imports. Tries fenced ```json blocks
    first, then a bare object match.
    """
    # Fenced code block
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        # Bare object — find the first balanced { ... }
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
