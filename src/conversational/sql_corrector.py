"""Critic-driven SQL corrector for the conversational pipeline.

Runs AFTER the post-execution plausibility critic
(:func:`src.conversational.critic.critique`) flags a ``warn``/``block``
whose concern is a *fixable query bug* — e.g. mortality detected via an ICD
code instead of ``admissions.hospital_expire_flag``, or a drug filtered by
its class label (``'thiazide diuretic'``) instead of the agent names MIMIC
actually stores. The critic is good at *diagnosing* these; this module turns
the diagnosis into an executable corrected query the user can run with one
click.

Single public entry point :func:`propose_sql_correction`. Like the
pre-execution validator it is a single, no-tool LLM call (the reviewer's
diagnosis usually names the exact fix, so a tool loop buys little) and it
NEVER raises — on any failure (not fixable, empty/missing SQL, malformed
JSON, schema mismatch, API error) it returns ``None`` so the answer still
renders without a correction offer.

Model: Sonnet 4.6 — a single-shot SQL rewrite is extraction-shaped, same
tier as the decomposer / answerer / pre-execution validator (Haiku is
rejected for this class of work).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    CriticVerdict,
    SqlCorrection,
)
from src.conversational.prompts import SQL_CORRECTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


_CORRECTOR_MODEL = "claude-sonnet-4-6"
# SQL can be long; give the rewrite room to reproduce the query plus a
# one-sentence rationale without truncating mid-statement.
_MAX_TOKENS = 1500
_DEFAULT_TIMEOUT_SECONDS = 20.0
_RAW_RESPONSE_TRUNCATE = 800
_MAX_DATA_TABLE_ROWS = 10


def propose_sql_correction(
    client: anthropic.Anthropic,
    cq: CompetencyQuestion,
    answer: AnswerResult,
    critic_verdict: CriticVerdict,
    original_sql: str,
    *,
    fallback_warning: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> SqlCorrection | None:
    """Ask the model to rewrite ``original_sql`` per the critic's diagnosis.

    Parameters
    ----------
    client:
        The shared ``anthropic.Anthropic`` client from the pipeline.
    cq:
        The competency question that produced the (flagged) answer. Its
        ``original_question`` / ``return_type`` / ``interpretation_summary`` /
        ``aggregation`` are carried onto the returned :class:`SqlCorrection`
        so the re-run path can re-explain the corrected result.
    answer:
        The flagged AnswerResult (its text + data table give the corrector
        the observed — usually empty — result).
    critic_verdict:
        The critic's verdict; its ``concern`` + ``reference_used`` are the
        diagnosis the corrector applies.
    original_sql:
        The exact SQL that ran (values-inlined ``rendered_sql``). The
        corrected SQL must mirror its tables, dialect, and output columns.
    fallback_warning:
        Optional resolver/validator note threaded in for extra context.
    timeout:
        Soft-cap on the API call.

    Returns
    -------
    SqlCorrection on success; ``None`` when the model judges the concern
    unfixable, emits no usable SQL, or any failure occurs. Never raises.
    """
    try:
        user_message = _build_user_message(
            cq, answer, critic_verdict, original_sql,
            fallback_warning=fallback_warning,
        )
        response = client.messages.create(
            model=_CORRECTOR_MODEL,
            max_tokens=_MAX_TOKENS,
            timeout=timeout,
            system=[
                {
                    "type": "text",
                    "text": SQL_CORRECTOR_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        text = _extract_final_text(response.content)
        parsed = _extract_json(text)
        if parsed is None:
            logger.warning(
                "sql_corrector returned non-JSON response (truncated): %s",
                text[:_RAW_RESPONSE_TRUNCATE],
            )
            return None

        if not parsed.get("fixable", False):
            return None
        corrected_sql = parsed.get("corrected_sql")
        if not isinstance(corrected_sql, str) or not corrected_sql.strip():
            return None

        return SqlCorrection(
            corrected_sql=corrected_sql.strip(),
            rationale=(parsed.get("rationale") or "").strip()
            or "Corrected the query per the reviewer's diagnosis.",
            original_question=cq.original_question,
            return_type=cq.return_type,
            interpretation_summary=cq.interpretation_summary,
            aggregation=cq.aggregation,
        )

    except Exception as exc:  # noqa: BLE001 — never block answer rendering
        logger.warning(
            "sql_corrector call failed; no correction offered: %s (%s)",
            exc, type(exc).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_message(
    cq: CompetencyQuestion,
    answer: AnswerResult,
    critic_verdict: CriticVerdict,
    original_sql: str,
    *,
    fallback_warning: str | None,
) -> str:
    """Assemble the per-call user message.

    Sections are labeled so the corrector can attribute the fix to a slot:
    the question, the interpretation, the SQL that ran, its observed result
    (or an explicit empty marker), and the reviewer's diagnosis.
    """
    parts: list[str] = []
    parts.append(f"## Question\n{cq.original_question}")

    if cq.interpretation_summary:
        parts.append(f"## System interpretation\n{cq.interpretation_summary}")

    parts.append("## SQL that ran\n```sql\n" + original_sql + "\n```")

    if answer.data_table:
        rows = answer.data_table[:_MAX_DATA_TABLE_ROWS]
        parts.append(
            "## Result (first {} rows)\n```\n{}\n```".format(
                len(rows), json.dumps(rows, default=str, indent=2),
            )
        )
    else:
        parts.append(
            "## Result\n(empty — the query returned no rows)\n"
            f"Answer text shown to the user: {answer.text_summary}"
        )

    concern = critic_verdict.concern or "(no concern text)"
    parts.append(f"## Reviewer diagnosis (why the result is wrong)\n{concern}")
    if critic_verdict.reference_used:
        parts.append(f"## Reviewer reference\n{critic_verdict.reference_used}")

    if fallback_warning:
        parts.append(f"## System warning\n{fallback_warning}")

    parts.append(
        "Return the JSON object only: {fixable, corrected_sql, rationale}."
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
