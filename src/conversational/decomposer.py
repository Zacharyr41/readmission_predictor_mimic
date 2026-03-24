"""LLM-powered decomposition of clinical questions into structured queries."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.conversational.models import CompetencyQuestion, ReturnType

if TYPE_CHECKING:
    import anthropic
from src.conversational.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    build_decomposition_messages,
)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 1024

_VIZ_KEYWORDS = frozenset(
    ["plot", "chart", "graph", "visualize", "histogram", "distribution"]
)
_TREND_KEYWORDS = frozenset(
    ["trend", "over time", "trajectory", "day-by-day", "daily", "hourly", "change"]
)


def _extract_json(text: str) -> str:
    """Extract a JSON object from *text*, handling markdown code fences."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _validate_return_type(cq: CompetencyQuestion) -> CompetencyQuestion:
    """Lightweight post-processing to catch obvious return_type mismatches."""
    q = cq.original_question.lower()

    # Rule 1: VISUALIZATION without any temporal/trend/viz signal → downgrade
    if cq.return_type == ReturnType.VISUALIZATION:
        has_temporal = len(cq.temporal_constraints) > 0
        has_trend = any(kw in q for kw in _TREND_KEYWORDS)
        has_viz = any(kw in q for kw in _VIZ_KEYWORDS)
        if not (has_temporal or has_trend or has_viz):
            cq.return_type = ReturnType.TEXT_AND_TABLE

    # Rule 2: TEXT for cohort/comparison with non-count aggregation → upgrade
    if cq.return_type == ReturnType.TEXT:
        if (
            cq.scope in ("cohort", "comparison")
            and len(cq.clinical_concepts) > 0
            and cq.aggregation not in ("count", "exists")
        ):
            cq.return_type = ReturnType.TEXT_AND_TABLE

    return cq


def decompose(
    client: anthropic.Anthropic,
    question: str,
    conversation_history: list | None = None,
) -> CompetencyQuestion:
    """Decompose a clinical question into a structured CompetencyQuestion.

    Parameters
    ----------
    client:
        An initialised ``anthropic.Anthropic`` instance.
    question:
        The natural-language clinical question.
    conversation_history:
        Optional list of ``(CompetencyQuestion, AnswerResult)`` tuples from
        prior conversation turns.

    Returns
    -------
    CompetencyQuestion
        Parsed and validated structured representation.

    Raises
    ------
    json.JSONDecodeError | pydantic.ValidationError
        If the LLM fails to produce valid JSON after one retry.
    """
    messages = build_decomposition_messages(question, conversation_history)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=DECOMPOSITION_SYSTEM_PROMPT,
        messages=messages,
    )

    raw_text = response.content[0].text
    json_str = _extract_json(raw_text)

    try:
        data = json.loads(json_str)
        data["original_question"] = question
        cq = CompetencyQuestion.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        # Retry once — feed the error back so the LLM can self-correct.
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": (
                f"Your response was not valid JSON or failed validation: {exc}. "
                "Please return ONLY the corrected JSON object."
            ),
        })
        retry = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=DECOMPOSITION_SYSTEM_PROMPT,
            messages=messages,
        )
        retry_json = _extract_json(retry.content[0].text)
        data = json.loads(retry_json)
        data["original_question"] = question
        cq = CompetencyQuestion.model_validate(data)

    return _validate_return_type(cq)
