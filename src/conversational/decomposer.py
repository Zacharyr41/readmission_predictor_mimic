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
    
    ## TODO(zach): More robust prompt, with more filter/population options
    ## TODO(zach): We want to provide examples of decomposing an initial "human" Q into the competency questions
    ## TODO (zach): Provide broader context on what this pipeline is (self-awareness sort of thing)
    ## TODO (zach - phase 1): programmatic integration with external source of truth like SNOMED CT and for ICD codes
    ## TODO (zach): Many Many Many test cases -- broad test cases. Automate as part of test suite. 
    # TODO (zach): Filtering shouldn't be artificially set to num patients, but can include other fields (more robust filtering system or maybe less artiifcial filters?)
    ## TODO (zach): compell user to really understand what they ask / limited filtering/deterministic portions of interface with clear explanations
    ## For above - AI comprehension of questions (i.e. hey user, this is what I'm attempting to answer)
    ## UI should be clear as to what its doing, what it searches for , etc. 
    
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

    # Validate filter fields — retry if LLM used unsupported fields.
    # Source of truth is the OperationRegistry so this set stays in sync with
    # what the extractor can actually compile and what the prompt advertises.
    from src.conversational.operations import get_default_registry

    supported_fields = set(get_default_registry().supported_names("filter"))
    unsupported = {f.field for f in cq.patient_filters} - supported_fields
    if unsupported:
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": (
                f"These patient_filter fields are not supported: {unsupported}. "
                f"Supported fields are: {sorted(supported_fields)}. "
                "Please rephrase the filters using only supported fields "
                "(e.g. use 'diagnosis' to filter by condition). "
                "Return ONLY the corrected JSON."
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
