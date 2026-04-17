"""LLM-powered decomposition of clinical questions into structured queries."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.conversational.models import (
    CompetencyQuestion,
    DecompositionResult,
    ReturnType,
)

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


def _synthesise_interpretation(cq: CompetencyQuestion) -> str:
    """Build a one-sentence echo of the CQ's structured fields.

    Called when the LLM omits ``interpretation_summary``. The goal is that the
    clinician-facing "this is what I'm answering" echo is NEVER blank — if the
    LLM forgets, we reconstruct a serviceable restatement from the structured
    fields alone. Prose is intentionally mechanical; the LLM normally writes
    something better, this is the fallback.
    """
    # Lead with the aggregate verb if present, else a generic verb per return_type.
    verb_map = {
        "mean": "Mean",
        "avg": "Mean",
        "median": "Median",
        "max": "Maximum",
        "min": "Minimum",
        "count": "Count of",
        "sum": "Total",
        "exists": "Whether any",
    }
    if cq.aggregation:
        verb = verb_map.get(cq.aggregation, cq.aggregation.capitalize())
    elif cq.return_type.value in ("table", "text_and_table"):
        verb = "Listing of"
    elif cq.return_type == ReturnType.VISUALIZATION:
        verb = "Plot of"
    else:
        verb = "Lookup of"

    concepts = ", ".join(c.name for c in cq.clinical_concepts) or "records"

    filter_parts: list[str] = []
    for f in cq.patient_filters:
        if isinstance(f.value, list):
            val = "[" + ", ".join(str(v) for v in f.value) + "]"
        else:
            val = str(f.value)
        filter_parts.append(f"{f.field} {f.operator} {val}")
    filter_clause = f" for admissions where {' and '.join(filter_parts)}" if filter_parts else ""

    temporal_parts = [
        f"{tc.relation} {tc.reference_event}"
        + (f" within {tc.time_window}" if tc.time_window and tc.relation == "within" else "")
        for tc in cq.temporal_constraints
    ]
    temporal_clause = f" ({', '.join(temporal_parts)})" if temporal_parts else ""

    if cq.scope == "comparison" and cq.comparison_field:
        tail = f", grouped by {cq.comparison_field}"
    else:
        tail = ""

    return f"{verb} {concepts}{filter_clause}{temporal_clause}{tail}.".replace(" ,", ",")


def _postprocess_cq(cq: CompetencyQuestion) -> CompetencyQuestion:
    """Apply return-type validation + interpretation-summary synthesis.

    Broken out so it can be applied to each CQ of a Shape B decomposition
    as well as to the single CQ of a Shape A response.
    """
    cq = _validate_return_type(cq)
    # Phase 4: interpretation_summary is clinician-facing; make sure it's never
    # blank. The prompt always asks the LLM to populate it, but if it forgets
    # or returns whitespace we synthesise a mechanical restatement from the
    # structured fields so the UI has something to echo.
    if not cq.interpretation_summary or not cq.interpretation_summary.strip():
        cq.interpretation_summary = _synthesise_interpretation(cq)
    return cq


def _parse_response_payload(
    data: dict, original_question: str,
) -> DecompositionResult:
    """Parse an LLM response payload into a DecompositionResult.

    The LLM may return either Shape A (single CQ dict) or Shape B
    ({"narrative": ..., "competency_questions": [...]}). We detect by
    presence of the ``competency_questions`` key. In Shape A, the top-level
    ``original_question`` is always overwritten with the user's actual
    question; in Shape B, only the outer wrapper lacks that field and each
    sub-CQ keeps its own ``original_question`` (typically a sub-question
    authored by the LLM).
    """
    if "competency_questions" in data:
        # Shape B: big-question decomposition.
        cqs = [CompetencyQuestion.model_validate(c) for c in data.get("competency_questions", [])]
        return DecompositionResult(
            narrative=data.get("narrative"),
            competency_questions=cqs,
        )
    # Shape A: single CQ. Stamp the user's question over whatever the LLM
    # echoed, so downstream steps see the exact input text.
    data = {**data, "original_question": original_question}
    return DecompositionResult(
        narrative=None,
        competency_questions=[CompetencyQuestion.model_validate(data)],
    )


def decompose_question(
    client: anthropic.Anthropic,
    question: str,
    conversation_history: list | None = None,
) -> DecompositionResult:
    """Decompose a clinical question into a DecompositionResult.

    Handles both single-CQ (Shape A) and big-question (Shape B) LLM responses
    behind one entry point. Retry-once-on-failure behaviour is preserved from
    the pre-Phase-4.5 implementation. Unsupported filter fields trigger a
    retry across every sub-CQ in one batch.

    Returns
    -------
    DecompositionResult
        Always contains at least one CompetencyQuestion.
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
        decomp = _parse_response_payload(data, question)
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
        decomp = _parse_response_payload(data, question)

    # Validate filter fields across EVERY sub-CQ. One retry if any use
    # unsupported fields; the corrective message names every offender in
    # one round-trip rather than per-CQ.
    from src.conversational.operations import get_default_registry

    supported_fields = set(get_default_registry().supported_names("filter"))
    all_unsupported: set[str] = set()
    for cq in decomp.competency_questions:
        all_unsupported |= {f.field for f in cq.patient_filters} - supported_fields
    if all_unsupported:
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": (
                f"These patient_filter fields are not supported: {all_unsupported}. "
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
        decomp = _parse_response_payload(data, question)

    # Post-process every CQ: validate return_type + synthesise interpretation
    # if the LLM left one blank. Done after all retry paths so the fallback
    # only runs on the final parsed shape.
    decomp.competency_questions = [
        _postprocess_cq(cq) for cq in decomp.competency_questions
    ]
    return decomp


def decompose(
    client: anthropic.Anthropic,
    question: str,
    conversation_history: list | None = None,
) -> CompetencyQuestion:
    """Backward-compatible single-CQ entry point.

    Returns the FIRST CompetencyQuestion of whatever the LLM produced. For
    Shape A responses this is the only CQ; for Shape B it's the first
    sub-CQ, which discards the narrative and remaining sub-CQs. Callers
    that need the full decomposition — notably the orchestrator — should
    call ``decompose_question`` instead.

    Raises
    ------
    json.JSONDecodeError | pydantic.ValidationError
        If the LLM fails to produce valid JSON after one retry.
    """
    decomp = decompose_question(client, question, conversation_history)
    return decomp.competency_questions[0]
