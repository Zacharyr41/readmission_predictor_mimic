"""Clinical consult — disambiguation, clarification, contextualization.

Three thin functions atop :class:`EvidenceAgent`:

- :func:`disambiguate` decides whether an ambiguous concept name can be
  canonicalized using literature/registry tools. The orchestrator calls
  this BEFORE the existing clarify short-circuit; high-confidence
  resolutions allow the pipeline to proceed without asking the user.
- :func:`clarify` formats a literature-grounded user-facing clarifying
  message when ambiguity remains. Called inside the clarify short-circuit;
  on failure, the orchestrator falls through to the decomposer's raw text.
- :func:`contextualize` appends a 1-2 sentence literature-grounded note
  to a successful answer. Called after the critic; only fires when the
  critic's verdict is "info" (or absent) so the note doesn't drown a
  warning. Default-OFF.

All three NEVER raise; on any failure they return ``None``.

Like the critic, this module keeps ``pubmed_search`` as an importable
alias at the module top so tests that monkeypatch
``src.conversational.clinical_consult.pubmed_search`` take effect via
the per-call dispatch built below.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import anthropic
from pydantic import ValidationError

from src.conversational.health_evidence import (
    ALL_TOOL_DEFS,
    EvidenceAgent,
    TOOL_DISPATCH,
)
from src.conversational.health_evidence.tools import (  # noqa: F401  (test monkeypatch hook)
    loinc_reference_range,
    mimic_distribution_lookup,
    pubmed_search,
)
from src.conversational.models import (
    AnswerResult,
    ClarifyingMessage,
    ClinicalConcept,
    CompetencyQuestion,
    ContextualNote,
    Disambiguation,
)
from src.conversational.prompts import (
    CLARIFY_SYSTEM_PROMPT,
    CONTEXTUALIZE_SYSTEM_PROMPT,
    DISAMBIGUATE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


_CONSULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 500
_DEFAULT_TIMEOUT_SECONDS = 20.0
_DEFAULT_MAX_ITERATIONS = 3


def _consult_tool_dispatch() -> dict[str, Any]:
    """Build a per-call tool_dispatch that resolves names through THIS
    module's globals (so test monkeypatches on
    ``src.conversational.clinical_consult.pubmed_search`` etc. take effect).
    Same trick as the critic module."""
    module_dict = sys.modules[__name__].__dict__
    return {
        "pubmed_search": lambda **kw: module_dict["pubmed_search"](**kw),
        "mimic_distribution_lookup": lambda **kw: module_dict["mimic_distribution_lookup"](**kw),
        "loinc_reference_range": lambda **kw: module_dict["loinc_reference_range"](**kw),
    }


def _make_agent(client: anthropic.Anthropic) -> EvidenceAgent:
    return EvidenceAgent(
        client,
        model=_CONSULT_MODEL,
        max_tokens=_DEFAULT_MAX_TOKENS,
        max_iterations=_DEFAULT_MAX_ITERATIONS,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        tools=ALL_TOOL_DEFS,
        tool_dispatch=_consult_tool_dispatch(),
    )


# ---------------------------------------------------------------------------
# disambiguate
# ---------------------------------------------------------------------------


def disambiguate(
    client: anthropic.Anthropic,
    concept: ClinicalConcept,
    *,
    original_question: str,
) -> Disambiguation | None:
    """Try to canonicalize an ambiguous concept name.

    Returns a :class:`Disambiguation` describing the outcome (canonical
    name, alternates, optional resolved code, confidence). Returns
    ``None`` on any failure. Never raises.
    """
    try:
        agent = _make_agent(client)
        user_msg = (
            f"## Concept name (as user typed)\n{concept.name}\n\n"
            f"## Concept type\n{concept.concept_type}\n\n"
            f"## User's full question\n{original_question}\n\n"
            "Disambiguate per the schema."
        )
        result = agent.consult(DISAMBIGUATE_SYSTEM_PROMPT, user_msg)
        if result.parsed_json is None:
            logger.info(
                "disambiguate returned non-JSON (truncated): %s",
                (result.final_text or "")[:300],
            )
            return None
        parsed = dict(result.parsed_json)
        # Patch in the user-typed name if the model omitted it.
        parsed.setdefault("input_name", concept.name)
        # Filter cited_sources against observed citations and surface as
        # ``citations`` on the Disambiguation model.
        cited = result.filter_claimed_citations(parsed.pop("cited_sources", None))
        try:
            verdict = Disambiguation(**parsed, citations=cited)
        except ValidationError as exc:
            logger.info("disambiguate schema fail: %s", exc)
            return None
        return verdict
    except Exception as exc:  # noqa: BLE001
        logger.info("disambiguate failed: %s (%s)", exc, type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# clarify
# ---------------------------------------------------------------------------


def clarify(
    client: anthropic.Anthropic,
    original_question: str,
    raw_clarifying_question: str,
    partial_disambiguations: list[Disambiguation],
) -> ClarifyingMessage | None:
    """Format a clinically-grounded clarifying message for the user.

    The ``partial_disambiguations`` argument carries any low/medium-
    confidence Disambiguation objects from the disambiguation pass so
    the message can offer the alternates the literature surfaced.

    Returns ``None`` on any failure; the orchestrator falls back to the
    decomposer's raw clarifying question text in that case.
    """
    try:
        agent = _make_agent(client)
        partials_repr = json.dumps(
            [d.model_dump(mode="json") for d in partial_disambiguations],
            default=str,
            indent=2,
        )
        user_msg = (
            f"## Original question\n{original_question}\n\n"
            f"## Raw clarifying question (from decomposer)\n{raw_clarifying_question}\n\n"
            f"## Partial disambiguations\n```json\n{partials_repr}\n```\n\n"
            "Format the user-facing clarifying message per the schema."
        )
        result = agent.consult(CLARIFY_SYSTEM_PROMPT, user_msg)
        if result.parsed_json is None:
            return None
        parsed = dict(result.parsed_json)
        cited = result.filter_claimed_citations(parsed.pop("cited_sources", None))
        try:
            return ClarifyingMessage(**parsed, citations=cited)
        except ValidationError as exc:
            logger.info("clarify schema fail: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.info("clarify failed: %s (%s)", exc, type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# contextualize
# ---------------------------------------------------------------------------


def contextualize(
    client: anthropic.Anthropic,
    answer: AnswerResult,
    cq: CompetencyQuestion,
) -> ContextualNote | None:
    """Append a brief literature-grounded note to a successful answer.

    Returns a :class:`ContextualNote` whose ``text`` is non-empty; returns
    ``None`` when the model decided there was nothing useful to add (empty
    text) or on any failure. Never raises.
    """
    try:
        agent = _make_agent(client)
        user_msg = (
            f"## Original question\n{cq.original_question}\n\n"
            f"## Answer text\n{answer.text_summary}\n\n"
        )
        if answer.data_table:
            sample = answer.data_table[:5]
            user_msg += (
                "## Data table (first rows)\n```json\n"
                + json.dumps(sample, default=str, indent=2)
                + "\n```\n\n"
            )
        user_msg += "Append a contextual note per the schema."

        result = agent.consult(CONTEXTUALIZE_SYSTEM_PROMPT, user_msg)
        if result.parsed_json is None:
            return None
        parsed = dict(result.parsed_json)
        text = (parsed.get("text") or "").strip()
        if not text:
            # Model declined to add a note — equivalent to "no useful context".
            return None
        cited = result.filter_claimed_citations(parsed.pop("cited_sources", None))
        try:
            return ContextualNote(text=text, citations=cited)
        except ValidationError as exc:
            logger.info("contextualize schema fail: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.info("contextualize failed: %s (%s)", exc, type(exc).__name__)
        return None
