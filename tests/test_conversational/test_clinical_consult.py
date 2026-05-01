"""Tests for the clinical-consult module (disambiguate/clarify/contextualize).

All three are thin wrappers around ``EvidenceAgent``. They:
- Use the agent's tool-use loop (PubMed + LOINC + MIMIC distribution).
- Filter cited sources against ``EvidenceResult.observed_citations``
  (anti-hallucination guard).
- NEVER raise; return None on any failure.

These tests mock the Anthropic client at the message-create level — same
pattern as the critic and validator tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.conversational.clinical_consult import (
    clarify,
    contextualize,
    disambiguate,
)
from src.conversational.models import (
    AnswerResult,
    ClarifyingMessage,
    ClinicalConcept,
    CompetencyQuestion,
    ContextualNote,
    Disambiguation,
)
from tests.test_conversational.conftest import mock_anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disamb_response(
    *,
    canonical_name: str,
    confidence: str,
    resolved_code: str | None = None,
    code_system: str | None = None,
    alternates: list[str] | None = None,
    reasoning: str | None = None,
    cited_sources: list[dict] | None = None,
) -> str:
    payload = {
        "input_name": "lactate",
        "canonical_name": canonical_name,
        "alternates": alternates or [],
        "resolved_code": resolved_code,
        "code_system": code_system,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    if cited_sources is not None:
        payload["cited_sources"] = cited_sources
    return json.dumps(payload)


def _clarify_response(
    text: str,
    *,
    alternates_offered: list[str] | None = None,
    cited_sources: list[dict] | None = None,
) -> str:
    payload = {
        "text": text,
        "alternates_offered": alternates_offered or [],
    }
    if cited_sources is not None:
        payload["cited_sources"] = cited_sources
    return json.dumps(payload)


def _ctx_response(
    text: str,
    *,
    cited_sources: list[dict] | None = None,
) -> str:
    payload = {"text": text}
    if cited_sources is not None:
        payload["cited_sources"] = cited_sources
    return json.dumps(payload)


def _make_concept() -> ClinicalConcept:
    return ClinicalConcept(name="lactate", concept_type="biomarker")


def _make_cq() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="average lactate for sepsis patients",
        clinical_concepts=[_make_concept()],
    )


def _make_answer() -> AnswerResult:
    return AnswerResult(text_summary="Mean lactate was 2.1 mmol/L (n=1234).")


# ---------------------------------------------------------------------------
# disambiguate
# ---------------------------------------------------------------------------


class TestDisambiguateHappyPath:
    def test_high_confidence_returns_resolved_code(self):
        client = mock_anthropic([_disamb_response(
            canonical_name="serum lactate",
            confidence="high",
            resolved_code="32693-4",
            code_system="loinc",
            reasoning="In sepsis context, serum is overwhelmingly intended.",
        )])
        result = disambiguate(
            client, _make_concept(),
            original_question="average lactate for sepsis patients",
        )
        assert result is not None
        assert isinstance(result, Disambiguation)
        assert result.canonical_name == "serum lactate"
        assert result.confidence == "high"
        assert result.resolved_code == "32693-4"
        assert result.code_system == "loinc"

    def test_low_confidence_no_resolved_code(self):
        client = mock_anthropic([_disamb_response(
            canonical_name="lactate (specimen ambiguous)",
            confidence="low",
            alternates=["serum lactate", "CSF lactate", "plasma lactate"],
        )])
        result = disambiguate(
            client, _make_concept(),
            original_question="lactate level",
        )
        assert result is not None
        assert result.confidence == "low"
        assert result.resolved_code is None
        assert "serum lactate" in result.alternates


class TestDisambiguateFailureModes:
    def test_api_error_returns_none(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = disambiguate(
            client, _make_concept(),
            original_question="x",
        )
        assert result is None

    def test_malformed_json_returns_none(self):
        client = mock_anthropic(["not JSON"])
        result = disambiguate(
            client, _make_concept(),
            original_question="x",
        )
        assert result is None

    def test_invalid_confidence_returns_none(self):
        client = mock_anthropic([json.dumps({
            "input_name": "lactate",
            "canonical_name": "lactate",
            "confidence": "BOGUS_LEVEL",
        })])
        result = disambiguate(
            client, _make_concept(),
            original_question="x",
        )
        assert result is None


class TestDisambiguateAntiHallucination:
    def test_only_observed_citations_kept(self, monkeypatch):
        # Mock pubmed_search to return one PMID.
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": [
                {"pmid": "REAL", "title": "Real study", "url": "u"},
            ]}

        monkeypatch.setattr(
            "src.conversational.clinical_consult.pubmed_search",
            fake_pubmed,
        )

        # Model: tool_use → end_turn citing both REAL and FAKE PMIDs.
        client = mock_anthropic([
            {
                "tool_use": [{"id": "tu1", "name": "pubmed_search",
                              "input": {"query": "lactate sepsis"}}],
                "stop_reason": "tool_use",
            },
            _disamb_response(
                canonical_name="serum lactate",
                confidence="high",
                resolved_code="32693-4",
                code_system="loinc",
                cited_sources=[
                    {"type": "pubmed", "pmid": "REAL", "title": "Real"},
                    {"type": "pubmed", "pmid": "FAKE", "title": "Hallucinated"},
                ],
            ),
        ])
        result = disambiguate(
            client, _make_concept(),
            original_question="x",
        )
        assert result is not None
        assert result.citations is not None
        kept_ids = [c["pmid"] for c in result.citations]
        assert "REAL" in kept_ids
        assert "FAKE" not in kept_ids


# ---------------------------------------------------------------------------
# clarify
# ---------------------------------------------------------------------------


class TestClarifyHappyPath:
    def test_returns_clarifying_message_with_alternates(self):
        client = mock_anthropic([_clarify_response(
            text="Did you mean serum lactate or CSF lactate?",
            alternates_offered=["serum lactate", "CSF lactate"],
        )])
        partial = [
            Disambiguation(
                input_name="lactate",
                canonical_name="lactate",
                alternates=["serum lactate", "CSF lactate"],
                confidence="low",
            ),
        ]
        result = clarify(
            client, "what is the lactate?", "specimen ambiguous", partial,
        )
        assert result is not None
        assert isinstance(result, ClarifyingMessage)
        assert "serum" in result.text.lower()
        assert "serum lactate" in result.alternates_offered

    def test_works_with_empty_partial_disambiguations(self):
        client = mock_anthropic([_clarify_response(
            text="Could you specify which patient cohort?",
        )])
        result = clarify(client, "what's the level?", "vague", [])
        assert result is not None
        assert "cohort" in result.text


class TestClarifyFailureModes:
    def test_api_error_returns_none(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = clarify(client, "x", "y", [])
        assert result is None

    def test_malformed_json_returns_none(self):
        client = mock_anthropic(["not JSON"])
        result = clarify(client, "x", "y", [])
        assert result is None


# ---------------------------------------------------------------------------
# contextualize
# ---------------------------------------------------------------------------


class TestContextualizeHappyPath:
    def test_returns_contextual_note(self):
        client = mock_anthropic([_ctx_response(
            text="Mean lactate of 2.1 mmol/L is in line with published "
                 "sepsis-cohort distributions.",
        )])
        result = contextualize(client, _make_answer(), _make_cq())
        assert result is not None
        assert isinstance(result, ContextualNote)
        assert "2.1" in result.text or "lactate" in result.text.lower()

    def test_empty_text_returns_none(self):
        """If the model decides there's nothing useful to add, it returns
        an empty text. The caller should treat that as 'no note'."""
        client = mock_anthropic([_ctx_response(text="")])
        result = contextualize(client, _make_answer(), _make_cq())
        assert result is None


class TestContextualizeFailureModes:
    def test_api_error_returns_none(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = contextualize(client, _make_answer(), _make_cq())
        assert result is None

    def test_malformed_json_returns_none(self):
        client = mock_anthropic(["not JSON"])
        result = contextualize(client, _make_answer(), _make_cq())
        assert result is None


class TestContextualizeAntiHallucination:
    def test_only_observed_citations_kept(self, monkeypatch):
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": [
                {"pmid": "REAL", "title": "Real", "url": "u"},
            ]}

        monkeypatch.setattr(
            "src.conversational.clinical_consult.pubmed_search",
            fake_pubmed,
        )

        client = mock_anthropic([
            {
                "tool_use": [{"id": "tu1", "name": "pubmed_search",
                              "input": {"query": "lactate sepsis"}}],
                "stop_reason": "tool_use",
            },
            _ctx_response(
                text="Lactate 2.1 is typical for sepsis (PMID REAL).",
                cited_sources=[
                    {"type": "pubmed", "pmid": "REAL"},
                    {"type": "pubmed", "pmid": "FAKE"},
                ],
            ),
        ])
        result = contextualize(client, _make_answer(), _make_cq())
        assert result is not None
        assert result.citations is not None
        kept = [c["pmid"] for c in result.citations]
        assert "REAL" in kept
        assert "FAKE" not in kept
