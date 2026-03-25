"""Tests for the conversational pipeline orchestrator."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# anthropic is not installed in the test environment — inject a mock module
# so that `import anthropic` inside ConversationalPipeline.__init__ succeeds.
sys.modules.setdefault("anthropic", MagicMock())

from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionResult,
    TemporalConstraint,
)
from src.conversational.orchestrator import ConversationalPipeline
from src.conversational.reasoner import ReasoningResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DB_PATH = Path("/tmp/test.duckdb")
_ONTOLOGY_DIR = Path("/tmp/ontology")


def _make_cq(question: str = "What is the creatinine?") -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question=question,
        clinical_concepts=[
            ClinicalConcept(name="Creatinine", concept_type="biomarker"),
        ],
    )


def _make_answer(summary: str = "Creatinine was 1.1 mg/dL.") -> AnswerResult:
    return AnswerResult(text_summary=summary)


def _make_reasoning() -> ReasoningResult:
    return ReasoningResult(
        rows=[{"value": 1.1, "unit": "mg/dL"}],
        columns=["value", "unit"],
        sparql_queries=["SELECT ?value ..."],
        template_names=["value_lookup"],
    )


def _patch_all(fn):
    """Stack all 5 stage patches needed for orchestrator tests."""
    fn = patch("src.conversational.orchestrator.decompose")(fn)
    fn = patch("src.conversational.orchestrator.extract")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    return fn


def _setup_mocks(mock_decompose, mock_extract, mock_build, mock_reason, mock_answer):
    """Wire up default return values for all stage mocks."""
    cq = _make_cq()
    reasoning = _make_reasoning()
    answer = _make_answer()

    mock_decompose.return_value = cq
    mock_extract.return_value = ExtractionResult()
    mock_build.return_value = (MagicMock(), {"triples": 50})
    mock_reason.return_value = reasoning
    mock_answer.return_value = answer

    return cq, reasoning, answer


# ---------------------------------------------------------------------------
# TestAsk
# ---------------------------------------------------------------------------


class TestAsk:
    @_patch_all
    def test_single_question_flow(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        cq, reasoning, answer = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("What is the creatinine?")

        assert result is answer
        mock_decompose.assert_called_once_with(
            pipeline._client, "What is the creatinine?",
            conversation_history=None,
        )
        mock_extract.assert_called_once()
        extract_args = mock_extract.call_args
        assert extract_args[0] == (_DB_PATH, cq)
        assert extract_args[1]["config"] is None
        mock_build.assert_called_once()
        mock_reason.assert_called_once()
        mock_answer.assert_called_once_with(
            pipeline._client, cq, reasoning.rows,
            {"triples": 50}, reasoning.sparql_queries,
        )
        assert len(pipeline.conversation_history) == 1
        assert pipeline.conversation_history[0] == (cq, answer)

    @_patch_all
    def test_follow_up_passes_history(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        cq1 = _make_cq("What is the creatinine?")
        cq2 = _make_cq("Now show sodium")
        answer1 = _make_answer("Creatinine was 1.1")
        answer2 = _make_answer("Sodium was 140")
        reasoning = _make_reasoning()

        mock_decompose.side_effect = [cq1, cq2]
        mock_extract.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {})
        mock_reason.return_value = reasoning
        mock_answer.side_effect = [answer1, answer2]

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("What is the creatinine?")
        pipeline.ask("Now show sodium")

        second_call = mock_decompose.call_args_list[1]
        assert second_call.kwargs["conversation_history"] == [(cq1, answer1)]

    @_patch_all
    def test_error_returns_error_answer(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        mock_decompose.side_effect = RuntimeError("LLM failed")

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("bad question")

        assert isinstance(result, AnswerResult)
        assert "error" in result.text_summary.lower()
        assert result.data_table is None
        assert len(pipeline.conversation_history) == 0

    @_patch_all
    def test_max_history_trimming(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.max_history = 3

        for i in range(5):
            pipeline.ask(f"Question {i}")

        assert len(pipeline.conversation_history) == 3


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    @_patch_all
    def test_reset_clears_history(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("First question")
        assert len(pipeline.conversation_history) == 1

        pipeline.reset()
        assert len(pipeline.conversation_history) == 0


# ---------------------------------------------------------------------------
# TestAllenRelationsPassthrough
# ---------------------------------------------------------------------------


class TestAllenRelationsPassthrough:
    @_patch_all
    def test_no_temporal_constraints_skips_allen(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        """CQ without temporal_constraints passes skip_allen_relations=True."""
        cq = CompetencyQuestion(
            original_question="What is the creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        mock_decompose.return_value = cq
        mock_extract.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {})
        mock_reason.return_value = _make_reasoning()
        mock_answer.return_value = _make_answer()

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("What is the creatinine?")

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("skip_allen_relations") is True

    @_patch_all
    def test_temporal_constraints_computes_allen(
        self, mock_decompose, mock_extract,
        mock_build, mock_reason, mock_answer,
    ):
        """CQ with temporal_constraints passes skip_allen_relations=False."""
        cq = CompetencyQuestion(
            original_question="Creatinine before intubation",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            temporal_constraints=[
                TemporalConstraint(relation="before", reference_event="intubation"),
            ],
        )
        mock_decompose.return_value = cq
        mock_extract.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {})
        mock_reason.return_value = _make_reasoning()
        mock_answer.return_value = _make_answer()

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("Creatinine before intubation")

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("skip_allen_relations") is False
