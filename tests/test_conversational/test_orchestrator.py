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
    """Stack all 5 stage patches needed for orchestrator tests.

    Phase 4.5: the orchestrator calls ``decompose_question`` (returns a
    ``DecompositionResult``) instead of the legacy single-CQ ``decompose``.
    Tests keep their argument name ``mock_decompose`` but what they mock is
    now the question-level entrypoint; ``_setup_mocks`` wraps the returned
    CQ in a single-CQ DecompositionResult so legacy test bodies work
    unchanged.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator.extract")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    return fn


def _setup_mocks(mock_decompose, mock_extract, mock_build, mock_reason, mock_answer):
    """Wire up default return values for all stage mocks.

    For backward compat with legacy test bodies that access
    ``mock_decompose.return_value`` as if it were a bare CompetencyQuestion,
    we also attach the CQ directly (via the wrapping DecompositionResult's
    first element). Tests that mutate attributes on the CQ still see the
    change because both references point at the same object.
    """
    from src.conversational.models import DecompositionResult

    cq = _make_cq()
    reasoning = _make_reasoning()
    answer = _make_answer()

    mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
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
        from src.conversational.models import DecompositionResult

        cq1 = _make_cq("What is the creatinine?")
        cq2 = _make_cq("Now show sodium")
        answer1 = _make_answer("Creatinine was 1.1")
        answer2 = _make_answer("Sodium was 140")
        reasoning = _make_reasoning()

        mock_decompose.side_effect = [
            DecompositionResult(competency_questions=[cq1]),
            DecompositionResult(competency_questions=[cq2]),
        ]
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
        from src.conversational.models import DecompositionResult

        cq = CompetencyQuestion(
            original_question="What is the creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
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
        from src.conversational.models import DecompositionResult

        cq = CompetencyQuestion(
            original_question="Creatinine before intubation",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            temporal_constraints=[
                TemporalConstraint(relation="before", reference_event="intubation"),
            ],
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_extract.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {})
        mock_reason.return_value = _make_reasoning()
        mock_answer.return_value = _make_answer()

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("Creatinine before intubation")

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("skip_allen_relations") is False


# ---------------------------------------------------------------------------
# Phase 4: interpretation echo + clarifying-question short-circuit
# ---------------------------------------------------------------------------


class TestPhase4InterpretationAndClarify:
    @_patch_all
    def test_clarifying_question_short_circuits_pipeline(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """When the decomposer sets ``clarifying_question``, extract / graph /
        reason / answer must NOT run. The returned AnswerResult's body is the
        clarifying question verbatim."""
        from src.conversational.models import DecompositionResult

        cq = _make_cq("show me the labs")
        cq.clinical_concepts = []
        cq.interpretation_summary = "Unclear: no specific lab or cohort provided."
        cq.clarifying_question = "Which lab and which patients?"
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        # Booby-trap: if downstream stages run, these will raise.
        mock_extract.side_effect = AssertionError("extract must not run on clarify short-circuit")
        mock_build.side_effect = AssertionError("graph build must not run on clarify short-circuit")
        mock_reason.side_effect = AssertionError("reason must not run on clarify short-circuit")
        mock_answer.side_effect = AssertionError("answer must not run on clarify short-circuit")

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("show me the labs")

        assert result.clarifying_question == "Which lab and which patients?"
        assert result.interpretation_summary == "Unclear: no specific lab or cohort provided."
        assert result.text_summary == "Which lab and which patients?"
        # The turn is still recorded so follow-ups can reference it.
        assert len(pipeline.conversation_history) == 1

    @_patch_all
    def test_empty_clarifying_question_runs_full_pipeline(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """``clarifying_question=None`` and ``clarifying_question=""`` must both
        fall through to the full pipeline — the short-circuit only fires on
        a non-empty string."""
        _setup_mocks(mock_decompose, mock_extract, mock_build, mock_reason, mock_answer)
        # Decomposer returned CQ with no clarifying_question (None).
        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("avg creatinine")

        mock_extract.assert_called_once()
        mock_build.assert_called_once()
        mock_reason.assert_called_once()
        mock_answer.assert_called_once()

    @_patch_all
    def test_whitespace_only_clarifying_question_does_not_trigger_short_circuit(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """LLMs sometimes emit ``"   "`` when "no clarifying question" is meant.
        The orchestrator's ``.strip()`` guard means whitespace-only values
        fall through to the full pipeline."""
        cq, _, _ = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )
        cq.clarifying_question = "   \n  "
        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("avg creatinine")

        mock_extract.assert_called_once()

    @_patch_all
    def test_interpretation_summary_propagated_to_answer(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """The decomposer's interpretation must flow onto the AnswerResult
        before the answerer returns, so the UI can echo it above the summary."""
        cq, _, _ = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )
        cq.interpretation_summary = "Mean serum creatinine over the cohort."
        # answerer returns a fresh AnswerResult — no interpretation on it yet.
        fresh_answer = AnswerResult(text_summary="1.1 mg/dL mean.")
        mock_answer.return_value = fresh_answer

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("avg creatinine")

        assert result.interpretation_summary == "Mean serum creatinine over the cohort."


# ---------------------------------------------------------------------------
# Phase 4.5 — multi-CQ (big-question) orchestration
# ---------------------------------------------------------------------------


def _patch_all_multi(fn):
    """Stack patches for the multi-CQ orchestrator path.

    ``decompose_question`` is the new entrypoint; ``merge_extractions`` is the
    new helper. Both need to be patchable so we can verify the multi-CQ flow
    without a real graph or real LLM.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator.extract")(fn)
    fn = patch("src.conversational.orchestrator.merge_extractions")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    return fn


def _make_decomp(*cqs, narrative=None):
    from src.conversational.models import DecompositionResult

    return DecompositionResult(
        narrative=narrative,
        competency_questions=list(cqs),
    )


class TestMultiCQOrchestration:
    """For a big-question (Shape B) turn:
      - extract() is called once per sub-CQ
      - merge_extractions() unions those into one ExtractionResult
      - build_query_graph() is called EXACTLY ONCE on the merged result
      - reason() is called once per sub-CQ against the shared graph
      - generate_answer() is called once per sub-CQ
      - the returned AnswerResult has sub_answers=[sub_1, …, sub_n] and its
        top-level text_summary reflects the narrative
    """

    @_patch_all_multi
    def test_big_question_builds_ONE_graph_and_reasons_per_cq(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
    ):
        cq1 = _make_cq("count readmitted sepsis")
        cq2 = _make_cq("compare labs")
        mock_decompose_q.return_value = _make_decomp(
            cq1, cq2,
            narrative="Break down: cohort size, then lab comparisons.",
        )

        # Each extract() call returns a distinct ExtractionResult.
        mock_extract.side_effect = [ExtractionResult(), ExtractionResult()]
        merged_marker = ExtractionResult()
        mock_merge.return_value = merged_marker
        mock_build.return_value = (MagicMock(), {"triples": 99})
        mock_reason.side_effect = [_make_reasoning(), _make_reasoning()]
        mock_answer.side_effect = [
            _make_answer("sub-answer 1"),
            _make_answer("sub-answer 2"),
        ]

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("why do sepsis pts get readmitted")

        # Two extractions merged once into one build_query_graph call.
        assert mock_extract.call_count == 2
        mock_merge.assert_called_once()
        merged_arg = mock_merge.call_args.args[0]
        assert len(merged_arg) == 2  # list of 2 ExtractionResults
        mock_build.assert_called_once()
        build_call = mock_build.call_args
        # Second positional or kwarg: the merged extraction object identity.
        passed_extraction = (
            build_call.args[1] if len(build_call.args) > 1 else build_call.kwargs["extraction"]
        )
        assert passed_extraction is merged_marker

        # Reason was called once per sub-CQ, against the same graph.
        assert mock_reason.call_count == 2
        graphs_reasoned_on = [call.args[0] for call in mock_reason.call_args_list]
        assert graphs_reasoned_on[0] is graphs_reasoned_on[1]

        # generate_answer called once per sub-CQ.
        assert mock_answer.call_count == 2

        # Result carries sub_answers and the narrative.
        assert result.sub_answers is not None
        assert len(result.sub_answers) == 2
        assert result.sub_answers[0].text_summary == "sub-answer 1"
        assert result.sub_answers[1].text_summary == "sub-answer 2"
        assert "Break down" in (result.text_summary or "")

    @_patch_all_multi
    def test_skip_allen_is_false_when_any_sub_cq_has_temporal(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
    ):
        """If ANY sub-CQ has temporal constraints, the shared graph must be
        built WITH Allen relations so all sub-CQs can query them."""
        cq_no_temporal = _make_cq("count")
        cq_with_temporal = _make_cq("creatinine during icu")
        cq_with_temporal.temporal_constraints = [
            TemporalConstraint(relation="during", reference_event="ICU stay"),
        ]
        mock_decompose_q.return_value = _make_decomp(cq_no_temporal, cq_with_temporal)
        mock_extract.side_effect = [ExtractionResult(), ExtractionResult()]
        mock_merge.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {})
        mock_reason.side_effect = [_make_reasoning(), _make_reasoning()]
        mock_answer.side_effect = [_make_answer(), _make_answer()]

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        pipeline.ask("mixed question")

        _, kwargs = mock_build.call_args
        assert kwargs.get("skip_allen_relations") is False

    @_patch_all_multi
    def test_single_cq_path_does_not_call_merge_or_wrap(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
    ):
        """Shape A decomposition (1 CQ, no narrative): don't bother calling
        merge_extractions, and return the sub-answer directly without
        wrapping it in a synthesised top-level shell."""
        cq = _make_cq()
        mock_decompose_q.return_value = _make_decomp(cq)
        mock_extract.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {})
        mock_reason.return_value = _make_reasoning()
        leaf_answer = _make_answer("single answer")
        mock_answer.return_value = leaf_answer

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("avg creatinine")

        # Merge is NOT called when there's only one extraction — single-CQ
        # fast path keeps the legacy shape.
        mock_merge.assert_not_called()
        assert result.sub_answers is None
        assert result.text_summary == "single answer"

    @_patch_all_multi
    def test_any_clarifying_question_short_circuits_multi_cq(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
    ):
        """If the decomposer returns Shape B but ANY sub-CQ has a
        clarifying_question set, treat the whole turn as a clarify: surface
        the first clarifying sub-CQ's question and skip all downstream
        stages."""
        cq_ok = _make_cq("count")
        cq_ambig = _make_cq("wtf")
        cq_ambig.clarifying_question = "Which cohort exactly?"
        cq_ambig.interpretation_summary = "Unclear."
        mock_decompose_q.return_value = _make_decomp(
            cq_ok, cq_ambig,
            narrative="two parts",
        )
        # Booby-trap every downstream stage.
        mock_extract.side_effect = AssertionError("extract must not run")
        mock_merge.side_effect = AssertionError("merge must not run")
        mock_build.side_effect = AssertionError("build must not run")
        mock_reason.side_effect = AssertionError("reason must not run")
        mock_answer.side_effect = AssertionError("answer must not run")

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("big ambiguous question")

        assert result.clarifying_question == "Which cohort exactly?"
        assert result.text_summary == "Which cohort exactly?"
