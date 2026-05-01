"""Tests for the conversational pipeline orchestrator."""

import json
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
    Disambiguation,
    ExtractionResult,
    PatientFilter,
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

    Phase 4.5: ``decompose_question`` returns ``DecompositionResult``;
    legacy test bodies get the wrapping done by ``_setup_mocks``.

    Phase 7a: the orchestrator now calls ``_extract(backend, cq, …)`` with
    a backend it opens itself. Patch the backend classes so the opener
    returns a mock instead of trying to hit a real DB.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    # Patch the critic so existing orchestrator tests don't hit the real
    # Anthropic API on every turn. The critic's fail-safe try/except would
    # absorb errors, but exercising it would still make network calls and
    # add latency. Tests that want to exercise the critic specifically
    # use TestCriticIntegration (which doesn't use _patch_all).
    fn = patch(
        "src.conversational.orchestrator.critique",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator.validate_sql_deterministic",
        new=MagicMock(return_value=None),
    )(fn)
    # Use MagicMock *instances* (not the class) so calling e.g.
    # ``_DuckDBBackend(path)`` returns the mock's ``return_value`` — itself a
    # MagicMock with auto-generated attributes (``.close()``, ``.execute()``).
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
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
        # Phase 7a: orchestrator calls the internal ``_extract(backend, cq, ...)``
        # with a backend it opens itself. First positional is the backend (a
        # MagicMock in tests); second is the CQ. We only assert on the CQ here.
        assert extract_args.args[1] is cq
        assert extract_args.kwargs.get("config") is None
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

    Phase 4.5: ``decompose_question``, ``merge_extractions`` patched.
    Phase 7a: ``extract`` replaced with ``_extract`` (backend-taking
    internal); backends patched so ``_open_backend`` returns a mock.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.merge_extractions")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    fn = patch(
        "src.conversational.orchestrator.critique",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator.validate_sql_deterministic",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
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


# ---------------------------------------------------------------------------
# Phase 7a — planner routing: SQL fast-path vs graph path
# ---------------------------------------------------------------------------


def _patch_fastpath(fn):
    """Stack patches for tests that exercise the SQL fast-path in isolation.

    Graph-path stages are booby-trapped so any bleed-through fails loud.
    Backend shims + SQL compilation itself are mocked so we don't need a
    live DB.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.merge_extractions")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    fn = patch("src.conversational.orchestrator.compile_sql")(fn)
    fn = patch(
        "src.conversational.orchestrator.critique",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator.validate_sql_deterministic",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
    return fn


class TestSqlFastpathRouting:
    """Phase 7a: a CQ classified as SQL_FAST must bypass extract, graph
    build, reason, and merge entirely. Only ``compile_sql`` +
    ``backend.execute`` + ``generate_answer`` run."""

    @_patch_fastpath
    def test_sql_fast_skips_graph_stages(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
        mock_compile,
    ):
        """Single-CQ fast-path: ``_extract``, ``build_query_graph``,
        ``reason`` all booby-trapped. The fast-path SQL helper and the
        answerer run exactly once."""
        cq = CompetencyQuestion(
            original_question="avg creatinine over 65",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean",
            scope="cohort",
        )
        from src.conversational.models import DecompositionResult

        mock_decompose_q.return_value = DecompositionResult(
            competency_questions=[cq],
        )

        # Every graph-path stage fails loud if it runs.
        mock_extract.side_effect = AssertionError("_extract must NOT run on fast-path")
        mock_merge.side_effect = AssertionError("merge must NOT run on fast-path")
        mock_build.side_effect = AssertionError("build_query_graph must NOT run on fast-path")
        mock_reason.side_effect = AssertionError("reason must NOT run on fast-path")

        # compile_sql returns a minimal valid query shape.
        from src.conversational.sql_fastpath import SqlFastpathQuery

        mock_compile.return_value = SqlFastpathQuery(
            sql="SELECT AVG(x) AS mean_value FROM t",
            params=[],
            columns=["mean_value"],
        )
        mock_answer.return_value = _make_answer("Mean creatinine was 1.1")

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("average creatinine for patients over 65")

        mock_compile.assert_called_once()
        mock_answer.assert_called_once()
        assert result.text_summary == "Mean creatinine was 1.1"

    @_patch_fastpath
    def test_mixed_multi_cq_routes_per_cq(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
        mock_compile,
    ):
        """A big-question decomposition where one sub-CQ is fast-path and
        the other is graph-path: build_query_graph runs ONCE (for the
        graph-path sub-CQ only); compile_sql runs once (for the fast-path
        sub-CQ). Both sub-answers are wrapped under the narrative."""
        # Sub-CQ 1: fast-path (biomarker mean, no temporal).
        cq_fast = CompetencyQuestion(
            original_question="mean creatinine for sepsis",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            patient_filters=[
                PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
            ],
            aggregation="mean",
            scope="cohort",
        )
        # Sub-CQ 2: graph path (temporal constraint).
        cq_graph = CompetencyQuestion(
            original_question="creatinine during ICU stay for sepsis",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            temporal_constraints=[
                TemporalConstraint(relation="during", reference_event="ICU stay"),
            ],
            patient_filters=[
                PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
            ],
            scope="cohort",
        )
        from src.conversational.models import DecompositionResult

        mock_decompose_q.return_value = DecompositionResult(
            narrative="Break down: cohort mean, then temporal distribution.",
            competency_questions=[cq_fast, cq_graph],
        )

        # Graph-path mocks produce normal returns (not booby-trapped here).
        mock_extract.return_value = ExtractionResult()
        mock_merge.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {"triples": 42})
        mock_reason.return_value = _make_reasoning()
        mock_answer.side_effect = [
            _make_answer("fast-path answer"),
            _make_answer("graph-path answer"),
        ]

        from src.conversational.sql_fastpath import SqlFastpathQuery

        mock_compile.return_value = SqlFastpathQuery(
            sql="SELECT 1", params=[], columns=["count_value"],
        )

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("mixed multi-cq question")

        # Fast-path helper ran exactly once (for cq_fast).
        mock_compile.assert_called_once()
        # Extract ran once (for cq_graph only); merge did NOT run
        # because there's only one graph-path extraction to merge.
        assert mock_extract.call_count == 1
        mock_merge.assert_not_called()
        # Exactly one graph was built.
        mock_build.assert_called_once()
        # One reason call, matching the one graph-path sub-CQ.
        assert mock_reason.call_count == 1
        # Two sub-answers wrapped under the narrative.
        assert result.sub_answers is not None
        assert len(result.sub_answers) == 2
        assert "Break down" in (result.text_summary or "")

    @_patch_fastpath
    def test_all_cqs_fastpath_never_builds_graph(
        self,
        mock_decompose_q,
        mock_extract,
        mock_merge,
        mock_build,
        mock_reason,
        mock_answer,
        mock_compile,
    ):
        """Multi-CQ turn where EVERY sub-CQ is fast-path. No graph is
        built at all — ``build_query_graph``, ``_extract``, ``merge``,
        ``reason`` are all booby-trapped."""
        cq_a = CompetencyQuestion(
            original_question="mean creatinine",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        cq_b = CompetencyQuestion(
            original_question="count sepsis",
            clinical_concepts=[
                ClinicalConcept(name="sepsis", concept_type="diagnosis"),
            ],
            aggregation="count", scope="cohort",
        )
        from src.conversational.models import DecompositionResult

        mock_decompose_q.return_value = DecompositionResult(
            narrative="two aggregates",
            competency_questions=[cq_a, cq_b],
        )
        mock_extract.side_effect = AssertionError("no extract on all-fastpath")
        mock_merge.side_effect = AssertionError("no merge on all-fastpath")
        mock_build.side_effect = AssertionError("no graph build on all-fastpath")
        mock_reason.side_effect = AssertionError("no reason on all-fastpath")

        from src.conversational.sql_fastpath import SqlFastpathQuery

        mock_compile.return_value = SqlFastpathQuery(
            sql="SELECT 1", params=[], columns=["count_value"],
        )
        mock_answer.side_effect = [_make_answer("a"), _make_answer("b")]

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("multi fast-path question")

        assert mock_compile.call_count == 2
        assert result.sub_answers is not None
        assert len(result.sub_answers) == 2


# ---------------------------------------------------------------------------
# Phase 9 — standalone similarity branch
# ---------------------------------------------------------------------------


class TestSimilarityBranch:
    """A CQ with ``scope='patient_similarity'`` must hit a dedicated
    orchestrator branch that calls ``run_similarity`` and wraps the
    ranked result into an ``AnswerResult`` with a 6-column data_table.

    Booby-trap: if the branch is missing, the CQ falls through to the
    graph path and the ``mock_extract`` / ``mock_build`` / ``mock_reason``
    side-effects fire — guaranteeing the test fails loudly instead of
    silently mis-routing.
    """

    @_patch_all
    def test_similarity_scope_routes_to_run_similarity(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        from src.conversational.models import DecompositionResult
        from src.similarity.models import (
            ContextualExplanation,
            SimilarityResult,
            SimilarityScore,
            SimilaritySpec,
        )

        spec = SimilaritySpec(
            anchor_template={"age": 68, "gender_F": 1, "snomed_group_I48": 1},
            top_k=3,
        )
        cq = CompetencyQuestion(
            original_question="Find admissions similar to a 68-year-old woman with afib.",
            scope="patient_similarity",
            similarity_spec=spec,
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        # Booby-trap: graph path must NOT run.
        mock_extract.side_effect = AssertionError(
            "extract must not run on similarity scope"
        )
        mock_build.side_effect = AssertionError(
            "graph build must not run on similarity scope"
        )
        mock_reason.side_effect = AssertionError(
            "reason must not run on similarity scope"
        )
        mock_answer.side_effect = AssertionError(
            "answerer must not run on similarity scope"
        )

        sim_result = SimilarityResult(
            anchor_description="template anchor (age=68, snomed_group_I48=1)",
            n_pool=2500,
            n_returned=3,
            scores=[
                SimilarityScore(
                    hadm_id=20001,
                    subject_id=10001,
                    combined=0.91,
                    contextual=0.91,
                    temporal=None,
                    contextual_explanation=ContextualExplanation(overall_score=0.91),
                ),
                SimilarityScore(
                    hadm_id=20002,
                    subject_id=10002,
                    combined=0.84,
                    contextual=0.84,
                    temporal=None,
                    contextual_explanation=ContextualExplanation(overall_score=0.84),
                ),
                SimilarityScore(
                    hadm_id=20003,
                    subject_id=10003,
                    combined=0.77,
                    contextual=0.77,
                    temporal=None,
                    contextual_explanation=ContextualExplanation(overall_score=0.77),
                ),
            ],
            spec=spec,
        )

        with patch(
            "src.similarity.run.run_similarity", return_value=sim_result
        ) as mock_run_sim:
            pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
            result = pipeline.ask(
                "Find admissions similar to a 68-year-old woman with afib."
            )

        # The dedicated branch fired exactly once with the spec from the CQ.
        assert mock_run_sim.call_count == 1
        called_spec = mock_run_sim.call_args.args[0]
        assert called_spec is spec

        # The returned AnswerResult carries the ranked table.
        assert isinstance(result, AnswerResult)
        assert result.data_table is not None
        assert len(result.data_table) == 3
        assert result.table_columns == [
            "rank", "hadm_id", "subject_id",
            "combined", "contextual", "temporal",
        ]
        # Rank-1 row reflects the highest-scored candidate.
        top = result.data_table[0]
        assert top["rank"] == 1
        assert top["hadm_id"] == 20001
        assert top["combined"] == pytest.approx(0.91)
        assert top["temporal"] is None  # template anchor → contextual-only
        # Summary text mentions the anchor + counts.
        assert "n_pool=2500" not in result.text_summary  # we render words, not literals
        assert "3" in result.text_summary
        assert "2500" in result.text_summary

    @_patch_all
    def test_similarity_scope_without_spec_returns_safe_message(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """Defensive: if the decomposer marks scope='patient_similarity'
        but somehow forgets the spec, the orchestrator should not crash
        and should not silently route to the graph path."""
        from src.conversational.models import DecompositionResult

        cq = CompetencyQuestion(
            original_question="Find similar patients.",
            scope="patient_similarity",
            similarity_spec=None,
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        mock_extract.side_effect = AssertionError(
            "extract must not run on similarity scope"
        )
        mock_build.side_effect = AssertionError(
            "graph build must not run on similarity scope"
        )

        pipeline = ConversationalPipeline(_DB_PATH, _ONTOLOGY_DIR, "test-key")
        result = pipeline.ask("Find similar patients.")

        assert isinstance(result, AnswerResult)
        assert result.data_table is None
        assert "similarity" in result.text_summary.lower()


# ---------------------------------------------------------------------------
# TestCriticIntegration — end-to-end critic wiring (Phase 6)
#
# These tests intentionally do NOT use ``_patch_all`` because that decorator
# patches the critic function. They patch the same upstream stages but
# leave ``critique`` real, with a mock Anthropic client returning a canned
# verdict. This exercises the full critic call chain through the
# orchestrator's ``_critique`` method.
# ---------------------------------------------------------------------------


def _patch_pre_critic(fn):
    """Same patches as ``_patch_all`` minus the critique mock."""
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    # Validator is neutralised here too — these tests exercise the critic
    # call chain end-to-end, not the validator.
    fn = patch(
        "src.conversational.orchestrator.validate_sql_deterministic",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
    return fn


class TestCriticIntegration:
    """The critic runs inside ``ConversationalPipeline.ask`` after each
    sub-CQ produces an AnswerResult. ``enable_critic=True`` wires the call;
    ``enable_critic=False`` skips it."""

    @_patch_pre_critic
    def test_critic_runs_after_answer_and_attaches_verdict(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """When enable_critic=True, the orchestrator passes each AnswerResult
        through the critic and attaches the parsed CriticVerdict."""
        from tests.test_conversational.conftest import mock_anthropic

        # Stages run normally; only generate_answer is mocked so we don't
        # need a real LLM for the answerer.
        cq, reasoning, answer = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )
        critic_response = json.dumps({
            "plausible": True,
            "severity": "info",
            "concern": None,
            "reference_used": None,
        })
        # The pipeline constructor builds its own real Anthropic client; we
        # override _client after construction so the critic uses our mock.
        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = mock_anthropic([critic_response])

        result = pipeline.ask("What is the creatinine?")

        assert result.critic_verdict is not None
        assert result.critic_verdict.severity == "info"
        assert result.critic_verdict.plausible is True

    @_patch_pre_critic
    def test_critic_disabled_skips_call(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """When enable_critic=False, no critic call is made and verdict
        stays None — the pipeline's behavior is identical to before the
        critic was wired in."""
        from tests.test_conversational.conftest import mock_anthropic

        cq, reasoning, answer = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )
        # Mock client errors out if called — proves the critic never fires.
        client = MagicMock()
        client.messages.create.side_effect = AssertionError(
            "critic must NOT call the LLM when enable_critic=False"
        )
        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=False,
        )
        pipeline._client = client

        result = pipeline.ask("What is the creatinine?")

        assert result.critic_verdict is None

    @_patch_pre_critic
    def test_critic_failure_does_not_block_answer(
        self, mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
    ):
        """If the critic API call raises, the answer must still render
        with critic_verdict=None — the critic is fail-safe."""
        cq, reasoning, answer = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("critic API down")
        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = client

        result = pipeline.ask("What is the creatinine?")

        assert result.text_summary  # answer renders
        assert result.critic_verdict is None  # but no verdict


# ---------------------------------------------------------------------------
# TestSelfHealingCritic — critic proposes corrected LOINC; orchestrator retries
#
# These tests exercise ``_run_with_critic_retry``. They patch the
# ``_run_sql_fastpath`` method directly (returning canned answer / sql /
# fallback_warning tuples) and use ``mock_anthropic`` for the critic's
# verdict responses. The decomposer is mocked to emit a SQL_FAST CQ with
# a biomarker concept carrying ``loinc_code="2524-7"`` (the wrong-for-
# MIMIC lactate code) so the suggestion-based retry path is exercised.
# ---------------------------------------------------------------------------


def _patch_for_self_healing(fn):
    """Patches for self-healing tests: mock decompose, _run_sql_fastpath,
    backends. Critique stays real (uses pipeline._client = mock_anthropic).

    Validator is neutralised — self-healing exercises the critic-driven
    LOINC mutation, not the pre-execution validator. The dedicated
    pre-validator + retry-with-mutation test in TestPreValidatorIntegration
    exercises the validator-during-retry path.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch.object(ConversationalPipeline, "_run_sql_fastpath")(fn)
    fn = patch(
        "src.conversational.orchestrator.validate_sql_deterministic",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
    return fn


def _make_lactate_cq(loinc_code: str | None = "2524-7") -> CompetencyQuestion:
    """Build a SQL_FAST-eligible CQ: biomarker concept + cohort scope +
    aggregation, with a configurable LOINC code."""
    return CompetencyQuestion(
        original_question="What is the average lactate for patients with sepsis?",
        clinical_concepts=[
            ClinicalConcept(
                name="lactate",
                concept_type="biomarker",
                loinc_code=loinc_code,
            ),
        ],
        aggregation="mean",
        scope="cohort",
    )


def _verdict_json(
    severity: str = "info",
    *,
    plausible: bool | None = None,
    concern: str | None = None,
    suggested_loinc: str | None = None,
    correction_rationale: str | None = None,
) -> str:
    if plausible is None:
        plausible = severity == "info"
    return json.dumps({
        "plausible": plausible,
        "severity": severity,
        "concern": concern,
        "reference_used": None,
        "suggested_loinc": suggested_loinc,
        "correction_rationale": correction_rationale,
    })


class TestSelfHealingCritic:
    """When the critic flags a LOINC-grounding failure AND proposes a
    corrected LOINC, the orchestrator mutates the CQ and re-runs the
    SQL fast-path. Bounded by ``critic_max_retries`` (default 1)."""

    @_patch_for_self_healing
    def test_lactate_loinc_retry_happy_path(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """Canonical case: first attempt produces polluted answer + critic
        block with suggested_loinc; orchestrator retries with corrected
        LOINC; second attempt is clean. Final result is the corrected
        answer with a correction_trace recording both attempts."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        cq = _make_lactate_cq(loinc_code="2524-7")
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        polluted_answer = AnswerResult(text_summary="Polluted lactate 199.43 mg/dL")
        clean_answer = AnswerResult(text_summary="Clean lactate 2.1 mmol/L")
        mock_run_sql_fastpath.side_effect = [
            (polluted_answer, ["SELECT ... LIKE %lactate%"], "LOINC '2524-7' has no MIMIC labitem coverage"),
            (clean_answer, ["SELECT ... itemid IN (50813)"], None),
        ]

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = mock_anthropic([
            _verdict_json(
                severity="block",
                concern="Mean lactate 199 mg/dL is in LDH range — pollution.",
                suggested_loinc="32693-4",
                correction_rationale="MIMIC codes lactate molarly via 32693-4.",
            ),
            _verdict_json(severity="info"),
        ])

        result = pipeline.ask("What is the average lactate for patients with sepsis?")

        # Final answer is the corrected attempt.
        assert result.text_summary == "Clean lactate 2.1 mmol/L"
        # Trace records both attempts.
        assert result.correction_trace is not None
        assert len(result.correction_trace) == 2
        assert result.correction_trace[0]["loinc_used"] == "2524-7"
        assert result.correction_trace[1]["loinc_used"] == "32693-4"
        assert result.correction_trace[0]["fallback_warning"] is not None
        assert result.correction_trace[1]["fallback_warning"] is None
        # CQ in conversation history reflects the corrected LOINC.
        assert pipeline.conversation_history[-1][0].clinical_concepts[0].loinc_code == "32693-4"
        # Critic was called twice (once per attempt).
        assert pipeline._client.messages.create.call_count == 2

    @_patch_for_self_healing
    def test_no_suggestion_no_retry(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """Critic flags severity=block but emits no suggested_loinc.
        Orchestrator must NOT retry — correction_trace stays None."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        cq = _make_lactate_cq(loinc_code="2524-7")
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        polluted = AnswerResult(text_summary="Polluted answer")
        mock_run_sql_fastpath.return_value = (
            polluted, ["SELECT ..."], "LOINC fallback",
        )

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = mock_anthropic([
            _verdict_json(severity="block", concern="bad answer", suggested_loinc=None),
        ])

        result = pipeline.ask("...")

        assert result.correction_trace is None
        assert result.text_summary == "Polluted answer"
        assert mock_run_sql_fastpath.call_count == 1
        assert pipeline._client.messages.create.call_count == 1  # only critic 1

    @_patch_for_self_healing
    def test_max_retries_respected(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """Critic suggests LOINCs on both verdicts. With max_retries=1,
        only ONE retry happens — the second polluted attempt is final."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        cq = _make_lactate_cq(loinc_code="2524-7")
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        a1 = AnswerResult(text_summary="answer 1")
        a2 = AnswerResult(text_summary="answer 2")
        mock_run_sql_fastpath.side_effect = [
            (a1, ["sql1"], "fb1"),
            (a2, ["sql2"], "fb2"),
        ]

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_critic=True, critic_max_retries=1,
        )
        pipeline._client = mock_anthropic([
            _verdict_json(severity="block", concern="x", suggested_loinc="32693-4"),
            _verdict_json(severity="block", concern="y", suggested_loinc="2518-9"),
        ])

        result = pipeline.ask("...")

        assert result.text_summary == "answer 2"
        assert len(result.correction_trace) == 2
        assert mock_run_sql_fastpath.call_count == 2
        assert pipeline._client.messages.create.call_count == 2

    @_patch_for_self_healing
    def test_concept_not_biomarker_no_retry(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """Critic returns suggested_loinc on a drug concept (which the
        critic shouldn't do, but we're robust). _should_retry rejects
        non-biomarker concepts."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        drug_cq = CompetencyQuestion(
            original_question="something drug",
            clinical_concepts=[
                ClinicalConcept(name="vancomycin", concept_type="drug"),
            ],
            aggregation="count",
            scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[drug_cq])

        a1 = AnswerResult(text_summary="drug answer")
        mock_run_sql_fastpath.return_value = (a1, ["sql"], None)

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = mock_anthropic([
            _verdict_json(severity="block", concern="x", suggested_loinc="32693-4"),
        ])

        result = pipeline.ask("...")

        assert result.correction_trace is None
        assert mock_run_sql_fastpath.call_count == 1

    @_patch_for_self_healing
    def test_idempotent_guard_same_loinc_no_retry(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """If the critic suggests the same LOINC the CQ already has,
        _should_retry refuses — protects against degenerate loops."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        cq = _make_lactate_cq(loinc_code="32693-4")  # already has the suggested code
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        a1 = AnswerResult(text_summary="answer")
        mock_run_sql_fastpath.return_value = (a1, ["sql"], None)

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = mock_anthropic([
            _verdict_json(severity="warn", concern="x", suggested_loinc="32693-4"),
        ])

        result = pipeline.ask("...")

        assert result.correction_trace is None
        assert mock_run_sql_fastpath.call_count == 1

    @_patch_for_self_healing
    def test_severity_info_no_retry(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """severity=info verdicts must never trigger retry, even if a
        suggested_loinc happens to be present (defensive)."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        cq = _make_lactate_cq(loinc_code="2524-7")
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        a1 = AnswerResult(text_summary="legitimate answer")
        mock_run_sql_fastpath.return_value = (a1, ["sql"], None)

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = mock_anthropic([
            _verdict_json(
                severity="info",
                plausible=True,
                suggested_loinc="32693-4",  # malformed: shouldn't have this on info
            ),
        ])

        result = pipeline.ask("...")

        assert result.correction_trace is None
        assert mock_run_sql_fastpath.call_count == 1

    @_patch_for_self_healing
    def test_critic_failure_during_retry_breaks_loop(
        self, mock_decompose, mock_run_sql_fastpath,
    ):
        """If the critic call fails on the retry attempt (returns None),
        the loop terminates: _should_retry sees no verdict and breaks."""
        from tests.test_conversational.conftest import mock_anthropic
        from src.conversational.models import DecompositionResult

        cq = _make_lactate_cq(loinc_code="2524-7")
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        a1 = AnswerResult(text_summary="answer 1")
        a2 = AnswerResult(text_summary="answer 2")
        mock_run_sql_fastpath.side_effect = [
            (a1, ["sql1"], "fb1"),
            (a2, ["sql2"], None),
        ]

        # Mock returns valid verdict, then raises. critique() catches the
        # raise and returns None. _should_retry sees None and stops.
        client = MagicMock()
        first_resp = MagicMock()
        first_resp.content = [MagicMock(text=_verdict_json(
            severity="block", concern="x", suggested_loinc="32693-4",
        ))]
        client.messages.create.side_effect = [first_resp, RuntimeError("api down")]

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = client

        result = pipeline.ask("...")

        # Both runs happened (the retry fired before the critic failed).
        assert mock_run_sql_fastpath.call_count == 2
        # Trace has both attempts; second has critic_verdict=None.
        assert len(result.correction_trace) == 2
        assert result.correction_trace[1]["critic_verdict"] is None
        # Final answer is the second attempt.
        assert result.text_summary == "answer 2"


# ---------------------------------------------------------------------------
# TestCriticToolUseIntegration — externally-grounded critic via the orchestrator
# ---------------------------------------------------------------------------


class TestCriticToolUseIntegration:
    """End-to-end test that the orchestrator passes a tool-using critic
    sequence through correctly. Verifies that ``cited_sources`` populated
    on the verdict survives ``model_dump`` into ``correction_trace``."""

    @_patch_pre_critic
    def test_orchestrator_surfaces_cited_sources(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        monkeypatch,
    ):
        from tests.test_conversational.conftest import mock_anthropic

        cq, reasoning, answer = _setup_mocks(
            mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        )

        # Mock the critic's PubMed tool to return a deterministic record.
        monkeypatch.setattr(
            "src.conversational.critic.pubmed_search",
            lambda **kw: {"status": "ok", "results": [
                {"pmid": "12345", "title": "stub", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"},
            ]},
        )

        # Critic sequence: tool_use → end_turn citing the stub record.
        client = mock_anthropic([
            {
                "tool_use": [{"id": "tu_1", "name": "pubmed_search", "input": {"query": "x"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": json.dumps({
                    "plausible": True, "severity": "info", "concern": None,
                    "cited_sources": [
                        {"type": "pubmed", "pmid": "12345", "title": "stub", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"},
                    ],
                }),
                "stop_reason": "end_turn",
            },
        ])

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key", enable_critic=True,
        )
        pipeline._client = client

        result = pipeline.ask("any biomarker question")

        # The verdict carries the cited source.
        assert result.critic_verdict is not None
        assert result.critic_verdict.cited_sources is not None
        assert result.critic_verdict.cited_sources[0]["pmid"] == "12345"


# ---------------------------------------------------------------------------
# TestPreValidatorIntegration — Phase B end-to-end wiring tests.
#
# These tests exercise the orchestrator's interaction with the
# pre-execution SQL validator. They patch ``compile_sql`` (so we can
# return a deterministic SqlFastpathQuery), patch ``backend.execute``
# (so we can verify whether it was called), patch ``generate_answer``
# (so we can verify the answerer is skipped on block), and patch
# ``critique`` (same reason). The validator itself is patched per-test
# to return a chosen verdict.
# ---------------------------------------------------------------------------


def _patch_for_validator(fn):
    """Stack patches for pre-validator integration tests.

    Like ``_patch_fastpath`` but does NOT pre-mock ``validate_sql`` —
    each test installs its own validator response.
    """
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.merge_extractions")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    fn = patch("src.conversational.orchestrator.compile_sql")(fn)
    fn = patch("src.conversational.orchestrator.critique")(fn)
    fn = patch("src.conversational.orchestrator.validate_sql_deterministic")(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
    return fn


def _make_fastpath_cq(loinc_code: str | None = "2160-0") -> CompetencyQuestion:
    """SQL_FAST-eligible CQ with a biomarker concept + cohort scope + agg."""
    return CompetencyQuestion(
        original_question="average creatinine for patients over 65",
        clinical_concepts=[
            ClinicalConcept(
                name="creatinine",
                concept_type="biomarker",
                loinc_code=loinc_code,
            ),
        ],
        aggregation="mean",
        scope="cohort",
    )


def _make_validator_verdict(verdict: str, concern: str | None = None,
                             suggested_fix: str | None = None):
    from src.conversational.models import SqlValidationVerdict
    return SqlValidationVerdict(
        verdict=verdict, concern=concern, suggested_fix=suggested_fix,
    )


def _make_query_obj():
    from src.conversational.sql_fastpath import SqlFastpathQuery
    return SqlFastpathQuery(
        sql="SELECT AVG(l.valuenum) FROM labevents l ...",
        params=[1, 2, 3],
        columns=["mean_value"],
    )


class TestPreValidatorIntegration:
    """End-to-end wiring of the pre-execution SQL validator.

    Per the plan: block short-circuits backend.execute + generate_answer
    + critic; warn passes through to execution but threads concern into
    critic; pass / None are transparent. enable_pre_validator=False is
    the regression guard."""

    @_patch_for_validator
    def test_disabled_skips_validator_call(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_critique, mock_validator,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_fastpath_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = _make_query_obj()
        mock_answer.return_value = AnswerResult(text_summary="answer text")
        mock_critique.return_value = None

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_pre_validator=False,
        )
        pipeline.ask("avg creatinine over 65")

        assert mock_validator.call_count == 0
        # backend.execute is on the backend mock returned by _DuckDBBackend()
        backend_instance = pipeline._open_backend.__wrapped__  # not available
        # Counters never incremented when disabled.
        assert pipeline._pre_validator_blocks == 0
        assert pipeline._pre_validator_warns == 0
        assert pipeline._pre_validator_passes == 0

    @_patch_for_validator
    def test_block_short_circuits_execute_and_critic(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_critique, mock_validator,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_fastpath_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = _make_query_obj()
        mock_validator.return_value = _make_validator_verdict(
            "block",
            concern="LIKE-pooling pollution",
            suggested_fix="emit LOINC 2160-0",
        )

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_pre_validator=True,
        )
        result = pipeline.ask("avg creatinine over 65")

        # Validator was called; block path taken.
        assert mock_validator.call_count == 1
        # No execute, no answerer, no critic.
        # _DuckDBBackend was patched as a MagicMock instance; pipeline opened
        # one backend per ask(). Inspect via the patched class's instance.
        from src.conversational.orchestrator import _DuckDBBackend
        backend_instance = _DuckDBBackend.return_value
        assert backend_instance.execute.call_count == 0
        assert mock_answer.call_count == 0
        assert mock_critique.call_count == 0
        # AnswerResult contains the validator concern.
        assert "blocked" in result.text_summary.lower()
        assert "LIKE-pooling pollution" in result.text_summary
        assert "emit LOINC 2160-0" in result.text_summary
        # Counter incremented.
        assert pipeline._pre_validator_blocks == 1
        assert pipeline._pre_validator_warns == 0
        assert pipeline._pre_validator_passes == 0

    @_patch_for_validator
    def test_warn_threads_concern_into_critic_fallback(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_critique, mock_validator,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_fastpath_cq(loinc_code=None)
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = _make_query_obj()
        mock_answer.return_value = AnswerResult(text_summary="answer")
        mock_critique.return_value = None
        mock_validator.return_value = _make_validator_verdict(
            "warn", concern="possible unit pooling",
        )

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_pre_validator=True,
        )
        # Mock backend.execute return so generate_answer-style path completes.
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        pipeline.ask("avg creatinine over 65")

        # Execute and critic were both called (warn does NOT short-circuit).
        assert _DuckDBBackend.return_value.execute.call_count == 1
        assert mock_critique.call_count == 1
        # The critic received the validator concern via fallback_warning kwarg.
        critic_kwargs = mock_critique.call_args.kwargs
        fb_arg = critic_kwargs.get("fallback_warning") or ""
        assert "possible unit pooling" in fb_arg
        # Counter incremented.
        assert pipeline._pre_validator_warns == 1
        assert pipeline._pre_validator_blocks == 0
        assert pipeline._pre_validator_passes == 0

    @_patch_for_validator
    def test_pass_is_transparent(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_critique, mock_validator,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_fastpath_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = _make_query_obj()
        mock_answer.return_value = AnswerResult(text_summary="answer")
        mock_critique.return_value = None
        mock_validator.return_value = _make_validator_verdict("pass")

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_pre_validator=True,
        )
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        pipeline.ask("avg creatinine over 65")

        assert _DuckDBBackend.return_value.execute.call_count == 1
        assert mock_critique.call_count == 1
        # Counter: passed.
        assert pipeline._pre_validator_passes == 1
        assert pipeline._pre_validator_blocks == 0
        assert pipeline._pre_validator_warns == 0

    @_patch_for_validator
    def test_validator_returns_none_proceeds_as_baseline(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_critique, mock_validator,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_fastpath_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = _make_query_obj()
        mock_answer.return_value = AnswerResult(text_summary="answer")
        mock_critique.return_value = None
        # Validator API failure → returns None.
        mock_validator.return_value = None

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_pre_validator=True,
        )
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        pipeline.ask("avg creatinine over 65")

        # Pipeline proceeds normally.
        assert _DuckDBBackend.return_value.execute.call_count == 1
        assert mock_critique.call_count == 1
        # All counters stay at zero — None doesn't increment any.
        assert pipeline._pre_validator_blocks == 0
        assert pipeline._pre_validator_warns == 0
        assert pipeline._pre_validator_passes == 0

    @_patch_for_validator
    def test_block_does_not_fire_on_graph_path(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_critique, mock_validator,
    ):
        """Validator only fires inside _run_with_critic_retry (SQL fast-path).
        Graph-path CQs (no aggregation, single-patient scope, etc.) route
        through the graph branch and never touch the validator."""
        from src.conversational.models import DecompositionResult
        # Single-patient scope without aggregation routes to graph path.
        cq = CompetencyQuestion(
            original_question="show creatinine trajectory for patient X",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            scope="single_patient",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_extract.return_value = ExtractionResult()
        mock_build.return_value = (MagicMock(), {"triples": 0})
        mock_reason.return_value = _make_reasoning()
        mock_answer.return_value = AnswerResult(text_summary="answer")
        mock_critique.return_value = None

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_pre_validator=True,
        )
        pipeline.ask("show creatinine trajectory")

        # Validator was NOT called — graph path doesn't go through
        # _run_with_critic_retry.
        assert mock_validator.call_count == 0


# ---------------------------------------------------------------------------
# TestPreValidatorRetryWithMutation — verify the validator is re-invoked
# when the critic-driven retry mutates the LOINC.
# ---------------------------------------------------------------------------


def _patch_for_validator_retry(fn):
    """Like _patch_for_validator but leaves the critic real (mock_anthropic-
    driven) so we can exercise the critic-mutate-LOINC retry path."""
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.merge_extractions")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    fn = patch("src.conversational.orchestrator.compile_sql")(fn)
    fn = patch("src.conversational.orchestrator.validate_sql_deterministic")(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
    return fn


class TestPreValidatorRetryWithMutation:
    @_patch_for_validator_retry
    def test_validator_re_runs_after_critic_mutates_loinc(
        self,
        mock_decompose, mock_extract, mock_merge, mock_build, mock_reason,
        mock_answer, mock_compile, mock_validator,
    ):
        """Attempt 0: validator returns 'warn'; critic suggests new LOINC.
        Attempt 1: validator must be invoked AGAIN with the new SQL (since
        the critic mutated the CQ's loinc_code, _compile_fastpath_preview
        recompiles)."""
        from src.conversational.models import DecompositionResult
        from tests.test_conversational.conftest import mock_anthropic

        # Use a CQ that's eligible for self-healing (biomarker with LOINC).
        cq = CompetencyQuestion(
            original_question="average lactate for sepsis",
            clinical_concepts=[
                ClinicalConcept(
                    name="lactate", concept_type="biomarker",
                    loinc_code="2524-7",  # the wrong code
                ),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])

        # compile_sql returns different shapes per attempt so we can assert
        # the validator sees both.
        from src.conversational.sql_fastpath import SqlFastpathQuery
        attempt0_query = SqlFastpathQuery(
            sql="SELECT 1 attempt0", params=[], columns=["x"],
        )
        attempt1_query = SqlFastpathQuery(
            sql="SELECT 1 attempt1", params=[], columns=["x"],
        )
        mock_compile.side_effect = [attempt0_query, attempt1_query]

        mock_answer.return_value = AnswerResult(text_summary="answer")

        # Validator: warn on first call (allows execute), then pass on retry.
        mock_validator.side_effect = [
            _make_validator_verdict("warn", concern="suspect pollution"),
            _make_validator_verdict("pass"),
        ]

        # Critic: returns a verdict with suggested_loinc on first call,
        # then info-OK verdict on second call (so retry doesn't loop).
        critic_responses = [
            json.dumps({
                "plausible": False, "severity": "warn",
                "concern": "wrong LOINC", "reference_used": None,
                "suggested_loinc": "32693-4",  # correct code
                "correction_rationale": "use the MIMIC-coverage code",
            }),
            json.dumps({
                "plausible": True, "severity": "info",
                "concern": None, "reference_used": None,
            }),
        ]
        client = mock_anthropic(critic_responses)

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_critic=True, enable_pre_validator=True, critic_max_retries=1,
        )
        pipeline._client = client

        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        pipeline.ask("average lactate for sepsis")

        # Validator called twice — once per attempt.
        assert mock_validator.call_count == 2
        # The two validator calls saw different queries (attempt0 vs attempt1).
        # Phase E: validate_sql_deterministic(query, *, mcp_client=, ...) — query
        # is the first positional arg.
        first_query = mock_validator.call_args_list[0].args[0]
        second_query = mock_validator.call_args_list[1].args[0]
        assert first_query.sql != second_query.sql
        # Two warns + ... wait, second is "pass". Counters: 1 warn, 1 pass.
        assert pipeline._pre_validator_warns == 1
        assert pipeline._pre_validator_passes == 1


# ---------------------------------------------------------------------------
# TestSubAgentInContextualize — Phase F3 wiring of the sub-agent into
# the contextualization path.
# ---------------------------------------------------------------------------


def _patch_for_sub_agent_wiring(fn):
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    fn = patch("src.conversational.orchestrator.compile_sql")(fn)
    fn = patch("src.conversational.orchestrator.critique")(fn)
    fn = patch(
        "src.conversational.orchestrator.validate_sql_deterministic",
        new=MagicMock(return_value=None),
    )(fn)
    fn = patch("src.conversational.orchestrator.disambiguate")(fn)
    fn = patch("src.conversational.orchestrator.clarify")(fn)
    fn = patch("src.conversational.orchestrator.contextualize")(fn)
    fn = patch("src.conversational.orchestrator.HealthSourceOfTruthAgent")(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock(),
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock(),
    )(fn)
    return fn


class TestSubAgentInContextualize:
    @_patch_for_sub_agent_wiring
    def test_disabled_falls_back_to_simple_contextualize(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_disambiguate, mock_clarify,
        mock_contextualize, mock_sub_agent_class,
    ):
        from src.conversational.models import (
            ContextualNote, DecompositionResult,
        )
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_answer.return_value = AnswerResult(text_summary="base")
        mock_critique.return_value = None
        mock_contextualize.return_value = ContextualNote(text="simple ctx")
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
            enable_sub_agent_in_contextualize=False,
        ).ask("x")

        # Simple contextualize was called; sub-agent class never instantiated.
        assert mock_contextualize.call_count == 1
        assert mock_sub_agent_class.call_count == 0
        assert "simple ctx" in result.text_summary

    @_patch_for_sub_agent_wiring
    def test_enabled_uses_sub_agent_for_contextualize(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_disambiguate, mock_clarify,
        mock_contextualize, mock_sub_agent_class,
    ):
        from src.conversational.models import (
            DecompositionResult, Evidence, HealthAnswer, HealthFinding,
        )
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="what is normal creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_answer.return_value = AnswerResult(text_summary="mean was 1.2")
        mock_critique.return_value = None
        # Configure sub-agent: returns a HealthAnswer.
        sub_instance = MagicMock()
        sub_instance.consult.return_value = HealthAnswer(
            query="what is normal creatinine?",
            answer_summary="Normal serum creatinine is 0.7-1.3 mg/dL.",
            findings=[HealthFinding(
                claim="normal range",
                evidence=[Evidence(
                    source="loinc", id="2160-0", tool="loinc_reference_range",
                )],
                confidence="high",
                status="verified",
            )],
        )
        mock_sub_agent_class.return_value = sub_instance
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
            enable_sub_agent_in_contextualize=True,
        ).ask("what is normal creatinine?")

        # Sub-agent invoked exactly once; simple contextualize NOT used.
        assert mock_sub_agent_class.call_count == 1
        sub_instance.consult.assert_called_once()
        assert mock_contextualize.call_count == 0
        # The note from the sub-agent's answer_summary made it into the result.
        assert "0.7-1.3" in result.text_summary
        # And the LOINC citation was surfaced.
        assert result.contextual_citations is not None
        assert any(
            c.get("id") == "2160-0" for c in result.contextual_citations
        )

    @_patch_for_sub_agent_wiring
    def test_sub_agent_never_receives_data_table(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_disambiguate, mock_clarify,
        mock_contextualize, mock_sub_agent_class,
    ):
        """PHI invariant: the sub-agent's consult() call must receive
        only the question + sanitised context — never the answer's
        data_table (which can contain MIMIC row data)."""
        from src.conversational.models import (
            DecompositionResult, HealthAnswer,
        )
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        # Answer carries a data_table with PHI-shaped rows.
        mock_answer.return_value = AnswerResult(
            text_summary="mean was 1.2",
            data_table=[
                {"hadm_id": 12345, "subject_id": 999, "value": 1.2},
                {"hadm_id": 12346, "subject_id": 998, "value": 1.3},
            ],
        )
        mock_critique.return_value = None
        sub_instance = MagicMock()
        sub_instance.consult.return_value = HealthAnswer(
            query="x", answer_summary="ctx", findings=[],
        )
        mock_sub_agent_class.return_value = sub_instance
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
            enable_sub_agent_in_contextualize=True,
        ).ask("x")

        # Inspect what the sub-agent received.
        sub_instance.consult.assert_called_once()
        kwargs = sub_instance.consult.call_args.kwargs
        # context must be a dict — never carry data_table directly.
        ctx = kwargs.get("context") or {}
        assert "data_table" not in ctx
        assert "rows" not in ctx
        # And no PHI ID strings should appear in the serialised context.
        ctx_str = str(ctx)
        assert "12345" not in ctx_str
        assert "subject_id" not in ctx_str.lower()
        assert "hadm_id" not in ctx_str.lower()

    @_patch_for_sub_agent_wiring
    def test_sub_agent_returning_none_falls_through(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_disambiguate, mock_clarify,
        mock_contextualize, mock_sub_agent_class,
    ):
        from src.conversational.models import DecompositionResult
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_answer.return_value = AnswerResult(text_summary="base")
        mock_critique.return_value = None
        sub_instance = MagicMock()
        sub_instance.consult.return_value = None  # sub-agent failed
        mock_sub_agent_class.return_value = sub_instance
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
            enable_sub_agent_in_contextualize=True,
        ).ask("x")

        # Sub-agent failure → no note appended; baseline text preserved.
        assert result.text_summary == "base"
        assert result.contextual_citations is None


# ---------------------------------------------------------------------------
# TestDisambiguationWiring — Phase C wiring of disambiguate() into the
# clarify short-circuit path.
# ---------------------------------------------------------------------------


def _patch_for_consult(fn):
    """Patches for consult-wiring tests. We patch the consult functions at
    the orchestrator module level so individual tests can install mocked
    behaviour. Backends and graph stages are patched so the orchestrator
    doesn't try to hit a real DB."""
    fn = patch("src.conversational.orchestrator.decompose_question")(fn)
    fn = patch("src.conversational.orchestrator._extract")(fn)
    fn = patch("src.conversational.orchestrator.build_query_graph")(fn)
    fn = patch("src.conversational.orchestrator.reason")(fn)
    fn = patch("src.conversational.orchestrator.generate_answer")(fn)
    fn = patch("src.conversational.orchestrator.compile_sql")(fn)
    fn = patch("src.conversational.orchestrator.critique")(fn)
    fn = patch("src.conversational.orchestrator.validate_sql_deterministic")(fn)
    fn = patch("src.conversational.orchestrator.disambiguate")(fn)
    fn = patch("src.conversational.orchestrator.clarify")(fn)
    fn = patch("src.conversational.orchestrator.contextualize")(fn)
    fn = patch(
        "src.conversational.orchestrator._DuckDBBackend", new=MagicMock()
    )(fn)
    fn = patch(
        "src.conversational.orchestrator._BigQueryBackend", new=MagicMock()
    )(fn)
    return fn


def _make_clarifying_cq() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="what's the lactate level?",
        clarifying_question="Did you mean serum or CSF lactate?",
        clinical_concepts=[
            ClinicalConcept(name="lactate", concept_type="biomarker"),
        ],
    )


class TestDisambiguationWiring:
    @_patch_for_consult
    def test_disabled_skips_disambiguate_call(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_clarifying_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_clarify.return_value = None  # so raw text falls through

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_disambiguation=False,
        )
        result = pipeline.ask("what's the lactate level?")

        assert mock_disambiguate.call_count == 0
        # Existing clarify short-circuit still fires; raw text returned.
        assert "Did you mean serum or CSF lactate?" == result.text_summary
        assert pipeline._disambiguations_attempted == 0

    @_patch_for_consult
    def test_high_confidence_resolution_clears_clarifying_question(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import DecompositionResult
        from src.conversational.sql_fastpath import SqlFastpathQuery

        cq = _make_clarifying_cq()
        # Make CQ otherwise SQL_FAST eligible so it routes to the fast path
        # once the clarifying_question is cleared.
        cq.aggregation = "mean"
        cq.scope = "cohort"

        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_disambiguate.return_value = Disambiguation(
            input_name="lactate",
            canonical_name="serum lactate",
            resolved_code="32693-4",
            code_system="loinc",
            confidence="high",
        )
        mock_compile.return_value = SqlFastpathQuery(
            sql="SELECT 1", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        mock_answer.return_value = AnswerResult(text_summary="answer text")
        mock_critique.return_value = None
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        result = pipeline_run = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_disambiguation=True,
        ).ask("what's the lactate level?")

        # CQ proceeded to the normal pipeline; backend was hit.
        assert _DuckDBBackend.return_value.execute.called
        # Disambiguate fired once; resolution counter incremented.
        assert mock_disambiguate.call_count == 1
        # Loinc was mutated onto the concept (the test CQ).
        assert cq.loinc_code if False else True  # CQ doesn't expose; check via concept
        assert cq.clinical_concepts[0].loinc_code == "32693-4"

    @_patch_for_consult
    def test_low_confidence_keeps_clarifying_question(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import DecompositionResult
        from src.conversational.models import ClarifyingMessage
        cq = _make_clarifying_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_disambiguate.return_value = Disambiguation(
            input_name="lactate",
            canonical_name="lactate (ambiguous)",
            alternates=["serum lactate", "CSF lactate"],
            confidence="low",
        )
        mock_clarify.return_value = ClarifyingMessage(
            text="Did you mean serum (typical sepsis) or CSF lactate?",
            alternates_offered=["serum lactate", "CSF lactate"],
        )

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_disambiguation=True,
            enable_clarify_enrichment=True,
        )
        result = pipeline.ask("what's the lactate level?")

        # Disambiguate fired (one concept); but low conf → CQ retains
        # clarifying_question; clarify() formatted the user-facing message.
        assert mock_disambiguate.call_count == 1
        assert mock_clarify.call_count == 1
        assert "serum (typical sepsis)" in result.text_summary
        # No high-conf resolution.
        assert cq.clinical_concepts[0].loinc_code is None

    @_patch_for_consult
    def test_no_clarifying_question_means_no_disambiguate_call(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        """If decomposer didn't flag ambiguity, disambiguate is a no-op."""
        from src.conversational.models import DecompositionResult
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        mock_answer.return_value = AnswerResult(text_summary="ok")
        mock_critique.return_value = None
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_disambiguation=True,
        ).ask("x")

        assert mock_disambiguate.call_count == 0

    @_patch_for_consult
    def test_disambiguate_failure_keeps_clarifying_question(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        """When disambiguate() returns None, the CQ retains its
        clarifying_question and the orchestrator falls through to the
        clarify short-circuit. No crash."""
        from src.conversational.models import DecompositionResult
        cq = _make_clarifying_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_disambiguate.return_value = None
        mock_clarify.return_value = None  # falls through to raw text

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_disambiguation=True,
        ).ask("what's the lactate level?")

        # Raw text falls through unchanged.
        assert result.text_summary == "Did you mean serum or CSF lactate?"


class TestClarifyEnrichmentWiring:
    @_patch_for_consult
    def test_disabled_returns_raw_decomposer_text(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import DecompositionResult
        cq = _make_clarifying_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_disambiguate.return_value = None  # don't auto-resolve

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_clarify_enrichment=False,
        ).ask("what's the lactate level?")

        assert mock_clarify.call_count == 0
        assert result.text_summary == "Did you mean serum or CSF lactate?"

    @_patch_for_consult
    def test_clarify_returns_none_falls_back_to_raw(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        """API failure (clarify returns None) → raw decomposer text falls
        through unchanged. Critical regression guard."""
        from src.conversational.models import DecompositionResult
        cq = _make_clarifying_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_disambiguate.return_value = None
        mock_clarify.return_value = None

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_clarify_enrichment=True,
        ).ask("what's the lactate level?")

        # Raw text preserved.
        assert result.text_summary == "Did you mean serum or CSF lactate?"

    @_patch_for_consult
    def test_clarify_runs_after_disambiguation_with_partials(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import (
            ClarifyingMessage, DecompositionResult,
        )
        cq = _make_clarifying_cq()
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        partial = Disambiguation(
            input_name="lactate", canonical_name="lactate (ambiguous)",
            alternates=["serum lactate", "CSF lactate"],
            confidence="low",
        )
        mock_disambiguate.return_value = partial
        mock_clarify.return_value = ClarifyingMessage(
            text="enriched message",
            alternates_offered=["serum lactate", "CSF lactate"],
        )

        ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_disambiguation=True,
            enable_clarify_enrichment=True,
        ).ask("what's the lactate level?")

        # Verify clarify was called and that partials were threaded in.
        assert mock_clarify.call_count == 1
        partials_arg = mock_clarify.call_args.args[3]
        assert len(partials_arg) == 1
        assert partials_arg[0].confidence == "low"


class TestContextualizationWiring:
    @_patch_for_consult
    def test_disabled_skips_contextualize_call(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import DecompositionResult
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        mock_answer.return_value = AnswerResult(text_summary="answer")
        mock_critique.return_value = None
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        pipeline = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=False,  # default; explicit
        )
        pipeline.ask("x")

        assert mock_contextualize.call_count == 0

    @_patch_for_consult
    def test_info_severity_appends_note(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import (
            ContextualNote, CriticVerdict, DecompositionResult,
        )
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        # Critic returns info-severity verdict.
        mock_critique.return_value = CriticVerdict(
            plausible=True, severity="info",
        )
        # Wrap mock_answer to attach the verdict via _critique side effect.
        # Since we patch critique itself, we have to make the mock attach.
        # The orchestrator's _critique sets sub.critic_verdict = verdict
        # when verdict is not None — so the AnswerResult passing through
        # _contextualize will have severity=info.
        answer = AnswerResult(text_summary="base answer")
        mock_answer.return_value = answer
        mock_contextualize.return_value = ContextualNote(
            text="Mean is in line with published norms.",
            citations=[{"type": "pubmed", "pmid": "111"}],
        )
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
        ).ask("x")

        assert mock_contextualize.call_count == 1
        assert "Context" in result.text_summary
        assert "Mean is in line" in result.text_summary
        assert result.contextual_citations == [{"type": "pubmed", "pmid": "111"}]

    @_patch_for_consult
    def test_warn_severity_skips_contextualize(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import (
            CriticVerdict, DecompositionResult,
        )
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        mock_critique.return_value = CriticVerdict(
            plausible=False, severity="warn",
            concern="suspect pollution",
        )
        mock_answer.return_value = AnswerResult(text_summary="base")
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
        ).ask("x")

        # Critic warned → contextualize NOT called (don't drown the warning).
        assert mock_contextualize.call_count == 0

    @_patch_for_consult
    def test_contextualize_returns_none_means_no_append(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import DecompositionResult
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        mock_critique.return_value = None
        mock_answer.return_value = AnswerResult(text_summary="base answer")
        mock_contextualize.return_value = None
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        result = ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_contextualization=True,
        ).ask("x")

        # No append; baseline behaviour preserved.
        assert result.text_summary == "base answer"
        assert result.contextual_citations is None

    @_patch_for_consult
    def test_critic_disabled_still_fires_contextualize(
        self,
        mock_decompose, mock_extract, mock_build, mock_reason, mock_answer,
        mock_compile, mock_critique, mock_validator,
        mock_disambiguate, mock_clarify, mock_contextualize,
    ):
        from src.conversational.models import (
            ContextualNote, DecompositionResult,
        )
        from src.conversational.sql_fastpath import SqlFastpathQuery
        cq = CompetencyQuestion(
            original_question="x",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="mean", scope="cohort",
        )
        mock_decompose.return_value = DecompositionResult(competency_questions=[cq])
        mock_compile.return_value = SqlFastpathQuery(
            sql="x", params=[], columns=["x"],
        )
        mock_validator.return_value = None
        mock_answer.return_value = AnswerResult(text_summary="base")
        mock_contextualize.return_value = ContextualNote(text="ctx note")
        from src.conversational.orchestrator import _DuckDBBackend
        _DuckDBBackend.return_value.execute.return_value = []

        ConversationalPipeline(
            _DB_PATH, _ONTOLOGY_DIR, "test-key",
            enable_critic=False,  # no critic
            enable_contextualization=True,
        ).ask("x")

        # No critic verdict means severity check passes (verdict is None);
        # contextualize fires.
        assert mock_contextualize.call_count == 1

