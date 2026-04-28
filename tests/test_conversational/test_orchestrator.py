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
