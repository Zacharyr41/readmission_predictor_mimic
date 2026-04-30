"""Pipeline orchestrator for the conversational analytics layer.

Wires together decomposer, extractor, graph_builder, reasoner, and answerer
into a single ``ask(question) -> AnswerResult`` interface with conversation
history management.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from src.conversational.answerer import generate_answer
from src.conversational.concept_resolver import ConceptResolver
from src.conversational.decomposer import decompose_question
from src.conversational.extractor import (
    _BigQueryBackend,
    _DuckDBBackend,
    _extract,
    extract,
    extract_bigquery,
    merge_extractions,
)
from src.conversational.graph_builder import build_query_graph
from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    DecompositionResult,
    ExtractionConfig,
)
from src.conversational.operations import get_default_registry
from src.conversational.planner import QueryPlan, QueryPlanner
from src.conversational.reasoner import reason
from src.conversational.sql_fastpath import compile_sql

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


class ConversationalPipeline:
    """Chains decomposer -> extractor -> graph_builder -> reasoner -> answerer."""

    def __init__(
        self,
        db_path: Path,
        ontology_dir: Path,
        api_key: str,
        *,
        data_source: str = "local",
        bigquery_project: str | None = None,
        extraction_config: ExtractionConfig | None = None,
        max_workers: int = 1,
    ) -> None:
        import anthropic as _anthropic

        self._db_path = db_path
        self._ontology_dir = ontology_dir
        self._data_source = data_source
        self._bigquery_project = bigquery_project
        self._extraction_config = extraction_config
        self._max_workers = max_workers
        repo_root = Path(__file__).parent.parent.parent
        # Phase 5: wire SNOMED hierarchy if the cached JSON is present.
        # The SnomedHierarchy class itself degrades gracefully on missing
        # files, but constructing one when the file is absent produces
        # spurious warnings — so we only instantiate when the file exists.
        hierarchy = None
        hierarchy_path = repo_root / "data" / "ontology_cache" / "snomed_hierarchy.json"
        if hierarchy_path.exists():
            from src.graph_construction.terminology.snomed_hierarchy import (
                SnomedHierarchy,
            )
            hierarchy = SnomedHierarchy(hierarchy_path)
        self._resolver = ConceptResolver(
            mappings_dir=repo_root / "data" / "mappings",
            hierarchy=hierarchy,
        )
        self._registry = get_default_registry()
        self._planner = QueryPlanner(registry=self._registry)
        self._client: anthropic.Anthropic = _anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[tuple[CompetencyQuestion, AnswerResult]] = []
        self.max_history: int = 10

    def ask(self, question: str) -> AnswerResult:
        """Run the full pipeline for a natural-language question.

        Phase 4.5 + Phase 7a: the decomposer returns one or more
        CompetencyQuestions. Each sub-CQ is independently classified by
        the planner:

          - **SQL fast-path** (``QueryPlan.SQL_FAST``) — single-concept
            aggregate / comparison / diagnosis list / mortality. Skips
            extract + graph + SPARQL; one direct SQL call.
          - **Graph path** (``QueryPlan.GRAPH``) — anything needing the
            RDF knowledge graph (temporal Allen relations, time-series
            viz, median, multi-concept).

        Graph-path sub-CQs share ONE graph for the turn (Phase 4.5
        contract preserved). SQL-fast sub-CQs skip merge+build+reason
        entirely but still contribute to the final multi-CQ AnswerResult.

        Clarify short-circuit: if ANY sub-CQ has ``clarifying_question``,
        no downstream stages run.
        """
        try:
            decomp = decompose_question(
                self._client, question,
                conversation_history=list(self.conversation_history) or None,
            )

            # Clarify short-circuit: any sub-CQ with a non-empty
            # clarifying_question wins; the whole turn becomes a clarify.
            clarifying_cq = next(
                (
                    cq for cq in decomp.competency_questions
                    if cq.clarifying_question and cq.clarifying_question.strip()
                ),
                None,
            )
            if clarifying_cq is not None:
                answer = AnswerResult(
                    text_summary=clarifying_cq.clarifying_question,
                    interpretation_summary=clarifying_cq.interpretation_summary,
                    clarifying_question=clarifying_cq.clarifying_question,
                )
                # Record the first CQ in history so conversation context is preserved.
                self._remember(decomp.competency_questions[0], answer)
                return answer

            # Pre-resolve concepts once per CQ. SQL fast-path needs the
            # resolved name list for OR-matching (category → concrete names);
            # graph path's ``extract`` re-resolves internally but benefits
            # from the ``resolved_from_category`` marker being set here.
            per_cq_resolved: list[list[list[str]]] = []  # per CQ, per concept
            for cq in decomp.competency_questions:
                per_concept: list[list[str]] = []
                for concept in cq.clinical_concepts:
                    resolved = self._resolver.resolve(concept)
                    if len(resolved) > 1 or (
                        len(resolved) == 1 and resolved[0] != concept.name
                    ):
                        concept.resolved_from_category = True
                    per_concept.append(resolved)
                per_cq_resolved.append(per_concept)

            # Open one backend for the whole turn so SQL fast-path and
            # graph-path extractions share the same connection/client.
            with self._open_backend() as backend:
                sub_answers: list[AnswerResult | None] = [None] * len(
                    decomp.competency_questions
                )
                graph_cqs: list[tuple[int, CompetencyQuestion]] = []
                graph_extractions: list = []
                fastpath_sparql: list[str] = []

                for idx, cq in enumerate(decomp.competency_questions):
                    plan = self._planner.classify(cq)
                    if plan == QueryPlan.SQL_FAST:
                        sub, sql_used = self._run_sql_fastpath(
                            cq, backend, resolved_names=per_cq_resolved[idx][0],
                        )
                        sub.interpretation_summary = cq.interpretation_summary
                        sub_answers[idx] = sub
                        fastpath_sparql.extend(sql_used)  # stash for aggregation
                    elif plan == QueryPlan.CAUSAL:
                        # Phase 8a: wired to a stub that returns a
                        # well-shaped but NaN-valued CausalEffectResult so
                        # the UI can be built against the real contract
                        # before estimators land in 8d. When the stub fires
                        # in a live session, the summary text makes it
                        # obvious we're not returning a real estimate.
                        sub = self._run_causal(cq)
                        sub.interpretation_summary = cq.interpretation_summary
                        sub_answers[idx] = sub
                    elif plan == QueryPlan.SIMILARITY:
                        # Phase 9: standalone similarity ranking. The
                        # decomposer marks scope='patient_similarity' and
                        # populates ``similarity_spec``; ``_run_similarity``
                        # calls into ``src.similarity.run.run_similarity``
                        # and wraps the ranked output as a 6-column
                        # ``data_table``. Causal CQs that *also* carry a
                        # similarity_spec stay on QueryPlan.CAUSAL — the
                        # spec is consumed there as a cohort-narrowing
                        # directive (src/causal/run.py:187-219).
                        sub = self._run_similarity(cq, backend)
                        sub.interpretation_summary = cq.interpretation_summary
                        sub_answers[idx] = sub
                    else:
                        graph_cqs.append((idx, cq))
                        graph_extractions.append(
                            _extract(
                                backend, cq,
                                config=self._extraction_config,
                                resolver=self._resolver,
                            )
                        )

                graph_stats: dict = {}
                graph_sparql: list[str] = []
                if graph_cqs:
                    merged = (
                        graph_extractions[0] if len(graph_extractions) == 1
                        else merge_extractions(graph_extractions)
                    )
                    any_temporal = any(
                        bool(cq.temporal_constraints) for _, cq in graph_cqs
                    )
                    graph, graph_stats = build_query_graph(
                        self._ontology_dir, merged,
                        skip_allen_relations=not any_temporal,
                        max_workers=self._max_workers,
                    )
                    for idx, cq in graph_cqs:
                        reasoning = reason(graph, cq)
                        graph_sparql.extend(reasoning.sparql_queries)
                        sub = generate_answer(
                            self._client, cq, reasoning.rows,
                            graph_stats, reasoning.sparql_queries,
                        )
                        sub.interpretation_summary = cq.interpretation_summary
                        sub_answers[idx] = sub

            # All slots filled now (planner produces exactly one plan per CQ).
            completed: list[AnswerResult] = [a for a in sub_answers if a is not None]

            if not decomp.is_multi:
                answer = completed[0]
            else:
                answer = AnswerResult(
                    text_summary=decomp.narrative or "Multi-part answer:",
                    interpretation_summary=decomp.narrative,
                    graph_stats=graph_stats,
                    sparql_queries_used=list(fastpath_sparql) + list(graph_sparql),
                    sub_answers=completed,
                )

            self._remember(decomp.competency_questions[0], answer)
            return answer

        except Exception:
            logger.exception("Pipeline failed for question: %s", question)
            return AnswerResult(
                text_summary=(
                    "An error occurred while processing your question. "
                    "Please try rephrasing."
                ),
            )

    # -- internal helpers --------------------------------------------------

    @contextmanager
    def _open_backend(self):
        """Open the right backend for this pipeline's data source.

        Used once per ``ask()`` call; both SQL fast-path and graph-path
        extractions share the connection/client to avoid repeated setup
        cost. Close is guaranteed by the contextmanager.
        """
        if self._data_source == "bigquery":
            backend = _BigQueryBackend(self._bigquery_project)
        else:
            backend = _DuckDBBackend(self._db_path)
        try:
            yield backend
        finally:
            backend.close()

    def _run_sql_fastpath(
        self,
        cq: CompetencyQuestion,
        backend,
        *,
        resolved_names: list[str],
    ) -> tuple[AnswerResult, list[str]]:
        """Compile and execute a SQL fast-path CQ; wrap in AnswerResult.

        Returns (answer, [sql_text]) so the top-level aggregation of
        ``sparql_queries_used`` across a multi-CQ turn can include the
        SQL we ran on the fast-path (the Query Details expander shows it).

        Biomarker concepts that carry a LOINC code go through
        ``ConceptResolver.resolve_biomarker`` before compilation so the
        WHERE clause can use ``itemid IN (...)`` rather than a
        unit-pooling ``LIKE`` filter. When grounding fails (LOINC absent
        from the mapping table or no MIMIC labitem coverage), we fall
        back to LIKE and surface a visible warning to the user via
        ``AnswerResult.text_summary``.
        """
        resolved_itemids: list[int] | None = None
        fallback_warning: str | None = None
        if cq.clinical_concepts:
            concept = cq.clinical_concepts[0]
            if concept.concept_type == "biomarker" and concept.loinc_code:
                biom = self._resolver.resolve_biomarker(concept)
                resolved_itemids = biom.itemids
                fallback_warning = biom.fallback_reason

        query = compile_sql(
            cq, backend, self._registry,
            resolved_names=resolved_names,
            resolved_itemids=resolved_itemids,
        )
        raw_rows = backend.execute(query.sql, query.params)
        rows = [dict(zip(query.columns, r)) for r in raw_rows]
        answer = generate_answer(
            self._client, cq, rows,
            {},  # no graph_stats on the fast-path
            [query.sql],  # surface the SQL alongside any SPARQL
        )
        if fallback_warning:
            # Append a user-visible note explaining the answer may pool
            # variants. Phase 9b proper warning surface lives behind
            # AnswerResult; for now we co-locate with text_summary so the
            # warning is visible in chat without a model change.
            answer.text_summary = (
                (answer.text_summary or "")
                + f"\n\n⚠️ Note: {fallback_warning}"
            ).strip()
        return answer, [query.sql]

    def _run_causal(self, cq: CompetencyQuestion) -> AnswerResult:
        """Phase 8a: wrap a ``CausalEffectResult`` into an ``AnswerResult``.

        The wrapping is deliberately minimal — just enough so the existing
        Streamlit UI renders *something* recognisable on a CAUSAL plan.
        Phase 8i replaces this with a proper causal-effect panel (pairwise
        τ heatmap, per-outcome breakdown, diagnostic block, mode badge).
        The stub-vs-real distinction is surfaced to the user via the
        ``is_stub`` flag bleeding into the text summary so no one mistakes
        8a output for a real estimate.
        """
        from src.causal.run import run_causal

        result = run_causal(cq)
        if result.is_stub:
            summary = (
                "Causal-inference pipeline is wired (Phase 8a). Schema and "
                "routing are in place, but the estimator itself lands in "
                "Phase 8d — this result is a shape-only placeholder with "
                "NaN point estimates. The final system will return μ_c "
                "point estimates + uncertainty intervals + pairwise τ "
                "contrasts + diagnostics here."
            )
        else:
            # Phase 8d+ summary; 8a never reaches this branch.
            summary = (
                f"Causal-effect estimates for {len(result.mu_c)} interventions "
                f"(mode={result.mode}). See the causal panel for pairwise "
                f"contrasts and diagnostics."
            )

        # Flatten mu_c into a data_table the existing UI can render.
        data_table = [
            {
                "intervention": label,
                "mu_point": ui.point,
                "mu_lower": ui.lower,
                "mu_upper": ui.upper,
            }
            for label, ui in result.mu_c.items()
        ]
        return AnswerResult(
            text_summary=summary,
            data_table=data_table,
            table_columns=["intervention", "mu_point", "mu_lower", "mu_upper"],
        )

    def _run_similarity(self, cq: CompetencyQuestion, backend) -> AnswerResult:
        """Phase 9: wrap a ``SimilarityResult`` into an ``AnswerResult``.

        Lazy import of ``run_similarity`` mirrors the ``_run_causal``
        pattern above — keeps the heavy ``src.similarity`` graph out of
        the orchestrator import path until a similarity CQ actually
        arrives. The flat 6-column data_table is what the existing
        Streamlit dataframe renderer in ``src/conversational/app.py``
        already knows how to display; richer per-bucket breakdowns
        ride along on ``SimilarityScore.contextual_explanation`` and
        ``temporal_explanation`` for a future explanation panel
        (Phase 9b).
        """
        from src.similarity.run import run_similarity

        if cq.similarity_spec is None:
            # Defensive: scope='patient_similarity' but no spec means the
            # decomposer produced an underspecified CQ. Surface a plain
            # message rather than crashing or silently mis-routing.
            return AnswerResult(
                text_summary=(
                    "This question is scoped to patient similarity but no "
                    "similarity specification was produced. Please rephrase "
                    "with a clearer anchor (a specific patient or a "
                    "covariate template)."
                ),
            )

        result = run_similarity(cq.similarity_spec, backend)
        data_table = [
            {
                "rank": i + 1,
                "hadm_id": s.hadm_id,
                "subject_id": s.subject_id,
                "combined": s.combined,
                "contextual": s.contextual,
                "temporal": s.temporal,
            }
            for i, s in enumerate(result.scores)
        ]
        summary = (
            f"Found {result.n_returned} of {result.n_pool} candidates "
            f"similar to {result.anchor_description}. "
            f"See the table for ranked combined / contextual / temporal "
            f"scores."
        )
        return AnswerResult(
            text_summary=summary,
            data_table=data_table,
            table_columns=[
                "rank", "hadm_id", "subject_id",
                "combined", "contextual", "temporal",
            ],
        )

    def _extract_one(self, cq: CompetencyQuestion):
        """Legacy single-CQ extractor kept for external callers. Modern
        code inside ``ask()`` uses ``_extract(backend, cq, ...)`` directly
        so the backend is shared across sub-CQs."""
        if self._data_source == "bigquery":
            return extract_bigquery(
                cq, project=self._bigquery_project,
                config=self._extraction_config,
                resolver=self._resolver,
            )
        return extract(
            self._db_path, cq, config=self._extraction_config,
            resolver=self._resolver,
        )

    def _remember(self, cq: CompetencyQuestion, answer: AnswerResult) -> None:
        """Append to conversation_history, enforcing ``max_history``."""
        self.conversation_history.append((cq, answer))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()


def create_pipeline_from_settings() -> ConversationalPipeline:
    """Create a pipeline from config/settings.py defaults."""
    from config.settings import Settings

    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set in .env or environment.")

    ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"

    return ConversationalPipeline(
        db_path=settings.duckdb_path,
        ontology_dir=ontology_dir,
        api_key=settings.anthropic_api_key,
    )
