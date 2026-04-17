"""Pipeline orchestrator for the conversational analytics layer.

Wires together decomposer, extractor, graph_builder, reasoner, and answerer
into a single ``ask(question) -> AnswerResult`` interface with conversation
history management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.conversational.answerer import generate_answer
from src.conversational.concept_resolver import ConceptResolver
from src.conversational.decomposer import decompose_question
from src.conversational.extractor import (
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
from src.conversational.reasoner import reason

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
        self._client: anthropic.Anthropic = _anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[tuple[CompetencyQuestion, AnswerResult]] = []
        self.max_history: int = 10

    def ask(self, question: str) -> AnswerResult:
        """Run the full pipeline for a natural-language question.

        Phase 4.5: the decomposer may return a single CompetencyQuestion
        (Shape A, the common case) OR a big-question decomposition (Shape B:
        narrative + N sub-CQs that share ONE downstream knowledge graph).

        Flow:
          1. decompose_question → DecompositionResult
          2. If ANY sub-CQ has a clarifying_question, short-circuit and
             surface that question — skipping extract / graph / reason / answer.
          3. Otherwise extract per sub-CQ, then merge into one
             ExtractionResult. For a single-CQ decomposition, merging is a
             no-op and is skipped.
          4. Build exactly ONE RDF graph from the merged extraction. Allen
             relations are built if ANY sub-CQ has temporal constraints.
          5. Reason once per sub-CQ against the shared graph.
          6. Generate per-CQ answers. For single-CQ: return that answer
             directly. For multi-CQ: wrap in a top-level AnswerResult whose
             ``text_summary`` is the narrative and whose ``sub_answers``
             carries the per-CQ details.
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

            # Mark category-resolved concepts across every sub-CQ. This is
            # per-concept, idempotent, and cheap — safe to do before extraction.
            for cq in decomp.competency_questions:
                for concept in cq.clinical_concepts:
                    resolved = self._resolver.resolve(concept)
                    if len(resolved) > 1 or (
                        len(resolved) == 1 and resolved[0] != concept.name
                    ):
                        concept.resolved_from_category = True

            # Per-CQ extract; single-CQ fast path skips the merge step.
            extractions = [self._extract_one(cq) for cq in decomp.competency_questions]
            merged_extraction = (
                extractions[0] if len(extractions) == 1
                else merge_extractions(extractions)
            )

            # Build ONE graph. Allen relations iff any sub-CQ needs them.
            any_temporal = any(
                bool(cq.temporal_constraints) for cq in decomp.competency_questions
            )
            graph, graph_stats = build_query_graph(
                self._ontology_dir, merged_extraction,
                skip_allen_relations=not any_temporal,
                max_workers=self._max_workers,
            )

            # Reason + answer per sub-CQ against the shared graph.
            sub_answers: list[AnswerResult] = []
            all_sparql: list[str] = []
            for cq in decomp.competency_questions:
                reasoning = reason(graph, cq)
                all_sparql.extend(reasoning.sparql_queries)
                sub = generate_answer(
                    self._client, cq, reasoning.rows,
                    graph_stats, reasoning.sparql_queries,
                )
                sub.interpretation_summary = cq.interpretation_summary
                sub_answers.append(sub)

            if not decomp.is_multi:
                # Single-CQ: return the one sub-answer directly (legacy shape).
                answer = sub_answers[0]
            else:
                # Multi-CQ: wrap sub-answers under a narrative-led top-level.
                answer = AnswerResult(
                    text_summary=decomp.narrative or "Multi-part answer:",
                    interpretation_summary=decomp.narrative,
                    graph_stats=graph_stats,
                    sparql_queries_used=all_sparql,
                    sub_answers=sub_answers,
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

    def _extract_one(self, cq: CompetencyQuestion):
        """Dispatch to the right extractor backend for a single CQ."""
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
