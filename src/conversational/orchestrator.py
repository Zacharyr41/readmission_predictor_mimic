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
from src.conversational.decomposer import decompose
from src.conversational.extractor import extract, extract_bigquery
from src.conversational.graph_builder import build_query_graph
from src.conversational.models import AnswerResult, CompetencyQuestion, ExtractionConfig
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
        self._client: anthropic.Anthropic = _anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[tuple[CompetencyQuestion, AnswerResult]] = []
        self.max_history: int = 10

    def ask(self, question: str) -> AnswerResult:
        """Run the full pipeline for a natural-language question."""
        try:
            cq = decompose(
                self._client, question,
                conversation_history=list(self.conversation_history) or None,
            )

            if self._data_source == "bigquery":
                extraction = extract_bigquery(
                    cq, project=self._bigquery_project,
                    config=self._extraction_config,
                )
            else:
                extraction = extract(
                    self._db_path, cq, config=self._extraction_config,
                )

            graph, graph_stats = build_query_graph(
                self._ontology_dir, extraction,
                skip_allen_relations=not bool(cq.temporal_constraints),
                max_workers=self._max_workers,
            )

            reasoning = reason(graph, cq)

            answer = generate_answer(
                self._client, cq, reasoning.rows,
                graph_stats, reasoning.sparql_queries,
            )

            self.conversation_history.append((cq, answer))
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return answer

        except Exception:
            logger.exception("Pipeline failed for question: %s", question)
            return AnswerResult(
                text_summary=(
                    "An error occurred while processing your question. "
                    "Please try rephrasing."
                ),
            )

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
