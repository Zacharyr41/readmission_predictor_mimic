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
from src.conversational.extractor import extract
from src.conversational.graph_builder import build_query_graph
from src.conversational.models import AnswerResult, CompetencyQuestion
from src.conversational.reasoner import reason

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


class ConversationalPipeline:
    """Chains decomposer -> extractor -> graph_builder -> reasoner -> answerer."""

    def __init__(self, db_path: Path, ontology_dir: Path, api_key: str) -> None:
        import anthropic as _anthropic

        self._db_path = db_path
        self._ontology_dir = ontology_dir
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

            extraction = extract(self._db_path, cq)

            graph, graph_stats = build_query_graph(self._ontology_dir, extraction)

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
