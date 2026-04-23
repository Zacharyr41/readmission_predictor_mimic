"""Explanation formatters (Phase 9).

Translate the structured explanation payloads into human-readable text
snippets for the chat response and Streamlit UI. Structured data stays
on the ``SimilarityScore`` object for programmatic consumers;
formatters here are pure-presentation.

Commit 5 of the Phase 9 TDD trail fills in the real formatting;
commit 2 stubs for import resolution.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.similarity.models import (
    ContextualExplanation,
    SimilarityScore,
    TemporalExplanation,
)


def format_contextual_text(exp: ContextualExplanation) -> str:
    """One-paragraph summary of a contextual explanation."""
    raise NotImplementedError(
        "format_contextual_text — ships in commit 5 of the Phase 9 TDD trail."
    )


def format_temporal_text(exp: TemporalExplanation) -> str:
    """One-paragraph summary of a temporal explanation.

    Special case: when ``exp.temporal_available`` is False, returns
    text explicitly flagging the contextual-only fallback.
    """
    raise NotImplementedError(
        "format_temporal_text — ships in commit 5 of the Phase 9 TDD trail."
    )


def format_similarity_text(score: SimilarityScore) -> str:
    """Combined text summary referencing both axes + overall score.

    Used as the chat-response similarity blurb when the CQ carries a
    ``similarity_spec``.
    """
    raise NotImplementedError(
        "format_similarity_text — ships in commit 5 of the Phase 9 TDD trail."
    )


__all__ = [
    "format_contextual_text",
    "format_similarity_text",
    "format_temporal_text",
]
