"""Human-readable explanation formatters (Phase 9).

Convert ``ContextualExplanation`` / ``TemporalExplanation`` /
``SimilarityScore`` payloads into short plain-text snippets for the
chat response. Structured data stays on the model objects for the
Streamlit panel + machine consumers — these helpers are pure
presentation.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.similarity.models import (
    ContextualExplanation,
    SimilarityScore,
    TemporalExplanation,
)


def format_contextual_text(exp: ContextualExplanation) -> str:
    """Short summary for a contextual explanation.

    Lists every group score (sorted by strength) so readers see the
    full 5-group decomposition, not just the top. Caps at the top
    three contributors / detractors to keep chat output tight.
    """
    lines = [f"Contextual similarity: {exp.overall_score:.2f}"]
    if exp.per_group:
        groups_sorted = sorted(exp.per_group.items(), key=lambda kv: -kv[1])
        group_parts = [f"{name}={score:.2f}" for name, score in groups_sorted]
        lines.append("Group scores: " + ", ".join(group_parts))
    if exp.top_contributors:
        names = ", ".join(f for f, _ in exp.top_contributors[:3])
        lines.append(f"Top contributors: {names}")
    if exp.top_detractors:
        names = ", ".join(f for f, _ in exp.top_detractors[:3])
        lines.append(f"Top detractors: {names}")
    return "\n".join(lines)


def format_temporal_text(exp: TemporalExplanation) -> str:
    """Short summary for a temporal explanation.

    Handles the template-anchor fallback with explicit wording
    ("unavailable", "template", "contextual-only") so chat consumers
    can match on any of those.
    """
    if not exp.temporal_available:
        return (
            "Temporal similarity unavailable (template anchor — no trajectory "
            "to compare against). Falling back to contextual-only scoring."
        )
    lines = [f"Temporal similarity: {exp.overall_score:.2f}"]
    lines.append(f"LOS gap: {exp.los_gap_days} day(s)")
    if exp.shared_events:
        shared = ", ".join(e for _, e in exp.shared_events[:5])
        lines.append(f"Shared events: {shared}")
    if exp.candidate_only:
        cand_unique = ", ".join(e for _, e in exp.candidate_only[:3])
        lines.append(f"Candidate-only events: {cand_unique}")
    return "\n".join(lines)


def format_similarity_text(score: SimilarityScore) -> str:
    """Top-of-stack summary referencing both axes + overall score."""
    parts = [f"Overall similarity: {score.combined:.2f}"]
    parts.append(format_contextual_text(score.contextual_explanation))
    if score.temporal_explanation is not None:
        parts.append(format_temporal_text(score.temporal_explanation))
    return "\n".join(parts)


__all__ = [
    "format_contextual_text",
    "format_similarity_text",
    "format_temporal_text",
]
