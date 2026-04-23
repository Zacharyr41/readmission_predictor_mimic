"""Combined similarity — weighted sum with α (Phase 9).

Given a ``ContextualExplanation`` and a ``TemporalExplanation``
(both optional), return the final 0–1 scalar. When the temporal
side is unavailable — the template-anchor fallback — the caller's
``temporal_weight`` is ignored and the combined score degrades to
contextual-only.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.similarity.models import ContextualExplanation, TemporalExplanation


def combine_scores(
    contextual: ContextualExplanation,
    temporal: TemporalExplanation | None,
    temporal_weight: float = 0.5,
) -> float:
    """Return ``α · s_temp + (1 - α) · s_ctx`` clipped to ``[0, 1]``.

    Template-anchor fallback: if ``temporal`` is ``None`` OR
    ``temporal.temporal_available`` is ``False``, returns
    ``contextual.overall_score`` clipped — independent of
    ``temporal_weight``.
    """
    ctx = contextual.overall_score
    if temporal is None or not temporal.temporal_available:
        return float(max(0.0, min(1.0, ctx)))
    combined = temporal_weight * temporal.overall_score + (1.0 - temporal_weight) * ctx
    return float(max(0.0, min(1.0, combined)))


__all__ = ["combine_scores"]
