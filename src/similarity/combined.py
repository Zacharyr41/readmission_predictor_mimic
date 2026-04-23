"""Combined similarity — weighted sum with α (Phase 9).

``combine_scores(contextual, temporal, temporal_weight)`` returns the
final 0–1 scalar. When the temporal explanation is unavailable
(template anchor), the combination degrades to contextual-only
regardless of the caller's requested ``temporal_weight``.

Commit 5 of the Phase 9 TDD trail ships the real implementation;
commit 2 carries the stub so imports resolve.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.similarity.models import ContextualExplanation, TemporalExplanation


def combine_scores(
    contextual: ContextualExplanation,
    temporal: TemporalExplanation | None,
    temporal_weight: float = 0.5,
) -> float:
    """Return ``α · s_temp + (1 - α) · s_ctx``.

    If ``temporal`` is ``None`` or ``temporal.temporal_available`` is
    ``False``, the result is ``contextual.overall_score`` (contextual
    only — the template-anchor fallback). The returned value is
    clipped to ``[0, 1]`` to handle minor floating-point drift.
    """
    raise NotImplementedError(
        "combine_scores — ships in commit 5 of the Phase 9 TDD trail. "
        "See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
    )


__all__ = ["combine_scores"]
