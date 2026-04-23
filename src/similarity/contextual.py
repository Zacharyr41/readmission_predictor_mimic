"""Contextual similarity — grouped Gower-like distance (Phase 9).

Five feature groups (defined in ``feature_groups.py``), each computing
a 0-1 group-similarity score plus per-feature contribution lists for
explanations. Overall contextual similarity is a weighted mean across
groups; group weights are user-overrideable via
``SimilaritySpec.contextual_weights``.

Commit 2 of the Phase 9 TDD trail: stubbed. Commit 3 ships the real
per-group distance functions + explanation population.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

import pandas as pd

from src.similarity.models import ContextualExplanation


def compute_contextual_similarity(
    anchor_features: dict,
    candidate_features_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict[int, ContextualExplanation]:
    """Score each candidate's contextual similarity to the anchor.

    Args:
        anchor_features: feature dict for the anchor row (keys must
            match ``candidate_features_df`` columns).
        candidate_features_df: one row per candidate; must include
            ``hadm_id`` as a column.
        weights: optional per-group override. When provided, must sum
            to 1.0 across the five groups; missing groups take their
            defaults from ``DEFAULT_GROUP_WEIGHTS``.

    Returns:
        ``dict[hadm_id, ContextualExplanation]`` — one explanation per
        candidate row, carrying overall score + per-group breakdown +
        top contributors / detractors.
    """
    raise NotImplementedError(
        "compute_contextual_similarity — ships in commit 3 of the Phase 9 TDD "
        "trail. See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
    )


__all__ = ["compute_contextual_similarity"]
