"""Combined temporal + contextual similarity (Phase 9).

Weighted sum with α (``temporal_weight``). When temporal is
unavailable (template anchor), forces contextual-only.
"""

from __future__ import annotations

import math

import pytest

from src.similarity.combined import combine_scores
from src.similarity.models import ContextualExplanation, TemporalExplanation


def _ctx(score: float) -> ContextualExplanation:
    return ContextualExplanation(
        overall_score=score, per_group={}, top_contributors=[], top_detractors=[],
    )


def _temp(score: float, available: bool = True) -> TemporalExplanation:
    return TemporalExplanation(
        overall_score=score, per_bucket=[], shared_events=[],
        anchor_only=[], candidate_only=[], los_gap_days=0,
        temporal_available=available,
    )


class TestCombinedWeighting:
    def test_alpha_half_is_simple_average(self):
        combined = combine_scores(
            contextual=_ctx(0.8), temporal=_temp(0.6), temporal_weight=0.5,
        )
        assert combined == pytest.approx(0.7)

    def test_alpha_zero_uses_contextual_only(self):
        combined = combine_scores(
            contextual=_ctx(0.9), temporal=_temp(0.2), temporal_weight=0.0,
        )
        assert combined == 0.9

    def test_alpha_one_uses_temporal_only(self):
        combined = combine_scores(
            contextual=_ctx(0.1), temporal=_temp(0.8), temporal_weight=1.0,
        )
        assert combined == 0.8


class TestTemporalUnavailableFallback:
    def test_unavailable_temporal_forces_contextual_only(self):
        """When a template anchor yields no trajectory, temporal_available
        = False; combine_scores must degrade to the contextual score
        regardless of the caller's temporal_weight."""
        combined = combine_scores(
            contextual=_ctx(0.75),
            temporal=_temp(float("nan"), available=False),
            temporal_weight=0.5,
        )
        assert combined == 0.75

    def test_none_temporal_also_degrades(self):
        combined = combine_scores(
            contextual=_ctx(0.62),
            temporal=None,
            temporal_weight=0.5,
        )
        assert combined == 0.62


class TestOutputBounds:
    def test_combined_in_unit_interval(self):
        for c in (0.0, 0.3, 1.0):
            for t in (0.0, 0.5, 1.0):
                for a in (0.0, 0.5, 1.0):
                    combined = combine_scores(
                        contextual=_ctx(c), temporal=_temp(t), temporal_weight=a,
                    )
                    assert 0.0 <= combined <= 1.0
                    assert not math.isnan(combined)
