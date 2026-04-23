"""Temporal similarity — hybrid-bucketed weighted Jaccard (Phase 9).

Per-bucket Jaccard over SNOMED-grouped event sets, aggregated via a
decay-weighted mean (earlier buckets count more). Bucket alignment
uses the hybrid scheme from decision #2 (ICU-day for ICU admissions,
admission-relative for floor-only); bucket assignment lives in
``src.similarity.bucketing``.

Commit 2 of the Phase 9 TDD trail: stubbed. Commit 4 ships the real
implementation.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.similarity.models import TemporalExplanation


def compute_temporal_similarity(
    anchor_buckets: dict[str, set[str]],
    candidate_buckets_by_hadm: dict[int, dict[str, set[str]]],
    decay: float = 0.9,
) -> dict[int, TemporalExplanation]:
    """Score each candidate's temporal similarity to the anchor.

    Args:
        anchor_buckets: ``bucket_label → set[snomed_coded_event]``.
        candidate_buckets_by_hadm: ``hadm_id → bucket_label → set[event]``.
        decay: geometric decay factor per bucket (``1.0`` ⇒ equal
            weighting). ``0.9`` default weights earlier buckets more.

    Returns:
        ``dict[hadm_id, TemporalExplanation]`` — one per candidate,
        carrying the overall score + per-bucket Jaccard list +
        shared/anchor-only/candidate-only event partition + LOS gap.

    When ``anchor_buckets`` is empty (template anchor), every entry
    comes back with ``temporal_available=False``.
    """
    raise NotImplementedError(
        "compute_temporal_similarity — ships in commit 4 of the Phase 9 TDD "
        "trail. See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
    )


__all__ = ["compute_temporal_similarity"]
