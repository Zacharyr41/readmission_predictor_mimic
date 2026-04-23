"""Temporal similarity — hybrid-bucketed weighted Jaccard (Phase 9).

Per-bucket Jaccard over SNOMED-grouped event sets, aggregated via
decay-weighted mean. Default ``decay=0.9`` weights earlier buckets
more (clinical intuition: first hours / first ICU day carry more
diagnostic signal than the tail of a long stay).

Bucket alignment: the scorer uses the union of bucket labels from
anchor + candidate, iterating in anchor-insertion order first (so
the time-order intended by ``assign_buckets`` is preserved). A
bucket present in only one side contributes 0 to the Jaccard
(no overlap possible); this also naturally handles unequal trajectory
lengths (``los_gap_days`` captures the count gap for the UI).

When ``anchor_buckets`` is empty (template-anchor fallback), every
returned explanation carries ``temporal_available=False`` and
``overall_score=0.0`` so ``combine_scores`` degrades to
contextual-only.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.similarity.models import TemporalExplanation


def _bucket_jaccard(a: set[str], c: set[str]) -> float:
    """Jaccard with empty-set conventions: both empty → 1.0; one
    empty → 0.0; else |a ∩ c| / |a ∪ c|."""
    if not a and not c:
        return 1.0
    if not a or not c:
        return 0.0
    inter = a & c
    union = a | c
    return len(inter) / len(union) if union else 1.0


def _ordered_labels(
    anchor_buckets: dict[str, set[str]],
    candidate_buckets: dict[str, set[str]],
) -> list[str]:
    """Anchor insertion order first (time-order preserving), then
    candidate-only labels in their own insertion order."""
    ordered = list(anchor_buckets.keys())
    seen = set(ordered)
    for label in candidate_buckets.keys():
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def compute_temporal_similarity(
    anchor_buckets: dict[str, set[str]],
    candidate_buckets_by_hadm: dict[int, dict[str, set[str]]],
    decay: float = 0.9,
) -> dict[int, TemporalExplanation]:
    """Score each candidate's temporal similarity to the anchor.

    See module docstring + the plan file for bucket alignment,
    decay semantics, and the template-anchor fallback.
    """
    temporal_available = bool(anchor_buckets)
    out: dict[int, TemporalExplanation] = {}

    for hadm_id, cand_buckets in candidate_buckets_by_hadm.items():
        if not temporal_available:
            out[hadm_id] = TemporalExplanation(
                overall_score=0.0,
                per_bucket=[],
                shared_events=[],
                anchor_only=[],
                candidate_only=[],
                los_gap_days=0,
                temporal_available=False,
            )
            continue

        labels = _ordered_labels(anchor_buckets, cand_buckets)
        per_bucket: list[tuple[str, float]] = []
        shared: list[tuple[str, str]] = []
        anchor_only: list[tuple[str, str]] = []
        candidate_only: list[tuple[str, str]] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for i, label in enumerate(labels):
            a_set = anchor_buckets.get(label, set())
            c_set = cand_buckets.get(label, set())
            jaccard = _bucket_jaccard(a_set, c_set)
            per_bucket.append((label, jaccard))

            w = decay ** i
            weighted_sum += w * jaccard
            weight_total += w

            for evt in sorted(a_set & c_set):
                shared.append((label, evt))
            for evt in sorted(a_set - c_set):
                anchor_only.append((label, evt))
            for evt in sorted(c_set - a_set):
                candidate_only.append((label, evt))

        overall = weighted_sum / weight_total if weight_total > 0 else 0.0
        overall = float(max(0.0, min(1.0, overall)))
        los_gap = abs(len(anchor_buckets) - len(cand_buckets))

        out[hadm_id] = TemporalExplanation(
            overall_score=overall,
            per_bucket=per_bucket,
            shared_events=shared,
            anchor_only=anchor_only,
            candidate_only=candidate_only,
            los_gap_days=los_gap,
            temporal_available=True,
        )
    return out


__all__ = ["compute_temporal_similarity"]
