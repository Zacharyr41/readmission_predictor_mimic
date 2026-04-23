"""Temporal similarity scoring (Phase 9).

Bucketed weighted-Jaccard with decay. Verifies per-bucket scores +
aggregation + shared/only event partitioning.
"""

from __future__ import annotations

import math

from src.similarity.temporal import compute_temporal_similarity


class TestTemporalPerBucketJaccard:
    def test_identical_buckets_score_one(self, anchor_buckets):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm={999: anchor_buckets},
        )
        assert out[999].overall_score == 1.0
        for _, s in out[999].per_bucket:
            assert s == 1.0

    def test_disjoint_buckets_score_zero(self, anchor_buckets):
        disjoint = {
            "icu_day_0": {"snomed_drug:diuretic"},
            "icu_day_1": {"snomed_drug:insulin"},
            "icu_day_2": {"snomed_drug:paralytic"},
        }
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm={999: disjoint},
        )
        assert out[999].overall_score == 0.0

    def test_partial_overlap_intermediate_score(
        self, anchor_buckets, candidate_buckets_by_hadm,
    ):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
        )
        # 2001 = identical → highest
        # 2002 = partial → intermediate
        # 2003 = disjoint → lowest
        assert out[2001].overall_score > out[2002].overall_score
        assert out[2002].overall_score > out[2003].overall_score


class TestTemporalDecay:
    def test_default_decay_weights_earlier_buckets_more(
        self, anchor_buckets,
    ):
        """A candidate that matches only the FIRST bucket should score
        higher than one that matches only the LAST bucket under the
        default decay=0.9."""
        only_first = {
            "icu_day_0": anchor_buckets["icu_day_0"],
            "icu_day_1": set(),
            "icu_day_2": set(),
        }
        only_last = {
            "icu_day_0": set(),
            "icu_day_1": set(),
            "icu_day_2": anchor_buckets["icu_day_2"],
        }
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm={
                1: only_first,
                2: only_last,
            },
        )
        assert out[1].overall_score > out[2].overall_score

    def test_equal_weighting_when_decay_is_one(self, anchor_buckets):
        only_first = {
            "icu_day_0": anchor_buckets["icu_day_0"],
            "icu_day_1": set(), "icu_day_2": set(),
        }
        only_last = {
            "icu_day_0": set(),
            "icu_day_1": set(), "icu_day_2": anchor_buckets["icu_day_2"],
        }
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm={1: only_first, 2: only_last},
            decay=1.0,
        )
        # Both patients match exactly one bucket perfectly; under
        # equal weighting they must tie.
        assert out[1].overall_score == out[2].overall_score


class TestTemporalExplanationPartitioning:
    def test_shared_events_correctly_partitioned(
        self, anchor_buckets, candidate_buckets_by_hadm,
    ):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
        )
        # 2002's d0 shared: abx_broad + vasopressor. anchor-only: lactate_abn.
        shared = {(b, e) for b, e in out[2002].shared_events}
        anchor_only = {(b, e) for b, e in out[2002].anchor_only}
        assert ("icu_day_0", "snomed_drug:abx_broad") in shared
        assert ("icu_day_0", "snomed_drug:vasopressor") in shared
        assert ("icu_day_0", "snomed_lab:lactate_abn") in anchor_only

    def test_candidate_only_surfaces_uniques(
        self, anchor_buckets, candidate_buckets_by_hadm,
    ):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
        )
        # 2003 introduces diuretic + insulin — neither in anchor.
        cand_only = {(b, e) for b, e in out[2003].candidate_only}
        assert ("icu_day_0", "snomed_drug:diuretic") in cand_only
        assert ("icu_day_0", "snomed_drug:insulin") in cand_only


class TestTemporalLosGap:
    def test_los_gap_reported_when_buckets_differ_in_length(
        self, anchor_buckets, candidate_buckets_by_hadm,
    ):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
        )
        # Anchor has 3 buckets, candidate 2003 has 2.
        assert out[2003].los_gap_days == 1

    def test_los_gap_zero_when_matching_lengths(
        self, anchor_buckets, candidate_buckets_by_hadm,
    ):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
        )
        # Anchor + 2001 both have 3 buckets.
        assert out[2001].los_gap_days == 0


class TestTemporalAvailableFlag:
    def test_flag_true_when_anchor_buckets_present(
        self, anchor_buckets, candidate_buckets_by_hadm,
    ):
        out = compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
        )
        for exp in out.values():
            assert exp.temporal_available is True

    def test_empty_anchor_yields_available_false(self):
        out = compute_temporal_similarity(
            anchor_buckets={},
            candidate_buckets_by_hadm={1: {"icu_day_0": {"snomed_drug:abx_broad"}}},
        )
        assert out[1].temporal_available is False
        assert math.isnan(out[1].overall_score) or out[1].overall_score == 0.0
