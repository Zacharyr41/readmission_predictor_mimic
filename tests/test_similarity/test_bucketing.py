"""Hybrid temporal bucketing (Phase 9).

ICU-day buckets for ICU admissions; admission-relative 0-24h / 24-48h /
daily for floor-only admissions. See locked decision #2.
"""

from __future__ import annotations

from datetime import datetime

from src.similarity.bucketing import assign_buckets, get_bucketing_mode


class TestBucketingMode:
    def test_icu_admission_uses_icu_day_mode(self):
        icu_stays = [{"intime": datetime(2150, 1, 15, 10), "outtime": datetime(2150, 1, 18, 8)}]
        mode = get_bucketing_mode(icu_stays=icu_stays)
        assert mode == "icu_day"

    def test_floor_only_admission_uses_admission_relative(self):
        mode = get_bucketing_mode(icu_stays=[])
        assert mode == "admission_relative"

    def test_none_icu_stays_equivalent_to_empty(self):
        assert get_bucketing_mode(icu_stays=None) == "admission_relative"


class TestIcuDayBuckets:
    def test_events_inside_icu_stay_bucketed_by_icu_day(self):
        icu_stays = [{"intime": datetime(2150, 1, 15, 10), "outtime": datetime(2150, 1, 18, 8)}]
        events = [
            {"code": "snomed_drug:abx", "timestamp": datetime(2150, 1, 15, 12)},  # icu_day_0
            {"code": "snomed_drug:vaso", "timestamp": datetime(2150, 1, 16, 6)},  # icu_day_1
            {"code": "snomed_drug:steroid", "timestamp": datetime(2150, 1, 17, 4)},  # icu_day_2
        ]
        buckets, mode = assign_buckets(
            events=events,
            admission_start=datetime(2150, 1, 15, 8),
            admission_end=datetime(2150, 1, 18, 20),
            icu_stays=icu_stays,
        )
        assert mode == "icu_day"
        assert "snomed_drug:abx" in buckets["icu_day_0"]
        assert "snomed_drug:vaso" in buckets["icu_day_1"]
        assert "snomed_drug:steroid" in buckets["icu_day_2"]

    def test_events_before_icu_use_admission_relative_bucketing(self):
        """Events in the pre-ICU window stack in admission-relative
        buckets per the plan (pre-ICU → 0-24h / 24-48h buckets)."""
        icu_stays = [{"intime": datetime(2150, 1, 16, 10), "outtime": datetime(2150, 1, 18, 8)}]
        events = [
            {"code": "snomed_drug:preicu", "timestamp": datetime(2150, 1, 15, 12)},  # pre-ICU, first day
        ]
        buckets, mode = assign_buckets(
            events=events,
            admission_start=datetime(2150, 1, 15, 8),
            admission_end=datetime(2150, 1, 18, 20),
            icu_stays=icu_stays,
        )
        assert mode == "icu_day"
        assert any(
            "snomed_drug:preicu" in evts
            for label, evts in buckets.items()
            if label.startswith("pre_icu")
        )


class TestAdmissionRelativeBuckets:
    def test_first_24h_bucket_captures_early_events(self):
        events = [
            {"code": "snomed_drug:tpa", "timestamp": datetime(2150, 1, 15, 10)},  # 2h in
            {"code": "snomed_drug:abx", "timestamp": datetime(2150, 1, 15, 20)},  # 12h in
        ]
        buckets, mode = assign_buckets(
            events=events,
            admission_start=datetime(2150, 1, 15, 8),
            admission_end=datetime(2150, 1, 18, 8),
            icu_stays=None,
        )
        assert mode == "admission_relative"
        assert "snomed_drug:tpa" in buckets["h_0_24"]
        assert "snomed_drug:abx" in buckets["h_0_24"]

    def test_second_24h_bucket_separated(self):
        events = [
            {"code": "snomed_drug:warfarin", "timestamp": datetime(2150, 1, 16, 12)},  # 28h in
        ]
        buckets, mode = assign_buckets(
            events=events,
            admission_start=datetime(2150, 1, 15, 8),
            admission_end=datetime(2150, 1, 18, 8),
            icu_stays=None,
        )
        assert mode == "admission_relative"
        assert "snomed_drug:warfarin" in buckets["h_24_48"]

    def test_day_level_buckets_after_48h(self):
        events = [
            {"code": "snomed_drug:steroid", "timestamp": datetime(2150, 1, 17, 12)},  # day 3
            {"code": "snomed_drug:mab", "timestamp": datetime(2150, 1, 18, 6)},  # day 4
        ]
        buckets, mode = assign_buckets(
            events=events,
            admission_start=datetime(2150, 1, 15, 8),
            admission_end=datetime(2150, 1, 20, 0),
            icu_stays=None,
        )
        assert mode == "admission_relative"
        # Day-level bucket labels are "day_2", "day_3" (0-indexed from admission).
        day_buckets = {k: v for k, v in buckets.items() if k.startswith("day_")}
        all_events_in_day_buckets = set().union(*day_buckets.values())
        assert "snomed_drug:steroid" in all_events_in_day_buckets
        assert "snomed_drug:mab" in all_events_in_day_buckets


class TestEmptyInputs:
    def test_no_events_returns_empty_dict(self):
        buckets, mode = assign_buckets(
            events=[],
            admission_start=datetime(2150, 1, 15, 8),
            admission_end=datetime(2150, 1, 18, 8),
            icu_stays=None,
        )
        assert mode == "admission_relative"
        # Empty buckets for each canonical label expected (implementation
        # may emit labels with empty sets, or no labels at all — both OK;
        # assert no events leaked in).
        for evts in buckets.values():
            assert evts == set()
