"""Hybrid temporal bucketing (Phase 9, decision #2).

``assign_buckets`` dispatches on whether the admission has an ICU
stay: ICU-day buckets for ICU admissions (reuses the conventions of
the existing hetero graph's ``icu_day`` nodes at
``data/processed/full_hetero_graph.meta.json``), admission-relative
windows (``h_0_24``, ``h_24_48``, then daily ``day_N``) for
floor-only admissions. Pre-ICU events for ICU admissions land in
``pre_icu_*`` buckets so the trajectory alignment isn't left-padded
with empty days.

Commit 2 of the Phase 9 TDD trail: stubbed. Commit 4 implements.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

BucketingMode = Literal["icu_day", "admission_relative"]


def get_bucketing_mode(icu_stays: list[dict] | None) -> BucketingMode:
    """Pick the bucketing mode for an admission.

    ICU stays present → ``"icu_day"``. Else → ``"admission_relative"``.
    ``None`` is treated as empty.
    """
    if icu_stays:
        return "icu_day"
    return "admission_relative"


def assign_buckets(
    events: list[dict],
    admission_start: datetime,
    admission_end: datetime,
    icu_stays: list[dict] | None = None,
) -> tuple[dict[str, set[str]], BucketingMode]:
    """Partition events into temporal buckets.

    Each event must carry at least ``code`` (str) and ``timestamp``
    (datetime). Returns ``(bucket_map, mode)`` where ``bucket_map``
    is ``{bucket_label → set[event_codes]}``.
    """
    raise NotImplementedError(
        "assign_buckets — ships in commit 4 of the Phase 9 TDD trail. "
        "See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
    )


__all__ = ["BucketingMode", "assign_buckets", "get_bucketing_mode"]
