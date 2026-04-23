"""Hybrid temporal bucketing (Phase 9, decision #2).

``assign_buckets`` dispatches on whether the admission has an ICU
stay:

* **ICU admissions** (``icu_stays`` non-empty) — events inside any ICU
  stay land in ``icu_day_N`` buckets (N is the 0-indexed calendar-day
  offset from ``intime``). Pre-ICU events land in ``pre_icu_0_24`` /
  ``pre_icu_24_plus`` based on hour-offset from admission start, so
  the trajectory isn't left-padded with empty ICU days. Post-ICU
  events are deliberately not bucketed by 8d scope — they're rare
  in the causal / similarity workflow and 8c's event extractor
  doesn't surface them by default.

* **Floor-only admissions** — events bucket as ``h_0_24`` /
  ``h_24_48`` (hour-offset from admission) for the first 48 hours,
  then ``day_N`` (N ≥ 2, 0-indexed) afterward.

``get_bucketing_mode`` is the standalone dispatch helper used by both
this module and by future callers in ``src.similarity.run`` that need
to know the mode before extracting events.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

BucketingMode = Literal["icu_day", "admission_relative"]


def get_bucketing_mode(icu_stays: list[dict] | None) -> BucketingMode:
    """Return ``"icu_day"`` if any ICU stay is present, else
    ``"admission_relative"``."""
    if icu_stays:
        return "icu_day"
    return "admission_relative"


def _admission_relative_bucket(event_ts: datetime, admission_start: datetime) -> str:
    """Map a timestamp to the canonical admission-relative bucket label."""
    hours = (event_ts - admission_start).total_seconds() / 3600.0
    if hours < 24:
        return "h_0_24"
    if hours < 48:
        return "h_24_48"
    day = int(hours // 24)  # 2, 3, 4, ...
    return f"day_{day}"


def _pre_icu_bucket(event_ts: datetime, admission_start: datetime) -> str:
    """Bucket for events before the first ICU stay — hour-offset based."""
    hours = (event_ts - admission_start).total_seconds() / 3600.0
    if hours < 24:
        return "pre_icu_0_24"
    return "pre_icu_24_plus"


def _icu_day_bucket(event_ts: datetime, icu_intime: datetime) -> str:
    """Bucket for events inside an ICU stay — 0-indexed calendar day."""
    day_offset = (event_ts.date() - icu_intime.date()).days
    return f"icu_day_{day_offset}"


def assign_buckets(
    events: list[dict],
    admission_start: datetime,
    admission_end: datetime,
    icu_stays: list[dict] | None = None,
) -> tuple[dict[str, set[str]], BucketingMode]:
    """Partition ``events`` into temporal buckets.

    Each event must carry ``code`` (str) and ``timestamp`` (datetime).
    Events outside ``[admission_start, admission_end]`` are silently
    skipped.

    Returns ``(bucket_map, mode)`` where ``bucket_map`` is
    ``{bucket_label → set[event_codes]}``. Buckets with zero events
    are not materialised (no empty-set placeholders) — the temporal
    scorer treats "missing" and "empty" equivalently via its
    bucket-union logic.
    """
    mode = get_bucketing_mode(icu_stays)
    buckets: dict[str, set[str]] = {}

    for ev in events:
        ts: datetime = ev["timestamp"]
        code: str = ev["code"]
        if ts < admission_start or ts > admission_end:
            continue

        if mode == "icu_day":
            assert icu_stays is not None  # narrowed by get_bucketing_mode
            in_icu = False
            for stay in icu_stays:
                if stay["intime"] <= ts <= stay["outtime"]:
                    label = _icu_day_bucket(ts, stay["intime"])
                    in_icu = True
                    break
            if not in_icu:
                first_intime = min(s["intime"] for s in icu_stays)
                if ts < first_intime:
                    label = _pre_icu_bucket(ts, admission_start)
                else:
                    # Post-ICU — skip. Deliberate scope limit for 8d.
                    continue
        else:
            label = _admission_relative_bucket(ts, admission_start)

        buckets.setdefault(label, set()).add(code)

    return buckets, mode


__all__ = ["BucketingMode", "assign_buckets", "get_bucketing_mode"]
