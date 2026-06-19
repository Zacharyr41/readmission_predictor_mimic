"""Tests for the shared temporal-window generator (``src.conversational.temporal``).

Part A of the query-triage fix: a ``TemporalConstraint`` is *window/anchor*
(SQL-bound-able against a structural interval — ICU stay or hospital admission)
or *relational/Allen* (``before``/``after`` an arbitrary clinical event, which
needs the graph). This module owns the single source of truth so the planner,
the fast-path compiler, and the extractor all agree — parity by construction.
"""

from __future__ import annotations

import pytest

from src.conversational.models import TemporalConstraint
from src.conversational.temporal import (
    is_sql_window,
    parse_time_window,
    temporal_where_predicates,
)


class _FakeBackend:
    """Minimal duck-type: the generator only ever calls ``.table(name)``."""

    def table(self, name: str) -> str:
        return name


def _tc(relation: str, ref: str, window: str | None = None) -> TemporalConstraint:
    return TemporalConstraint(
        relation=relation, reference_event=ref, time_window=window
    )


# ---------------------------------------------------------------------------
# is_sql_window — the planner's window-vs-relational discriminator
# ---------------------------------------------------------------------------


class TestIsSqlWindow:
    @pytest.mark.parametrize("ref", [
        "ICU stay", "icu stay", "the ICU", "ICU admission", "admission",
        "hospital admission", "hospital stay", "hospitalization", "admit",
    ])
    def test_recognized_anchors_are_windows(self, ref):
        assert is_sql_window(_tc("during", ref)) is True

    @pytest.mark.parametrize("ref", [
        "intubation", "extubation", "surgery", "dialysis", "vasopressor start",
        "death", "discharge",
    ])
    def test_arbitrary_events_are_relational(self, ref):
        assert is_sql_window(_tc("before", ref)) is False

    def test_icu_takes_priority_over_admission(self):
        # "ICU admission" contains both keywords; ICU is the more specific
        # anchor and must win so the bound uses intime/outtime (matches the
        # extractor's existing behaviour for test_temporal_constraint_within).
        preds = temporal_where_predicates(
            [_tc("within", "ICU admission", "24h")],
            "l.charttime", "l.hadm_id", _FakeBackend(),
        )
        assert len(preds) == 1
        assert "icustays" in preds[0]
        assert "intime" in preds[0]
        assert "admissions" not in preds[0]


# ---------------------------------------------------------------------------
# parse_time_window — moved verbatim from the extractor
# ---------------------------------------------------------------------------


class TestParseTimeWindow:
    @pytest.mark.parametrize("window,expected", [
        ("24h", "INTERVAL 24 HOUR"),
        ("48 hours", "INTERVAL 48 HOUR"),
        ("7d", "INTERVAL 7 DAY"),
        ("30m", "INTERVAL 30 MINUTE"),
    ])
    def test_parses(self, window, expected):
        assert parse_time_window(window) == expected

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_time_window("soon")


# ---------------------------------------------------------------------------
# temporal_where_predicates — bare predicates (no leading AND), parameterless
# ---------------------------------------------------------------------------


class TestWherePredicates:
    def test_empty_for_no_constraints(self):
        assert temporal_where_predicates(
            [], "l.charttime", "l.hadm_id", _FakeBackend()
        ) == []

    def test_relational_constraint_emits_nothing(self):
        # "before intubation" needs the graph; the generator produces no bound.
        assert temporal_where_predicates(
            [_tc("before", "intubation")], "l.charttime", "l.hadm_id", _FakeBackend()
        ) == []

    def test_predicates_have_no_leading_and(self):
        # The compiler list-joins with " AND " — a leading AND would double-glue.
        preds = temporal_where_predicates(
            [_tc("during", "ICU stay")], "l.charttime", "l.hadm_id", _FakeBackend()
        )
        assert preds and not preds[0].lstrip().startswith("AND")
        assert preds[0].startswith("EXISTS")

    def test_during_icu_bounds_intime_outtime(self):
        preds = temporal_where_predicates(
            [_tc("during", "ICU stay")], "l.charttime", "l.hadm_id", _FakeBackend()
        )
        sql = preds[0]
        assert "icustays" in sql
        assert "_win.hadm_id = l.hadm_id" in sql
        assert "l.charttime >= _win.intime" in sql
        assert "l.charttime <= _win.outtime" in sql

    def test_during_admission_bounds_admittime_dischtime(self):
        preds = temporal_where_predicates(
            [_tc("during", "admission")], "l.charttime", "l.hadm_id", _FakeBackend()
        )
        sql = preds[0]
        assert "admissions" in sql
        assert "l.charttime >= _win.admittime" in sql
        assert "l.charttime <= _win.dischtime" in sql

    def test_within_admission_uses_interval_from_admittime(self):
        preds = temporal_where_predicates(
            [_tc("within", "admission", "24h")],
            "l.charttime", "l.hadm_id", _FakeBackend(),
        )
        sql = preds[0]
        assert "l.charttime >= _win.admittime" in sql
        assert "INTERVAL 24 HOUR" in sql

    def test_within_without_window_emits_nothing(self):
        assert temporal_where_predicates(
            [_tc("within", "ICU stay")], "l.charttime", "l.hadm_id", _FakeBackend()
        ) == []

    def test_before_after_icu_bound(self):
        before = temporal_where_predicates(
            [_tc("before", "ICU stay")], "l.charttime", "l.hadm_id", _FakeBackend()
        )[0]
        after = temporal_where_predicates(
            [_tc("after", "ICU stay")], "l.charttime", "l.hadm_id", _FakeBackend()
        )[0]
        assert "l.charttime < _win.intime" in before
        assert "l.charttime > _win.outtime" in after
