"""Tests for the frozen similarity reference-range artifact (plan task II-E).

Locked decision #6: Gower normalization ranges are **frozen** on a fixed
reference population and never re-learned per query batch (fit/transform
separation). This module pins two halves of that contract:

  * the *builder* (`compute_reference_ranges`) computes robust p1/p99 ranges
    over the SAME per-admission aggregates the cohort feature extractor pulls
    (`src/similarity/run.py:_fetch_admission_features`), so the frozen range a
    trait is normalized against matches the column it scores;
  * the *loader* (`load_reference_ranges`) reads the committed JSON artifact
    into the ``{feature: (low, high)}`` shape ``run_cohort`` consumes, and never
    raises on a missing / malformed file.

Synthetic cohort (``synthetic_duckdb_with_events``) per-admission feature
distributions the builder sees:

    age            : 65, 65, 72, 58, 45, 80         (n=6, real range)
    creatinine_max : 1.2(101), 0.9(103), 1.5(106)   (n=3, real range)
    icu_los_hours  : 69.6(101), 148.8(103), 174.0   (n=3, real range)
    sodium_mean    : 140.0(101)                      (n=1 → degenerate, skipped)
    platelet_min   : (no rows)                       (n=0 → degenerate, skipped)
"""

from __future__ import annotations

import json

import pytest

from src.similarity.reference_ranges import (
    compute_reference_ranges,
    load_reference_ranges,
)


# ---------------------------------------------------------------------------
# Backend fixture — wrap the synthetic connection in the real _DuckDBBackend so
# the builder runs the exact same ``.execute(sql, params)`` contract as
# production (and as ``_fetch_admission_features``).
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_backend(synthetic_duckdb_with_events):
    from src.conversational.extractor import _DuckDBBackend

    backend = _DuckDBBackend.__new__(_DuckDBBackend)
    backend._conn = synthetic_duckdb_with_events
    return backend


# ---------------------------------------------------------------------------
# Builder — robust percentiles over the per-admission aggregates
# ---------------------------------------------------------------------------


class TestComputeReferenceRanges:
    def test_returns_ranges_for_features_with_spread(self, rich_backend):
        ranges = compute_reference_ranges(rich_backend)
        # Only features with ≥2 distinct measured values get a usable range.
        assert set(ranges) == {"age", "creatinine_max", "icu_los_hours"}

    def test_age_range_brackets_population(self, rich_backend):
        r = compute_reference_ranges(rich_backend)["age"]
        # ages 45..80; p1/p99 (linear interp) land just inside the extremes.
        assert 45.0 <= r["low"] <= 47.0
        assert 79.0 <= r["high"] <= 80.0
        assert r["low"] < r["high"]
        assert r["n"] == 6

    def test_creatinine_range_matches_per_admission_max(self, rich_backend):
        r = compute_reference_ranges(rich_backend)["creatinine_max"]
        # per-hadm MAX creatinine over {1.2, 0.9, 1.5}.
        assert 0.9 <= r["low"] <= 1.0
        assert 1.4 <= r["high"] <= 1.5
        assert r["n"] == 3

    def test_icu_los_uses_summed_hours(self, rich_backend):
        r = compute_reference_ranges(rich_backend)["icu_los_hours"]
        # SUM(los)*24 over {69.6, 148.8, 174.0}.
        assert 69.0 <= r["low"] <= 75.0
        assert 170.0 <= r["high"] <= 174.0
        assert r["n"] == 3

    def test_degenerate_features_are_skipped(self, rich_backend):
        # sodium_mean has a single value (140) → zero-width; platelet_min has
        # no rows → undefined. Neither yields a usable normalization range, so
        # the builder drops them rather than emitting R=0 (a divide-by-zero
        # trap for the Gower kernel).
        ranges = compute_reference_ranges(rich_backend)
        assert "sodium_mean" not in ranges
        assert "platelet_min" not in ranges

    def test_feature_subset_is_configurable(self, rich_backend):
        ranges = compute_reference_ranges(rich_backend, features=["age"])
        assert set(ranges) == {"age"}

    def test_percentiles_are_configurable(self, rich_backend):
        # Wider tails (p0/p100) push the age range out to the hard extremes.
        r = compute_reference_ranges(
            rich_backend, features=["age"], percentiles=(0.0, 1.0)
        )["age"]
        assert r["low"] == pytest.approx(45.0)
        assert r["high"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Artifact assembly + loader round-trip
# ---------------------------------------------------------------------------


class TestArtifactRoundTrip:
    def test_build_artifact_has_metadata_and_ranges(self, rich_backend):
        from src.similarity.reference_ranges import build_artifact

        art = build_artifact(rich_backend, cohort="all_admissions")
        assert art["cohort"] == "all_admissions"
        assert art["percentiles"] == {"low": 0.01, "high": 0.99}
        assert "generated_at" in art
        assert "age" in art["ranges"]

    def test_loader_parses_low_high_tuples(self, tmp_path, monkeypatch):
        import src.similarity.reference_ranges as rr

        artifact = {
            "version": "1",
            "cohort": "all_admissions",
            "percentiles": {"low": 0.01, "high": 0.99},
            "ranges": {
                "age": {"low": 18.0, "high": 91.0, "n": 1000},
                "creatinine_max": {"low": 0.4, "high": 9.5, "n": 800},
            },
        }
        path = tmp_path / "ranges.json"
        path.write_text(json.dumps(artifact))
        monkeypatch.setattr(rr, "REFERENCE_RANGES_PATH", path)

        loaded = load_reference_ranges()
        assert loaded == {"age": (18.0, 91.0), "creatinine_max": (0.4, 9.5)}

    def test_loader_missing_file_returns_empty(self, tmp_path, monkeypatch):
        import src.similarity.reference_ranges as rr

        monkeypatch.setattr(rr, "REFERENCE_RANGES_PATH", tmp_path / "nope.json")
        assert load_reference_ranges() == {}

    def test_loader_malformed_file_returns_empty(self, tmp_path, monkeypatch):
        import src.similarity.reference_ranges as rr

        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        monkeypatch.setattr(rr, "REFERENCE_RANGES_PATH", path)
        assert load_reference_ranges() == {}

    def test_compute_then_load_round_trip(self, rich_backend, tmp_path, monkeypatch):
        import src.similarity.reference_ranges as rr

        art = rr.build_artifact(rich_backend)
        path = tmp_path / "ranges.json"
        rr.write_artifact(art, path)
        monkeypatch.setattr(rr, "REFERENCE_RANGES_PATH", path)

        loaded = load_reference_ranges()
        assert "age" in loaded
        low, high = loaded["age"]
        assert low < high
