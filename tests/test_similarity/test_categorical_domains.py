"""Tests for the frozen categorical-domain artifact (schema-grounded nominals).

Mirror of ``test_reference_ranges.py``. Where reference-ranges freeze the
*numeric* scale a quantitative trait is normalized against, this artifact freezes
the *legal value set* of each categorical column (``admission_type``, ``gender``)
so the cohort prompt teaches — and the validate-and-repair guard enforces — the
vocabulary the live MIMIC-IV schema actually uses.

The whole point is **data-driven derivation**: the domain is whatever distinct
values the column holds, most-common first. The fixture below seeds *real*
MIMIC-IV ``admission_type`` literals (``EW EMER.``, ``DIRECT EMER.``, ``URGENT``,
``ELECTIVE``) — deliberately **not** the stale MIMIC-III ``EMERGENCY`` — so a
regression that reintroduces a hardcoded MIMIC-III literal is caught here.

Fixture domain the builder sees:

    admission_type : EW EMER.(4), DIRECT EMER.(3), URGENT(2), ELECTIVE(1)  (n=10, 1 NULL dropped)
    gender         : M(3), F(2)                                            (n=5)
"""

from __future__ import annotations

import json

import duckdb
import pytest

from src.similarity.categorical_domains import (
    compute_categorical_domains,
    load_categorical_domains,
)


# ---------------------------------------------------------------------------
# Backend fixture — a dedicated DuckDB seeded with REAL MIMIC-IV nominal values
# (not the legacy EMERGENCY/ELECTIVE/URGENT of the shared synthetic_duckdb), so
# these tests prove the domain is derived from the data, and a stale MIMIC-III
# literal creeping back in is caught. Wrapped in the real _DuckDBBackend so the
# builder runs the exact ``.execute(sql, params)`` contract as production.
# ---------------------------------------------------------------------------


@pytest.fixture
def domains_backend(tmp_path):
    from src.conversational.extractor import _DuckDBBackend

    conn = duckdb.connect(str(tmp_path / "domains.duckdb"))
    conn.execute(
        "CREATE TABLE patients (subject_id INTEGER, gender VARCHAR, anchor_age INTEGER)"
    )
    conn.execute(
        "INSERT INTO patients VALUES "
        "(1,'M',65),(2,'M',72),(3,'M',58),(4,'F',45),(5,'F',80)"
    )
    conn.execute(
        "CREATE TABLE admissions (hadm_id INTEGER, subject_id INTEGER, admission_type VARCHAR)"
    )
    # EW EMER.×4, DIRECT EMER.×3, URGENT×2, ELECTIVE×1, plus one NULL (dropped).
    conn.execute(
        "INSERT INTO admissions VALUES "
        "(101,1,'EW EMER.'),(102,1,'EW EMER.'),(103,2,'EW EMER.'),(104,3,'EW EMER.'),"
        "(105,3,'DIRECT EMER.'),(106,4,'DIRECT EMER.'),(107,4,'DIRECT EMER.'),"
        "(108,5,'URGENT'),(109,5,'URGENT'),"
        "(110,2,'ELECTIVE'),"
        "(111,1,NULL)"
    )

    backend = _DuckDBBackend.__new__(_DuckDBBackend)
    backend._conn = conn
    yield backend
    conn.close()


# ---------------------------------------------------------------------------
# Builder — distinct values, most-common first
# ---------------------------------------------------------------------------


class TestComputeCategoricalDomains:
    def test_derives_values_present_in_data(self, domains_backend):
        # Proves data-driven derivation: the domain is exactly the distinct
        # non-null values the column holds — not a hardcoded tuple.
        domains = compute_categorical_domains(domains_backend)
        assert set(domains["admission_type"]["values"]) == {
            "EW EMER.",
            "DIRECT EMER.",
            "URGENT",
            "ELECTIVE",
        }

    def test_values_ordered_most_common_first(self, domains_backend):
        # Nearest-first by count, so the most representative literal heads the
        # list the prompt/example draws from.
        d = compute_categorical_domains(domains_backend)["admission_type"]
        assert d["values"] == ["EW EMER.", "DIRECT EMER.", "URGENT", "ELECTIVE"]

    def test_counts_and_total(self, domains_backend):
        d = compute_categorical_domains(domains_backend)["admission_type"]
        assert d["counts"] == {
            "EW EMER.": 4,
            "DIRECT EMER.": 3,
            "URGENT": 2,
            "ELECTIVE": 1,
        }
        assert d["n"] == 10  # 11 rows minus the 1 NULL

    def test_nulls_excluded(self, domains_backend):
        d = compute_categorical_domains(domains_backend)["admission_type"]
        assert None not in d["values"]

    def test_no_stale_mimic3_literal(self, domains_backend):
        # Regression guard: MIMIC-IV has no ``EMERGENCY`` (that's MIMIC-III).
        d = compute_categorical_domains(domains_backend)["admission_type"]
        assert "EMERGENCY" not in d["values"]

    def test_gender_domain(self, domains_backend):
        d = compute_categorical_domains(domains_backend)["gender"]
        assert d["values"] == ["M", "F"]
        assert d["counts"] == {"M": 3, "F": 2}
        assert d["n"] == 5

    def test_column_subset_is_configurable(self, domains_backend):
        domains = compute_categorical_domains(domains_backend, columns=["gender"])
        assert set(domains) == {"gender"}


# ---------------------------------------------------------------------------
# Artifact assembly + loader round-trip
# ---------------------------------------------------------------------------


class TestArtifactRoundTrip:
    def test_build_artifact_has_metadata_and_domains(self, domains_backend):
        from src.similarity.categorical_domains import build_artifact

        art = build_artifact(domains_backend, source="test")
        assert art["source"] == "test"
        assert "generated_at" in art
        assert "version" in art
        assert "admission_type" in art["domains"]
        assert art["domains"]["admission_type"]["values"][0] == "EW EMER."

    def test_loader_parses_values_into_tuples(self, tmp_path, monkeypatch):
        import src.similarity.categorical_domains as cd

        artifact = {
            "version": "1",
            "source": "all_admissions",
            "domains": {
                "admission_type": {
                    "values": ["EW EMER.", "DIRECT EMER.", "URGENT"],
                    "counts": {"EW EMER.": 177459, "DIRECT EMER.": 21973, "URGENT": 5000},
                    "n": 204432,
                },
                "gender": {"values": ["M", "F"], "counts": {"M": 1, "F": 1}, "n": 2},
            },
        }
        path = tmp_path / "domains.json"
        path.write_text(json.dumps(artifact))
        monkeypatch.setattr(cd, "CATEGORICAL_DOMAINS_PATH", path)

        loaded = load_categorical_domains()
        assert loaded == {
            "admission_type": ("EW EMER.", "DIRECT EMER.", "URGENT"),
            "gender": ("M", "F"),
        }

    def test_loader_missing_file_returns_empty(self, tmp_path, monkeypatch):
        import src.similarity.categorical_domains as cd

        monkeypatch.setattr(cd, "CATEGORICAL_DOMAINS_PATH", tmp_path / "nope.json")
        assert load_categorical_domains() == {}

    def test_loader_malformed_file_returns_empty(self, tmp_path, monkeypatch):
        import src.similarity.categorical_domains as cd

        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        monkeypatch.setattr(cd, "CATEGORICAL_DOMAINS_PATH", path)
        assert load_categorical_domains() == {}

    def test_compute_then_load_round_trip(self, domains_backend, tmp_path, monkeypatch):
        import src.similarity.categorical_domains as cd

        art = cd.build_artifact(domains_backend, source="fixture")
        path = tmp_path / "domains.json"
        cd.write_artifact(art, path)
        monkeypatch.setattr(cd, "CATEGORICAL_DOMAINS_PATH", path)

        loaded = load_categorical_domains()
        assert loaded["admission_type"] == (
            "EW EMER.",
            "DIRECT EMER.",
            "URGENT",
            "ELECTIVE",
        )
        assert loaded["gender"] == ("M", "F")
