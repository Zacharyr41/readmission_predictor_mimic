"""Tests for the health_evidence tools.

Three tools are exposed to the EvidenceAgent: ``pubmed_search``,
``mimic_distribution_lookup``, and ``loinc_reference_range``. Every tool
follows the envelope contract::

    {"status": "ok", "results": [...]}

or::

    {"status": "unavailable", "error": "..."}

Tools never raise — the EvidenceAgent's loop relies on this for simple
failure handling. Result payloads must fit in 4 KB when JSON-serialized
so the agent's context doesn't explode.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from src.conversational.health_evidence import tools as he_tools
from src.conversational.health_evidence.tools import (
    _MAX_TOOL_RESULT_BYTES,
    loinc_reference_range,
    mimic_distribution_lookup,
    pubmed_search,
)

# Imported below in test class — kept out of the top-level imports so
# the import error in pre-RED state surfaces as a class-level fail
# rather than blocking the whole module.


# ---------------------------------------------------------------------------
# Helpers (parallel to test_critic_tools.py shape)
# ---------------------------------------------------------------------------


def _make_response(
    json_payload: dict | None = None,
    *,
    raise_on_call: Exception | None = None,
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    if raise_on_call is not None:
        resp.raise_for_status.side_effect = raise_on_call
    else:
        resp.raise_for_status.return_value = None
    if json_payload is not None:
        resp.json.return_value = json_payload
    return resp


def _esearch_payload(idlist: list[str]) -> dict:
    return {"esearchresult": {"idlist": idlist, "count": str(len(idlist))}}


def _esummary_payload(records: list[dict]) -> dict:
    uids = [r["pmid"] for r in records]
    result: dict = {"uids": uids}
    for r in records:
        result[r["pmid"]] = {
            "uid": r["pmid"],
            "title": r["title"],
            "source": r.get("source", "Test Journal"),
            "pubdate": r.get("pubdate", "2024"),
        }
    return {"result": result}


# ===========================================================================
# pubmed_search — moved verbatim from critic_tools; re-tested at new path
# ===========================================================================


class TestPubmedSearchHappyPath:
    def test_happy_path_returns_records(self, monkeypatch):
        responses = iter([
            _make_response(_esearch_payload(["12345", "67890"])),
            _make_response(_esummary_payload([
                {"pmid": "12345", "title": "First study"},
                {"pmid": "67890", "title": "Second study"},
            ])),
        ])
        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get",
            lambda *a, **kw: next(responses),
        )

        result = pubmed_search("lactate sepsis")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        pmids = [r["pmid"] for r in result["results"]]
        assert pmids == ["12345", "67890"]
        assert result["results"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345/"

    def test_passes_query_and_retmax_to_esearch(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured["url"] = url
            captured["params"] = params or {}
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get",
            fake_get,
        )
        pubmed_search("creatinine kidney injury", max_results=3)
        assert "esearch" in captured["url"]
        assert captured["params"].get("term") == "creatinine kidney injury"
        assert int(captured["params"].get("retmax")) == 3
        assert captured["params"].get("db") == "pubmed"


class TestPubmedSearchEdgeCases:
    def test_empty_results_skips_esummary(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            if "esearch" in url:
                return _make_response(_esearch_payload([]))
            return _make_response(_esummary_payload([]))

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = pubmed_search("nonexistent analyte")
        assert result == {"status": "ok", "results": []}
        assert len(calls) == 1
        assert "esearch" in calls[0]

    def test_network_error_returns_unavailable(self, monkeypatch):
        def fake_get(*a, **kw):
            raise requests.RequestException("connection refused")

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = pubmed_search("anything")
        assert result["status"] == "unavailable"
        assert "connection refused" in result["error"]

    def test_malformed_esummary_returns_unavailable(self, monkeypatch):
        responses = iter([
            _make_response(_esearch_payload(["12345"])),
            _make_response({"unexpected": "shape"}),
        ])
        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get",
            lambda *a, **kw: next(responses),
        )
        result = pubmed_search("anything")
        assert result["status"] == "unavailable"

    def test_esearch_http_error_returns_unavailable(self, monkeypatch):
        def fake_get(*a, **kw):
            return _make_response(
                None,
                raise_on_call=requests.HTTPError("503 Service Unavailable"),
            )

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = pubmed_search("anything")
        assert result["status"] == "unavailable"


class TestPubmedSearchSizeAndKeys:
    def test_truncates_serialized_result_to_4kb(self, monkeypatch):
        big_title = "X" * 50_000
        responses = iter([
            _make_response(_esearch_payload(["1"])),
            _make_response(_esummary_payload([
                {"pmid": "1", "title": big_title},
            ])),
        ])
        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get",
            lambda *a, **kw: next(responses),
        )
        result = pubmed_search("x")
        assert result["status"] == "ok"
        size = len(json.dumps(result).encode("utf-8"))
        assert size <= _MAX_TOOL_RESULT_BYTES, (
            f"serialized size {size} exceeds budget"
        )

    def test_max_results_capped_at_five(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured["params"] = params or {}
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        pubmed_search("x", max_results=99)
        assert int(captured["params"]["retmax"]) <= 5

    def test_max_results_default_is_five(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured["params"] = params or {}
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        pubmed_search("x")
        assert int(captured["params"]["retmax"]) == 5


# ===========================================================================
# mimic_distribution_lookup
# ===========================================================================


class TestMimicDistributionLookup:
    def test_missing_registry_returns_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            he_tools, "MIMIC_DISTRIBUTIONS_PATH",
            tmp_path / "absent_lab_distributions.json",
        )
        result = mimic_distribution_lookup(50912)
        assert result["status"] == "unavailable"
        assert "not found" in result["error"]

    def test_malformed_registry_returns_unavailable(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        result = mimic_distribution_lookup(50912)
        assert result["status"] == "unavailable"

    def test_non_dict_registry_returns_unavailable(self, monkeypatch, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        result = mimic_distribution_lookup(50912)
        assert result["status"] == "unavailable"

    def test_unknown_itemid_returns_unavailable(self, monkeypatch, tmp_path):
        path = tmp_path / "dist.json"
        path.write_text(json.dumps({
            "50912": {"n": 1234, "mean": 1.4, "p50": 1.1, "p95": 4.0, "units": "mg/dL"},
        }))
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        result = mimic_distribution_lookup(99999)
        assert result["status"] == "unavailable"
        assert "not in registry" in result["error"]

    def test_invalid_itemid_returns_unavailable(self, monkeypatch, tmp_path):
        path = tmp_path / "dist.json"
        path.write_text(json.dumps({}))
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        result = mimic_distribution_lookup("not-an-int")  # type: ignore[arg-type]
        assert result["status"] == "unavailable"

    def test_happy_path_returns_distribution(self, monkeypatch, tmp_path):
        path = tmp_path / "dist.json"
        path.write_text(json.dumps({
            "50912": {
                "n": 12345, "mean": 1.42, "p50": 1.10,
                "p95": 4.05, "units": "mg/dL",
            },
        }))
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        result = mimic_distribution_lookup(50912)
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        rec = result["results"][0]
        assert rec["itemid"] == 50912
        assert rec["mean"] == pytest.approx(1.42)
        assert rec["units"] == "mg/dL"

    def test_size_budget_enforced(self, monkeypatch, tmp_path):
        path = tmp_path / "dist.json"
        oversized_units = "x" * 50_000
        path.write_text(json.dumps({
            "1": {
                "n": 1, "mean": 0, "p50": 0, "p95": 0,
                "units": oversized_units,
            },
        }))
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        result = mimic_distribution_lookup(1)
        size = len(json.dumps(result).encode("utf-8"))
        assert size <= _MAX_TOOL_RESULT_BYTES

    def test_default_path_when_file_absent(self):
        """Ship-as-is: data/processed/lab_distributions.json may not exist
        in v1; the tool returns unavailable cleanly without the test
        having to monkeypatch."""
        # We don't touch MIMIC_DISTRIBUTIONS_PATH here intentionally — this
        # is a smoke test that the *default* path's absent-case returns
        # the right envelope (the file genuinely does not exist in v1).
        result = mimic_distribution_lookup(50912)
        # If the file ever does land, this test will need adjusting; for
        # now it confirms the unavailable envelope under realistic state.
        assert result["status"] in {"ok", "unavailable"}
        if result["status"] == "unavailable":
            assert "error" in result


# ===========================================================================
# mimic_distribution_lookup — cohort-stratified (Phase H Tier D)
# ===========================================================================


def _flat_fixture(tmp_path, itemid: int, stats: dict):
    """Helper: write a legacy flat-schema fixture file."""
    path = tmp_path / "dist_flat.json"
    path.write_text(json.dumps({str(itemid): stats}))
    return path


def _nested_fixture(tmp_path, itemid: int, by_cohort: dict[str, dict]):
    """Helper: write a nested-schema fixture file."""
    path = tmp_path / "dist_nested.json"
    path.write_text(json.dumps({str(itemid): by_cohort}))
    return path


class TestCohortStratifiedLookup:
    """Phase H Tier D: cohort= parameter on mimic_distribution_lookup.

    Both schemas (legacy flat, new nested) must be supported. Result
    records gain ``cohort`` and ``source`` fields so the agent's
    citation tracking + telemetry can attribute the slice."""

    def test_flat_schema_returns_all_bucket_with_cohort_field(
        self, monkeypatch, tmp_path,
    ):
        """Old flat fixture: cohort=None returns those stats tagged
        with cohort='all', source='catalog' so downstream code never
        sees a result without the new fields."""
        path = _flat_fixture(tmp_path, 50912, {
            "n": 1234, "mean": 1.42, "p50": 1.10,
            "p95": 4.05, "units": "mg/dL",
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912)
        assert r["status"] == "ok"
        assert r["results"][0]["cohort"] == "all"
        assert r["results"][0]["source"] == "catalog"
        assert r["results"][0]["mean"] == pytest.approx(1.42)

    def test_nested_schema_default_cohort_returns_all_bucket(
        self, monkeypatch, tmp_path,
    ):
        path = _nested_fixture(tmp_path, 50912, {
            "all":    {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
            "sepsis": {"n": 200,  "mean": 2.5, "p50": 2.1, "p95": 6.5, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912)
        assert r["status"] == "ok"
        assert r["results"][0]["cohort"] == "all"
        assert r["results"][0]["mean"] == pytest.approx(1.0)
        assert r["results"][0]["source"] == "catalog"

    def test_nested_schema_with_explicit_cohort(self, monkeypatch, tmp_path):
        path = _nested_fixture(tmp_path, 50912, {
            "all":    {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
            "sepsis": {"n": 200,  "mean": 2.5, "p50": 2.1, "p95": 6.5, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912, cohort="sepsis")
        assert r["status"] == "ok"
        assert r["results"][0]["cohort"] == "sepsis"
        assert r["results"][0]["mean"] == pytest.approx(2.5)
        assert r["results"][0]["source"] == "catalog"

    def test_alias_resolves_to_canonical_cohort(self, monkeypatch, tmp_path):
        """Critic passes 'MI' / 'myocardial infarction' / 'heart attack' —
        all resolve to canonical 'mi_acute'. Result reports the
        canonical name, not the user's phrase."""
        path = _nested_fixture(tmp_path, 51301, {
            "all":      {"n": 9000, "mean": 9.0, "p50": 8.0, "p95": 17.0, "units": "K/uL"},
            "mi_acute": {"n": 1500, "mean": 11.5, "p50": 10.5, "p95": 22.0, "units": "K/uL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        for natural in ("MI", "myocardial infarction", "heart attack"):
            r = mimic_distribution_lookup(51301, cohort=natural)
            assert r["status"] == "ok", f"failed for {natural!r}"
            assert r["results"][0]["cohort"] == "mi_acute", natural

    def test_alias_resolution_case_insensitive(self, monkeypatch, tmp_path):
        path = _nested_fixture(tmp_path, 50912, {
            "all":           {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
            "heart_failure": {"n": 200,  "mean": 1.8, "p50": 1.6, "p95": 4.5, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        for phrase in ("CHF", "chf", "Chf", "Congestive Heart Failure"):
            r = mimic_distribution_lookup(50912, cohort=phrase)
            assert r["status"] == "ok", phrase
            assert r["results"][0]["cohort"] == "heart_failure", phrase

    def test_explicit_all_cohort_works(self, monkeypatch, tmp_path):
        """The reserved name 'all' is not in the registry, but it IS
        a valid cohort= value (the unstratified bucket)."""
        path = _nested_fixture(tmp_path, 50912, {
            "all":    {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
            "sepsis": {"n": 200,  "mean": 2.5, "p50": 2.1, "p95": 6.5, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912, cohort="all")
        assert r["status"] == "ok"
        assert r["results"][0]["cohort"] == "all"
        assert r["results"][0]["mean"] == pytest.approx(1.0)

    def test_unknown_cohort_returns_helpful_error(self, monkeypatch, tmp_path):
        path = _nested_fixture(tmp_path, 50912, {
            "all": {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912, cohort="not_a_cohort")
        assert r["status"] == "unavailable"
        # Error must mention the registered names so the model can
        # recover, AND mention icd10_prefixes as the escape hatch.
        err = r["error"]
        assert "sepsis" in err
        assert "icd10_prefixes" in err

    def test_cohort_registered_but_not_in_catalog_for_itemid(
        self, monkeypatch, tmp_path,
    ):
        """Cohort name is in the registry, but THIS itemid's catalog
        entry doesn't have a stratified bucket for it. For Inc 3 this
        returns unavailable; Inc 4 will turn it into an on-the-fly
        compute call, but the tool API contract is the same — the
        return shape stays unavailable here."""
        path = _nested_fixture(tmp_path, 50912, {
            # Only 'all' bucket — sepsis is registered but missing.
            "all": {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912, cohort="sepsis")
        # Inc 3 returns unavailable; Inc 4 will fall back to compute.
        # Either is acceptable — but if ok, the cohort label is sepsis.
        if r["status"] == "ok":
            assert r["results"][0]["cohort"] == "sepsis"
        else:
            # Error message should mention this itemid lacks the cohort.
            assert "sepsis" in r["error"] or "cohort" in r["error"].lower()

    def test_size_budget_under_4kb_with_nested_fixture(
        self, monkeypatch, tmp_path,
    ):
        """Single-cohort lookup must stay under 4KB regardless of how
        many cohorts the catalog contains for this itemid."""
        # 15 cohorts × ~100 bytes each → fits, but the lookup ONLY
        # returns one record so the response is tiny.
        many_cohorts = {
            name: {"n": 100, "mean": 1.0, "p50": 1.0, "p95": 1.0,
                   "units": "x" * 200}
            for name in (
                "all", "sepsis", "septic_shock", "aki", "mi_acute",
                "heart_failure", "hepatic_failure", "stroke_ischemic",
                "ards", "pneumonia", "copd", "diabetes", "ckd",
                "atrial_fibrillation", "coagulopathy", "respiratory_failure",
                "covid_19",
            )
        }
        path = _nested_fixture(tmp_path, 50912, many_cohorts)
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912, cohort="sepsis")
        assert r["status"] == "ok"
        assert len(json.dumps(r).encode("utf-8")) <= _MAX_TOOL_RESULT_BYTES

    def test_result_has_all_required_fields(self, monkeypatch, tmp_path):
        """Result records always carry: itemid, cohort, source, n,
        mean, p50, p95, units."""
        path = _nested_fixture(tmp_path, 50912, {
            "all": {"n": 1000, "mean": 1.0, "p50": 0.9, "p95": 3.0, "units": "mg/dL"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", path)
        r = mimic_distribution_lookup(50912)
        rec = r["results"][0]
        for field in ("itemid", "cohort", "source", "n", "mean", "p50", "p95", "units"):
            assert field in rec, f"missing {field!r}"


# ===========================================================================
# mimic_distribution_lookup — on-the-fly compute (Inc 4)
# ===========================================================================


def _make_compute_duckdb(
    tmp_path, *,
    labevents_rows=(),
    chartevents_rows=(),
    diagnoses_rows=(),
):
    """Build a tiny MIMIC-shaped duckdb fixture for the compute path.

    Schema mirrors the real MIMIC labevents/chartevents/diagnoses_icd
    tables but with only the columns the compute helper touches."""
    import duckdb
    db_path = tmp_path / "mimic_compute.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE labevents ("
        "  subject_id INTEGER, hadm_id INTEGER, itemid INTEGER,"
        "  valuenum DOUBLE, valueuom VARCHAR"
        ")"
    )
    con.execute(
        "CREATE TABLE chartevents ("
        "  subject_id INTEGER, hadm_id INTEGER, itemid INTEGER,"
        "  valuenum DOUBLE, valueuom VARCHAR"
        ")"
    )
    con.execute(
        "CREATE TABLE diagnoses_icd ("
        "  subject_id INTEGER, hadm_id INTEGER,"
        "  icd_code VARCHAR, icd_version INTEGER"
        ")"
    )
    for row in labevents_rows:
        con.execute("INSERT INTO labevents VALUES (?,?,?,?,?)", row)
    for row in chartevents_rows:
        con.execute("INSERT INTO chartevents VALUES (?,?,?,?,?)", row)
    for row in diagnoses_rows:
        con.execute("INSERT INTO diagnoses_icd VALUES (?,?,?,?)", row)
    con.close()
    return db_path


def _seed_sepsis_lactate_fixture(tmp_path, *, n_sepsis=50, n_other=50):
    """Build a duckdb fixture with sepsis-coded and non-sepsis hadm_ids
    each carrying lactate (itemid=50813). Returns the duckdb path.

    Sepsis hadms get higher lactate values (mean ~5) — non-sepsis get
    lower (mean ~1.5) — so the cohort-stratified result differs from
    the all-cohort result and we can distinguish cache hits from
    compute hits."""
    labevents = []
    diagnoses = []
    # Sepsis hadm_ids 1..n_sepsis with lactate ~5 mmol/L
    for hadm_id in range(1, n_sepsis + 1):
        labevents.append((100 + hadm_id, hadm_id, 50813, 5.0 + 0.01 * hadm_id, "mmol/L"))
        diagnoses.append((100 + hadm_id, hadm_id, "A419", 10))  # sepsis ICD-10
    # Non-sepsis hadm_ids 1001..1001+n_other with lactate ~1.5 mmol/L
    for i, hadm_id in enumerate(range(1001, 1001 + n_other), start=1):
        labevents.append((1100 + i, hadm_id, 50813, 1.0 + 0.01 * i, "mmol/L"))
        # No sepsis diagnosis for these — give them another code
        diagnoses.append((1100 + i, hadm_id, "Z000", 10))
    return _make_compute_duckdb(
        tmp_path,
        labevents_rows=labevents,
        diagnoses_rows=diagnoses,
    )


class TestOnTheFlyComputeFallback:
    """Phase H Tier D — Inc 4. The catalog is a hot cache; when a
    requested (itemid, cohort) pair isn't in it, the tool opens the
    MIMIC duckdb read-only and computes the stats fresh. Two entry
    paths: named cohort (registry-resolved) and raw ICD prefixes
    (caller-supplied)."""

    def test_named_cohort_falls_back_to_compute_when_missing(
        self, monkeypatch, tmp_path,
    ):
        """Catalog only has 'all' bucket for itemid 50813; cohort='sepsis'
        triggers compute via the registry's prefixes. Result tagged
        source='computed', cohort='sepsis'."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "ok"
        rec = r["results"][0]
        assert rec["cohort"] == "sepsis"
        assert rec["source"] == "computed"
        # Sepsis cohort lactate ~5 mmol/L (higher than the 'all' mean=3
        # in the catalog)
        assert rec["mean"] > 4.0
        assert rec["units"] == "mmol/L"

    def test_alias_phrase_falls_back_to_compute(
        self, monkeypatch, tmp_path,
    ):
        """Critic passes 'septicemia' (a sepsis alias). The tool
        resolves to canonical 'sepsis' and computes on the fly."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(50813, cohort="septicemia")
        assert r["status"] == "ok"
        # Result records the CANONICAL name, not the user's phrase
        assert r["results"][0]["cohort"] == "sepsis"
        assert r["results"][0]["source"] == "computed"

    def test_raw_icd10_prefixes_compute_directly(
        self, monkeypatch, tmp_path,
    ):
        """Caller passes icd10_prefixes — no registry lookup needed.
        Skip the catalog entirely. Result cohort='custom',
        source='computed', icd_prefixes echoed back."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(
            50813, icd10_prefixes=["A41."],
        )
        assert r["status"] == "ok"
        rec = r["results"][0]
        assert rec["cohort"] == "custom"
        assert rec["source"] == "computed"
        assert rec["icd_prefixes"] == ["A41."]

    def test_raw_icd9_prefixes_alone(self, monkeypatch, tmp_path):
        """ICD-9-only filter still works (older subsets / pre-2015 data)."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        # Build a fixture using ICD-9 sepsis codes
        labevents = [
            (100 + h, h, 50813, 5.0 + 0.01 * h, "mmol/L") for h in range(1, 51)
        ]
        diagnoses = [
            (100 + h, h, "99591", 9) for h in range(1, 51)
        ]
        compute_db = _make_compute_duckdb(
            tmp_path,
            labevents_rows=labevents,
            diagnoses_rows=diagnoses,
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(
            50813, icd9_prefixes=["995.91"],
        )
        assert r["status"] == "ok"
        assert r["results"][0]["cohort"] == "custom"
        assert r["results"][0]["source"] == "computed"

    def test_raw_prefixes_override_named_cohort(
        self, monkeypatch, tmp_path,
    ):
        """If both cohort= and icd10_prefixes= are given, prefixes win.
        cohort label becomes 'custom', not the registered name."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(
            50813, cohort="sepsis", icd10_prefixes=["A41."],
        )
        assert r["status"] == "ok"
        assert r["results"][0]["cohort"] == "custom"
        assert r["results"][0]["icd_prefixes"] == ["A41."]

    def test_unknown_cohort_with_no_prefixes_errors_helpfully(
        self, monkeypatch, tmp_path,
    ):
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        # No duckdb monkeypatch needed — error happens before compute.
        r = mimic_distribution_lookup(50813, cohort="not_a_cohort")
        assert r["status"] == "unavailable"
        err = r["error"]
        assert "icd10_prefixes" in err
        assert "sepsis" in err  # at least one example registered name

    def test_compute_skipped_when_duckdb_missing(
        self, monkeypatch, tmp_path,
    ):
        """If MIMIC_COMPUTE_DUCKDB_PATH points at a non-existent file,
        the compute path must NOT raise — return unavailable."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        monkeypatch.setattr(
            he_tools, "MIMIC_COMPUTE_DUCKDB_PATH",
            tmp_path / "absent.duckdb",
        )
        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "unavailable"

    def test_cache_hit_preferred_over_compute(
        self, monkeypatch, tmp_path,
    ):
        """If the (itemid, cohort) pair is in the catalog, the tool
        must NOT open duckdb. Validate by pointing duckdb at a path
        that doesn't exist — if compute were attempted, it would
        skip and return unavailable. Cache hit must return ok."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
            "sepsis": {"n": 50, "mean": 5.0, "p50": 4.5, "p95": 8.5, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        monkeypatch.setattr(
            he_tools, "MIMIC_COMPUTE_DUCKDB_PATH",
            tmp_path / "absent_for_cache_hit.duckdb",
        )
        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "ok"
        assert r["results"][0]["source"] == "catalog"
        assert r["results"][0]["mean"] == pytest.approx(5.0)

    def test_compute_below_n_threshold_returns_unavailable(
        self, monkeypatch, tmp_path,
    ):
        """When the cohort filter matches < 30 rows for this itemid,
        suppress noisy stats. Same threshold the generator uses."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        # Only 5 sepsis-coded rows — well below the n=30 floor.
        compute_db = _seed_sepsis_lactate_fixture(tmp_path, n_sepsis=5, n_other=0)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "unavailable"

    def test_prefix_validation_rejects_sql_injection(
        self, monkeypatch, tmp_path,
    ):
        """ICD prefix must match [A-Za-z0-9.] — anything else is
        rejected before any duckdb file is opened (security gate).
        Also: the duckdb file must be unmodified after the call."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        size_before = compute_db.stat().st_size
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(
            50813, icd10_prefixes=["A41.", "'; DROP TABLE labevents; --"],
        )
        assert r["status"] == "unavailable"
        assert "invalid" in r["error"].lower() or "prefix" in r["error"].lower()

        # File unchanged.
        assert compute_db.stat().st_size == size_before

    def test_prefix_validation_rejects_wildcard_chars(
        self, monkeypatch, tmp_path,
    ):
        """SQL LIKE wildcards in the prefix would let the caller match
        unintended codes. Reject them at validation time."""
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        for bad in ("%", "_", "%41", "_41", "A%41"):
            r = mimic_distribution_lookup(
                50813, icd10_prefixes=[bad],
            )
            assert r["status"] == "unavailable", f"failed for {bad!r}"

    def test_missing_itemid_in_compute_is_handled(
        self, monkeypatch, tmp_path,
    ):
        """itemid that doesn't exist in labevents/chartevents → 0 rows
        → unavailable with insufficient-data error."""
        catalog = _nested_fixture(tmp_path, 99999, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "x"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        # itemid 99999 is in the catalog but not in the duckdb fixture
        r = mimic_distribution_lookup(99999, cohort="sepsis")
        assert r["status"] == "unavailable"


class TestComputeBackendDispatch:
    """The compute helper must match the user's session backend
    (DATA_SOURCE=local vs DATA_SOURCE=bigquery). Otherwise the critic's
    distributions wouldn't reflect the data the answer was computed
    against."""

    def test_local_backend_uses_duckdb(self, monkeypatch, tmp_path):
        """DATA_SOURCE=local (or unset) routes to the duckdb path."""
        monkeypatch.setenv("DATA_SOURCE", "local")
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        compute_db = _seed_sepsis_lactate_fixture(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", compute_db)

        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "ok"
        assert r["results"][0]["source"] == "computed"

    def test_bigquery_backend_uses_bigquery_client(
        self, monkeypatch, tmp_path,
    ):
        """DATA_SOURCE=bigquery routes to the BigQuery path. The
        bigquery client is mocked — no live BQ call. Verify the
        fully-qualified table names appear in the issued query."""
        monkeypatch.setenv("DATA_SOURCE", "bigquery")
        monkeypatch.setenv("BIGQUERY_PROJECT", "test-project")
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)

        # Fake bigquery module: client.query(sql).result() returns a row
        # iterable with attributes (n, mean, p50, p95, units).
        from unittest.mock import MagicMock
        fake_row = MagicMock()
        fake_row.n = 50
        fake_row.mean = 5.2
        fake_row.p50 = 5.0
        fake_row.p95 = 7.8
        fake_row.units = "mmol/L"
        fake_job = MagicMock()
        fake_job.result.return_value = iter([fake_row])
        fake_client = MagicMock()
        fake_client.query.return_value = fake_job
        fake_bq_module = MagicMock()
        fake_bq_module.Client.return_value = fake_client
        monkeypatch.setattr(
            he_tools, "_get_bigquery_module",
            lambda: fake_bq_module,
        )

        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "ok"
        assert r["results"][0]["source"] == "computed"
        assert r["results"][0]["mean"] == pytest.approx(5.2)

        # The query must have used the fully-qualified BigQuery tables,
        # not the bare names (which would 404 on BQ).
        sql = fake_client.query.call_args[0][0]
        assert "physionet-data" in sql or "mimiciv_3_1_hosp" in sql
        assert "labevents" in sql

    def test_bigquery_missing_project_returns_unavailable(
        self, monkeypatch, tmp_path,
    ):
        """DATA_SOURCE=bigquery without BIGQUERY_PROJECT → unavailable."""
        monkeypatch.setenv("DATA_SOURCE", "bigquery")
        monkeypatch.delenv("BIGQUERY_PROJECT", raising=False)
        catalog = _nested_fixture(tmp_path, 50813, {
            "all": {"n": 100, "mean": 3.0, "p50": 2.5, "p95": 7.0, "units": "mmol/L"},
        })
        monkeypatch.setattr(he_tools, "MIMIC_DISTRIBUTIONS_PATH", catalog)
        r = mimic_distribution_lookup(50813, cohort="sepsis")
        assert r["status"] == "unavailable"
        assert "BIGQUERY_PROJECT" in r["error"] or "project" in r["error"].lower()


# ===========================================================================
# loinc_reference_range
# ===========================================================================


class TestLoincReferenceRange:
    def test_invalid_loinc_format_returns_unavailable(self):
        result = loinc_reference_range("not-a-loinc")
        assert result["status"] == "unavailable"
        assert "invalid" in result["error"].lower()

    def test_non_string_loinc_returns_unavailable(self):
        result = loinc_reference_range(12345)  # type: ignore[arg-type]
        assert result["status"] == "unavailable"

    def test_missing_catalog_returns_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            he_tools, "LOINC_CATALOG_PATH",
            tmp_path / "absent_loinc.json",
        )
        result = loinc_reference_range("2160-0")
        assert result["status"] == "unavailable"
        assert "not found" in result["error"]

    def test_unknown_loinc_returns_unavailable(self, monkeypatch, tmp_path):
        path = tmp_path / "loinc.json"
        path.write_text(json.dumps({
            "2160-0": {"low": 0.7, "high": 1.3, "units": "mg/dL"},
        }))
        monkeypatch.setattr(he_tools, "LOINC_CATALOG_PATH", path)
        result = loinc_reference_range("99999-9")
        assert result["status"] == "unavailable"

    def test_happy_path_returns_range(self, monkeypatch, tmp_path):
        path = tmp_path / "loinc.json"
        path.write_text(json.dumps({
            "2160-0": {"low": 0.7, "high": 1.3, "units": "mg/dL"},
        }))
        monkeypatch.setattr(he_tools, "LOINC_CATALOG_PATH", path)
        result = loinc_reference_range("2160-0")
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        rec = result["results"][0]
        assert rec["loinc_code"] == "2160-0"
        assert rec["low"] == pytest.approx(0.7)
        assert rec["high"] == pytest.approx(1.3)
        assert rec["units"] == "mg/dL"

    def test_malformed_catalog_returns_unavailable(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json")
        monkeypatch.setattr(he_tools, "LOINC_CATALOG_PATH", path)
        result = loinc_reference_range("2160-0")
        assert result["status"] == "unavailable"

    def test_size_budget_enforced(self, monkeypatch, tmp_path):
        path = tmp_path / "loinc.json"
        path.write_text(json.dumps({
            "1-1": {"low": 0, "high": 0, "units": "u" * 50_000},
        }))
        monkeypatch.setattr(he_tools, "LOINC_CATALOG_PATH", path)
        result = loinc_reference_range("1-1")
        size = len(json.dumps(result).encode("utf-8"))
        assert size <= _MAX_TOOL_RESULT_BYTES


# ===========================================================================
# Tool dispatch / tool defs
# ===========================================================================


class TestToolDefs:
    def test_all_three_tools_in_dispatch(self):
        from src.conversational.health_evidence import TOOL_DISPATCH

        assert "pubmed_search" in TOOL_DISPATCH
        assert "mimic_distribution_lookup" in TOOL_DISPATCH
        assert "loinc_reference_range" in TOOL_DISPATCH
        assert TOOL_DISPATCH["pubmed_search"] is pubmed_search
        assert TOOL_DISPATCH["mimic_distribution_lookup"] is mimic_distribution_lookup
        assert TOOL_DISPATCH["loinc_reference_range"] is loinc_reference_range

    def test_all_tool_defs_have_required_fields(self):
        from src.conversational.health_evidence import ALL_TOOL_DEFS

        for tool_def in ALL_TOOL_DEFS:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def
            assert tool_def["input_schema"]["type"] == "object"
            assert "properties" in tool_def["input_schema"]
            assert "required" in tool_def["input_schema"]


# ===========================================================================
# mimic_itemid_search — analyte-name → itemid lookup (Phase H follow-up)
# ===========================================================================


def _make_itemid_search_duckdb(
    tmp_path,
    *,
    labitems: list[tuple] = (),
    chartitems: list[tuple] = (),
):
    """Build a tiny duckdb fixture with d_labitems + d_items so tests
    can exercise the real SQL path against deterministic data."""
    import duckdb
    db_path = tmp_path / "mimic_for_itemid_search.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE d_labitems ("
        "  itemid INTEGER, label VARCHAR,"
        "  fluid VARCHAR, category VARCHAR"
        ")"
    )
    con.execute(
        "CREATE TABLE d_items ("
        "  itemid INTEGER, label VARCHAR,"
        "  abbreviation VARCHAR, linksto VARCHAR,"
        "  category VARCHAR, unitname VARCHAR,"
        "  param_type VARCHAR,"
        "  lownormalvalue DOUBLE, highnormalvalue DOUBLE"
        ")"
    )
    for row in labitems:
        # row: (itemid, label, fluid, category)
        con.execute("INSERT INTO d_labitems VALUES (?, ?, ?, ?)", row)
    for row in chartitems:
        # row: (itemid, label, abbreviation, linksto, category, unitname,
        #       param_type, lownormalvalue, highnormalvalue)
        con.execute(
            "INSERT INTO d_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    con.close()
    return db_path


class TestMimicItemidSearch:
    """Phase H follow-up: maps a free-text analyte/measurement name to
    canonical MIMIC itemid candidates. Backend dispatch mirrors the
    on-the-fly cohort compute (DATA_SOURCE local vs bigquery)."""

    def test_finds_lab_item_by_label(self, monkeypatch, tmp_path):
        """Common case — query 'creatinine' returns the canonical lab
        itemid (50912 in MIMIC-IV) with match='exact'."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[
                (50912, "Creatinine", "Blood", "Chemistry"),
                (51067, "24 hr Creatinine", "Urine", "Chemistry"),
            ],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="creatinine")
        assert r["status"] == "ok"
        # At least one match
        assert any(rec["itemid"] == 50912 for rec in r["results"])
        # Exact match is reported as such
        canonical = next(rec for rec in r["results"] if rec["itemid"] == 50912)
        assert canonical["label"] == "Creatinine"
        assert canonical["match"] == "exact"
        assert canonical["table"] == "labevents"
        assert canonical["fluid"] == "Blood"
        assert canonical["category"] == "Chemistry"

    def test_finds_chart_item(self, monkeypatch, tmp_path):
        """Vital signs — query 'heart rate' returns chartevents itemid
        with table='chartevents'."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            chartitems=[
                (220045, "Heart Rate", "HR", "chartevents",
                 "Routine Vital Signs", "bpm", "numeric", 60.0, 100.0),
            ],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="heart rate")
        assert r["status"] == "ok"
        assert any(
            rec["itemid"] == 220045 and rec["table"] == "chartevents"
            for rec in r["results"]
        )

    def test_substring_match(self, monkeypatch, tmp_path):
        """Substring queries find labels that contain the substring."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[
                (50912, "Creatinine", "Blood", "Chemistry"),
            ],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        # 'creat' is a substring of 'Creatinine'
        r = mimic_itemid_search(query="creat")
        assert r["status"] == "ok"
        assert any(rec["itemid"] == 50912 for rec in r["results"])

    def test_ranking_exact_before_prefix_before_substring(
        self, monkeypatch, tmp_path,
    ):
        """When several rows match, exact-label hits rank above prefix
        hits, which rank above substring hits."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[
                (50912, "Creatinine", "Blood", "Chemistry"),                 # exact
                (51067, "Creatinine, Whole Blood", "Blood", "Chemistry"),    # prefix
                (52546, "Urine Creatinine", "Urine", "Chemistry"),           # substring
            ],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="creatinine", max_results=10)
        assert r["status"] == "ok"
        # Order: exact → prefix → substring
        ids_in_order = [rec["itemid"] for rec in r["results"]]
        assert ids_in_order.index(50912) < ids_in_order.index(51067) < ids_in_order.index(52546)
        # Match types reported
        match_by_id = {rec["itemid"]: rec["match"] for rec in r["results"]}
        assert match_by_id[50912] == "exact"
        assert match_by_id[51067] == "prefix"
        assert match_by_id[52546] == "substring"

    def test_max_results_caps(self, monkeypatch, tmp_path):
        """max_results limits the returned list."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        labitems = [
            (50000 + i, f"Creatinine variant {i}", "Blood", "Chemistry")
            for i in range(20)
        ]
        db = _make_itemid_search_duckdb(tmp_path, labitems=labitems)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="creatinine", max_results=3)
        assert r["status"] == "ok"
        assert len(r["results"]) == 3

    def test_loinc_enrichment_for_lab_items(self, monkeypatch, tmp_path):
        """Lab itemids in labitem_to_snomed.json get enriched with
        their LOINC code in the result record."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[(50912, "Creatinine", "Blood", "Chemistry")],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        # Override the labitem→loinc mapping path with a small fixture.
        mapping_path = tmp_path / "labitem_to_snomed.json"
        mapping_path.write_text(json.dumps({
            "_metadata": {"source": "test fixture"},
            "50912": {"loinc": "2160-0", "label": "Creatinine"},
        }))
        monkeypatch.setattr(he_tools, "LABITEM_TO_SNOMED_PATH", mapping_path)
        r = mimic_itemid_search(query="creatinine")
        rec = next(x for x in r["results"] if x["itemid"] == 50912)
        assert rec.get("loinc") == "2160-0"

    def test_chart_items_have_no_loinc(self, monkeypatch, tmp_path):
        """Chart items aren't in labitem_to_snomed.json by convention,
        so the result record has no 'loinc' field. The tool must NOT
        invent one."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            chartitems=[
                (220045, "Heart Rate", "HR", "chartevents",
                 "Routine Vital Signs", "bpm", "numeric", 60.0, 100.0),
            ],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        # Mapping doesn't cover chart itemids.
        mapping_path = tmp_path / "labitem_to_snomed.json"
        mapping_path.write_text(json.dumps({"_metadata": {}}))
        monkeypatch.setattr(he_tools, "LABITEM_TO_SNOMED_PATH", mapping_path)
        r = mimic_itemid_search(query="heart rate")
        rec = next(x for x in r["results"] if x["itemid"] == 220045)
        # Field is absent OR explicitly None — both acceptable shapes.
        assert rec.get("loinc") is None

    def test_not_in_mimic_returns_empty_ok(self, monkeypatch, tmp_path):
        """Critical contract: when the analyte isn't in MIMIC at all
        (procalcitonin is genuinely absent), the tool returns
        status='ok' with results=[]. NOT status='unavailable'. The
        model needs the distinction so it pivots to PubMed instead of
        retrying with another guess."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        # Empty fixture — no labitems, no chartitems.
        db = _make_itemid_search_duckdb(tmp_path)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="procalcitonin")
        assert r["status"] == "ok", (
            "empty results MUST be ok (no match found), "
            "NOT unavailable (which signals backend failure)"
        )
        assert r["results"] == []

    def test_query_validation_rejects_sql_injection(
        self, monkeypatch, tmp_path,
    ):
        """The query string is interpolated into SQL via charset-gated
        sanitisation. Classic injection payloads must be refused
        BEFORE backend is opened, AND duckdb file is unmodified."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[(50912, "Creatinine", "Blood", "Chemistry")],
        )
        size_before = db.stat().st_size
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        for bad in (
            "'; DROP TABLE d_labitems; --",
            "creatinine' OR 1=1 --",
            "creat'; DELETE FROM d_labitems; --",
        ):
            r = mimic_itemid_search(query=bad)
            assert r["status"] == "unavailable"
            assert "invalid" in r["error"].lower() or "query" in r["error"].lower()
        # File unchanged.
        assert db.stat().st_size == size_before

    def test_query_validation_rejects_sql_wildcards(
        self, monkeypatch, tmp_path,
    ):
        """LIKE wildcards in the user's query would let the caller
        scan unrelated rows. Reject % and _ in the input."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[(50912, "Creatinine", "Blood", "Chemistry")],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        for bad in ("%", "_", "%creat", "creat_"):
            r = mimic_itemid_search(query=bad)
            assert r["status"] == "unavailable", f"failed for {bad!r}"

    def test_query_too_short_or_empty_rejected(
        self, monkeypatch, tmp_path,
    ):
        """Empty / 1-char queries would match too many rows. Require
        at least 2 chars after stripping whitespace."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[(50912, "Creatinine", "Blood", "Chemistry")],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        for bad in ("", " ", "a", "  "):
            r = mimic_itemid_search(query=bad)
            assert r["status"] == "unavailable"

    def test_local_backend_used_when_data_source_local(
        self, monkeypatch, tmp_path,
    ):
        """DATA_SOURCE=local routes to the duckdb path."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        db = _make_itemid_search_duckdb(
            tmp_path,
            labitems=[(50912, "Creatinine", "Blood", "Chemistry")],
        )
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="creatinine")
        assert r["status"] == "ok"
        assert any(rec["itemid"] == 50912 for rec in r["results"])

    def test_bigquery_backend_uses_bigquery_client(
        self, monkeypatch, tmp_path,
    ):
        """DATA_SOURCE=bigquery routes to the BigQuery path. The
        bigquery client is mocked — no live BQ call. Verify the FQN
        d_labitems / d_items table identifiers appear in the SQL."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        monkeypatch.setenv("DATA_SOURCE", "bigquery")
        monkeypatch.setenv("BIGQUERY_PROJECT", "test-project")

        fake_row = MagicMock()
        fake_row.itemid = 50912
        fake_row.label = "Creatinine"
        fake_row.fluid = "Blood"
        fake_row.category = "Chemistry"
        fake_row.source_table = "labevents"
        fake_row.match_rank = 0
        fake_job = MagicMock()
        fake_job.result.return_value = iter([fake_row])
        fake_client = MagicMock()
        fake_client.query.return_value = fake_job
        fake_bq_module = MagicMock()
        fake_bq_module.Client.return_value = fake_client
        monkeypatch.setattr(
            he_tools, "_get_bigquery_module", lambda: fake_bq_module,
        )

        r = mimic_itemid_search(query="creatinine")
        assert r["status"] == "ok"
        assert any(rec["itemid"] == 50912 for rec in r["results"])
        sql = fake_client.query.call_args[0][0]
        assert "physionet-data" in sql or "d_labitems" in sql
        assert "d_items" in sql

    def test_bigquery_missing_project_returns_unavailable(
        self, monkeypatch, tmp_path,
    ):
        """DATA_SOURCE=bigquery without BIGQUERY_PROJECT → unavailable."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        monkeypatch.setenv("DATA_SOURCE", "bigquery")
        monkeypatch.delenv("BIGQUERY_PROJECT", raising=False)
        r = mimic_itemid_search(query="creatinine")
        assert r["status"] == "unavailable"

    def test_duckdb_missing_returns_unavailable(
        self, monkeypatch, tmp_path,
    ):
        """If the local duckdb path doesn't exist, return unavailable
        cleanly — never raise."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        monkeypatch.setattr(
            he_tools, "MIMIC_COMPUTE_DUCKDB_PATH",
            tmp_path / "absent.duckdb",
        )
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="creatinine")
        assert r["status"] == "unavailable"

    def test_size_budget_respected(self, monkeypatch, tmp_path):
        """Result envelope stays under the 4 KB budget even with
        many matches."""
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        labitems = [
            (50000 + i, f"Creatinine variant {i:03d} long-label-text " * 2,
             "Blood", "Chemistry")
            for i in range(50)
        ]
        db = _make_itemid_search_duckdb(tmp_path, labitems=labitems)
        monkeypatch.setattr(he_tools, "MIMIC_COMPUTE_DUCKDB_PATH", db)
        monkeypatch.setenv("DATA_SOURCE", "local")
        r = mimic_itemid_search(query="creatinine", max_results=20)
        assert r["status"] == "ok"
        size = len(json.dumps(r).encode("utf-8"))
        assert size <= _MAX_TOOL_RESULT_BYTES, (
            f"result envelope {size} bytes > 4KB budget"
        )


class TestMimicItemidSearchRegistry:
    """Inc 3 — tool def + dispatch wiring."""

    def test_in_all_tool_defs(self):
        from src.conversational.health_evidence.tool_defs import ALL_TOOL_DEFS
        assert any(
            d["name"] == "mimic_itemid_search" for d in ALL_TOOL_DEFS
        )

    def test_in_tool_dispatch(self):
        from src.conversational.health_evidence.tool_defs import TOOL_DISPATCH
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search,
        )
        assert "mimic_itemid_search" in TOOL_DISPATCH
        assert TOOL_DISPATCH["mimic_itemid_search"] is mimic_itemid_search

    def test_tool_def_advertises_query_and_max_results(self):
        from src.conversational.health_evidence.tool_defs import ALL_TOOL_DEFS
        td = next(
            d for d in ALL_TOOL_DEFS if d["name"] == "mimic_itemid_search"
        )
        props = td["input_schema"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "max_results" in props
        assert "query" in td["input_schema"]["required"]

    def test_tool_def_description_explains_when_to_use(self):
        """Description must teach the model: call this BEFORE
        mimic_distribution_lookup when the itemid is unknown."""
        from src.conversational.health_evidence.tool_defs import ALL_TOOL_DEFS
        td = next(
            d for d in ALL_TOOL_DEFS if d["name"] == "mimic_itemid_search"
        )
        desc = td["description"].lower()
        assert "itemid" in desc
        # Must connect to mimic_distribution_lookup so the model
        # learns the chain.
        assert "mimic_distribution_lookup" in desc
        # Must say something about empty results signalling
        # not-in-MIMIC, not failure.
        assert (
            "empty" in desc
            or "not in mimic" in desc
            or "no match" in desc
        )

    def test_reexported_from_health_evidence_init(self):
        from src.conversational.health_evidence import mimic_itemid_search
        from src.conversational.health_evidence.tools import (
            mimic_itemid_search as direct,
        )
        assert mimic_itemid_search is direct


# ===========================================================================
# clinical_formula_lookup — PubMed-grounded derived-quantity definitions
# ===========================================================================


_SHOCK_INDEX_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33123456</PMID>
      <Article>
        <ArticleTitle>The shock index in emergency triage</ArticleTitle>
        <Abstract>
          <AbstractText>The shock index, defined as heart rate divided by
          systolic blood pressure, is considered elevated when it is greater
          than or equal to 0.9 and predicts in-hospital mortality.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _efetch_response(text: str):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.text = text
    return resp


class TestClinicalFormulaLookup:
    """The PubMed-backed formula/definition lookup: searches PubMed (esearch)
    then fetches ABSTRACTS (efetch), where the formula is stated. Envelope
    contract identical to the other tools; never raises."""

    def test_happy_path_returns_abstract_evidence(self, monkeypatch):
        def fake_get(url, params=None, timeout=None, **kw):
            if "esearch" in url:
                return _make_response(_esearch_payload(["33123456"]))
            return _efetch_response(_SHOCK_INDEX_EFETCH_XML)

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = he_tools.clinical_formula_lookup("shock index")
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        rec = result["results"][0]
        assert rec["pmid"] == "33123456"
        assert "heart rate divided by" in rec["abstract"].lower()
        assert "systolic blood pressure" in rec["abstract"].lower()
        assert rec["url"] == "https://pubmed.ncbi.nlm.nih.gov/33123456/"

    def test_queries_pubmed_esearch_and_efetch(self, monkeypatch):
        """Proves the definition is MCP/PubMed-SOURCED (not hardcoded): both
        NCBI endpoints are hit and the formula name is in the search term."""
        calls: list[str] = []

        def fake_get(url, params=None, timeout=None, **kw):
            calls.append(url)
            if "esearch" in url:
                assert "shock index" in (params or {}).get("term", "")
                return _make_response(_esearch_payload(["33123456"]))
            assert "33123456" in (params or {}).get("id", "")
            return _efetch_response(_SHOCK_INDEX_EFETCH_XML)

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        he_tools.clinical_formula_lookup("shock index")
        assert any("esearch" in u for u in calls)
        assert any("efetch" in u for u in calls)

    def test_empty_idlist_skips_efetch(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = he_tools.clinical_formula_lookup("not a real index xyzzy")
        assert result == {"status": "ok", "results": []}
        assert len(calls) == 1 and "esearch" in calls[0]

    def test_network_error_returns_unavailable(self, monkeypatch):
        def fake_get(*a, **kw):
            raise requests.RequestException("connection refused")

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = he_tools.clinical_formula_lookup("shock index")
        assert result["status"] == "unavailable"

    def test_malformed_efetch_xml_returns_unavailable(self, monkeypatch):
        def fake_get(url, **kw):
            if "esearch" in url:
                return _make_response(_esearch_payload(["33123456"]))
            return _efetch_response("<not-valid-xml ><<<")

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        result = he_tools.clinical_formula_lookup("shock index")
        assert result["status"] == "unavailable"

    def test_invalid_name_returns_unavailable_without_network(self, monkeypatch):
        def fake_get(*a, **kw):  # must NOT be called
            raise AssertionError("network hit for an invalid name")

        monkeypatch.setattr(
            "src.conversational.health_evidence.tools.requests.get", fake_get,
        )
        assert he_tools.clinical_formula_lookup("a")["status"] == "unavailable"
        assert he_tools.clinical_formula_lookup("bad; drop table")["status"] == "unavailable"

    def test_registered_in_dispatch_and_defs(self):
        from src.conversational.health_evidence.tool_defs import (
            ALL_TOOL_DEFS, TOOL_DISPATCH,
        )
        assert TOOL_DISPATCH["clinical_formula_lookup"] is he_tools.clinical_formula_lookup
        names = {d["name"] for d in ALL_TOOL_DEFS}
        assert "clinical_formula_lookup" in names
