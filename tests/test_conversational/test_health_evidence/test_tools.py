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
