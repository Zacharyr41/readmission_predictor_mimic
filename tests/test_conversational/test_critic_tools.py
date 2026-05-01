"""Tests for the critic's external evidence tools.

The critic gets a small set of HTTP-API-wrapper functions it can call
when its baked-in reference ranges aren't enough. v1 ships PubMed
search via NCBI E-utilities. These tests mock ``requests.get`` to keep
the suite offline + fast.

The contract is symmetric: every tool function returns either
``{"status": "ok", "results": [...]}`` or
``{"status": "unavailable", "error": "..."}`` — never raises. This
keeps the critic loop's failure handling simple and predictable.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.conversational.critic_tools import (
    PUBMED_SEARCH_TOOL_DEF,
    TOOL_DISPATCH,
    pubmed_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(json_payload: dict | None = None, *, raise_on_call: Exception | None = None) -> MagicMock:
    """Build a fake ``requests.Response`` returning *json_payload* on .json()."""
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
    """``records`` items: {pmid, title, source, pubdate, abstract?}.
    NCBI esummary returns a dict keyed by PMID with a `uids` list."""
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


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_pubmed_search_tool_def_shape(self):
        assert PUBMED_SEARCH_TOOL_DEF["name"] == "pubmed_search"
        assert "input_schema" in PUBMED_SEARCH_TOOL_DEF
        props = PUBMED_SEARCH_TOOL_DEF["input_schema"]["properties"]
        assert "query" in props
        assert "max_results" in props
        assert "query" in PUBMED_SEARCH_TOOL_DEF["input_schema"]["required"]

    def test_dispatch_registry_includes_pubmed_search(self):
        assert "pubmed_search" in TOOL_DISPATCH
        assert TOOL_DISPATCH["pubmed_search"] is pubmed_search


# ---------------------------------------------------------------------------
# pubmed_search behavior
# ---------------------------------------------------------------------------


class TestPubmedSearchHappyPath:
    def test_happy_path_returns_records(self, monkeypatch):
        """esearch returns 2 PMIDs; esummary returns metadata for both;
        pubmed_search returns a list of {pmid, title, abstract, url} dicts."""
        responses = iter([
            _make_response(_esearch_payload(["12345", "67890"])),
            _make_response(_esummary_payload([
                {"pmid": "12345", "title": "First study"},
                {"pmid": "67890", "title": "Second study"},
            ])),
        ])
        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get",
            lambda *a, **kw: next(responses),
        )

        result = pubmed_search("lactate sepsis")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        pmids = [r["pmid"] for r in result["results"]]
        assert pmids == ["12345", "67890"]
        # URL canonical
        assert result["results"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345/"
        # Required keys
        for r in result["results"]:
            assert "pmid" in r
            assert "title" in r
            assert "url" in r

    def test_passes_query_and_retmax_to_esearch(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured["url"] = url
            captured["params"] = params or {}
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        pubmed_search("creatinine kidney injury", max_results=3)
        assert "esearch" in captured["url"]
        assert captured["params"].get("term") == "creatinine kidney injury"
        assert int(captured["params"].get("retmax")) == 3
        assert captured["params"].get("db") == "pubmed"


class TestPubmedSearchEdgeCases:
    def test_empty_results_skips_esummary(self, monkeypatch):
        """No PMIDs from esearch → no esummary call, returns ok with []."""
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            if "esearch" in url:
                return _make_response(_esearch_payload([]))
            return _make_response(_esummary_payload([]))

        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        result = pubmed_search("nonexistent analyte")
        assert result == {"status": "ok", "results": []}
        # Only esearch should have been called.
        assert len(calls) == 1
        assert "esearch" in calls[0]

    def test_network_error_returns_unavailable(self, monkeypatch):
        def fake_get(*a, **kw):
            raise requests.RequestException("connection refused")

        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        result = pubmed_search("anything")
        assert result["status"] == "unavailable"
        assert "error" in result
        assert "connection refused" in result["error"]

    def test_malformed_esummary_returns_unavailable(self, monkeypatch):
        responses = iter([
            _make_response(_esearch_payload(["12345"])),
            _make_response({"unexpected": "shape"}),  # missing 'result' key
        ])
        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get",
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
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        result = pubmed_search("anything")
        assert result["status"] == "unavailable"


class TestPubmedSearchSizeAndKeys:
    def test_truncates_serialized_result_to_4kb(self, monkeypatch):
        """Even if individual abstracts are huge, the JSON-serialized result
        must fit within the 4KB tool-result budget. Per-record abstract is
        truncated with a sentinel suffix."""
        # 50 KB title, well over budget.
        big_title = "X" * 50_000
        responses = iter([
            _make_response(_esearch_payload(["1"])),
            _make_response(_esummary_payload([
                {"pmid": "1", "title": big_title},
            ])),
        ])
        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get",
            lambda *a, **kw: next(responses),
        )
        result = pubmed_search("x")
        assert result["status"] == "ok"
        # Serialized form must be ≤ 4096 bytes.
        size = len(json.dumps(result).encode("utf-8"))
        assert size <= 4096, f"serialized size {size} exceeds 4KB"

    def test_max_results_capped_at_five(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured["params"] = params or {}
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        pubmed_search("x", max_results=99)
        # Caller asks for 99 but the wrapper caps at 5 (per tool def).
        assert int(captured["params"]["retmax"]) <= 5

    def test_max_results_default_is_five(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured["params"] = params or {}
            return _make_response(_esearch_payload([]))

        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        pubmed_search("x")
        assert int(captured["params"]["retmax"]) == 5


class TestPubmedSearchAuth:
    def test_api_key_param_set_when_env_present(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured.setdefault("params_seen", []).append(params or {})
            return _make_response(_esearch_payload([]))

        monkeypatch.setenv("NCBI_API_KEY", "test-key-abc")
        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        pubmed_search("x")
        # Both esearch (and esummary if reached) carry api_key param.
        # esearch is the call we know fired.
        assert captured["params_seen"][0].get("api_key") == "test-key-abc"

    def test_api_key_absent_when_env_unset(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, timeout=None, **kw):
            captured.setdefault("params_seen", []).append(params or {})
            return _make_response(_esearch_payload([]))

        monkeypatch.delenv("NCBI_API_KEY", raising=False)
        monkeypatch.setattr(
            "src.conversational.critic_tools.requests.get", fake_get,
        )
        pubmed_search("x")
        assert "api_key" not in captured["params_seen"][0]
