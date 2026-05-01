"""Tests for the Phase G MCP-backed source-of-truth tools.

Five tools: snomed_search, rxnorm_lookup, trials_search,
openfda_drug_label, icd_lookup. Each follows the same shape:
- Returns ``{"status": "unavailable", ...}`` when its config env var
  / required binary is missing.
- Calls ``McpClient.call_tool`` once and normalises the response into a
  per-tool envelope shape.
- Passes through ``unavailable`` envelopes from the MCP layer.
- Never raises.

We mock ``_get_mcp_client`` to inject a fake McpClient — the real client
itself is exercised by ``test_mcp_client.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.conversational.health_evidence import tools as he_tools
from src.conversational.health_evidence.tools import (
    icd_lookup,
    openfda_drug_label,
    rxnorm_lookup,
    snomed_search,
    trials_search,
)


@pytest.fixture(autouse=True)
def _clear_mcp_cache(monkeypatch):
    """Reset the lazy MCP-client cache between tests."""
    monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {})


def _fake_client(envelope) -> MagicMock:
    client = MagicMock()
    client.call_tool.return_value = envelope
    return client


# ===========================================================================
# snomed_search
# ===========================================================================


class TestSnomedSearch:
    def test_unavailable_when_hermes_not_on_path(self, monkeypatch):
        """Default config (HERMES_MCP_COMMAND='hermes') falls through to
        ``shutil.which`` returning None on most systems."""
        monkeypatch.delenv("HERMES_MCP_COMMAND", raising=False)
        # Force shutil.which to return None.
        monkeypatch.setattr("shutil.which", lambda _: None)
        result = snomed_search("sepsis")
        assert result["status"] == "unavailable"
        assert "Hermes" in result["error"]

    def test_happy_path_normalizes_results(self, monkeypatch):
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "concept_id": "91302008",
                    "preferred_term": "Sepsis (disorder)",
                    "fsn": "Sepsis (disorder)",
                    "semantic_tag": "disorder",
                },
                {
                    "id": "238170002",
                    "name": "Septicaemia",
                    "fully_specified_name": "Septicaemia (disorder)",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = snomed_search("sepsis", max_results=10)
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        # Field name normalisation
        assert result["results"][0]["concept_id"] == "91302008"
        assert result["results"][1]["concept_id"] == "238170002"
        # Both shapes (concept_id vs id, fsn vs fully_specified_name) accepted
        assert "Sepsis" in result["results"][0]["preferred_term"]

    def test_passes_through_unavailable_envelope(self, monkeypatch):
        client = _fake_client({"status": "unavailable", "error": "down"})
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = snomed_search("x")
        assert result["status"] == "unavailable"
        assert "down" in result["error"]

    def test_drops_results_without_concept_id(self, monkeypatch):
        client = _fake_client({
            "status": "ok",
            "results": [
                {"concept_id": "1", "preferred_term": "valid"},
                {"preferred_term": "no id — drop me"},
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = snomed_search("x")
        assert len(result["results"]) == 1


# ===========================================================================
# rxnorm_lookup
# ===========================================================================


class TestRxnormLookup:
    def test_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("OMOPHUB_MCP_URL", raising=False)
        result = rxnorm_lookup("metformin")
        assert result["status"] == "unavailable"
        assert "OMOPHUB_MCP_URL" in result["error"]

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("OMOPHUB_MCP_URL", "https://omophub.example/mcp")
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "rxcui": "6809",
                    "name": "metformin",
                    "tty": "IN",
                    "vocabulary": "RxNorm",
                },
                {
                    "concept_code": "11929",
                    "concept_name": "Metformin Hydrochloride",
                    "term_type": "PIN",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = rxnorm_lookup("metformin")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        assert result["results"][0]["rxcui"] == "6809"
        assert result["results"][1]["rxcui"] == "11929"
        assert result["results"][1]["name"] == "Metformin Hydrochloride"
        assert result["results"][1]["tty"] == "PIN"


# ===========================================================================
# trials_search
# ===========================================================================


class TestTrialsSearch:
    def test_unavailable_when_no_launcher_on_path(self, monkeypatch):
        monkeypatch.delenv("CLINICALTRIALS_MCP_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)
        result = trials_search("metformin sepsis")
        assert result["status"] == "unavailable"

    def test_happy_path_normalizes_results(self, monkeypatch):
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "nct_id": "NCT12345678",
                    "brief_title": "Metformin in sepsis",
                    "status": "Recruiting",
                    "conditions": ["Sepsis", "Diabetes"],
                    "phase": "Phase 2",
                },
                {
                    "nctId": "NCT87654321",
                    "briefTitle": "Insulin trial",
                    "overall_status": "Completed",
                    "condition": "Type 2 Diabetes",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = trials_search("metformin sepsis")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        # snake_case normalisation
        assert result["results"][0]["nct_id"] == "NCT12345678"
        assert result["results"][1]["nct_id"] == "NCT87654321"
        assert result["results"][1]["brief_title"] == "Insulin trial"
        assert result["results"][1]["status"] == "Completed"
        # Single-string condition coerced to list
        assert result["results"][1]["conditions"] == ["Type 2 Diabetes"]


# ===========================================================================
# openfda_drug_label
# ===========================================================================


class TestOpenfdaDrugLabel:
    def test_unavailable_when_no_npx(self, monkeypatch):
        monkeypatch.delenv("OPENFDA_MCP_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)
        result = openfda_drug_label("metformin")
        assert result["status"] == "unavailable"

    def test_happy_path_normalizes_listy_fields(self, monkeypatch):
        # OpenFDA labels nest most fields as lists.
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "brand_name": ["GLUCOPHAGE"],
                    "generic_name": ["METFORMIN HYDROCHLORIDE"],
                    "indications_and_usage": ["Type 2 diabetes adjunct."],
                    "warnings": ["Risk of lactic acidosis."],
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = openfda_drug_label("metformin")
        assert result["status"] == "ok"
        assert result["results"][0]["brand_name"] == "GLUCOPHAGE"
        assert result["results"][0]["generic_name"] == "METFORMIN HYDROCHLORIDE"
        assert "Type 2 diabetes" in result["results"][0]["indications_and_usage"]
        assert "lactic acidosis" in result["results"][0]["warnings"]

    def test_drops_records_without_any_name(self, monkeypatch):
        client = _fake_client({
            "status": "ok",
            "results": [{"warnings": ["just warnings, no names"]}],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = openfda_drug_label("x")
        assert result["status"] == "ok"
        assert result["results"] == []


# ===========================================================================
# icd_lookup
# ===========================================================================


class TestIcdLookup:
    def test_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        result = icd_lookup("sepsis")
        assert result["status"] == "unavailable"
        assert "ICD_MCP_URL" in result["error"]

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("ICD_MCP_URL", "https://icd.example/mcp")
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "code": "A41.9",
                    "title": "Sepsis, unspecified organism",
                    "version": "10",
                    "chapter": "Certain infectious and parasitic diseases",
                },
                {
                    "icd_code": "1G40",
                    "description": "Sepsis without septic shock",
                    "version": "11",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_lookup("sepsis", version="10")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        assert result["results"][0]["code"] == "A41.9"
        assert result["results"][1]["code"] == "1G40"
        assert "Sepsis" in result["results"][1]["title"]


# ===========================================================================
# Tool defs + dispatch sanity
# ===========================================================================


class TestToolRegistration:
    def test_all_8_tools_in_dispatch(self):
        from src.conversational.health_evidence import TOOL_DISPATCH

        for name in [
            "pubmed_search",
            "mimic_distribution_lookup",
            "loinc_reference_range",
            "snomed_search",
            "rxnorm_lookup",
            "trials_search",
            "openfda_drug_label",
            "icd_lookup",
        ]:
            assert name in TOOL_DISPATCH, f"Missing dispatch for {name}"

    def test_all_8_tool_defs_listed(self):
        from src.conversational.health_evidence import ALL_TOOL_DEFS

        names = {td["name"] for td in ALL_TOOL_DEFS}
        assert names == {
            "pubmed_search", "mimic_distribution_lookup",
            "loinc_reference_range", "snomed_search", "rxnorm_lookup",
            "trials_search", "openfda_drug_label", "icd_lookup",
        }

    def test_each_def_has_required_fields(self):
        from src.conversational.health_evidence import ALL_TOOL_DEFS

        for td in ALL_TOOL_DEFS:
            assert "name" in td
            assert "description" in td
            assert "input_schema" in td
            schema = td["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
