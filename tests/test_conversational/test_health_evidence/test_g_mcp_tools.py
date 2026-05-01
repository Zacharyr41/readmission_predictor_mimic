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
    code_map,
    icd_autocode,
    icd_lookup,
    openfda_drug_label,
    rxnorm_lookup,
    snomed_expand_ecl,
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
    def test_all_11_tools_in_dispatch(self):
        from src.conversational.health_evidence import TOOL_DISPATCH

        for name in [
            "pubmed_search",
            "mimic_distribution_lookup",
            "loinc_reference_range",
            "snomed_search",
            "snomed_expand_ecl",
            "rxnorm_lookup",
            "code_map",
            "trials_search",
            "openfda_drug_label",
            "icd_lookup",
            "icd_autocode",
        ]:
            assert name in TOOL_DISPATCH, f"Missing dispatch for {name}"

    def test_all_11_tool_defs_listed(self):
        from src.conversational.health_evidence import ALL_TOOL_DEFS

        names = {td["name"] for td in ALL_TOOL_DEFS}
        assert names == {
            "pubmed_search", "mimic_distribution_lookup",
            "loinc_reference_range", "snomed_search", "snomed_expand_ecl",
            "rxnorm_lookup", "code_map", "trials_search",
            "openfda_drug_label", "icd_lookup", "icd_autocode",
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


# ===========================================================================
# snomed_expand_ecl (Hermes ECL — Expression Constraint Language)
# ===========================================================================


class TestSnomedExpandEcl:
    def test_unavailable_when_hermes_not_on_path(self, monkeypatch):
        monkeypatch.delenv("HERMES_MCP_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)
        result = snomed_expand_ecl("<<73211009")
        assert result["status"] == "unavailable"
        assert "Hermes" in result["error"]

    def test_happy_path_normalizes_ecl_results(self, monkeypatch):
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "concept_id": "73211009",
                    "preferred_term": "Diabetes mellitus",
                    "fully_specified_name": "Diabetes mellitus (disorder)",
                },
                {
                    "id": "44054006",
                    "name": "Type 2 diabetes mellitus",
                    "fsn": "Diabetes mellitus type 2 (disorder)",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = snomed_expand_ecl("<<73211009 |Diabetes mellitus|")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        # Field-name normalisation accepts either shape
        assert result["results"][0]["concept_id"] == "73211009"
        assert result["results"][1]["concept_id"] == "44054006"
        assert "Type 2" in result["results"][1]["preferred_term"]

    def test_passes_ecl_expression_through_to_server(self, monkeypatch):
        client = _fake_client({"status": "ok", "results": []})
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        snomed_expand_ecl("<<195967001")
        # Server is called with the ECL expression in the args
        call = client.call_tool.call_args
        assert call.args[0] == "expand_ecl"
        assert "<<195967001" in str(call.args[1])

    def test_passes_through_unavailable_envelope(self, monkeypatch):
        client = _fake_client({
            "status": "unavailable", "error": "ECL parser failed",
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = snomed_expand_ecl("garbage ECL")
        assert result["status"] == "unavailable"
        assert "ECL parser" in result["error"]

    def test_drops_results_without_concept_id(self, monkeypatch):
        client = _fake_client({
            "status": "ok",
            "results": [
                {"concept_id": "1", "preferred_term": "valid"},
                {"preferred_term": "no id — drop"},
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = snomed_expand_ecl("<<x")
        assert len(result["results"]) == 1


# ===========================================================================
# code_map (OMOPHub cross-vocabulary mapping)
# ===========================================================================


class TestCodeMap:
    def test_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("OMOPHUB_MCP_URL", raising=False)
        result = code_map(
            source_vocabulary="ICD10CM", source_code="E11.9",
            target_vocabulary="SNOMED",
        )
        assert result["status"] == "unavailable"
        assert "OMOPHUB_MCP_URL" in result["error"]

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("OMOPHUB_MCP_URL", "https://omophub.example/mcp")
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "source_code": "E11.9",
                    "source_vocabulary": "ICD10CM",
                    "target_code": "44054006",
                    "target_vocabulary": "SNOMED",
                    "target_name": "Type 2 diabetes mellitus",
                    "relationship": "Maps to",
                },
                {
                    "source_concept_code": "E11.9",
                    "source_vocabulary_id": "ICD10CM",
                    "target_concept_code": "201826",
                    "target_vocabulary_id": "RxNorm",
                    "target_concept_name": "metformin",
                    "relationship_id": "Treats",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = code_map(
            source_vocabulary="ICD10CM", source_code="E11.9",
            target_vocabulary="SNOMED",
        )
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        # Both upstream field-name shapes accepted
        assert result["results"][0]["target_code"] == "44054006"
        assert result["results"][0]["target_vocabulary"] == "SNOMED"
        assert result["results"][1]["target_code"] == "201826"
        assert result["results"][1]["target_vocabulary"] == "RxNorm"
        assert result["results"][1]["relationship"] == "Treats"

    def test_passes_args_to_server(self, monkeypatch):
        monkeypatch.setenv("OMOPHUB_MCP_URL", "https://x")
        client = _fake_client({"status": "ok", "results": []})
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        code_map(
            source_vocabulary="ICD10CM", source_code="E11.9",
            target_vocabulary="SNOMED",
        )
        call = client.call_tool.call_args
        assert call.args[0] == "map_code"
        payload = call.args[1]
        assert payload.get("source_vocabulary") == "ICD10CM"
        assert payload.get("source_code") == "E11.9"
        assert payload.get("target_vocabulary") == "SNOMED"

    def test_passes_through_unavailable_envelope(self, monkeypatch):
        monkeypatch.setenv("OMOPHUB_MCP_URL", "https://x")
        client = _fake_client({
            "status": "unavailable", "error": "rate limited",
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = code_map(
            source_vocabulary="ICD10CM", source_code="x",
            target_vocabulary="SNOMED",
        )
        assert result["status"] == "unavailable"

    def test_drops_results_without_target_code(self, monkeypatch):
        monkeypatch.setenv("OMOPHUB_MCP_URL", "https://x")
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "source_code": "x", "target_code": "y",
                    "target_vocabulary": "SNOMED",
                },
                {  # missing target_code
                    "source_code": "x", "target_vocabulary": "SNOMED",
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = code_map(
            source_vocabulary="ICD10CM", source_code="x",
            target_vocabulary="SNOMED",
        )
        assert len(result["results"]) == 1


# ===========================================================================
# icd_autocode (free-text → ICD code suggestions)
# ===========================================================================


class TestIcdAutocode:
    def test_unavailable_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        result = icd_autocode("type 2 diabetes mellitus")
        assert result["status"] == "unavailable"
        assert "ICD_MCP_URL" in result["error"]

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("ICD_MCP_URL", "https://icd.example/mcp")
        client = _fake_client({
            "status": "ok",
            "results": [
                {
                    "code": "E11.9",
                    "title": "Type 2 diabetes mellitus without complications",
                    "version": "10",
                    "confidence": 0.92,
                },
                {
                    "icd_code": "5A11",
                    "description": "Type 2 diabetes mellitus",
                    "version": "11",
                    "score": 0.88,
                },
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_autocode("type 2 diabetes mellitus", version="10")
        assert result["status"] == "ok"
        assert len(result["results"]) == 2
        # Both upstream field-name shapes accepted; confidence preserved.
        first = result["results"][0]
        second = result["results"][1]
        assert first["code"] == "E11.9"
        assert first["confidence"] == pytest.approx(0.92)
        assert second["code"] == "5A11"
        # 'score' alias also accepted as confidence.
        assert second["confidence"] == pytest.approx(0.88)
        assert "Type 2" in second["title"]

    def test_passes_text_and_version_to_server(self, monkeypatch):
        monkeypatch.setenv("ICD_MCP_URL", "https://x")
        client = _fake_client({"status": "ok", "results": []})
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        icd_autocode("severe sepsis", version="11")
        call = client.call_tool.call_args
        assert call.args[0] == "autocode"
        payload = call.args[1]
        assert payload.get("text") == "severe sepsis"
        assert payload.get("version") == "11"

    def test_passes_through_unavailable_envelope(self, monkeypatch):
        monkeypatch.setenv("ICD_MCP_URL", "https://x")
        client = _fake_client({
            "status": "unavailable", "error": "WHO API timeout",
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_autocode("anything")
        assert result["status"] == "unavailable"

    def test_drops_records_without_code(self, monkeypatch):
        monkeypatch.setenv("ICD_MCP_URL", "https://x")
        client = _fake_client({
            "status": "ok",
            "results": [
                {"code": "X1", "title": "valid"},
                {"title": "no code — drop"},
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_autocode("x")
        assert len(result["results"]) == 1

    def test_missing_confidence_omitted_or_none(self, monkeypatch):
        """If the upstream doesn't return a confidence/score, the
        normalised record should omit it (or default to None) — not
        synthesise a fake number."""
        monkeypatch.setenv("ICD_MCP_URL", "https://x")
        client = _fake_client({
            "status": "ok",
            "results": [{"code": "A1", "title": "no confidence here"}],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_autocode("x")
        rec = result["results"][0]
        assert rec.get("confidence") is None
