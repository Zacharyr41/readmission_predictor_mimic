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


class TestOmophubConfig:
    """Phase H follow-up: _omophub_config defaults to the hosted MCP
    endpoint (https://mcp.omophub.com) and constructs an Authorization:
    Bearer header from OMOPHUB_API_KEY. The legacy OMOPHUB_MCP_URL env
    var is preserved as an override path for users with a self-hosted
    Docker container."""

    def test_uses_hosted_default_when_url_unset(self, monkeypatch):
        monkeypatch.delenv("OMOPHUB_MCP_URL", raising=False)
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_test123")
        cfg = he_tools._omophub_config()
        assert cfg is not None
        assert cfg.url == "https://mcp.omophub.com"
        assert cfg.headers == {"Authorization": "Bearer oh_test123"}
        assert cfg.transport == "http"

    def test_respects_custom_url_for_self_hosted(self, monkeypatch):
        """Self-hosted Docker users override via OMOPHUB_MCP_URL."""
        monkeypatch.setenv("OMOPHUB_MCP_URL", "http://localhost:3100/mcp")
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_local_test")
        cfg = he_tools._omophub_config()
        assert cfg.url == "http://localhost:3100/mcp"
        # Bearer header still attached regardless of URL — works for
        # both hosted and self-hosted (the self-hosted container will
        # ignore client headers since it auths via env, but no harm).
        assert cfg.headers == {"Authorization": "Bearer oh_local_test"}

    def test_returns_none_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("OMOPHUB_API_KEY", raising=False)
        cfg = he_tools._omophub_config()
        assert cfg is None


class TestRxnormLookup:
    def test_unavailable_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("OMOPHUB_API_KEY", raising=False)
        monkeypatch.delenv("OMOPHUB_MCP_URL", raising=False)
        result = rxnorm_lookup("metformin")
        assert result["status"] == "unavailable"
        # Error message should point at the missing key.
        assert "OMOPHUB_API_KEY" in result["error"]

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

    def test_calls_omophub_search_concepts_with_vocabulary_ids(
        self, monkeypatch,
    ):
        """OMOPHub's actual MCP tool name is ``search_concepts``, not
        ``search`` (which doesn't exist on OMOPHub). The param key for
        the vocabulary filter is ``vocabulary_ids`` (plural, multi-vocab
        comma-separated). Phase H follow-up fix."""
        captured = []

        def fake_call_tool(name, payload, **kw):
            captured.append((name, payload))
            return {"status": "ok", "results": []}

        client = MagicMock()
        client.call_tool.side_effect = fake_call_tool
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        rxnorm_lookup("Levophed", max_results=3)
        assert len(captured) == 1
        tool_name, payload = captured[0]
        assert tool_name == "search_concepts", (
            f"OMOPHub MCP exposes 'search_concepts', not {tool_name!r}"
        )
        assert payload.get("vocabulary_ids") == "RxNorm"
        assert "vocabulary" not in payload, (
            "old single-key 'vocabulary' param is rejected by OMOPHub"
        )

    def test_normalises_omophub_response_shape(self, monkeypatch):
        """OMOPHub returns concept records with ``concept_code``,
        ``concept_name``, ``concept_class_id``, ``vocabulary_id``.
        Our wrapper remaps to our envelope's stable field names
        (``rxcui``, ``name``, ``tty``, ``vocabulary``)."""
        client = _fake_client({
            "status": "ok",
            "results": [{
                "concept_code": "7980",
                "concept_name": "norepinephrine",
                "concept_class_id": "Ingredient",
                "vocabulary_id": "RxNorm",
            }],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        r = rxnorm_lookup("Levophed")
        assert r["status"] == "ok"
        rec = r["results"][0]
        assert rec["rxcui"] == "7980"
        assert rec["name"] == "norepinephrine"
        assert rec["tty"] == "Ingredient"
        assert rec["vocabulary"] == "RxNorm"


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
    def test_unavailable_when_neither_icd_nor_omophub_configured(
        self, monkeypatch,
    ):
        """Phase H follow-up: ICD lookup now has a fallback chain —
        ICD_MCP_URL first (legacy self-hosted), then OMOPHUB_API_KEY
        (route through OMOPHub's search_concepts with
        vocabulary_ids=ICD10CM). Unavailable only when BOTH are unset."""
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        monkeypatch.delenv("OMOPHUB_API_KEY", raising=False)
        result = icd_lookup("sepsis")
        assert result["status"] == "unavailable"

    def test_legacy_path_when_icd_mcp_url_set(self, monkeypatch):
        """When ICD_MCP_URL is set, calls the legacy `lookup` tool name
        with the legacy payload shape — back-compat for users with a
        self-hosted ICD MCP."""
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
            ],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_lookup("sepsis", version="10")
        # Legacy `lookup` tool name + payload shape.
        call = client.call_tool.call_args
        assert call.args[0] == "lookup"
        assert call.args[1].get("query") == "sepsis"
        assert result["status"] == "ok"
        assert result["results"][0]["code"] == "A41.9"

    def test_omophub_path_when_only_omophub_configured(
        self, monkeypatch,
    ):
        """When ICD_MCP_URL is unset but OMOPHUB_API_KEY is set, route
        through OMOPHub's search_concepts with vocabulary_ids=ICD10CM."""
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_test")
        captured = []

        def fake(name, payload, **kw):
            captured.append((name, payload))
            return {
                "status": "ok",
                "results": [{
                    "concept_code": "A419",
                    "concept_name": "Sepsis, unspecified organism",
                    "vocabulary_id": "ICD10CM",
                    "domain_id": "Condition",
                }],
            }

        client = MagicMock()
        client.call_tool.side_effect = fake
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_lookup("sepsis", version="10")
        assert captured[0][0] == "search_concepts"
        assert captured[0][1].get("vocabulary_ids") == "ICD10CM"
        assert result["status"] == "ok"
        # Result normalized to the existing envelope shape (code/title/version).
        assert result["results"][0]["code"] == "A419"
        assert "Sepsis" in result["results"][0]["title"]
        assert result["results"][0]["version"] == "10"

    def test_omophub_path_rejects_icd_11(self, monkeypatch):
        """OMOPHub doesn't carry ICD-11 vocabulary. version='11' on the
        OMOPHub path returns unavailable with a clear note."""
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_test")
        # Even with a client mocked, the version check should short-circuit.
        monkeypatch.setattr(
            he_tools, "_get_mcp_client",
            lambda *a, **kw: MagicMock(),
        )
        result = icd_lookup("anything", version="11")
        assert result["status"] == "unavailable"
        # Error mentions ICD-11 + the OMOPHub limitation
        assert "11" in result["error"]
        assert "icd" in result["error"].lower()

    def test_legacy_path_supports_icd_11(self, monkeypatch):
        """When using a self-hosted ICD MCP via ICD_MCP_URL, ICD-11
        IS supported (the user's self-host knows ICD-11). Only the
        OMOPHub path is restricted."""
        monkeypatch.setenv("ICD_MCP_URL", "https://icd.example/mcp")
        client = _fake_client({
            "status": "ok",
            "results": [{
                "code": "1G40", "title": "Sepsis without septic shock",
                "version": "11",
            }],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_lookup("sepsis", version="11")
        assert result["status"] == "ok"
        assert result["results"][0]["code"] == "1G40"

    def test_icd_mcp_url_takes_precedence_over_omophub(self, monkeypatch):
        """When both ICD_MCP_URL AND OMOPHUB_API_KEY are set, the
        legacy ICD path wins (so users with their own ICD MCP keep
        getting their existing behaviour)."""
        monkeypatch.setenv("ICD_MCP_URL", "https://icd.example/mcp")
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_test")
        client = _fake_client({
            "status": "ok",
            "results": [{"code": "A41.9", "title": "Sepsis"}],
        })
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        icd_lookup("sepsis")
        call = client.call_tool.call_args
        # Legacy tool name (NOT search_concepts).
        assert call.args[0] == "lookup"


# ===========================================================================
# Tool defs + dispatch sanity
# ===========================================================================


class TestToolRegistration:
    def test_all_tools_in_dispatch(self):
        from src.conversational.health_evidence import TOOL_DISPATCH

        for name in [
            "pubmed_search",
            "mimic_distribution_lookup",
            "mimic_itemid_search",
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

    def test_all_tool_defs_listed(self):
        from src.conversational.health_evidence import ALL_TOOL_DEFS

        names = {td["name"] for td in ALL_TOOL_DEFS}
        assert names == {
            "pubmed_search", "mimic_distribution_lookup",
            "mimic_itemid_search",
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
    def test_unavailable_when_api_key_missing(self, monkeypatch):
        """Phase H follow-up: OMOPHub now defaults to the hosted
        endpoint, so URL-missing isn't the error condition anymore —
        the missing API key is."""
        monkeypatch.delenv("OMOPHUB_API_KEY", raising=False)
        monkeypatch.delenv("OMOPHUB_MCP_URL", raising=False)
        result = code_map(
            source_vocabulary="ICD10CM", source_code="E11.9",
            target_vocabulary="SNOMED",
        )
        assert result["status"] == "unavailable"
        assert "OMOPHUB_API_KEY" in result["error"]

    def test_uses_2step_get_concept_by_code_then_explore_concept(
        self, monkeypatch,
    ):
        """OMOPHub's actual MCP exposes ``get_concept_by_code`` (code →
        OMOP concept_id) and ``explore_concept`` (concept_id → full
        relationships including ``concept_2`` with vocabulary-native
        target codes). Our wrapper does the 2-step pivot.

        We use ``explore_concept`` rather than ``map_concept`` because
        the latter returns OMOP concept_ids without vocab-native
        target codes, which would require a 3rd call per mapping."""
        import json as _json
        calls = []

        # Explore-concept's response wraps mappings in a text blob with
        # trailing JSON of shape {"concept": ..., "relationships":
        # {"relationships": [...]}}.
        explore_payload = {
            "concept": {"concept_id": 35206882},
            "relationships": {
                "relationships": [
                    {
                        "relationship_id": "Maps to",
                        "concept_2": {
                            "concept_id": 201826,
                            "concept_name": "Type 2 diabetes mellitus",
                            "concept_code": "44054006",
                            "vocabulary_id": "SNOMED",
                        },
                    },
                    # Decoy — different vocab, must be filtered out.
                    {
                        "relationship_id": "Subsumes",
                        "concept_2": {
                            "concept_code": "E11",
                            "vocabulary_id": "ICD10CM",
                        },
                    },
                ],
            },
        }
        get_by_code_payload = [{"concept_id": 35206882}]

        def fake(name, payload, **kw):
            calls.append((name, payload))
            if name == "get_concept_by_code":
                return {
                    "status": "ok",
                    "results": [{
                        "text": "some markdown content...\n" + _json.dumps(get_by_code_payload),
                    }],
                }
            if name == "explore_concept":
                return {
                    "status": "ok",
                    "results": [{
                        "text": "some markdown content...\n" + _json.dumps(explore_payload),
                    }],
                }
            return {"status": "unavailable", "error": f"unknown tool {name}"}

        client = MagicMock()
        client.call_tool.side_effect = fake
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = code_map(
            source_vocabulary="ICD10CM", source_code="E11.9",
            target_vocabulary="SNOMED",
        )
        # Two tool calls happened in the right order.
        assert calls[0][0] == "get_concept_by_code"
        assert calls[0][1].get("concept_code") == "E11.9"
        assert calls[0][1].get("vocabulary_id") == "ICD10CM"
        assert calls[1][0] == "explore_concept"
        assert calls[1][1].get("concept_id") == 35206882
        # Result filtered to SNOMED only (the ICD10CM decoy got dropped).
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        assert result["results"][0]["target_code"] == "44054006"
        assert result["results"][0]["target_vocabulary"] == "SNOMED"
        assert result["results"][0]["target_name"] == "Type 2 diabetes mellitus"
        assert result["results"][0]["source_code"] == "E11.9"
        assert result["results"][0]["source_vocabulary"] == "ICD10CM"
        assert result["results"][0]["relationship"] == "Maps to"

    def test_step1_miss_returns_unavailable_without_step2(
        self, monkeypatch,
    ):
        """If get_concept_by_code finds no concept for the source
        code, return unavailable cleanly — don't waste step 2."""
        import json as _json
        calls = []

        def fake(name, payload, **kw):
            calls.append(name)
            if name == "get_concept_by_code":
                # Empty-array text blob means "not found"
                return {"status": "ok", "results": [{"text": "[markdown][]"}]}
            return {"status": "ok", "results": [{"text": "[markdown][]"}]}

        client = MagicMock()
        client.call_tool.side_effect = fake
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        r = code_map(
            source_vocabulary="ICD10", source_code="ZZZZ",
            target_vocabulary="SNOMED",
        )
        assert r["status"] == "unavailable"
        # Step 2 NEVER fired
        assert "explore_concept" not in calls

    def test_step1_unavailable_propagates(self, monkeypatch):
        """If get_concept_by_code itself returns unavailable (e.g.
        OMOPHub hiccup), pass that through. Step 2 doesn't fire."""
        calls = []

        def fake(name, payload, **kw):
            calls.append(name)
            if name == "get_concept_by_code":
                return {"status": "unavailable", "error": "rate limited"}
            return {"status": "ok", "results": []}

        client = MagicMock()
        client.call_tool.side_effect = fake
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        r = code_map(
            source_vocabulary="ICD10", source_code="E11.9",
            target_vocabulary="SNOMED",
        )
        assert r["status"] == "unavailable"
        assert "rate limited" in r["error"]
        assert "explore_concept" not in calls

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

    def test_drops_step2_results_without_target_code(self, monkeypatch):
        """explore_concept entries whose ``concept_2`` lacks a
        concept_code are filtered out."""
        import json as _json

        def fake(name, payload, **kw):
            if name == "get_concept_by_code":
                return {
                    "status": "ok",
                    "results": [{
                        "text": "markdown blob..." + _json.dumps([{"concept_id": 999}]),
                    }],
                }
            if name == "explore_concept":
                payload = {
                    "concept": {"concept_id": 999},
                    "relationships": {
                        "relationships": [
                            {
                                "relationship_id": "Maps to",
                                "concept_2": {
                                    "concept_code": "y",
                                    "vocabulary_id": "SNOMED",
                                },
                            },
                            {
                                "relationship_id": "Maps to",
                                "concept_2": {
                                    "vocabulary_id": "SNOMED",
                                    # missing concept_code
                                },
                            },
                        ],
                    },
                }
                return {
                    "status": "ok",
                    "results": [{"text": "markdown blob..." + _json.dumps(payload)}],
                }
            return {"status": "unavailable"}

        client = MagicMock()
        client.call_tool.side_effect = fake
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = code_map(
            source_vocabulary="ICD10CM", source_code="x",
            target_vocabulary="SNOMED",
        )
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        assert result["results"][0]["target_code"] == "y"


# ===========================================================================
# icd_autocode (free-text → ICD code suggestions)
# ===========================================================================


class TestIcdAutocode:
    def test_unavailable_when_neither_icd_nor_omophub_configured(
        self, monkeypatch,
    ):
        """Same fallback chain as icd_lookup — unavailable only when
        BOTH ICD_MCP_URL and OMOPHUB_API_KEY are unset."""
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        monkeypatch.delenv("OMOPHUB_API_KEY", raising=False)
        result = icd_autocode("type 2 diabetes mellitus")
        assert result["status"] == "unavailable"

    def test_omophub_path_uses_semantic_search(self, monkeypatch):
        """When only OMOPHUB_API_KEY is set, route through OMOPHub's
        semantic_search filtered to vocabulary_ids=ICD10CM."""
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_test")
        captured = []

        def fake(name, payload, **kw):
            captured.append((name, payload))
            return {
                "status": "ok",
                "results": [{
                    "concept_code": "J96.00",
                    "concept_name": "Acute respiratory failure, unspecified",
                    "vocabulary_id": "ICD10CM",
                    "score": 0.91,
                }],
            }

        client = MagicMock()
        client.call_tool.side_effect = fake
        monkeypatch.setattr(
            he_tools, "_get_mcp_client", lambda *a, **kw: client,
        )
        result = icd_autocode("acute respiratory failure")
        assert captured[0][0] == "semantic_search"
        assert captured[0][1].get("vocabulary_ids") == "ICD10CM"
        assert result["status"] == "ok"
        rec = result["results"][0]
        assert rec["code"] == "J96.00"
        assert "respiratory failure" in rec["title"].lower()
        assert rec["confidence"] == pytest.approx(0.91)
        assert rec["version"] == "10"

    def test_omophub_path_rejects_icd_11(self, monkeypatch):
        monkeypatch.delenv("ICD_MCP_URL", raising=False)
        monkeypatch.setenv("OMOPHUB_API_KEY", "oh_test")
        monkeypatch.setattr(
            he_tools, "_get_mcp_client",
            lambda *a, **kw: MagicMock(),
        )
        result = icd_autocode("anything", version="11")
        assert result["status"] == "unavailable"
        assert "11" in result["error"]
        assert "icd" in result["error"].lower()

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
