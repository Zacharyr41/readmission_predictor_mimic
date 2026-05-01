"""Tests for the 3-backend pubmed_search dispatch (Phase F2).

Backends:
- ``"direct"`` (default) — NCBI E-utilities; tested in test_tools.py.
- ``"mcp_anthropic"`` — Anthropic-hosted PubMed MCP via HTTP.
- ``"mcp_self_host"`` — User-supplied MCP via HTTP.

We mock the McpClient layer so these tests don't require network or
subprocess. The end-to-end MCP path itself is exercised in
test_mcp_client.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.conversational.health_evidence import tools as he_tools
from src.conversational.health_evidence.tools import pubmed_search


@pytest.fixture(autouse=True)
def _clear_pubmed_mcp_cache(monkeypatch):
    """Reset the lazy MCP-client cache between tests so each test sees
    a fresh client and our patches take effect."""
    monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {})


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_default_backend_is_direct(self, monkeypatch):
        monkeypatch.delenv("PUBMED_BACKEND", raising=False)
        # Patch _pubmed_direct so we can confirm it was called.
        called = []
        monkeypatch.setattr(he_tools, "_pubmed_direct",
                            lambda *a, **kw: called.append("direct") or {"status": "ok", "results": []})
        pubmed_search("anything")
        assert called == ["direct"]

    def test_explicit_direct_routes_to_direct(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "direct")
        called = []
        monkeypatch.setattr(he_tools, "_pubmed_direct",
                            lambda *a, **kw: called.append("direct") or {"status": "ok", "results": []})
        pubmed_search("x")
        assert called == ["direct"]

    def test_mcp_anthropic_routes_via_mcp(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_anthropic")
        called = []

        def fake_via_mcp(query, max_results, *, backend_name, url):
            called.append((backend_name, url))
            return {"status": "ok", "results": []}

        monkeypatch.setattr(he_tools, "_pubmed_via_mcp", fake_via_mcp)
        pubmed_search("x")
        assert len(called) == 1
        assert called[0][0] == "mcp_anthropic"
        assert "pubmed.mcp.claude.com" in called[0][1]

    def test_mcp_self_host_requires_url(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_self_host")
        monkeypatch.delenv("PUBMED_MCP_URL", raising=False)
        result = pubmed_search("x")
        assert result["status"] == "unavailable"
        assert "PUBMED_MCP_URL" in result["error"]

    def test_mcp_self_host_with_url_routes_correctly(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_self_host")
        monkeypatch.setenv("PUBMED_MCP_URL", "https://my.pubmed.local/mcp")
        called = []

        def fake_via_mcp(query, max_results, *, backend_name, url):
            called.append((backend_name, url))
            return {"status": "ok", "results": []}

        monkeypatch.setattr(he_tools, "_pubmed_via_mcp", fake_via_mcp)
        pubmed_search("x")
        assert called[0] == ("mcp_self_host", "https://my.pubmed.local/mcp")


# ---------------------------------------------------------------------------
# MCP backend behaviour (envelope normalisation, failure passthrough)
# ---------------------------------------------------------------------------


class TestMcpBackendBehaviour:
    def test_normalizes_mcp_results_to_pubmed_envelope(self, monkeypatch):
        """The MCP server may return slightly-different field names
        (id vs pmid, journal vs source). We normalise to the standard
        pubmed envelope shape."""
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_anthropic")

        fake_client = MagicMock()
        fake_client.call_tool.return_value = {
            "status": "ok",
            "results": [
                {
                    "id": "12345",
                    "title": "Sepsis lactate study",
                    "journal": "JAMA",
                    "date": "2024",
                },
            ],
        }
        # Pre-populate the cache with our fake.
        monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {
            "mcp_anthropic": fake_client,
        })

        result = pubmed_search("lactate sepsis", max_results=5)
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        rec = result["results"][0]
        # id → pmid
        assert rec["pmid"] == "12345"
        # journal → source
        assert rec["source"] == "JAMA"
        # date → pubdate
        assert rec["pubdate"] == "2024"
        # URL synthesised from PMID
        assert "12345" in rec["url"]

    def test_passthrough_pmid_field(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_anthropic")
        fake_client = MagicMock()
        fake_client.call_tool.return_value = {
            "status": "ok",
            "results": [{"pmid": "67890", "title": "X"}],
        }
        monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {
            "mcp_anthropic": fake_client,
        })
        result = pubmed_search("x")
        assert result["results"][0]["pmid"] == "67890"

    def test_unavailable_envelope_passes_through(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_anthropic")
        fake_client = MagicMock()
        fake_client.call_tool.return_value = {
            "status": "unavailable",
            "error": "MCP server down",
        }
        monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {
            "mcp_anthropic": fake_client,
        })
        result = pubmed_search("x")
        assert result["status"] == "unavailable"
        assert "down" in result["error"]

    def test_drops_results_without_pmid(self, monkeypatch):
        monkeypatch.setenv("PUBMED_BACKEND", "mcp_anthropic")
        fake_client = MagicMock()
        fake_client.call_tool.return_value = {
            "status": "ok",
            "results": [
                {"pmid": "111", "title": "real"},
                {"title": "no id — should drop"},
                {"id": "222", "title": "id-shaped"},
            ],
        }
        monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {
            "mcp_anthropic": fake_client,
        })
        result = pubmed_search("x", max_results=5)
        ids = [r["pmid"] for r in result["results"]]
        assert "111" in ids
        assert "222" in ids
        assert len(result["results"]) == 2


# ---------------------------------------------------------------------------
# Identical envelope shape across backends (parametrised smoke)
# ---------------------------------------------------------------------------


class TestEnvelopeShapeConsistency:
    def test_all_backends_return_status_field(self, monkeypatch):
        # Force all three to fail; confirm the unavailable shape is the same.
        for backend, env_setup in [
            ("direct", lambda: None),
            ("mcp_anthropic", lambda: None),
            ("mcp_self_host", lambda: monkeypatch.setenv("PUBMED_MCP_URL", "https://x")),
        ]:
            monkeypatch.setenv("PUBMED_BACKEND", backend)
            env_setup()

            if backend == "direct":
                # Force a network failure.
                import requests as _requests

                def raise_(*a, **kw):
                    raise _requests.RequestException("forced fail")

                monkeypatch.setattr(he_tools, "requests",
                                    type("R", (), {"get": raise_,
                                                   "RequestException": _requests.RequestException}))
            else:
                fake_client = MagicMock()
                fake_client.call_tool.return_value = {
                    "status": "unavailable", "error": "forced",
                }
                monkeypatch.setattr(he_tools, "_MCP_CLIENTS", {
                    backend: fake_client,
                })

            result = pubmed_search("x")
            assert "status" in result, f"backend={backend} missing status"
            assert result["status"] in {"ok", "unavailable"}, (
                f"backend={backend} bad status {result['status']!r}"
            )
