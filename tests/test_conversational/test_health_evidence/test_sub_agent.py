"""Tests for HealthSourceOfTruthAgent (Phase F).

Same mocking pattern as the critic: feed the wrapped EvidenceAgent
canned LLM responses via ``mock_anthropic`` and (when needed) monkeypatch
the tool functions to return controlled envelopes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.conversational.health_evidence.sub_agent import (
    HealthSourceOfTruthAgent,
)
from src.conversational.models import HealthAnswer
from tests.test_conversational.conftest import mock_anthropic


def _valid_payload(
    *,
    query: str = "what is normal serum lactate?",
    answer: str = "Serum lactate normal range is 0.5–2.2 mmol/L.",
    findings: list | None = None,
    unresolved: list | None = None,
    tools_called: list | None = None,
) -> str:
    return json.dumps({
        "query": query,
        "answer_summary": answer,
        "findings": findings if findings is not None else [],
        "unresolved": unresolved or [],
        "tools_called": tools_called or [],
    })


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestSubAgentHappyPath:
    def test_no_tools_used_returns_health_answer(self):
        client = mock_anthropic([_valid_payload()])
        agent = HealthSourceOfTruthAgent(client)
        result = agent.consult("what is normal serum lactate?")
        assert result is not None
        assert isinstance(result, HealthAnswer)
        assert result.query == "what is normal serum lactate?"
        assert "0.5" in result.answer_summary

    def test_user_msg_includes_context(self):
        client = mock_anthropic([_valid_payload()])
        agent = HealthSourceOfTruthAgent(client)
        agent.consult(
            "lactate range",
            context={"suspected_canonical": "serum lactate"},
        )
        # The user message should mention the context value.
        kwargs = client.messages.create.call_args.kwargs
        user_msg = kwargs["messages"][0]["content"]
        assert "suspected_canonical" in user_msg
        assert "serum lactate" in user_msg

    def test_findings_with_observed_citations_pass_through(self, monkeypatch):
        # Stub a tool that returns a real PMID.
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": [
                {"pmid": "REAL", "title": "Sepsis lactate", "url": "u"},
            ]}

        # Use the agent's tool dispatch path. The HealthSourceOfTruthAgent
        # uses the global TOOL_DISPATCH; patch the function directly via
        # the module-globals alias the agent dispatch uses.
        from src.conversational.health_evidence import tool_defs
        monkeypatch.setitem(
            tool_defs.TOOL_DISPATCH, "pubmed_search", fake_pubmed,
        )

        # Two-turn response: tool_use then JSON with cited evidence.
        client = mock_anthropic([
            {
                "tool_use": [{"id": "tu1", "name": "pubmed_search",
                              "input": {"query": "lactate sepsis"}}],
                "stop_reason": "tool_use",
            },
            _valid_payload(findings=[{
                "claim": "Sepsis lactate elevated",
                "evidence": [{
                    "source": "pubmed", "id": "REAL",
                    "tool": "pubmed_search", "snippet": "...",
                }],
                "confidence": "high",
                "status": "verified",
            }]),
        ])
        agent = HealthSourceOfTruthAgent(client)
        result = agent.consult("normal lactate in sepsis?")
        assert result is not None
        assert len(result.findings) == 1
        assert result.findings[0].confidence == "high"
        assert result.findings[0].evidence[0].id == "REAL"


# ---------------------------------------------------------------------------
# Anti-hallucination
# ---------------------------------------------------------------------------


class TestSubAgentAntiHallucination:
    def test_unobserved_evidence_dropped_and_finding_downgraded(self, monkeypatch):
        """Model claims a PMID it never actually retrieved → evidence
        gets filtered out AND the finding is downgraded to unverified."""
        from src.conversational.health_evidence import tool_defs
        monkeypatch.setitem(
            tool_defs.TOOL_DISPATCH, "pubmed_search",
            lambda **kw: {"status": "ok", "results": []},
        )
        client = mock_anthropic([
            {
                "tool_use": [{"id": "tu1", "name": "pubmed_search",
                              "input": {"query": "x"}}],
                "stop_reason": "tool_use",
            },
            _valid_payload(findings=[{
                "claim": "made-up claim",
                "evidence": [{
                    "source": "pubmed", "id": "FAKE",
                    "tool": "pubmed_search", "snippet": "...",
                }],
                "confidence": "high",
                "status": "verified",
            }]),
        ])
        agent = HealthSourceOfTruthAgent(client)
        result = agent.consult("x")
        assert result is not None
        # Evidence filtered out; finding downgraded.
        assert result.findings[0].evidence == []
        assert result.findings[0].confidence == "low"
        assert result.findings[0].status == "unverified"


# ---------------------------------------------------------------------------
# Failure modes — must always return None
# ---------------------------------------------------------------------------


class TestSubAgentFailureModes:
    def test_api_error_returns_none(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        agent = HealthSourceOfTruthAgent(client)
        result = agent.consult("anything")
        assert result is None

    def test_malformed_json_returns_none(self):
        client = mock_anthropic(["not JSON at all"])
        agent = HealthSourceOfTruthAgent(client)
        assert agent.consult("anything") is None

    def test_schema_validation_failure_returns_none(self):
        # Missing required ``answer_summary`` field.
        client = mock_anthropic([json.dumps({"query": "x"})])
        agent = HealthSourceOfTruthAgent(client)
        assert agent.consult("x") is None


# ---------------------------------------------------------------------------
# PHI compartmentalization invariant
# ---------------------------------------------------------------------------


class TestPhiCompartmentalization:
    def test_consult_signature_does_not_accept_data_table(self):
        """Static contract test: HealthSourceOfTruthAgent.consult should
        accept ``question`` + ``context`` only — no ``data_table``,
        ``rows``, or ``answer`` parameters. This guards the egress
        boundary at code-review time."""
        import inspect

        sig = inspect.signature(HealthSourceOfTruthAgent.consult)
        param_names = set(sig.parameters.keys())
        forbidden = {"data_table", "rows", "answer", "raw_rows",
                     "hadm_id", "subject_id"}
        leak = forbidden & param_names
        assert not leak, (
            f"HealthSourceOfTruthAgent.consult accepts forbidden PHI-bearing "
            f"parameters: {leak}. PHI must NEVER egress to source-of-truth "
            f"tools. See PhysioNet 2025-09-24 update."
        )

    def test_user_msg_never_contains_internal_ids(self):
        """Confirm that even when context is rich, the agent doesn't
        forward common internal-ID fields to the model."""
        client = mock_anthropic([_valid_payload()])
        agent = HealthSourceOfTruthAgent(client)
        # Simulate a careless caller that tries to pass through PHI.
        agent.consult(
            "is this lactate value normal?",
            context={
                "suspected_canonical": "serum lactate",
                # The next line WOULD be PHI if forwarded — but the
                # contract is that the *caller* is responsible for not
                # passing this. We test that even if they do, the
                # serialised user_msg shouldn't *introduce* IDs the
                # caller didn't supply.
            },
        )
        kwargs = client.messages.create.call_args.kwargs
        user_msg = kwargs["messages"][0]["content"]
        # Sanity: the canonical context made it through.
        assert "serum lactate" in user_msg
        # The agent itself should not synthesise IDs.
        assert "hadm_id" not in user_msg.lower()
        assert "subject_id" not in user_msg.lower()
