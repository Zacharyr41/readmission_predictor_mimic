"""Tests for EvidenceAgent — the reusable Anthropic tool-use loop.

Behaviour contract:
- ``consult`` NEVER raises. Failures return an EvidenceResult with empty
  fields and ``parsed_json=None``.
- The system prompt is wrapped with ``cache_control: ephemeral`` so callers
  don't repeat that ceremony.
- Up to ``max_iterations`` tool-use rounds plus a final text turn (so the
  total round-trip count is at most ``max_iterations + 1``).
- On the cap iteration the agent forces ``tool_choice={"type":"none"}`` to
  guarantee the model produces text.
- The ``messages`` list passed to the FINAL ``messages.create`` is not
  mutated post-call (so tests inspecting ``call_args.kwargs["messages"]``
  see the right state).
- Observed citations are deduplicated by ``(source, id)`` and only contain
  records returned by tool calls that actually fired.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.conversational.health_evidence import EvidenceAgent, EvidenceResult
from src.conversational.health_evidence.models import Citation


# ---------------------------------------------------------------------------
# Mock Anthropic client (mirrors tests/test_conversational/conftest.py shape).
# Local copy so this test file is self-contained for the new package.
# ---------------------------------------------------------------------------


def _build_response(item):
    resp = MagicMock()
    if isinstance(item, str):
        text_block = MagicMock(type="text")
        text_block.text = item
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        return resp
    if not isinstance(item, dict):
        raise TypeError(f"items must be str or dict; got {type(item).__name__}")
    blocks = []
    text = item.get("text")
    if text is not None:
        text_block = MagicMock(type="text")
        text_block.text = text
        blocks.append(text_block)
    for tu in item.get("tool_use", []) or []:
        block = MagicMock(type="tool_use")
        block.id = tu.get("id", "tu_anonymous")
        block.name = tu["name"]
        block.input = tu.get("input", {})
        blocks.append(block)
    resp.content = blocks
    resp.stop_reason = item.get("stop_reason", "end_turn")
    return resp


def mock_client(responses: list) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = [_build_response(r) for r in responses]
    return client


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestZeroToolUse:
    def test_text_only_response_yields_evidence_result(self):
        client = mock_client(['{"verdict": "pass", "concern": null}'])
        agent = EvidenceAgent(client)
        result = agent.consult("system", "user")
        assert isinstance(result, EvidenceResult)
        assert result.parsed_json == {"verdict": "pass", "concern": None}
        assert result.observed_citations == []
        assert result.tool_calls == []
        assert client.messages.create.call_count == 1

    def test_system_prompt_carries_cache_control(self):
        client = mock_client(['{"ok": true}'])
        agent = EvidenceAgent(client)
        agent.consult("MY_SYSTEM_PROMPT", "user")
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["system"][0]["text"] == "MY_SYSTEM_PROMPT"

    def test_tools_passed_to_messages_create(self):
        client = mock_client(['{"ok": true}'])
        agent = EvidenceAgent(client)
        agent.consult("system", "user")
        kwargs = client.messages.create.call_args.kwargs
        # ALL_TOOL_DEFS by default
        names = [t["name"] for t in kwargs["tools"]]
        assert "pubmed_search" in names
        assert "mimic_distribution_lookup" in names
        assert "loinc_reference_range" in names

    def test_returns_empty_when_text_has_no_json(self):
        client = mock_client(["just some prose with no braces"])
        agent = EvidenceAgent(client)
        result = agent.consult("system", "user")
        assert result.parsed_json is None
        assert result.final_text == "just some prose with no braces"


class TestOneToolUseIteration:
    def test_pubmed_call_records_observed_citation(self, monkeypatch):
        # Mock the actual pubmed_search tool function via dispatch.
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": [
                {"pmid": "12345", "title": "Sepsis lactate study", "url": "u"},
            ]}

        client = mock_client([
            {
                "stop_reason": "tool_use",
                "tool_use": [
                    {"id": "tu1", "name": "pubmed_search",
                     "input": {"query": "lactate sepsis"}},
                ],
            },
            '{"verdict": "pass", "cited_sources": [{"type": "pubmed", "pmid": "12345"}]}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={"pubmed_search": fake_pubmed},
            tools=[{
                "name": "pubmed_search",
                "description": "x",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }],
        )
        result = agent.consult("system", "user")
        assert client.messages.create.call_count == 2
        assert len(result.observed_citations) == 1
        cite = result.observed_citations[0]
        assert cite.source == "pubmed"
        assert cite.id == "12345"
        assert cite.title == "Sepsis lactate study"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "pubmed_search"
        assert result.tool_calls[0].status == "ok"
        assert result.tool_calls[0].n_results == 1

    def test_unknown_tool_returns_unavailable_result(self):
        client = mock_client([
            {
                "stop_reason": "tool_use",
                "tool_use": [
                    {"id": "tu1", "name": "nonexistent_tool", "input": {}},
                ],
            },
            '{"verdict": "pass"}',
        ])
        # Pass empty dispatch so any tool name is "unknown"
        agent = EvidenceAgent(client, tool_dispatch={}, tools=[])
        result = agent.consult("system", "user")
        # No observed citations because the tool returned unavailable.
        assert result.observed_citations == []
        # But the call IS recorded as unavailable for telemetry.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].status == "unavailable"


class TestMultiToolUseIteration:
    def test_two_tool_iterations_both_observed(self):
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": [
                {"pmid": "1", "title": "T1"},
            ]}

        def fake_loinc(loinc_code):
            return {"status": "ok", "results": [
                {"loinc_code": "2160-0", "low": 0.7, "high": 1.3, "units": "mg/dL"},
            ]}

        client = mock_client([
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu1", "name": "pubmed_search", "input": {"query": "x"}}]},
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu2", "name": "loinc_reference_range",
                           "input": {"loinc_code": "2160-0"}}]},
            '{"verdict": "pass"}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={
                "pubmed_search": fake_pubmed,
                "loinc_reference_range": fake_loinc,
            },
            tools=[
                {"name": "pubmed_search", "description": "x",
                 "input_schema": {"type": "object", "properties": {}, "required": []}},
                {"name": "loinc_reference_range", "description": "x",
                 "input_schema": {"type": "object", "properties": {}, "required": []}},
            ],
        )
        result = agent.consult("system", "user")
        assert client.messages.create.call_count == 3
        sources = sorted(c.source for c in result.observed_citations)
        assert sources == ["loinc", "pubmed"]


class TestIterationCap:
    def test_tool_choice_none_set_on_cap_iter(self):
        # Configure max_iterations=1 → we get 1 tool round + 1 forced-text round.
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": []}

        client = mock_client([
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu1", "name": "pubmed_search", "input": {"query": "x"}}]},
            '{"verdict": "pass"}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={"pubmed_search": fake_pubmed},
            tools=[{"name": "pubmed_search", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
            max_iterations=1,
        )
        agent.consult("system", "user")
        # Last call (the cap iter) should have tool_choice={"type":"none"}.
        last_kwargs = client.messages.create.call_args_list[-1].kwargs
        assert last_kwargs.get("tool_choice") == {"type": "none"}
        # First call (the regular tool-use iter) should NOT.
        first_kwargs = client.messages.create.call_args_list[0].kwargs
        assert "tool_choice" not in first_kwargs


class TestDeduplication:
    def test_same_pmid_observed_twice_dedupes(self):
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": [
                {"pmid": "1", "title": "T"},
                {"pmid": "1", "title": "T"},  # duplicate
            ]}

        client = mock_client([
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu1", "name": "pubmed_search", "input": {"query": "x"}}]},
            '{"verdict": "pass"}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={"pubmed_search": fake_pubmed},
            tools=[{"name": "pubmed_search", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
        )
        result = agent.consult("system", "user")
        assert len(result.observed_citations) == 1
        assert result.observed_citations[0].id == "1"


# ---------------------------------------------------------------------------
# Failure modes — ALL must return empty EvidenceResult, never raise
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_api_error_returns_empty_result(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        agent = EvidenceAgent(client)
        result = agent.consult("system", "user")
        assert isinstance(result, EvidenceResult)
        assert result.parsed_json is None
        assert result.final_text == ""
        assert result.observed_citations == []

    def test_timeout_returns_empty_result(self):
        client = MagicMock()
        client.messages.create.side_effect = TimeoutError("slow")
        agent = EvidenceAgent(client)
        result = agent.consult("system", "user")
        assert result.parsed_json is None

    def test_malformed_json_in_final_turn(self):
        client = mock_client(["{not valid json}"])
        agent = EvidenceAgent(client)
        result = agent.consult("system", "user")
        assert result.parsed_json is None
        assert result.final_text == "{not valid json}"

    def test_tool_returns_unavailable_propagates_no_citations(self):
        def fake_pubmed(query, max_results=5):
            return {"status": "unavailable", "error": "rate limited"}

        client = mock_client([
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu1", "name": "pubmed_search", "input": {"query": "x"}}]},
            '{"verdict": "pass"}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={"pubmed_search": fake_pubmed},
            tools=[{"name": "pubmed_search", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
        )
        result = agent.consult("system", "user")
        assert result.observed_citations == []
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].status == "unavailable"
        # parsed_json still derived from the final-text turn.
        assert result.parsed_json == {"verdict": "pass"}

    def test_tool_function_raising_does_not_crash_agent(self):
        def raising_tool(**kw):
            raise RuntimeError("oh no")

        client = mock_client([
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu1", "name": "broken", "input": {}}]},
            '{"verdict": "pass"}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={"broken": raising_tool},
            tools=[{"name": "broken", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
        )
        result = agent.consult("system", "user")
        # Should NOT raise; tool call recorded as unavailable.
        assert result.tool_calls[0].status == "unavailable"


# ---------------------------------------------------------------------------
# filter_claimed_citations
# ---------------------------------------------------------------------------


class TestFilterClaimedCitations:
    def test_filters_unobserved_pmids(self):
        result = EvidenceResult(
            observed_citations=[
                Citation(source="pubmed", id="111"),
                Citation(source="pubmed", id="222"),
            ],
        )
        claimed = [
            {"type": "pubmed", "pmid": "111", "title": "real"},
            {"type": "pubmed", "pmid": "999", "title": "hallucinated"},
        ]
        kept = result.filter_claimed_citations(claimed)
        assert kept is not None
        assert len(kept) == 1
        assert kept[0]["pmid"] == "111"

    def test_returns_none_when_input_none(self):
        assert EvidenceResult().filter_claimed_citations(None) is None

    def test_returns_none_when_all_filtered(self):
        result = EvidenceResult()  # no observed
        claimed = [{"type": "pubmed", "pmid": "111"}]
        # Empty list collapses to None per existing critic contract.
        assert result.filter_claimed_citations(claimed) is None

    def test_accepts_new_source_id_shape(self):
        result = EvidenceResult(
            observed_citations=[Citation(source="loinc", id="2160-0")],
        )
        claimed = [{"source": "loinc", "id": "2160-0"}]
        kept = result.filter_claimed_citations(claimed)
        assert kept is not None and len(kept) == 1


# ---------------------------------------------------------------------------
# Mutation discipline
# ---------------------------------------------------------------------------


class TestMutationDiscipline:
    def test_messages_list_not_mutated_after_final_call(self):
        """Critical for tests that inspect call_args.kwargs['messages'].
        After end_turn, the list passed to the final messages.create call
        must be exactly what the agent sent — not appended to."""
        client = mock_client(['{"verdict": "pass"}'])
        agent = EvidenceAgent(client)
        agent.consult("system", "user")
        sent = client.messages.create.call_args.kwargs["messages"]
        # Should be the original [user] message, not appended-to with assistant.
        assert len(sent) == 1
        assert sent[0]["role"] == "user"

    def test_messages_list_grows_across_tool_iterations(self):
        def fake_pubmed(query, max_results=5):
            return {"status": "ok", "results": []}

        client = mock_client([
            {"stop_reason": "tool_use",
             "tool_use": [{"id": "tu1", "name": "pubmed_search", "input": {"query": "x"}}]},
            '{"verdict": "pass"}',
        ])
        agent = EvidenceAgent(
            client,
            tool_dispatch={"pubmed_search": fake_pubmed},
            tools=[{"name": "pubmed_search", "description": "x",
                    "input_schema": {"type": "object", "properties": {}, "required": []}}],
        )
        agent.consult("system", "user")
        # First call: just [user].
        first_messages = client.messages.create.call_args_list[0].kwargs["messages"]
        # Second (final) call: [user, assistant(tool_use), user(tool_result)].
        last_messages = client.messages.create.call_args_list[1].kwargs["messages"]
        assert len(first_messages) == 1
        assert len(last_messages) == 3
        assert last_messages[0]["role"] == "user"
        assert last_messages[1]["role"] == "assistant"
        assert last_messages[2]["role"] == "user"
