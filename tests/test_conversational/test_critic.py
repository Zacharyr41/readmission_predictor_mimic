"""Tests for the critic / LLM-as-judge module.

The critic is a second-pass plausibility check: after the conversational
pipeline produces an AnswerResult, the critic reviews the answer for
clinical plausibility (catches lactate=199 / age=380 / mortality>1 /
units mismatches / pollution artifacts).

These tests exercise the critic CONTRACT — schema, error handling, what
the critic sends to Claude — using a mock client that returns canned
JSON verdicts. Real-LLM judgment quality is verified manually in Phase 7.

The naming `test_catches_X` reflects what the test SCENARIO models, not
what the test verifies; the canned mock response asserts the parse +
schema flow holds for that scenario shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.conversational.critic import critique
from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
    CriticVerdict,
)
from src.conversational.prompts import CRITIC_SYSTEM_PROMPT
from tests.test_conversational.conftest import mock_anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cq(question: str = "test question") -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question=question,
        clinical_concepts=[ClinicalConcept(name="x", concept_type="biomarker")],
        return_type="text_and_table",
        scope="cohort",
    )


def _make_answer(
    text_summary: str,
    data_table: list[dict] | None = None,
) -> AnswerResult:
    return AnswerResult(
        text_summary=text_summary,
        data_table=data_table,
    )


def _verdict_response(
    severity: str,
    concern: str | None = None,
    *,
    plausible: bool | None = None,
    reference_used: str | None = None,
) -> str:
    if plausible is None:
        plausible = severity == "info"
    return json.dumps({
        "plausible": plausible,
        "severity": severity,
        "concern": concern,
        "reference_used": reference_used,
    })


# ---------------------------------------------------------------------------
# Verdicts that should fire (mock returns warn/block for the scenario)
# ---------------------------------------------------------------------------


class TestCriticCatches:
    """Inputs that model real failure scenarios. The mock returns the
    matching warn/block JSON verdict; tests confirm the parse + schema
    pass through correctly."""

    def test_catches_pooled_lactate_199(self):
        client = mock_anthropic([_verdict_response(
            severity="warn",
            concern=(
                "Mean lactate of 199 mg/dL is biologically impossible. The "
                "system warning notes a LOINC fallback; this matches typical "
                "Lactate Dehydrogenase (LDH) U/L ranges — likely LIKE-pollution."
            ),
            reference_used="serum lactate normal 0.5-2.0 mmol/L; max ~25 in shock",
        )])
        cq = _make_cq("What is the average lactate for ICU patients?")
        answer = _make_answer("Mean lactate 199.43 mg/dL")
        verdict = critique(
            client, cq, answer,
            fallback_warning="LOINC '2524-7' has no MIMIC labitem coverage",
        )
        assert verdict is not None
        assert isinstance(verdict, CriticVerdict)
        assert verdict.severity == "warn"
        assert verdict.plausible is False
        assert "199" in verdict.concern or "pollution" in verdict.concern.lower()

    def test_catches_pooled_creatinine_pre_fix(self):
        client = mock_anthropic([_verdict_response(
            severity="warn",
            concern=(
                "Mean serum creatinine of 4.95 mg/dL is ~4× normal; if this "
                "is a population mean it suggests pollution or a cohort issue."
            ),
        )])
        cq = _make_cq("What is the average creatinine for patients over 65?")
        answer = _make_answer("Mean creatinine 4.95 mg/dL")
        verdict = critique(client, cq, answer)
        assert verdict.severity == "warn"
        assert verdict.plausible is False

    def test_catches_age_380(self):
        client = mock_anthropic([_verdict_response(
            severity="block",
            concern="Mean age 380 years is biologically impossible.",
        )])
        cq = _make_cq("What is the average age in the cohort?")
        answer = _make_answer("Mean age 380 years")
        verdict = critique(client, cq, answer)
        assert verdict.severity == "block"
        assert verdict.plausible is False

    def test_catches_unit_mismatch(self):
        client = mock_anthropic([_verdict_response(
            severity="warn",
            concern=(
                "Summary states 'mg/dL' but the value 8.4 is in the mmol/L "
                "range for sodium. Either the value or the unit is wrong."
            ),
        )])
        cq = _make_cq("What is the average sodium?")
        answer = _make_answer("Mean sodium 8.4 mg/dL")
        verdict = critique(client, cq, answer)
        assert verdict.severity == "warn"

    def test_catches_mortality_above_one(self):
        client = mock_anthropic([_verdict_response(
            severity="block",
            concern="Mortality rate of 1.4 is impossible (proportion must be in [0, 1]).",
        )])
        cq = _make_cq("What is the mortality rate?")
        answer = _make_answer("Mortality rate 1.4")
        verdict = critique(client, cq, answer)
        assert verdict.severity == "block"


# ---------------------------------------------------------------------------
# Verdicts that should pass (mock returns info — calibration)
# ---------------------------------------------------------------------------


class TestCriticPasses:
    """Legitimate values get severity='info' — the critic shouldn't fire on
    legitimately abnormal-but-possible ICU labs."""

    def test_passes_legitimate_high_icu_lactate(self):
        client = mock_anthropic([_verdict_response(severity="info")])
        cq = _make_cq("What is the average lactate in shock patients?")
        answer = _make_answer("Mean lactate 8.2 mmol/L")
        verdict = critique(client, cq, answer)
        assert verdict.severity == "info"
        assert verdict.plausible is True
        assert verdict.concern is None

    def test_passes_normal_serum_creatinine(self):
        client = mock_anthropic([_verdict_response(severity="info")])
        cq = _make_cq("What is the average creatinine for patients over 65?")
        answer = _make_answer("Mean creatinine 1.4 mg/dL")
        verdict = critique(client, cq, answer)
        assert verdict.severity == "info"


# ---------------------------------------------------------------------------
# Failure handling — must never raise; return None and let the answer render
# ---------------------------------------------------------------------------


class TestCriticFailures:

    def test_returns_none_on_api_failure(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is None

    def test_returns_none_on_malformed_json(self):
        client = mock_anthropic(["this is not valid json at all {{{"])
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is None

    def test_returns_none_on_timeout(self):
        client = MagicMock()
        client.messages.create.side_effect = TimeoutError("timeout")
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is None

    def test_returns_none_on_schema_validation_failure(self):
        # Valid JSON but missing required fields like `severity`.
        client = mock_anthropic([json.dumps({"foo": "bar"})])
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is None


# ---------------------------------------------------------------------------
# Input forwarding — verify the critic actually sends the right inputs
# ---------------------------------------------------------------------------


class TestCriticInputForwarding:
    """Beyond parsing: the critic must pass the question, answer, and any
    fallback warning into the user message so Claude can reason over them."""

    def test_user_message_includes_question(self):
        client = mock_anthropic([_verdict_response("info")])
        cq = _make_cq("very specific question text about creatinine")
        answer = _make_answer("Some answer")
        critique(client, cq, answer)
        kwargs = client.messages.create.call_args.kwargs
        user_content = kwargs["messages"][-1]["content"]
        if isinstance(user_content, list):  # may be content blocks
            user_content = " ".join(b.get("text", "") for b in user_content)
        assert "very specific question text about creatinine" in user_content

    def test_user_message_includes_answer_text(self):
        client = mock_anthropic([_verdict_response("info")])
        cq = _make_cq()
        answer = _make_answer("Mean lactate 199.43 mg/dL")
        critique(client, cq, answer)
        kwargs = client.messages.create.call_args.kwargs
        user_content = kwargs["messages"][-1]["content"]
        if isinstance(user_content, list):
            user_content = " ".join(b.get("text", "") for b in user_content)
        assert "199.43" in user_content

    def test_user_message_includes_fallback_warning(self):
        client = mock_anthropic([_verdict_response("info")])
        cq = _make_cq()
        answer = _make_answer("test")
        critique(
            client, cq, answer,
            fallback_warning="LOINC '2524-7' unknown — falling back to LIKE",
        )
        kwargs = client.messages.create.call_args.kwargs
        user_content = kwargs["messages"][-1]["content"]
        if isinstance(user_content, list):
            user_content = " ".join(b.get("text", "") for b in user_content)
        assert "LOINC '2524-7'" in user_content or "falling back" in user_content


# ---------------------------------------------------------------------------
# Phase 5: critic system-prompt snapshot
#
# Mirrors ``TestPromptSnapshot`` at test_decomposer_contract.py:381-406.
# Any prompt change forces an explicit snapshot regen step, so changes
# show up in PR diffs rather than buried in a string constant edit.
# ---------------------------------------------------------------------------


class TestCriticPromptSnapshot:
    """The committed ``critic_prompt_snapshot.txt`` is the rendered critic
    system prompt verbatim. When the prompt changes — adding a reference
    range, edits to the failure-mode taxonomy, etc. — this test fails
    loudly and forces snapshot regeneration in the same commit."""

    SNAPSHOT_PATH = (
        Path(__file__).parent / "fixtures" / "critic_prompt_snapshot.txt"
    )

    def test_critic_prompt_matches_snapshot(self):
        if not self.SNAPSHOT_PATH.exists():
            self.SNAPSHOT_PATH.write_text(CRITIC_SYSTEM_PROMPT)
            pytest.skip(
                f"Snapshot written to {self.SNAPSHOT_PATH}. Re-run to verify."
            )
        expected = self.SNAPSHOT_PATH.read_text()
        if expected != CRITIC_SYSTEM_PROMPT:
            diff_preview = (
                f"Critic prompt changed. To accept, re-run with:\n"
                f"    rm {self.SNAPSHOT_PATH}\n"
                f"    pytest {__file__}::TestCriticPromptSnapshot\n"
                f"Length: expected {len(expected)} chars, got "
                f"{len(CRITIC_SYSTEM_PROMPT)}."
            )
            assert expected == CRITIC_SYSTEM_PROMPT, diff_preview


# ---------------------------------------------------------------------------
# Self-healing critic — verdict carries a corrective LOINC suggestion
# ---------------------------------------------------------------------------
#
# These tests pin the *contract* for the new fields on CriticVerdict:
# ``suggested_loinc`` + ``correction_rationale``. Whether the orchestrator
# actually retries on a suggestion is tested in test_orchestrator.py
# (TestSelfHealingCritic). Here we verify the parse + validation layer.


class TestCriticSuggests:
    def test_critic_emits_suggested_loinc_round_trips(self):
        """When the critic returns suggested_loinc + correction_rationale,
        both fields populate on the parsed verdict."""
        client = mock_anthropic([json.dumps({
            "plausible": False,
            "severity": "block",
            "concern": "Mean lactate of 199 mg/dL is in the LDH range.",
            "reference_used": "Serum lactate ICU-plausible upper ~135 mg/dL",
            "suggested_loinc": "32693-4",
            "correction_rationale": (
                "MIMIC codes lactate molarly via LOINC 32693-4 (mmol/L blood); "
                "the prompt's 2524-7 (mg/dL serum) doesn't ground."
            ),
        })])
        cq = _make_cq("What is the average lactate?")
        answer = _make_answer("Mean lactate 199.43 mg/dL")
        verdict = critique(
            client, cq, answer,
            fallback_warning="LOINC '2524-7' not found in mapping table",
        )
        assert verdict is not None
        assert verdict.suggested_loinc == "32693-4"
        assert verdict.correction_rationale is not None
        assert "32693-4" in verdict.correction_rationale or "LOINC" in verdict.correction_rationale

    def test_critic_omits_suggestion_on_plausible_answer(self):
        """severity='info' verdicts shouldn't carry a corrective suggestion.
        The contract is permissive — we don't enforce — but the round-trip
        of None values must work cleanly."""
        client = mock_anthropic([json.dumps({
            "plausible": True,
            "severity": "info",
            "concern": None,
            "reference_used": None,
            "suggested_loinc": None,
            "correction_rationale": None,
        })])
        cq = _make_cq()
        answer = _make_answer("Mean creatinine 1.4 mg/dL")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert verdict.suggested_loinc is None
        assert verdict.correction_rationale is None

    def test_critic_handles_malformed_loinc_gracefully(self):
        """A malformed suggested_loinc (e.g. typo, freeform text) must be
        coerced to None by the field_validator — never crash a turn,
        never let the orchestrator try to use a bogus code."""
        client = mock_anthropic([json.dumps({
            "plausible": False,
            "severity": "warn",
            "concern": "Suspect pollution.",
            "suggested_loinc": "not-a-real-loinc",
            "correction_rationale": "could not pin down",
        })])
        cq = _make_cq()
        answer = _make_answer("Mean lactate 199 mg/dL")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert verdict.suggested_loinc is None  # coerced
        # rationale survives — it's free-text and might still be useful
        assert verdict.correction_rationale == "could not pin down"

    def test_correction_trace_field_default_is_none(self):
        """AnswerResult.correction_trace defaults to None — no retry happened.
        This is the regression guard for the common path."""
        from src.conversational.models import AnswerResult

        a = AnswerResult(text_summary="hello")
        assert a.correction_trace is None


# ---------------------------------------------------------------------------
# Externally-grounded critic — cited_sources schema
# ---------------------------------------------------------------------------


class TestCriticCitedSources:
    """The ``cited_sources`` field carries PubMed records the critic
    consulted. Pydantic-side validation does shape only; identity
    validation (filtering fabricated PMIDs) is critic.py's responsibility."""

    def test_cited_sources_default_is_none(self):
        v = CriticVerdict(plausible=True, severity="info")
        assert v.cited_sources is None

    def test_cited_sources_round_trip(self):
        v = CriticVerdict(
            plausible=True, severity="info",
            cited_sources=[
                {"type": "pubmed", "pmid": "27621888", "title": "T", "url": "https://pubmed.ncbi.nlm.nih.gov/27621888/"},
            ],
        )
        assert v.cited_sources is not None
        assert len(v.cited_sources) == 1
        assert v.cited_sources[0]["pmid"] == "27621888"

    def test_cited_sources_drops_malformed_entries(self):
        v = CriticVerdict(
            plausible=True, severity="info",
            cited_sources=[
                {"type": "pubmed", "pmid": "12345", "title": "ok"},
                "not a dict",
                {"type": "pubmed", "pmid": "abc-not-digits", "title": "bad"},
                {"type": "pubmed", "title": "no pmid"},
            ],
        )
        assert v.cited_sources is not None
        assert len(v.cited_sources) == 1
        assert v.cited_sources[0]["pmid"] == "12345"

    def test_cited_sources_empty_list_coerces_to_none(self):
        v = CriticVerdict(
            plausible=True, severity="info", cited_sources=[],
        )
        assert v.cited_sources is None

    def test_cited_sources_all_malformed_coerces_to_none(self):
        v = CriticVerdict(
            plausible=True, severity="info",
            cited_sources=[{"no_pmid": "x"}, "junk"],
        )
        assert v.cited_sources is None

    def test_cited_sources_pmid_coerced_to_string(self):
        """Models could feed integer pmids; we coerce to string for stable round-trip."""
        v = CriticVerdict(
            plausible=True, severity="info",
            cited_sources=[{"type": "pubmed", "pmid": 12345, "title": "x"}],
        )
        assert v.cited_sources is not None
        assert v.cited_sources[0]["pmid"] == "12345"


# ---------------------------------------------------------------------------
# Tool-use loop in critique() — externally-grounded critic
# ---------------------------------------------------------------------------
#
# These tests pin the contract for the critic's tool-use loop. We mock
# ``mock_anthropic`` with multi-step response sequences (tool_use →
# end_turn) and monkeypatch ``pubmed_search`` so the suite stays offline.
# The critic must never raise; tool failures degrade to fallback.


def _tool_use_response(
    tool_name: str = "pubmed_search",
    tool_input: dict | None = None,
    tool_id: str = "tu_1",
    text: str | None = None,
) -> dict:
    return {
        "tool_use": [{"id": tool_id, "name": tool_name, "input": tool_input or {}}],
        "stop_reason": "tool_use",
        **({"text": text} if text else {}),
    }


def _end_turn_response(verdict_dict: dict) -> dict:
    return {"text": json.dumps(verdict_dict), "stop_reason": "end_turn"}


class TestCriticToolUse:

    def test_critic_calls_pubmed_when_uncertain(self, monkeypatch):
        """Happy path: model first emits tool_use; we execute pubmed_search;
        model emits end_turn with verdict citing the records."""
        # Fake pubmed_search return.
        fake_records = [
            {"pmid": "27621888", "title": "Lactate as a marker", "url": "https://pubmed.ncbi.nlm.nih.gov/27621888/"},
            {"pmid": "29100563", "title": "Sepsis vasopressors", "url": "https://pubmed.ncbi.nlm.nih.gov/29100563/"},
        ]
        monkeypatch.setattr(
            "src.conversational.critic.pubmed_search",
            lambda **kw: {"status": "ok", "results": fake_records},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_input={"query": "lactate sepsis ICU mortality"},
                tool_id="tu_1",
            ),
            _end_turn_response({
                "plausible": True,
                "severity": "info",
                "concern": None,
                "reference_used": "PubMed: 27621888, 29100563",
                "cited_sources": [
                    {"type": "pubmed", "pmid": "27621888", "title": "Lactate as a marker", "url": "https://pubmed.ncbi.nlm.nih.gov/27621888/"},
                    {"type": "pubmed", "pmid": "29100563", "title": "Sepsis vasopressors", "url": "https://pubmed.ncbi.nlm.nih.gov/29100563/"},
                ],
            }),
        ])
        cq = _make_cq()
        answer = _make_answer("Mean lactate 7.99 mmol/L")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert verdict.severity == "info"
        assert verdict.cited_sources is not None
        assert {s["pmid"] for s in verdict.cited_sources} == {"27621888", "29100563"}
        # Two LLM calls happened.
        assert client.messages.create.call_count == 2

    def test_critic_no_tool_call_simple_case(self):
        """Model goes straight to end_turn — no tool call. Behaves like
        the pre-tool-use critic; ``cited_sources`` is None."""
        client = mock_anthropic([
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
            }),
        ])
        cq = _make_cq()
        answer = _make_answer("Mean creatinine 1.4 mg/dL")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert verdict.severity == "info"
        assert verdict.cited_sources is None
        assert client.messages.create.call_count == 1

    def test_max_tool_calls_capped(self, monkeypatch):
        """Model keeps requesting tool_use; loop forces a final response
        on the cap iteration via tool_choice={'type':'none'}."""
        monkeypatch.setattr(
            "src.conversational.critic.pubmed_search",
            lambda **kw: {"status": "ok", "results": []},
        )
        # Three tool_use rounds + one forced final = 4 LLM calls.
        client = mock_anthropic([
            _tool_use_response(tool_input={"query": "q1"}, tool_id="t1"),
            _tool_use_response(tool_input={"query": "q2"}, tool_id="t2"),
            _tool_use_response(tool_input={"query": "q3"}, tool_id="t3"),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
            }),
        ])
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert client.messages.create.call_count == 4
        # Final call carries tool_choice={"type":"none"}.
        final_call_kwargs = client.messages.create.call_args_list[-1].kwargs
        assert final_call_kwargs.get("tool_choice") == {"type": "none"}

    def test_tool_failure_does_not_block_verdict(self, monkeypatch):
        """When pubmed_search returns an unavailable envelope, the critic
        proceeds with the reference table; verdict still produced."""
        monkeypatch.setattr(
            "src.conversational.critic.pubmed_search",
            lambda **kw: {"status": "unavailable", "error": "API down"},
        )
        client = mock_anthropic([
            _tool_use_response(tool_input={"query": "anything"}),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
            }),
        ])
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert verdict.severity == "info"
        assert verdict.cited_sources is None  # nothing real to cite

    def test_unknown_tool_name_returns_unavailable(self):
        """Model invents a tool name not in TOOL_DISPATCH; critic loop
        injects an unavailable result instead of crashing."""
        client = mock_anthropic([
            _tool_use_response(tool_name="invented_tool", tool_input={"x": 1}),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
            }),
        ])
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        # Loop completed; verdict produced via fallback.

    def test_fabricated_pmid_filtered_out(self, monkeypatch):
        """Model cites a PMID that wasn't in any tool_result. The critic
        loop strips it before constructing CriticVerdict."""
        monkeypatch.setattr(
            "src.conversational.critic.pubmed_search",
            lambda **kw: {"status": "ok", "results": [
                {"pmid": "11111", "title": "real", "url": "https://pubmed.ncbi.nlm.nih.gov/11111/"},
            ]},
        )
        client = mock_anthropic([
            _tool_use_response(tool_input={"query": "x"}),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
                "cited_sources": [
                    {"type": "pubmed", "pmid": "11111", "title": "real", "url": "https://pubmed.ncbi.nlm.nih.gov/11111/"},
                    {"type": "pubmed", "pmid": "99999999", "title": "fake", "url": "https://pubmed.ncbi.nlm.nih.gov/99999999/"},
                ],
            }),
        ])
        cq = _make_cq()
        answer = _make_answer("test")
        verdict = critique(client, cq, answer)
        assert verdict is not None
        assert verdict.cited_sources is not None
        assert len(verdict.cited_sources) == 1
        assert verdict.cited_sources[0]["pmid"] == "11111"

    def test_critic_uses_30s_timeout(self):
        """The critic's tool-use loop has a longer per-call timeout (30s)
        than the pre-tool-use single-call critic (10s) to accommodate tool
        execution latency."""
        client = mock_anthropic([
            _end_turn_response({"plausible": True, "severity": "info"}),
        ])
        cq = _make_cq()
        answer = _make_answer("test")
        critique(client, cq, answer)
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs.get("timeout") == 30.0

    def test_tools_parameter_passed_to_messages_create(self):
        """The critic always passes the tools list (PubMed) so the model
        can choose to invoke it. Per the design: pass tools always; let
        the model decide; cap iters in our loop."""
        client = mock_anthropic([
            _end_turn_response({"plausible": True, "severity": "info"}),
        ])
        cq = _make_cq()
        answer = _make_answer("test")
        critique(client, cq, answer)
        kwargs = client.messages.create.call_args.kwargs
        tools = kwargs.get("tools") or []
        tool_names = {t["name"] for t in tools}
        assert "pubmed_search" in tool_names


# ---------------------------------------------------------------------------
# _CRITIC_TOOLS dispatch coverage (Phase H)
# ---------------------------------------------------------------------------


class TestCriticToolDispatchCoverage:
    """Guards the critic's tool dispatch infrastructure.

    The dispatch is built as a dict-comprehension over ``_CRITIC_TOOLS``;
    these tests catch (a) drift between the tuple and the dispatch dict
    and (b) the classic late-binding closure pitfall where every lambda
    in a comprehension captures the same loop variable."""

    def test_dispatch_includes_one_entry_per_critic_tool(self):
        from src.conversational.critic import (
            _CRITIC_TOOLS,
            _critic_tool_dispatch,
        )
        dispatch = _critic_tool_dispatch()
        assert set(dispatch.keys()) == set(_CRITIC_TOOLS)

    def test_dispatch_each_lambda_routes_to_its_own_tool(self, monkeypatch):
        """Late-binding guard. If lambdas all close over the same name,
        every dispatch entry would call the LAST tool. Patch each tool to
        return its own marker; verify each dispatch entry returns the
        correct marker."""
        from src.conversational.critic import (
            _CRITIC_TOOLS,
            _critic_tool_dispatch,
        )
        for name in _CRITIC_TOOLS:
            monkeypatch.setattr(
                f"src.conversational.critic.{name}",
                lambda _marker=name, **_kw: {
                    "status": "ok", "results": [], "_marker": _marker,
                },
            )
        dispatch = _critic_tool_dispatch()
        for name in _CRITIC_TOOLS:
            result = dispatch[name]()
            assert result.get("_marker") == name, (
                f"dispatch[{name!r}] routed to {result.get('_marker')!r}"
            )

    def test_critic_tool_defs_match_critic_tools(self):
        from src.conversational.critic import (
            _CRITIC_TOOLS,
            _CRITIC_TOOL_DEFS,
        )
        assert {d["name"] for d in _CRITIC_TOOL_DEFS} == set(_CRITIC_TOOLS)


# ---------------------------------------------------------------------------
# Critic with broader tool access (Phase H)
# ---------------------------------------------------------------------------


class TestCriticLoincReferenceRange:
    """Critic gains access to ``loinc_reference_range`` so it can cite
    published normal ranges instead of recalling them from training data."""

    def test_critic_calls_loinc_reference_range(self, monkeypatch):
        """Happy path: model emits tool_use(loinc_reference_range);
        verdict cites the LOINC source with id=loinc_code."""
        monkeypatch.setattr(
            "src.conversational.critic.loinc_reference_range",
            lambda **kw: {"status": "ok", "results": [{
                "loinc_code": "33747-0",
                "low": 0.0, "high": 0.5, "units": "ng/mL",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="loinc_reference_range",
                tool_input={"loinc_code": "33747-0"},
                tool_id="lt1",
            ),
            _end_turn_response({
                "plausible": True,
                "severity": "info",
                "concern": None,
                "reference_used": "LOINC 33747-0 (procalcitonin)",
                "cited_sources": [
                    {"source": "loinc", "id": "33747-0", "title": "procalcitonin"},
                ],
            }),
        ])
        verdict = critique(
            client,
            _make_cq("typical procalcitonin in sepsis"),
            _make_answer("Mean procalcitonin 1.2 ng/mL"),
        )
        assert verdict is not None
        assert verdict.severity == "info"
        assert verdict.cited_sources is not None
        assert any(
            s.get("source") == "loinc" and s.get("id") == "33747-0"
            for s in verdict.cited_sources
        )
        assert client.messages.create.call_count == 2

    def test_critic_loinc_unavailable_falls_through(self, monkeypatch):
        """LOINC tool returns unavailable: critic still produces a verdict.
        Graceful degradation is the agent's existing contract."""
        monkeypatch.setattr(
            "src.conversational.critic.loinc_reference_range",
            lambda **kw: {"status": "unavailable", "error": "catalog missing"},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="loinc_reference_range",
                tool_input={"loinc_code": "33747-0"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.severity == "info"

    def test_loinc_tool_def_in_critic_system_block(self):
        """LOINC tool def is passed to messages.create so the model knows
        the tool exists."""
        client = mock_anthropic([
            _end_turn_response({"plausible": True, "severity": "info"}),
        ])
        critique(client, _make_cq(), _make_answer("test"))
        kwargs = client.messages.create.call_args.kwargs
        tool_names = {t["name"] for t in (kwargs.get("tools") or [])}
        assert "loinc_reference_range" in tool_names


class TestCriticMimicDistributionLookup:
    """Critic gains access to ``mimic_distribution_lookup`` so it can
    distinguish severity-shifted-but-plausible from polluted aggregates
    by checking the cohort-typical distribution in MIMIC itself.

    Canonical scenario from
    ``memory/project_critic_external_grounding.md``: lactate ~7.99 mmol/L
    in a sepsis cohort. Without the tool, the critic would either flag
    it (false positive) or accept it (true positive only by luck). With
    the tool, it can confirm the value sits inside MIMIC's sepsis-cohort
    typical range and emit ``severity=info`` with a citation."""

    def test_critic_uses_mimic_distribution_for_cohort_shifted_value(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            "src.conversational.critic.mimic_distribution_lookup",
            lambda **kw: {"status": "ok", "results": [{
                "itemid": 50813,
                "n": 1842,
                "mean": 4.1,
                "p50": 2.1,
                "p95": 8.2,
                "units": "mmol/L",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="mimic_distribution_lookup",
                tool_input={"itemid": 50813},
                tool_id="mt1",
            ),
            _end_turn_response({
                "plausible": True,
                "severity": "info",
                "concern": None,
                "reference_used": "MIMIC sepsis-cohort lactate p95=8.2 mmol/L",
                "cited_sources": [
                    {"source": "mimic_distribution", "id": "50813",
                     "title": "lactate (sepsis cohort)"},
                ],
            }),
        ])
        verdict = critique(
            client,
            _make_cq("mean lactate in sepsis"),
            _make_answer("Mean lactate 7.99 mmol/L (n=512)"),
        )
        assert verdict is not None
        assert verdict.severity == "info"
        assert verdict.plausible is True
        assert verdict.cited_sources is not None
        assert any(
            s.get("source") == "mimic_distribution" and s.get("id") == "50813"
            for s in verdict.cited_sources
        )

    def test_mimic_tool_def_in_critic_system_block(self):
        client = mock_anthropic([
            _end_turn_response({"plausible": True, "severity": "info"}),
        ])
        critique(client, _make_cq(), _make_answer("test"))
        kwargs = client.messages.create.call_args.kwargs
        tool_names = {t["name"] for t in (kwargs.get("tools") or [])}
        assert "mimic_distribution_lookup" in tool_names


class TestCriticPhaseGMcpTools:
    """Phase-G source-of-truth MCPs are reachable from the critic.

    One end-to-end happy-path test per tool, plus one
    backend-unavailable test to confirm graceful degradation. The
    critic is the production consumer that gives these MCPs reach
    (Bucket 2 of the Phase H plan)."""

    def test_critic_calls_snomed_search(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.snomed_search",
            lambda **kw: {"status": "ok", "results": [{
                "concept_id": "76571007",
                "preferred_term": "Septic shock",
                "fully_specified_name": "Septic shock (disorder)",
                "semantic_tag": "disorder",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="snomed_search",
                tool_input={"term": "septic shock"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
                "cited_sources": [
                    {"source": "snomed", "id": "76571007",
                     "title": "Septic shock"},
                ],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None
        assert any(
            s.get("source") == "snomed" and s.get("id") == "76571007"
            for s in verdict.cited_sources
        )

    def test_critic_calls_rxnorm_lookup(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.rxnorm_lookup",
            lambda **kw: {"status": "ok", "results": [{
                "rxcui": "7980", "name": "norepinephrine",
                "tty": "IN", "vocabulary": "RxNorm",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="rxnorm_lookup",
                tool_input={"drug_name": "norepinephrine"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "cited_sources": [
                    {"source": "rxnorm", "id": "7980", "title": "norepinephrine"},
                ],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None
        assert any(
            s.get("source") == "rxnorm" and s.get("id") == "7980"
            for s in verdict.cited_sources
        )

    def test_critic_calls_openfda_drug_label(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.openfda_drug_label",
            lambda **kw: {"status": "ok", "results": [{
                "brand_name": "Levophed",
                "generic_name": "norepinephrine bitartrate",
                "indications_and_usage": "shock",
                "warnings": "extravasation risk",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="openfda_drug_label",
                tool_input={"drug_name": "Levophed"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "cited_sources": [
                    {"source": "openfda", "id": "Levophed",
                     "title": "norepinephrine bitartrate"},
                ],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None
        assert any(s.get("source") == "openfda" for s in verdict.cited_sources)

    def test_critic_calls_icd_lookup(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.icd_lookup",
            lambda **kw: {"status": "ok", "results": [{
                "code": "I50.9",
                "title": "Heart failure, unspecified",
                "version": "10",
                "chapter": "Diseases of the circulatory system",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="icd_lookup",
                tool_input={"query": "heart failure"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "cited_sources": [
                    {"source": "icd", "id": "I50.9",
                     "title": "Heart failure, unspecified"},
                ],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None
        assert any(
            s.get("source") == "icd" and s.get("id") == "I50.9"
            for s in verdict.cited_sources
        )

    def test_critic_calls_trials_search(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.trials_search",
            lambda **kw: {"status": "ok", "results": [{
                "nct_id": "NCT04244266",
                "brief_title": "Sepsis treatment study",
                "status": "Completed",
                "conditions": ["Sepsis"],
                "phase": "Phase 3",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="trials_search",
                tool_input={"query": "sepsis"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "cited_sources": [
                    {"source": "clinicaltrials", "id": "NCT04244266",
                     "title": "Sepsis treatment study"},
                ],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None
        assert any(
            s.get("source") == "clinicaltrials" for s in verdict.cited_sources
        )

    def test_critic_calls_snomed_expand_ecl(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.snomed_expand_ecl",
            lambda **kw: {"status": "ok", "results": [{
                "concept_id": "44054006",
                "preferred_term": "Diabetes mellitus type 2",
                "fully_specified_name": "Diabetes mellitus type 2 (disorder)",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="snomed_expand_ecl",
                tool_input={"expression": "<<73211009"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "cited_sources": [
                    {"source": "snomed", "id": "44054006"},
                ],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None

    def test_critic_calls_icd_autocode(self, monkeypatch):
        monkeypatch.setattr(
            "src.conversational.critic.icd_autocode",
            lambda **kw: {"status": "ok", "results": [{
                "code": "J96.00",
                "title": "Acute respiratory failure",
                "version": "10",
                "confidence": 0.91,
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="icd_autocode",
                tool_input={"text": "patient developed respiratory failure"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "cited_sources": [{"source": "icd", "id": "J96.00"}],
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.cited_sources is not None

    def test_critic_calls_code_map(self, monkeypatch):
        """code_map produces a cross-vocab mapping. Citation tracking is
        intentionally not done for code_map (it's a mapping, not a primary
        record), so cited_sources stays None — but the verdict still
        renders cleanly."""
        monkeypatch.setattr(
            "src.conversational.critic.code_map",
            lambda **kw: {"status": "ok", "results": [{
                "source_code": "E11.9",
                "source_vocabulary": "ICD10",
                "target_code": "44054006",
                "target_vocabulary": "SNOMED",
                "target_name": "Diabetes mellitus type 2",
                "relationship": "Maps to",
            }]},
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="code_map",
                tool_input={
                    "source_vocabulary": "ICD10",
                    "source_code": "E11.9",
                    "target_vocabulary": "SNOMED",
                },
            ),
            _end_turn_response({
                "plausible": True, "severity": "info",
                "concern": None, "reference_used": "ICD10 E11.9 → SNOMED 44054006",
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.severity == "info"

    def test_phase_g_tool_unavailable_is_graceful(self, monkeypatch):
        """When an MCP backend is not configured, the tool returns
        ``unavailable``; the critic still produces a verdict (no raise)."""
        monkeypatch.setattr(
            "src.conversational.critic.snomed_search",
            lambda **kw: {
                "status": "unavailable",
                "error": "Hermes MCP not configured",
            },
        )
        client = mock_anthropic([
            _tool_use_response(
                tool_name="snomed_search",
                tool_input={"term": "anything"},
            ),
            _end_turn_response({
                "plausible": True, "severity": "info", "concern": None,
            }),
        ])
        verdict = critique(client, _make_cq(), _make_answer("text"))
        assert verdict is not None
        assert verdict.severity == "info"

    def test_all_phase_g_tools_in_critic_system_block(self):
        """Every Phase-G MCP tool is in the system block sent to the model."""
        client = mock_anthropic([
            _end_turn_response({"plausible": True, "severity": "info"}),
        ])
        critique(client, _make_cq(), _make_answer("test"))
        kwargs = client.messages.create.call_args.kwargs
        tool_names = {t["name"] for t in (kwargs.get("tools") or [])}
        for expected in (
            "snomed_search", "snomed_expand_ecl",
            "rxnorm_lookup", "code_map",
            "trials_search", "openfda_drug_label",
            "icd_lookup", "icd_autocode",
        ):
            assert expected in tool_names, f"missing {expected!r}"

    def test_critic_tool_count_is_eleven(self):
        """Sanity: the critic now has exactly the 11 health-evidence tools."""
        from src.conversational.critic import _CRITIC_TOOLS
        assert len(_CRITIC_TOOLS) == 11
        assert len(set(_CRITIC_TOOLS)) == 11  # no duplicates


class TestCriticPromptShape:
    """Phase H prompt shrink. The reference table that hard-coded
    population-typical and ICU-plausible ranges per analyte is replaced
    with: (a) a much smaller universal-impossibility floor table (the
    physics of incompatible-with-life values), and (b) explicit guidance
    to call ``loinc_reference_range`` / ``mimic_distribution_lookup``
    when range-checking. This guards the shape of those changes — the
    full text is held by the snapshot test."""

    def test_prompt_enumerates_all_critic_tools(self):
        """Every tool in _CRITIC_TOOLS is named in the system prompt so
        the model knows what's available without inspecting the tool defs."""
        from src.conversational.critic import _CRITIC_TOOLS
        for name in _CRITIC_TOOLS:
            assert name in CRITIC_SYSTEM_PROMPT, f"prompt missing tool {name!r}"

    def test_prompt_drops_icu_plausible_recall_table(self):
        """The pre-Phase-H prompt baked ICU-plausible upper bounds per
        analyte (e.g. creatinine 'up to ~10'). Phase H removes this — the
        critic looks them up instead. Marker substring guards this stays
        out."""
        # Pre-Phase-H markers (ICU-plausible ranges, recall-from-training):
        assert "up to ~10" not in CRITIC_SYSTEM_PROMPT
        assert "up to ~150" not in CRITIC_SYSTEM_PROMPT  # BUN row
        assert "up to ~135" not in CRITIC_SYSTEM_PROMPT  # lactate mg/dL row
        # Universal-impossibility floor preserved (or analogous wording):
        prompt_lower = CRITIC_SYSTEM_PROMPT.lower()
        assert (
            "biologically impossible" in prompt_lower
            or "incompatible with life" in prompt_lower
        )

    def test_prompt_directs_to_tools_for_ranges(self):
        """Critic is told to use the tools for population-typical and
        cohort-typical ranges instead of recalling them."""
        assert "loinc_reference_range" in CRITIC_SYSTEM_PROMPT
        assert "mimic_distribution_lookup" in CRITIC_SYSTEM_PROMPT

    def test_prompt_preserves_cohort_selection_principle(self):
        """The cohort-selection adjustment rule (added in f23f5bf) is the
        most important calibration concept — must survive the shrink."""
        prompt_lower = CRITIC_SYSTEM_PROMPT.lower()
        assert "cohort selection" in prompt_lower or "cohort-selection" in prompt_lower
        assert "selection bias" in prompt_lower

    def test_prompt_preserves_self_healing_loinc_section(self):
        """The self-healing LOINC suggestion path (suggested_loinc field)
        is orthogonal to the externally-grounded critic and must remain."""
        assert "suggested_loinc" in CRITIC_SYSTEM_PROMPT
        assert "correction_rationale" in CRITIC_SYSTEM_PROMPT

    def test_prompt_documents_multi_source_cited_sources_shape(self):
        """cited_sources schema in the prompt advertises the multi-source
        ``{source, id}`` shape so the model emits citations the validator
        will accept."""
        # The schema example mentions either the legacy or the new shape;
        # at minimum, multi-source citations are documented somewhere.
        assert (
            '"source"' in CRITIC_SYSTEM_PROMPT
            or "source =" in CRITIC_SYSTEM_PROMPT
            or "registry" in CRITIC_SYSTEM_PROMPT.lower()
        )

    def test_prompt_documents_cohort_param_for_mimic_lookup(self):
        """Phase H Tier D — prompt teaches the model that
        mimic_distribution_lookup accepts cohort= for natural medical
        phrases AND icd10_prefixes for arbitrary ICD-defined cohorts.
        Goal: the user types medical terminology and the model is the
        translator. The user must never need to know ICD codes."""
        # cohort= parameter explicitly mentioned for mimic_distribution_lookup
        assert "cohort" in CRITIC_SYSTEM_PROMPT
        assert "mimic_distribution_lookup" in CRITIC_SYSTEM_PROMPT

        # The prompt explicitly tells the model natural phrases work
        # (so it doesn't ask the user for canonical names).
        prompt_lower = CRITIC_SYSTEM_PROMPT.lower()
        assert (
            "natural" in prompt_lower
            or "medical phrase" in prompt_lower
            or "alias" in prompt_lower
        )

        # The icd_prefixes escape hatch is advertised so the model can
        # handle cohorts not in the registry.
        assert "icd10_prefixes" in CRITIC_SYSTEM_PROMPT

        # At least three example cohort phrases so the model has a
        # vocabulary to draw on without consulting the registry first.
        cohort_phrases_present = sum(
            1 for c in (
                "sepsis", "aki", "myocardial infarction",
                "heart failure", "ards", "pneumonia", "covid",
            )
            if c.lower() in prompt_lower
        )
        assert cohort_phrases_present >= 3, (
            f"only {cohort_phrases_present} cohort phrases in prompt"
        )
