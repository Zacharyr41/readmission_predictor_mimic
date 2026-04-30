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
