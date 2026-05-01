"""Tests for the pre-execution SQL validator.

The validator runs between ``compile_sql`` and ``backend.execute``. It
returns ``SqlValidationVerdict | None``:
- ``None`` ⇒ proceed (any failure mode — API error, malformed JSON, etc.)
- ``verdict="pass"`` ⇒ proceed
- ``verdict="warn"`` ⇒ proceed but thread the concern into the critic
- ``verdict="block"`` ⇒ orchestrator short-circuits (no execute / answer / critic)

Critical safety property: NEVER raises. Failure-on-error is silent
(returns None) so a degraded validator can never break a turn.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    SqlValidationVerdict,
)
from src.conversational.sql_fastpath import SqlFastpathQuery
from src.conversational.sql_validator import validate_sql
from tests.test_conversational.conftest import mock_anthropic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cq(
    *,
    question: str = "average creatinine for patients over 65",
    concept_name: str = "creatinine",
    loinc_code: str | None = "2160-0",
) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question=question,
        clinical_concepts=[
            ClinicalConcept(
                name=concept_name,
                concept_type="biomarker",
                loinc_code=loinc_code,
            ),
        ],
        aggregation="mean",
        interpretation_summary=question,
    )


def _make_query(
    *,
    sql: str = (
        "SELECT AVG(l.valuenum) AS mean_value "
        "FROM labevents l JOIN d_labitems d ON l.itemid = d.itemid "
        "JOIN admissions a ON l.hadm_id = a.hadm_id "
        "WHERE l.itemid IN (?, ?, ?) AND l.valuenum IS NOT NULL"
    ),
    params: list | None = None,
    columns: list[str] | None = None,
) -> SqlFastpathQuery:
    return SqlFastpathQuery(
        sql=sql,
        params=params if params is not None else [50912, 51081, 52546],
        columns=columns if columns is not None else ["mean_value"],
    )


def _verdict_response(
    verdict: str,
    concern: str | None = None,
    suggested_fix: str | None = None,
    reference_used: str | None = None,
) -> str:
    """Build a JSON validator response for the mock client."""
    return json.dumps({
        "verdict": verdict,
        "concern": concern,
        "suggested_fix": suggested_fix,
        "reference_used": reference_used,
    })


# ---------------------------------------------------------------------------
# Happy path verdicts
# ---------------------------------------------------------------------------


class TestValidatorVerdicts:
    def test_pass_verdict(self):
        client = mock_anthropic([_verdict_response("pass")])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912, 51081, 52546],
        )
        assert result is not None
        assert result.verdict == "pass"
        assert result.concern is None

    def test_block_concept_pollution(self):
        client = mock_anthropic([_verdict_response(
            "block",
            concern="LIKE-on-label pools serum and urine creatinine",
            suggested_fix="Decomposer should emit LOINC 2160-0",
            reference_used="taxonomy:concept-pollution",
        )])
        result = validate_sql(
            client,
            _make_cq(loinc_code=None),  # no LOINC; LIKE fallback
            _make_query(
                sql="SELECT AVG(l.valuenum) FROM labevents l JOIN d_labitems d "
                    "ON l.itemid = d.itemid WHERE LOWER(d.label) LIKE '%creatinine%'",
            ),
            fallback_warning=None, resolved_itemids=None,
        )
        assert result is not None
        assert result.verdict == "block"
        assert "creatinine" in (result.concern or "").lower()
        assert result.reference_used == "taxonomy:concept-pollution"

    def test_block_agg_column_mismatch(self):
        client = mock_anthropic([_verdict_response(
            "block",
            concern="AVG over a COUNT(*) result is not meaningful",
            reference_used="taxonomy:agg-column-mismatch",
        )])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result.verdict == "block"

    def test_block_missing_join(self):
        client = mock_anthropic([_verdict_response(
            "block",
            concern="WHERE references rl.* but readmission_labels not joined",
            reference_used="taxonomy:reference-without-join",
        )])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result.verdict == "block"

    def test_warn_unit_pooling_on_fallback(self):
        """The validator should WARN (not block) on LIKE-fallback unit pooling
        so the critic's self-healing retry can fix it."""
        client = mock_anthropic([_verdict_response(
            "warn",
            concern="Lactate LIKE-fallback may pool serum vs CSF lactate",
            reference_used="taxonomy:unit-pooling-fallback",
        )])
        result = validate_sql(
            client,
            _make_cq(concept_name="lactate", loinc_code=None),
            _make_query(),
            fallback_warning="LOINC grounding failed for lactate; falling back to LIKE",
            resolved_itemids=None,
        )
        assert result.verdict == "warn"
        assert "pool" in (result.concern or "").lower()


# ---------------------------------------------------------------------------
# Failure modes — must always return None, never raise
# ---------------------------------------------------------------------------


class TestValidatorFailureModes:
    def test_returns_none_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is None

    def test_returns_none_on_timeout(self):
        client = MagicMock()
        client.messages.create.side_effect = TimeoutError("slow")
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is None

    def test_returns_none_on_malformed_json(self):
        client = mock_anthropic(["this is not JSON"])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is None

    def test_returns_none_on_schema_validation_failure(self):
        # JSON object but verdict field has invalid value
        client = mock_anthropic([json.dumps({
            "verdict": "totally-not-a-valid-verdict",
        })])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is None

    def test_returns_none_when_verdict_field_missing(self):
        client = mock_anthropic([json.dumps({"concern": "no verdict field"})])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is None


# ---------------------------------------------------------------------------
# Input forwarding & request shape
# ---------------------------------------------------------------------------


class TestValidatorRequestShape:
    def test_uses_sonnet_4_6(self):
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_system_prompt_is_cached(self):
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_tools_passed_in_v1(self):
        """v1 validator is instruction-based (no tools) for latency / cost."""
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        kwargs = client.messages.create.call_args.kwargs
        # Either tools is missing entirely, or it's an empty list.
        assert not kwargs.get("tools")

    def test_user_message_includes_sql(self):
        sentinel_sql = "SELECT very_specific_string_to_grep_for FROM x"
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client, _make_cq(),
            _make_query(sql=sentinel_sql),
            fallback_warning=None, resolved_itemids=[50912],
        )
        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert sentinel_sql in user_msg

    def test_user_message_includes_fallback_warning_when_set(self):
        client = mock_anthropic([_verdict_response("warn")])
        validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning="LOINC grounding failed for creatinine",
            resolved_itemids=None,
        )
        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "LOINC grounding failed" in user_msg

    def test_user_message_includes_question(self):
        sentinel_q = "what is the average uniquely-identifiable-question-text?"
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client,
            _make_cq(question=sentinel_q),
            _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert sentinel_q in user_msg

    def test_user_message_indicates_resolved_itemids(self):
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client, _make_cq(),
            _make_query(),
            fallback_warning=None, resolved_itemids=[50912, 51081, 52546],
        )
        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        # The user message should communicate that LOINC grounding succeeded.
        assert "50912" in user_msg or "resolved" in user_msg.lower()

    def test_no_resolved_itemids_communicated(self):
        client = mock_anthropic([_verdict_response("pass")])
        validate_sql(
            client, _make_cq(loinc_code=None),
            _make_query(),
            fallback_warning=None, resolved_itemids=None,
        )
        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        # When no LOINC grounding happened, the message should signal that
        # so the validator knows the LIKE-fallback rule applies.
        assert "no" in user_msg.lower() or "none" in user_msg.lower() or "fallback" in user_msg.lower()


# ---------------------------------------------------------------------------
# JSON extraction robustness
# ---------------------------------------------------------------------------


class TestValidatorJsonExtraction:
    def test_handles_fenced_json_block(self):
        client = mock_anthropic([
            f"Here is my verdict:\n```json\n{_verdict_response('pass')}\n```",
        ])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is not None
        assert result.verdict == "pass"

    def test_handles_bare_json_with_surrounding_prose(self):
        client = mock_anthropic([
            f"My analysis: {_verdict_response('warn', concern='soft pool')}",
        ])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        assert result is not None
        assert result.verdict == "warn"

    def test_truncates_raw_response_in_verdict(self):
        # The raw_response should be truncated if very long.
        long_response = _verdict_response("pass") + " " + ("x" * 5000)
        client = mock_anthropic([long_response])
        result = validate_sql(
            client, _make_cq(), _make_query(),
            fallback_warning=None, resolved_itemids=[50912],
        )
        if result is not None and result.raw_response is not None:
            # Defensive: should be bounded.
            assert len(result.raw_response) <= 1000


# ---------------------------------------------------------------------------
# System-prompt snapshot — drift guard
# ---------------------------------------------------------------------------


class TestSqlValidatorPromptSnapshot:
    """The committed ``sql_validator_prompt_snapshot.txt`` is the rendered
    validator system prompt verbatim. Any change to the prompt forces a
    snapshot regen in the same commit, so taxonomy edits are explicit."""

    from pathlib import Path as _Path
    SNAPSHOT_PATH = (
        _Path(__file__).parent / "fixtures" / "sql_validator_prompt_snapshot.txt"
    )

    def test_validator_prompt_matches_snapshot(self):
        from src.conversational.prompts import SQL_VALIDATOR_SYSTEM_PROMPT
        if not self.SNAPSHOT_PATH.exists():
            self.SNAPSHOT_PATH.write_text(SQL_VALIDATOR_SYSTEM_PROMPT)
            pytest.skip(
                f"Snapshot written to {self.SNAPSHOT_PATH}. Re-run to verify."
            )
        expected = self.SNAPSHOT_PATH.read_text()
        if expected != SQL_VALIDATOR_SYSTEM_PROMPT:
            diff_preview = (
                f"SQL validator prompt changed. To accept, re-run with:\n"
                f"    rm {self.SNAPSHOT_PATH}\n"
                f"    pytest {__file__}::TestSqlValidatorPromptSnapshot\n"
                f"Length: expected {len(expected)} chars, got "
                f"{len(SQL_VALIDATOR_SYSTEM_PROMPT)}."
            )
            assert expected == SQL_VALIDATOR_SYSTEM_PROMPT, diff_preview
