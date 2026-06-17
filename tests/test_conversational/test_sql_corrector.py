"""Unit tests for the critic-driven SQL corrector.

``propose_sql_correction`` runs after the plausibility critic flags a
``warn``/``block`` whose concern is a *query bug*. It asks an LLM (single
no-tool call) to rewrite the offending SQL into a directly-executable
corrected query, returning a :class:`SqlCorrection` or ``None``.

Design contract mirrored from ``sql_validator.validate_sql``: NEVER raises;
on any failure (not fixable, empty SQL, malformed JSON, API error) returns
``None`` so the answer still renders without a correction offer.
"""

from __future__ import annotations

import json

from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
    CriticVerdict,
    ReturnType,
    SqlCorrection,
)
from src.conversational.sql_corrector import propose_sql_correction
from tests.test_conversational.conftest import mock_anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The real failure from the bug report: thiazide diuretic counts split by
# in-hospital mortality among HFpEF admissions. The original SQL detected
# mortality via an ICD code and matched the drug-class label string, so it
# returned an empty result the critic correctly diagnosed as a query bug.
_ORIGINAL_SQL = (
    "SELECT CASE WHEN EXISTS (SELECT 1 FROM diagnoses_icd di2 "
    "WHERE di2.hadm_id = a.hadm_id AND di2.icd_code LIKE 'in-hospital death%') "
    "THEN 'yes' ELSE 'no' END AS group_value, COUNT(DISTINCT a.hadm_id) AS count "
    "FROM prescriptions pr JOIN admissions a ON pr.hadm_id = a.hadm_id "
    "WHERE LOWER(pr.drug) LIKE LOWER('%thiazide diuretic%') GROUP BY group_value"
)

_CORRECTED_SQL = (
    "SELECT CASE WHEN a.hospital_expire_flag = 1 THEN 'yes' ELSE 'no' END "
    "AS group_value, COUNT(DISTINCT a.hadm_id) AS count "
    "FROM prescriptions pr JOIN admissions a ON pr.hadm_id = a.hadm_id "
    "WHERE LOWER(pr.drug) LIKE LOWER('%hydrochlorothiazide%') "
    "OR LOWER(pr.drug) LIKE LOWER('%metolazone%') GROUP BY group_value"
)

_CONCERN = (
    "The SQL has two bugs producing the empty result: (1) mortality is detected "
    "via ICD code LIKE 'in-hospital death%' but in-hospital mortality in MIMIC-IV "
    "is admissions.hospital_expire_flag; (2) prescriptions.drug stores agent "
    "names, not the class label 'thiazide diuretic'."
)


def _make_cq() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question=(
            "Within each mortality group of patients with HFpEF, how many "
            "were prescribed a thiazide diuretic?"
        ),
        clinical_concepts=[ClinicalConcept(name="thiazide diuretic", concept_type="drug")],
        return_type=ReturnType.TABLE,
        scope="cohort",
        aggregation="count",
        interpretation_summary=(
            "Count of HFpEF admissions prescribed a thiazide diuretic, split by "
            "in-hospital mortality."
        ),
    )


def _make_answer(text: str = "No matching data was found.") -> AnswerResult:
    return AnswerResult(
        text_summary=text,
        sparql_queries_used=[_ORIGINAL_SQL],
    )


def _warn_verdict() -> CriticVerdict:
    return CriticVerdict(
        plausible=False,
        severity="warn",
        concern=_CONCERN,
        reference_used="MIMIC-IV: admissions.hospital_expire_flag; prescriptions.drug agent names",
    )


def _correction_response(
    *,
    fixable: bool = True,
    corrected_sql: str | None = _CORRECTED_SQL,
    rationale: str = "Use hospital_expire_flag for mortality and match agent names.",
) -> str:
    return json.dumps({
        "fixable": fixable,
        "corrected_sql": corrected_sql,
        "rationale": rationale,
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestProposesCorrection:
    def test_proposes_corrected_sql_for_buggy_query(self):
        client = mock_anthropic([_correction_response()])
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
            fallback_warning=None,
        )
        assert correction is not None
        assert isinstance(correction, SqlCorrection)
        assert correction.corrected_sql == _CORRECTED_SQL
        assert "hospital_expire_flag" in correction.corrected_sql
        assert correction.rationale

    def test_carries_rerun_context_from_cq(self):
        cq = _make_cq()
        client = mock_anthropic([_correction_response()])
        correction = propose_sql_correction(
            client, cq, _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
        )
        assert correction is not None
        assert correction.original_question == cq.original_question
        assert correction.return_type == cq.return_type
        assert correction.interpretation_summary == cq.interpretation_summary
        assert correction.aggregation == cq.aggregation

    def test_works_for_block_severity(self):
        client = mock_anthropic([_correction_response()])
        verdict = _warn_verdict()
        verdict.severity = "block"
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), verdict, _ORIGINAL_SQL,
        )
        assert correction is not None


# ---------------------------------------------------------------------------
# Failure modes — must return None, never raise
# ---------------------------------------------------------------------------


class TestReturnsNone:
    def test_not_fixable(self):
        client = mock_anthropic([_correction_response(fixable=False, corrected_sql=None)])
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
        )
        assert correction is None

    def test_empty_corrected_sql(self):
        client = mock_anthropic([_correction_response(corrected_sql="   ")])
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
        )
        assert correction is None

    def test_missing_corrected_sql_key(self):
        client = mock_anthropic([json.dumps({"fixable": True, "rationale": "x"})])
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
        )
        assert correction is None

    def test_malformed_json(self):
        client = mock_anthropic(["I think the query should use hospital_expire_flag."])
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
        )
        assert correction is None

    def test_api_error_returns_none(self):
        client = mock_anthropic([_correction_response()])
        client.messages.create.side_effect = RuntimeError("boom")
        correction = propose_sql_correction(
            client, _make_cq(), _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
        )
        assert correction is None


# ---------------------------------------------------------------------------
# Prompt assembly — the corrector must see the bug context
# ---------------------------------------------------------------------------


class TestUserMessage:
    def _sent_user_message(self, client) -> str:
        kwargs = client.messages.create.call_args.kwargs
        content = kwargs["messages"][0]["content"]
        if isinstance(content, list):  # tolerate block-list content shape
            return "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        return content

    def test_message_includes_original_sql_concern_and_interpretation(self):
        cq = _make_cq()
        client = mock_anthropic([_correction_response()])
        propose_sql_correction(
            client, cq, _make_answer(), _warn_verdict(), _ORIGINAL_SQL,
            fallback_warning="the precise assay had no values in this cohort",
        )
        msg = self._sent_user_message(client)
        assert "thiazide diuretic" in msg  # original SQL present
        assert "hospital_expire_flag" in msg  # critic concern present
        assert cq.interpretation_summary in msg
        assert "the precise assay had no values" in msg  # fallback warning threaded

    def test_message_marks_empty_result(self):
        client = mock_anthropic([_correction_response()])
        propose_sql_correction(
            client, _make_cq(), _make_answer("No matching data was found."),
            _warn_verdict(), _ORIGINAL_SQL,
        )
        msg = self._sent_user_message(client).lower()
        assert "empty" in msg or "no matching data" in msg
