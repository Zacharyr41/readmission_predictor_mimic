"""Tier 1 (cheap, ungated) — corrected-query proposal + one-click re-run.

Pre-seeds ``session_state["messages"]`` with an ``AnswerResult`` that carries
a critic-driven ``suggested_correction`` and injects a stub pipeline whose
``run_corrected_query`` returns a canned corrected answer. No real pipeline,
no Anthropic API, no DuckDB — fast-feedback coverage for the UI wiring:

  1. the "Suggested corrected query" expander shows the rationale + SQL,
  2. a "Run corrected query" button is rendered (addressable by key),
  3. clicking it runs the corrected query and appends a new assistant turn,
  4. a corrected answer that is itself still flagged surfaces a second,
     distinctly-keyed button (the chain is bounded by user clicks).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from src.conversational.models import AnswerResult, SqlCorrection

_APP = str(
    Path(__file__).resolve().parents[2] / "src" / "conversational" / "app.py"
)

_CORRECTED_SQL = (
    "SELECT CASE WHEN a.hospital_expire_flag = 1 THEN 'yes' ELSE 'no' END "
    "AS group_value, COUNT(DISTINCT a.hadm_id) AS count FROM admissions a "
    "GROUP BY group_value"
)


def _correction(corrected_sql: str = _CORRECTED_SQL) -> SqlCorrection:
    return SqlCorrection(
        corrected_sql=corrected_sql,
        rationale=(
            "Mortality should use admissions.hospital_expire_flag, not an ICD "
            "code; match thiazide agents by name."
        ),
        original_question="thiazide diuretic counts by mortality group",
        interpretation_summary="Count of HFpEF admissions on a thiazide, by mortality.",
        aggregation="count",
    )


def _flagged_answer() -> AnswerResult:
    """An answer the critic flagged, carrying a corrected-query proposal."""
    a = AnswerResult(
        text_summary="No matching data was found.",
        sparql_queries_used=["SELECT ... LIKE '%thiazide diuretic%'"],
        suggested_correction=_correction(),
    )
    return a


def _seed(at: AppTest, mock_pipeline) -> AppTest:
    at.session_state["pipeline"] = mock_pipeline
    at.session_state["messages"] = [
        {"role": "assistant", "content": _flagged_answer()},
    ]
    return at.run()


def _run_correction_buttons(at: AppTest):
    return [b for b in at.button if (b.key or "").endswith("_run_correction")]


def test_correction_expander_and_button_render():
    at = AppTest.from_file(_APP, default_timeout=30)
    at = _seed(at, MagicMock())
    assert not at.exception, at.exception

    # Expander introducing the proposed fix.
    assert any(
        "Suggested corrected query" in (e.label or "") for e in at.expander
    ), [e.label for e in at.expander]

    # The corrected SQL is exposed (st.code lands in at.code).
    assert any(
        "hospital_expire_flag" in (c.value or "") for c in at.code
    ), [c.value for c in at.code]

    # The run button is present and addressable by key.
    buttons = _run_correction_buttons(at)
    assert len(buttons) == 1, [b.key for b in at.button]


def test_clicking_button_runs_corrected_query_and_appends_turn():
    mock_pipeline = MagicMock()
    corrected = AnswerResult(
        text_summary="3 admissions expired, 5 did not.",
        data_table=[{"group_value": "yes", "count": 3}, {"group_value": "no", "count": 5}],
    )
    mock_pipeline.run_corrected_query.return_value = corrected

    at = AppTest.from_file(_APP, default_timeout=30)
    at = _seed(at, mock_pipeline)

    button = _run_correction_buttons(at)[0]
    button.click().run()
    assert not at.exception, at.exception

    # The corrected query ran exactly once, with the seeded SqlCorrection.
    assert mock_pipeline.run_corrected_query.call_count == 1
    passed = mock_pipeline.run_corrected_query.call_args.args[0]
    assert isinstance(passed, SqlCorrection)
    assert passed.corrected_sql == _CORRECTED_SQL

    # A user "run" message + the corrected assistant answer were appended.
    msgs = at.session_state["messages"]
    assert len(msgs) == 3
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] is corrected

    # The corrected answer's summary is rendered.
    assert any(
        "3 admissions expired" in (m.value or "") for m in at.markdown
    ), [m.value for m in at.markdown]


def test_still_flagged_correction_offers_second_button():
    """A corrected answer that is itself flagged surfaces a fresh, distinctly-
    keyed button — bounded by user clicks, never auto-looped."""
    mock_pipeline = MagicMock()
    corrected_still_flagged = AnswerResult(
        text_summary="Still looks off.",
        sparql_queries_used=[_CORRECTED_SQL],
        suggested_correction=_correction(
            corrected_sql="SELECT a.hospital_expire_flag, COUNT(*) FROM admissions a GROUP BY 1",
        ),
    )
    mock_pipeline.run_corrected_query.return_value = corrected_still_flagged

    at = AppTest.from_file(_APP, default_timeout=30)
    at = _seed(at, mock_pipeline)

    _run_correction_buttons(at)[0].click().run()
    assert not at.exception, at.exception

    buttons = _run_correction_buttons(at)
    keys = {b.key for b in buttons}
    # The original answer's button plus the corrected answer's new button.
    assert len(keys) == 2, keys
