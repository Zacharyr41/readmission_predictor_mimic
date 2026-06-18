"""Tier 1 (cheap, ungated) — corrected-query proposal + one-click batch re-run.

Pre-seeds ``session_state["messages"]`` with answers that carry critic-driven
``suggested_correction``s and injects a stub pipeline whose
``run_corrected_queries`` returns a canned corrected answer. No real pipeline,
no Anthropic API, no DuckDB — fast UI-wiring coverage:

  1. a flagged answer shows the corrected SQL + rationale,
  2. a SINGLE turn-level "Run corrected quer{y/ies}" button (not one per part),
  3. clicking it calls ``run_corrected_queries`` with ALL of the turn's
     corrections and appends one combined corrected turn,
  4. a multi-part turn with a correction under each part still shows exactly
     ONE button that batches both.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from src.conversational.models import AnswerResult, SqlCorrection

_APP = str(
    Path(__file__).resolve().parents[2] / "src" / "conversational" / "app.py"
)

_SQL_1 = "SELECT COUNT(DISTINCT di.hadm_id) AS count FROM diagnoses_icd di WHERE di.icd_code IN ('2762','E872')"
_SQL_2 = "SELECT COUNT(DISTINCT di.hadm_id) AS count FROM diagnoses_icd di WHERE di.icd_code IN ('2762')"


def _correction(corrected_sql: str, q: str = "q") -> SqlCorrection:
    return SqlCorrection(
        corrected_sql=corrected_sql,
        rationale="Match the Acidosis ICD codes directly instead of a LIKE on a title that doesn't exist.",
        original_question=q,
        interpretation_summary="Count of acidosis admissions.",
        aggregation="count",
    )


def _flagged_single() -> AnswerResult:
    return AnswerResult(
        text_summary="A count of 0 is implausible.",
        sparql_queries_used=["SELECT ... LIKE '%high anion gap%'"],
        suggested_correction=_correction(_SQL_1),
    )


def _flagged_multi() -> AnswerResult:
    """A multi-part turn (Part 1 + Part 2), each part carrying a correction."""
    sub1 = AnswerResult(
        text_summary="Part 1: 0 (implausible).",
        sparql_queries_used=["SELECT ... LIKE '%high anion gap%'"],
        suggested_correction=_correction(_SQL_1),
    )
    sub2 = AnswerResult(
        text_summary="Part 2: 0 (implausible).",
        sparql_queries_used=["SELECT ... LIKE '%non-high anion gap%'"],
        suggested_correction=_correction(_SQL_2),
    )
    return AnswerResult(text_summary="Multi-part answer:", sub_answers=[sub1, sub2])


def _seed(at: AppTest, mock_pipeline, answer: AnswerResult) -> AppTest:
    at.session_state["pipeline"] = mock_pipeline
    at.session_state["messages"] = [{"role": "assistant", "content": answer}]
    return at.run()


def _run_buttons(at: AppTest):
    return [b for b in at.button if (b.key or "").endswith("_run_correction")]


def test_single_flagged_shows_sql_and_one_button():
    at = _seed(AppTest.from_file(_APP, default_timeout=30), MagicMock(), _flagged_single())
    assert not at.exception, at.exception
    assert any("Suggested corrected quer" in (e.label or "") for e in at.expander)
    assert any("di.icd_code IN" in (c.value or "") for c in at.code)
    assert len(_run_buttons(at)) == 1, [b.key for b in at.button]


def test_clicking_single_runs_batch_of_one():
    mock_pipeline = MagicMock()
    corrected = AnswerResult(text_summary="8,308 acidosis admissions.")
    mock_pipeline.run_corrected_queries.return_value = corrected

    at = _seed(AppTest.from_file(_APP, default_timeout=30), mock_pipeline, _flagged_single())
    _run_buttons(at)[0].click().run()
    assert not at.exception, at.exception

    # The batch entrypoint ran once with the single correction.
    assert mock_pipeline.run_corrected_queries.call_count == 1
    passed = mock_pipeline.run_corrected_queries.call_args.args[0]
    assert isinstance(passed, list) and len(passed) == 1
    assert passed[0].corrected_sql == _SQL_1

    msgs = at.session_state["messages"]
    assert len(msgs) == 3 and msgs[2]["content"] is corrected
    assert any("8,308 acidosis admissions" in (m.value or "") for m in at.markdown)


def test_multi_part_turn_has_one_button_batching_all_parts():
    mock_pipeline = MagicMock()
    combined = AnswerResult(
        text_summary="Re-ran 2 corrected queries:",
        sub_answers=[
            AnswerResult(text_summary="Part 1 corrected: 8,000"),
            AnswerResult(text_summary="Part 2 corrected: 300"),
        ],
    )
    mock_pipeline.run_corrected_queries.return_value = combined

    at = _seed(AppTest.from_file(_APP, default_timeout=30), mock_pipeline, _flagged_multi())
    assert not at.exception, at.exception

    # BOTH corrected SQLs are visible (one per part)...
    code_blob = " ".join(c.value or "" for c in at.code)
    assert _SQL_1 in code_blob and _SQL_2 in code_blob
    # ...but there is exactly ONE run button for the whole turn.
    buttons = _run_buttons(at)
    assert len(buttons) == 1, [b.key for b in at.button]

    buttons[0].click().run()
    assert not at.exception, at.exception
    # It batched BOTH corrections in one call.
    passed = mock_pipeline.run_corrected_queries.call_args.args[0]
    assert [c.corrected_sql for c in passed] == [_SQL_1, _SQL_2]
    # One combined corrected turn was appended.
    msgs = at.session_state["messages"]
    assert msgs[-1]["content"] is combined


def test_no_correction_no_button():
    at = _seed(
        AppTest.from_file(_APP, default_timeout=30), MagicMock(),
        AnswerResult(text_summary="A normal answer with no correction."),
    )
    assert not at.exception, at.exception
    assert _run_buttons(at) == []
