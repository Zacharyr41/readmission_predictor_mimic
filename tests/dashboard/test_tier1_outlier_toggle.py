"""Tier 1 (cheap, ungated) — outlier expander + include/exclude toggle.

Unlike the live Tier-1 suite (``test_tier1_dashboard_e2e.py``, gated on
``RUN_LIVE_DASHBOARD`` because it drives a real pipeline), this test
pre-seeds ``session_state["messages"]`` with a hand-built ``AnswerResult``
that already carries an ``OutlierReport`` and asserts the *render* tree.
No pipeline, no Anthropic API, no DuckDB — so it runs in the default
invocation and gives fast-feedback regression coverage for the UI wiring:

  1. the "Removed N impossible outlier(s)" expander + provenance caption,
  2. the "Include outliers in the result" checkbox, and
  3. the live swap: checking the box flips the displayed value from the
     clean mean to the precomputed with-outliers mean, with a templated
     note — no backend round-trip.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.conversational.models import AnswerResult, OutlierReport

_APP = str(
    Path(__file__).resolve().parents[2] / "src" / "conversational" / "app.py"
)

# Clean mean (poison excluded) vs. polluted mean (1e6 included).
_CLEAN_MEAN = 5.233
_POLLUTED_MEAN = 250003.925


def _seed_answer() -> AnswerResult:
    report = OutlierReport(
        analyte="lactate",
        bound_low=0.0,
        bound_high=40.0,
        units="mmol/L",
        source="seed:literature",
        method="biological_limits",
        n_removed=1,
        n_total=4,
        removed_rows=[
            {
                "valuenum": 1_000_000.0,
                "subject_id": 1,
                "hadm_id": 101,
                "label": "Lactate",
                "valueuom": "mmol/L",
            }
        ],
        value_with_outliers=_POLLUTED_MEAN,
        data_table_with_outliers=[{"Mean Value": _POLLUTED_MEAN}],
    )
    return AnswerResult(
        text_summary="The mean lactate is 5.23 mmol/L.",
        data_table=[{"Mean Value": _CLEAN_MEAN}],
        outlier_report=report,
    )


def _run_seeded() -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=30)
    at.session_state["messages"] = [
        {"role": "assistant", "content": _seed_answer()},
    ]
    return at.run()


def test_outlier_expander_and_toggle_render():
    at = _run_seeded()
    assert not at.exception, at.exception

    # Expander labeled with the removed count.
    assert any(
        "Removed 1 impossible outlier" in (e.label or "") for e in at.expander
    ), [e.label for e in at.expander]

    # The provenance caption names the analyte and the screening envelope.
    caption_text = " ".join((c.value or "") for c in at.caption).lower()
    assert "lactate" in caption_text
    assert "biological-possibility" in caption_text

    # The include/exclude toggle is present.
    assert any(
        "Include outliers" in (c.label or "") for c in at.checkbox
    ), [c.label for c in at.checkbox]


def test_toggle_swaps_to_with_outliers_value():
    at = _run_seeded()

    # Default (unchecked): the main table shows the clean mean and there is
    # no with-outliers info note.
    assert float(at.dataframe[0].value.iloc[0, 0]) < 100.0
    assert not any("Including" in (i.value or "") for i in at.info)

    # Flip the toggle and re-run: the precomputed polluted value surfaces in
    # both the templated note and the (swapped) main table.
    ckbox = next(c for c in at.checkbox if "Include outliers" in (c.label or ""))
    ckbox.check().run()

    assert any("Including 1 outlier" in (i.value or "") for i in at.info), [
        i.value for i in at.info
    ]
    assert float(at.dataframe[0].value.iloc[0, 0]) > 1000.0
