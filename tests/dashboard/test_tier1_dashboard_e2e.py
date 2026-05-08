"""Tier 1 — AppTest end-to-end smoke against the live dashboard.

Drives ``src/conversational/app.py`` via ``streamlit.testing.v1.AppTest``
with a real ``ConversationalPipeline`` injected into session_state. Tests
exercise the full path: chat input → decompose → resolve (with OMOPHub)
→ compile → execute against DuckDB → answer → critique → render.

Gated by ``RUN_LIVE_DASHBOARD=1`` because they're slow (30-90s/test) and
expensive (~$0.50/run end-to-end). The default test invocation skips
this module entirely so the cheap tiers (Tier 3 + Tier 2) stay
fast-feedback-friendly.

Per the AppTest guide §1, this framework can't render visuals — we
assert against ``at.session_state["messages"]`` (the canonical truth
location for the pipeline output) plus the element-tree shape
(``at.markdown``, ``at.warning``, etc.) for UX-level checks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


_HAS_DASHBOARD_GATE = bool(os.environ.get("RUN_LIVE_DASHBOARD"))
pytestmark = pytest.mark.skipif(
    not _HAS_DASHBOARD_GATE,
    reason="Set RUN_LIVE_DASHBOARD=1 + OMOPHUB_API_KEY + ANTHROPIC_API_KEY to run",
)


_SEPSIS_FAMILY_PREFIXES = ("A41", "R65", "A40", "A42")


def test_dashboard_initial_state_renders(at_dashboard, reporter):
    """Sanity: the script runs cleanly with no exceptions, the sidebar
    + chat input are wired, and the title block is present. Catches
    import-time / Streamlit-API-version regressions before any of the
    more expensive pipeline tests run."""
    at = at_dashboard
    at.run()

    no_exception = not at.exception
    reporter.add_assertion("Script runs without exception", no_exception)
    assert no_exception, f"App raised: {at.exception[0].value if at.exception else '?'}"

    has_title = any("NeuroGraph" in (t.value or "") for t in at.title)
    reporter.add_assertion("Title block contains 'NeuroGraph'", has_title)
    assert has_title

    has_chat_input = len(at.chat_input) >= 1
    reporter.add_assertion("Chat input widget present", has_chat_input)
    assert has_chat_input

    has_sidebar = len(at.sidebar.text_input) >= 1
    reporter.add_assertion("Sidebar text-input widgets present", has_sidebar)
    assert has_sidebar

    reporter.add_note(
        f"Sanity smoke. {len(at.title)} title block(s), "
        f"{len(at.chat_input)} chat input(s), "
        f"{len(at.sidebar.text_input)} sidebar inputs."
    )


def test_dashboard_lactate_in_sepsis_uses_in_list_grounding(
    at_dashboard, real_pipeline, reporter,
):
    """Inc 9 smoking-gun verified through the live UI.

    Original failing query: "What is the mean lactate in our sepsis
    cohort?" returned 7.99 mmol/L (LIKE-pollution); after Inc 9 should
    return ~2.42 mmol/L with ``di.icd_code IN (...)`` grounding.

    Test injects a real pipeline via ``session_state["pipeline"]`` to
    skip the sidebar Connect form, then drives via
    ``at.chat_input[0].set_value(...).run()``. Assertions read the
    AnswerResult stashed in ``session_state["messages"][-1]["content"]``.
    """
    from tests.dashboard.lib.scenarios import SMOKING_GUN_QUESTION

    at = at_dashboard
    at.session_state["pipeline"] = real_pipeline
    at.run()

    at.chat_input[0].set_value(SMOKING_GUN_QUESTION).run()
    no_exception = not at.exception
    reporter.add_assertion("Script runs without exception after chat submit", no_exception)
    assert no_exception, f"App raised: {at.exception[0].value if at.exception else '?'}"

    reporter.add_question(SMOKING_GUN_QUESTION)

    msgs = (
        at.session_state["messages"]
        if "messages" in at.session_state else []
    )
    has_answer = any(m.get("role") == "assistant" for m in msgs)
    reporter.add_assertion("Pipeline produced an assistant message", has_answer)
    assert has_answer, "no assistant message in session_state['messages']"

    answer = next(
        m["content"] for m in reversed(msgs) if m.get("role") == "assistant"
    )
    reporter.add_answer(answer.text_summary)
    for sql in (answer.sparql_queries_used or []):
        reporter.add_sql(sql, [])

    has_in_list = any(
        "di.icd_code IN (" in (sql or "")
        for sql in (answer.sparql_queries_used or [])
    )
    reporter.add_assertion(
        "SQL contains 'di.icd_code IN (' (Inc 9 grounding active)",
        has_in_list,
    )
    assert has_in_list, (
        "SQL did not contain ICD IN-list — Inc 9 grounding may be unwired"
    )

    # The answer should mention a number; loose-check that it's in a
    # plausible range. The exact mean lactate from a real DuckDB
    # depends on the dataset; use a wide range so the test isn't
    # brittle to data updates.
    text = (answer.text_summary or "").lower()
    has_plausible_value = any(
        f"{n}." in text for n in range(1, 5)
    )
    reporter.add_assertion(
        "Answer text mentions a plausible mmol/L value (1-5 range)",
        has_plausible_value,
        detail=f"answer: {(answer.text_summary or '')[:200]!r}",
    )
    assert has_plausible_value


def test_dashboard_count_diagnosis_emits_grounded_sql(
    at_dashboard, real_pipeline, reporter,
):
    """Diagnosis-count path verified through the live UI. "How many
    sepsis patients?" decomposes to a diagnosis-typed CQ
    (``concept_type='diagnosis'``); the Inc 4 wiring should emit
    ``WHERE ((di.icd_code IN (...)) OR (<existing LIKE>))``."""
    from tests.dashboard.lib.scenarios import DIAGNOSIS_COUNT_QUESTION

    at = at_dashboard
    at.session_state["pipeline"] = real_pipeline
    at.run()

    at.chat_input[0].set_value(DIAGNOSIS_COUNT_QUESTION).run()
    no_exception = not at.exception
    reporter.add_assertion("Script runs without exception after chat submit", no_exception)
    assert no_exception, f"App raised: {at.exception[0].value if at.exception else '?'}"

    reporter.add_question(DIAGNOSIS_COUNT_QUESTION)
    msgs = (
        at.session_state["messages"]
        if "messages" in at.session_state else []
    )
    answer = next(
        m["content"] for m in reversed(msgs) if m.get("role") == "assistant"
    )
    reporter.add_answer(answer.text_summary)
    for sql in (answer.sparql_queries_used or []):
        reporter.add_sql(sql, [])

    has_in_list = any(
        "di.icd_code IN (" in (sql or "")
        for sql in (answer.sparql_queries_used or [])
    )
    reporter.add_assertion(
        "Diagnosis-count SQL contains 'di.icd_code IN (' (Inc 4 grounding)",
        has_in_list,
    )
    assert has_in_list


def test_dashboard_critic_warn_renders_yellow_warning_block(
    at_dashboard, reporter,
):
    """UX regression. When an AnswerResult carries a warn-severity
    critic_verdict, the dashboard's ``_render_answer`` helper should
    render a yellow warning block. Inject a pre-canned message
    directly to avoid spending an LLM round-trip on UI testing."""
    from src.conversational.models import AnswerResult, CriticVerdict

    answer = AnswerResult(
        text_summary="Test answer with warn verdict",
        critic_verdict=CriticVerdict(
            plausible=False, severity="warn",
            concern="Test plausibility note for UX regression check",
        ),
    )
    at = at_dashboard
    at.session_state["pipeline"] = object()  # any truthy stand-in; not used
    at.session_state["messages"] = [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": answer},
    ]
    at.run()

    reporter.add_question("[UX harness] warn verdict rendering")
    reporter.add_answer(answer.text_summary)
    reporter.add_verdict(answer.critic_verdict)

    no_exception = not at.exception
    reporter.add_assertion("Script renders pre-canned warn message without exception", no_exception)
    assert no_exception, f"App raised: {at.exception[0].value if at.exception else '?'}"

    has_warning = len(at.warning) >= 1 and any(
        "Test plausibility note" in (w.value or "")
        for w in at.warning
    )
    reporter.add_assertion(
        "Yellow warning block contains the critic concern text",
        has_warning,
        detail=(
            f"n_warnings={len(at.warning)}, "
            f"values={[w.value[:80] for w in at.warning]!r}"
        ),
    )
    assert has_warning


def test_dashboard_query_details_expander_shows_sql(
    at_dashboard, reporter,
):
    """UX regression. The ``Query Details`` expander should expose the
    emitted SQL to the user via ``st.code(sql, language='sparql')``.
    Inject a pre-canned message with ``sparql_queries_used`` populated
    and confirm ``at.code`` (or matching markdown) carries the SQL."""
    from src.conversational.models import AnswerResult

    expected_sql = (
        "SELECT COUNT(*) FROM diagnoses_icd di "
        "WHERE di.icd_code IN ('A41.9', 'R65.21')"
    )
    answer = AnswerResult(
        text_summary="Test answer for SQL exposure check",
        sparql_queries_used=[expected_sql],
    )
    at = at_dashboard
    at.session_state["pipeline"] = object()
    at.session_state["messages"] = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": answer},
    ]
    at.run()

    reporter.add_question("[UX harness] Query Details expander")
    reporter.add_answer(answer.text_summary)
    reporter.add_sql(expected_sql, [])

    no_exception = not at.exception
    reporter.add_assertion("Script renders without exception", no_exception)
    assert no_exception, f"App raised: {at.exception[0].value if at.exception else '?'}"

    # ``st.code`` lands in at.code; some Streamlit versions also surface
    # it via at.markdown (the cheat sheet flags Query Details / expander
    # content under the limitations section). Accept either.
    code_carries_sql = any(
        expected_sql in (c.value or "") for c in at.code
    )
    md_carries_sql = any(
        expected_sql in (m.value or "") for m in at.markdown
    )
    has_sql_exposed = code_carries_sql or md_carries_sql
    reporter.add_assertion(
        "Emitted SQL is exposed via st.code or st.markdown",
        has_sql_exposed,
        detail=(
            f"n_code={len(at.code)} carries={code_carries_sql}, "
            f"n_md={len(at.markdown)} carries={md_carries_sql}"
        ),
    )
    assert has_sql_exposed
