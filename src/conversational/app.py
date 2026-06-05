"""Streamlit chat UI for the conversational clinical analytics pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.conversational.models import AnswerResult, ExtractionConfig
from src.conversational.orchestrator import (
    PROGRESS_INTERPRETING,
    ConversationalPipeline,
)
from src.conversational.query_log import log_query_run

# Fixed pixel height of the scrollable chat transcript. A bounded container
# keeps the message area scrolling *within itself* (auto-anchored to the
# latest turn) instead of growing the page and jolting the viewport on every
# rerun once the conversation runs past the fold. Override with the env var
# for taller/shorter displays.
CHAT_HEIGHT = int(os.environ.get("NEUROGRAPH_CHAT_HEIGHT", "560"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NeuroGraph", layout="wide")
st.title("NeuroGraph \u2014 Conversational Clinical Analytics")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline: ConversationalPipeline | None = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Connection")

    data_source = st.radio(
        "Data source",
        options=["Local DuckDB", "BigQuery"],
        horizontal=True,
    )

    if data_source == "Local DuckDB":
        db_path = st.text_input(
            "DuckDB path",
            value="data/processed/mimiciv.duckdb",
        )
        bq_project = None
    else:
        db_path = "data/processed/mimiciv.duckdb"  # unused but required by init
        bq_project = st.text_input(
            "GCP project ID",
            value=os.environ.get("BIGQUERY_PROJECT", ""),
        )

    api_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
    )

    with st.expander("Advanced Settings"):
        batch_size = st.number_input(
            "Batch size",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
            help=(
                "Admissions per IN-clause batch sent to the database. "
                "Lower if you hit BigQuery parameter limits; higher for fewer round-trips."
            ),
        )
        max_concurrent_batches = st.number_input(
            "Concurrent batches",
            min_value=1,
            max_value=32,
            value=8,
            step=1,
            help=(
                "Phase 7b. Graph-path extraction runs this many batch "
                "fetches in parallel via a thread pool. BigQuery benefits "
                "most (network-bound); local DuckDB also benefits via "
                "per-thread cursors. Drop to 1 for sequential debugging."
            ),
        )
        cohort_strategy = st.selectbox(
            "Cohort strategy",
            options=["recent", "random"],
            help=(
                "'recent' orders admissions by admittime DESC; 'random' shuffles "
                "them. Affects batch composition but not final result set."
            ),
        )
        max_workers = st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            help="Graph build parallelism. Increase for large cohorts.",
        )

    if st.button("Connect"):
        if not api_key:
            st.error("Please provide an Anthropic API key.")
        elif data_source == "Local DuckDB" and not Path(db_path).exists():
            st.error(f"Database not found: {db_path}")
        elif data_source == "BigQuery" and not bq_project:
            st.error("Please provide a GCP project ID.")
        else:
            try:
                ontology_dir = (
                    Path(__file__).parent.parent.parent / "ontology" / "definition"
                )
                ds = "bigquery" if data_source == "BigQuery" else "local"
                st.session_state.pipeline = ConversationalPipeline(
                    db_path=Path(db_path),
                    ontology_dir=ontology_dir,
                    api_key=api_key,
                    data_source=ds,
                    bigquery_project=bq_project,
                    extraction_config=ExtractionConfig(
                        batch_size=batch_size,
                        cohort_strategy=cohort_strategy,
                        max_concurrent_batches=max_concurrent_batches,
                    ),
                    max_workers=max_workers,
                )
                st.success("Connected!")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

    if st.session_state.pipeline is not None:
        st.success("Connected")
    else:
        st.warning("Not connected")

    if st.button("New Conversation"):
        if st.session_state.pipeline is not None:
            st.session_state.pipeline.reset()
        st.session_state.messages = []
        st.rerun()

    with st.expander("About"):
        st.markdown(
            "**NeuroGraph** lets you ask natural-language clinical questions "
            "against MIMIC-IV ICU data. Questions are decomposed into "
            "structured queries, executed against an RDF knowledge graph, "
            "and answered with text summaries, data tables, and "
            "optional visualizations."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_plotly(spec: dict, data: list[dict]) -> None:
    """Render a Plotly chart from a visualization spec."""
    chart_fn = {
        "scatter": px.scatter,
        "line": px.line,
        "bar": px.bar,
        "histogram": px.histogram,
        "box": px.box,
    }.get(spec.get("chart_type", ""), None)

    if chart_fn is None:
        st.warning(f"Unsupported chart type: {spec.get('chart_type')}")
        return

    kwargs: dict = {"data_frame": pd.DataFrame(data)}
    if "x" in spec:
        kwargs["x"] = spec["x"]
    if "y" in spec:
        kwargs["y"] = spec["y"]
    if "color" in spec:
        kwargs["color"] = spec["color"]
    if "title" in spec:
        kwargs["title"] = spec["title"]

    try:
        fig = chart_fn(**kwargs)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Could not render chart.")


def _render_outlier_panel(report, *, key: str, include: bool) -> None:
    """Render the removed-outliers expander + the live include/exclude toggle.

    ``report`` is an ``OutlierReport``. The clean answer is the default; the
    checkbox flips to the precomputed *with-outliers* value/table — an instant
    re-render (no backend round-trip, no second LLM call). ``key`` is a stable
    per-message widget key so multiple answers in the chat history don't clash.
    """
    plural = "s" if report.n_removed != 1 else ""
    with st.expander(
        f"🔍 Removed {report.n_removed} impossible outlier{plural}",
        expanded=False,
    ):
        units = f" {report.units}" if report.units else ""
        st.caption(
            f"Screened out values outside the biological-possibility envelope "
            f"[{report.bound_low:g}, {report.bound_high:g}]{units} for "
            f"**{report.analyte}** "
            f"(source: {report.source or 'unknown'}). These are physiologically "
            f"impossible / data-entry errors, not high-but-real values."
        )
        if report.removed_rows:
            st.dataframe(pd.DataFrame(report.removed_rows))

    st.checkbox("Include outliers in the result", key=key)
    if include:
        if report.value_with_outliers is not None:
            units = f" {report.units}" if report.units else ""
            st.info(
                f"Including {report.n_removed} outlier{plural}, the value is "
                f"**{report.value_with_outliers:g}{units}** "
                f"(vs. the screened answer above)."
            )
        else:
            st.info(
                f"Including {report.n_removed} outlier{plural} "
                "(table reflects the unscreened values)."
            )


def _render_answer(
    answer: AnswerResult, *, is_sub: bool = False, key_prefix: str = "msg",
) -> None:
    """Display an AnswerResult inside the current chat message context.

    Phase 4 rendering:
      - ``interpretation_summary`` (when present) surfaces as an info block
        *above* the summary so the clinician can verify what was actually
        answered before reading the content.
      - ``clarifying_question`` replaces the normal body: when the pipeline
        short-circuits on ambiguity, the UI renders the follow-up question
        prominently and shows a hint that a reply is expected.

    Phase 4.5 rendering:
      - When ``sub_answers`` is set, the top-level ``text_summary`` is the
        big-question narrative. Each sub-answer is rendered below inside
        its own expander so the clinician can drill down to any component
        of the decomposition.

    ``is_sub`` is passed when called recursively for a sub-answer; it
    suppresses the per-sub query-details expander (the top-level already
    shows aggregated stats) and uses tighter visual styling.
    """
    if answer.interpretation_summary:
        label = "Sub-question interpretation" if is_sub else "Interpreting as"
        st.info(f"**{label}:** {answer.interpretation_summary}")

    if answer.clarifying_question:
        # Short-circuit: no data to render, only the question.
        st.markdown(f"**{answer.clarifying_question}**")
        st.caption("Reply with more detail and I'll re-run the analysis.")
        return

    st.markdown(answer.text_summary)

    if answer.correction_trace:
        with st.expander(
            "🔄 Self-healed answer — original attempt was rejected",
            expanded=False,
        ):
            st.markdown(
                "The critic flagged the first attempt and proposed a corrected "
                "LOINC code. We re-ran with the correction; the answer above "
                "reflects the corrected attempt. Earlier attempts are kept "
                "for transparency."
            )
            for entry in answer.correction_trace:
                st.markdown(
                    f"**Attempt {entry['attempt'] + 1}** — "
                    f"LOINC `{entry.get('loinc_used') or 'none'}`"
                )
                if entry.get("fallback_warning"):
                    st.caption(f"Fallback warning: {entry['fallback_warning']}")
                st.markdown(entry.get("text_summary") or "*(no answer text)*")
                v = entry.get("critic_verdict")
                if v:
                    st.caption(
                        f"Critic verdict: **{v.get('severity')}** — "
                        f"{v.get('concern') or '—'}"
                    )
                else:
                    st.caption("Critic verdict: *not produced (call failed or skipped)*")
                st.markdown("---")

    if answer.critic_verdict is not None:
        verdict = answer.critic_verdict
        if verdict.severity == "block":
            st.error(
                f"🛑 **Plausibility critic flagged this answer:** {verdict.concern}"
                + (f"\n\n*Reference:* {verdict.reference_used}" if verdict.reference_used else "")
            )
        elif verdict.severity == "warn":
            st.warning(
                f"⚠️ **Plausibility note:** {verdict.concern}"
                + (f"\n\n*Reference:* {verdict.reference_used}" if verdict.reference_used else "")
            )
        elif verdict.concern:  # severity=info but with non-null concern
            with st.expander("Reviewer notes"):
                st.caption(verdict.concern)

        # Externally-grounded citations: surfaced when the critic invoked
        # pubmed_search (or other future tools) to reach its verdict.
        # Renders below the severity banner so the user can audit the
        # evidence the critic relied on.
        if verdict.cited_sources:
            with st.expander(
                f"📚 Sources cited by reviewer ({len(verdict.cited_sources)})",
                expanded=False,
            ):
                for src in verdict.cited_sources:
                    title = src.get("title") or src.get("pmid") or "(untitled)"
                    url = src.get("url") or "#"
                    st.markdown(f"- [{title}]({url})")

    # Biological-impossibility screening: when outliers were removed, the
    # default view is the *clean* table; a live checkbox flips to the
    # precomputed with-outliers table. Read the checkbox state before the
    # table renders so the swap happens in the same re-run.
    report = answer.outlier_report
    has_outliers = bool(report and report.n_removed > 0)
    outlier_key = f"{key_prefix}_include_outliers"
    include_outliers = bool(
        has_outliers and st.session_state.get(outlier_key, False)
    )

    display_table = answer.data_table
    if include_outliers and report.data_table_with_outliers:
        display_table = report.data_table_with_outliers
    if display_table:
        st.dataframe(pd.DataFrame(display_table))

    # Cohort take-away (plan III-C): when the answer carries a cohort CSV,
    # offer it as a download. This is the one place the database keys
    # (subject_id / hadm_id) are exposed — as a file for downstream analysis,
    # never as something the clinician must read or type in chat.
    if answer.download_csv:
        st.download_button(
            "⬇ Download cohort (CSV)",
            data=answer.download_csv,
            file_name=answer.download_filename or "cohort.csv",
            mime="text/csv",
            key=f"{key_prefix}_cohort_download",
        )

    # Visualize the clean data only — a plot dominated by a 1e6 data-entry
    # error is uninformative regardless of the toggle.
    if answer.visualization_spec and answer.data_table:
        _render_plotly(answer.visualization_spec, answer.data_table)

    if has_outliers:
        _render_outlier_panel(
            report, key=outlier_key, include=include_outliers,
        )

    # Multi-CQ: render each sub-answer below the narrative in its own expander.
    # The top-level wraps sub-answers, not sub-answers wrap more sub-answers,
    # so we don't recurse into sub.sub_answers even if set.
    if answer.sub_answers:
        st.markdown("---")
        st.markdown("**Breakdown:**")
        for i, sub in enumerate(answer.sub_answers, 1):
            # Use the sub-CQ's original question in the header when available
            # (it's echoed on AnswerResult.interpretation_summary for context).
            header = f"Part {i}"
            with st.expander(header, expanded=(i == 1)):
                _render_answer(
                    sub, is_sub=True, key_prefix=f"{key_prefix}_sub{i}",
                )

    # Query-details expander only at the top level; aggregated across sub-CQs.
    if not is_sub and (answer.graph_stats or answer.sparql_queries_used):
        with st.expander("Query Details"):
            if answer.graph_stats:
                st.json(answer.graph_stats)
            for i, q in enumerate(answer.sparql_queries_used, 1):
                st.code(q, language="sparql")


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

# A single fixed-height container holds the whole transcript so it scrolls
# within itself. The live turn (below) renders into this same container so a
# just-submitted exchange lands inside the scroll box, consistent with how it
# re-renders from history on the next run.
chat_container = st.container(height=CHAT_HEIGHT)

with chat_container:
    for _msg_idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                _render_answer(msg["content"], key_prefix=f"msg{_msg_idx}")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

# ``st.chat_input`` must stay at the page top level (not inside the height
# container) to keep its bottom-pinned position.
if question := st.chat_input("Ask a clinical question..."):
    if st.session_state.pipeline is None:
        st.warning("Please connect to a database first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # ``st.status`` (not a bare spinner) so the pipeline can report
                # *which* stage is live. A cohort turn's BigQuery scoring can
                # block for minutes; an opaque spinner can't tell "working" from
                # "hung". ``ask`` drives the label via ``progress_callback``.
                with st.status(PROGRESS_INTERPRETING, expanded=False) as status:
                    _started = time.monotonic()

                    def _on_progress(stage: str) -> None:
                        status.update(label=stage)

                    try:
                        answer = st.session_state.pipeline.ask(
                            question, progress_callback=_on_progress
                        )
                    except Exception as exc:
                        status.update(label="Analysis failed", state="error")
                        log_query_run(
                            question,
                            duration_s=time.monotonic() - _started,
                            error=str(exc),
                        )
                        raise
                    log_query_run(
                        question,
                        duration_s=time.monotonic() - _started,
                        answer=answer,
                    )
                    # ``ask`` swallows all exceptions and returns a sentinel
                    # AnswerResult, so the ``except`` above never fires for a
                    # real pipeline failure. Drive the final state off the
                    # structured ``error`` flag instead, so an error message
                    # doesn't render under a green "complete".
                    if answer.error:
                        status.update(label="Analysis failed", state="error")
                    else:
                        status.update(
                            label="Analysis complete", state="complete"
                        )
                # Key by the index this message will occupy once appended, so
                # the history loop reuses the same widget keys on the next
                # re-run and the outlier toggle's state survives.
                _render_answer(
                    answer, key_prefix=f"msg{len(st.session_state.messages)}",
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})
