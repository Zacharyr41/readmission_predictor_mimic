"""Streamlit chat UI for the conversational clinical analytics pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.conversational.models import AnswerResult, ExtractionConfig
from src.conversational.orchestrator import ConversationalPipeline

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


def _render_answer(answer: AnswerResult, *, is_sub: bool = False) -> None:
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

    if answer.data_table:
        st.dataframe(pd.DataFrame(answer.data_table))

    if answer.visualization_spec and answer.data_table:
        _render_plotly(answer.visualization_spec, answer.data_table)

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
                _render_answer(sub, is_sub=True)

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

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            _render_answer(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if question := st.chat_input("Ask a clinical question..."):
    if st.session_state.pipeline is None:
        st.warning("Please connect to a database first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = st.session_state.pipeline.ask(question)
            _render_answer(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
