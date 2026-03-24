"""Streamlit chat UI for the conversational clinical analytics pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.conversational.models import AnswerResult
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


def _render_answer(answer: AnswerResult) -> None:
    """Display an AnswerResult inside the current chat message context."""
    st.markdown(answer.text_summary)

    if answer.data_table:
        st.dataframe(pd.DataFrame(answer.data_table))

    if answer.visualization_spec and answer.data_table:
        _render_plotly(answer.visualization_spec, answer.data_table)

    if answer.graph_stats or answer.sparql_queries_used:
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
