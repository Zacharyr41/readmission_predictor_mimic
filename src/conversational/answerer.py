"""Answer generation layer for the conversational analytics pipeline.

Translates structured SPARQL query results into human-readable summaries,
data tables with clean column names, and optional Plotly visualization specs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

from src.conversational.models import AnswerResult, CompetencyQuestion, ReturnType
from src.conversational.prompts import ANSWER_GENERATION_SYSTEM_PROMPT

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS_SUMMARY = 500
_MAX_TOKENS_VIZ = 1024
_MAX_ROWS_FOR_LLM = 50

# ---------------------------------------------------------------------------
# Column rename mapping (ontology property names → human-readable)
# ---------------------------------------------------------------------------

_COLUMN_MAP: dict[str, str] = {
    "value": "Value",
    "unit": "Unit",
    "timestamp": "Timestamp",
    "mean_value": "Mean Value",
    "max_value": "Max Value",
    "min_value": "Min Value",
    "count_value": "Count",
    "median_value": "Median Value",
    "avg_value": "Average",
    "drugName": "Drug",
    "startTime": "Start Time",
    "endTime": "End Time",
    "dose": "Dose",
    "doseUnit": "Dose Unit",
    "route": "Route",
    "subjectId": "Patient ID",
    "hadmId": "Admission ID",
    "stayId": "Stay ID",
    "icdCode": "ICD Code",
    "longTitle": "Diagnosis",
    "specimenType": "Specimen",
    "organismName": "Organism",
    "losDays": "LOS (days)",
    "admissionType": "Admission Type",
    "dischargeLocation": "Discharge Location",
    "readmitted30": "Readmitted (30d)",
    "readmitted60": "Readmitted (60d)",
    "age": "Age",
    "gender": "Gender",
    "type": "Event Type",
    "count": "Count",
    "eventType": "Event Type",
    "eventLabel": "Event",
    "label": "Label",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camel_to_title(name: str) -> str:
    """Convert a camelCase or PascalCase string to Title Case with spaces."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return spaced.replace("_", " ").title()


def _rename_columns(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rename ontology column names to human-readable display names.

    Returns (renamed_rows, ordered_column_names).
    """
    if not results:
        return [], []

    renamed: list[dict[str, Any]] = []
    for row in results:
        renamed.append({
            _COLUMN_MAP.get(k, _camel_to_title(k)): v
            for k, v in row.items()
        })

    columns = list(renamed[0].keys())
    return renamed, columns


def _extract_json(text: str) -> dict | None:
    """Extract and parse a JSON object from LLM text output."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    raw = match.group(1).strip() if match else text

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _build_summary_prompt(
    cq: CompetencyQuestion,
    results: list[dict[str, Any]],
) -> str:
    """Build the user message for the text-summary API call."""
    parts = [f"Question: {cq.original_question}"]

    if cq.aggregation:
        parts.append(f"Aggregation: {cq.aggregation}")
    if cq.scope != "single_patient":
        parts.append(f"Scope: {cq.scope}")

    if results:
        truncated = results[:_MAX_ROWS_FOR_LLM]
        parts.append(f"Results ({len(results)} total rows):")
        parts.append(json.dumps(truncated, indent=2, default=str))
    else:
        parts.append("Results: No matching data was found.")

    return "\n\n".join(parts)


def _build_viz_prompt(
    cq: CompetencyQuestion,
    columns: list[str],
    sample_rows: list[dict[str, Any]],
) -> str:
    """Build the user message for the Plotly viz-spec API call."""
    return (
        f"Question: {cq.original_question}\n\n"
        f"Available columns: {columns}\n\n"
        f"Sample data (first {len(sample_rows)} rows):\n"
        f"{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '- "chart_type": one of "scatter", "line", "bar", "histogram", "box"\n'
        '- "x": column name for the x-axis\n'
        '- "y": column name for the y-axis\n'
        '- "color": (optional) column name for color grouping\n'
        '- "title": a descriptive chart title\n'
        "No explanation, just the JSON."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_answer(
    client: anthropic.Anthropic,
    cq: CompetencyQuestion,
    results: list[dict[str, Any]],
    graph_stats: dict,
    sparql_queries: list[str],
) -> AnswerResult:
    """Generate a human-readable answer from structured query results.

    Parameters
    ----------
    client:
        An initialised ``anthropic.Anthropic`` instance.
    cq:
        The structured competency question.
    results:
        Row dicts from the SPARQL reasoning engine.
    graph_stats:
        Statistics about the RDF subgraph (triples, nodes, etc.).
    sparql_queries:
        The SPARQL queries that were executed.

    Returns
    -------
    AnswerResult
        Text summary, optional data table, optional visualization spec.
    """
    # Step 1: Text summary (always)
    summary_prompt = _build_summary_prompt(cq, results)
    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS_SUMMARY,
        system=ANSWER_GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    text_summary = response.content[0].text

    # Step 2: Data table (if results non-empty)
    data_table = None
    table_columns = None
    if results:
        data_table, table_columns = _rename_columns(results)

    # Step 3: Visualization spec (if requested and data exists)
    visualization_spec = None
    if cq.return_type == ReturnType.VISUALIZATION and results and table_columns:
        viz_prompt = _build_viz_prompt(cq, table_columns, data_table[:5])
        viz_response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS_VIZ,
            system="You are a data visualization assistant. Return only valid JSON.",
            messages=[{"role": "user", "content": viz_prompt}],
        )
        visualization_spec = _extract_json(viz_response.content[0].text)

    return AnswerResult(
        text_summary=text_summary,
        data_table=data_table,
        table_columns=table_columns,
        visualization_spec=visualization_spec,
        graph_stats=graph_stats,
        sparql_queries_used=sparql_queries,
    )
