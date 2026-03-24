"""Tests for the conversational answer generation layer."""

import json
from unittest.mock import MagicMock

import pytest

from src.conversational.answerer import (
    _rename_columns,
    generate_answer,
)
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ReturnType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(*response_texts: str) -> MagicMock:
    """Create a mock Anthropic client returning *response_texts* in sequence."""
    client = MagicMock()
    responses = []
    for text in response_texts:
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        responses.append(resp)
    client.messages.create.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# TestRenameColumns
# ---------------------------------------------------------------------------


class TestRenameColumns:
    def test_known_columns_renamed(self):
        rows = [{"value": 1.1, "unit": "mg/dL", "timestamp": "2150-06-02"}]
        renamed, columns = _rename_columns(rows)
        assert renamed == [{"Value": 1.1, "Unit": "mg/dL", "Timestamp": "2150-06-02"}]
        assert columns == ["Value", "Unit", "Timestamp"]

    def test_unknown_columns_titlecased(self):
        rows = [{"someCustomField": 42, "value": 1.0}]
        renamed, columns = _rename_columns(rows)
        assert "Some Custom Field" in renamed[0]
        assert "Value" in renamed[0]
        assert renamed[0]["Value"] == 1.0
        assert renamed[0]["Some Custom Field"] == 42


# ---------------------------------------------------------------------------
# TestGenerateAnswer
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    def test_text_and_table_basic(self):
        client = _mock_client("Serum creatinine was 1.1 mg/dL.")
        cq = CompetencyQuestion(
            original_question="What is the creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            return_type=ReturnType.TEXT_AND_TABLE,
        )
        results = [{"value": 1.1, "unit": "mg/dL"}]

        answer = generate_answer(
            client, cq, results,
            graph_stats={"triples": 100},
            sparql_queries=["SELECT ?value ..."],
        )

        assert answer.text_summary == "Serum creatinine was 1.1 mg/dL."
        assert answer.data_table is not None
        assert len(answer.data_table) == 1
        assert "Value" in answer.data_table[0]
        assert answer.table_columns == ["Value", "Unit"]
        assert answer.visualization_spec is None
        assert answer.graph_stats == {"triples": 100}
        assert answer.sparql_queries_used == ["SELECT ?value ..."]

    def test_empty_results(self):
        client = _mock_client("No matching data was found for this query.")
        cq = CompetencyQuestion(
            original_question="Show hemoglobin values",
            clinical_concepts=[
                ClinicalConcept(name="hemoglobin", concept_type="biomarker"),
            ],
        )

        answer = generate_answer(client, cq, [], graph_stats={}, sparql_queries=[])

        assert answer.text_summary == "No matching data was found for this query."
        assert answer.data_table is None
        assert answer.table_columns is None
        assert answer.visualization_spec is None

    def test_visualization_makes_second_call(self):
        viz_spec = json.dumps({
            "chart_type": "scatter",
            "x": "Timestamp",
            "y": "Value",
            "title": "Creatinine over time",
        })
        client = _mock_client(
            "Creatinine trended upward over the ICU stay.",
            viz_spec,
        )
        cq = CompetencyQuestion(
            original_question="Plot creatinine over time",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            return_type=ReturnType.VISUALIZATION,
        )
        results = [
            {"value": 1.0, "timestamp": "2150-06-01"},
            {"value": 1.3, "timestamp": "2150-06-02"},
        ]

        answer = generate_answer(
            client, cq, results, graph_stats={}, sparql_queries=[],
        )

        assert client.messages.create.call_count == 2
        assert answer.visualization_spec is not None
        assert answer.visualization_spec["chart_type"] == "scatter"
        assert answer.text_summary == "Creatinine trended upward over the ICU stay."

    def test_visualization_empty_results_skips_viz(self):
        client = _mock_client("No data found.")
        cq = CompetencyQuestion(
            original_question="Plot creatinine over time",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            return_type=ReturnType.VISUALIZATION,
        )

        answer = generate_answer(
            client, cq, [], graph_stats={}, sparql_queries=[],
        )

        assert client.messages.create.call_count == 1
        assert answer.visualization_spec is None

    def test_results_truncated_for_llm(self):
        client = _mock_client("Summary of 100 creatinine values.")
        cq = CompetencyQuestion(
            original_question="Show all creatinine values",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        results = [{"value": float(i), "unit": "mg/dL"} for i in range(100)]

        generate_answer(client, cq, results, graph_stats={}, sparql_queries=[])

        # Inspect the user message sent to the LLM
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1]["messages"]
        user_content = messages[0]["content"]
        # The JSON payload should contain at most 50 rows
        parsed_rows = json.loads(
            user_content[user_content.index("["):user_content.rindex("]") + 1]
        )
        assert len(parsed_rows) <= 50

    def test_graph_stats_and_sparql_passed_through(self):
        client = _mock_client("Summary.")
        cq = CompetencyQuestion(original_question="test")
        stats = {"triples": 500, "nodes": 120}
        queries = ["PREFIX rdf: ... SELECT ...", "PREFIX rdf: ... SELECT ..."]

        answer = generate_answer(
            client, cq, [{"value": 1}],
            graph_stats=stats,
            sparql_queries=queries,
        )

        assert answer.graph_stats == stats
        assert answer.sparql_queries_used == queries
