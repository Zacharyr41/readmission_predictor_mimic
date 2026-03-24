"""Unit tests for conversational pipeline Pydantic models."""

import pytest

from pydantic import ValidationError

from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionResult,
    PatientFilter,
    ReturnType,
    TemporalConstraint,
)


# --- ClinicalConcept ---


class TestClinicalConcept:
    def test_basic_construction(self):
        c = ClinicalConcept(name="lactate", concept_type="biomarker")
        assert c.name == "lactate"
        assert c.concept_type == "biomarker"
        assert c.attributes == []

    def test_with_attributes(self):
        c = ClinicalConcept(
            name="heart_rate", concept_type="vital", attributes=["min", "max"]
        )
        assert c.attributes == ["min", "max"]

    @pytest.mark.parametrize(
        "concept_type",
        ["biomarker", "vital", "drug", "diagnosis", "microbiology"],
    )
    def test_valid_concept_types(self, concept_type):
        c = ClinicalConcept(name="x", concept_type=concept_type)
        assert c.concept_type == concept_type

    def test_invalid_concept_type_rejected(self):
        with pytest.raises(ValidationError):
            ClinicalConcept(name="x", concept_type="invalid")

    def test_json_round_trip(self):
        c = ClinicalConcept(
            name="vancomycin", concept_type="drug", attributes=["dose"]
        )
        data = c.model_dump_json()
        c2 = ClinicalConcept.model_validate_json(data)
        assert c == c2


# --- TemporalConstraint ---


class TestTemporalConstraint:
    def test_basic(self):
        t = TemporalConstraint(relation="before", reference_event="ICU admission")
        assert t.relation == "before"
        assert t.time_window is None

    def test_with_time_window(self):
        t = TemporalConstraint(
            relation="within", reference_event="surgery", time_window="24h"
        )
        assert t.time_window == "24h"

    def test_invalid_relation_rejected(self):
        with pytest.raises(ValidationError):
            TemporalConstraint(relation="overlaps", reference_event="x")

    def test_json_round_trip(self):
        t = TemporalConstraint(
            relation="after", reference_event="discharge", time_window="7d"
        )
        assert t == TemporalConstraint.model_validate_json(t.model_dump_json())


# --- PatientFilter ---


class TestPatientFilter:
    def test_basic(self):
        f = PatientFilter(field="age", operator=">", value="65")
        assert f.field == "age"

    @pytest.mark.parametrize("op", [">", "<", "=", ">=", "<=", "contains", "in"])
    def test_valid_operators(self, op):
        f = PatientFilter(field="x", operator=op, value="1")
        assert f.operator == op

    def test_invalid_operator_rejected(self):
        with pytest.raises(ValidationError):
            PatientFilter(field="x", operator="!=", value="1")

    def test_json_round_trip(self):
        f = PatientFilter(field="gender", operator="=", value="F")
        assert f == PatientFilter.model_validate_json(f.model_dump_json())


# --- ReturnType ---


class TestReturnType:
    def test_enum_values(self):
        assert ReturnType.TEXT == "text"
        assert ReturnType.TABLE == "table"
        assert ReturnType.TEXT_AND_TABLE == "text_and_table"
        assert ReturnType.VISUALIZATION == "visualization"

    def test_string_comparison(self):
        assert ReturnType.TEXT == "text"
        assert ReturnType("table") is ReturnType.TABLE


# --- CompetencyQuestion ---


class TestCompetencyQuestion:
    def test_minimal(self):
        q = CompetencyQuestion(original_question="What is the lactate trend?")
        assert q.clinical_concepts == []
        assert q.return_type == ReturnType.TEXT
        assert q.scope == "single_patient"

    def test_full_construction(self):
        q = CompetencyQuestion(
            original_question="Compare lactate in sepsis vs non-sepsis",
            clinical_concepts=[
                ClinicalConcept(name="lactate", concept_type="biomarker")
            ],
            temporal_constraints=[
                TemporalConstraint(relation="during", reference_event="ICU stay")
            ],
            patient_filters=[
                PatientFilter(field="age", operator=">", value="18")
            ],
            aggregation="mean",
            return_type=ReturnType.TEXT_AND_TABLE,
            scope="comparison",
        )
        assert len(q.clinical_concepts) == 1
        assert q.aggregation == "mean"
        assert q.scope == "comparison"

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValidationError):
            CompetencyQuestion(
                original_question="x", scope="population"
            )

    def test_json_round_trip(self):
        q = CompetencyQuestion(
            original_question="Show vitals",
            return_type=ReturnType.TABLE,
            scope="cohort",
        )
        assert q == CompetencyQuestion.model_validate_json(q.model_dump_json())


# --- ExtractionResult ---


class TestExtractionResult:
    def test_defaults_empty(self):
        e = ExtractionResult()
        assert e.patients == []
        assert e.events == {}

    def test_with_data(self):
        e = ExtractionResult(
            patients=[{"subject_id": 1}],
            events={"biomarker": [{"itemid": 50813, "value": 2.1}]},
        )
        assert len(e.patients) == 1
        assert "biomarker" in e.events

    def test_json_round_trip(self):
        e = ExtractionResult(
            admissions=[{"hadm_id": 100}],
            icu_stays=[{"stay_id": 200}],
        )
        assert e == ExtractionResult.model_validate_json(e.model_dump_json())


# --- AnswerResult ---


class TestAnswerResult:
    def test_minimal(self):
        a = AnswerResult(text_summary="Lactate peaked at 4.2 mmol/L.")
        assert a.data_table is None
        assert a.sparql_queries_used == []

    def test_full(self):
        a = AnswerResult(
            text_summary="Results below.",
            data_table=[{"patient": 1, "lactate": 2.1}],
            table_columns=["patient", "lactate"],
            visualization_spec={"type": "line"},
            graph_stats={"nodes": 42, "edges": 100},
            sparql_queries_used=["SELECT ?s WHERE { ?s a :Patient }"],
        )
        assert len(a.data_table) == 1
        assert a.graph_stats["nodes"] == 42

    def test_json_round_trip(self):
        a = AnswerResult(
            text_summary="ok",
            graph_stats={"triples": 10},
            sparql_queries_used=["ASK { ?s ?p ?o }"],
        )
        assert a == AnswerResult.model_validate_json(a.model_dump_json())
