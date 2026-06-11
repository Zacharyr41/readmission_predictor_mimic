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

    def test_icd_codes_defaults_to_none(self):
        """Front-half OMOPHub plumbing: icd_codes is a new optional field on
        ClinicalConcept that lets the resolver carry grounded ICD codes for
        diagnosis concepts. Default is None for back-compat with existing
        constructions that omit the field."""
        c = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        assert c.icd_codes is None

    def test_icd_codes_accepts_list(self):
        c = ClinicalConcept(
            name="sepsis", concept_type="diagnosis",
            icd_codes=["A41.9", "R65.21", "A40", "A41"],
        )
        assert c.icd_codes == ["A41.9", "R65.21", "A40", "A41"]

    def test_icd_codes_round_trip(self):
        c = ClinicalConcept(
            name="sepsis", concept_type="diagnosis",
            icd_codes=["A41.9"],
        )
        c2 = ClinicalConcept.model_validate_json(c.model_dump_json())
        assert c == c2

    def test_icd_codes_rejects_empty_list(self):
        """Empty list is meaningless — use None for ungrounded. Forcing the
        distinction at the validator avoids a class of bugs where an empty
        list is treated as 'we tried but found nothing' vs. 'we never tried'."""
        with pytest.raises(ValidationError):
            ClinicalConcept(
                name="sepsis", concept_type="diagnosis", icd_codes=[],
            )

    def test_icd_codes_rejects_malformed_code(self):
        """ICD-10 codes follow a loose pattern: leading letter (not U), digit,
        digit-or-A/B, optional dot + up to 4 alphanumerics. Reject obvious
        garbage early to surface upstream resolver bugs (e.g. accidental
        round-tripping of LOINC codes through the wrong field)."""
        with pytest.raises(ValidationError):
            ClinicalConcept(
                name="sepsis", concept_type="diagnosis",
                icd_codes=["not-a-code"],
            )

    def test_icd_codes_accepts_codes_without_dot(self):
        """ICD-10 codes are commonly written both with and without the dot
        separator (A419 vs A41.9). Accept both — the SQL fast-path matches
        verbatim against MIMIC's diagnoses_icd table which stores codes
        without dots."""
        c = ClinicalConcept(
            name="sepsis", concept_type="diagnosis",
            icd_codes=["A419", "A41"],
        )
        assert c.icd_codes == ["A419", "A41"]


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

    def test_measurement_grounding_fields_default_none(self):
        """Structural filters carry no measurement grounding — byte-identical
        to before the measurement-value-filter feature."""
        f = PatientFilter(field="age", operator=">", value="65")
        assert f.measurement is None
        assert f.loinc_code is None
        assert f.sub_filters is None

    def test_lab_value_filter_carries_analyte_and_loinc(self):
        f = PatientFilter(
            field="lab_value", operator="<", value="50",
            measurement="platelet count", loinc_code="777-3",
        )
        assert f.measurement == "platelet count"
        assert f.loinc_code == "777-3"

    def test_loinc_code_format_validated(self):
        with pytest.raises(ValidationError):
            PatientFilter(
                field="lab_value", operator="<", value="50",
                measurement="platelet count", loinc_code="not-a-loinc",
            )

    def test_or_any_carries_nested_sub_filters(self):
        """The or_any composite holds child filters (a union cohort); the model
        is self-referencing and JSON round-trips."""
        f = PatientFilter(
            field="or_any", operator="in", value="any",
            sub_filters=[
                PatientFilter(field="lab_value", operator="<", value="50",
                              measurement="platelet count", loinc_code="777-3"),
                PatientFilter(field="vital_value", operator="<", value="65",
                              measurement="mean arterial pressure", loinc_code="8478-0"),
            ],
        )
        assert len(f.sub_filters) == 2
        assert f.sub_filters[1].measurement == "mean arterial pressure"
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
        assert q.return_type == ReturnType.TEXT_AND_TABLE
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


# ---------------------------------------------------------------------------
# Phase 4.5 — multi-CQ support
# ---------------------------------------------------------------------------


class TestAnswerResultSubAnswers:
    """Phase 4.5 adds ``sub_answers`` to carry per-CQ answers inside one
    top-level AnswerResult. Recursive (a sub-answer is itself an AnswerResult),
    optional, defaulting to None so single-CQ behaviour is unchanged."""

    def test_default_is_none(self):
        a = AnswerResult(text_summary="single")
        assert a.sub_answers is None

    def test_can_carry_children(self):
        from src.conversational.models import AnswerResult as AR

        parent = AR(
            text_summary="combined",
            sub_answers=[
                AR(text_summary="sub 1"),
                AR(text_summary="sub 2", interpretation_summary="echo 2"),
            ],
        )
        assert parent.sub_answers is not None
        assert len(parent.sub_answers) == 2
        assert parent.sub_answers[1].interpretation_summary == "echo 2"

    def test_json_round_trip_preserves_nesting(self):
        from src.conversational.models import AnswerResult as AR

        original = AR(
            text_summary="top",
            sub_answers=[AR(text_summary="child")],
        )
        restored = AR.model_validate_json(original.model_dump_json())
        assert restored == original
        assert restored.sub_answers[0].text_summary == "child"


class TestDecompositionResult:
    """Phase 4.5: the decomposer returns ``DecompositionResult`` — a (narrative,
    list[CompetencyQuestion]) wrapper. Single-CQ decompositions set narrative
    to None and list length 1; big-question decompositions carry both."""

    def test_single_cq_case(self):
        from src.conversational.models import DecompositionResult

        cq = CompetencyQuestion(original_question="avg creatinine")
        d = DecompositionResult(competency_questions=[cq])
        assert d.narrative is None
        assert len(d.competency_questions) == 1
        assert d.is_multi is False

    def test_big_question_case(self):
        from src.conversational.models import DecompositionResult

        d = DecompositionResult(
            narrative="Break down: (1) cohort, (2) differences.",
            competency_questions=[
                CompetencyQuestion(original_question="count"),
                CompetencyQuestion(original_question="differences"),
            ],
        )
        assert d.narrative.startswith("Break down")
        assert len(d.competency_questions) == 2
        assert d.is_multi is True

    def test_empty_competency_questions_is_rejected(self):
        """A DecompositionResult must always contain at least one CQ —
        empty is never a valid state (ambiguity is modelled with a single CQ
        whose ``clarifying_question`` is set)."""
        from src.conversational.models import DecompositionResult

        with pytest.raises(ValidationError):
            DecompositionResult(competency_questions=[])

    def test_json_round_trip(self):
        from src.conversational.models import DecompositionResult

        d = DecompositionResult(
            narrative="a short narrative",
            competency_questions=[
                CompetencyQuestion(
                    original_question="q1",
                    interpretation_summary="i1",
                ),
                CompetencyQuestion(
                    original_question="q2",
                    clarifying_question="unclear",
                ),
            ],
        )
        restored = DecompositionResult.model_validate_json(d.model_dump_json())
        assert restored == d
