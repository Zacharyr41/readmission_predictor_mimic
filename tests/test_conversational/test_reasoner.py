"""Tests for the conversational SPARQL reasoning engine."""

from datetime import datetime
from pathlib import Path

import pytest

from src.conversational.graph_builder import build_query_graph
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionResult,
    PatientFilter,
    TemporalConstraint,
)
from src.conversational.reasoner import (
    ReasoningResult,
    build_sparql,
    reason,
    select_templates,
    TEMPLATES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


@pytest.fixture
def sample_extraction():
    """1 patient, 1 admission, 1 ICU stay (3 days), 1 event per type."""
    return ExtractionResult(
        patients=[{"subject_id": 9001, "gender": "M", "anchor_age": 60}],
        admissions=[{
            "hadm_id": 9101, "subject_id": 9001,
            "admittime": datetime(2150, 6, 1, 8, 0),
            "dischtime": datetime(2150, 6, 10, 14, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
        }],
        icu_stays=[{
            "stay_id": 9201, "hadm_id": 9101, "subject_id": 9001,
            "intime": datetime(2150, 6, 1, 10, 0),
            "outtime": datetime(2150, 6, 4, 10, 0),
            "los": 3.0,
        }],
        events={
            "biomarker": [{
                "labevent_id": 90001, "subject_id": 9001, "hadm_id": 9101,
                "itemid": 50912, "charttime": datetime(2150, 6, 2, 6, 0),
                "label": "Creatinine", "fluid": "Blood", "category": "Chemistry",
                "valuenum": 1.1, "valueuom": "mg/dL",
                "ref_range_lower": 0.7, "ref_range_upper": 1.3,
            }],
            "vital": [{
                "stay_id": 9201, "subject_id": 9001, "hadm_id": 9101,
                "itemid": 220045, "charttime": datetime(2150, 6, 1, 14, 0),
                "label": "Heart Rate", "category": "Routine Vital Signs",
                "valuenum": 82.0,
            }],
            "drug": [{
                "hadm_id": 9101, "subject_id": 9001,
                "drug": "Vancomycin", "starttime": datetime(2150, 6, 1, 12, 0),
                "stoptime": datetime(2150, 6, 3, 12, 0),
                "dose_val_rx": 1000.0, "dose_unit_rx": "mg", "route": "IV",
            }],
            "diagnosis": [{
                "hadm_id": 9101, "subject_id": 9001, "seq_num": 1,
                "icd_code": "I63.0", "icd_version": 10,
                "long_title": "Cerebral infarction due to thrombosis",
            }],
            "microbiology": [{
                "microevent_id": 95001, "subject_id": 9001, "hadm_id": 9101,
                "charttime": datetime(2150, 6, 2, 12, 0),
                "spec_type_desc": "BLOOD CULTURE",
                "org_name": "STAPHYLOCOCCUS AUREUS",
            }],
        },
    )


@pytest.fixture
def sample_graph(sample_extraction):
    """Build an RDF graph from the sample extraction."""
    graph, _stats = build_query_graph(ONTOLOGY_DIR, sample_extraction)
    return graph


# ---------------------------------------------------------------------------
# Template selection tests
# ---------------------------------------------------------------------------


class TestSelectTemplates:
    def test_biomarker_no_aggregation(self):
        cq = CompetencyQuestion(
            original_question="What are the creatinine values?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        templates = select_templates(cq)
        assert "value_with_timestamps" in templates

    def test_aggregation_mean(self):
        cq = CompetencyQuestion(
            original_question="Average creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            aggregation="mean",
        )
        templates = select_templates(cq)
        assert "aggregation_mean" in templates

    def test_aggregation_count(self):
        cq = CompetencyQuestion(
            original_question="How many creatinine measurements?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            aggregation="count",
        )
        templates = select_templates(cq)
        assert "aggregation_count" in templates

    def test_aggregation_median(self):
        cq = CompetencyQuestion(
            original_question="Median creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            aggregation="median",
        )
        templates = select_templates(cq)
        # Median not in SPARQL — falls back to value_lookup for Python post-processing
        assert "value_lookup" in templates

    def test_drug_concept(self):
        cq = CompetencyQuestion(
            original_question="What drugs were given?",
            clinical_concepts=[
                ClinicalConcept(name="Vancomycin", concept_type="drug"),
            ],
        )
        templates = select_templates(cq)
        assert "drug_lookup" in templates

    def test_diagnosis_concept(self):
        cq = CompetencyQuestion(
            original_question="Patients with stroke?",
            clinical_concepts=[
                ClinicalConcept(name="I63", concept_type="diagnosis"),
            ],
            scope="cohort",
        )
        templates = select_templates(cq)
        assert "patient_list_by_diagnosis" in templates

    def test_microbiology_concept(self):
        cq = CompetencyQuestion(
            original_question="Blood culture results?",
            clinical_concepts=[
                ClinicalConcept(name="BLOOD CULTURE", concept_type="microbiology"),
            ],
        )
        templates = select_templates(cq)
        assert "microbiology_results" in templates

    def test_comparison_scope(self):
        cq = CompetencyQuestion(
            original_question="Compare creatinine between readmitted and not?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
        )
        templates = select_templates(cq)
        assert "comparison_two_groups" in templates

    def test_no_concepts_age_filter(self):
        cq = CompetencyQuestion(
            original_question="Patients over 65?",
            patient_filters=[
                PatientFilter(field="age", operator=">", value="65"),
            ],
        )
        templates = select_templates(cq)
        assert "patient_demographics" in templates


# ---------------------------------------------------------------------------
# SPARQL building tests
# ---------------------------------------------------------------------------


class TestBuildSparql:
    def test_fills_concept_name(self):
        cq = CompetencyQuestion(
            original_question="Creatinine values?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        sparql = build_sparql("value_lookup", cq, concept_index=0)
        assert 'FILTER(STR(?label) = "Creatinine")' in sparql
        assert sparql.startswith("PREFIX")

    def test_sanitizes_quotes(self):
        cq = CompetencyQuestion(
            original_question="test",
            clinical_concepts=[
                ClinicalConcept(name='He said "hello"', concept_type="biomarker"),
            ],
        )
        sparql = build_sparql("value_lookup", cq, concept_index=0)
        # Double quotes should be escaped
        assert r'\"hello\"' in sparql

    def test_no_concept_template(self):
        cq = CompetencyQuestion(original_question="Admission details?")
        sparql = build_sparql("admission_details", cq)
        assert "PREFIX" in sparql
        assert "hasAdmissionId" in sparql


# ---------------------------------------------------------------------------
# Integration tests — execute SPARQL against a real graph
# ---------------------------------------------------------------------------


class TestReasonIntegration:
    def test_value_lookup_creatinine(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Creatinine values?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        result = reason(sample_graph, cq)
        assert isinstance(result, ReasoningResult)
        assert len(result.rows) == 1
        # Value should be approximately 1.1
        val = float(result.rows[0]["value"])
        assert abs(val - 1.1) < 0.01

    def test_value_with_timestamps(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Creatinine values?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        result = reason(sample_graph, cq)
        assert len(result.rows) == 1
        assert "timestamp" in result.rows[0]

    def test_aggregation_mean(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Average creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            aggregation="mean",
        )
        result = reason(sample_graph, cq)
        assert len(result.rows) == 1
        mean_val = float(result.rows[0]["mean_value"])
        assert abs(mean_val - 1.1) < 0.01

    def test_aggregation_count(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="How many creatinine measurements?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            aggregation="count",
        )
        result = reason(sample_graph, cq)
        assert len(result.rows) == 1
        assert int(result.rows[0]["count_value"]) == 1

    def test_diagnosis_lookup_by_icd(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Patients with stroke?",
            clinical_concepts=[
                ClinicalConcept(name="I63", concept_type="diagnosis"),
            ],
            scope="cohort",
        )
        result = reason(sample_graph, cq)
        assert len(result.rows) >= 1
        subject_ids = {int(r["subjectId"]) for r in result.rows}
        assert 9001 in subject_ids

    def test_drug_lookup_vancomycin(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Vancomycin prescriptions?",
            clinical_concepts=[
                ClinicalConcept(name="Vancomycin", concept_type="drug"),
            ],
        )
        result = reason(sample_graph, cq)
        assert len(result.rows) == 1
        assert "Vancomycin" in str(result.rows[0]["drugName"])

    def test_patient_demographics(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Patient demographics?",
            patient_filters=[
                PatientFilter(field="age", operator=">", value="50"),
            ],
        )
        result = reason(sample_graph, cq)
        assert len(result.rows) >= 1
        row = result.rows[0]
        assert int(row["age"]) == 60
        assert str(row["gender"]) == "M"

    def test_empty_result_nonexistent(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Nonexistent lab?",
            clinical_concepts=[
                ClinicalConcept(name="Unobtainium", concept_type="biomarker"),
            ],
        )
        result = reason(sample_graph, cq)
        assert result.rows == []
        assert len(result.sparql_queries) > 0

    def test_sparql_queries_populated(self, sample_graph):
        cq = CompetencyQuestion(
            original_question="Creatinine?",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
        )
        result = reason(sample_graph, cq)
        assert len(result.sparql_queries) >= 1
        assert all(q.startswith("PREFIX") for q in result.sparql_queries)
        assert len(result.template_names) >= 1


# ---------------------------------------------------------------------------
# Comparison template selection
# ---------------------------------------------------------------------------


class TestComparisonTemplateSelection:
    def test_comparison_with_field_uses_parameterized_template(self):
        """comparison_field set → selects comparison_by_field template."""
        cq = CompetencyQuestion(
            original_question="Compare creatinine by gender",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
            comparison_field="gender",
        )
        templates = select_templates(cq)
        assert "comparison_by_field" in templates

    def test_comparison_without_field_uses_readmission_default(self):
        """No comparison_field → falls back to comparison_two_groups."""
        cq = CompetencyQuestion(
            original_question="Compare creatinine readmitted vs not",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
        )
        templates = select_templates(cq)
        assert "comparison_two_groups" in templates

    def test_comparison_by_field_sparql_contains_group_property(self):
        """Built SPARQL for comparison_by_field contains the mapped property."""
        cq = CompetencyQuestion(
            original_question="Compare creatinine by gender",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
            comparison_field="gender",
        )
        sparql = build_sparql("comparison_by_field", cq)
        assert "hasGender" in sparql
        assert "GROUP BY" in sparql
        assert "Creatinine" in sparql

    def test_comparison_by_admission_type(self):
        """comparison_field=admission_type maps to hasAdmissionType."""
        cq = CompetencyQuestion(
            original_question="Compare creatinine by admission type",
            clinical_concepts=[
                ClinicalConcept(name="Creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
            comparison_field="admission_type",
        )
        sparql = build_sparql("comparison_by_field", cq)
        assert "hasAdmissionType" in sparql


# ---------------------------------------------------------------------------
# Mortality template
# ---------------------------------------------------------------------------


class TestMortalityTemplate:
    def test_outcome_concept_selects_mortality_template(self):
        cq = CompetencyQuestion(
            original_question="How many died?",
            clinical_concepts=[
                ClinicalConcept(name="mortality", concept_type="outcome"),
            ],
            aggregation="count",
        )
        templates = select_templates(cq)
        assert "mortality_count" in templates

    def test_mortality_sparql_has_expire_flag(self):
        cq = CompetencyQuestion(
            original_question="How many died?",
            clinical_concepts=[
                ClinicalConcept(name="mortality", concept_type="outcome"),
            ],
        )
        sparql = build_sparql("mortality_count", cq)
        assert "hasHospitalExpireFlag" in sparql


# ---------------------------------------------------------------------------
# LOS as queryable property
# ---------------------------------------------------------------------------


class TestLOSTemplateSelection:
    def test_no_concepts_with_aggregation_selects_los(self):
        """Aggregation without concepts defaults to LOS template."""
        cq = CompetencyQuestion(
            original_question="Average length of stay",
            aggregation="mean",
            scope="cohort",
        )
        templates = select_templates(cq)
        assert "icu_length_of_stay" in templates

    def test_no_concepts_no_aggregation_still_defaults(self):
        """No concepts, no aggregation → admission_details (existing behavior)."""
        cq = CompetencyQuestion(
            original_question="Show admission details",
            scope="cohort",
        )
        templates = select_templates(cq)
        assert "admission_details" in templates
