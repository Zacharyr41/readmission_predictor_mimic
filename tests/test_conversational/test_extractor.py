"""Tests for the conversational data extractor."""

from pathlib import Path

import pytest

from src.conversational.extractor import extract
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
    TemporalConstraint,
)


@pytest.fixture
def synthetic_db_path(tmp_path, synthetic_duckdb_with_events):
    """Return the file path to the synthetic DuckDB with all event tables."""
    synthetic_duckdb_with_events.close()
    return tmp_path / "test.duckdb"


class TestExtract:
    def test_biomarker_extraction(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="What are the creatinine values?",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        assert len(result.events["biomarker"]) == 3
        assert all(
            "creatinine" in e["label"].lower()
            for e in result.events["biomarker"]
        )
        assert len(result.patients) > 0
        assert len(result.admissions) > 0

    def test_patient_filter_age(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Creatinine for patients over 70",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            patient_filters=[
                PatientFilter(field="age", operator=">", value="70")
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        # Patients over 70: subject 2 (72), subject 5 (80)
        assert len(result.events["biomarker"]) == 2
        subject_ids = {p["subject_id"] for p in result.patients}
        assert subject_ids == {2, 5}

    def test_temporal_constraint_within(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Creatinine within 24 hours of ICU admission",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            temporal_constraints=[
                TemporalConstraint(
                    relation="within",
                    reference_event="ICU admission",
                    time_window="24h",
                )
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        # Patient 1: charttime 20h after intime -> YES
        # Patient 2: charttime 24h after intime -> YES (inclusive)
        # Patient 5: charttime 34h after intime -> NO
        assert len(result.events["biomarker"]) == 2

    def test_empty_result(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Show hemoglobin values",
            clinical_concepts=[
                ClinicalConcept(name="hemoglobin", concept_type="biomarker")
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        assert result.events.get("biomarker", []) == []

    def test_multiple_concepts(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Show creatinine and heart rate",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
                ClinicalConcept(name="heart rate", concept_type="vital"),
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        assert "biomarker" in result.events
        assert "vital" in result.events
        assert len(result.events["biomarker"]) == 3
        assert len(result.events["vital"]) == 3
