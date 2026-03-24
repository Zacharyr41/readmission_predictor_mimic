"""Tests for the conversational data extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.conversational.extractor import (
    _BigQueryBackend,
    _DuckDBBackend,
    extract,
    extract_bigquery,
)
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


# ---------------------------------------------------------------------------
# BigQuery backend tests (mocked — no real BQ needed)
# ---------------------------------------------------------------------------


def _make_mock_bq():
    """Create a mock google.cloud.bigquery module."""
    mock_bq = MagicMock()
    mock_bq.ScalarQueryParameter = MagicMock(
        side_effect=lambda n, t, v: (n, t, v)
    )
    mock_bq.QueryJobConfig = MagicMock()
    return mock_bq


class TestBigQueryBackend:
    @patch("src.conversational.extractor._get_bigquery_module")
    def test_table_resolution(self, mock_get_bq):
        """Verify fully-qualified BigQuery table names."""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        assert backend.table("patients") == "`physionet-data.mimiciv_3_1_hosp.patients`"
        assert backend.table("icustays") == "`physionet-data.mimiciv_3_1_icu.icustays`"
        assert backend.table("labevents") == "`physionet-data.mimiciv_3_1_hosp.labevents`"
        assert backend.table("chartevents") == "`physionet-data.mimiciv_3_1_icu.chartevents`"
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_param_conversion(self, mock_get_bq):
        """Verify ? → @pN conversion and typed parameters."""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        sql = "SELECT * FROM t WHERE age > ? AND name ILIKE ?"
        converted_sql, bq_params = backend._convert_params(sql, [70, "%creatinine%"])

        assert "@p0" in converted_sql
        assert "@p1" in converted_sql
        assert "?" not in converted_sql
        assert bq_params[0] == ("p0", "INT64", 70)
        assert bq_params[1] == ("p1", "STRING", "%creatinine%")
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_extract_bigquery_generates_qualified_sql(self, mock_get_bq):
        """Full extract_bigquery call with mocked client verifies table names in SQL."""
        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq

        mock_client = mock_bq.Client.return_value
        mock_result = MagicMock()
        mock_result.result.return_value = []
        mock_client.query.return_value = mock_result

        cq = CompetencyQuestion(
            original_question="Show creatinine",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            scope="cohort",
        )
        result = extract_bigquery(cq, project="test-project")

        # With no admissions returned, we get an empty result
        assert result.patients == []
        assert result.events == {}

        # Verify at least one query was sent to BigQuery
        assert mock_client.query.called
        first_sql = mock_client.query.call_args_list[0][0][0]
        assert "physionet-data.mimiciv_3_1_hosp.admissions" in first_sql
