"""Tests for the conversational data extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pydantic import ValidationError

from src.conversational.extractor import (
    _BigQueryBackend,
    _DuckDBBackend,
    _fetch_admissions,
    _get_filtered_hadm_ids,
    extract,
    extract_bigquery,
)
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionConfig,
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


class TestBackendIlike:
    def test_duckdb_uses_ilike(self):
        assert "ILIKE" in _DuckDBBackend.ilike("col")

    def test_bigquery_uses_lower_like(self):
        result = _BigQueryBackend.ilike("col")
        assert "ILIKE" not in result
        assert "LOWER" in result
        assert "LIKE" in result


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


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------


class TestExtractionConfig:
    def test_default_values(self):
        cfg = ExtractionConfig()
        assert cfg.max_cohort_size == 500
        assert cfg.cohort_strategy == "recent"

    def test_custom_values(self):
        cfg = ExtractionConfig(max_cohort_size=1000, cohort_strategy="random")
        assert cfg.max_cohort_size == 1000
        assert cfg.cohort_strategy == "random"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionConfig(cohort_strategy="invalid")


# ---------------------------------------------------------------------------
# Readmission filters
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_db_path_with_readmission(tmp_path, synthetic_duckdb_with_events):
    """Extend synthetic DuckDB with readmission_labels table."""
    conn = synthetic_duckdb_with_events
    conn.execute("""
        CREATE TABLE readmission_labels (
            subject_id INTEGER,
            hadm_id INTEGER,
            dischtime TIMESTAMP,
            next_admittime TIMESTAMP,
            days_to_readmission INTEGER,
            readmitted_30d INTEGER,
            readmitted_60d INTEGER
        )
    """)
    # Patient 1, hadm 101: readmitted within 30d (next admit 26 days later)
    # Patient 1, hadm 102: not readmitted
    # Patient 2, hadm 103: readmitted within 60d but not 30d
    # Patient 3, hadm 104: not readmitted
    # Patient 4, hadm 105: not readmitted
    # Patient 5, hadm 106: not readmitted
    conn.execute("""
        INSERT INTO readmission_labels VALUES
        (1, 101, '2150-01-20 14:00:00', '2150-02-10 10:00:00', 21, 1, 1),
        (1, 102, '2150-02-15 12:00:00', NULL, NULL, 0, 0),
        (2, 103, '2151-03-10 16:00:00', '2151-04-25 08:00:00', 46, 0, 1),
        (3, 104, '2152-05-25 10:00:00', NULL, NULL, 0, 0),
        (4, 105, '2150-07-05 11:00:00', NULL, NULL, 0, 0),
        (5, 106, '2151-04-20 08:00:00', NULL, NULL, 0, 0)
    """)
    conn.close()
    return tmp_path / "test.duckdb"


class TestReadmissionFilters:
    def test_readmitted_30d_filter(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="readmitted_30d", operator="=", value="1")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        # Only hadm 101 has readmitted_30d=1
        assert set(hadm_ids) == {101}

    def test_readmitted_60d_filter(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="readmitted_60d", operator="=", value="1")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        # hadm 101 and 103 have readmitted_60d=1
        assert set(hadm_ids) == {101, 103}

    def test_readmitted_not_readmitted(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="readmitted_30d", operator="=", value="0")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        assert set(hadm_ids) == {102, 103, 104, 105, 106}

    def test_unknown_filter_skipped_gracefully(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="invented_field", operator="=", value="x")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        # Unknown filter skipped → all 6 admissions returned
        assert len(hadm_ids) == 6


# ---------------------------------------------------------------------------
# Admission readmission labels in fetch
# ---------------------------------------------------------------------------


class TestFetchAdmissionsIncludesReadmission:
    def test_admissions_contain_readmission_labels(
        self, synthetic_db_path_with_readmission,
    ):
        """_fetch_admissions must include readmitted_30d/60d from readmission_labels.

        Regression: previously these were always defaulted to False by
        _augment_admission, making comparison SPARQL queries return empty.
        """
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        admissions = _fetch_admissions(backend, [101, 102, 103])
        backend.close()

        by_hadm = {a["hadm_id"]: a for a in admissions}
        # hadm 101: readmitted_30d=1, readmitted_60d=1
        assert by_hadm[101]["readmitted_30d"] == 1
        assert by_hadm[101]["readmitted_60d"] == 1
        # hadm 102: not readmitted
        assert by_hadm[102]["readmitted_30d"] == 0
        assert by_hadm[102]["readmitted_60d"] == 0
        # hadm 103: readmitted within 60d only
        assert by_hadm[103]["readmitted_30d"] == 0
        assert by_hadm[103]["readmitted_60d"] == 1

    def test_admissions_without_readmission_table_default_to_zero(
        self, synthetic_db_path,
    ):
        """When readmission_labels table doesn't exist, defaults are 0."""
        backend = _DuckDBBackend(synthetic_db_path)
        admissions = _fetch_admissions(backend, [101])
        backend.close()

        assert admissions[0].get("readmitted_30d", 0) == 0
        assert admissions[0].get("readmitted_60d", 0) == 0


# ---------------------------------------------------------------------------
# Cohort capping
# ---------------------------------------------------------------------------


class TestCohortCappingSqlCompat:
    """Ensure cohort capping SQL is valid for both DuckDB and BigQuery."""

    def test_order_by_admittime_included_in_select(
        self, synthetic_db_path_with_readmission,
    ):
        """ORDER BY admittime must reference a column in the SELECT list.

        BigQuery rejects ORDER BY on a column not in SELECT DISTINCT.
        Regression test for: google.api_core.exceptions.BadRequest 400.
        """
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        cfg = ExtractionConfig(max_cohort_size=3, cohort_strategy="recent")
        # Should not raise — admittime must be in SELECT or use a subquery
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=cfg)
        backend.close()
        assert len(hadm_ids) == 3


class TestCohortCapping:
    def test_cap_applied_when_exceeds_max(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        cfg = ExtractionConfig(max_cohort_size=3, cohort_strategy="recent")
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=cfg)
        backend.close()
        assert len(hadm_ids) == 3

    def test_no_cap_when_under_max(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        cfg = ExtractionConfig(max_cohort_size=500)
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=cfg)
        backend.close()
        # 6 admissions total, all returned
        assert len(hadm_ids) == 6

    def test_recent_strategy_returns_newest(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        cfg = ExtractionConfig(max_cohort_size=2, cohort_strategy="recent")
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=cfg)
        backend.close()
        # Most recent by admittime: 104 (2152-05-20), 106 (2151-04-10)
        assert set(hadm_ids) == {104, 106}

    def test_random_strategy_returns_correct_count(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        cfg = ExtractionConfig(max_cohort_size=3, cohort_strategy="random")
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=cfg)
        backend.close()
        assert len(hadm_ids) == 3

    def test_cap_with_filters(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        cfg = ExtractionConfig(max_cohort_size=2, cohort_strategy="recent")
        filters = [PatientFilter(field="readmitted_30d", operator="=", value="0")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters, config=cfg)
        backend.close()
        # 5 match readmitted_30d=0, cap to 2 most recent
        assert len(hadm_ids) == 2
