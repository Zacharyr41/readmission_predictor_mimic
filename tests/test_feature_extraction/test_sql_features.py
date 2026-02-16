"""Tests for DuckDB SQL feature extractors.

These tests verify that the SQL-based extractors produce equivalent results
to the SPARQL-based extractors, using the same synthetic data.
"""

import pytest
import duckdb
import pandas as pd

from src.ingestion.derived_tables import (
    create_age_table,
    create_readmission_labels,
    select_neurology_cohort,
)
from src.feature_extraction.sql_features import (
    extract_demographics_sql,
    extract_stay_features_sql,
    extract_lab_summary_sql,
    extract_vital_summary_sql,
    extract_medication_features_sql,
    extract_diagnosis_features_sql,
    extract_labels_sql,
    extract_subject_ids_sql,
    extract_temporal_features_sql,
)


@pytest.fixture
def sql_feature_db(synthetic_duckdb_with_events):
    """DuckDB with derived tables and cohort registered for SQL feature tests.

    Cohort includes:
    - hadm 101 (patient 1, stroke I639, ICU stay 1001, age 65, male)
    - hadm 103 (patient 2, stroke I630, ICU stay 1002, age 72, female)
    - hadm 106 (patient 5, stroke I639, ICU stay 1003, age 80, male, died)
    """
    conn = synthetic_duckdb_with_events

    # Create derived tables
    create_age_table(conn)
    create_readmission_labels(conn)

    # Select cohort and register as temp table
    cohort_df = select_neurology_cohort(conn, ["I63", "I61", "I60"])
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM cohort_df"
    )

    return conn


# ---------------------------------------------------------------------------
# TestExtractDemographicsSql
# ---------------------------------------------------------------------------


class TestExtractDemographicsSql:
    def test_columns(self, sql_feature_db):
        df = extract_demographics_sql(sql_feature_db)
        assert set(df.columns) >= {"hadm_id", "age", "gender_M", "gender_F"}

    def test_age_values(self, sql_feature_db):
        df = extract_demographics_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["age"] == 65

        row_103 = df[df["hadm_id"] == 103].iloc[0]
        assert row_103["age"] == 72

    def test_gender_encoding(self, sql_feature_db):
        df = extract_demographics_sql(sql_feature_db)
        # Patient 1 is male
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["gender_M"] == 1
        assert row_101["gender_F"] == 0

        # Patient 2 is female
        row_103 = df[df["hadm_id"] == 103].iloc[0]
        assert row_103["gender_M"] == 0
        assert row_103["gender_F"] == 1


# ---------------------------------------------------------------------------
# TestExtractStayFeaturesSql
# ---------------------------------------------------------------------------


class TestExtractStayFeaturesSql:
    def test_columns(self, sql_feature_db):
        df = extract_stay_features_sql(sql_feature_db)
        assert "hadm_id" in df.columns
        assert "icu_los_hours" in df.columns
        assert "num_icu_days" in df.columns

    def test_los_hours(self, sql_feature_db):
        df = extract_stay_features_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # LOS 2.9 days * 24 = 69.6 hours
        assert abs(row_101["icu_los_hours"] - 69.6) < 0.1

    def test_admission_type_dummies(self, sql_feature_db):
        df = extract_stay_features_sql(sql_feature_db)
        # hadm 101 is EMERGENCY, hadm 103 is ELECTIVE
        admission_type_cols = [c for c in df.columns if c.startswith("admission_type_")]
        assert len(admission_type_cols) >= 2

        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["admission_type_EMERGENCY"] == 1

        row_103 = df[df["hadm_id"] == 103].iloc[0]
        assert row_103["admission_type_ELECTIVE"] == 1


# ---------------------------------------------------------------------------
# TestExtractLabSummarySql
# ---------------------------------------------------------------------------


class TestExtractLabSummarySql:
    def test_columns(self, sql_feature_db):
        df = extract_lab_summary_sql(sql_feature_db)
        assert "hadm_id" in df.columns
        # Should have creatinine columns at minimum
        creatinine_cols = [c for c in df.columns if "Creatinine" in c]
        assert len(creatinine_cols) > 0

    def test_creatinine_values(self, sql_feature_db):
        df = extract_lab_summary_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # hadm 101 has 1 creatinine value of 1.2
        assert abs(row_101["Creatinine_mean"] - 1.2) < 0.01
        assert row_101["Creatinine_count"] == 1

    def test_abnormal_rate(self, sql_feature_db):
        df = extract_lab_summary_sql(sql_feature_db)
        # hadm 101 creatinine 1.2 is within ref range [0.7, 1.3] → normal
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["Creatinine_abnormal_rate"] == 0.0

        # hadm 106 creatinine 1.5 is above ref range upper 1.3 → abnormal
        row_106 = df[df["hadm_id"] == 106].iloc[0]
        assert row_106["Creatinine_abnormal_rate"] == 1.0


# ---------------------------------------------------------------------------
# TestExtractVitalSummarySql
# ---------------------------------------------------------------------------


class TestExtractVitalSummarySql:
    def test_columns(self, sql_feature_db):
        df = extract_vital_summary_sql(sql_feature_db)
        assert "hadm_id" in df.columns
        hr_cols = [c for c in df.columns if "Heart Rate" in c]
        assert len(hr_cols) > 0

    def test_heart_rate_mean(self, sql_feature_db):
        df = extract_vital_summary_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # hadm 101 has 1 HR value of 78.0
        assert abs(row_101["Heart Rate_mean"] - 78.0) < 0.01

    def test_cv_computation(self, sql_feature_db):
        df = extract_vital_summary_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # Single measurement → std=0, cv=0
        assert row_101["Heart Rate_cv"] == 0.0


# ---------------------------------------------------------------------------
# TestExtractMedicationFeaturesSql
# ---------------------------------------------------------------------------


class TestExtractMedicationFeaturesSql:
    def test_columns(self, sql_feature_db):
        df = extract_medication_features_sql(sql_feature_db)
        assert set(df.columns) >= {
            "hadm_id",
            "num_distinct_meds",
            "total_prescription_days",
            "has_prescription",
        }

    def test_with_prescription(self, sql_feature_db):
        df = extract_medication_features_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["has_prescription"] == 1
        assert row_101["num_distinct_meds"] == 1
        # Vancomycin: 2150-01-15 12:00 to 2150-01-18 12:00 = 3.0 days
        assert abs(row_101["total_prescription_days"] - 3.0) < 0.01

    def test_without_prescription(self, sql_feature_db):
        df = extract_medication_features_sql(sql_feature_db)
        row_106 = df[df["hadm_id"] == 106].iloc[0]
        assert row_106["has_prescription"] == 0
        assert row_106["num_distinct_meds"] == 0
        assert row_106["total_prescription_days"] == 0.0


# ---------------------------------------------------------------------------
# TestExtractDiagnosisFeaturesSql
# ---------------------------------------------------------------------------


class TestExtractDiagnosisFeaturesSql:
    def test_columns(self, sql_feature_db):
        df = extract_diagnosis_features_sql(sql_feature_db)
        assert "hadm_id" in df.columns
        assert "num_diagnoses" in df.columns

    def test_diagnosis_count(self, sql_feature_db):
        df = extract_diagnosis_features_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["num_diagnoses"] == 1

    def test_icd_chapter_one_hot(self, sql_feature_db):
        df = extract_diagnosis_features_sql(sql_feature_db)
        # All cohort diagnoses start with 'I' → icd_chapter_I should be 1
        icd_cols = [c for c in df.columns if c.startswith("icd_chapter_")]
        assert len(icd_cols) >= 1

        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["icd_chapter_I"] == 1


# ---------------------------------------------------------------------------
# TestExtractLabelsSql and TestExtractSubjectIdsSql
# ---------------------------------------------------------------------------


class TestExtractLabelsSql:
    def test_readmission_labels(self, sql_feature_db):
        df = extract_labels_sql(sql_feature_db)
        assert set(df.columns) >= {"hadm_id", "readmitted_30d", "readmitted_60d"}

        # hadm 101: patient 1 readmitted within 30 days (discharged 01-20, readmitted 02-10 = 21 days)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["readmitted_30d"] == 1
        assert row_101["readmitted_60d"] == 1

        # hadm 103: patient 2, no subsequent admission
        row_103 = df[df["hadm_id"] == 103].iloc[0]
        assert row_103["readmitted_30d"] == 0
        assert row_103["readmitted_60d"] == 0

    def test_excludes_deceased(self, sql_feature_db):
        """hadm 106 (patient died) should not have readmission labels."""
        df = extract_labels_sql(sql_feature_db)
        assert 106 not in df["hadm_id"].values


class TestExtractSubjectIdsSql:
    def test_subject_ids(self, sql_feature_db):
        df = extract_subject_ids_sql(sql_feature_db)
        assert set(df.columns) >= {"hadm_id", "subject_id"}

        row_101 = df[df["hadm_id"] == 101].iloc[0]
        assert row_101["subject_id"] == 1

        row_103 = df[df["hadm_id"] == 103].iloc[0]
        assert row_103["subject_id"] == 2


# ---------------------------------------------------------------------------
# TestExtractTemporalFeaturesSql
# ---------------------------------------------------------------------------


class TestExtractTemporalFeaturesSql:
    """Test SQL-based temporal feature extraction.

    Fixture data for hadm 101 (stay 1001, ICU 2150-01-15 10:00 → 2150-01-18 08:00):
        Events within ICU window:
        - Lab Creatinine @ 2150-01-16 06:00
        - Lab Sodium     @ 2150-01-16 06:00
        - Chart HR       @ 2150-01-16 08:00
        - Chart BP       @ 2150-01-16 08:00

    ICU days: DATE_DIFF('day', '2150-01-15', '2150-01-18') + 1 = 4
    Unordered pairs (C(4,2) = 6):
        Before (a.time < b.time): Creat→HR, Creat→BP, Sod→HR, Sod→BP = 4
        Meets  (a.time = b.time): Creat&Sod, HR&BP = 2
    """

    def test_columns(self, sql_feature_db):
        df = extract_temporal_features_sql(sql_feature_db)
        assert set(df.columns) >= {
            "hadm_id",
            "events_per_icu_day",
            "num_before_relations",
            "num_during_relations",
            "total_temporal_edges",
        }

    def test_events_per_icu_day(self, sql_feature_db):
        df = extract_temporal_features_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # 4 events / 4 ICU days = 1.0
        assert row_101["events_per_icu_day"] == pytest.approx(1.0)

    def test_before_relations(self, sql_feature_db):
        df = extract_temporal_features_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # 2 events at 06:00 × 2 events at 08:00 = 4 before pairs
        assert row_101["num_before_relations"] == 4

    def test_during_relations_zero_for_points(self, sql_feature_db):
        df = extract_temporal_features_sql(sql_feature_db)
        # All events are point events → num_during_relations = 0
        for hadm_id in [101, 103, 106]:
            row = df[df["hadm_id"] == hadm_id].iloc[0]
            assert row["num_during_relations"] == 0

    def test_total_temporal_edges(self, sql_feature_db):
        df = extract_temporal_features_sql(sql_feature_db)
        row_101 = df[df["hadm_id"] == 101].iloc[0]
        # 4 before + 2 meets = 6 total
        assert row_101["total_temporal_edges"] == 6

    def test_no_events_admission(self, sql_feature_db):
        """Admission with no events returns zeros for all temporal columns."""
        # Add a dummy admission to the cohort that has no events
        sql_feature_db.execute(
            "CREATE OR REPLACE TEMP TABLE cohort AS "
            "SELECT * FROM cohort "
            "UNION ALL "
            "SELECT 999 AS subject_id, 999 AS hadm_id, 999 AS stay_id"
        )
        df = extract_temporal_features_sql(sql_feature_db)
        row_999 = df[df["hadm_id"] == 999].iloc[0]
        assert row_999["events_per_icu_day"] == 0.0
        assert row_999["num_before_relations"] == 0
        assert row_999["num_during_relations"] == 0
        assert row_999["total_temporal_edges"] == 0
