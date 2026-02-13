"""Tests for derived tables — age, cohort selection, readmission labels."""

import pytest
import duckdb
import pandas as pd
from pathlib import Path

from src.ingestion.derived_tables import (
    create_age_table,
    select_neurology_cohort,
    create_readmission_labels,
)


class TestCreateAgeTable:
    """Tests for create_age_table function."""

    def test_create_age_table(self, synthetic_duckdb: duckdb.DuckDBPyConnection):
        """Verify age table exists with correct columns and computed ages."""
        create_age_table(synthetic_duckdb)

        # Verify table exists
        tables = synthetic_duckdb.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'age'"
        ).fetchall()
        assert len(tables) == 1, "age table should exist"

        # Verify columns
        columns = synthetic_duckdb.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'age'"
        ).fetchall()
        column_names = {c[0] for c in columns}
        assert {"subject_id", "hadm_id", "age"}.issubset(column_names)

        # Verify computed ages
        # anchor_age + (YEAR(admittime) - anchor_year)
        ages = synthetic_duckdb.execute(
            "SELECT subject_id, hadm_id, age FROM age ORDER BY subject_id, hadm_id"
        ).fetchall()

        # Patient 1, hadm 101: anchor_age=65, anchor_year=2150, admittime=2150 → age=65
        # Patient 1, hadm 102: anchor_age=65, anchor_year=2150, admittime=2150 → age=65
        # Patient 2, hadm 103: anchor_age=72, anchor_year=2151, admittime=2151 → age=72
        # Patient 3, hadm 104: anchor_age=58, anchor_year=2152, admittime=2152 → age=58
        # Patient 4, hadm 105: anchor_age=45, anchor_year=2150, admittime=2150 → age=45
        # Patient 5, hadm 106: anchor_age=80, anchor_year=2151, admittime=2151 → age=80
        expected = [
            (1, 101, 65),
            (1, 102, 65),
            (2, 103, 72),
            (3, 104, 58),
            (4, 105, 45),
            (5, 106, 80),
        ]
        assert ages == expected


class TestSelectNeurologyCohort:
    """Tests for select_neurology_cohort function."""

    def test_select_stroke_cohort(self, synthetic_duckdb: duckdb.DuckDBPyConnection):
        """Verify cohort selection filters by ICD codes."""
        # First create the age table (required for cohort selection)
        create_age_table(synthetic_duckdb)

        # Select stroke cohort (ICD-10 codes I63*, I61*, I60*)
        df = select_neurology_cohort(
            synthetic_duckdb,
            icd_prefixes=["I63", "I61", "I60"],
        )

        # Verify returns DataFrame with required columns
        assert isinstance(df, pd.DataFrame)
        assert {"subject_id", "hadm_id", "stay_id"}.issubset(df.columns)

        # From fixture data:
        # - Patient 1, hadm 101: I639 (stroke) - has ICU stay 1001, age 65 ✓
        # - Patient 1, hadm 102: I634 (stroke) - no ICU stay ✗
        # - Patient 2, hadm 103: I630 (stroke) - has ICU stay 1002, age 72 ✓
        # - Patient 3, hadm 104: G409 (epilepsy) - excluded
        # - Patient 4, hadm 105: I251 (cardiac) - excluded
        # - Patient 5, hadm 106: I639 (stroke) - has ICU stay 1003, age 80, died ✓
        # Note: Patient 5 has hospital_expire_flag=1 but that's for readmission labels, not cohort

        # Should return patients with stroke diagnosis AND ICU stay
        assert len(df) >= 2  # At least patients 1 and 2 with ICU stays

        # Verify stroke patients are included
        subject_ids = set(df["subject_id"].tolist())
        assert 1 in subject_ids  # Patient 1 with I639
        assert 2 in subject_ids  # Patient 2 with I630

        # Verify non-stroke patients are excluded
        assert 3 not in subject_ids  # Patient 3 with epilepsy G409
        assert 4 not in subject_ids  # Patient 4 with cardiac I251

    def test_cohort_respects_inclusion_criteria(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify cohort respects age 18-89, ICU stay >24h, first stay per admission."""
        # Add a patient outside age range and one with short ICU stay
        synthetic_duckdb.execute("""
            INSERT INTO patients VALUES
            (10, 'M', 15, 2150, NULL),
            (11, 'F', 50, 2150, NULL)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO admissions VALUES
            (110, 10, '2150-01-15 08:00:00', '2150-01-20 14:00:00', 'EMERGENCY', 'HOME', 0),
            (111, 11, '2150-01-15 08:00:00', '2150-01-20 14:00:00', 'EMERGENCY', 'HOME', 0)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO icustays VALUES
            (1010, 10, 110, '2150-01-15 10:00:00', '2150-01-18 08:00:00', 2.9),
            (1011, 11, 111, '2150-01-15 10:00:00', '2150-01-15 20:00:00', 0.4)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO diagnoses_icd VALUES
            (10, 110, 1, 'I630', 10),
            (11, 111, 1, 'I630', 10)
        """)

        create_age_table(synthetic_duckdb)
        df = select_neurology_cohort(
            synthetic_duckdb,
            icd_prefixes=["I63", "I61", "I60"],
        )

        subject_ids = set(df["subject_id"].tolist())

        # Patient 10 (age 15) should be excluded - age < 18
        assert 10 not in subject_ids, "Patient age 15 should be excluded (age < 18)"

        # Patient 11 (10h ICU stay) should be excluded - ICU stay < 24h
        assert 11 not in subject_ids, "Patient with 10h ICU stay should be excluded"

    def test_cohort_excludes_secondary_diagnoses(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify patients with stroke only as secondary diagnosis are excluded."""
        # Patient 12: primary=pneumonia (J189), secondary=stroke (I639)
        synthetic_duckdb.execute("""
            INSERT INTO patients VALUES (12, 'M', 60, 2150, NULL)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO admissions VALUES
            (112, 12, '2150-01-15 08:00:00', '2150-01-25 14:00:00', 'EMERGENCY', 'HOME', 0)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO icustays VALUES
            (1012, 12, 112, '2150-01-15 10:00:00', '2150-01-18 08:00:00', 2.9)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO diagnoses_icd VALUES
            (12, 112, 1, 'J189', 10),
            (12, 112, 2, 'I639', 10)
        """)

        create_age_table(synthetic_duckdb)
        df = select_neurology_cohort(
            synthetic_duckdb, icd_prefixes=["I63", "I61", "I60"]
        )

        subject_ids = set(df["subject_id"].tolist())
        assert 12 not in subject_ids, (
            "Patient with stroke only as secondary diagnosis should be excluded"
        )

    def test_cohort_includes_icd9_codes(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify ICD-9 stroke codes are matched via automatic mapping."""
        # 3 patients with ICD-9 stroke codes at seq_num=1
        synthetic_duckdb.execute("""
            INSERT INTO patients VALUES
            (13, 'M', 60, 2150, NULL),
            (14, 'F', 55, 2150, NULL),
            (15, 'M', 70, 2150, NULL)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO admissions VALUES
            (113, 13, '2150-01-15 08:00:00', '2150-01-25 14:00:00', 'EMERGENCY', 'HOME', 0),
            (114, 14, '2150-02-01 08:00:00', '2150-02-10 14:00:00', 'EMERGENCY', 'HOME', 0),
            (115, 15, '2150-03-01 08:00:00', '2150-03-10 14:00:00', 'EMERGENCY', 'HOME', 0)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO icustays VALUES
            (1013, 13, 113, '2150-01-15 10:00:00', '2150-01-18 08:00:00', 2.9),
            (1014, 14, 114, '2150-02-01 10:00:00', '2150-02-04 08:00:00', 2.9),
            (1015, 15, 115, '2150-03-01 10:00:00', '2150-03-04 08:00:00', 2.9)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO diagnoses_icd VALUES
            (13, 113, 1, '43391', 9),
            (14, 114, 1, '430', 9),
            (15, 115, 1, '431', 9)
        """)

        create_age_table(synthetic_duckdb)
        df = select_neurology_cohort(
            synthetic_duckdb, icd_prefixes=["I63", "I61", "I60"]
        )

        subject_ids = set(df["subject_id"].tolist())
        assert 13 in subject_ids, "ICD-9 43391 (ischemic stroke) should be included"
        assert 14 in subject_ids, "ICD-9 430 (subarachnoid hemorrhage) should be included"
        assert 15 in subject_ids, "ICD-9 431 (intracerebral hemorrhage) should be included"


class TestReadmissionLabels:
    """Tests for create_readmission_labels function."""

    def test_create_readmission_labels_30day(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify 30-day readmission labels are computed correctly."""
        # Patient 1: hadm 101 discharged Jan 20, hadm 102 admitted Feb 10 (21 days) → readmitted_30d = 1
        # Patient 2: single admission → readmitted_30d = 0
        # Patient 5: hospital_expire_flag=1 → excluded

        create_readmission_labels(synthetic_duckdb, windows=[30, 60])

        # Verify table exists
        tables = synthetic_duckdb.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'readmission_labels'"
        ).fetchall()
        assert len(tables) == 1, "readmission_labels table should exist"

        # Check Patient 1's first admission
        result = synthetic_duckdb.execute("""
            SELECT subject_id, hadm_id, readmitted_30d
            FROM readmission_labels
            WHERE hadm_id = 101
        """).fetchone()
        assert result is not None
        assert result[2] == 1, "Patient 1 hadm 101 should be readmitted_30d=1 (21 days to next)"

        # Check Patient 1's second admission
        result = synthetic_duckdb.execute("""
            SELECT subject_id, hadm_id, readmitted_30d
            FROM readmission_labels
            WHERE hadm_id = 102
        """).fetchone()
        assert result is not None
        assert result[2] == 0, "Patient 1 hadm 102 should be readmitted_30d=0 (no next admission)"

        # Check Patient 2's admission
        result = synthetic_duckdb.execute("""
            SELECT subject_id, hadm_id, readmitted_30d
            FROM readmission_labels
            WHERE hadm_id = 103
        """).fetchone()
        assert result is not None
        assert result[2] == 0, "Patient 2 should be readmitted_30d=0 (single admission)"

        # Check Patient 5 is excluded (died in hospital)
        result = synthetic_duckdb.execute("""
            SELECT * FROM readmission_labels WHERE hadm_id = 106
        """).fetchone()
        assert result is None, "Patient 5 (died in hospital) should be excluded"

    def test_create_readmission_labels_60day(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify 60-day readmission labels are computed correctly."""
        # Add a patient with readmission at 45 days
        synthetic_duckdb.execute("""
            INSERT INTO patients VALUES (20, 'M', 60, 2150, NULL)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO admissions VALUES
            (201, 20, '2150-01-01 08:00:00', '2150-01-10 14:00:00', 'EMERGENCY', 'HOME', 0),
            (202, 20, '2150-02-25 10:00:00', '2150-03-05 12:00:00', 'EMERGENCY', 'HOME', 0)
        """)

        create_readmission_labels(synthetic_duckdb, windows=[30, 60])

        # Check Patient 20's first admission (45 days to readmission)
        result = synthetic_duckdb.execute("""
            SELECT subject_id, hadm_id, readmitted_30d, readmitted_60d, days_to_readmission
            FROM readmission_labels
            WHERE hadm_id = 201
        """).fetchone()
        assert result is not None
        # 45 days: >30 so readmitted_30d=0, <=60 so readmitted_60d=1
        assert result[2] == 0, "45 days should not trigger 30-day readmission"
        assert result[3] == 1, "45 days should trigger 60-day readmission"

    def test_readmission_labels_schema(self, synthetic_duckdb: duckdb.DuckDBPyConnection):
        """Verify readmission_labels table has correct columns."""
        create_readmission_labels(synthetic_duckdb, windows=[30, 60])

        columns = synthetic_duckdb.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'readmission_labels'"
        ).fetchall()
        column_names = {c[0] for c in columns}

        expected_columns = {
            "subject_id",
            "hadm_id",
            "dischtime",
            "next_admittime",
            "days_to_readmission",
            "readmitted_30d",
            "readmitted_60d",
        }
        assert expected_columns.issubset(column_names), f"Missing columns: {expected_columns - column_names}"

    def test_readmission_excludes_hospice_discharge(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify patients discharged to hospice are excluded from readmission labels."""
        synthetic_duckdb.execute("""
            INSERT INTO patients VALUES (16, 'F', 70, 2150, NULL)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO admissions VALUES
            (116, 16, '2150-01-15 08:00:00', '2150-01-25 14:00:00', 'EMERGENCY', 'HOSPICE', 0)
        """)

        create_readmission_labels(synthetic_duckdb, windows=[30, 60])

        result = synthetic_duckdb.execute("""
            SELECT * FROM readmission_labels WHERE hadm_id = 116
        """).fetchone()
        assert result is None, "Patient discharged to hospice should be excluded"

    def test_readmission_excludes_ama_discharge(
        self, synthetic_duckdb: duckdb.DuckDBPyConnection
    ):
        """Verify patients discharged against medical advice are excluded."""
        synthetic_duckdb.execute("""
            INSERT INTO patients VALUES (17, 'M', 55, 2150, NULL)
        """)
        synthetic_duckdb.execute("""
            INSERT INTO admissions VALUES
            (117, 17, '2150-02-01 08:00:00', '2150-02-10 14:00:00', 'EMERGENCY', 'AGAINST ADVICE', 0)
        """)

        create_readmission_labels(synthetic_duckdb, windows=[30, 60])

        result = synthetic_duckdb.execute("""
            SELECT * FROM readmission_labels WHERE hadm_id = 117
        """).fetchone()
        assert result is None, "Patient discharged against medical advice should be excluded"


class TestDerivedTablesIntegration:
    """Integration tests for derived tables with real MIMIC-IV data."""

    # Path to pre-loaded MIMIC-IV DuckDB database
    MIMIC_DUCKDB_PATH = Path("/Users/zacharyrothstein/Code/readmission_predictor_mimic/data/processed/mimiciv.duckdb")

    @pytest.mark.integration
    def test_stroke_cohort_with_real_data(self):
        """Integration test with real MIMIC-IV data (uses pre-loaded database)."""
        if not self.MIMIC_DUCKDB_PATH.exists():
            pytest.skip(f"Pre-loaded MIMIC-IV database not found at {self.MIMIC_DUCKDB_PATH}")

        conn = duckdb.connect(str(self.MIMIC_DUCKDB_PATH))

        try:
            # Create age table
            create_age_table(conn)

            # Select stroke cohort
            df = select_neurology_cohort(
                conn,
                icd_prefixes=["I63", "I61", "I60"],
            )

            # Sanity check cohort size
            cohort_size = len(df)
            print(f"\nStroke cohort size: {cohort_size}")
            assert 500 <= cohort_size <= 10000, f"Unexpected cohort size: {cohort_size}"

            # Create readmission labels
            create_readmission_labels(conn, windows=[30, 60])

            # Get readmission rates for cohort
            readmission_stats = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(readmitted_30d) as readmit_30d,
                    SUM(readmitted_60d) as readmit_60d
                FROM readmission_labels r
                WHERE r.hadm_id IN (SELECT hadm_id FROM age)
            """).fetchone()

            total, readmit_30d, readmit_60d = readmission_stats
            rate_30d = readmit_30d / total * 100 if total > 0 else 0
            rate_60d = readmit_60d / total * 100 if total > 0 else 0

            print(f"Total admissions with readmission labels: {total}")
            print(f"30-day readmission rate: {rate_30d:.1f}%")
            print(f"60-day readmission rate: {rate_60d:.1f}%")

        finally:
            conn.close()
