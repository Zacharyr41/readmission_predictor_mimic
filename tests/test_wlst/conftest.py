"""Fixtures for WLST pipeline tests."""

import pytest
import duckdb
from datetime import datetime
from pathlib import Path

from config.settings import Settings


@pytest.fixture
def wlst_settings(tmp_path: Path) -> Settings:
    """Settings configured for WLST pipeline testing."""
    return Settings(
        mimic_iv_path=tmp_path / "mimic",
        duckdb_path=tmp_path / "test.duckdb",
        clinical_tkg_repo=tmp_path / "tkg",
        data_source="local",
        wlst_mode=True,
        wlst_gcs_threshold=8,
        wlst_observation_window_hours=48,
    )


@pytest.fixture
def wlst_duckdb(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """DuckDB with synthetic MIMIC tables for TBI/WLST testing.

    Creates a small cohort of TBI patients with GCS data, code status changes,
    and clinical events within 48h windows.
    """
    db_path = tmp_path / "wlst_test.duckdb"
    conn = duckdb.connect(str(db_path))

    # ── patients ──
    conn.execute("""
        CREATE TABLE patients (
            subject_id INTEGER PRIMARY KEY,
            gender VARCHAR,
            anchor_age INTEGER,
            anchor_year INTEGER,
            dod DATE
        )
    """)
    conn.execute("""
        INSERT INTO patients VALUES
        (10, 'M', 45, 2150, NULL),
        (20, 'F', 62, 2150, NULL),
        (30, 'M', 55, 2150, '2150-02-28'),
        (40, 'F', 70, 2150, NULL)
    """)

    # ── admissions ──
    conn.execute("""
        CREATE TABLE admissions (
            hadm_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            admittime TIMESTAMP,
            dischtime TIMESTAMP,
            admission_type VARCHAR,
            discharge_location VARCHAR,
            hospital_expire_flag INTEGER,
            deathtime TIMESTAMP,
            language VARCHAR,
            insurance VARCHAR,
            race VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO admissions VALUES
        (201, 10, '2150-01-10 08:00:00', '2150-01-25 14:00:00', 'EMERGENCY', 'HOME', 0, NULL, 'ENGLISH', 'Medicare', 'WHITE'),
        (202, 20, '2150-02-01 10:00:00', '2150-02-20 12:00:00', 'EMERGENCY', 'SNF', 0, NULL, 'SPANISH', 'Medicaid', 'HISPANIC'),
        (203, 30, '2150-02-15 06:00:00', '2150-02-28 08:00:00', 'EMERGENCY', 'DIED', 1, '2150-02-28 06:00:00', 'ENGLISH', 'Medicare', 'BLACK'),
        (204, 40, '2150-03-01 09:00:00', '2150-03-15 11:00:00', 'EMERGENCY', 'HOSPICE', 0, NULL, 'ENGLISH', 'Private', 'WHITE')
    """)

    # ── icustays (all in neuro/trauma ICU) ──
    conn.execute("""
        CREATE TABLE icustays (
            stay_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            hadm_id INTEGER,
            first_careunit VARCHAR,
            intime TIMESTAMP,
            outtime TIMESTAMP,
            los DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO icustays VALUES
        (3001, 10, 201, 'Trauma SICU (TSICU)', '2150-01-10 10:00:00', '2150-01-20 10:00:00', 10.0),
        (3002, 20, 202, 'Neuro Surgical Intensive Care Unit (Neuro SICU)', '2150-02-01 12:00:00', '2150-02-15 12:00:00', 14.0),
        (3003, 30, 203, 'Trauma SICU (TSICU)', '2150-02-15 08:00:00', '2150-02-27 08:00:00', 12.0),
        (3004, 40, 204, 'Neuro Stepdown', '2150-03-01 11:00:00', '2150-03-10 11:00:00', 9.0)
    """)

    # ── diagnoses_icd (TBI codes S06.x) ──
    conn.execute("""
        CREATE TABLE diagnoses_icd (
            subject_id INTEGER,
            hadm_id INTEGER,
            seq_num INTEGER,
            icd_code VARCHAR,
            icd_version INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO diagnoses_icd VALUES
        (10, 201, 1, 'S065', 10),
        (10, 201, 2, 'S0250A', 10),
        (20, 202, 1, 'S062', 10),
        (30, 203, 1, 'S064', 10),
        (30, 203, 2, 'S065', 10),
        (40, 204, 1, 'S061', 10)
    """)

    # ── d_icd_diagnoses ──
    conn.execute("""
        CREATE TABLE d_icd_diagnoses (
            icd_code VARCHAR,
            icd_version INTEGER,
            long_title VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO d_icd_diagnoses VALUES
        ('S065', 10, 'Traumatic subdural hemorrhage'),
        ('S062', 10, 'Diffuse traumatic brain injury'),
        ('S064', 10, 'Epidural hemorrhage'),
        ('S061', 10, 'Traumatic cerebral edema')
    """)

    # ── d_items ──
    conn.execute("""
        CREATE TABLE d_items (
            itemid INTEGER PRIMARY KEY,
            label VARCHAR,
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO d_items VALUES
        (220739, 'GCS - Eye Opening', 'Neurological'),
        (223900, 'GCS - Verbal Response', 'Neurological'),
        (223901, 'GCS - Motor Response', 'Neurological'),
        (223758, 'Code Status', 'Neurological'),
        (220052, 'Arterial Blood Pressure mean', 'Hemodynamic'),
        (220181, 'Non Invasive Blood Pressure mean', 'Routine Vital Signs')
    """)

    # ── chartevents (GCS, MAP, code status) ──
    conn.execute("""
        CREATE TABLE chartevents (
            subject_id INTEGER,
            hadm_id INTEGER,
            stay_id INTEGER,
            itemid INTEGER,
            charttime TIMESTAMP,
            valuenum DOUBLE,
            value VARCHAR
        )
    """)
    # Patient 10: GCS 3+1+4=8, stays full code
    # Patient 20: GCS 1+1+3=5, changes to DNR
    # Patient 30: GCS 1+1+2=4, changes to CMO, dies
    # Patient 40: GCS 2+1+3=6, discharged to hospice (no code change)
    conn.execute("""
        INSERT INTO chartevents VALUES
        -- Patient 10: GCS components at admission (within 24h)
        (10, 201, 3001, 220739, '2150-01-10 11:00:00', 3, NULL),
        (10, 201, 3001, 223900, '2150-01-10 11:00:00', 1, NULL),
        (10, 201, 3001, 223901, '2150-01-10 11:00:00', 4, NULL),
        -- Patient 10: GCS at 36h (within 48h window)
        (10, 201, 3001, 220739, '2150-01-11 22:00:00', 3, NULL),
        (10, 201, 3001, 223900, '2150-01-11 22:00:00', 2, NULL),
        (10, 201, 3001, 223901, '2150-01-11 22:00:00', 5, NULL),
        -- Patient 10: MAP
        (10, 201, 3001, 220052, '2150-01-10 12:00:00', 75.0, NULL),
        (10, 201, 3001, 220052, '2150-01-11 06:00:00', 80.0, NULL),
        -- Patient 10: Code status stays full
        (10, 201, 3001, 223758, '2150-01-10 12:00:00', NULL, 'Full code'),

        -- Patient 20: GCS components (severe)
        (20, 202, 3002, 220739, '2150-02-01 13:00:00', 1, NULL),
        (20, 202, 3002, 223900, '2150-02-01 13:00:00', 1, NULL),
        (20, 202, 3002, 223901, '2150-02-01 13:00:00', 3, NULL),
        -- Patient 20: MAP
        (20, 202, 3002, 220181, '2150-02-01 14:00:00', 65.0, NULL),
        -- Patient 20: Code status changes to DNR (at day 5, after 48h)
        (20, 202, 3002, 223758, '2150-02-01 14:00:00', NULL, 'Full code'),
        (20, 202, 3002, 223758, '2150-02-06 10:00:00', NULL, 'DNR'),

        -- Patient 30: GCS components (very severe)
        (30, 203, 3003, 220739, '2150-02-15 09:00:00', 1, NULL),
        (30, 203, 3003, 223900, '2150-02-15 09:00:00', 1, NULL),
        (30, 203, 3003, 223901, '2150-02-15 09:00:00', 2, NULL),
        -- Patient 30: Code status changes to CMO (at day 3, within 48h is close)
        (30, 203, 3003, 223758, '2150-02-15 10:00:00', NULL, 'Full code'),
        (30, 203, 3003, 223758, '2150-02-18 08:00:00', NULL, 'Comfort measures only'),

        -- Patient 40: GCS components
        (40, 204, 3004, 220739, '2150-03-01 12:00:00', 2, NULL),
        (40, 204, 3004, 223900, '2150-03-01 12:00:00', 1, NULL),
        (40, 204, 3004, 223901, '2150-03-01 12:00:00', 3, NULL),
        -- Patient 40: Code status stays full, but hospice discharge
        (40, 204, 3004, 223758, '2150-03-01 13:00:00', NULL, 'Full code')
    """)

    # ── d_labitems ──
    conn.execute("""
        CREATE TABLE d_labitems (
            itemid INTEGER PRIMARY KEY,
            label VARCHAR,
            fluid VARCHAR,
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO d_labitems VALUES
        (50983, 'Sodium', 'Blood', 'Chemistry'),
        (50813, 'Lactate', 'Blood', 'Chemistry'),
        (50912, 'Creatinine', 'Blood', 'Chemistry')
    """)

    # ── labevents (within 48h of admission) ──
    conn.execute("""
        CREATE TABLE labevents (
            labevent_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            hadm_id INTEGER,
            itemid INTEGER,
            charttime TIMESTAMP,
            valuenum DOUBLE,
            valueuom VARCHAR,
            ref_range_lower DOUBLE,
            ref_range_upper DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO labevents VALUES
        (1, 10, 201, 50983, '2150-01-10 12:00:00', 142.0, 'mEq/L', 136.0, 145.0),
        (2, 10, 201, 50813, '2150-01-10 12:00:00', 2.1, 'mmol/L', 0.5, 2.0),
        (3, 20, 202, 50983, '2150-02-01 14:00:00', 138.0, 'mEq/L', 136.0, 145.0),
        (4, 30, 203, 50912, '2150-02-15 10:00:00', 1.8, 'mg/dL', 0.7, 1.3)
    """)

    # ── Create age table ──
    conn.execute("""
        CREATE TABLE age AS
        SELECT
            p.subject_id,
            a.hadm_id,
            p.anchor_age + (YEAR(a.admittime) - p.anchor_year) AS age
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
    """)

    yield conn
    conn.close()
