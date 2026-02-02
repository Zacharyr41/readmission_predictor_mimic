"""Tests for MIMIC-IV DuckDB loader."""

import pytest
import duckdb
from pathlib import Path

from src.ingestion.mimic_loader import load_mimic_to_duckdb


# Required tables from the plan
REQUIRED_TABLES = [
    "patients",
    "admissions",
    "icustays",
    "labevents",
    "d_labitems",
    "chartevents",
    "d_items",
    "microbiologyevents",
    "prescriptions",
    "diagnoses_icd",
    "d_icd_diagnoses",
    "procedures_icd",
    "d_icd_procedures",
]

# Expected key columns for schema validation
EXPECTED_SCHEMAS = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    "admissions": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "discharge_location",
        "hospital_expire_flag",
    ],
    "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
}


class TestLoadMimicToDuckDB:
    """Tests for load_mimic_to_duckdb function."""

    def test_load_creates_duckdb_file(self, synthetic_mimic_dir: Path, tmp_path: Path):
        """Verify .duckdb file is created."""
        db_path = tmp_path / "test.duckdb"
        assert not db_path.exists()

        conn = load_mimic_to_duckdb(synthetic_mimic_dir, db_path)

        assert db_path.exists()
        conn.close()

    def test_all_required_tables_exist(self, synthetic_mimic_dir: Path, tmp_path: Path):
        """Check all 13 required tables exist."""
        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_to_duckdb(synthetic_mimic_dir, db_path)

        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}

        for required_table in REQUIRED_TABLES:
            assert required_table in table_names, f"Missing table: {required_table}"

        conn.close()

    def test_table_schemas_match_expected(self, synthetic_mimic_dir: Path, tmp_path: Path):
        """Verify key columns for patients, admissions, icustays."""
        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_to_duckdb(synthetic_mimic_dir, db_path)

        for table_name, expected_columns in EXPECTED_SCHEMAS.items():
            columns = conn.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
            ).fetchall()
            actual_columns = {c[0] for c in columns}

            for expected_col in expected_columns:
                assert (
                    expected_col in actual_columns
                ), f"Missing column {expected_col} in table {table_name}"

        conn.close()

    def test_idempotent_load(self, synthetic_mimic_dir: Path, tmp_path: Path):
        """Running twice doesn't duplicate data (CREATE OR REPLACE)."""
        db_path = tmp_path / "test.duckdb"

        # First load
        conn1 = load_mimic_to_duckdb(synthetic_mimic_dir, db_path)
        first_count = conn1.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        conn1.close()

        # Second load
        conn2 = load_mimic_to_duckdb(synthetic_mimic_dir, db_path)
        second_count = conn2.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        conn2.close()

        assert first_count == second_count, "Row count changed after second load"

    @pytest.mark.integration
    def test_row_counts_nonzero(self, real_mimic_dir: Path, tmp_path: Path):
        """Integration test with real data - verify tables have rows."""
        db_path = tmp_path / "mimiciv.duckdb"
        conn = load_mimic_to_duckdb(real_mimic_dir, db_path)

        for table in REQUIRED_TABLES:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count > 0, f"Table {table} has no rows"

        conn.close()
