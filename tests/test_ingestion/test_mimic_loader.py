"""Tests for MIMIC-IV DuckDB loader."""

import duckdb
import pytest
from pathlib import Path

from src.ingestion.mimic_loader import load_mimic_to_duckdb, get_loaded_tables

# Path to pre-loaded MIMIC-IV DuckDB database
MIMIC_DUCKDB_PATH = Path("/Users/zacharyrothstein/Code/readmission_predictor_mimic/data/processed/mimiciv.duckdb")


# Tables that must exist in synthetic test data (subset of full MIMIC)
SYNTHETIC_TABLES = [
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

    def test_discovers_and_loads_all_tables(self, synthetic_mimic_dir: Path, tmp_path: Path):
        """Check all tables from synthetic data are discovered and loaded."""
        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_to_duckdb(synthetic_mimic_dir, db_path)

        loaded_tables = get_loaded_tables(conn)

        # All synthetic tables should be loaded
        for table in SYNTHETIC_TABLES:
            assert table in loaded_tables, f"Missing table: {table}"

        # Should have exactly the tables we put in synthetic data
        assert len(loaded_tables) == len(SYNTHETIC_TABLES)

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
    def test_loads_all_mimic_tables(self):
        """Integration test - verify all MIMIC-IV tables are loaded with data."""
        if not MIMIC_DUCKDB_PATH.exists():
            pytest.skip(f"Pre-loaded MIMIC-IV database not found at {MIMIC_DUCKDB_PATH}")

        conn = duckdb.connect(str(MIMIC_DUCKDB_PATH), read_only=True)

        try:
            loaded_tables = get_loaded_tables(conn)

            # Should have many tables (hosp + icu subdirs)
            assert len(loaded_tables) >= 25, f"Expected 25+ tables, got {len(loaded_tables)}"

            # Verify all tables have rows
            for table in loaded_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert count > 0, f"Table {table} has no rows"
        finally:
            conn.close()
