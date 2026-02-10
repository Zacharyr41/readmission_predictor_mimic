"""Integration tests that hit real BigQuery.

Require:
  - gcloud auth application-default login
  - PhysioNet MIMIC-IV access linked to Google account
  - BIGQUERY_PROJECT set in .env (or environment)

Run with:
  pytest -m bigquery -v
"""

import pytest
import duckdb

from google.cloud import bigquery
from google.api_core.exceptions import Forbidden, NotFound

from config.settings import Settings
from src.ingestion.bigquery_loader import (
    TABLE_REGISTRY,
    MIMIC_SOURCE,
    load_mimic_from_bigquery,
)


def _get_bigquery_project():
    """Read BIGQUERY_PROJECT from settings, skip if not configured."""
    try:
        settings = Settings(data_source="bigquery")
        return settings.bigquery_project
    except Exception:
        # Settings might fail if no .env — try env var directly
        import os
        return os.environ.get("BIGQUERY_PROJECT")


@pytest.fixture(scope="module")
def bq_project():
    """Get BigQuery project ID, skip entire module if not available."""
    project = _get_bigquery_project()
    if not project:
        pytest.skip("BIGQUERY_PROJECT not configured")
    return project


@pytest.fixture(scope="module")
def bq_client(bq_project):
    """Create a BigQuery client, skip if auth fails."""
    try:
        client = bigquery.Client(project=bq_project)
        # Quick smoke test — list datasets to verify auth works
        list(client.list_datasets("physionet-data", max_results=1))
        return client
    except Exception as e:
        pytest.skip(f"BigQuery auth failed: {e}")


# ---------------------------------------------------------------------------
# Test 1: All datasets we reference actually exist
# ---------------------------------------------------------------------------

@pytest.mark.bigquery
class TestBigQueryDatasetAccess:
    """Verify we can access the real PhysioNet datasets."""

    def test_hosp_dataset_exists(self, bq_client):
        """mimiciv_3_1_hosp dataset should be accessible."""
        dataset_ref = bigquery.DatasetReference("physionet-data", "mimiciv_3_1_hosp")
        dataset = bq_client.get_dataset(dataset_ref)
        assert dataset.dataset_id == "mimiciv_3_1_hosp"

    def test_icu_dataset_exists(self, bq_client):
        """mimiciv_3_1_icu dataset should be accessible."""
        dataset_ref = bigquery.DatasetReference("physionet-data", "mimiciv_3_1_icu")
        dataset = bq_client.get_dataset(dataset_ref)
        assert dataset.dataset_id == "mimiciv_3_1_icu"


# ---------------------------------------------------------------------------
# Test 2: Every table in TABLE_REGISTRY exists (metadata only, no query cost)
# ---------------------------------------------------------------------------

@pytest.mark.bigquery
class TestBigQueryTableRegistry:
    """Verify every table in TABLE_REGISTRY exists on BigQuery.

    Uses get_table() metadata calls (free) instead of SELECT queries
    to avoid scanning large tables like chartevents (35 GB per scan).
    """

    @pytest.mark.parametrize(
        "table_name,dataset",
        [(t, ds) for t, (ds, _) in TABLE_REGISTRY.items()],
        ids=list(TABLE_REGISTRY.keys()),
    )
    def test_table_exists(self, bq_client, table_name, dataset):
        """Each registered table should exist and have rows."""
        table_ref = f"{MIMIC_SOURCE}.{dataset}.{table_name}"
        try:
            table = bq_client.get_table(table_ref)
        except (Forbidden, NotFound) as e:
            pytest.fail(f"Cannot access {table_ref}: {e}")

        assert table.num_rows > 0, f"{table_ref} has 0 rows"


# ---------------------------------------------------------------------------
# Test 3: Key columns exist in the real tables
# ---------------------------------------------------------------------------

@pytest.mark.bigquery
class TestBigQuerySchemas:
    """Verify key columns exist in real BigQuery tables."""

    EXPECTED_COLUMNS = {
        "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
        "admissions": ["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"],
        "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        "diagnoses_icd": ["subject_id", "hadm_id", "icd_code", "icd_version"],
        "labevents": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        "chartevents": ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"],
    }

    @pytest.mark.parametrize(
        "table_name,expected_cols",
        list(EXPECTED_COLUMNS.items()),
        ids=list(EXPECTED_COLUMNS.keys()),
    )
    def test_table_has_expected_columns(self, bq_client, table_name, expected_cols):
        """Real BigQuery table should contain the columns our code depends on."""
        dataset = dict((t, ds) for t, (ds, _) in TABLE_REGISTRY.items())[table_name]
        table_ref = f"{MIMIC_SOURCE}.{dataset}.{table_name}"
        table = bq_client.get_table(table_ref)
        actual_cols = {field.name for field in table.schema}

        for col in expected_cols:
            assert col in actual_cols, (
                f"Column '{col}' missing from {table_ref}. "
                f"Actual columns: {sorted(actual_cols)}"
            )


# ---------------------------------------------------------------------------
# Test 4: End-to-end loader with small patient limit
# ---------------------------------------------------------------------------

@pytest.mark.bigquery
@pytest.mark.bigquery_e2e
class TestBigQueryLoaderEndToEnd:
    """Run the real loader with a tiny cohort.

    Skipped by default — scans ~55 GB (~$0.35). Run explicitly with:
        pytest -m bigquery_e2e -v
    """

    def test_load_small_cohort(self, bq_project, tmp_path):
        """Load 5 patients from real BigQuery and verify DuckDB output."""
        db_path = tmp_path / "bq_test.duckdb"

        conn = load_mimic_from_bigquery(
            bigquery_project=bq_project,
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
            patients_limit=5,
        )

        # All 13 tables should exist
        tables = [
            r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]

        for expected in TABLE_REGISTRY:
            assert expected in tables, f"Missing table: {expected}"

        # patients table should have data
        patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        assert patient_count > 0, "patients table is empty"

        # Derived age table should have been created during cohort derivation
        age_count = conn.execute("SELECT COUNT(*) FROM age").fetchone()[0]
        assert age_count > 0, "age table is empty"

        # Verify timestamps work with DuckDB functions
        result = conn.execute(
            "SELECT YEAR(admittime) FROM admissions LIMIT 1"
        ).fetchone()
        assert result is not None

        conn.close()
