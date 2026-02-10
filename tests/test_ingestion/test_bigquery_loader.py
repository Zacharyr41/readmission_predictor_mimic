"""Tests for BigQuery MIMIC-IV loader.

Covers:
- Step 3: Table queries, two-phase loading, cohort filtering, patients_limit, all tables written
- Step 4: DuckDB schema compatibility
- Step 5: Derived tables contract tests
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import duckdb
import pandas as pd
import pytest

from tests.test_ingestion.test_mimic_loader import SYNTHETIC_TABLES, EXPECTED_SCHEMAS


# ---------------------------------------------------------------------------
# Helpers: mock DataFrames matching BigQuery output
# ---------------------------------------------------------------------------

def _patients_df():
    return pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5],
        "gender": ["M", "F", "M", "F", "M"],
        "anchor_age": [65, 72, 58, 45, 80],
        "anchor_year": [2150, 2151, 2152, 2150, 2151],
        "anchor_year_group": ["2017 - 2019"] * 5,
        "dod": [None, None, None, None, "2151-06-15"],
    })


def _admissions_df():
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4, 5],
        "hadm_id": [101, 102, 103, 104, 105, 106],
        "admittime": pd.to_datetime([
            "2150-01-15 08:00:00", "2150-02-10 10:00:00", "2151-03-01 06:00:00",
            "2152-05-20 14:00:00", "2150-07-01 09:00:00", "2151-04-10 12:00:00",
        ]),
        "dischtime": pd.to_datetime([
            "2150-01-20 14:00:00", "2150-02-15 12:00:00", "2151-03-10 16:00:00",
            "2152-05-25 10:00:00", "2150-07-05 11:00:00", "2151-04-20 08:00:00",
        ]),
        "deathtime": [None] * 6,
        "admission_type": ["EMERGENCY", "EMERGENCY", "ELECTIVE", "EMERGENCY", "URGENT", "EMERGENCY"],
        "discharge_location": ["HOME", "HOME", "SNF", "HOME", "HOME", "HOSPICE"],
        "hospital_expire_flag": [0, 0, 0, 0, 0, 1],
    })


def _icustays_df():
    return pd.DataFrame({
        "subject_id": [1, 2, 5],
        "hadm_id": [101, 103, 106],
        "stay_id": [1001, 1002, 1003],
        "intime": pd.to_datetime([
            "2150-01-15 10:00:00", "2151-03-02 08:00:00", "2151-04-11 00:00:00",
        ]),
        "outtime": pd.to_datetime([
            "2150-01-18 08:00:00", "2151-03-08 12:00:00", "2151-04-18 06:00:00",
        ]),
        "los": [2.9, 6.2, 7.25],
    })


def _diagnoses_icd_df():
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4, 5],
        "hadm_id": [101, 102, 103, 104, 105, 106],
        "seq_num": [1, 1, 1, 1, 1, 1],
        "icd_code": ["I639", "I634", "I630", "G409", "I251", "I639"],
        "icd_version": [10, 10, 10, 10, 10, 10],
    })


def _d_icd_diagnoses_df():
    return pd.DataFrame({
        "icd_code": ["I639", "I634", "I630", "G409", "I251"],
        "icd_version": [10, 10, 10, 10, 10],
        "long_title": [
            "Cerebral infarction, unspecified",
            "Cerebral infarction due to embolism of cerebral arteries",
            "Cerebral infarction due to thrombosis of precerebral arteries",
            "Epilepsy, unspecified",
            "Atherosclerotic heart disease of native coronary artery",
        ],
    })


def _procedures_icd_df():
    return pd.DataFrame({
        "subject_id": [1, 2, 3],
        "hadm_id": [101, 103, 104],
        "seq_num": [1, 1, 1],
        "icd_code": ["0016070", "02H63JZ", "00JU0ZZ"],
        "icd_version": [10, 10, 10],
    })


def _d_icd_procedures_df():
    return pd.DataFrame({
        "icd_code": ["0016070", "02H63JZ", "00JU0ZZ"],
        "icd_version": [10, 10, 10],
        "long_title": [
            "Bypass Cerebral Ventricle to Nasopharynx with Autologous Tissue",
            "Insertion of Pacemaker Lead into Right Atrium Percutaneous Approach",
            "Inspection of Spinal Canal Open Approach",
        ],
    })


def _d_labitems_df():
    return pd.DataFrame({
        "itemid": [50912, 50971, 51265],
        "label": ["Creatinine", "Sodium", "Platelet Count"],
        "fluid": ["Blood", "Blood", "Blood"],
        "category": ["Chemistry", "Chemistry", "Hematology"],
    })


def _d_items_df():
    return pd.DataFrame({
        "itemid": [220045, 220179, 220180],
        "label": ["Heart Rate", "Non Invasive Blood Pressure systolic", "Non Invasive Blood Pressure diastolic"],
        "category": ["Routine Vital Signs", "Routine Vital Signs", "Routine Vital Signs"],
    })


def _labevents_df(subject_ids=None):
    df = pd.DataFrame({
        "labevent_id": [1, 2, 3, 4],
        "subject_id": [1, 1, 2, 5],
        "hadm_id": [101, 101, 103, 106],
        "itemid": [50912, 50971, 50912, 50912],
        "charttime": pd.to_datetime([
            "2150-01-16 06:00:00", "2150-01-16 06:00:00",
            "2151-03-03 08:00:00", "2151-04-12 10:00:00",
        ]),
        "valuenum": [1.2, 140.0, 0.9, 1.5],
        "valueuom": ["mg/dL", "mEq/L", "mg/dL", "mg/dL"],
        "ref_range_lower": [0.7, 136.0, 0.7, 0.7],
        "ref_range_upper": [1.3, 145.0, 1.3, 1.3],
    })
    if subject_ids is not None:
        df = df[df["subject_id"].isin(subject_ids)]
    return df


def _chartevents_df(subject_ids=None):
    df = pd.DataFrame({
        "subject_id": [1, 1, 2, 5],
        "hadm_id": [101, 101, 103, 106],
        "stay_id": [1001, 1001, 1002, 1003],
        "itemid": [220045, 220179, 220045, 220045],
        "charttime": pd.to_datetime([
            "2150-01-16 08:00:00", "2150-01-16 08:00:00",
            "2151-03-03 10:00:00", "2151-04-12 12:00:00",
        ]),
        "valuenum": [78.0, 120.0, 82.0, 95.0],
    })
    if subject_ids is not None:
        df = df[df["subject_id"].isin(subject_ids)]
    return df


def _microbiologyevents_df(subject_ids=None):
    df = pd.DataFrame({
        "microevent_id": [1, 2],
        "subject_id": [1, 2],
        "hadm_id": [101, 103],
        "charttime": pd.to_datetime(["2150-01-16 12:00:00", "2151-03-04 14:00:00"]),
        "spec_type_desc": ["BLOOD CULTURE", "URINE"],
        "org_name": ["STAPHYLOCOCCUS AUREUS", "ESCHERICHIA COLI"],
    })
    if subject_ids is not None:
        df = df[df["subject_id"].isin(subject_ids)]
    return df


def _prescriptions_df(subject_ids=None):
    df = pd.DataFrame({
        "subject_id": [1, 2],
        "hadm_id": [101, 103],
        "starttime": pd.to_datetime(["2150-01-15 12:00:00", "2151-03-02 10:00:00"]),
        "stoptime": pd.to_datetime(["2150-01-18 12:00:00", "2151-03-07 10:00:00"]),
        "drug": ["Vancomycin", "Ceftriaxone"],
        "dose_val_rx": [1000.0, 2000.0],
        "dose_unit_rx": ["mg", "mg"],
        "route": ["IV", "IV"],
    })
    if subject_ids is not None:
        df = df[df["subject_id"].isin(subject_ids)]
    return df


# Map table names to their mock DataFrame generators
SMALL_TABLE_DFS = {
    "patients": _patients_df,
    "admissions": _admissions_df,
    "icustays": _icustays_df,
    "diagnoses_icd": _diagnoses_icd_df,
    "d_icd_diagnoses": _d_icd_diagnoses_df,
    "procedures_icd": _procedures_icd_df,
    "d_icd_procedures": _d_icd_procedures_df,
    "d_labitems": _d_labitems_df,
    "d_items": _d_items_df,
}

LARGE_TABLE_DFS = {
    "labevents": _labevents_df,
    "chartevents": _chartevents_df,
    "microbiologyevents": _microbiologyevents_df,
    "prescriptions": _prescriptions_df,
}


def _make_mock_client(cohort_subject_ids=None):
    """Create a mock BigQuery client that returns appropriate DataFrames.

    The mock intercepts client.query(sql) and returns the right DataFrame
    based on which table is referenced in the SQL.
    """
    client = MagicMock()

    def _fake_query(sql, **kwargs):
        result = MagicMock()
        # Determine which table is being queried from the SQL string
        for table_name, df_fn in SMALL_TABLE_DFS.items():
            if f".{table_name}`" in sql:
                result.to_dataframe.return_value = df_fn()
                return result
        for table_name, df_fn in LARGE_TABLE_DFS.items():
            if f".{table_name}`" in sql:
                result.to_dataframe.return_value = df_fn(cohort_subject_ids)
                return result
        # Fallback: empty DataFrame
        result.to_dataframe.return_value = pd.DataFrame()
        return result

    client.query.side_effect = _fake_query
    return client


# ---------------------------------------------------------------------------
# TABLE_REGISTRY validation (catches dataset name typos without auth)
# ---------------------------------------------------------------------------

class TestTableRegistryConsistency:
    """Validate TABLE_REGISTRY dataset names match real BigQuery naming."""

    def test_datasets_use_versioned_names(self):
        """Dataset names must be versioned (e.g. mimiciv_3_1_hosp, not mimiciv_hosp)."""
        from src.ingestion.bigquery_loader import TABLE_REGISTRY

        for table_name, (dataset, _) in TABLE_REGISTRY.items():
            assert "_3_1_" in dataset, (
                f"TABLE_REGISTRY['{table_name}'] uses unversioned dataset '{dataset}'. "
                f"BigQuery datasets are versioned (e.g. 'mimiciv_3_1_hosp')."
            )

    def test_hosp_tables_in_hosp_dataset(self):
        """Hospital tables should reference the hosp dataset."""
        from src.ingestion.bigquery_loader import TABLE_REGISTRY

        hosp_tables = [
            "patients", "admissions", "diagnoses_icd", "d_icd_diagnoses",
            "procedures_icd", "d_icd_procedures", "d_labitems",
            "labevents", "microbiologyevents", "prescriptions",
        ]
        for table in hosp_tables:
            dataset, _ = TABLE_REGISTRY[table]
            assert "hosp" in dataset, f"{table} should be in a hosp dataset, got {dataset}"

    def test_icu_tables_in_icu_dataset(self):
        """ICU tables should reference the icu dataset."""
        from src.ingestion.bigquery_loader import TABLE_REGISTRY

        icu_tables = ["icustays", "chartevents", "d_items"]
        for table in icu_tables:
            dataset, _ = TABLE_REGISTRY[table]
            assert "icu" in dataset, f"{table} should be in an icu dataset, got {dataset}"

    def test_registry_has_all_expected_tables(self):
        """TABLE_REGISTRY should contain all 13 required tables."""
        from src.ingestion.bigquery_loader import TABLE_REGISTRY

        for table in SYNTHETIC_TABLES:
            assert table in TABLE_REGISTRY, f"Missing table in TABLE_REGISTRY: {table}"


# ---------------------------------------------------------------------------
# Step 3: Table query tests
# ---------------------------------------------------------------------------

class TestBigQueryTableQueries:
    """Test that correct BigQuery tables are queried."""

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_queries_correct_fully_qualified_tables(self, mock_bq_module, tmp_path):
        """Each table should be queried from its correct physionet-data dataset."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery, TABLE_REGISTRY

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        # Collect all SQL queries
        all_queries = [c.args[0] for c in mock_client.query.call_args_list]
        all_sql = "\n".join(all_queries)

        # Each table should reference its correct dataset
        for table_name, (dataset, _) in TABLE_REGISTRY.items():
            expected_ref = f"`physionet-data.{dataset}.{table_name}`"
            assert expected_ref in all_sql, (
                f"Expected query for {expected_ref} not found in SQL queries"
            )

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_two_phase_loading(self, mock_bq_module, tmp_path):
        """Phase 1 loads small tables, phase 2 loads large tables with subject_id filter."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery, TABLE_REGISTRY

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        all_queries = [c.args[0] for c in mock_client.query.call_args_list]

        # Large tables should have WHERE subject_id IN clause
        large_tables = {t for t, (_, is_large) in TABLE_REGISTRY.items() if is_large}
        for query in all_queries:
            for table_name in large_tables:
                if f".{table_name}`" in query:
                    assert "subject_id IN" in query, (
                        f"Large table {table_name} query missing subject_id IN filter"
                    )

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_patients_limit_truncates_cohort(self, mock_bq_module, tmp_path):
        """When patients_limit > 0, fewer subject_ids should be used for large tables."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery

        mock_client = _make_mock_client(cohort_subject_ids={1})
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
            patients_limit=1,
        )

        # With patients_limit=1, large table queries should reference only 1 subject_id
        all_queries = [c.args[0] for c in mock_client.query.call_args_list]
        large_queries = [q for q in all_queries if "subject_id IN" in q]
        assert len(large_queries) > 0

        # Each large table query should have at most 1 subject_id
        for q in large_queries:
            # Extract the IN (...) part
            in_clause = q.split("subject_id IN (")[1].split(")")[0]
            ids = [x.strip() for x in in_clause.split(",") if x.strip()]
            assert len(ids) <= 1

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_all_tables_written_to_duckdb(self, mock_bq_module, tmp_path):
        """All 13 MIMIC tables should exist in DuckDB after loading."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        tables = [r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()]

        for table in SYNTHETIC_TABLES:
            assert table in tables, f"Missing table: {table}"

        conn.close()

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_returns_duckdb_connection(self, mock_bq_module, tmp_path):
        """Function should return an open DuckDB connection."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        assert isinstance(conn, duckdb.DuckDBPyConnection)
        # Should be open (can execute a query)
        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)
        conn.close()


# ---------------------------------------------------------------------------
# Step 4: DuckDB schema compatibility
# ---------------------------------------------------------------------------

class TestSchemaCompatibility:
    """Verify BigQuery-loaded tables have expected schemas."""

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_expected_schemas_match(self, mock_bq_module, tmp_path):
        """Key columns for patients, admissions, icustays should match EXPECTED_SCHEMAS."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        for table_name, expected_columns in EXPECTED_SCHEMAS.items():
            columns = conn.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
            ).fetchall()
            actual_columns = {c[0] for c in columns}

            for expected_col in expected_columns:
                assert expected_col in actual_columns, (
                    f"Missing column {expected_col} in table {table_name}"
                )

        conn.close()

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_timestamp_columns_work_with_duckdb_functions(self, mock_bq_module, tmp_path):
        """Timestamp columns should work with YEAR() and DATE_DIFF()."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        # YEAR() on admissions.admittime
        result = conn.execute(
            "SELECT YEAR(admittime) FROM admissions LIMIT 1"
        ).fetchone()
        assert result is not None
        assert isinstance(result[0], int)

        # DATE_DIFF on icustays
        result = conn.execute(
            "SELECT DATE_DIFF('hour', intime, outtime) FROM icustays LIMIT 1"
        ).fetchone()
        assert result is not None

        conn.close()


# ---------------------------------------------------------------------------
# Step 5: Derived tables contract tests
# ---------------------------------------------------------------------------

class TestDerivedTablesContract:
    """Contract tests: BigQuery output is compatible with derived_tables.py."""

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_create_age_table_succeeds(self, mock_bq_module, tmp_path):
        """create_age_table should produce subject_id, hadm_id, age columns."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery
        from src.ingestion.derived_tables import create_age_table

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        create_age_table(conn)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'age'"
        ).fetchall()
        column_names = {c[0] for c in columns}
        assert {"subject_id", "hadm_id", "age"} <= column_names

        conn.close()

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_select_neurology_cohort_succeeds(self, mock_bq_module, tmp_path):
        """select_neurology_cohort should return DataFrame with expected columns."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery
        from src.ingestion.derived_tables import create_age_table, select_neurology_cohort

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        create_age_table(conn)
        cohort_df = select_neurology_cohort(conn, ["I63", "I61", "I60"])

        assert "subject_id" in cohort_df.columns
        assert "hadm_id" in cohort_df.columns
        assert "stay_id" in cohort_df.columns

        conn.close()

    @patch("src.ingestion.bigquery_loader.bigquery")
    def test_create_readmission_labels_succeeds(self, mock_bq_module, tmp_path):
        """create_readmission_labels should produce expected columns."""
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery
        from src.ingestion.derived_tables import create_readmission_labels

        mock_client = _make_mock_client()
        mock_bq_module.Client.return_value = mock_client

        db_path = tmp_path / "test.duckdb"
        conn = load_mimic_from_bigquery(
            bigquery_project="my-proj",
            db_path=db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
        )

        create_readmission_labels(conn)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'readmission_labels'"
        ).fetchall()
        column_names = {c[0] for c in columns}
        assert {"subject_id", "hadm_id", "readmitted_30d", "readmitted_60d"} <= column_names

        conn.close()
