"""BigQuery data source for MIMIC-IV ingestion.

Loads MIMIC-IV data from Google BigQuery (physionet-data project)
into a local DuckDB database. Uses two-phase loading:
  Phase 1: Small/dimension tables loaded in full
  Phase 2: Large tables filtered by cohort subject_ids
"""

import logging
import time
from pathlib import Path

import duckdb
import pandas as pd
from google.cloud import bigquery

from src.ingestion.derived_tables import create_age_table, select_neurology_cohort

logger = logging.getLogger(__name__)

MIMIC_SOURCE = "physionet-data"

# Table -> (dataset, is_large)
# Dataset names are versioned on BigQuery (e.g. mimiciv_3_1_hosp)
TABLE_REGISTRY = {
    # Hospital tables
    "patients": ("mimiciv_3_1_hosp", False),
    "admissions": ("mimiciv_3_1_hosp", False),
    "diagnoses_icd": ("mimiciv_3_1_hosp", False),
    "d_icd_diagnoses": ("mimiciv_3_1_hosp", False),
    "procedures_icd": ("mimiciv_3_1_hosp", False),
    "d_icd_procedures": ("mimiciv_3_1_hosp", False),
    "d_labitems": ("mimiciv_3_1_hosp", False),
    "labevents": ("mimiciv_3_1_hosp", True),
    "microbiologyevents": ("mimiciv_3_1_hosp", True),
    "prescriptions": ("mimiciv_3_1_hosp", True),
    # ICU tables
    "icustays": ("mimiciv_3_1_icu", False),
    "chartevents": ("mimiciv_3_1_icu", True),
    "d_items": ("mimiciv_3_1_icu", False),
}


def _run_query(
    client: bigquery.Client, sql: str, label: str
) -> pd.DataFrame:
    """Run a BigQuery query with progress logging."""
    t0 = time.time()
    logger.info("  [%s] Query submitted...", label)
    job = client.query(sql)

    # Wait for query to finish, log progress
    while not job.done():
        time.sleep(5)
        elapsed = time.time() - t0
        logger.info("  [%s] Query running... %.0fs elapsed", label, elapsed)

    query_time = time.time() - t0
    billed = job.total_bytes_billed or 0
    logger.info(
        "  [%s] Query complete in %.0fs (%.1f MB billed). Downloading results...",
        label,
        query_time,
        billed / 1e6,
    )

    # Download results
    df = job.to_dataframe()
    total_time = time.time() - t0
    logger.info(
        "  [%s] Downloaded %d rows in %.0fs (%.0fs query + %.0fs download)",
        label,
        len(df),
        total_time,
        query_time,
        total_time - query_time,
    )
    return df


def _query_full_table(
    client: bigquery.Client, dataset: str, table_name: str
) -> pd.DataFrame:
    """Query an entire BigQuery table."""
    fq_table = f"`{MIMIC_SOURCE}.{dataset}.{table_name}`"
    sql = f"SELECT * FROM {fq_table}"
    return _run_query(client, sql, table_name)


def _query_filtered_table(
    client: bigquery.Client,
    dataset: str,
    table_name: str,
    subject_ids: list[int],
) -> pd.DataFrame:
    """Query a BigQuery table filtered by subject_ids."""
    fq_table = f"`{MIMIC_SOURCE}.{dataset}.{table_name}`"
    ids_str = ", ".join(str(sid) for sid in subject_ids)
    sql = f"SELECT * FROM {fq_table} WHERE subject_id IN ({ids_str})"
    return _run_query(client, sql, f"{table_name} (filtered, {len(subject_ids)} subj)")


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BigQuery-specific dtypes to standard pandas types for DuckDB.

    BigQuery's to_dataframe() returns db-dtypes types (dbdate, dbtime, etc.)
    that DuckDB doesn't recognize. Convert them to standard pandas types.
    """
    for col in df.columns:
        dtype_name = str(df[col].dtype)
        if dtype_name == "dbdate":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dtype_name == "dbtime":
            df[col] = df[col].astype(str)
        elif dtype_name.startswith("Int") or dtype_name.startswith("UInt"):
            # BigQuery nullable integers (Int64, etc.) -> standard numpy
            df[col] = df[col].astype("object").where(df[col].notna(), None)
    return df


def _insert_dataframe(
    conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame
) -> None:
    """Insert a DataFrame into DuckDB, handling empty DataFrames."""
    if df.empty:
        # Create empty table with correct column names/types
        cols = ", ".join(f"{col} VARCHAR" for col in df.columns)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} ({cols})")
    else:
        df = _normalize_dtypes(df)
        conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df"
        )
    logger.info(f"  Wrote {len(df)} rows to {table_name}")


def load_mimic_from_bigquery(
    bigquery_project: str,
    db_path: Path | str,
    cohort_icd_codes: list[str] | None = None,
    patients_limit: int = 0,
) -> duckdb.DuckDBPyConnection:
    """Load MIMIC-IV data from BigQuery into DuckDB.

    Two-phase loading:
      Phase 1: Load small/dimension tables in full.
      Phase 2: Derive cohort, then load large tables filtered by cohort subject_ids.

    Args:
        bigquery_project: GCP project ID for billing.
        db_path: Path to output DuckDB file.
        cohort_icd_codes: ICD-10 prefixes for cohort selection.
        patients_limit: Max patients (0 = no limit).

    Returns:
        Open DuckDB connection with all tables loaded.
    """
    if cohort_icd_codes is None:
        cohort_icd_codes = ["I63", "I61", "I60"]

    client = bigquery.Client(project=bigquery_project)
    conn = duckdb.connect(str(db_path))

    # Phase 1: Load small tables
    logger.info("Phase 1: Loading base tables from BigQuery...")
    small_tables = {t: (ds, lg) for t, (ds, lg) in TABLE_REGISTRY.items() if not lg}
    for table_name, (dataset, _) in small_tables.items():
        df = _query_full_table(client, dataset, table_name)
        _insert_dataframe(conn, table_name, df)

    # Cohort derivation
    logger.info("Deriving cohort...")
    create_age_table(conn)
    cohort_df = select_neurology_cohort(conn, cohort_icd_codes)
    subject_ids = cohort_df["subject_id"].unique().tolist()

    if patients_limit > 0:
        subject_ids = subject_ids[:patients_limit]
        logger.info(f"  Truncated cohort to {len(subject_ids)} patients (limit={patients_limit})")

    logger.info(f"  Cohort: {len(subject_ids)} unique subjects")

    # Phase 2: Load large tables filtered by cohort
    logger.info("Phase 2: Loading large tables filtered by cohort...")
    large_tables = {t: (ds, lg) for t, (ds, lg) in TABLE_REGISTRY.items() if lg}

    if not subject_ids:
        # No cohort — create empty large tables
        for table_name, (dataset, _) in large_tables.items():
            _insert_dataframe(conn, table_name, pd.DataFrame())
    else:
        for table_name, (dataset, _) in large_tables.items():
            df = _query_filtered_table(client, dataset, table_name, subject_ids)
            _insert_dataframe(conn, table_name, df)

    logger.info("BigQuery ingestion complete.")
    return conn
