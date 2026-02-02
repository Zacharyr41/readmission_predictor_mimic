"""MIMIC-IV DuckDB loader.

Load MIMIC-IV CSV/CSV.GZ files into DuckDB for downstream analysis.
"""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# Required tables and their subdirectory locations
REQUIRED_TABLES = {
    # hosp/ tables
    "patients": "hosp",
    "admissions": "hosp",
    "labevents": "hosp",
    "d_labitems": "hosp",
    "microbiologyevents": "hosp",
    "prescriptions": "hosp",
    "diagnoses_icd": "hosp",
    "d_icd_diagnoses": "hosp",
    "procedures_icd": "hosp",
    "d_icd_procedures": "hosp",
    # icu/ tables
    "icustays": "icu",
    "chartevents": "icu",
    "d_items": "icu",
}


def load_mimic_to_duckdb(
    source_dir: Path,
    db_path: Path,
) -> duckdb.DuckDBPyConnection:
    """Load MIMIC-IV CSV/CSV.GZ files into DuckDB.

    Args:
        source_dir: Path to MIMIC-IV directory containing hosp/ and icu/ subdirs.
        db_path: Path where DuckDB database file will be created.

    Returns:
        DuckDB connection to the loaded database.
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))

    for table_name, subdir in REQUIRED_TABLES.items():
        file_path = _find_table_file(source_dir / subdir, table_name)
        if file_path is None:
            logger.warning(f"File not found for table {table_name} in {source_dir / subdir}")
            continue

        _load_table(conn, table_name, file_path)

    return conn


def _find_table_file(directory: Path, table_name: str) -> Path | None:
    """Find the file for a table, supporting .csv, .csv.gz, and .parquet extensions."""
    extensions = [".csv.gz", ".csv", ".parquet"]

    for ext in extensions:
        file_path = directory / f"{table_name}{ext}"
        if file_path.exists():
            return file_path

    return None


def _load_table(conn: duckdb.DuckDBPyConnection, table_name: str, file_path: Path) -> None:
    """Load a single file into a DuckDB table."""
    file_str = str(file_path)

    if file_path.suffix == ".parquet" or file_str.endswith(".parquet"):
        query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file_str}')"
    else:
        # CSV or CSV.GZ - DuckDB handles gzip automatically
        query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_str}', header=true)"

    conn.execute(query)

    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"Loaded {table_name}: {row_count} rows from {file_path.name}")
