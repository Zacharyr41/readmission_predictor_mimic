"""MIMIC-IV DuckDB loader.

Load MIMIC-IV CSV/CSV.GZ files into DuckDB for downstream analysis.
"""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# Subdirectories to scan for data files
MIMIC_SUBDIRS = ["hosp", "icu"]

# Supported file extensions (in order of preference)
SUPPORTED_EXTENSIONS = [".csv.gz", ".csv", ".parquet"]


def load_mimic_to_duckdb(
    source_dir: Path,
    db_path: Path,
) -> duckdb.DuckDBPyConnection:
    """Load all MIMIC-IV CSV/CSV.GZ files into DuckDB.

    Automatically discovers and loads all data files from hosp/ and icu/
    subdirectories.

    Args:
        source_dir: Path to MIMIC-IV directory containing hosp/ and icu/ subdirs.
        db_path: Path where DuckDB database file will be created.

    Returns:
        DuckDB connection to the loaded database.
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))

    # Discover and load all tables from each subdirectory
    for subdir in MIMIC_SUBDIRS:
        subdir_path = source_dir / subdir
        if not subdir_path.exists():
            logger.warning(f"Subdirectory not found: {subdir_path}")
            continue

        data_files = _discover_data_files(subdir_path)
        logger.info(f"Found {len(data_files)} data files in {subdir}/")

        for table_name, file_path in data_files.items():
            _load_table(conn, table_name, file_path)

    return conn


def _discover_data_files(directory: Path) -> dict[str, Path]:
    """Discover all data files in a directory.

    Returns:
        Dict mapping table names to file paths.
    """
    data_files = {}

    for ext in SUPPORTED_EXTENSIONS:
        if ext == ".csv.gz":
            # Glob for .csv.gz files
            for file_path in directory.glob("*.csv.gz"):
                table_name = file_path.name.replace(".csv.gz", "")
                if table_name not in data_files:
                    data_files[table_name] = file_path
        else:
            for file_path in directory.glob(f"*{ext}"):
                table_name = file_path.stem
                if table_name not in data_files:
                    data_files[table_name] = file_path

    return data_files


def _load_table(conn: duckdb.DuckDBPyConnection, table_name: str, file_path: Path) -> None:
    """Load a single file into a DuckDB table."""
    file_str = str(file_path)

    if file_path.suffix == ".parquet" or file_str.endswith(".parquet"):
        query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file_str}')"
    else:
        # CSV or CSV.GZ - DuckDB handles gzip automatically
        # ignore_errors=true skips rows with conversion errors (e.g., '___' in numeric cols)
        # null_padding=true handles rows with missing columns
        query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto(
                '{file_str}',
                header=true,
                ignore_errors=true,
                null_padding=true
            )
        """

    conn.execute(query)

    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"Loaded {table_name}: {row_count:,} rows from {file_path.name}")


def get_loaded_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Get list of all loaded table names."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return sorted([t[0] for t in tables])
