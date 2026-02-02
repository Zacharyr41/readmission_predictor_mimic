#!/usr/bin/env python
"""Verify MIMIC-IV DuckDB loader.

This script loads MIMIC-IV data into DuckDB and runs verification queries
to ensure the data was loaded correctly.

Usage:
    python scripts/verify_mimic_load.py [--source PATH] [--db PATH]

Examples:
    # Use default paths from settings
    python scripts/verify_mimic_load.py

    # Specify custom paths
    python scripts/verify_mimic_load.py \
        --source /path/to/mimiciv/3.1 \
        --db data/processed/mimiciv.duckdb
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.mimic_loader import load_mimic_to_duckdb, REQUIRED_TABLES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_tables(conn) -> bool:
    """Verify all required tables exist and have rows."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = {t[0] for t in tables}

    logger.info(f"Found {len(table_names)} tables: {sorted(table_names)}")

    all_ok = True
    for table in REQUIRED_TABLES:
        if table not in table_names:
            logger.error(f"Missing table: {table}")
            all_ok = False
            continue

        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table}: {count:,} rows")
        if count == 0:
            logger.warning(f"  Table {table} has no rows!")
            all_ok = False

    return all_ok


def verify_joins(conn) -> bool:
    """Verify key tables can be joined correctly."""
    logger.info("Verifying join queries...")

    # Test patient -> admission -> icustay join
    result = conn.execute("""
        SELECT COUNT(DISTINCT p.subject_id) as patients,
               COUNT(DISTINCT a.hadm_id) as admissions,
               COUNT(DISTINCT i.stay_id) as icustays
        FROM patients p
        INNER JOIN admissions a ON a.subject_id = p.subject_id
        INNER JOIN icustays i ON i.hadm_id = a.hadm_id
    """).fetchone()

    logger.info(f"  Patients with ICU stays: {result[0]:,}")
    logger.info(f"  Admissions with ICU stays: {result[1]:,}")
    logger.info(f"  Total ICU stays: {result[2]:,}")

    # Test diagnosis join
    result = conn.execute("""
        SELECT COUNT(*) as diagnosis_count
        FROM diagnoses_icd d
        INNER JOIN d_icd_diagnoses dd
            ON d.icd_code = dd.icd_code
            AND d.icd_version = dd.icd_version
    """).fetchone()
    logger.info(f"  Diagnoses with descriptions: {result[0]:,}")

    # Sample joined data
    logger.info("Sample patient-admission-icustay join:")
    sample = conn.execute("""
        SELECT p.subject_id, a.hadm_id, i.stay_id,
               p.gender, p.anchor_age,
               a.admission_type, i.los
        FROM patients p
        INNER JOIN admissions a ON a.subject_id = p.subject_id
        INNER JOIN icustays i ON i.hadm_id = a.hadm_id
        LIMIT 5
    """).fetchdf()
    print(sample.to_string(index=False))

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify MIMIC-IV DuckDB loader")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Users/zacharyrothstein/Code/NeuroResearch/physionet.org/files/mimiciv/3.1"),
        help="Path to MIMIC-IV source directory",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/mimiciv.duckdb"),
        help="Path to DuckDB database file",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip loading and just verify existing database",
    )
    args = parser.parse_args()

    if args.skip_load:
        import duckdb
        if not args.db.exists():
            logger.error(f"Database not found: {args.db}")
            sys.exit(1)
        logger.info(f"Opening existing database: {args.db}")
        conn = duckdb.connect(str(args.db))
    else:
        if not args.source.exists():
            logger.error(f"Source directory not found: {args.source}")
            sys.exit(1)

        logger.info(f"Loading MIMIC-IV from: {args.source}")
        logger.info(f"Database path: {args.db}")
        conn = load_mimic_to_duckdb(args.source, args.db)

    try:
        tables_ok = verify_tables(conn)
        joins_ok = verify_joins(conn)

        if tables_ok and joins_ok:
            logger.info("All verifications passed!")
            sys.exit(0)
        else:
            logger.error("Some verifications failed!")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
