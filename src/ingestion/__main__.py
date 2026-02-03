"""CLI entry point for MIMIC-IV ingestion module.

Usage:
    python -m src.ingestion [--source-dir PATH] [--db-path PATH]
"""

import argparse
import logging
from pathlib import Path

from config.settings import Settings
from src.ingestion.mimic_loader import load_mimic_to_duckdb, get_loaded_tables
from src.ingestion.derived_tables import (
    create_age_table,
    create_readmission_labels,
)


logger = logging.getLogger(__name__)


def main():
    """Load MIMIC-IV data into DuckDB and create derived tables."""
    parser = argparse.ArgumentParser(
        description="Load MIMIC-IV CSV files into DuckDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Path to MIMIC-IV directory (default: from settings)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to output DuckDB file (default: from settings)",
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip creating derived tables (age, readmission_labels)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load settings
    settings = Settings()

    source_dir = args.source_dir or settings.mimic_iv_path
    db_path = args.db_path or settings.duckdb_path

    logger.info(f"Loading MIMIC-IV from {source_dir}")
    logger.info(f"Output DuckDB: {db_path}")

    # Load data
    conn = load_mimic_to_duckdb(source_dir, db_path)

    # Create derived tables
    if not args.skip_derived:
        logger.info("Creating derived tables...")
        try:
            create_age_table(conn)
            logger.info("  Created age table")
        except Exception as e:
            logger.warning(f"  Could not create age table: {e}")

        try:
            create_readmission_labels(conn)
            logger.info("  Created readmission_labels table")
        except Exception as e:
            logger.warning(f"  Could not create readmission_labels: {e}")

    # Print summary
    tables = get_loaded_tables(conn)
    conn.close()

    print(f"\nLoaded {len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")


if __name__ == "__main__":
    main()
