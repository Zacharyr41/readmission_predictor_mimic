"""CLI entry point for MIMIC-IV ingestion module.

Usage:
    python -m src.ingestion [--source-dir PATH] [--db-path PATH] [--data-source local|bigquery]
"""

import argparse
import logging
from pathlib import Path

from config.settings import Settings
from src.ingestion import load_mimic_data
from src.ingestion.mimic_loader import get_loaded_tables
from src.ingestion.derived_tables import (
    create_age_table,
    create_readmission_labels,
)


logger = logging.getLogger(__name__)


def main():
    """Load MIMIC-IV data into DuckDB and create derived tables."""
    parser = argparse.ArgumentParser(
        description="Load MIMIC-IV data into DuckDB",
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
        "--data-source",
        choices=["local", "bigquery"],
        default=None,
        help="Data source for MIMIC-IV (default: from settings)",
    )
    parser.add_argument(
        "--bigquery-project",
        type=str,
        default=None,
        help="GCP project ID for BigQuery billing",
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

    # Load settings with CLI overrides
    updates = {}
    if args.source_dir:
        updates["mimic_iv_path"] = args.source_dir
    if args.db_path:
        updates["duckdb_path"] = args.db_path
    if args.data_source:
        updates["data_source"] = args.data_source
    if args.bigquery_project:
        updates["bigquery_project"] = args.bigquery_project

    settings = Settings()
    if updates:
        settings = settings.model_copy(update=updates)

    logger.info(f"Loading MIMIC-IV from {settings.data_source}")
    logger.info(f"Output DuckDB: {settings.duckdb_path}")

    # Load data via dispatch
    conn = load_mimic_data(settings)

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
