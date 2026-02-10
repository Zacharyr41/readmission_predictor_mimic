"""MIMIC-IV data ingestion with pluggable data sources."""

import duckdb

from config.settings import Settings


def load_mimic_data(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Load MIMIC-IV data using the configured data source."""
    if settings.data_source == "local":
        from src.ingestion.mimic_loader import load_mimic_to_duckdb

        return load_mimic_to_duckdb(settings.mimic_iv_path, settings.duckdb_path)
    elif settings.data_source == "bigquery":
        from src.ingestion.bigquery_loader import load_mimic_from_bigquery

        return load_mimic_from_bigquery(
            bigquery_project=settings.bigquery_project,
            db_path=settings.duckdb_path,
            cohort_icd_codes=settings.cohort_icd_codes,
            patients_limit=settings.patients_limit,
        )
    else:
        raise ValueError(f"Unknown data_source: {settings.data_source}")
