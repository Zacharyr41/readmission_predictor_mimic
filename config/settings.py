from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    mimic_iv_path: Path = Field(default=Path("data/raw"))
    duckdb_path: Path = Field(default=Path("data/processed/mimiciv.duckdb"))
    clinical_tkg_repo: Path = Field(default=Path("clinical-tkg-cmls2025"))

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # Data source
    data_source: Literal["local", "bigquery"] = Field(default="bigquery")
    bigquery_project: str | None = Field(default=None)

    # SNOMED-CT mappings
    snomed_mappings_dir: Path | None = Field(default=Path("data/mappings"))

    # UMLS API
    umls_api_key: str | None = Field(default=None)

    # Cohort configuration
    cohort_icd_codes: list[str] = Field(default=["I63", "I61", "I60"])
    readmission_window_days: int = Field(default=30)
    patients_limit: int = Field(default=0)  # 0 = no limit
    biomarkers_limit: int = Field(default=0)  # 0 = no limit
    vitals_limit: int = Field(default=0)  # 0 = no limit
    diagnoses_limit: int = Field(default=0)  # 0 = no limit
    skip_allen_relations: bool = Field(default=False)  # Skip Allen relation computation

    @model_validator(mode="after")
    def _validate_bigquery_config(self) -> "Settings":
        if self.data_source == "bigquery" and not self.bigquery_project:
            raise ValueError(
                "BIGQUERY_PROJECT is required when DATA_SOURCE='bigquery'. "
                "Set DATA_SOURCE=local to use local CSV files instead."
            )
        return self
