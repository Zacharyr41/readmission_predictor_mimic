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

    # Anthropic API (conversational analytics)
    anthropic_api_key: str | None = Field(default=None)

    # Cohort configuration
    cohort_icd_codes: list[str] = Field(default=["I63", "I61", "I60"])
    readmission_window_days: int = Field(default=30)
    patients_limit: int = Field(default=0)  # 0 = no limit
    biomarkers_limit: int = Field(default=0)  # 0 = no limit
    vitals_limit: int = Field(default=0)  # 0 = no limit
    diagnoses_limit: int = Field(default=0)  # 0 = no limit
    skip_allen_relations: bool = Field(default=False)  # Skip Allen relation computation

    # WLST pipeline configuration
    wlst_mode: bool = Field(default=False)
    wlst_icd_prefixes: list[str] = Field(default=["S06"])
    wlst_gcs_threshold: int = Field(default=8)
    wlst_observation_window_hours: int = Field(default=48)
    wlst_icu_types: list[str] = Field(default=[
        "Neuro Stepdown",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        "Trauma SICU (TSICU)",
    ])
    wlst_stage: Literal["stage1", "stage2"] = Field(default="stage1")

    @model_validator(mode="after")
    def _validate_bigquery_config(self) -> "Settings":
        if self.data_source == "bigquery" and not self.bigquery_project:
            raise ValueError(
                "BIGQUERY_PROJECT is required when DATA_SOURCE='bigquery'. "
                "Set DATA_SOURCE=local to use local CSV files instead."
            )
        return self

    @model_validator(mode="after")
    def _sync_wlst_cohort_codes(self) -> "Settings":
        if self.wlst_mode:
            self.cohort_icd_codes = list(self.wlst_icd_prefixes)
        return self
