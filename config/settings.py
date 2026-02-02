from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    mimic_iv_path: Path = Field(default=Path("/Users/zacharyrothstein/Code/NeuroResearch"))
    duckdb_path: Path = Field(default=Path("data/processed/mimiciv.duckdb"))
    clinical_tkg_repo: Path = Field(default=Path("/Users/zacharyrothstein/Code/clinical-tkg-cmls2025"))

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # Cohort configuration
    cohort_icd_codes: list[str] = Field(default=["I63", "I61", "I60"])
    readmission_window_days: int = Field(default=30)
    patients_limit: int = Field(default=0)  # 0 = no limit
    biomarkers_limit: int = Field(default=0)  # 0 = no limit


settings = Settings()
