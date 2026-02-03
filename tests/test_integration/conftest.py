"""Integration test fixtures."""

import pytest
from pathlib import Path


# Default artifact paths used by the pipeline
DEFAULT_PATHS = {
    "duckdb": Path("data/processed/mimiciv.duckdb"),
    "rdf": Path("data/processed/knowledge_graph.rdf"),
    "features": Path("data/features/feature_matrix.parquet"),
    "analysis_report": Path("outputs/reports/graph_analysis.md"),
    "model_lr": Path("outputs/models/logistic_regression.pkl"),
    "model_xgb": Path("outputs/models/xgboost.json"),
    "eval_lr": Path("outputs/reports/evaluation_lr.md"),
    "eval_xgb": Path("outputs/reports/evaluation_xgb.md"),
}


@pytest.fixture
def temp_artifact_paths(tmp_path: Path) -> dict[str, Path]:
    """Override default paths to use tmp_path for testing."""
    return {
        "duckdb": tmp_path / "mimiciv.duckdb",
        "rdf": tmp_path / "knowledge_graph.rdf",
        "features": tmp_path / "feature_matrix.parquet",
        "analysis_report": tmp_path / "graph_analysis.md",
        "model_lr": tmp_path / "logistic_regression.pkl",
        "model_xgb": tmp_path / "xgboost.json",
        "eval_lr": tmp_path / "evaluation_lr.md",
        "eval_xgb": tmp_path / "evaluation_xgb.md",
    }
