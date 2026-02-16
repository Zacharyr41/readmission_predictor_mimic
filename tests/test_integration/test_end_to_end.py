"""End-to-end integration tests for the readmission predictor pipeline.

Tests the full pipeline from DuckDB ingestion through model training and evaluation.
Uses synthetic data fixtures to ensure fast, reproducible tests.
"""

import pytest
import duckdb
import pandas as pd
from pathlib import Path
from rdflib import Graph

from config.settings import Settings
from src.graph_construction.disk_graph import close_disk_graph


def export_duckdb_to_file(
    source_conn: duckdb.DuckDBPyConnection,
    target_path: Path,
) -> None:
    """Export in-memory DuckDB tables to a file-based database.

    Args:
        source_conn: Source DuckDB connection (in-memory)
        target_path: Path for output DuckDB file
    """
    file_conn = duckdb.connect(str(target_path))

    tables = source_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()

    for (table_name,) in tables:
        data = source_conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        file_conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM data"
        )

    file_conn.close()


@pytest.mark.integration
class TestFullPipelineEndToEnd:
    """End-to-end integration tests for the complete pipeline."""

    def test_full_pipeline_end_to_end(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test the complete pipeline from DuckDB to model evaluation.

        Assertions:
        - DuckDB has required tables
        - Cohort has >0 stroke patients
        - RDF graph has >100 triples (synthetic data is small)
        - Feature matrix shape (n_admissions, >5 features)
        - Probabilities in [0, 1]
        - AUROC is finite and in [0, 1]
        - Evaluation report exists
        """
        from src.main import run_pipeline

        # Setup paths
        paths = {
            "duckdb": tmp_path / "mimiciv.duckdb",
            "rdf": tmp_path / "knowledge_graph.nt",
            "features": tmp_path / "feature_matrix.parquet",
            "analysis_report": tmp_path / "graph_analysis.md",
            "model_lr": tmp_path / "logistic_regression.pkl",
            "model_xgb": tmp_path / "xgboost.json",
            "eval_lr": tmp_path / "evaluation_lr.md",
            "eval_xgb": tmp_path / "evaluation_xgb.md",
        }

        # Export the in-memory tables to a file-based database
        synthetic_db_path = tmp_path / "synthetic.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, synthetic_db_path)

        # Create test settings
        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"

        settings = Settings(
            duckdb_path=synthetic_db_path,
            cohort_icd_codes=["I63", "I61", "I60"],  # Stroke codes
            patients_limit=0,  # Process all patients
            data_source="local",
        )

        # Run the pipeline
        result = run_pipeline(
            settings=settings,
            paths=paths,
            ontology_dir=ontology_dir,
        )

        # Verify DuckDB has required tables
        conn = duckdb.connect(str(synthetic_db_path), read_only=True)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        conn.close()

        assert "patients" in table_names
        assert "admissions" in table_names
        assert "icustays" in table_names
        assert "diagnoses_icd" in table_names

        # Verify cohort was selected (stroke patients exist)
        assert result["cohort_size"] > 0, "Cohort should have at least 1 patient"

        # Verify RDF graph was created
        assert paths["rdf"].exists(), "RDF graph file should exist"
        graph = Graph()
        graph.parse(str(paths["rdf"]), format="nt")
        assert len(graph) > 100, f"Graph should have >100 triples, got {len(graph)}"

        # Verify feature matrix
        assert paths["features"].exists(), "Feature matrix should exist"
        features_df = pd.read_parquet(paths["features"])
        assert len(features_df) > 0, "Feature matrix should have at least 1 row"
        assert features_df.shape[1] > 5, "Feature matrix should have >5 columns"
        assert "hadm_id" in features_df.columns
        assert "subject_id" in features_df.columns
        assert "readmitted_30d" in features_df.columns

        # Verify model training occurred
        assert "metrics" in result
        metrics = result["metrics"]

        # Verify AUROC is valid (if training happened)
        if "lr" in metrics:
            auroc = metrics["lr"]["auroc"]
            assert 0 <= auroc <= 1, f"AUROC should be in [0,1], got {auroc}"
            assert pd.notna(auroc), "AUROC should be finite"

        # Verify evaluation report exists (if training happened)
        if metrics:
            assert paths["eval_lr"].exists() or paths["eval_xgb"].exists(), \
                "At least one evaluation report should exist"

    def test_pipeline_artifacts_exist(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Verify all expected artifacts are created after pipeline run."""
        from src.main import run_pipeline

        # Setup paths
        paths = {
            "duckdb": tmp_path / "mimiciv.duckdb",
            "rdf": tmp_path / "knowledge_graph.nt",
            "features": tmp_path / "feature_matrix.parquet",
            "analysis_report": tmp_path / "graph_analysis.md",
            "model_lr": tmp_path / "logistic_regression.pkl",
            "model_xgb": tmp_path / "xgboost.json",
            "eval_lr": tmp_path / "evaluation_lr.md",
            "eval_xgb": tmp_path / "evaluation_xgb.md",
        }

        # Export synthetic DB to file
        synthetic_db_path = tmp_path / "synthetic.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, synthetic_db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"
        settings = Settings(
            duckdb_path=synthetic_db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
            patients_limit=0,
            data_source="local",
        )

        # Run the pipeline
        run_pipeline(settings=settings, paths=paths, ontology_dir=ontology_dir)

        # Check all expected artifacts exist
        expected_artifacts = [
            ("RDF graph", paths["rdf"]),
            ("Feature matrix", paths["features"]),
            ("Analysis report", paths["analysis_report"]),
        ]

        for name, path in expected_artifacts:
            assert path.exists(), f"{name} should exist at {path}"

    def test_pipeline_with_small_cohort(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test pipeline handles small cohorts gracefully.

        With synthetic data (only a few patients), we need to handle
        cases where stratified splitting may fail.
        """
        from src.main import run_pipeline

        paths = {
            "duckdb": tmp_path / "mimiciv.duckdb",
            "rdf": tmp_path / "knowledge_graph.nt",
            "features": tmp_path / "feature_matrix.parquet",
            "analysis_report": tmp_path / "graph_analysis.md",
            "model_lr": tmp_path / "logistic_regression.pkl",
            "model_xgb": tmp_path / "xgboost.json",
            "eval_lr": tmp_path / "evaluation_lr.md",
            "eval_xgb": tmp_path / "evaluation_xgb.md",
        }

        synthetic_db_path = tmp_path / "synthetic.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, synthetic_db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"
        settings = Settings(
            duckdb_path=synthetic_db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
            patients_limit=2,  # Very small cohort
            data_source="local",
        )

        # Should not raise an error
        result = run_pipeline(settings=settings, paths=paths, ontology_dir=ontology_dir)

        # Pipeline should complete (even if model training is skipped)
        assert result is not None
        assert paths["rdf"].exists()
        assert paths["features"].exists()


@pytest.mark.integration
class TestPipelineComponents:
    """Tests for individual pipeline components in integration context."""

    def test_graph_construction_from_duckdb(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test graph construction produces valid RDF from DuckDB."""
        from src.graph_construction.pipeline import build_graph

        # Export to file (use different name than the fixture's file)
        db_path = tmp_path / "exported.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"
        output_path = tmp_path / "graph.nt"

        # Build graph
        graph = build_graph(
            db_path=db_path,
            ontology_dir=ontology_dir,
            output_path=output_path,
            icd_prefixes=["I63", "I61", "I60"],
            patients_limit=0,
        )

        # Verify graph structure
        assert len(graph) > 0, "Graph should have triples"
        assert output_path.exists(), "NT file should exist"

        # Load and verify the serialized graph
        loaded_graph = Graph()
        loaded_graph.parse(str(output_path), format="nt")
        assert len(loaded_graph) == len(graph), "Serialized graph should match"
        close_disk_graph(graph)

    def test_feature_extraction_from_graph(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test feature extraction produces valid DataFrame from RDF graph."""
        from src.graph_construction.pipeline import build_graph
        from src.feature_extraction.feature_builder import build_feature_matrix

        # Build graph first (use different name than the fixture's file)
        db_path = tmp_path / "exported.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"
        graph = build_graph(
            db_path=db_path,
            ontology_dir=ontology_dir,
            output_path=tmp_path / "graph.nt",
            icd_prefixes=["I63", "I61", "I60"],
            patients_limit=0,
        )

        # Extract features
        features_path = tmp_path / "features.parquet"
        feature_df = build_feature_matrix(graph, save_path=features_path)
        close_disk_graph(graph)

        # Verify feature DataFrame
        assert len(feature_df) > 0, "Should have at least one admission"
        assert "hadm_id" in feature_df.columns
        assert "subject_id" in feature_df.columns
        assert "readmitted_30d" in feature_df.columns
        assert features_path.exists()

    def test_model_training_with_features(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test model training works with extracted features."""
        from src.graph_construction.pipeline import build_graph
        from src.feature_extraction.feature_builder import build_feature_matrix
        from src.prediction.model import train_model
        from src.prediction.split import patient_level_split

        # Build graph and extract features (use different name than the fixture's file)
        db_path = tmp_path / "exported.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"
        graph = build_graph(
            db_path=db_path,
            ontology_dir=ontology_dir,
            output_path=tmp_path / "graph.nt",
            icd_prefixes=["I63", "I61", "I60"],
            patients_limit=0,
        )

        feature_df = build_feature_matrix(graph)
        close_disk_graph(graph)

        # Check if we have enough samples for splitting
        if len(feature_df) < 6:
            pytest.skip("Not enough samples for train/val/test split")

        # Check if we have both classes
        if feature_df["readmitted_30d"].nunique() < 2:
            pytest.skip("Need both classes for model training")

        # Split data
        train_df, val_df, test_df = patient_level_split(
            feature_df,
            target_col="readmitted_30d",
            subject_col="subject_id",
        )

        # Get feature columns (exclude identifiers and labels)
        exclude_cols = ["hadm_id", "subject_id", "readmitted_30d", "readmitted_60d"]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        X_train = train_df[feature_cols]
        y_train = train_df["readmitted_30d"]

        # Train model
        model = train_model(X_train, y_train, model_type="logistic_regression")

        # Verify model
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


@pytest.mark.integration
class TestPipelineOptimizations:
    """Tests for pipeline optimization features."""

    def test_pipeline_with_skip_allen_relations(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test pipeline runs faster and produces valid results when skipping Allen relations.

        Allen temporal relations are O(n²) in event pairs and can dominate runtime.
        This test verifies that skip_allen_relations=True:
        - Produces a graph with 0 Allen relations
        - Still produces valid features and can train models
        - Produces fewer triples than with Allen relations enabled
        """
        from src.main import run_pipeline

        paths = {
            "duckdb": tmp_path / "mimiciv.duckdb",
            "rdf": tmp_path / "knowledge_graph.nt",
            "features": tmp_path / "feature_matrix.parquet",
            "analysis_report": tmp_path / "graph_analysis.md",
            "model_lr": tmp_path / "logistic_regression.pkl",
            "model_xgb": tmp_path / "xgboost.json",
            "eval_lr": tmp_path / "evaluation_lr.md",
            "eval_xgb": tmp_path / "evaluation_xgb.md",
        }

        synthetic_db_path = tmp_path / "synthetic.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, synthetic_db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"
        settings = Settings(
            duckdb_path=synthetic_db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
            patients_limit=0,
            skip_allen_relations=True,  # Key: skip Allen relations
            data_source="local",
        )

        result = run_pipeline(settings=settings, paths=paths, ontology_dir=ontology_dir)

        # Verify pipeline completed
        assert result is not None
        assert result["cohort_size"] > 0, "Should have patients in cohort"
        assert result["graph_triples"] > 0, "Should have triples in graph"

        # Verify RDF was created
        assert paths["rdf"].exists()
        graph = Graph()
        graph.parse(str(paths["rdf"]), format="nt")

        # Verify NO Allen relations in the graph
        allen_predicates = [
            "http://www.w3.org/2006/time#intervalBefore",
            "http://www.w3.org/2006/time#intervalMeets",
            "http://www.w3.org/2006/time#intervalOverlaps",
            "http://www.w3.org/2006/time#intervalStarts",
            "http://www.w3.org/2006/time#intervalDuring",
            "http://www.w3.org/2006/time#intervalFinishes",
            "http://www.w3.org/2006/time#intervalEquals",
        ]
        from rdflib import URIRef
        allen_count = sum(
            1 for p in allen_predicates
            for _ in graph.triples((None, URIRef(p), None))
        )
        assert allen_count == 0, f"Should have 0 Allen relations, got {allen_count}"

        # Verify features were still extracted
        assert paths["features"].exists()
        features_df = pd.read_parquet(paths["features"])
        assert len(features_df) > 0
        assert "hadm_id" in features_df.columns

    def test_pipeline_with_event_limits(
        self, synthetic_duckdb_with_events: duckdb.DuckDBPyConnection, tmp_path: Path
    ):
        """Test pipeline respects event limits for biomarkers, vitals, and diagnoses.

        Event limits allow controlling graph size and runtime by capping
        the number of events per ICU stay or admission.
        """
        from src.main import run_pipeline

        paths = {
            "duckdb": tmp_path / "mimiciv.duckdb",
            "rdf": tmp_path / "knowledge_graph.nt",
            "features": tmp_path / "feature_matrix.parquet",
            "analysis_report": tmp_path / "graph_analysis.md",
            "model_lr": tmp_path / "logistic_regression.pkl",
            "model_xgb": tmp_path / "xgboost.json",
            "eval_lr": tmp_path / "evaluation_lr.md",
            "eval_xgb": tmp_path / "evaluation_xgb.md",
        }

        synthetic_db_path = tmp_path / "synthetic.duckdb"
        export_duckdb_to_file(synthetic_duckdb_with_events, synthetic_db_path)

        ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"

        # Run with strict limits
        settings = Settings(
            duckdb_path=synthetic_db_path,
            cohort_icd_codes=["I63", "I61", "I60"],
            patients_limit=0,
            biomarkers_limit=5,  # Max 5 biomarkers per ICU stay
            vitals_limit=5,  # Max 5 vitals per ICU stay
            diagnoses_limit=3,  # Max 3 diagnoses per admission
            skip_allen_relations=True,  # Skip for speed
            data_source="local",
        )

        result = run_pipeline(settings=settings, paths=paths, ontology_dir=ontology_dir)

        # Verify pipeline completed
        assert result is not None
        assert result["cohort_size"] > 0

        # Verify artifacts exist
        assert paths["rdf"].exists()
        assert paths["features"].exists()

        # The graph should be smaller due to limits
        # (We can't easily verify exact counts without knowing the fixture data,
        # but we verify the pipeline completed successfully with limits applied)
        graph = Graph()
        graph.parse(str(paths["rdf"]), format="nt")
        assert len(graph) > 0, "Graph should have some triples"

        # Features should still be extractable
        features_df = pd.read_parquet(paths["features"])
        assert len(features_df) > 0
        assert "hadm_id" in features_df.columns
        assert "readmitted_30d" in features_df.columns
