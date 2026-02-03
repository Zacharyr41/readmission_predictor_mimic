"""Main pipeline orchestrator for hospital readmission prediction.

Orchestrates all 5 layers of the pipeline:
1. Ingestion: Load MIMIC-IV data to DuckDB (or use existing DB)
2. Graph: Build RDF knowledge graph from DuckDB
3. Analysis: Generate graph structure analysis report
4. Features: Extract feature matrix from RDF graph
5. Prediction: Train models and generate evaluation reports
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph

from config.settings import Settings


logger = logging.getLogger(__name__)


# Default artifact paths
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

# Default ontology directory
DEFAULT_ONTOLOGY_DIR = Path(__file__).parent.parent / "ontology" / "definition"


def run_pipeline(
    settings: Settings,
    paths: dict[str, Path] | None = None,
    ontology_dir: Path | None = None,
    skip_ingestion: bool = True,
) -> dict[str, Any]:
    """Orchestrate the complete readmission prediction pipeline.

    Pipeline stages:
    1. Ingestion: Load MIMIC-IV CSVs to DuckDB (optional, skip if DB exists)
    2. Graph Construction: Build RDF graph with cohort filtering
    3. Graph Analysis: Generate structure analysis report
    4. Feature Extraction: Build feature matrix from RDF
    5. Prediction: Train models, evaluate, generate reports

    Args:
        settings: Pipeline configuration settings
        paths: Override default artifact paths (uses DEFAULT_PATHS if None)
        ontology_dir: Path to ontology files (uses DEFAULT_ONTOLOGY_DIR if None)
        skip_ingestion: If True, skip CSV loading and use existing DuckDB

    Returns:
        Dictionary containing:
            - cohort_size: Number of patients in cohort
            - graph_triples: Number of RDF triples
            - feature_shape: (n_admissions, n_features)
            - metrics: Model evaluation metrics
            - artifact_paths: Paths to generated artifacts
    """
    paths = paths or DEFAULT_PATHS.copy()
    ontology_dir = ontology_dir or DEFAULT_ONTOLOGY_DIR

    result = {
        "cohort_size": 0,
        "graph_triples": 0,
        "feature_shape": (0, 0),
        "metrics": {},
        "artifact_paths": paths,
    }

    # Use the DuckDB path from settings
    db_path = settings.duckdb_path

    # Stage 1: Ingestion (optional)
    if not skip_ingestion and settings.mimic_iv_path.exists():
        logger.info("Stage 1: Loading MIMIC-IV to DuckDB...")
        from src.ingestion.mimic_loader import load_mimic_to_duckdb

        conn = load_mimic_to_duckdb(settings.mimic_iv_path, db_path)
        conn.close()
    else:
        logger.info("Stage 1: Skipping ingestion (using existing DuckDB)")

    # Stage 2: Graph Construction
    logger.info("Stage 2: Building RDF knowledge graph...")
    from src.graph_construction.pipeline import build_graph

    graph = build_graph(
        db_path=db_path,
        ontology_dir=ontology_dir,
        output_path=paths["rdf"],
        icd_prefixes=settings.cohort_icd_codes,
        patients_limit=settings.patients_limit,
        biomarkers_limit=settings.biomarkers_limit,
        vitals_limit=settings.vitals_limit,
        diagnoses_limit=settings.diagnoses_limit,
        skip_allen_relations=settings.skip_allen_relations,
    )

    result["graph_triples"] = len(graph)
    logger.info(f"  Graph has {len(graph)} triples")

    # Count cohort size from graph
    cohort_query = """
    SELECT (COUNT(DISTINCT ?patient) AS ?count)
    WHERE {
        ?patient rdf:type mimic:Patient .
    }
    """
    cohort_result = list(graph.query(cohort_query))
    if cohort_result:
        result["cohort_size"] = int(cohort_result[0][0])
    logger.info(f"  Cohort size: {result['cohort_size']} patients")

    # Stage 3: Graph Analysis
    logger.info("Stage 3: Generating graph analysis report...")
    from src.graph_analysis.analysis import generate_analysis_report

    paths["analysis_report"].parent.mkdir(parents=True, exist_ok=True)
    generate_analysis_report(graph, output_path=paths["analysis_report"])
    logger.info(f"  Report saved to {paths['analysis_report']}")

    # Stage 4: Feature Extraction
    logger.info("Stage 4: Extracting features...")
    from src.feature_extraction.feature_builder import build_feature_matrix

    feature_df = build_feature_matrix(graph, save_path=paths["features"])
    result["feature_shape"] = feature_df.shape
    logger.info(f"  Feature matrix shape: {feature_df.shape}")

    # Stage 5: Model Training and Evaluation
    logger.info("Stage 5: Training and evaluating models...")

    # Check if we have enough data for training
    if len(feature_df) < 6:
        logger.warning(
            f"Only {len(feature_df)} samples - too few for train/val/test split. "
            "Skipping model training."
        )
        return result

    if "subject_id" not in feature_df.columns:
        logger.warning("subject_id not in feature matrix. Skipping model training.")
        return result

    # Check class distribution
    target_col = "readmitted_30d"
    if target_col not in feature_df.columns:
        logger.warning(f"Target column {target_col} not found. Skipping training.")
        return result

    n_positive = feature_df[target_col].sum()
    n_negative = len(feature_df) - n_positive

    if n_positive == 0 or n_negative == 0:
        logger.warning(
            f"Only one class present ({n_positive} positive, {n_negative} negative). "
            "Skipping model training."
        )
        return result

    # Need at least 2 samples of each class for stratified split
    if n_positive < 2 or n_negative < 2:
        logger.warning(
            f"Insufficient class samples ({n_positive} positive, {n_negative} negative). "
            "Skipping model training."
        )
        return result

    result["metrics"] = _train_and_evaluate(feature_df, paths)

    return result


def _train_and_evaluate(
    feature_df: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Train models and generate evaluation reports.

    Args:
        feature_df: Feature matrix with labels
        paths: Artifact paths

    Returns:
        Dictionary with model metrics
    """
    from src.prediction.split import patient_level_split
    from src.prediction.model import train_model, save_model
    from src.prediction.evaluate import (
        evaluate_model,
        get_feature_importance,
        generate_evaluation_report,
    )

    metrics = {}
    target_col = "readmitted_30d"

    # Identify feature columns (exclude identifiers and labels)
    exclude_cols = ["hadm_id", "subject_id", "readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    # Patient-level split
    try:
        train_df, val_df, test_df = patient_level_split(
            feature_df,
            target_col=target_col,
            subject_col="subject_id",
        )
    except ValueError as e:
        logger.warning(f"Could not perform stratified split: {e}")
        return metrics

    logger.info(
        f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    if len(test_df) == 0:
        logger.warning("Empty test set. Skipping evaluation.")
        return metrics

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train Logistic Regression
    try:
        logger.info("  Training Logistic Regression...")
        lr_model = train_model(X_train, y_train, model_type="logistic_regression")
        save_model(lr_model, paths["model_lr"])

        lr_metrics = evaluate_model(lr_model, X_test, y_test)
        lr_importance = get_feature_importance(lr_model, feature_cols)
        generate_evaluation_report(lr_metrics, lr_importance, paths["eval_lr"])

        metrics["lr"] = lr_metrics
        logger.info(f"    AUROC: {lr_metrics['auroc']:.4f}")
    except Exception as e:
        logger.warning(f"  LR training failed: {e}")

    # Train XGBoost
    try:
        logger.info("  Training XGBoost...")
        xgb_model = train_model(X_train, y_train, model_type="xgboost")
        save_model(xgb_model, paths["model_xgb"])

        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        xgb_importance = get_feature_importance(xgb_model, feature_cols)
        generate_evaluation_report(xgb_metrics, xgb_importance, paths["eval_xgb"])

        metrics["xgb"] = xgb_metrics
        logger.info(f"    AUROC: {xgb_metrics['auroc']:.4f}")
    except Exception as e:
        logger.warning(f"  XGBoost training failed: {e}")

    return metrics


def main():
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the hospital readmission prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".env"),
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--patients-limit",
        type=int,
        default=0,
        help="Maximum number of patients to process (0 = no limit)",
    )
    parser.add_argument(
        "--biomarkers-limit",
        type=int,
        default=0,
        help="Maximum biomarker events per ICU stay (0 = no limit)",
    )
    parser.add_argument(
        "--vitals-limit",
        type=int,
        default=0,
        help="Maximum vital events per ICU stay (0 = no limit)",
    )
    parser.add_argument(
        "--diagnoses-limit",
        type=int,
        default=0,
        help="Maximum diagnoses per admission (0 = no limit)",
    )
    parser.add_argument(
        "--skip-allen",
        action="store_true",
        help="Skip Allen temporal relation computation (faster)",
    )
    parser.add_argument(
        "--icd-codes",
        nargs="+",
        default=["I63", "I61", "I60"],
        help="ICD-10 code prefixes for cohort selection",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        default=True,
        help="Skip MIMIC CSV loading (use existing DuckDB)",
    )
    parser.add_argument(
        "--run-ingestion",
        action="store_true",
        help="Run MIMIC CSV ingestion (overrides --skip-ingestion)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--verbose",
        "-v",
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

    # Override settings from CLI args
    updates = {}
    if args.patients_limit > 0:
        updates["patients_limit"] = args.patients_limit
    if args.biomarkers_limit > 0:
        updates["biomarkers_limit"] = args.biomarkers_limit
    if args.vitals_limit > 0:
        updates["vitals_limit"] = args.vitals_limit
    if args.diagnoses_limit > 0:
        updates["diagnoses_limit"] = args.diagnoses_limit
    if args.skip_allen:
        updates["skip_allen_relations"] = True
    if args.icd_codes:
        updates["cohort_icd_codes"] = args.icd_codes

    if updates:
        settings = settings.model_copy(update=updates)

    # Setup paths
    paths = DEFAULT_PATHS.copy()
    if args.output_dir:
        paths = {
            "duckdb": args.output_dir / "mimiciv.duckdb",
            "rdf": args.output_dir / "knowledge_graph.rdf",
            "features": args.output_dir / "feature_matrix.parquet",
            "analysis_report": args.output_dir / "graph_analysis.md",
            "model_lr": args.output_dir / "logistic_regression.pkl",
            "model_xgb": args.output_dir / "xgboost.json",
            "eval_lr": args.output_dir / "evaluation_lr.md",
            "eval_xgb": args.output_dir / "evaluation_xgb.md",
        }

    # Determine ingestion behavior
    skip_ingestion = not args.run_ingestion

    # Run the pipeline
    logger.info("Starting readmission prediction pipeline...")
    result = run_pipeline(
        settings=settings,
        paths=paths,
        skip_ingestion=skip_ingestion,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Cohort size: {result['cohort_size']} patients")
    print(f"Graph triples: {result['graph_triples']}")
    print(f"Feature matrix: {result['feature_shape'][0]} x {result['feature_shape'][1]}")

    if result["metrics"]:
        print("\nModel Performance:")
        for model_name, metrics in result["metrics"].items():
            print(f"  {model_name.upper()}: AUROC={metrics['auroc']:.4f}")

    print("\nArtifacts:")
    for name, path in result["artifact_paths"].items():
        status = "exists" if Path(path).exists() else "not created"
        print(f"  {name}: {path} ({status})")


if __name__ == "__main__":
    main()
