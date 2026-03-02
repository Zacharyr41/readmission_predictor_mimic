"""CLI entry point for the WLST prediction pipeline.

Usage:
    python -m src.wlst.main --patients-limit 20 --wlst-stage stage1
    python -m src.wlst.main --wlst-stage stage2 --skip-graph
"""

import argparse
import logging
import time
from pathlib import Path

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DB_PATH = Path("data/processed/mimiciv.duckdb")
RDF_PATH = Path("data/processed/wlst_knowledge_graph.nt")
FEATURES_PATH = Path("data/features/wlst_feature_matrix.parquet")
MAPPINGS_DIR = Path("data/mappings")
ONTOLOGY_DIR = Path("ontology/definition")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="WLST Prediction Pipeline")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--rdf-path", type=Path, default=RDF_PATH)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--mappings-dir", type=Path, default=MAPPINGS_DIR)
    parser.add_argument("--ontology-dir", type=Path, default=ONTOLOGY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--patients-limit", type=int, default=0)
    parser.add_argument("--wlst-stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--gcs-threshold", type=int, default=8)
    parser.add_argument("--window-hours", type=int, default=48)
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph construction")
    parser.add_argument("--skip-allen", action="store_true", help="Skip Allen relations")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipeline_start = time.time()
    logger.info("=== WLST Pipeline: stage=%s ===", args.wlst_stage)

    # ── Step 1: Cohort selection + WLST labels ──
    step_start = time.time()
    logger.info("Step 1: TBI cohort selection + WLST label derivation...")
    from src.wlst.cohort import (
        create_wlst_labels,
        generate_cohort_summary,
        select_tbi_cohort,
    )

    conn = duckdb.connect(str(args.db_path))

    # Ensure age table exists
    from src.ingestion.derived_tables import create_age_table
    try:
        conn.execute("SELECT 1 FROM age LIMIT 1")
    except duckdb.CatalogException:
        create_age_table(conn)

    cohort_df = select_tbi_cohort(
        conn,
        gcs_threshold=args.gcs_threshold,
        observation_window_hours=args.window_hours,
        patients_limit=args.patients_limit,
    )

    if len(cohort_df) == 0:
        logger.error("No patients in TBI cohort. Check data or criteria.")
        conn.close()
        return

    labels_df = create_wlst_labels(conn, cohort_df)
    conn.close()

    # Save cohort summary
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = generate_cohort_summary(labels_df)
    (report_dir / "wlst_cohort_summary.md").write_text(summary)
    logger.info("Step 1 complete (%.1fs)", time.time() - step_start)

    # ── Step 2: Graph construction ──
    if not args.skip_graph:
        step_start = time.time()
        logger.info("Step 2: Building WLST RDF graph...")
        from src.wlst.graph_pipeline import build_wlst_graph

        graph, labels_df = build_wlst_graph(
            db_path=args.db_path,
            ontology_dir=args.ontology_dir,
            output_path=args.rdf_path,
            gcs_threshold=args.gcs_threshold,
            observation_window_hours=args.window_hours,
            patients_limit=args.patients_limit,
            skip_allen_relations=args.skip_allen,
            snomed_mappings_dir=args.mappings_dir,
            n_workers=args.n_workers,
            stage=args.wlst_stage,
        )
        logger.info("Step 2 complete: %d triples (%.1fs)", len(graph), time.time() - step_start)
    else:
        logger.info("Step 2: Skipping graph construction")

    # ── Step 3: Feature extraction ──
    step_start = time.time()
    logger.info("Step 3: Extracting 48h tabular features...")
    from src.wlst.features import extract_wlst_features

    feat_conn = duckdb.connect(str(args.db_path), read_only=True)
    feature_df = extract_wlst_features(
        feat_conn, labels_df,
        observation_window_hours=args.window_hours,
        stage=args.wlst_stage,
        mappings_dir=args.mappings_dir,
    )
    feat_conn.close()

    # Save feature matrix
    args.features_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(args.features_path)
    logger.info("Step 3 complete: %s features (%.1fs)", feature_df.shape, time.time() - step_start)

    # ── Step 4: Classical ML baselines ──
    if not args.skip_training:
        step_start = time.time()
        logger.info("Step 4: Training classical ML baselines...")
        from src.wlst.experiments import run_classical_baselines

        baseline_dir = args.output_dir / "wlst" / args.wlst_stage / "baselines"
        baseline_results = run_classical_baselines(
            feature_df, output_dir=baseline_dir, seed=args.seed,
        )

        # Save baseline evaluation reports
        from src.wlst.evaluate import compute_wlst_metrics, generate_wlst_evaluation_report
        for model_name, result in baseline_results.items():
            report = generate_wlst_evaluation_report(
                result["metrics"], model_name, args.wlst_stage,
            )
            (report_dir / f"wlst_{args.wlst_stage}_{model_name}_evaluation.md").write_text(report)

        logger.info("Step 4 complete (%.1fs)", time.time() - step_start)
    else:
        logger.info("Step 4: Skipping training")

    elapsed = time.time() - pipeline_start
    logger.info("=== WLST Pipeline complete: %.1f seconds (%.1f min) ===", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
