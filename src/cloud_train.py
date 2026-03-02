"""Cloud training orchestrator for Vertex AI.

Runs the full pipeline: BigQuery ingestion -> RDF graph -> feature extraction
-> GNN preparation -> experiment training, then uploads outputs to GCS.

Supports two pipeline modes:
  - "readmission" (default): Stroke readmission prediction
  - "wlst": WLST prediction in severe TBI patients

Environment variables:
    GCP_PROJECT (required): GCP project ID for BigQuery billing
    GCS_OUTPUT_BUCKET (required): GCS bucket for saving results
    ADC_CREDENTIALS_B64 (required): Base64-encoded ADC JSON credentials
    PIPELINE_MODE (default: readmission): Pipeline mode ("readmission" or "wlst")
    EXPERIMENT (default: E6_full_model): Experiment name from registry
    SEED (default: 42): Random seed
    PATIENTS_LIMIT (default: 0): Limit cohort size (0 = no limit)
    SKIP_ALLEN (default: 1): Skip Allen temporal relation computation
    RUN_ALL (default: 0): Run all ablation experiments
    WLST_STAGE (default: stage1): WLST stage ("stage1" or "stage2")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Artifact paths (local to the container)
DB_PATH = Path("data/processed/mimiciv.duckdb")
RDF_PATH = Path("data/processed/knowledge_graph.nt")
FEATURES_PATH = Path("data/features/feature_matrix.parquet")
EMBEDDINGS_PATH = Path("data/processed/concept_embeddings.pt")
HETERO_PATH = Path("data/processed/full_hetero_graph.pt")
MAPPINGS_DIR = Path("data/mappings")
ONTOLOGY_DIR = Path("ontology/definition")
OUTPUT_DIR = Path("outputs")

# WLST-specific paths
WLST_RDF_PATH = Path("data/processed/wlst_knowledge_graph.nt")
WLST_FEATURES_PATH = Path("data/features/wlst_feature_matrix.parquet")


def _write_adc_credentials() -> Path:
    """Decode ADC credentials from base64 env var and write to a temp file."""
    adc_b64 = os.environ.get("ADC_CREDENTIALS_B64")
    if not adc_b64:
        raise RuntimeError(
            "ADC_CREDENTIALS_B64 environment variable is required. "
            "The submission script should inject this from Secret Manager."
        )
    creds_json = base64.b64decode(adc_b64).decode("utf-8")
    creds_path = Path(tempfile.mktemp(suffix=".json", prefix="adc_"))
    creds_path.write_text(creds_json)
    logger.info("ADC credentials written to %s", creds_path)
    return creds_path


def _upload_to_gcs(bucket_name: str, run_id: str, project: str | None = None) -> None:
    """Upload the outputs/ directory to GCS."""
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    output_dir = OUTPUT_DIR

    if not output_dir.exists():
        logger.warning("No outputs/ directory to upload")
        return

    count = 0
    for local_path in output_dir.rglob("*"):
        if local_path.is_file():
            blob_path = f"runs/{run_id}/{local_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            count += 1

    logger.info("Uploaded %d files to gs://%s/runs/%s/", count, bucket_name, run_id)


def main() -> None:
    pipeline_start = time.time()

    # Read environment
    gcp_project = os.environ.get("GCP_PROJECT")
    gcs_bucket = os.environ.get("GCS_OUTPUT_BUCKET")
    pipeline_mode = os.environ.get("PIPELINE_MODE", "readmission")
    experiment = os.environ.get("EXPERIMENT", "E6_full_model")
    seed = int(os.environ.get("SEED", "42"))
    patients_limit = int(os.environ.get("PATIENTS_LIMIT", "0"))
    skip_allen = os.environ.get("SKIP_ALLEN", "1") == "1"
    run_all = os.environ.get("RUN_ALL", "0") == "1"
    wlst_stage = os.environ.get("WLST_STAGE", "stage1")

    if not gcp_project:
        raise RuntimeError("GCP_PROJECT environment variable is required")
    if not gcs_bucket:
        raise RuntimeError("GCS_OUTPUT_BUCKET environment variable is required")

    run_id = f"{pipeline_mode}_{experiment}_{int(time.time())}"
    logger.info("=== Cloud Training Run: %s (mode=%s) ===", run_id, pipeline_mode)
    logger.info(
        "Config: project=%s bucket=%s experiment=%s patients_limit=%d seed=%d skip_allen=%s run_all=%s",
        gcp_project, gcs_bucket, experiment, patients_limit, seed, skip_allen, run_all,
    )

    # ── Step 1: Write ADC credentials from env var ──
    logger.info("Step 1: Setting up ADC credentials...")
    creds_path = _write_adc_credentials()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    # Dispatch to appropriate pipeline
    if pipeline_mode == "wlst":
        _run_wlst_pipeline(
            gcp_project, gcs_bucket, run_id, seed, patients_limit,
            skip_allen, wlst_stage, pipeline_start,
            run_all=run_all, experiment=experiment,
        )
    else:
        _run_readmission_pipeline(
            gcp_project, gcs_bucket, run_id, experiment, seed,
            patients_limit, skip_allen, run_all, pipeline_start,
        )


def _run_wlst_pipeline(
    gcp_project: str,
    gcs_bucket: str,
    run_id: str,
    seed: int,
    patients_limit: int,
    skip_allen: bool,
    wlst_stage: str,
    pipeline_start: float,
    *,
    run_all: bool = False,
    experiment: str = "",
) -> None:
    """Run the WLST prediction pipeline."""
    import shutil

    import duckdb

    step_timings: dict[str, float] = {}

    # ── Step 2: BigQuery ingestion ──
    step_start = time.time()
    logger.info("Step 2: BigQuery ingestion...")
    from config.settings import Settings
    from src.ingestion import load_mimic_data

    settings = Settings(
        data_source="bigquery",
        bigquery_project=gcp_project,
        duckdb_path=DB_PATH,
        patients_limit=patients_limit,
        skip_allen_relations=skip_allen,
        snomed_mappings_dir=MAPPINGS_DIR,
        wlst_mode=True,
    )
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = load_mimic_data(settings)

    # Fail-fast: verify TBI diagnoses were actually loaded
    tbi_count = conn.execute(
        "SELECT COUNT(DISTINCT hadm_id) FROM diagnoses_icd "
        "WHERE icd_version = 10 AND icd_code LIKE 'S06%'"
    ).fetchone()[0]
    conn.close()
    if tbi_count == 0:
        raise RuntimeError(
            "No TBI diagnoses (S06.x) found after ingestion. "
            "Verify cohort_icd_codes includes 'S06' and BigQuery data is accessible."
        )
    logger.info("  TBI diagnoses found: %d admissions", tbi_count)
    step_timings["ingestion"] = time.time() - step_start
    logger.info("Step 2 complete: Ingestion (%.1f min)", step_timings["ingestion"] / 60)

    # ── Step 3: WLST graph construction ──
    step_start = time.time()
    logger.info("Step 3: Building WLST RDF knowledge graph (stage=%s)...", wlst_stage)
    from src.wlst.graph_pipeline import build_wlst_graph

    WLST_RDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph, labels_df = build_wlst_graph(
        db_path=DB_PATH,
        ontology_dir=ONTOLOGY_DIR,
        output_path=WLST_RDF_PATH,
        patients_limit=patients_limit,
        skip_allen_relations=skip_allen,
        snomed_mappings_dir=MAPPINGS_DIR,
        stage=wlst_stage,
    )
    logger.info("Step 3 complete: WLST graph — %d triples (%.1f min)", len(graph), (time.time() - step_start) / 60)

    if len(labels_df) == 0:
        raise RuntimeError(
            "WLST graph produced 0 patients. Check GCS threshold/ICU types."
        )

    # Graph analysis (before closing disk graph)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = OUTPUT_DIR / "wlst" / wlst_stage
    report_dir.mkdir(parents=True, exist_ok=True)

    from src.graph_analysis.analysis import generate_analysis_report
    graph_report_md, _nx_graph = generate_analysis_report(graph)
    (report_dir / "graph_analysis.md").write_text(graph_report_md)

    # Release disk-backed graph
    from src.graph_construction.disk_graph import close_disk_graph
    close_disk_graph(graph)
    step_timings["graph_construction"] = time.time() - step_start

    # ── Step 4: Feature extraction ──
    step_start = time.time()
    logger.info("Step 4: Extracting WLST 48h features...")
    from src.wlst.features import extract_wlst_features

    feat_conn = duckdb.connect(str(DB_PATH), read_only=True)
    WLST_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df = extract_wlst_features(
        feat_conn, labels_df,
        stage=wlst_stage,
        mappings_dir=MAPPINGS_DIR,
    )
    feature_df.to_parquet(WLST_FEATURES_PATH)
    feat_conn.close()
    step_timings["feature_extraction"] = time.time() - step_start
    logger.info("Step 4 complete: Features %s (%.1f min)", feature_df.shape, step_timings["feature_extraction"] / 60)

    # Save patient IDs
    patient_ids = sorted(labels_df["subject_id"].unique().tolist())
    (report_dir / "patient_ids.json").write_text(json.dumps(patient_ids))

    # Save feature summary stats
    feature_stats = feature_df.describe(include="all").to_json()
    (report_dir / "feature_stats.json").write_text(feature_stats)

    # ── Step 5: Cohort summary ──
    from src.wlst.cohort import generate_cohort_summary

    summary = generate_cohort_summary(labels_df)
    (report_dir / "cohort_summary.md").write_text(summary)

    # ── Step 6: Classical ML baselines ──
    step_start = time.time()
    logger.info("Step 6: Training classical ML baselines...")
    from src.wlst.experiments import run_classical_baselines

    baseline_dir = report_dir / "baselines"
    baseline_results = run_classical_baselines(
        feature_df, output_dir=baseline_dir, seed=seed,
    )

    from src.wlst.evaluate import generate_wlst_evaluation_report
    for model_name, result in baseline_results.items():
        report = generate_wlst_evaluation_report(
            result["metrics"], model_name, wlst_stage,
        )
        (report_dir / f"{model_name}_evaluation.md").write_text(report)

    step_timings["baselines"] = time.time() - step_start
    logger.info("Step 6 complete: Baselines (%.1f min)", step_timings["baselines"] / 60)

    # ── Step 7: GNN preparation (SapBERT embeddings + HeteroData) ──
    wlst_embeddings_path = Path("data/processed/wlst_concept_embeddings.pt")
    wlst_hetero_path = Path("data/processed/wlst_hetero_graph.pt")

    step_start = time.time()
    logger.info("Step 7: Preparing GNN data (embeddings + HeteroData export)...")
    from src.gnn.__main__ import prepare as gnn_prepare

    gnn_prepare(
        rdf_path=WLST_RDF_PATH,
        features_path=WLST_FEATURES_PATH,
        embeddings_path=wlst_embeddings_path,
        output_path=wlst_hetero_path,
        mappings_dir=MAPPINGS_DIR,
        label_mode="wlst",
    )
    step_timings["gnn_preparation"] = time.time() - step_start
    logger.info("Step 7 complete: HeteroData exported (%.1f min)", step_timings["gnn_preparation"] / 60)

    # Save HeteroData metadata
    meta_path = wlst_hetero_path.with_suffix(".meta.json")
    if meta_path.exists():
        shutil.copy(meta_path, report_dir / "hetero_meta.json")

    # ── Step 8: GNN training ──
    step_start = time.time()
    logger.info("Step 8: Running WLST GNN experiments...")
    from src.gnn.experiments import ExperimentRunner
    from src.wlst.experiments import WLST_EXPERIMENT_REGISTRY, get_wlst_gnn_registry

    wlst_gnn_registry = get_wlst_gnn_registry()
    runner = ExperimentRunner(
        wlst_hetero_path,
        base_output_dir=report_dir / "gnn_experiments",
        registry=wlst_gnn_registry,
    )

    if run_all:
        # Run all GNN experiments matching the current stage
        stage_experiments = [
            name for name, cfg in WLST_EXPERIMENT_REGISTRY.items()
            if cfg.stage == wlst_stage and name in wlst_gnn_registry
        ]
        results = runner.run_all(experiments=stage_experiments, seed=seed)
        comparison = ExperimentRunner.compare(results)
        logger.info("All WLST experiments complete:\n%s", comparison)
        (report_dir / "comparison.md").write_text(comparison)
        wlst_experiment = f"all({','.join(stage_experiments)})"
    elif experiment in wlst_gnn_registry:
        result = runner.run(experiment, seed=seed)
        auroc = result["eval_metrics"].get("auroc", "N/A")
        logger.info("Experiment %s complete: AUROC=%s", experiment, auroc)
        wlst_experiment = experiment
    else:
        # Default: W4 (Stage 1 full model) or W6 (Stage 2 full model)
        wlst_experiment = "W4_full_model" if wlst_stage == "stage1" else "W6_stage2_full_model"
        if wlst_experiment in wlst_gnn_registry:
            result = runner.run(wlst_experiment, seed=seed)
            auroc = result["eval_metrics"].get("auroc", "N/A")
            logger.info("Experiment %s complete: AUROC=%s", wlst_experiment, auroc)

    step_timings["gnn_training"] = time.time() - step_start
    logger.info("Step 8 complete: GNN training (%.1f min)", step_timings["gnn_training"] / 60)

    elapsed = time.time() - pipeline_start
    logger.info("Total WLST pipeline time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Save run metadata
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps({
        "run_id": run_id,
        "pipeline_mode": "wlst",
        "wlst_stage": wlst_stage,
        "wlst_experiment": wlst_experiment,
        "run_all": run_all,
        "experiment": experiment,
        "gcp_project": gcp_project,
        "seed": seed,
        "patients_limit": patients_limit,
        "skip_allen": skip_allen,
        "elapsed_seconds": elapsed,
        "step_timings_seconds": step_timings,
    }, indent=2))

    # Upload to GCS
    logger.info("Uploading outputs to GCS...")
    _upload_to_gcs(gcs_bucket, run_id, project=gcp_project)
    logger.info("=== WLST cloud training complete: %s ===", run_id)


def _run_readmission_pipeline(
    gcp_project: str,
    gcs_bucket: str,
    run_id: str,
    experiment: str,
    seed: int,
    patients_limit: int,
    skip_allen: bool,
    run_all: bool,
    pipeline_start: float,
) -> None:
    """Run the original readmission prediction pipeline."""

    # ── Step 2: BigQuery ingestion ──
    step_start = time.time()
    logger.info("Step 2/6: BigQuery ingestion...")
    from config.settings import Settings
    from src.ingestion import load_mimic_data

    settings = Settings(
        data_source="bigquery",
        bigquery_project=gcp_project,
        duckdb_path=DB_PATH,
        patients_limit=patients_limit,
        skip_allen_relations=skip_allen,
        snomed_mappings_dir=MAPPINGS_DIR,
    )
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = load_mimic_data(settings)
    conn.close()
    logger.info("Step 2/6 complete: Ingestion (%.1f min)", (time.time() - step_start) / 60)

    # ── Step 3: RDF graph construction ──
    step_start = time.time()
    logger.info("Step 3/6: Building RDF knowledge graph...")
    from src.graph_construction.pipeline import build_graph

    RDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph(
        db_path=DB_PATH,
        ontology_dir=ONTOLOGY_DIR,
        output_path=RDF_PATH,
        icd_prefixes=settings.cohort_icd_codes,
        patients_limit=settings.patients_limit,
        biomarkers_limit=settings.biomarkers_limit,
        vitals_limit=settings.vitals_limit,
        diagnoses_limit=settings.diagnoses_limit,
        skip_allen_relations=settings.skip_allen_relations,
        snomed_mappings_dir=settings.snomed_mappings_dir,
        umls_api_key=settings.umls_api_key,
    )
    logger.info("Step 3/6 complete: RDF graph — %d triples (%.1f min)", len(graph), (time.time() - step_start) / 60)

    # ── Step 4: Feature extraction (graph must stay open for SPARQL queries) ──
    step_start = time.time()
    logger.info("Step 4/6: Extracting features...")
    import duckdb

    from src.feature_extraction.feature_builder import build_feature_matrix
    from src.graph_analysis.analysis import generate_analysis_report
    from src.ingestion.derived_tables import select_neurology_cohort

    # Generate nx_graph for graph-structural features
    _report, nx_graph = generate_analysis_report(graph)

    feat_conn = duckdb.connect(str(DB_PATH), read_only=True)
    cohort_df = select_neurology_cohort(feat_conn, settings.cohort_icd_codes)
    if settings.patients_limit > 0:
        limited = cohort_df["subject_id"].unique()[: settings.patients_limit]
        cohort_df = cohort_df[cohort_df["subject_id"].isin(limited)]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df = build_feature_matrix(
        graph,
        conn=feat_conn,
        cohort_df=cohort_df,
        save_path=FEATURES_PATH,
        nx_graph=nx_graph,
    )
    feat_conn.close()

    # Release the disk-backed graph (no longer needed after feature extraction)
    from src.graph_construction.disk_graph import close_disk_graph
    close_disk_graph(graph)

    logger.info("Step 4/6 complete: Features %s (%.1f min)", feature_df.shape, (time.time() - step_start) / 60)

    # ── Step 5: GNN preparation (SapBERT embeddings + HeteroData) ──
    step_start = time.time()
    logger.info("Step 5/6: Preparing GNN data (embeddings + HeteroData export)...")
    from src.gnn.__main__ import prepare

    prepare(
        rdf_path=RDF_PATH,
        features_path=FEATURES_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        output_path=HETERO_PATH,
        mappings_dir=MAPPINGS_DIR,
    )
    logger.info("Step 5/6 complete: HeteroData exported (%.1f min)", (time.time() - step_start) / 60)

    # ── Step 6: GNN training ──
    step_start = time.time()
    logger.info("Step 6/6: Running GNN experiments...")
    from src.gnn.experiments import ExperimentRunner

    runner = ExperimentRunner(HETERO_PATH, base_output_dir=OUTPUT_DIR / "gnn_experiments")

    if run_all:
        results = runner.run_all(seed=seed)
        comparison = ExperimentRunner.compare(results)
        logger.info("All experiments complete:\n%s", comparison)
        # Save comparison table
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "comparison.md").write_text(comparison)
    else:
        result = runner.run(experiment, seed=seed)
        auroc = result["eval_metrics"].get("auroc", "N/A")
        logger.info("Experiment %s complete: AUROC=%s", experiment, auroc)
    logger.info("Step 6/6 complete: Training (%.1f min)", (time.time() - step_start) / 60)

    elapsed = time.time() - pipeline_start
    logger.info("Total pipeline time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Save run metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps({
        "run_id": run_id,
        "gcp_project": gcp_project,
        "experiment": experiment,
        "run_all": run_all,
        "seed": seed,
        "patients_limit": patients_limit,
        "skip_allen": skip_allen,
        "elapsed_seconds": elapsed,
    }, indent=2))

    # ── Upload outputs to GCS ──
    logger.info("Uploading outputs to GCS...")
    _upload_to_gcs(gcs_bucket, run_id, project=gcp_project)
    logger.info("=== Cloud training complete: %s ===", run_id)


if __name__ == "__main__":
    main()
