"""Cloud training orchestrator for Vertex AI.

Runs the full pipeline: BigQuery ingestion -> RDF graph -> feature extraction
-> GNN preparation -> experiment training, then uploads outputs to GCS.

Environment variables:
    GCP_PROJECT (required): GCP project ID for BigQuery billing
    GCS_OUTPUT_BUCKET (required): GCS bucket for saving results
    ADC_CREDENTIALS_B64 (required): Base64-encoded ADC JSON credentials
    EXPERIMENT (default: E6_full_model): Experiment name from registry
    SEED (default: 42): Random seed
    PATIENTS_LIMIT (default: 0): Limit cohort size (0 = no limit)
    SKIP_ALLEN (default: 1): Skip Allen temporal relation computation
    RUN_ALL (default: 0): Run all 6 ablation experiments
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
RDF_PATH = Path("data/processed/knowledge_graph.rdf")
FEATURES_PATH = Path("data/features/feature_matrix.parquet")
EMBEDDINGS_PATH = Path("data/processed/concept_embeddings.pt")
HETERO_PATH = Path("data/processed/full_hetero_graph.pt")
MAPPINGS_DIR = Path("data/mappings")
ONTOLOGY_DIR = Path("ontology/definition")
OUTPUT_DIR = Path("outputs")


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


def _upload_to_gcs(bucket_name: str, run_id: str) -> None:
    """Upload the outputs/ directory to GCS."""
    from google.cloud import storage

    client = storage.Client()
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
    experiment = os.environ.get("EXPERIMENT", "E6_full_model")
    seed = int(os.environ.get("SEED", "42"))
    patients_limit = int(os.environ.get("PATIENTS_LIMIT", "0"))
    skip_allen = os.environ.get("SKIP_ALLEN", "1") == "1"
    run_all = os.environ.get("RUN_ALL", "0") == "1"

    if not gcp_project:
        raise RuntimeError("GCP_PROJECT environment variable is required")
    if not gcs_bucket:
        raise RuntimeError("GCS_OUTPUT_BUCKET environment variable is required")

    run_id = f"{experiment}_{int(time.time())}"
    logger.info("=== Cloud Training Run: %s ===", run_id)
    logger.info(
        "Config: project=%s bucket=%s experiment=%s patients_limit=%d seed=%d skip_allen=%s run_all=%s",
        gcp_project, gcs_bucket, experiment, patients_limit, seed, skip_allen, run_all,
    )

    # ── Step 1: Write ADC credentials from env var ──
    logger.info("Step 1/6: Setting up ADC credentials...")
    creds_path = _write_adc_credentials()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

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

    # ── Step 4: Feature extraction ──
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
    _upload_to_gcs(gcs_bucket, run_id)
    logger.info("=== Cloud training complete: %s ===", run_id)


if __name__ == "__main__":
    main()
