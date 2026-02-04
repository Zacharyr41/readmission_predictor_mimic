#!/usr/bin/env python3
"""Quick Start Example: Run the minimal end-to-end pipeline.

This example demonstrates how to run the complete readmission prediction
pipeline with a small cohort for testing purposes.

Usage:
    python docs/examples/01_quick_start.py
"""

from pathlib import Path

from config.settings import Settings
from src.main import run_pipeline


def main():
    """Run a minimal end-to-end pipeline."""
    # Load default settings from .env
    settings = Settings()

    # Override for quick testing:
    # - Limit to 50 patients
    # - Skip Allen relations (much faster)
    # - Limit events to reduce graph size
    settings = settings.model_copy(update={
        "patients_limit": 50,
        "skip_allen_relations": True,
        "biomarkers_limit": 500,
        "vitals_limit": 500,
        "diagnoses_limit": 50,
    })

    print("=" * 60)
    print("Quick Start: Hospital Readmission Prediction Pipeline")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Patients limit: {settings.patients_limit}")
    print(f"  - Skip Allen relations: {settings.skip_allen_relations}")
    print(f"  - ICD codes: {settings.cohort_icd_codes}")
    print()

    # Run the pipeline
    # skip_ingestion=True assumes DuckDB is already loaded
    # Set to False if this is the first run
    result = run_pipeline(
        settings=settings,
        skip_ingestion=True,  # Set to False if DuckDB not yet created
    )

    # Print results
    print("\n" + "=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    print(f"\nCohort size: {result['cohort_size']} patients")
    print(f"Graph triples: {result['graph_triples']:,}")
    print(f"Feature matrix shape: {result['feature_shape']}")

    if result["metrics"]:
        print("\nModel Performance:")
        for model_name, metrics in result["metrics"].items():
            print(f"\n  {model_name.upper()}:")
            print(f"    AUROC: {metrics['auroc']:.4f}")
            print(f"    AUPRC: {metrics['auprc']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
    else:
        print("\nNo model metrics (insufficient data for training)")

    print("\nArtifacts created:")
    for name, path in result["artifact_paths"].items():
        exists = Path(path).exists()
        status = "OK" if exists else "Not created"
        print(f"  [{status}] {name}: {path}")

    return result


if __name__ == "__main__":
    main()
