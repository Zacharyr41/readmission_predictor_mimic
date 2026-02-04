#!/usr/bin/env python3
"""Custom Cohort Example: Use different ICD codes for cohort selection.

This example demonstrates how to run the pipeline with a custom cohort
defined by different ICD-10 diagnosis codes.

Examples:
    - Epilepsy: G40, G41
    - Heart Failure: I50
    - Diabetes: E10, E11
    - Pneumonia: J12-J18

Usage:
    python docs/examples/02_custom_cohort.py
"""

from pathlib import Path

from config.settings import Settings
from src.main import run_pipeline


def run_with_cohort(icd_codes: list[str], name: str, patients_limit: int = 100):
    """Run pipeline with a specific ICD cohort.

    Args:
        icd_codes: List of ICD-10 code prefixes
        name: Human-readable cohort name
        patients_limit: Maximum patients to process
    """
    print(f"\n{'=' * 60}")
    print(f"Running pipeline for: {name}")
    print(f"ICD codes: {icd_codes}")
    print(f"{'=' * 60}")

    settings = Settings()
    settings = settings.model_copy(update={
        "cohort_icd_codes": icd_codes,
        "patients_limit": patients_limit,
        "skip_allen_relations": True,
        "biomarkers_limit": 500,
        "vitals_limit": 500,
    })

    # Custom output paths for this cohort
    cohort_slug = name.lower().replace(" ", "_")
    paths = {
        "duckdb": Path("data/processed/mimiciv.duckdb"),  # Shared
        "rdf": Path(f"data/processed/kg_{cohort_slug}.rdf"),
        "features": Path(f"data/features/features_{cohort_slug}.parquet"),
        "analysis_report": Path(f"outputs/reports/analysis_{cohort_slug}.md"),
        "model_lr": Path(f"outputs/models/lr_{cohort_slug}.pkl"),
        "model_xgb": Path(f"outputs/models/xgb_{cohort_slug}.json"),
        "eval_lr": Path(f"outputs/reports/eval_lr_{cohort_slug}.md"),
        "eval_xgb": Path(f"outputs/reports/eval_xgb_{cohort_slug}.md"),
    }

    result = run_pipeline(
        settings=settings,
        paths=paths,
        skip_ingestion=True,
    )

    print(f"\nResults for {name}:")
    print(f"  Cohort size: {result['cohort_size']} patients")
    print(f"  Graph triples: {result['graph_triples']:,}")

    if result["metrics"] and "xgb" in result["metrics"]:
        print(f"  XGBoost AUROC: {result['metrics']['xgb']['auroc']:.4f}")
    else:
        print("  No model trained (insufficient data)")

    return result


def main():
    """Demonstrate custom cohort selection."""

    # Example 1: Stroke patients (default cohort)
    stroke_result = run_with_cohort(
        icd_codes=["I63", "I61", "I60"],  # Ischemic, hemorrhagic, subarachnoid
        name="Stroke",
        patients_limit=100,
    )

    # Example 2: Epilepsy patients
    epilepsy_result = run_with_cohort(
        icd_codes=["G40", "G41"],  # Epilepsy, Status epilepticus
        name="Epilepsy",
        patients_limit=100,
    )

    # Example 3: Heart failure patients
    hf_result = run_with_cohort(
        icd_codes=["I50"],  # Heart failure
        name="Heart Failure",
        patients_limit=100,
    )

    # Example 4: Respiratory infections
    pneumonia_result = run_with_cohort(
        icd_codes=["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
        name="Pneumonia",
        patients_limit=100,
    )

    # Summary comparison
    print("\n" + "=" * 60)
    print("Cohort Comparison Summary")
    print("=" * 60)

    cohorts = [
        ("Stroke", stroke_result),
        ("Epilepsy", epilepsy_result),
        ("Heart Failure", hf_result),
        ("Pneumonia", pneumonia_result),
    ]

    print(f"\n{'Cohort':<15} {'Patients':<10} {'Triples':<12} {'AUROC':<8}")
    print("-" * 50)

    for name, result in cohorts:
        n_patients = result["cohort_size"]
        n_triples = result["graph_triples"]

        if result["metrics"] and "xgb" in result["metrics"]:
            auroc = f"{result['metrics']['xgb']['auroc']:.4f}"
        else:
            auroc = "N/A"

        print(f"{name:<15} {n_patients:<10} {n_triples:<12,} {auroc:<8}")


if __name__ == "__main__":
    main()
