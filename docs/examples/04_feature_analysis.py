#!/usr/bin/env python3
"""Feature Analysis Example: Analyze and visualize extracted features.

This example demonstrates how to load the feature matrix and perform
exploratory data analysis on the extracted features.

Usage:
    python docs/examples/04_feature_analysis.py

Prerequisites:
    - Run the pipeline first to create the feature matrix:
      python -m src.main --patients-limit 100 --skip-allen
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_features(path: str = "data/features/feature_matrix.parquet") -> pd.DataFrame:
    """Load the feature matrix.

    Args:
        path: Path to the parquet file

    Returns:
        Feature DataFrame
    """
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. "
            "Run the pipeline first: python -m src.main --patients-limit 100 --skip-allen"
        )

    print(f"Loading features from {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")
    return df


def basic_statistics(df: pd.DataFrame) -> None:
    """Print basic statistics about the feature matrix."""
    print("=" * 60)
    print("Basic Statistics")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Identify column types
    id_cols = ["hadm_id", "subject_id"]
    label_cols = ["readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in df.columns if c not in id_cols + label_cols]

    print(f"\nColumn breakdown:")
    print(f"  Identifiers: {len(id_cols)}")
    print(f"  Labels: {len(label_cols)}")
    print(f"  Features: {len(feature_cols)}")


def class_distribution(df: pd.DataFrame) -> None:
    """Analyze class (label) distribution."""
    print("\n" + "=" * 60)
    print("Class Distribution")
    print("=" * 60)

    for label in ["readmitted_30d", "readmitted_60d"]:
        if label not in df.columns:
            continue

        counts = df[label].value_counts()
        total = len(df)

        print(f"\n{label}:")
        print(f"  Negative (0): {counts.get(0, 0)} ({100*counts.get(0, 0)/total:.1f}%)")
        print(f"  Positive (1): {counts.get(1, 0)} ({100*counts.get(1, 0)/total:.1f}%)")
        print(f"  Imbalance ratio: {counts.get(0, 1) / max(counts.get(1, 1), 1):.1f}:1")


def feature_categories(df: pd.DataFrame) -> None:
    """Categorize and count features by type."""
    print("\n" + "=" * 60)
    print("Feature Categories")
    print("=" * 60)

    id_cols = ["hadm_id", "subject_id"]
    label_cols = ["readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in df.columns if c not in id_cols + label_cols]

    categories = {
        "Demographics": ["age", "gender_M", "gender_F"],
        "Stay": [c for c in feature_cols if c.startswith(("icu_", "num_icu", "admission_type"))],
        "Lab values": [c for c in feature_cols if any(x in c for x in ["_mean", "_min", "_max", "_std", "_count", "_first", "_last", "_abnormal"])
                       and not any(v in c for v in ["Heart", "Resp", "SpO2", "Temp", "BP"])],
        "Vitals": [c for c in feature_cols if any(v in c for v in ["Heart", "Resp", "SpO2", "Temp", "BP", "_cv"])],
        "Medications": [c for c in feature_cols if any(x in c for x in ["antibiotic", "distinct_meds"])],
        "Diagnoses": [c for c in feature_cols if any(x in c for x in ["num_diagnoses", "icd_chapter"])],
        "Temporal": [c for c in feature_cols if any(x in c for x in ["events_per", "before_rel", "during_rel", "temporal_edges"])],
        "Graph structure": [c for c in feature_cols if any(x in c for x in ["subgraph", "degree"])],
    }

    # Categorize
    categorized = set()
    for cat, cols in categories.items():
        matching = [c for c in cols if c in feature_cols]
        categories[cat] = matching
        categorized.update(matching)

    # Find uncategorized
    uncategorized = [c for c in feature_cols if c not in categorized]
    if uncategorized:
        categories["Other"] = uncategorized

    print("\nFeatures by category:")
    for cat, cols in categories.items():
        if cols:
            print(f"  {cat}: {len(cols)} features")


def missing_values(df: pd.DataFrame) -> None:
    """Analyze missing values in the dataset."""
    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)

    missing = df.isna().sum()
    total_missing = missing.sum()

    print(f"\nTotal missing values: {total_missing}")

    if total_missing > 0:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        print(f"\nColumns with missing values: {len(missing_cols)}")
        print("\nTop 10 columns by missing count:")
        for col, count in missing_cols.head(10).items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("No missing values in the dataset.")


def feature_statistics(df: pd.DataFrame) -> None:
    """Print detailed statistics for key features."""
    print("\n" + "=" * 60)
    print("Key Feature Statistics")
    print("=" * 60)

    key_features = [
        "age",
        "icu_los_hours",
        "num_icu_days",
        "num_distinct_meds",
        "total_antibiotic_days",
        "num_diagnoses",
        "events_per_icu_day",
        "total_temporal_edges",
        "patient_subgraph_nodes",
    ]

    available = [f for f in key_features if f in df.columns]

    print(f"\n{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for feature in available:
        mean = df[feature].mean()
        std = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"{feature:<25} {mean:>10.2f} {std:>10.2f} {min_val:>10.2f} {max_val:>10.2f}")


def correlation_with_target(df: pd.DataFrame, target: str = "readmitted_30d") -> None:
    """Calculate correlation of features with target variable."""
    print("\n" + "=" * 60)
    print(f"Feature Correlation with {target}")
    print("=" * 60)

    if target not in df.columns:
        print(f"\nTarget column '{target}' not found.")
        return

    # Get numeric columns only
    id_cols = ["hadm_id", "subject_id"]
    label_cols = ["readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in df.columns if c not in id_cols + label_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("\nNo numeric features found.")
        return

    # Calculate correlations
    correlations = df[numeric_cols + [target]].corr()[target].drop(target)
    correlations = correlations.sort_values(key=abs, ascending=False)

    print("\nTop 15 features by absolute correlation:")
    print(f"\n{'Feature':<35} {'Correlation':>12}")
    print("-" * 50)

    for feature, corr in correlations.head(15).items():
        direction = "+" if corr > 0 else ""
        print(f"{feature:<35} {direction}{corr:>11.4f}")


def variance_analysis(df: pd.DataFrame) -> None:
    """Analyze feature variance to identify low-information features."""
    print("\n" + "=" * 60)
    print("Variance Analysis")
    print("=" * 60)

    id_cols = ["hadm_id", "subject_id"]
    label_cols = ["readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in df.columns if c not in id_cols + label_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("\nNo numeric features found.")
        return

    variances = df[numeric_cols].var().sort_values()

    # Zero variance features
    zero_var = variances[variances == 0]
    print(f"\nZero-variance features (constant): {len(zero_var)}")
    if len(zero_var) > 0 and len(zero_var) <= 10:
        for col in zero_var.index:
            print(f"  - {col}")

    # Low variance features
    threshold = 0.01
    low_var = variances[(variances > 0) & (variances < threshold)]
    print(f"\nLow-variance features (var < {threshold}): {len(low_var)}")

    # High variance features
    print("\nTop 10 highest-variance features:")
    for col, var in variances.tail(10).items():
        print(f"  {col}: {var:.4f}")


def admission_type_analysis(df: pd.DataFrame) -> None:
    """Analyze admission types and their readmission rates."""
    print("\n" + "=" * 60)
    print("Admission Type Analysis")
    print("=" * 60)

    adm_type_cols = [c for c in df.columns if c.startswith("admission_type_")]

    if not adm_type_cols:
        print("\nNo admission type columns found.")
        return

    print("\nReadmission rates by admission type:")
    print(f"{'Admission Type':<25} {'N':>8} {'Readmit 30d':>12} {'Rate':>8}")
    print("-" * 55)

    for col in adm_type_cols:
        adm_type = col.replace("admission_type_", "")
        subset = df[df[col] == 1]
        n = len(subset)

        if n > 0 and "readmitted_30d" in df.columns:
            rate = 100 * subset["readmitted_30d"].mean()
            readmit_n = int(subset["readmitted_30d"].sum())
            print(f"{adm_type:<25} {n:>8} {readmit_n:>12} {rate:>7.1f}%")


def main():
    """Run feature analysis."""
    try:
        df = load_features()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    basic_statistics(df)
    class_distribution(df)
    feature_categories(df)
    missing_values(df)
    feature_statistics(df)
    correlation_with_target(df)
    variance_analysis(df)
    admission_type_analysis(df)

    print("\n" + "=" * 60)
    print("Feature analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
