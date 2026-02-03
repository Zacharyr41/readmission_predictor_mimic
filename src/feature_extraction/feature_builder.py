"""Feature matrix builder combining all feature extractors.

This module combines tabular and graph features into a single feature matrix
for machine learning model training.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from rdflib import Graph

from src.feature_extraction.tabular_features import (
    extract_demographics,
    extract_stay_features,
    extract_lab_summary,
    extract_vital_summary,
    extract_medication_features,
    extract_diagnosis_features,
)
from src.feature_extraction.graph_features import (
    extract_temporal_features,
    extract_graph_structure_features,
)


def _extract_labels(graph: Graph) -> pd.DataFrame:
    """Extract readmission labels for each admission.

    Args:
        graph: RDF graph containing admission data with readmission labels.

    Returns:
        DataFrame with columns: hadm_id, readmitted_30d, readmitted_60d
    """
    query = """
    SELECT ?hadmId ?readmitted30 ?readmitted60
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:readmittedWithin30Days ?readmitted30 ;
                   mimic:readmittedWithin60Days ?readmitted60 .
    }
    """

    results = list(graph.query(query))

    data = []
    for row in results:
        hadm_id = int(row[0])
        # Convert RDF boolean literals to Python bool
        readmitted_30d = str(row[1]).lower() == "true"
        readmitted_60d = str(row[2]).lower() == "true"

        data.append({
            "hadm_id": hadm_id,
            "readmitted_30d": int(readmitted_30d),
            "readmitted_60d": int(readmitted_60d),
        })

    return pd.DataFrame(data)


def _fill_missing_values(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    """Fill missing values with appropriate defaults.

    Strategy:
    - Count/boolean columns: fill with 0
    - Continuous columns: fill with median

    Args:
        df: DataFrame with potential missing values.
        label_cols: List of label column names to exclude from filling.

    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()

    for col in df.columns:
        if col in label_cols or col == "hadm_id":
            continue

        if df[col].isna().any():
            # Identify column type based on name patterns
            is_count = any(pattern in col.lower() for pattern in [
                "_count", "num_", "has_", "total_"
            ])
            is_binary = any(pattern in col.lower() for pattern in [
                "gender_", "admission_type_", "icd_chapter_"
            ])

            if is_count or is_binary:
                df[col] = df[col].fillna(0)
            else:
                # Use median for continuous features
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    return df


def build_feature_matrix(
    graph: Graph,
    save_path: Path | None = None
) -> pd.DataFrame:
    """Build complete feature matrix from RDF graph.

    Combines all feature extractors and readmission labels into a single
    DataFrame suitable for machine learning.

    Args:
        graph: RDF graph containing the clinical knowledge graph.
        save_path: Optional path to save the feature matrix as parquet.

    Returns:
        DataFrame with all features and labels, one row per admission.
    """
    # Extract all feature groups
    demographics = extract_demographics(graph)
    stay_features = extract_stay_features(graph)
    lab_summary = extract_lab_summary(graph)
    vital_summary = extract_vital_summary(graph)
    medication_features = extract_medication_features(graph)
    diagnosis_features = extract_diagnosis_features(graph)
    temporal_features = extract_temporal_features(graph)
    graph_structure_features = extract_graph_structure_features(graph)

    # Extract labels
    labels = _extract_labels(graph)

    # Merge all features on hadm_id
    # Start with labels as the base (ensures we have all admissions)
    df = labels.copy()

    feature_dfs = [
        demographics,
        stay_features,
        lab_summary,
        vital_summary,
        medication_features,
        diagnosis_features,
        temporal_features,
        graph_structure_features,
    ]

    for feature_df in feature_dfs:
        if not feature_df.empty and "hadm_id" in feature_df.columns:
            df = df.merge(feature_df, on="hadm_id", how="left")

    # Fill missing values
    label_cols = ["readmitted_30d", "readmitted_60d"]
    df = _fill_missing_values(df, label_cols)

    # Ensure labels are integers (0 or 1)
    for col in label_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Save to parquet if path provided
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)

    return df
