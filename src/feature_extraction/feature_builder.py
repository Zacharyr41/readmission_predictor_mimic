"""Feature matrix builder combining all feature extractors.

This module combines tabular and graph features into a single feature matrix
for machine learning model training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
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
    extract_snomed_diagnosis_features,
    extract_snomed_medication_features,
)
from src.feature_extraction.graph_features import (
    extract_temporal_features,
    extract_graph_structure_features,
)
from src.feature_extraction.sql_features import (
    extract_demographics_sql,
    extract_stay_features_sql,
    extract_lab_summary_sql,
    extract_vital_summary_sql,
    extract_medication_features_sql,
    extract_diagnosis_features_sql,
    extract_labels_sql,
    extract_subject_ids_sql,
)

logger = logging.getLogger(__name__)


def _extract_subject_ids(graph: Graph) -> pd.DataFrame:
    """Extract subject_id for each admission via SPARQL.

    This is needed for patient-level train/test splitting to prevent data leakage.

    Args:
        graph: RDF graph containing patient and admission data.

    Returns:
        DataFrame with columns: hadm_id, subject_id
    """
    query = """
    SELECT ?hadmId ?subjectId
    WHERE {
        ?patient rdf:type mimic:Patient ;
                 mimic:hasSubjectId ?subjectId ;
                 mimic:hasAdmission ?admission .
        ?admission mimic:hasAdmissionId ?hadmId .
    }
    """

    results = list(graph.query(query))

    data = []
    for row in results:
        hadm_id = int(row[0])
        subject_id = int(row[1])

        data.append({
            "hadm_id": hadm_id,
            "subject_id": subject_id,
        })

    return pd.DataFrame(data)


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

    Missing value handling strategy based on clinical domain knowledge:

    1. Count/binary features (fill with 0):
       - Count columns (_count, num_, total_): Missing means no events recorded
       - Binary indicators (has_, gender_, admission_type_, icd_chapter_): Missing
         means the condition/category was not present

    2. Continuous features (fill with median):
       - Lab values, vital statistics: Use cohort median to avoid introducing
         extreme values that could bias the model
       - If median is NaN (all values missing), default to 0

    This approach preserves the clinical interpretation of missing data while
    avoiding information leakage from future/test data.

    Args:
        df: DataFrame with potential missing values.
        label_cols: List of label column names to exclude from filling.

    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()

    for col in df.columns:
        # Skip identifiers and labels - these should never have missing values
        if col in label_cols or col in ["hadm_id", "subject_id"]:
            continue

        if df[col].isna().any():
            # Identify column type based on naming conventions
            # Count columns: represent number of events (0 if no events)
            is_count = any(pattern in col.lower() for pattern in [
                "_count", "num_", "has_", "total_"
            ])
            # Binary/one-hot columns: represent categorical membership
            is_binary = any(pattern in col.lower() for pattern in [
                "gender_", "admission_type_", "icd_chapter_"
            ])

            if is_count or is_binary:
                # Missing counts/indicators imply absence (0)
                df[col] = df[col].fillna(0)
            else:
                # Continuous features: use median to avoid extreme value bias
                # Example: missing lab values imputed with cohort median
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    return df


def build_feature_matrix(
    graph: Graph,
    conn: duckdb.DuckDBPyConnection | None = None,
    cohort_df: pd.DataFrame | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Build complete feature matrix from RDF graph and/or DuckDB.

    When ``conn`` and ``cohort_df`` are provided, the 6 tabular feature
    extractors run as DuckDB SQL (orders of magnitude faster than SPARQL).
    The 4 graph-dependent extractors (SNOMED mappings, temporal relations,
    graph topology) still use SPARQL.

    When ``conn`` is None, all extractors use SPARQL (backward compatible).

    Args:
        graph: RDF graph containing the clinical knowledge graph.
        conn: Optional DuckDB connection with MIMIC tables and derived tables.
        cohort_df: Optional cohort DataFrame (subject_id, hadm_id, stay_id).
            Required when ``conn`` is provided.
        save_path: Optional path to save the feature matrix as parquet.

    Returns:
        DataFrame with all features and labels, one row per admission.
    """
    use_sql = conn is not None and cohort_df is not None

    if use_sql:
        # Register cohort as temp table for SQL extractors
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM cohort_df"
        )
        logger.info("Using DuckDB SQL for tabular feature extraction")

        # SQL path: 6 tabular extractors + labels + subject IDs
        demographics = extract_demographics_sql(conn)
        stay_features = extract_stay_features_sql(conn)
        lab_summary = extract_lab_summary_sql(conn)
        vital_summary = extract_vital_summary_sql(conn)
        medication_features = extract_medication_features_sql(conn)
        diagnosis_features = extract_diagnosis_features_sql(conn)
        labels = extract_labels_sql(conn)
        subject_ids = extract_subject_ids_sql(conn)
    else:
        logger.info("Using SPARQL for all feature extraction")

        # SPARQL path: all extractors via RDF graph
        demographics = extract_demographics(graph)
        stay_features = extract_stay_features(graph)
        lab_summary = extract_lab_summary(graph)
        vital_summary = extract_vital_summary(graph)
        medication_features = extract_medication_features(graph)
        diagnosis_features = extract_diagnosis_features(graph)
        labels = _extract_labels(graph)
        subject_ids = _extract_subject_ids(graph)

    # Graph-dependent extractors always use SPARQL
    snomed_diagnosis_features = extract_snomed_diagnosis_features(graph)
    snomed_medication_features = extract_snomed_medication_features(graph)
    temporal_features = extract_temporal_features(graph)
    graph_structure_features = extract_graph_structure_features(graph)

    # Merge all features on hadm_id
    # Start with labels as the base (ensures we have all admissions)
    df = labels.copy()

    # Add subject_id for patient-level splitting
    if not subject_ids.empty:
        df = df.merge(subject_ids, on="hadm_id", how="left")

    feature_dfs = [
        demographics,
        stay_features,
        lab_summary,
        vital_summary,
        medication_features,
        diagnosis_features,
        snomed_diagnosis_features,
        snomed_medication_features,
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
