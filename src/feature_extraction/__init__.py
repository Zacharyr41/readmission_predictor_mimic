"""Feature extraction module for hospital readmission prediction.

This module provides functions to extract tabular and graph structure features
from RDF clinical knowledge graphs for use in machine learning models.

Tabular Features:
- Demographics (age, gender)
- ICU stay characteristics (LOS, admission type)
- Lab value summaries (mean, min, max, std, abnormal rate)
- Vital sign summaries (mean, min, max, std, coefficient of variation)
- Medication features (antibiotic exposure)
- Diagnosis features (count, primary ICD chapter)

Graph Features:
- Temporal relation counts (Allen algebra)
- Graph structure metrics (density, degree distribution)

Feature Matrix:
- Combines all features with readmission labels
- Handles missing values with appropriate defaults
"""

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
from src.feature_extraction.feature_builder import (
    build_feature_matrix,
)

__all__ = [
    # Tabular feature extractors
    "extract_demographics",
    "extract_stay_features",
    "extract_lab_summary",
    "extract_vital_summary",
    "extract_medication_features",
    "extract_diagnosis_features",
    # Graph feature extractors
    "extract_temporal_features",
    "extract_graph_structure_features",
    # Feature matrix builder
    "build_feature_matrix",
]
