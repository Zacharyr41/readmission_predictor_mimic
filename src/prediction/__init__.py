"""Prediction module for hospital readmission prediction.

This module provides functions for model training, evaluation, and reporting
for predicting 30-day hospital readmissions using clinical features.

Data Splitting:
- Patient-level stratified splits to prevent data leakage
- Preserves class balance across train/val/test sets

Model Training:
- Logistic Regression with balanced class weights
- XGBoost with automatic scale_pos_weight computation
- Model save/load with JSON (XGBoost) or pickle (sklearn)

Evaluation:
- Standard metrics: AUROC, AUPRC, precision, recall, F1
- Youden's J threshold selection
- Feature importance extraction
- Calibration curve analysis
- Markdown report generation
"""

from src.prediction.split import (
    patient_level_split,
)
from src.prediction.model import (
    train_model,
    save_model,
    load_model,
)
from src.prediction.evaluate import (
    compute_metrics,
    evaluate_model,
    get_feature_importance,
    calibration_data,
    generate_evaluation_report,
)

__all__ = [
    # Data splitting
    "patient_level_split",
    # Model training and persistence
    "train_model",
    "save_model",
    "load_model",
    # Evaluation and reporting
    "compute_metrics",
    "evaluate_model",
    "get_feature_importance",
    "calibration_data",
    "generate_evaluation_report",
]
