"""Model evaluation and reporting for hospital readmission prediction.

Provides functions for computing evaluation metrics, extracting feature
importance, calibration analysis, and generating markdown evaluation reports.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


def evaluate_model(
    model: Union[LogisticRegression, XGBClassifier],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> dict:
    """Evaluate a trained model and compute performance metrics.

    Uses Youden's J statistic to select optimal classification threshold.

    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing:
            - auroc: Area under ROC curve
            - auprc: Area under precision-recall curve
            - precision: Precision at optimal threshold
            - recall: Recall at optimal threshold
            - f1: F1 score at optimal threshold
            - threshold: Optimal threshold (Youden's J)
            - confusion_matrix: Confusion matrix at optimal threshold
    """
    y_test = np.asarray(y_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # AUROC
    auroc = roc_auc_score(y_test, y_proba)

    # AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall_curve, precision_curve)

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Compute metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": optimal_threshold,
        "confusion_matrix": cm,
    }


def get_feature_importance(
    model: Union[LogisticRegression, XGBClassifier],
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importance from a trained model.

    For LogisticRegression, uses absolute coefficient values.
    For XGBoost, uses built-in feature importances.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        DataFrame with columns [feature, importance], sorted by importance descending
    """
    if isinstance(model, LogisticRegression):
        importances = np.abs(model.coef_[0])
    elif isinstance(model, XGBClassifier):
        importances = model.feature_importances_
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def calibration_data(
    model: Union[LogisticRegression, XGBClassifier],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        n_bins: Number of bins for calibration curve

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) arrays
    """
    y_test = np.asarray(y_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=n_bins, strategy="uniform"
    )

    return fraction_of_positives, mean_predicted_value


def generate_evaluation_report(
    metrics: dict,
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a markdown evaluation report.

    Creates a report with:
    - Model performance metrics (AUROC, AUPRC, precision, recall, F1)
    - Confusion matrix
    - Top-20 most important features

    Args:
        metrics: Dictionary of evaluation metrics from evaluate_model()
        feature_importance: DataFrame from get_feature_importance()
        output_path: Path to write the markdown report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format confusion matrix
    cm = metrics["confusion_matrix"]
    cm_str = f"""| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | {cm[0, 0]} | {cm[0, 1]} |
| **Actual Positive** | {cm[1, 0]} | {cm[1, 1]} |"""

    # Format top-20 features
    top_features = feature_importance.head(20)
    feature_rows = "\n".join(
        f"| {i+1} | {row['feature']} | {row['importance']:.4f} |"
        for i, row in top_features.iterrows()
    )
    features_table = f"""| Rank | Feature | Importance |
|---|---|---|
{feature_rows}"""

    report = f"""# Model Evaluation Report

## Performance Metrics

| Metric | Value |
|---|---|
| **AUROC** | {metrics['auroc']:.4f} |
| **AUPRC** | {metrics['auprc']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1 Score** | {metrics['f1']:.4f} |
| **Optimal Threshold** | {metrics['threshold']:.4f} |

## Confusion Matrix

{cm_str}

## Top-20 Feature Importance

{features_table}
"""

    output_path.write_text(report)
