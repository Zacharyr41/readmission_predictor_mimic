#!/usr/bin/env python3
"""Model Comparison Example: Compare Logistic Regression vs XGBoost.

This example demonstrates how to load trained models, evaluate them on test
data, and compare their performance.

Usage:
    python docs/examples/05_model_comparison.py

Prerequisites:
    - Run the pipeline first to train models:
      python -m src.main --patients-limit 200 --skip-allen
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.model import load_model
from src.prediction.split import patient_level_split
from src.prediction.evaluate import evaluate_model, get_feature_importance


def load_data_and_models():
    """Load feature matrix and trained models.

    Returns:
        Tuple of (feature_df, lr_model, xgb_model)
    """
    # Check paths
    feature_path = Path("data/features/feature_matrix.parquet")
    lr_path = Path("outputs/models/logistic_regression.pkl")
    xgb_path = Path("outputs/models/xgboost.json")

    missing = []
    if not feature_path.exists():
        missing.append(str(feature_path))
    if not lr_path.exists():
        missing.append(str(lr_path))
    if not xgb_path.exists():
        missing.append(str(xgb_path))

    if missing:
        raise FileNotFoundError(
            f"Missing files: {', '.join(missing)}. "
            "Run the pipeline first: python -m src.main --patients-limit 200 --skip-allen"
        )

    print("Loading data and models...")
    df = pd.read_parquet(feature_path)
    lr_model = load_model(lr_path)
    xgb_model = load_model(xgb_path)

    print(f"  Feature matrix: {df.shape}")
    print(f"  Logistic Regression: loaded")
    print(f"  XGBoost: loaded")

    return df, lr_model, xgb_model


def prepare_data(df: pd.DataFrame):
    """Prepare train/test split.

    Returns:
        Tuple of (X_test, y_test, feature_cols)
    """
    # Define columns
    exclude_cols = ["hadm_id", "subject_id", "readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = "readmitted_30d"

    # Patient-level split
    train_df, val_df, test_df = patient_level_split(
        df,
        target_col=target_col,
        subject_col="subject_id",
    )

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    return X_test, y_test, feature_cols


def compare_metrics(lr_model, xgb_model, X_test, y_test):
    """Compare model performance metrics.

    Args:
        lr_model: Trained Logistic Regression
        xgb_model: Trained XGBoost
        X_test: Test features
        y_test: Test labels
    """
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)

    # Evaluate both models
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    # Print comparison table
    metrics_list = ["auroc", "auprc", "precision", "recall", "f1", "threshold"]

    print(f"\n{'Metric':<15} {'Logistic Reg':>15} {'XGBoost':>15} {'Winner':<15}")
    print("-" * 60)

    for metric in metrics_list:
        lr_val = lr_metrics[metric]
        xgb_val = xgb_metrics[metric]

        # Determine winner (higher is better for all these metrics)
        if metric == "threshold":
            winner = "-"  # No winner for threshold
        elif lr_val > xgb_val:
            winner = "LR"
        elif xgb_val > lr_val:
            winner = "XGBoost"
        else:
            winner = "Tie"

        print(f"{metric:<15} {lr_val:>15.4f} {xgb_val:>15.4f} {winner:<15}")

    # Overall winner
    print("\n" + "-" * 60)
    lr_auroc = lr_metrics["auroc"]
    xgb_auroc = xgb_metrics["auroc"]

    if xgb_auroc > lr_auroc:
        diff = xgb_auroc - lr_auroc
        print(f"XGBoost outperforms LR by {diff:.4f} AUROC ({100*diff/lr_auroc:.1f}% relative improvement)")
    elif lr_auroc > xgb_auroc:
        diff = lr_auroc - xgb_auroc
        print(f"LR outperforms XGBoost by {diff:.4f} AUROC ({100*diff/xgb_auroc:.1f}% relative improvement)")
    else:
        print("Models perform equally on AUROC.")

    return lr_metrics, xgb_metrics


def compare_confusion_matrices(lr_metrics, xgb_metrics):
    """Compare confusion matrices."""
    print("\n" + "=" * 60)
    print("Confusion Matrix Comparison")
    print("=" * 60)

    models = [("Logistic Regression", lr_metrics), ("XGBoost", xgb_metrics)]

    for name, metrics in models:
        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()

        print(f"\n{name}:")
        print(f"  True Negatives:  {tn:>5}")
        print(f"  False Positives: {fp:>5}")
        print(f"  False Negatives: {fn:>5}")
        print(f"  True Positives:  {tp:>5}")

        # Additional derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"  Specificity:     {specificity:>5.4f}")
        print(f"  NPV:             {npv:>5.4f}")


def compare_feature_importance(lr_model, xgb_model, feature_cols, top_n: int = 15):
    """Compare feature importance between models.

    Args:
        lr_model: Trained Logistic Regression
        xgb_model: Trained XGBoost
        feature_cols: List of feature names
        top_n: Number of top features to show
    """
    print("\n" + "=" * 60)
    print(f"Top {top_n} Feature Importance Comparison")
    print("=" * 60)

    # Get importance for both models
    lr_importance = get_feature_importance(lr_model, feature_cols)
    xgb_importance = get_feature_importance(xgb_model, feature_cols)

    # Normalize importance to sum to 1
    lr_importance["importance_norm"] = lr_importance["importance"] / lr_importance["importance"].sum()
    xgb_importance["importance_norm"] = xgb_importance["importance"] / xgb_importance["importance"].sum()

    # Get top features from each model
    lr_top = set(lr_importance.head(top_n)["feature"])
    xgb_top = set(xgb_importance.head(top_n)["feature"])

    print(f"\nLogistic Regression top {top_n}:")
    for i, row in lr_importance.head(top_n).iterrows():
        marker = "*" if row["feature"] in xgb_top else " "
        print(f"  {marker} {row['feature']:<35} {row['importance_norm']:.4f}")

    print(f"\nXGBoost top {top_n}:")
    for i, row in xgb_importance.head(top_n).iterrows():
        marker = "*" if row["feature"] in lr_top else " "
        print(f"  {marker} {row['feature']:<35} {row['importance_norm']:.4f}")

    # Agreement analysis
    overlap = lr_top & xgb_top
    print(f"\nFeature agreement:")
    print(f"  Features in both top-{top_n}: {len(overlap)}")
    print(f"  Agreement rate: {100*len(overlap)/top_n:.1f}%")

    if overlap:
        print(f"\n  Shared important features: {', '.join(sorted(overlap))}")


def probability_distribution(lr_model, xgb_model, X_test, y_test):
    """Compare predicted probability distributions."""
    print("\n" + "=" * 60)
    print("Predicted Probability Distribution")
    print("=" * 60)

    # Get predictions
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    models = [("Logistic Regression", lr_proba), ("XGBoost", xgb_proba)]

    for name, proba in models:
        print(f"\n{name}:")
        print(f"  Mean predicted probability: {proba.mean():.4f}")
        print(f"  Std predicted probability:  {proba.std():.4f}")
        print(f"  Min predicted probability:  {proba.min():.4f}")
        print(f"  Max predicted probability:  {proba.max():.4f}")

        # Quantiles
        quantiles = [0.25, 0.5, 0.75, 0.9]
        print(f"  Quantiles: ", end="")
        print(", ".join([f"P{int(q*100)}={np.quantile(proba, q):.3f}" for q in quantiles]))

    # Correlation between model predictions
    corr = np.corrcoef(lr_proba, xgb_proba)[0, 1]
    print(f"\nPrediction correlation between models: {corr:.4f}")


def main():
    """Run model comparison."""
    try:
        df, lr_model, xgb_model = load_data_and_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    try:
        X_test, y_test, feature_cols = prepare_data(df)
    except ValueError as e:
        print(f"Error preparing data: {e}")
        print("Likely insufficient data for train/test split.")
        return

    if len(X_test) == 0:
        print("Error: Empty test set. Cannot compare models.")
        return

    lr_metrics, xgb_metrics = compare_metrics(lr_model, xgb_model, X_test, y_test)
    compare_confusion_matrices(lr_metrics, xgb_metrics)
    compare_feature_importance(lr_model, xgb_model, feature_cols)
    probability_distribution(lr_model, xgb_model, X_test, y_test)

    print("\n" + "=" * 60)
    print("Model comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
