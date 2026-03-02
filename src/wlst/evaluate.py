"""WLST-specific evaluation metrics and Stage 1 vs Stage 2 comparison.

Extends the base prediction evaluation with WLST-specific analyses:
- Sensitivity at high specificity (clinical utility)
- Subgroup analysis by outcome category
- Stage 1 vs Stage 2 comparison with statistical testing
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_wlst_metrics(
    y_proba: np.ndarray,
    y_test: np.ndarray,
    outcome_categories: pd.Series | None = None,
) -> dict:
    """Compute WLST-specific evaluation metrics.

    Args:
        y_proba: Predicted probabilities for WLST (positive class).
        y_test: True binary WLST labels.
        outcome_categories: Optional series of outcome categories for subgroup analysis.

    Returns:
        Dictionary of metrics.
    """
    y_test = np.asarray(y_test)
    y_proba = np.asarray(y_proba)

    # Standard metrics
    auroc = roc_auc_score(y_test, y_proba)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall_curve, precision_curve)

    brier = brier_score_loss(y_test, y_proba)

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Sensitivity at high specificity thresholds
    sens_at_90_spec = _sensitivity_at_specificity(fpr, tpr, target_specificity=0.90)
    sens_at_95_spec = _sensitivity_at_specificity(fpr, tpr, target_specificity=0.95)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "optimal_threshold": optimal_threshold,
        "sensitivity_at_90_specificity": sens_at_90_spec,
        "sensitivity_at_95_specificity": sens_at_95_spec,
        "confusion_matrix": cm,
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
        "n_negative": int(len(y_test) - y_test.sum()),
    }

    # Subgroup analysis
    if outcome_categories is not None:
        metrics["subgroup_analysis"] = _subgroup_analysis(
            y_proba, y_test, outcome_categories,
        )

    return metrics


def _sensitivity_at_specificity(
    fpr: np.ndarray, tpr: np.ndarray, target_specificity: float,
) -> float:
    """Find sensitivity at a given specificity threshold."""
    target_fpr = 1.0 - target_specificity
    # Find the tpr at the closest fpr <= target_fpr
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def _subgroup_analysis(
    y_proba: np.ndarray, y_test: np.ndarray, categories: pd.Series,
) -> dict:
    """Compute metrics per outcome category subgroup."""
    results = {}
    for cat in categories.unique():
        mask = categories == cat
        n = mask.sum()
        if n < 5:
            continue
        sub_y = y_test[mask]
        sub_proba = y_proba[mask]
        # Only compute AUROC if both classes present
        if len(np.unique(sub_y)) < 2:
            results[cat] = {"n": n, "auroc": None, "mean_proba": float(sub_proba.mean())}
        else:
            results[cat] = {
                "n": n,
                "auroc": float(roc_auc_score(sub_y, sub_proba)),
                "mean_proba": float(sub_proba.mean()),
            }
    return results


def compare_stages(
    stage1_metrics: dict,
    stage2_metrics: dict,
    stage1_proba: np.ndarray | None = None,
    stage2_proba: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare Stage 1 vs Stage 2 performance.

    Args:
        stage1_metrics: Metrics dict from Stage 1 evaluation.
        stage2_metrics: Metrics dict from Stage 2 evaluation.
        stage1_proba: Stage 1 predicted probabilities (for bootstrap).
        stage2_proba: Stage 2 predicted probabilities (for bootstrap).
        y_test: True labels (same test set for both stages).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Comparison results dict.
    """
    comparison = {
        "stage1_auroc": stage1_metrics["auroc"],
        "stage2_auroc": stage2_metrics["auroc"],
        "auroc_delta": stage2_metrics["auroc"] - stage1_metrics["auroc"],
        "stage1_auprc": stage1_metrics["auprc"],
        "stage2_auprc": stage2_metrics["auprc"],
        "auprc_delta": stage2_metrics["auprc"] - stage1_metrics["auprc"],
    }

    # Bootstrap confidence interval for AUROC difference
    if stage1_proba is not None and stage2_proba is not None and y_test is not None:
        rng = np.random.RandomState(seed)
        auroc_diffs = []
        n = len(y_test)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_y = y_test[idx]
            if len(np.unique(boot_y)) < 2:
                continue
            s1_auroc = roc_auc_score(boot_y, stage1_proba[idx])
            s2_auroc = roc_auc_score(boot_y, stage2_proba[idx])
            auroc_diffs.append(s2_auroc - s1_auroc)

        auroc_diffs = np.array(auroc_diffs)
        comparison["auroc_diff_ci_lower"] = float(np.percentile(auroc_diffs, 2.5))
        comparison["auroc_diff_ci_upper"] = float(np.percentile(auroc_diffs, 97.5))
        comparison["auroc_diff_p_value"] = float((auroc_diffs <= 0).mean())

    return comparison


def generate_wlst_evaluation_report(
    metrics: dict,
    model_name: str,
    stage: str = "stage1",
) -> str:
    """Generate markdown evaluation report for a WLST model.

    Args:
        metrics: Dict from compute_wlst_metrics().
        model_name: Name of the model/experiment.
        stage: "stage1" or "stage2".

    Returns:
        Markdown-formatted report string.
    """
    cm = metrics.get("confusion_matrix")
    cm_str = ""
    if cm is not None:
        cm_str = f"""
## Confusion Matrix

| | Predicted No WLST | Predicted WLST |
|---|---|---|
| **Actual No WLST** | {cm[0, 0]} | {cm[0, 1]} |
| **Actual WLST** | {cm[1, 0]} | {cm[1, 1]} |
"""

    subgroup_str = ""
    if "subgroup_analysis" in metrics:
        lines = ["## Subgroup Analysis\n", "| Category | N | AUROC | Mean P(WLST) |", "|---|---|---|---|"]
        for cat, data in metrics["subgroup_analysis"].items():
            auroc_str = f"{data['auroc']:.4f}" if data.get("auroc") is not None else "N/A"
            lines.append(f"| {cat} | {data['n']} | {auroc_str} | {data['mean_proba']:.4f} |")
        subgroup_str = "\n".join(lines)

    report = f"""# WLST Evaluation Report — {model_name} ({stage})

## Performance Metrics

| Metric | Value |
|---|---|
| **AUROC** | {metrics['auroc']:.4f} |
| **AUPRC** | {metrics['auprc']:.4f} |
| **Brier Score** | {metrics['brier_score']:.4f} |
| **Sensitivity** | {metrics['sensitivity']:.4f} |
| **Specificity** | {metrics['specificity']:.4f} |
| **Sens @ 90% Spec** | {metrics['sensitivity_at_90_specificity']:.4f} |
| **Sens @ 95% Spec** | {metrics['sensitivity_at_95_specificity']:.4f} |
| **Test Set Size** | {metrics['n_test']} (pos={metrics['n_positive']}, neg={metrics['n_negative']}) |
{cm_str}
{subgroup_str}
"""
    return report


def generate_stage_comparison_report(comparison: dict) -> str:
    """Generate markdown report comparing Stage 1 vs Stage 2."""
    ci_str = ""
    if "auroc_diff_ci_lower" in comparison:
        ci_str = f"""
## Statistical Comparison (Bootstrap, n=1000)

| Metric | Value |
|---|---|
| **AUROC Difference** | {comparison['auroc_delta']:.4f} |
| **95% CI** | [{comparison['auroc_diff_ci_lower']:.4f}, {comparison['auroc_diff_ci_upper']:.4f}] |
| **p-value** | {comparison['auroc_diff_p_value']:.4f} |
"""

    report = f"""# WLST Stage 1 vs Stage 2 Comparison

## Summary

| Metric | Stage 1 | Stage 2 | Delta |
|---|---|---|---|
| **AUROC** | {comparison['stage1_auroc']:.4f} | {comparison['stage2_auroc']:.4f} | {comparison['auroc_delta']:+.4f} |
| **AUPRC** | {comparison['stage1_auprc']:.4f} | {comparison['stage2_auprc']:.4f} | {comparison['auprc_delta']:+.4f} |
{ci_str}
"""
    return report
