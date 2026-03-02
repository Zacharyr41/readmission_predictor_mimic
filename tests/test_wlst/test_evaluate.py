"""Tests for WLST evaluation metrics and stage comparison."""

import numpy as np
import pytest

from src.wlst.evaluate import (
    compare_stages,
    compute_wlst_metrics,
    generate_stage_comparison_report,
    generate_wlst_evaluation_report,
)


@pytest.fixture
def binary_predictions():
    """Synthetic binary predictions for testing metrics."""
    rng = np.random.RandomState(42)
    n = 100
    y_test = np.array([0] * 60 + [1] * 40)
    y_proba = rng.beta(2, 5, size=n)
    y_proba[y_test == 1] += 0.3
    y_proba = np.clip(y_proba, 0, 1)
    return y_proba, y_test


class TestComputeWlstMetrics:
    def test_returns_expected_keys(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        expected_keys = {
            "auroc", "auprc", "brier_score", "sensitivity", "specificity",
            "optimal_threshold", "sensitivity_at_90_specificity",
            "sensitivity_at_95_specificity", "confusion_matrix",
            "n_test", "n_positive", "n_negative",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_auroc_range(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_brier_score_range(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_sensitivity_at_specificity(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        assert 0.0 <= metrics["sensitivity_at_90_specificity"] <= 1.0
        assert 0.0 <= metrics["sensitivity_at_95_specificity"] <= 1.0

    def test_confusion_matrix_shape(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_perfect_prediction(self):
        y_test = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        metrics = compute_wlst_metrics(y_proba, y_test)
        assert metrics["auroc"] == 1.0

    def test_sample_counts(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        assert metrics["n_test"] == len(y_test)
        assert metrics["n_positive"] == y_test.sum()


class TestCompareStages:
    def test_basic_comparison(self):
        s1 = {"auroc": 0.75, "auprc": 0.60}
        s2 = {"auroc": 0.80, "auprc": 0.65}
        result = compare_stages(s1, s2)
        assert result["auroc_delta"] == pytest.approx(0.05)
        assert result["auprc_delta"] == pytest.approx(0.05)

    def test_bootstrap_ci(self, binary_predictions):
        y_proba, y_test = binary_predictions
        s1 = compute_wlst_metrics(y_proba, y_test)
        # Slightly better stage 2
        y_proba2 = np.clip(y_proba + 0.05, 0, 1)
        s2 = compute_wlst_metrics(y_proba2, y_test)

        result = compare_stages(
            s1, s2,
            stage1_proba=y_proba,
            stage2_proba=y_proba2,
            y_test=y_test,
            n_bootstrap=100,
        )
        assert "auroc_diff_ci_lower" in result
        assert "auroc_diff_ci_upper" in result


class TestReportGeneration:
    def test_evaluation_report(self, binary_predictions):
        y_proba, y_test = binary_predictions
        metrics = compute_wlst_metrics(y_proba, y_test)
        report = generate_wlst_evaluation_report(metrics, "test_model")
        assert "# WLST Evaluation Report" in report
        assert "AUROC" in report

    def test_comparison_report(self):
        comparison = {
            "stage1_auroc": 0.75, "stage2_auroc": 0.80, "auroc_delta": 0.05,
            "stage1_auprc": 0.60, "stage2_auprc": 0.65, "auprc_delta": 0.05,
        }
        report = generate_stage_comparison_report(comparison)
        assert "Stage 1 vs Stage 2" in report
        assert "Stage 1" in report
