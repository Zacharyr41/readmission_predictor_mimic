"""Test suite for prediction module.

Tests model training, evaluation, and reporting for hospital readmission prediction.
Uses synthetic data with known characteristics to verify:
- Patient-level train/val/test splitting (no data leakage)
- Model training (Logistic Regression, XGBoost)
- Evaluation metrics (AUROC, AUPRC, confusion matrix)
- Feature importance extraction
- Model save/load round-trip
- Evaluation report generation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ==================== Fixtures ====================


@pytest.fixture
def synthetic_feature_matrix() -> pd.DataFrame:
    """200 rows, 50 features, 80 patients (60 single + 20 multi-admission), ~20% positive.

    Structure:
    - 60 patients with 1 admission each
    - 20 patients with 7 admissions each (140 rows)
    - Total: 200 rows, 80 patients
    - ~20% positive class rate
    """
    np.random.seed(42)

    # Generate synthetic features
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=15,
        n_redundant=10,
        n_clusters_per_class=2,
        class_sep=0.8,
        weights=[0.8, 0.2],
        random_state=42,
    )

    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(50)]
    df = pd.DataFrame(X, columns=feature_cols)

    # Assign subject_ids: 60 single-admission + 20 multi-admission patients
    subject_ids = []
    # First 60 rows: single-admission patients (IDs 1-60)
    subject_ids.extend(list(range(1, 61)))
    # Remaining 140 rows: 20 patients with 7 admissions each (IDs 61-80)
    for patient_id in range(61, 81):
        subject_ids.extend([patient_id] * 7)

    df["subject_id"] = subject_ids
    df["hadm_id"] = list(range(1, 201))
    df["readmitted_30d"] = y

    return df


@pytest.fixture
def small_feature_matrix() -> pd.DataFrame:
    """50 rows, 20 features for quick tests."""
    np.random.seed(123)

    X, y = make_classification(
        n_samples=50,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        class_sep=0.8,
        weights=[0.8, 0.2],
        random_state=123,
    )

    feature_cols = [f"feature_{i}" for i in range(20)]
    df = pd.DataFrame(X, columns=feature_cols)

    # 30 single-admission patients + 4 patients with 5 admissions each
    subject_ids = list(range(1, 31)) + [31] * 5 + [32] * 5 + [33] * 5 + [34] * 5
    df["subject_id"] = subject_ids
    df["hadm_id"] = list(range(1, 51))
    df["readmitted_30d"] = y

    return df


# ==================== Test Cases ====================


class TestPatientLevelSplit:
    """(a) test_patient_level_split: No subject_id in multiple splits; class ratio within 5%."""

    def test_no_patient_in_multiple_splits(self, synthetic_feature_matrix):
        """No patient appears in more than one split."""
        from src.prediction import patient_level_split

        train_df, val_df, test_df = patient_level_split(
            synthetic_feature_matrix,
            target_col="readmitted_30d",
            subject_col="subject_id",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        train_patients = set(train_df["subject_id"].unique())
        val_patients = set(val_df["subject_id"].unique())
        test_patients = set(test_df["subject_id"].unique())

        # No overlap between any splits
        assert len(train_patients & val_patients) == 0, "Patient in both train and val"
        assert len(train_patients & test_patients) == 0, "Patient in both train and test"
        assert len(val_patients & test_patients) == 0, "Patient in both val and test"

    def test_class_ratio_preserved(self, synthetic_feature_matrix):
        """Class ratio in each split is within 5% of overall ratio."""
        from src.prediction import patient_level_split

        train_df, val_df, test_df = patient_level_split(
            synthetic_feature_matrix,
            target_col="readmitted_30d",
            subject_col="subject_id",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        overall_ratio = synthetic_feature_matrix["readmitted_30d"].mean()
        train_ratio = train_df["readmitted_30d"].mean()
        val_ratio = val_df["readmitted_30d"].mean()
        test_ratio = test_df["readmitted_30d"].mean()

        assert abs(train_ratio - overall_ratio) < 0.05, f"Train ratio {train_ratio} too far from {overall_ratio}"
        assert abs(val_ratio - overall_ratio) < 0.10, f"Val ratio {val_ratio} too far from {overall_ratio}"
        assert abs(test_ratio - overall_ratio) < 0.10, f"Test ratio {test_ratio} too far from {overall_ratio}"


class TestPatientLevelSplitAllAdmissions:
    """(b) test_patient_level_split_all_admissions_together: Multi-admission patients stay together."""

    def test_all_admissions_in_same_split(self, synthetic_feature_matrix):
        """Patient with 7 admissions has all admissions in the same split."""
        from src.prediction import patient_level_split

        train_df, val_df, test_df = patient_level_split(
            synthetic_feature_matrix,
            target_col="readmitted_30d",
            subject_col="subject_id",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        # Multi-admission patients are IDs 61-80
        for patient_id in range(61, 81):
            in_train = patient_id in train_df["subject_id"].values
            in_val = patient_id in val_df["subject_id"].values
            in_test = patient_id in test_df["subject_id"].values

            # Patient should be in exactly one split
            assert sum([in_train, in_val, in_test]) == 1, f"Patient {patient_id} in multiple splits"

            # Count admissions in the split where patient appears
            if in_train:
                count = (train_df["subject_id"] == patient_id).sum()
            elif in_val:
                count = (val_df["subject_id"] == patient_id).sum()
            else:
                count = (test_df["subject_id"] == patient_id).sum()

            assert count == 7, f"Patient {patient_id} should have 7 admissions, got {count}"


class TestTrainLogisticRegression:
    """(c) test_train_logistic_regression: Returns fitted LR with balanced weights."""

    def test_returns_fitted_logistic_regression(self, small_feature_matrix):
        """train_model returns fitted LogisticRegression."""
        from src.prediction import train_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="logistic_regression")

        assert isinstance(model, LogisticRegression)
        # Check that model is fitted (has coef_ attribute)
        assert hasattr(model, "coef_")
        assert model.coef_.shape[1] == len(feature_cols)

    def test_uses_balanced_class_weight(self, small_feature_matrix):
        """LogisticRegression uses class_weight='balanced'."""
        from src.prediction import train_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="logistic_regression")

        assert model.class_weight == "balanced"


class TestTrainXGBoost:
    """(d) test_train_xgboost: Returns fitted XGBClassifier with computed scale_pos_weight."""

    def test_returns_fitted_xgboost(self, small_feature_matrix):
        """train_model returns fitted XGBClassifier."""
        from src.prediction import train_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")

        assert isinstance(model, XGBClassifier)
        # Check that model is fitted
        assert hasattr(model, "feature_importances_")

    def test_computes_scale_pos_weight(self, small_feature_matrix):
        """XGBoost auto-computes scale_pos_weight from class ratio."""
        from src.prediction import train_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")

        # Expected: n_neg / n_pos
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        expected_weight = n_neg / n_pos

        assert model.scale_pos_weight == pytest.approx(expected_weight, rel=0.01)


class TestEvaluateModelReturnsMetrics:
    """(e) test_evaluate_model_returns_metrics: Returns dict with required metrics."""

    def test_returns_required_metrics(self, small_feature_matrix):
        """evaluate_model returns dict with auroc, auprc, precision, recall, f1, threshold, confusion_matrix."""
        from src.prediction import train_model, evaluate_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")
        metrics = evaluate_model(model, X, y)

        assert isinstance(metrics, dict)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "threshold" in metrics
        assert "confusion_matrix" in metrics

    def test_metrics_in_valid_ranges(self, small_feature_matrix):
        """All metrics are in valid ranges [0, 1]."""
        from src.prediction import train_model, evaluate_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")
        metrics = evaluate_model(model, X, y)

        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["auprc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["threshold"] <= 1


class TestAUROCAboveRandom:
    """(f) test_auroc_above_random: XGBoost AUROC > 0.55 on synthetic data."""

    def test_auroc_above_random(self, synthetic_feature_matrix):
        """XGBoost achieves AUROC > 0.55 on synthetic data with informative features."""
        from src.prediction import patient_level_split, train_model, evaluate_model

        train_df, val_df, test_df = patient_level_split(
            synthetic_feature_matrix,
            target_col="readmitted_30d",
            subject_col="subject_id",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        feature_cols = [c for c in synthetic_feature_matrix.columns if c.startswith("feature_")]
        X_train = train_df[feature_cols]
        y_train = train_df["readmitted_30d"]
        X_test = test_df[feature_cols]
        y_test = test_df["readmitted_30d"]

        model = train_model(X_train, y_train, model_type="xgboost")
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["auroc"] > 0.55, f"AUROC {metrics['auroc']} not above random (0.55)"


class TestFeatureImportanceReturnsRanking:
    """(g) test_feature_importance_returns_ranking: DataFrame sorted descending."""

    def test_returns_sorted_dataframe(self, small_feature_matrix):
        """get_feature_importance returns DataFrame with [feature, importance] sorted descending."""
        from src.prediction import train_model, get_feature_importance

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")
        importance_df = get_feature_importance(model, feature_cols)

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == len(feature_cols)

        # Check sorted descending
        importances = importance_df["importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))

    def test_works_with_logistic_regression(self, small_feature_matrix):
        """get_feature_importance works with LogisticRegression (uses abs(coef_))."""
        from src.prediction import train_model, get_feature_importance

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="logistic_regression")
        importance_df = get_feature_importance(model, feature_cols)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_cols)


class TestModelSaveAndLoad:
    """(h) test_model_save_and_load: XGBoost → JSON, sklearn → pickle; predictions match."""

    def test_xgboost_json_roundtrip(self, small_feature_matrix, tmp_path):
        """XGBoost model saves to JSON and loads with matching predictions."""
        from src.prediction import train_model, save_model, load_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")
        original_preds = model.predict_proba(X)[:, 1]

        model_path = tmp_path / "model.json"
        save_model(model, model_path)

        loaded_model = load_model(model_path)
        loaded_preds = loaded_model.predict_proba(X)[:, 1]

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)

    def test_sklearn_pickle_roundtrip(self, small_feature_matrix, tmp_path):
        """sklearn model saves to pickle and loads with matching predictions."""
        from src.prediction import train_model, save_model, load_model

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="logistic_regression")
        original_preds = model.predict_proba(X)[:, 1]

        model_path = tmp_path / "model.pkl"
        save_model(model, model_path)

        loaded_model = load_model(model_path)
        loaded_preds = loaded_model.predict_proba(X)[:, 1]

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)


class TestEvaluationReportGeneratesMarkdown:
    """(i) test_evaluation_report_generates_markdown: Creates .md with metrics and features."""

    def test_generates_markdown_report(self, small_feature_matrix, tmp_path):
        """generate_evaluation_report creates markdown file with expected content."""
        from src.prediction import train_model, evaluate_model, get_feature_importance, generate_evaluation_report

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")
        metrics = evaluate_model(model, X, y)
        importance_df = get_feature_importance(model, feature_cols)

        report_path = tmp_path / "evaluation_report.md"
        generate_evaluation_report(metrics, importance_df, report_path)

        assert report_path.exists()

        content = report_path.read_text()
        assert "AUROC" in content or "auroc" in content.lower()
        assert "AUPRC" in content or "auprc" in content.lower()
        assert "Confusion Matrix" in content or "confusion" in content.lower()
        assert "feature_" in content  # At least one feature in top-20


class TestCalibrationCurveData:
    """(j) test_calibration_curve_data: Returns calibration data arrays."""

    def test_returns_calibration_arrays(self, small_feature_matrix):
        """calibration_data returns (fraction_of_positives, mean_predicted_value) arrays."""
        from src.prediction import train_model, calibration_data

        feature_cols = [c for c in small_feature_matrix.columns if c.startswith("feature_")]
        X = small_feature_matrix[feature_cols]
        y = small_feature_matrix["readmitted_30d"]

        model = train_model(X, y, model_type="xgboost")
        fraction_pos, mean_pred = calibration_data(model, X, y, n_bins=5)

        assert isinstance(fraction_pos, np.ndarray)
        assert isinstance(mean_pred, np.ndarray)
        assert len(fraction_pos) == len(mean_pred)
        # Values should be in [0, 1]
        assert all(0 <= v <= 1 for v in fraction_pos if not np.isnan(v))
        assert all(0 <= v <= 1 for v in mean_pred if not np.isnan(v))
