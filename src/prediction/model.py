"""Model training and persistence for hospital readmission prediction.

Supports Logistic Regression and XGBoost classifiers with automatic class
imbalance handling. Provides save/load functionality for model persistence.
"""

import pickle
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_model(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    model_type: Literal["logistic_regression", "xgboost"] = "xgboost",
    **kwargs,
) -> Union[LogisticRegression, XGBClassifier]:
    """Train a classification model for readmission prediction.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        model_type: Type of model to train ("logistic_regression" or "xgboost")
        **kwargs: Additional parameters passed to the model constructor

    Returns:
        Fitted model (LogisticRegression or XGBClassifier)
    """
    if model_type == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=kwargs.pop("random_state", 42),
            **kwargs,
        )
    elif model_type == "xgboost":
        # Compute scale_pos_weight from class ratio
        y_array = np.asarray(y_train)
        n_pos = y_array.sum()
        n_neg = len(y_array) - n_pos
        scale_pos_weight = kwargs.pop("scale_pos_weight", n_neg / n_pos if n_pos > 0 else 1.0)

        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=kwargs.pop("n_estimators", 100),
            max_depth=kwargs.pop("max_depth", 6),
            learning_rate=kwargs.pop("learning_rate", 0.1),
            random_state=kwargs.pop("random_state", 42),
            eval_metric="logloss",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def save_model(model: Union[LogisticRegression, XGBClassifier], path: Path) -> None:
    """Save a trained model to disk.

    XGBoost models are saved to JSON format for portability.
    sklearn models are saved to pickle format.

    Args:
        model: Trained model to save
        path: Path to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, XGBClassifier):
        model.save_model(str(path))
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(path: Path) -> Union[LogisticRegression, XGBClassifier]:
    """Load a trained model from disk.

    Determines model type from file extension: .json for XGBoost, .pkl for sklearn.

    Args:
        path: Path to the saved model

    Returns:
        Loaded model
    """
    path = Path(path)

    if path.suffix == ".json":
        model = XGBClassifier()
        model.load_model(str(path))
        return model
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
