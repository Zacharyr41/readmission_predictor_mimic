"""Input validation and coercion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    raise TypeError(f"Unsupported input type: {type(X)!r}")


def check_same_columns(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    if list(X.columns) != list(Y.columns):
        raise ValueError("X and Y must have identical columns in the same order")
