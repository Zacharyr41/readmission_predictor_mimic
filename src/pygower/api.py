"""Top-level functions: gower_matrix / gower_distances / gower_topn."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .core import gower_full
from .ranges import FittedColumn
from .spec import ColumnSpec, infer_spec
from .validation import check_same_columns, to_frame


def _fit_columns(df: pd.DataFrame, spec):
    spec = spec or infer_spec(df)
    missing = [c for c in df.columns if c not in spec]
    if missing:
        raise ValueError(f"No ColumnSpec for columns: {missing}")
    fitted = [FittedColumn(col, spec[col]).fit(df[col]) for col in df.columns]
    return fitted, spec


def _encode(df: pd.DataFrame, fitted):
    return [fc.encode(df[fc.name]) for fc in fitted]


def _resolve_weights(fitted, weights) -> list[float]:
    """Per-column weights: spec weight by default, overridden by ``weights``."""
    if weights is None:
        return [float(fc.spec.weight) for fc in fitted]
    if isinstance(weights, Mapping):
        return [float(weights.get(fc.name, fc.spec.weight)) for fc in fitted]
    if isinstance(weights, Sequence):
        if len(weights) != len(fitted):
            raise ValueError(
                f"weights length {len(weights)} != number of columns {len(fitted)}"
            )
        return [float(w) for w in weights]
    raise TypeError(f"Unsupported weights type: {type(weights)!r}")


def gower_distances(
    X,
    Y=None,
    *,
    spec: Mapping[str, ColumnSpec] | None = None,
    weights=None,
    chunk_size: int | None = None,
    return_contributions: bool = False,
):
    """Gower distances of ``X`` rows vs ``Y`` rows (``Y=None`` -> ``X`` vs ``X``).

    Ranges/categories are learned from ``X`` (or taken from frozen ``range_``
    in the spec) and reused for ``Y`` so distances are comparable across calls.
    With ``return_contributions`` returns ``(D, [ColumnContribution, ...])``.
    """
    X = to_frame(X)
    fitted, spec = _fit_columns(X, spec)
    col_weights = _resolve_weights(fitted, weights)
    if Y is None:
        Y = X
    else:
        Y = to_frame(Y)
        check_same_columns(X, Y)
    cols_X = _encode(X, fitted)
    cols_Y = _encode(Y, fitted)
    D, contribs = gower_full(
        cols_X, cols_Y, fitted, col_weights,
        chunk_size=chunk_size,
        collect_contributions=return_contributions,
    )
    if return_contributions:
        return D, contribs
    return D


def gower_matrix(
    X,
    *,
    spec=None,
    weights=None,
    return_similarity: bool = False,
    chunk_size: int | None = None,
    dtype: np.dtype = np.float64,
):
    """Symmetric pairwise distance matrix with an exact-zero diagonal."""
    D = gower_distances(X, None, spec=spec, weights=weights, chunk_size=chunk_size)
    np.fill_diagonal(D, 0.0)
    D = D.astype(dtype)
    if return_similarity:
        return 1.0 - D
    return D


def gower_topn(X, Y=None, *, n: int = 5, spec=None, weights=None):
    """Indices and distances of the ``n`` nearest ``Y`` rows for each ``X`` row."""
    D = gower_distances(X, Y, spec=spec, weights=weights)
    # NaN sorts last so all-missing pairs never masquerade as neighbours.
    order = np.argsort(np.where(np.isnan(D), np.inf, D), axis=1)[:, :n]
    dist = np.take_along_axis(D, order, axis=1)
    return order, dist
