"""scikit-learn compatible transformer wrapping the Gower engine."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

from .api import _encode, _fit_columns, _resolve_weights
from .core import gower_full
from .validation import check_same_columns, to_frame


class GowerDistance(BaseEstimator, TransformerMixin):
    """Learn ranges/categories on ``fit``; ``transform`` = distances vs fit data.

    The fit/transform split is what makes train/test correct: ranges come only
    from the reference (fit) data, never recomputed on the query.
    """

    def __init__(self, spec=None, weights=None, chunk_size=None):
        self.spec = spec
        self.weights = weights
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        self._X = to_frame(X)
        self.fitted_, self.spec_ = _fit_columns(self._X, self.spec)
        self.col_weights_ = _resolve_weights(self.fitted_, self.weights)
        self._cols_ref = _encode(self._X, self.fitted_)
        return self

    def transform(self, X):
        Xf = to_frame(X)
        check_same_columns(self._X, Xf)
        cols_X = _encode(Xf, self.fitted_)
        D, _ = gower_full(
            cols_X, self._cols_ref, self.fitted_, self.col_weights_,
            chunk_size=self.chunk_size,
        )
        return D
