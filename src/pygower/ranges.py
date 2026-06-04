"""Fit state: per-column range/category estimation and numeric encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._types import Kind
from .spec import ColumnSpec


def _safe_range(lo: float, hi: float) -> float:
    """Range guard: a constant / degenerate column gets range 1.0 so the
    division never blows up (all pairs then score partial distance 0)."""
    r = float(hi - lo)
    if not np.isfinite(r) or r == 0.0:
        return 1.0
    return r


class FittedColumn:
    """Encodes one column to a numeric array and carries fit-time state."""

    def __init__(self, name: str, spec: ColumnSpec) -> None:
        self.name = name
        self.spec = spec
        self.range_: float | None = None
        self.categories_: dict | None = None  # value -> rank (ordinal) / code (nominal)
        self._n_fit_cats: int = 0  # nominal: count of categories seen at fit

    def fit(self, s: pd.Series) -> "FittedColumn":
        k = self.spec.kind
        if k == Kind.QUANTITATIVE:
            if self.spec.range_ is not None:
                lo, hi = self.spec.range_
            else:
                arr = s.to_numpy(dtype=float)
                if np.all(np.isnan(arr)):
                    lo, hi = 0.0, 0.0
                else:
                    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            self.range_ = _safe_range(lo, hi)
        elif k == Kind.ORDINAL:
            cats = self.spec.categories or sorted(s.dropna().unique())
            self.categories_ = {c: i for i, c in enumerate(cats)}
            self.range_ = _safe_range(0.0, float(len(cats) - 1))
        elif k == Kind.NOMINAL:
            # Freeze a value->code map so X and Y (e.g. profile vs candidates)
            # encode to the SAME codes. Independent per-frame factorize would
            # scramble codes and make distinct categories collide.
            cats = sorted(s.dropna().unique())
            self.categories_ = {c: i for i, c in enumerate(cats)}
            self._n_fit_cats = len(cats)
        return self

    def encode(self, s: pd.Series) -> np.ndarray:
        """Return a float array where NaN marks missing; categoricals -> codes."""
        k = self.spec.kind
        if k == Kind.QUANTITATIVE:
            return s.to_numpy(dtype=float)
        if k == Kind.ORDINAL:
            return s.map(self.categories_).to_numpy(dtype=float)
        if k == Kind.BINARY:
            present = (s == self.spec.present_value)
            arr = present.to_numpy(dtype=float)
            arr[s.isna().to_numpy()] = np.nan
            return arr
        # nominal: map through the frozen fit codes; truly-missing -> NaN.
        # Categories unseen at fit get distinct codes past the fit range so they
        # never collide with a known category (always a mismatch), while real
        # NaN stays NaN (excluded by the validity mask).
        out = s.map(self.categories_).to_numpy(dtype=float)
        missing = s.isna().to_numpy()
        unseen = np.isnan(out) & ~missing
        if unseen.any():
            fresh, _ = pd.factorize(s[unseen])
            out[unseen] = self._n_fit_cats + fresh
        return out
