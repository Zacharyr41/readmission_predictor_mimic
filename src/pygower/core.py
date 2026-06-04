"""Aggregation engine: weighted Gower distance over encoded columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import Direction, Missing
from .kernels import select_kernel
from .ranges import FittedColumn


@dataclass
class ColumnContribution:
    """Per-column similarity exposed for explanation.

    ``signed()`` recovers the signed contribution ``w * (2 s - 1)`` (masked to
    0 where the pair was excluded), so a perfect match contributes ``+w`` and a
    perfect mismatch ``-w``.
    """

    name: str
    weight: float
    direction: Direction
    similarity: np.ndarray  # (n, m)
    delta: np.ndarray       # (n, m), float in {0, 1}

    def signed(self) -> np.ndarray:
        return np.where(self.delta > 0, self.weight * (2.0 * self.similarity - 1.0), 0.0)


def _apply_missing_policy(s, delta, a, b, policy):
    """Override the validity mask per the column's NaN policy.

    EXCLUDE leaves Gower's default (NaN -> delta 0). ZERO_SIMILARITY re-includes
    NaN pairs with similarity 0 so a missing value acts as a detractor.
    """
    if policy == Missing.ZERO_SIMILARITY:
        nan_mask = np.isnan(a)[:, None] | np.isnan(b)[None, :]
        s = np.where(nan_mask, 0.0, s)
        delta = delta | nan_mask
    return s, delta


def _accumulate(
    cols_X: list[np.ndarray],
    cols_Y: list[np.ndarray],
    fitted: list[FittedColumn],
    col_weights: list[float],
    *,
    collect_contributions: bool = False,
):
    """Return ``(D, contributions)`` for encoded column lists.

    ``D`` is ``(n, m)``; a pair with no shared valid variable is NaN.
    ``contributions`` is ``None`` unless ``collect_contributions``.
    """
    n = cols_X[0].shape[0]
    m = cols_Y[0].shape[0]
    num = np.zeros((n, m))
    den = np.zeros((n, m))
    contributions: list[ColumnContribution] | None = (
        [] if collect_contributions else None
    )

    for c, fc in enumerate(fitted):
        kernel = select_kernel(fc.spec)
        a, b = cols_X[c], cols_Y[c]
        s, delta = kernel(
            a, b,
            range_=fc.range_,
            asymmetric=fc.spec.asymmetric,
            direction=fc.spec.direction,
        )
        s, delta = _apply_missing_policy(s, delta, a, b, fc.spec.missing)
        w = col_weights[c]
        d = delta.astype(float)
        # np.where (not 0 * (1 - s)) so an excluded pair contributes exactly 0:
        # for NaN inputs s is NaN and 0 * NaN would poison the numerator.
        num += w * np.where(delta, 1.0 - s, 0.0)
        den += w * d
        if contributions is not None:
            contributions.append(
                ColumnContribution(
                    name=fc.name,
                    weight=w,
                    direction=fc.spec.direction,
                    similarity=s,
                    delta=d,
                )
            )

    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(den > 0, num / den, np.nan)
    return D, contributions


def gower_full(
    cols_X,
    cols_Y,
    fitted,
    col_weights,
    *,
    chunk_size: int | None = None,
    collect_contributions: bool = False,
):
    """Dense or row-chunked aggregation.

    Chunking bounds peak memory to ``O(chunk * m)``. Per-column contributions
    are only assembled in the dense path (the cohort use case scores a single
    profile row against many candidates, so chunking is unnecessary there).
    """
    if collect_contributions or chunk_size is None:
        return _accumulate(
            cols_X, cols_Y, fitted, col_weights,
            collect_contributions=collect_contributions,
        )
    n = cols_X[0].shape[0]
    D = np.empty((n, cols_Y[0].shape[0]))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sub_X = [c[start:stop] for c in cols_X]
        D[start:stop], _ = _accumulate(sub_X, cols_Y, fitted, col_weights)
    return D, None
