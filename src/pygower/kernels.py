"""Vectorized per-type partial-similarity kernels.

Each kernel takes one column of ``X`` (shape ``(n,)``) and one of ``Y``
(shape ``(m,)``) and returns two ``(n, m)`` arrays: partial **similarity**
``s`` (in [0, 1]) and a boolean validity mask ``delta``. Broadcasting does
the pairwise expansion.

Convention for the directional one-sided kernel: ``a`` is the *reference*
(the X side, e.g. a synthesized profile) and ``b`` is the *candidate*
(the Y side). The kernel is intentionally asymmetric in (a, b).
"""

from __future__ import annotations

import numpy as np

from ._types import Direction, Kind
from .spec import ColumnSpec


def _missing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """delta=False where either side is NaN."""
    return ~(np.isnan(a)[:, None] | np.isnan(b)[None, :])


def quantitative_kernel(a, b, *, range_, **_):
    diff = np.abs(a[:, None] - b[None, :]) / range_
    s = np.clip(1.0 - diff, 0.0, 1.0)
    return s, _missing(a, b)


def nominal_kernel(a, b, **_):
    s = (a[:, None] == b[None, :]).astype(float)
    return s, _missing(a, b)


# Ordinal columns are rank-encoded upstream, so the quantitative rule applies.
ordinal_kernel = quantitative_kernel


def binary_kernel(a, b, *, asymmetric=False, **_):
    eq = a[:, None] == b[None, :]
    s = eq.astype(float)
    delta = _missing(a, b)
    if asymmetric:
        # Exclude negative matches (both absent) from the average.
        both_absent = (a[:, None] == 0) & (b[None, :] == 0)
        delta = delta & ~both_absent
    return s, delta


def one_sided_kernel(a, b, *, range_, direction, **_):
    """Monotonic kernel: ``a`` reference, ``b`` candidate.

    ``HIGHER_MORE_SIMILAR``: a candidate at or above the reference is fully
    similar; only falling short of it is penalized. ``LOWER_MORE_SIMILAR``
    is the mirror.
    """
    ref = a[:, None]
    cand = b[None, :]
    if direction == Direction.HIGHER_MORE_SIMILAR:
        shortfall = np.maximum(0.0, ref - cand)
    elif direction == Direction.LOWER_MORE_SIMILAR:
        shortfall = np.maximum(0.0, cand - ref)
    else:  # pragma: no cover - select_kernel never routes SYMMETRIC here
        shortfall = np.abs(ref - cand)
    s = np.clip(1.0 - shortfall / range_, 0.0, 1.0)
    return s, _missing(a, b)


KERNELS = {
    Kind.QUANTITATIVE: quantitative_kernel,
    Kind.NOMINAL: nominal_kernel,
    Kind.ORDINAL: ordinal_kernel,
    Kind.BINARY: binary_kernel,
}


def select_kernel(spec: ColumnSpec):
    """Pick the kernel for a column: explicit override > directional > type."""
    if spec.kernel is not None:
        return spec.kernel
    if spec.kind == Kind.QUANTITATIVE and spec.direction != Direction.SYMMETRIC:
        return one_sided_kernel
    return KERNELS[spec.kind]
