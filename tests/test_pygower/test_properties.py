"""Property-based tests for ``pygower`` (Hypothesis).

These assert the mathematical invariants of Gower's distance independent of
any particular dataset: zero self-distance, symmetry (for symmetric kinds),
bounded range [0, 1], and permutation consistency.

Note: constant-column *invariance* is deliberately NOT asserted -- it is false
for averaged Gower. A constant column is a perfect match (s=1) that contributes
0 to the numerator but its weight to the denominator, so it dilutes every
distance toward 0 (the documented include-and-dilute semantics, matching daisy
and the ``gower`` PyPI package).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.pygower import ColumnSpec, gower_matrix

_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def mixed_frames(draw):
    """A small mixed-type frame with a quantitative, nominal and binary col."""
    n = draw(st.integers(min_value=2, max_value=8))
    nums = draw(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    cats = draw(
        st.lists(st.sampled_from(["a", "b", "c"]), min_size=n, max_size=n)
    )
    flags = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return pd.DataFrame({"num": nums, "cat": cats, "flag": flags})


_SPEC = {
    "num": ColumnSpec(kind="quantitative"),
    "cat": ColumnSpec(kind="nominal"),
    "flag": ColumnSpec(kind="binary"),  # symmetric -> matrix stays symmetric
}


@_SETTINGS
@given(mixed_frames())
def test_zero_diagonal(df):
    D = gower_matrix(df, spec=_SPEC)
    assert np.all(np.diag(D) == 0.0)


@_SETTINGS
@given(mixed_frames())
def test_symmetric(df):
    D = gower_matrix(df, spec=_SPEC)
    assert np.allclose(D, D.T, equal_nan=True)


@_SETTINGS
@given(mixed_frames())
def test_bounded_unit_interval(df):
    D = gower_matrix(df, spec=_SPEC)
    finite = D[np.isfinite(D)]
    assert np.all(finite >= -1e-12)
    assert np.all(finite <= 1.0 + 1e-12)


@st.composite
def frames_with_perm(draw):
    df = draw(mixed_frames())
    perm = draw(st.permutations(list(range(len(df)))))
    return df, np.asarray(perm)


@_SETTINGS
@given(frames_with_perm())
def test_permutation_consistency(args):
    """Relabeling rows permutes the distance matrix the same way (row/col)."""
    df, perm = args
    D = gower_matrix(df, spec=_SPEC)
    D_perm = gower_matrix(df.iloc[perm].reset_index(drop=True), spec=_SPEC)
    assert np.allclose(D_perm, D[np.ix_(perm, perm)], equal_nan=True)


@_SETTINGS
@given(mixed_frames())
def test_constant_column_dilutes(df):
    """A constant (perfect-match) column shrinks distances toward 0, never grows them.

    This pins the include-and-dilute semantics: averaged Gower folds the
    constant column's weight into the denominator while it adds 0 to the
    numerator, so every distance is scaled down by a positive factor.
    """
    base = gower_matrix(df, spec=_SPEC)
    df2 = df.copy()
    df2["constant"] = 7.0
    spec2 = dict(_SPEC)
    spec2["constant"] = ColumnSpec(kind="quantitative")
    diluted = gower_matrix(df2, spec=spec2)
    # 4 columns each weight 1 -> the constant adds 1 to a denominator that was 3.
    assert np.allclose(diluted, base * (3.0 / 4.0), equal_nan=True)
