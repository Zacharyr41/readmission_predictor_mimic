"""Deterministic tests for the ``pygower`` Gower's-distance module.

Expected values are hand-derived from the Gower formula

    d_ij = sum_k w_k * delta_ijk * (1 - s_ijk) / sum_k w_k * delta_ijk

so the goldens characterize the *spec*, not the implementation (no value is
copied from a running implementation). Partial similarities:

  * quantitative: s = clip(1 - |xi - xj| / R, 0, 1)
  * nominal:      s = 1[xi == xj]
  * binary sym:   s = 1[xi == xj]                    (both-absent counts, s=1)
  * binary asym:  s = 1[xi == xj == present]; both-absent excluded (delta=0)
  * one-sided (higher_more_similar), reference r, candidate x:
        s = clip(1 - max(0, r - x) / R, 0, 1)   (x >= r -> s = 1, not penalized)
  * one-sided (lower_more_similar): mirror.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pygower import (
    ColumnContribution,
    ColumnSpec,
    GowerDistance,
    gower_distances,
    gower_matrix,
    gower_topn,
)
from src.pygower._types import Direction, Kind, Missing
from src.pygower.kernels import (
    KERNELS,
    binary_kernel,
    nominal_kernel,
    one_sided_kernel,
    quantitative_kernel,
    select_kernel,
)


# --------------------------------------------------------------------------
# ColumnSpec & inference
# --------------------------------------------------------------------------


class TestColumnSpec:
    def test_kind_string_coerces_to_enum(self):
        spec = ColumnSpec(kind="quantitative")
        assert spec.kind is Kind.QUANTITATIVE

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ColumnSpec(kind="quantitative", weight=-1.0)

    def test_direction_defaults_symmetric(self):
        assert ColumnSpec(kind="quantitative").direction is Direction.SYMMETRIC

    def test_direction_string_coerces(self):
        spec = ColumnSpec(kind="quantitative", direction="higher_more_similar")
        assert spec.direction is Direction.HIGHER_MORE_SIMILAR

    def test_missing_defaults_exclude(self):
        assert ColumnSpec(kind="quantitative").missing is Missing.EXCLUDE

    def test_missing_string_coerces(self):
        spec = ColumnSpec(kind="quantitative", missing="zero_similarity")
        assert spec.missing is Missing.ZERO_SIMILARITY


class TestInferSpec:
    def test_infers_each_type(self):
        from src.pygower.spec import infer_spec

        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "flag": [True, False, True],
                "cat": ["a", "b", "a"],
                "ord": pd.Categorical(
                    ["lo", "hi", "lo"], categories=["lo", "hi"], ordered=True
                ),
            }
        )
        spec = infer_spec(df)
        assert spec["num"].kind is Kind.QUANTITATIVE
        assert spec["flag"].kind is Kind.BINARY
        assert spec["cat"].kind is Kind.NOMINAL
        assert spec["ord"].kind is Kind.ORDINAL
        assert list(spec["ord"].categories) == ["lo", "hi"]


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------


class TestKernels:
    def test_registry_select_by_kind(self):
        assert select_kernel(ColumnSpec(kind="nominal")) is nominal_kernel
        assert KERNELS[Kind.QUANTITATIVE] is quantitative_kernel

    def test_quantitative_basic(self):
        a = np.array([0.0, 10.0])
        b = np.array([5.0])
        s, delta = quantitative_kernel(a, b, range_=10.0)
        # |0-5|/10 = 0.5 -> s=0.5 ; |10-5|/10=0.5 -> s=0.5
        assert s.shape == (2, 1)
        assert s[0, 0] == pytest.approx(0.5)
        assert s[1, 0] == pytest.approx(0.5)
        assert delta.all()

    def test_quantitative_clips_to_zero_when_diff_exceeds_range(self):
        # |0 - 30| / 10 = 3.0 -> 1 - 3 = -2 -> clipped to 0 (guarantees d<=1)
        s, _ = quantitative_kernel(np.array([0.0]), np.array([30.0]), range_=10.0)
        assert s[0, 0] == pytest.approx(0.0)

    def test_nominal_match(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        s, delta = nominal_kernel(a, b)
        assert s[0, 0] == 1.0  # 1 == 1
        assert s[0, 1] == 0.0  # 1 != 3
        assert s[1, 0] == 0.0
        assert delta.all()

    def test_binary_symmetric_counts_both_absent(self):
        # symmetric: both-absent is a match (s=1) AND counted (delta=1)
        a = np.array([0.0])
        b = np.array([0.0])
        s, delta = binary_kernel(a, b, asymmetric=False)
        assert s[0, 0] == 1.0
        assert bool(delta[0, 0]) is True

    def test_binary_asymmetric_excludes_both_absent(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        s, delta = binary_kernel(a, b, asymmetric=True)
        # both present -> match, counted
        assert s[0, 0] == 1.0 and bool(delta[0, 0]) is True
        # both absent -> excluded
        assert bool(delta[1, 1]) is False
        # present vs absent -> mismatch, counted
        assert s[0, 1] == 0.0 and bool(delta[0, 1]) is True

    def test_one_sided_higher_does_not_penalize_exceeding(self):
        # reference r=2 (X side), candidates [1,2,3] (Y side), R=2
        r = np.array([2.0])
        x = np.array([1.0, 2.0, 3.0])
        s, delta = one_sided_kernel(r, x, range_=2.0, direction=Direction.HIGHER_MORE_SIMILAR)
        assert s[0, 0] == pytest.approx(0.5)  # 1 - (2-1)/2
        assert s[0, 1] == pytest.approx(1.0)  # x == r
        assert s[0, 2] == pytest.approx(1.0)  # x > r, NOT penalized
        assert delta.all()

    def test_one_sided_lower_mirror(self):
        r = np.array([2.0])
        x = np.array([1.0, 2.0, 3.0])
        s, _ = one_sided_kernel(r, x, range_=2.0, direction=Direction.LOWER_MORE_SIMILAR)
        assert s[0, 0] == pytest.approx(1.0)  # below ref, not penalized
        assert s[0, 1] == pytest.approx(1.0)
        assert s[0, 2] == pytest.approx(0.5)  # 1 - (3-2)/2

    def test_select_kernel_routes_directional(self):
        spec = ColumnSpec(kind="quantitative", direction="higher_more_similar")
        assert select_kernel(spec) is one_sided_kernel


# --------------------------------------------------------------------------
# gower_matrix — golden hand-computed mixed dataset
# --------------------------------------------------------------------------


def _golden_df():
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],   # quantitative, range = 20
            "city": ["A", "A", "B"],      # nominal
            "member": [True, False, True],  # binary asymmetric, present=True
        }
    )


def _golden_spec():
    return {
        "age": ColumnSpec(kind="quantitative"),
        "city": ColumnSpec(kind="nominal"),
        "member": ColumnSpec(kind="binary", asymmetric=True, present_value=True),
    }


class TestGowerMatrix:
    def test_golden_mixed(self):
        D = gower_matrix(_golden_df(), spec=_golden_spec())
        # Hand-derived (see module docstring), range_age = 20:
        #  d01: age (1-0.5)=0.5; city A=A ->0; member (1,0) asym counted ->1
        #       num=0.5+0+1=1.5 den=3 -> 0.5
        #  d02: age (1-0)=1; city A!=B ->1; member (1,1) match ->0
        #       num=1+1+0=2 den=3 -> 0.6667
        #  d12: age 0.5; city 1; member (0,1) counted ->1
        #       num=0.5+1+1=2.5 den=3 -> 0.8333
        assert D[0, 1] == pytest.approx(0.5)
        assert D[0, 2] == pytest.approx(2.0 / 3.0)
        assert D[1, 2] == pytest.approx(2.5 / 3.0)

    def test_diagonal_exactly_zero(self):
        D = gower_matrix(_golden_df(), spec=_golden_spec())
        assert np.all(np.diag(D) == 0.0)

    def test_symmetry(self):
        D = gower_matrix(_golden_df(), spec=_golden_spec())
        assert np.allclose(D, D.T, equal_nan=True)

    def test_weights_override(self):
        # age weight 2 -> d02: num = 2*1(age) + 1*1(city) + 1*0(member) = 3 ; den=4 -> 0.75
        D = gower_matrix(_golden_df(), spec=_golden_spec(), weights={"age": 2.0})
        assert D[0, 2] == pytest.approx(0.75)

    def test_zero_range_constant_column(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        D = gower_matrix(df, spec={"x": ColumnSpec(kind="quantitative")})
        # constant column -> range guarded to 1.0, all diffs 0 -> all distances 0
        assert np.allclose(D, 0.0)

    def test_all_missing_pair_is_nan(self):
        df = pd.DataFrame({"x": [np.nan, np.nan]})
        D = gower_matrix(df, spec={"x": ColumnSpec(kind="quantitative")})
        # off-diagonal pair shares no valid variable -> NaN (diagonal forced 0)
        assert np.isnan(D[0, 1])
        assert D[0, 0] == 0.0


# --------------------------------------------------------------------------
# Missing-data policy
# --------------------------------------------------------------------------


class TestMissingPolicy:
    def test_exclude_drops_column_from_average(self):
        # age missing on row 0; city matches -> with exclude, only city counts -> d=0
        df_x = pd.DataFrame({"age": [np.nan], "city": ["A"]})
        df_y = pd.DataFrame({"age": [5.0], "city": ["A"]})
        spec = {
            "age": ColumnSpec(kind="quantitative", range_=(0.0, 10.0), missing="exclude"),
            "city": ColumnSpec(kind="nominal"),
        }
        D = gower_distances(df_x, df_y, spec=spec)
        assert D[0, 0] == pytest.approx(0.0)

    def test_zero_similarity_counts_missing_as_detractor(self):
        # Same data but missing -> similarity 0 AND counted:
        # age: s=0 delta=1 (num=1) ; city: s=1 delta=1 (num=0) ; den=2 -> d=0.5
        df_x = pd.DataFrame({"age": [np.nan], "city": ["A"]})
        df_y = pd.DataFrame({"age": [5.0], "city": ["A"]})
        spec = {
            "age": ColumnSpec(
                kind="quantitative", range_=(0.0, 10.0), missing="zero_similarity"
            ),
            "city": ColumnSpec(kind="nominal"),
        }
        D = gower_distances(df_x, df_y, spec=spec)
        assert D[0, 0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# X vs Y, directional profile, frozen ranges, contributions
# --------------------------------------------------------------------------


class TestProfileVsCandidates:
    def test_directional_profile_end_to_end(self):
        # Mini "worsening lactate" scenario: profile lactate slope = +1.0,
        # range frozen at 2.0, direction higher_more_similar.
        profile = pd.DataFrame({"lactate_slope": [1.0]})
        candidates = pd.DataFrame({"lactate_slope": [0.0, 1.0, 5.0]})
        spec = {
            "lactate_slope": ColumnSpec(
                kind="quantitative",
                range_=(0.0, 2.0),
                direction="higher_more_similar",
            )
        }
        D = gower_distances(profile, candidates, spec=spec)
        # cand 0 (slope 0 < 1): s = 1-(1-0)/2 = 0.5 -> d=0.5
        # cand 1 (slope 1 == 1): d=0
        # cand 2 (slope 5 > 1): NOT penalized -> d=0
        assert D[0, 0] == pytest.approx(0.5)
        assert D[0, 1] == pytest.approx(0.0)
        assert D[0, 2] == pytest.approx(0.0)

    def test_nominal_codes_consistent_across_x_and_y(self):
        # Profile (X / fit side) and candidates (Y) list categories in DIFFERENT
        # orders. Factorizing each frame independently would scramble the integer
        # codes so "ED" could collide with "ICU"; the fit-time category map must
        # keep "ED" == "ED" and "ED" != {ICU, OR}. This is the load-bearing
        # cohort path (one profile row vs many candidates), which gower_matrix
        # (Y = X) never exercises.
        profile = pd.DataFrame({"unit": ["ED"]})
        candidates = pd.DataFrame({"unit": ["ICU", "ED", "OR"]})
        spec = {"unit": ColumnSpec(kind="nominal")}
        D = gower_distances(profile, candidates, spec=spec)
        assert D[0, 0] == pytest.approx(1.0)  # ED vs ICU -> mismatch
        assert D[0, 1] == pytest.approx(0.0)  # ED vs ED  -> match
        assert D[0, 2] == pytest.approx(1.0)  # ED vs OR  -> mismatch

    def test_frozen_range_used_not_recomputed_from_profile(self):
        # X is a single-row profile; a naive fit would derive range 0 from it.
        # The frozen range_ in the spec must be used instead.
        profile = pd.DataFrame({"age": [50.0]})
        candidates = pd.DataFrame({"age": [50.0, 70.0]})
        spec = {"age": ColumnSpec(kind="quantitative", range_=(0.0, 100.0))}
        D = gower_distances(profile, candidates, spec=spec)
        # |50-70|/100 = 0.2 -> d=0.2 (would be undefined/0 if range came from X)
        assert D[0, 1] == pytest.approx(0.2)

    def test_return_contributions_recovers_signed(self):
        profile = pd.DataFrame({"age": [50.0], "member": [True]})
        candidates = pd.DataFrame({"age": [70.0], "member": [True]})
        spec = {
            "age": ColumnSpec(kind="quantitative", range_=(0.0, 100.0), weight=1.0),
            "member": ColumnSpec(
                kind="binary", asymmetric=True, present_value=True, weight=2.0
            ),
        }
        D, contribs = gower_distances(
            profile, candidates, spec=spec, return_contributions=True
        )
        assert isinstance(contribs, list)
        by_name = {c.name: c for c in contribs}
        assert set(by_name) == {"age", "member"}
        assert all(isinstance(c, ColumnContribution) for c in contribs)
        # age similarity = 1 - 0.2 = 0.8 ; signed = w*(2s-1) = 1*(0.6) = 0.6
        age = by_name["age"]
        assert age.similarity[0, 0] == pytest.approx(0.8)
        assert age.signed()[0, 0] == pytest.approx(0.6)
        # member match s=1 ; signed = 2*(2*1-1) = +2
        member = by_name["member"]
        assert member.signed()[0, 0] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Chunking, top-n
# --------------------------------------------------------------------------


class TestChunkingAndTopN:
    def _big(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "a": rng.normal(size=25),
                "b": rng.integers(0, 4, size=25).astype(float),
                "c": rng.choice(["x", "y", "z"], size=25),
            }
        )

    def test_chunked_equals_dense(self):
        df = self._big()
        dense = gower_distances(df, chunk_size=None)
        chunked = gower_distances(df, chunk_size=4)
        assert np.allclose(dense, chunked, equal_nan=True)

    def test_topn_returns_sorted_neighbours(self):
        df = self._big()
        idx, dist = gower_topn(df, n=3)
        assert idx.shape == (25, 3)
        assert dist.shape == (25, 3)
        # distances along each row are non-decreasing
        assert np.all(np.diff(dist, axis=1) >= -1e-12)
        # nearest neighbour of each row (X vs X) is itself at distance 0
        assert np.allclose(dist[:, 0], 0.0)


# --------------------------------------------------------------------------
# sklearn transformer: fit/transform separation
# --------------------------------------------------------------------------


class TestGowerDistanceTransformer:
    def test_fit_transform_matches_gower_matrix(self):
        df = _golden_df()
        spec = _golden_spec()
        gd = GowerDistance(spec=spec)
        D_t = gd.fit_transform(df)
        D_m = gower_matrix(df, spec=spec)
        assert np.allclose(D_t, D_m, equal_nan=True, atol=1e-12)

    def test_ranges_come_from_train_only(self):
        train = pd.DataFrame({"age": [0.0, 10.0]})          # range 10
        test = pd.DataFrame({"age": [0.0, 100.0]})          # would be 100 if refit
        gd = GowerDistance(spec={"age": ColumnSpec(kind="quantitative")}).fit(train)
        D = gd.transform(test)  # (n_test, n_train)
        # test row age=100 vs train row age=0 -> |100-0|/10 = 10 -> clipped -> d=1
        assert D[1, 0] == pytest.approx(1.0)
        # test row age=0 vs train row age=0 -> d=0
        assert D[0, 0] == pytest.approx(0.0)
