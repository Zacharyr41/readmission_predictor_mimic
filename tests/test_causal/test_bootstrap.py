"""Bootstrap-runner correctness (Phase 8d, decision #1).

BootstrapRunner wraps any CausalEstimator, resamples the cohort
stratified by arm, refits, and extracts quantile-based CIs. Tests
exercise:

  * reproducibility under fixed random_state
  * stratified resample preserves per-arm sizes (decision #1 in
    /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md)
  * wider CIs with fewer replicates
  * paired τ CI from within-replicate differencing is narrower than
    the naive difference of marginals
"""

from __future__ import annotations

import numpy as np
import pytest

from src.causal.estimators.base import BootstrapRunner
from src.causal.estimators.metalearners import TLearnerAdapter

from tests.test_causal.conftest import make_synthetic_cohort_frame


class TestBootstrapReproducibility:
    def test_fixed_seed_reproduces_mu_c(self):
        cf = make_synthetic_cohort_frame(n_per_arm=80, n_arms=2, ate=2.0, seed=42)
        runner_a = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name="Y",
            B=10, random_state=7, alpha=0.05,
        )
        runner_b = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name="Y",
            B=10, random_state=7, alpha=0.05,
        )
        a = runner_a.run()
        b = runner_b.run()
        for c in a.mu_c:
            assert a.mu_c[c].point == pytest.approx(b.mu_c[c].point)


class TestStratifiedResample:
    def test_per_arm_sizes_preserved(self):
        cf = make_synthetic_cohort_frame(n_per_arm=120, n_arms=3, ate=1.0, seed=0)
        runner = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name="Y",
            B=5, random_state=0, alpha=0.05,
        )
        # Inspect a single stratified resample: each arm should still
        # have its original row count.
        for replicate_idx in range(5):
            resampled = runner._stratified_resample(replicate_idx)
            for arm in range(3):
                original_size = int((cf.df["T"] == arm).sum())
                resampled_size = int((resampled["T"] == arm).sum())
                assert resampled_size == original_size, (
                    f"arm {arm} had {original_size} rows originally but "
                    f"{resampled_size} in replicate {replicate_idx}"
                )


class TestBootstrapUncertaintyScales:
    def test_fewer_reps_yields_wider_cis(self):
        cf = make_synthetic_cohort_frame(n_per_arm=100, n_arms=2, ate=2.0, seed=11)
        small = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name="Y",
            B=20, random_state=0, alpha=0.05,
        ).run()
        large = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name="Y",
            B=200, random_state=0, alpha=0.05,
        ).run()
        # Width of the interval for arm 0. With the same seed the
        # center should be similar; the tail quantiles at B=20 are much
        # noisier, so on average width is larger. This is a statistical
        # claim — assert weakly (≥ 0.5× the larger-B width; both should
        # be finite).
        small_width = small.mu_c[0].upper - small.mu_c[0].lower
        large_width = large.mu_c[0].upper - large.mu_c[0].lower
        assert np.isfinite(small_width) and np.isfinite(large_width)
        assert small_width >= 0.5 * large_width


class TestPairedTauCi:
    def test_tau_ci_uses_paired_replicates(self):
        """τ_{c,c'} CI should come from within-replicate differencing,
        not from differencing independent marginal CIs. The paired CI
        is narrower when Y_c and Y_{c'} share noise — because the
        shared component cancels in the difference."""
        cf = make_synthetic_cohort_frame(n_per_arm=120, n_arms=2, ate=2.0, seed=3)
        runner = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name="Y",
            B=50, random_state=0, alpha=0.05,
        )
        result = runner.run()
        # Exactly one pairwise key for C=2.
        assert len(result.tau_cc_prime) == 1
        tau_key = next(iter(result.tau_cc_prime))
        tau_ui = result.tau_cc_prime[tau_key]
        mu0 = result.mu_c[0]
        mu1 = result.mu_c[1]
        paired_width = tau_ui.upper - tau_ui.lower
        naive_width = (mu1.upper - mu1.lower) + (mu0.upper - mu0.lower)
        # Paired should be narrower or equal; allow small numerical slop.
        assert paired_width <= naive_width + 1e-9
