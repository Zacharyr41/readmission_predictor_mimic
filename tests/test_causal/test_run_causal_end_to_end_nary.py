"""End-to-end Phase 8d: C=3 arms, scalar outcome (I5 shape, slimmed).

Flips the xfail at ``tests/test_conversational/test_intervention_effect.py:635``
once 8d lands.
"""

from __future__ import annotations

import math

from src.causal.estimators.base import BootstrapRunner
from src.causal.estimators.metalearners import TLearnerAdapter

from tests.test_causal.conftest import make_synthetic_cohort_frame


def test_three_arm_mu_c_all_non_nan():
    cf = make_synthetic_cohort_frame(n_per_arm=100, n_arms=3, ate=1.0, seed=7)
    runner = BootstrapRunner(
        TLearnerAdapter, cf, outcome_name="Y",
        B=50, random_state=0, alpha=0.05,
    )
    result = runner.run()
    assert len(result.mu_c) == 3
    for label, ui in result.mu_c.items():
        assert not math.isnan(ui.point)


def test_three_arm_pairwise_tau_matrix_populated():
    cf = make_synthetic_cohort_frame(n_per_arm=100, n_arms=3, ate=1.0, seed=7)
    runner = BootstrapRunner(
        TLearnerAdapter, cf, outcome_name="Y",
        B=50, random_state=0, alpha=0.05,
    )
    result = runner.run()
    # C(3,2) = 3.
    assert len(result.tau_cc_prime) == 3
    for ui in result.tau_cc_prime.values():
        assert not math.isnan(ui.point)
        assert ui.lower <= ui.point <= ui.upper
