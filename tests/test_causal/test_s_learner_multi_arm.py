"""S-learner on C=3 arms (Phase 8d).

S-learner fits one model with treatment encoded as a feature; the
adapter generalizes trivially to C≥2. Asserts the full μ_c / τ_{c,c'}
shape and the lexicographic key convention (matches the stub at
``src/causal/run.py:146-149``).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from src.causal.estimators.base import BootstrapRunner
from src.causal.estimators.metalearners import SLearnerAdapter

from tests.test_causal.conftest import make_synthetic_cohort_frame


def test_s_learner_produces_non_nan_mu_per_arm():
    cf = make_synthetic_cohort_frame(n_per_arm=80, n_arms=3, ate=1.5, seed=5)
    runner = BootstrapRunner(
        SLearnerAdapter, cf, outcome_name="Y",
        B=30, random_state=0, alpha=0.05,
    )
    result = runner.run()

    for label in cf.intervention_labels:
        assert label in result.mu_c
        ui = result.mu_c[label]
        assert not math.isnan(ui.point)
        assert ui.lower <= ui.point <= ui.upper


def test_s_learner_tau_matrix_has_correct_shape_and_keys():
    cf = make_synthetic_cohort_frame(n_per_arm=80, n_arms=3, ate=1.5, seed=5)
    runner = BootstrapRunner(
        SLearnerAdapter, cf, outcome_name="Y",
        B=30, random_state=0, alpha=0.05,
    )
    result = runner.run()

    # C(3,2) = 3 unordered pairs.
    assert len(result.tau_cc_prime) == 3
    # Keys must use lexicographic ordering: arm0 < arm1 < arm2 → keys
    # "arm0|arm1", "arm0|arm2", "arm1|arm2".
    labels = sorted(cf.intervention_labels)
    expected_keys = {
        f"{a}|{b}"
        for a, b in itertools.combinations(labels, 2)
    }
    assert set(result.tau_cc_prime.keys()) == expected_keys


def test_s_learner_ranking_length_matches_arm_count():
    cf = make_synthetic_cohort_frame(n_per_arm=80, n_arms=3, ate=1.5, seed=5)
    runner = BootstrapRunner(
        SLearnerAdapter, cf, outcome_name="Y",
        B=30, random_state=0, alpha=0.05,
    )
    result = runner.run()
    assert len(result.ranking) == 3
    assert set(result.ranking) == set(cf.intervention_labels)
