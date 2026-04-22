"""X-learner + always-fit propensity (Phase 8d, decision #3).

X-learner uses a propensity model internally. Decision #3 says we
always fit propensity inside run_causal even when the estimator
family (T/S) doesn't need one — so ``DiagnosticReport.overlap`` is
populated regardless of estimator choice. This test asserts the
X-learner path (and its propensity output) lands in the overlap
diagnostic correctly.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.causal.estimators.base import BootstrapRunner
from src.causal.estimators.metalearners import XLearnerAdapter

from tests.test_causal.conftest import make_synthetic_cohort_frame


def test_x_learner_overlap_dict_has_entry_per_arm():
    cf = make_synthetic_cohort_frame(n_per_arm=100, n_arms=2, ate=2.0, seed=1)
    runner = BootstrapRunner(
        XLearnerAdapter, cf, outcome_name="Y",
        B=20, random_state=0, alpha=0.05,
    )
    result = runner.run()

    for arm in range(2):
        min_key = f"arm_{arm}_min_propensity"
        max_key = f"arm_{arm}_max_propensity"
        assert min_key in result.diagnostics.overlap
        assert max_key in result.diagnostics.overlap


def test_x_learner_overlap_values_are_probabilities():
    cf = make_synthetic_cohort_frame(n_per_arm=100, n_arms=3, ate=1.0, seed=2)
    runner = BootstrapRunner(
        XLearnerAdapter, cf, outcome_name="Y",
        B=20, random_state=0, alpha=0.05,
    )
    result = runner.run()

    for arm in range(3):
        lo = result.diagnostics.overlap[f"arm_{arm}_min_propensity"]
        hi = result.diagnostics.overlap[f"arm_{arm}_max_propensity"]
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= hi
