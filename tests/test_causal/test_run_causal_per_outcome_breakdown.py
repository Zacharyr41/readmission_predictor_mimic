"""Per-outcome μ_{c,k} breakdown (Phase 8d).

n=3 scalar outcomes in one outcome_vector (I6 shape minus the
aggregation step and minus survival — survival lands in 8g,
aggregation in 8f). Flips the xfail at
``tests/test_conversational/test_intervention_effect.py:655``.
"""

from __future__ import annotations

import math

from src.causal.estimators.base import BootstrapRunner
from src.causal.estimators.metalearners import TLearnerAdapter

from tests.test_causal.conftest import make_synthetic_cohort_frame


def test_per_outcome_breakdown_has_cxk_entries():
    outcome_names = ["bleeding_score", "stroke_recurrence_score", "los_days"]
    cf = make_synthetic_cohort_frame(
        n_per_arm=80, n_arms=3, ate=1.0, seed=9,
        outcome_names=outcome_names,
    )

    # Fit one bootstrap run per outcome, merge.
    mu_c_k: dict[str, object] = {}
    for outcome in outcome_names:
        runner = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name=outcome,
            B=20, random_state=0, alpha=0.05,
        )
        result = runner.run()
        for arm_label in result.mu_c:
            mu_c_k[f"{arm_label}|{outcome}"] = result.mu_c[arm_label]

    # 3 arms × 3 outcomes.
    assert len(mu_c_k) == 9
    for key, ui in mu_c_k.items():
        assert not math.isnan(ui.point), f"{key} μ_{{c,k}} point is NaN"
        assert ui.lower <= ui.point <= ui.upper


def test_per_outcome_keys_follow_pipe_convention():
    """Key encoding convention from 8a: '<intervention>|<outcome>'.
    Documented at ``src/causal/models.py:93-97``."""
    outcome_names = ["y1", "y2"]
    cf = make_synthetic_cohort_frame(
        n_per_arm=60, n_arms=2, ate=1.0, seed=0,
        outcome_names=outcome_names,
    )

    keys: list[str] = []
    for outcome in outcome_names:
        runner = BootstrapRunner(
            TLearnerAdapter, cf, outcome_name=outcome,
            B=10, random_state=0, alpha=0.05,
        )
        result = runner.run()
        for arm_label in result.mu_c:
            keys.append(f"{arm_label}|{outcome}")

    for k in keys:
        assert "|" in k
        arm, outcome = k.split("|", 1)
        assert outcome in outcome_names
