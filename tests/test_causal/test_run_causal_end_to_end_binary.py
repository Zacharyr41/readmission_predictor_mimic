"""End-to-end Phase 8d: binary intervention, binary outcome.

Exercises run_causal's full dispatch path (validate → guard rails →
registry → bootstrap → merge) against a binary outcome. Uses
``make_synthetic_cohort_frame(binary_outcome=True)`` + the
``cohort_frame=`` injection API on ``run_causal`` so the test can
verify the pipeline without depending on DuckDB fixture outcome
variance (the 6-admission MIMIC-shape fixture lacks the class balance
LR needs in every bootstrap replicate).

The DuckDB-backend path is covered by 8c's
``test_cohort_frame.py`` / ``test_outcomes.py`` and by the non-causal
suite; 8d's focus here is the estimator + bootstrap + result-packaging
layer on top.
"""

from __future__ import annotations

import math

from src.causal.run import run_causal
from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)

from tests.test_causal.conftest import make_synthetic_cohort_frame


def _binary_tpa_cq() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="I3 — tPA effect on 30-day readmission",
        scope="causal_effect",
        intervention_set=[
            # The RxNorm code is irrelevant — we inject the cohort_frame
            # directly, skipping resolution. Labels must match the
            # synthetic cohort's intervention_labels ("arm0", "arm1").
            InterventionSpec(label="arm0", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="arm1", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ],
        outcome_vector=[
            OutcomeSpec(
                name="Y",
                outcome_type="binary",
                extractor_key="readmitted_30d",
            ),
        ],
        aggregation_spec=AggregationSpec(kind="identity"),
        estimator_family="t_learner",
        uncertainty_reps=30,
        random_state=0,
    )


def _binary_cohort():
    return make_synthetic_cohort_frame(
        n_per_arm=120, n_arms=2, ate=1.5, seed=0, binary_outcome=True,
    )


class TestRunCausalEndToEndBinary:
    def test_result_is_not_stub(self):
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        assert result.is_stub is False

    def test_mu_c_points_are_non_nan(self):
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        for label, ui in result.mu_c.items():
            assert not math.isnan(ui.point), f"{label} μ_c point is NaN"
            # Binary outcome → μ_c is P(Y=1 | X, T=c), bounded in [0, 1].
            assert 0.0 <= ui.point <= 1.0

    def test_tau_has_one_entry(self):
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        assert len(result.tau_cc_prime) == 1

    def test_ranking_has_both_arms(self):
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        assert len(result.ranking) == 2
        assert set(result.ranking) == {"arm0", "arm1"}

    def test_mode_stays_associative_until_8h(self):
        """8d doesn't compute assumption checks — mode stays
        associative. 8h owns the downgrade logic."""
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        assert result.mode == "associative"

    def test_uncertainty_kind_is_confidence(self):
        """8d ships bootstrap only; asymptotic + Bayesian are 8g."""
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        assert result.uncertainty_kind == "confidence"

    def test_overlap_diagnostic_populated(self):
        """Decision #3 — always-fit propensity means overlap shows up
        for every estimator, not just X-learner."""
        result = run_causal(
            _binary_tpa_cq(), backend=None, cohort_frame=_binary_cohort(),
        )
        assert "arm_0_min_propensity" in result.diagnostics.overlap
        assert "arm_1_min_propensity" in result.diagnostics.overlap
