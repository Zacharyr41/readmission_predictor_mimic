"""End-to-end Phase 8d: binary intervention, scalar outcome (I3 shape).

Runs against the real ``synthetic_duckdb_for_causal`` backend so the
test exercises 8b's intervention resolution + 8c's cohort assembly +
8d's estimator + bootstrap in one pass.
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


def _binary_tpa_cq() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="I3 — tPA effect on 30-day readmission",
        scope="causal_effect",
        intervention_set=[
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ],
        outcome_vector=[
            OutcomeSpec(
                name="readmitted_30d",
                outcome_type="binary",
                extractor_key="readmitted_30d",
            ),
        ],
        aggregation_spec=AggregationSpec(kind="identity"),
        estimator_family="t_learner",
    )


class TestRunCausalEndToEndBinary:
    def test_result_is_not_stub(self, duckdb_backend):
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        assert result.is_stub is False

    def test_mu_c_points_are_non_nan(self, duckdb_backend):
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        for label, ui in result.mu_c.items():
            assert not math.isnan(ui.point), f"{label} μ_c point is NaN"

    def test_tau_has_one_entry(self, duckdb_backend):
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        assert len(result.tau_cc_prime) == 1

    def test_ranking_has_both_arms(self, duckdb_backend):
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        assert len(result.ranking) == 2
        assert set(result.ranking) == {"tPA", "no_tPA"}

    def test_mode_stays_associative_until_8h(self, duckdb_backend):
        """8d doesn't compute assumption checks — mode stays
        associative. 8h owns the downgrade logic."""
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        assert result.mode == "associative"

    def test_uncertainty_kind_is_confidence(self, duckdb_backend):
        """8d ships bootstrap only; asymptotic + Bayesian are 8g."""
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        assert result.uncertainty_kind == "confidence"

    def test_overlap_diagnostic_populated(self, duckdb_backend):
        """Decision #3 — always-fit propensity means overlap shows up
        for every estimator, not just X-learner."""
        cq = _binary_tpa_cq()
        result = run_causal(cq, duckdb_backend)
        assert "arm_0_min_propensity" in result.diagnostics.overlap
        assert "arm_1_min_propensity" in result.diagnostics.overlap
