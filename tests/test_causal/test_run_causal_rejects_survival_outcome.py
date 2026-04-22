"""8d guard rail: survival outcomes raise ``SurvivalNotYetSupported``.

8c's ``build_cohort_frame`` correctly assembles survival columns, but
econml's metalearners in 8d don't consume ``(time, event)`` pairs —
survival estimators land in 8g. run_causal must fail fast with a
typed error pointing at 8g so users don't get silently-wrong results.
"""

from __future__ import annotations

import pytest

from src.causal.estimators import SurvivalNotYetSupported
from src.causal.run import run_causal
from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)


def _cq_with_survival() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="CQ with a survival outcome",
        scope="causal_effect",
        intervention_set=[
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ],
        outcome_vector=[
            OutcomeSpec(
                name="mortality_90d",
                outcome_type="time_to_event",
                extractor_key="mortality_time_to_event",
                censoring_horizon_days=90,
            ),
        ],
        aggregation_spec=AggregationSpec(kind="identity"),
    )


def test_survival_outcome_raises_typed_error(duckdb_backend):
    cq = _cq_with_survival()
    with pytest.raises(SurvivalNotYetSupported, match="Phase 8g"):
        run_causal(cq, duckdb_backend)
