"""8d guard rail: non-identity AggregationSpec raises.

Weighted-sum / dominant / utility aggregation is Phase 8f. 8d
emits per-outcome μ_{c,k} + per-arm μ_c but does not compose them
into a scalar composite yet. Typed error prevents silent misuse.
"""

from __future__ import annotations

import pytest

from src.causal.estimators import AggregationNotYetSupported
from src.causal.run import run_causal
from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)


def _cq_with_weighted_aggregation() -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="CQ with a weighted-sum aggregation",
        scope="causal_effect",
        intervention_set=[
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ],
        outcome_vector=[
            OutcomeSpec(name="y1", outcome_type="binary", extractor_key="readmitted_30d"),
            OutcomeSpec(name="y2", outcome_type="binary", extractor_key="readmitted_30d"),
        ],
        aggregation_spec=AggregationSpec(
            kind="weighted_sum",
            weights={"y1": 1.0, "y2": 2.0},
        ),
    )


def test_weighted_sum_aggregation_raises_typed_error(duckdb_backend):
    cq = _cq_with_weighted_aggregation()
    with pytest.raises(AggregationNotYetSupported, match="Phase 8f"):
        run_causal(cq, duckdb_backend)
