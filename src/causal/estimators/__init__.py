"""Phase 8d estimator stack — protocol, registry, adapters, bootstrap.

Public surface for the causal-inference estimator layer. Re-exports
the ``CausalEstimator`` protocol, the ``EstimatorRegistry`` (mirroring
``OperationRegistry`` at ``src/conversational/operations.py:264-328``),
the three typed exceptions that guard ``run_causal``, and the default
registry factory pre-populated with T/S/X-learner adapters.

See the plan at
``/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md``
for the five locked decisions driving this layer's design.
"""

from src.causal.estimators.base import (
    AggregationNotYetSupported,
    CausalEstimator,
    EstimatorOutcomeTypeMismatch,
    SurvivalNotYetSupported,
)
from src.causal.estimators.registry import (
    EstimatorRegistry,
    get_default_registry,
)

__all__ = [
    "AggregationNotYetSupported",
    "CausalEstimator",
    "EstimatorOutcomeTypeMismatch",
    "EstimatorRegistry",
    "SurvivalNotYetSupported",
    "get_default_registry",
]
