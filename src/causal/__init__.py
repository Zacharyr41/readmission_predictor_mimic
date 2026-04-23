"""Personalized causal / associative inference tool.

Implements the Neyman–Rubin potential-outcomes spec the user approved
2026-04-17. Phase 8 delivery plan:
``/Users/zacharyrothstein/.claude/plans/delegated-skipping-matsumoto.md``.

Phase 8a: output schema + dispatch stub.
Phase 8b: ontology-grounded intervention resolution + treatment assignment.
Phase 8c: cohort-frame assembly (covariates, outcomes incl. survival).
Phase 8d (this commit): real estimators (T/S/X-learner via econml) +
  BootstrapRunner with stratified resample + typed guard rails
  (``SurvivalNotYetSupported``, ``AggregationNotYetSupported``). See
  ``/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md``.

Downstream phases: 8f aggregation, 8g survival estimators + AIPW/TMLE/
Causal Forest + Bayesian uncertainty, 8h full diagnostics + mode
transitions, 8i LLM + Streamlit UX.
"""

from src.causal.cohort import CausalCohortError, CohortFrame, build_cohort_frame
from src.causal.covariates import (
    CovariateProfile,
    UnknownCovariateProfileError,
    build_covariate_matrix,
)
from src.causal.estimators import (
    AggregationNotYetSupported,
    CausalEstimator,
    EstimatorOutcomeTypeMismatch,
    EstimatorRegistry,
    SurvivalNotYetSupported,
    get_default_registry as get_default_estimator_registry,
)
from src.causal.estimators.base import BootstrapRunner
from src.causal.interventions import (
    InterventionResolutionError,
    InterventionResolver,
    ResolvedIntervention,
)
from src.causal.models import (
    AssumptionClaim,
    CausalEffectResult,
    DiagnosticReport,
    UncertaintyInterval,
)
from src.causal.outcomes import (
    OutcomeExtractionError,
    OutcomeExtractor,
    OutcomeRegistry,
    get_default_registry as get_default_outcome_registry,
)
from src.causal.run import run_causal
from src.causal.treatment_assignment import (
    InsufficientInterventionsError,
    TreatmentAssignment,
    assign_treatments,
)

__all__ = [
    "AggregationNotYetSupported",
    "AssumptionClaim",
    "BootstrapRunner",
    "CausalCohortError",
    "CausalEffectResult",
    "CausalEstimator",
    "CohortFrame",
    "CovariateProfile",
    "DiagnosticReport",
    "EstimatorOutcomeTypeMismatch",
    "EstimatorRegistry",
    "InsufficientInterventionsError",
    "InterventionResolutionError",
    "InterventionResolver",
    "OutcomeExtractionError",
    "OutcomeExtractor",
    "OutcomeRegistry",
    "ResolvedIntervention",
    "SurvivalNotYetSupported",
    "TreatmentAssignment",
    "UncertaintyInterval",
    "UnknownCovariateProfileError",
    "assign_treatments",
    "build_cohort_frame",
    "build_covariate_matrix",
    "get_default_estimator_registry",
    "get_default_outcome_registry",
    "run_causal",
]
