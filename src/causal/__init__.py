"""Personalized causal / associative inference tool.

Implements the Neyman–Rubin potential-outcomes spec the user approved
2026-04-17. Phase 8 delivery plan:
``/Users/zacharyrothstein/.claude/plans/delegated-skipping-matsumoto.md``.

Phase 8a (this module):
  * Output-side schema (``CausalEffectResult``, ``DiagnosticReport``,
    ``AssumptionClaim``, ``UncertaintyInterval``) in ``models``.
  * End-to-end dispatch stub (``run_causal``) so the orchestrator can
    wire up the ``QueryPlan.CAUSAL`` branch before any real estimator
    lands (8d+).

Downstream phases add real estimators (8d/8e/8f), diagnostics (8h),
LLM/UX integration (8i). Input-side schema
(``InterventionSpec``, ``OutcomeSpec``, ``AggregationSpec``) lives in
``src.conversational.models`` because it is part of the
``CompetencyQuestion`` extension surface, not the compute pipeline.
"""

from src.causal.cohort import CausalCohortError, CohortFrame, build_cohort_frame
from src.causal.covariates import (
    CovariateProfile,
    UnknownCovariateProfileError,
    build_covariate_matrix,
)
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
    "AssumptionClaim",
    "CausalCohortError",
    "CausalEffectResult",
    "CohortFrame",
    "CovariateProfile",
    "DiagnosticReport",
    "InsufficientInterventionsError",
    "InterventionResolutionError",
    "InterventionResolver",
    "OutcomeExtractionError",
    "OutcomeExtractor",
    "OutcomeRegistry",
    "ResolvedIntervention",
    "TreatmentAssignment",
    "UncertaintyInterval",
    "UnknownCovariateProfileError",
    "assign_treatments",
    "build_cohort_frame",
    "build_covariate_matrix",
    "get_default_outcome_registry",
    "run_causal",
]
