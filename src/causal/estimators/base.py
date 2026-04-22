"""CausalEstimator protocol + shared exceptions for Phase 8d.

This module owns two things:

1. The ``CausalEstimator`` Protocol — the interface every registered
   estimator class must satisfy. Thin: ``fit`` + ``predict_mu_per_arm``
   + ``n_arms``. Bootstrap is NOT part of the protocol — it's wrapped
   by ``BootstrapRunner`` (landing in commit 3 of the 8d TDD trail) so
   future analytic-variance estimators (AIPW / TMLE in 8g) don't carry
   a vestigial bootstrap field.

2. The three typed exceptions that ``run_causal`` uses to guardrail
   out-of-scope CQs. Each points at the later sub-phase that will
   eventually accept it:

   * ``SurvivalNotYetSupported`` → 8g (econml metalearners don't
     consume (time, event) pairs — 8c emits them correctly; 8d just
     refuses to fit on them).
   * ``AggregationNotYetSupported`` → 8f (non-identity composition
     of the outcome vector).
   * ``EstimatorOutcomeTypeMismatch`` → caller bug (registered
     estimator doesn't declare support for the CQ's outcome type).

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class CausalEstimator(Protocol):
    """Duck-typed contract for registrable estimators.

    A class satisfies this protocol iff it exposes:

      * ``__init__(cohort: CohortFrame, random_state: int = 0)``
        Constructor; may do no-op work beyond recording the cohort.
      * ``fit(cohort, outcome_name: str) -> None``
        Fits per-arm outcome models for the named outcome column in
        ``cohort.df``.
      * ``predict_mu_per_arm(X: pd.DataFrame) -> dict[int, np.ndarray]``
        Returns per-arm predicted outcomes for the rows in ``X``
        (``X`` has the same columns as ``cohort.covariate_columns``).
      * ``n_arms: int`` (property)
        Number of intervention arms C. Matches
        ``len(cohort.intervention_labels)``.

    Class-level attributes — registry uses these, not isinstance:
      * ``key: str`` — unique registry lookup key.
      * ``supported_outcome_types: tuple[str, ...]`` — values from
        ``OutcomeSpec.outcome_type`` this estimator can consume.
    """

    def fit(self, cohort, outcome_name: str) -> None: ...

    def predict_mu_per_arm(self, X: pd.DataFrame) -> dict[int, np.ndarray]: ...

    @property
    def n_arms(self) -> int: ...


class SurvivalNotYetSupported(NotImplementedError):
    """Raised when a CQ contains an ``OutcomeSpec(outcome_type="time_to_event")``.

    ``src/causal/cohort.py:100-240`` correctly assembles survival columns
    into the ``CohortFrame`` (``<name>_time`` + ``<name>_event`` per spec),
    but econml's metalearners in Phase 8d don't consume (time, event)
    pairs. Survival estimators land in Phase 8g.
    """


class AggregationNotYetSupported(NotImplementedError):
    """Raised when ``AggregationSpec.kind != "identity"``.

    Multi-outcome composition (``weighted_sum``, ``dominant``,
    ``utility``) is Phase 8f. 8d emits per-outcome μ_{c,k} but does
    not compose them into a scalar composite.
    """


class EstimatorOutcomeTypeMismatch(ValueError):
    """Raised when a registered estimator class does not declare
    support for a CQ's outcome type.

    Checked up-front in ``run_causal`` so the failure surfaces at the
    dispatch boundary rather than mid-bootstrap. The registered
    estimator class advertises its capabilities via the class-level
    ``supported_outcome_types`` tuple; the registry's
    ``check_outcome_type()`` enforces the contract.
    """
