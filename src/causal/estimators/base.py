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


# ---------------------------------------------------------------------------
# Bootstrap runner (commit 3 of the 8d TDD trail).
#
# Wraps any ``CausalEstimator`` in a stratified bootstrap (decision #1),
# computes per-arm μ_c, paired τ_{c,c'} (decision: paired within-replicate
# differencing — honest correlation), and the overlap diagnostic via the
# always-fit propensity helper (decision #3). Returns a fully-populated
# ``CausalEffectResult`` for the single outcome the runner was constructed
# with; ``run_causal`` runs the runner once per outcome to build the full
# ``μ_{c,k}`` breakdown.
# ---------------------------------------------------------------------------


import itertools
from dataclasses import replace as _replace_dataclass
from typing import Type

from src.causal.models import (
    AssumptionClaim,
    CausalEffectResult,
    DiagnosticReport,
    UncertaintyInterval,
)


class BootstrapRunner:
    """Stratified bootstrap over a single outcome (Phase 8d).

    Given an estimator class, a cohort, and a named outcome column,
    resamples the cohort ``B`` times (stratified by arm — decision #1
    in ``/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md``),
    refits the estimator each replicate, and returns a
    ``CausalEffectResult`` with μ_c + pairwise τ_{c,c'} + overlap
    diagnostic populated.

    Decision #2: if ``query_patient_covariates`` is None, the runner
    predicts μ_c at every row of the cohort covariate matrix and
    averages — the standard ATE-via-avg-CATE formulation. For a
    specific patient, pass the covariate dict.

    Decision #3: the overlap diagnostic is always fit (multinomial
    propensity over X → T), independent of whether the estimator
    family consumes propensity internally. Keeps
    ``DiagnosticReport.overlap`` shape uniform across all estimators.

    Paired τ: within each replicate, τ_{c,c'} = μ̂_c - μ̂_{c'} is
    computed and quantiles of those paired differences form the CI —
    honest under noise shared between arms (e.g. same XGBoost random
    splits), narrower than the naive difference of marginal CIs.

    Args:
        estimator_cls: a class satisfying ``CausalEstimator``.
            Instantiated per replicate with
            ``(resampled_cohort, random_state, outcome_type=...)``.
        cohort: the full ``CohortFrame`` from ``build_cohort_frame``.
        outcome_name: column in ``cohort.df`` holding the outcome
            values (or for survival, the ``<name>_time`` /
            ``<name>_event`` pair — 8d rejects survival at the
            ``run_causal`` boundary, so here ``outcome_name`` is a
            scalar column).
        outcome_type: one of ``continuous`` / ``binary`` / ``ordinal``
            (``time_to_event`` unreachable — 8g). Drives base-learner
            selection via ``_base_learners.make_base_learner``.
        B: bootstrap replicate count. 200 per the plan default.
        random_state: threaded through every replicate and to the
            overlap propensity fit for reproducibility.
        alpha: (1 - alpha) confidence interval.
        query_patient_covariates: see decision #2 above.
        higher_is_better: rank direction for the returned ``ranking``.
    """

    def __init__(
        self,
        estimator_cls: Type[CausalEstimator],
        cohort,
        *,
        outcome_name: str,
        outcome_type: str = "continuous",
        B: int = 200,
        random_state: int = 0,
        alpha: float = 0.05,
        query_patient_covariates: dict | None = None,
        higher_is_better: bool = False,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if B < 1:
            raise ValueError(f"B must be >= 1; got {B}")

        self._estimator_cls = estimator_cls
        self._cohort = cohort
        self._outcome_name = outcome_name
        self._outcome_type = outcome_type
        self._B = B
        self._random_state = random_state
        self._alpha = alpha
        self._higher_is_better = higher_is_better

        if query_patient_covariates is not None:
            missing = set(cohort.covariate_columns) - set(query_patient_covariates)
            if missing:
                raise ValueError(
                    f"query_patient_covariates missing covariate columns: "
                    f"{sorted(missing)}"
                )
        self._query_patient = query_patient_covariates

    # -- internal --------------------------------------------------------

    def _stratified_resample(self, replicate_idx: int) -> pd.DataFrame:
        """Per-arm stratified resample preserving per-arm row counts.

        Deterministic given ``(self._random_state, replicate_idx)``.
        Exposed (underscore-prefixed) for the regression test
        ``test_bootstrap.py::TestStratifiedResample``.
        """
        df = self._cohort.df
        tcol = self._cohort.treatment_column
        n_arms = len(self._cohort.intervention_labels)
        sub_rng = np.random.default_rng(
            self._random_state + replicate_idx * 1_000_003 + 1
        )
        parts = []
        for c in range(n_arms):
            arm_df = df[df[tcol] == c]
            if len(arm_df) == 0:
                continue
            picks = sub_rng.integers(0, len(arm_df), size=len(arm_df))
            parts.append(arm_df.iloc[picks])
        return pd.concat(parts, ignore_index=True)

    def _make_replicate_cohort(self, resampled_df: pd.DataFrame):
        """Build a ``CohortFrame``-shaped object from a resample.

        ``CohortFrame`` is a frozen dataclass; ``dataclasses.replace``
        returns a fresh instance with ``df`` swapped. Downstream code
        (the adapter's ``fit``) reads ``df`` + the schema attrs only,
        so the stale ``treatment_assignment`` is harmless.
        """
        return _replace_dataclass(self._cohort, df=resampled_df)

    def _target_X(self) -> pd.DataFrame:
        """Prediction target per decision #2."""
        cols = self._cohort.covariate_columns
        if self._query_patient is not None:
            return pd.DataFrame([self._query_patient])[cols]
        return self._cohort.df[cols].copy()

    # -- public ----------------------------------------------------------

    def run(self) -> CausalEffectResult:
        labels = list(self._cohort.intervention_labels)
        n_arms = len(labels)
        X_target = self._target_X()

        mu_per_rep = np.full((self._B, n_arms), np.nan, dtype=float)
        for b in range(self._B):
            replicate_cohort = self._make_replicate_cohort(
                self._stratified_resample(b)
            )
            est = self._estimator_cls(
                replicate_cohort,
                random_state=self._random_state + b,
                outcome_type=self._outcome_type,
            )
            est.fit(replicate_cohort, self._outcome_name)
            per_arm = est.predict_mu_per_arm(X_target)
            for c in range(n_arms):
                mu_per_rep[b, c] = float(np.mean(per_arm[c]))

        # μ_c with quantile CIs (per-arm marginal).
        mu_c: dict[str, UncertaintyInterval] = {}
        for c, label in enumerate(labels):
            col = mu_per_rep[:, c]
            point = float(np.mean(col))
            lo = float(np.quantile(col, self._alpha / 2))
            hi = float(np.quantile(col, 1 - self._alpha / 2))
            mu_c[label] = UncertaintyInterval(point=point, lower=lo, upper=hi)

        # μ_{c,k} for the single outcome — same numbers, outcome-suffixed keys.
        mu_c_k: dict[str, UncertaintyInterval] = {
            f"{label}|{self._outcome_name}": mu_c[label] for label in labels
        }

        # Paired τ_{c,c'} via within-replicate differencing, lexicographic keys.
        tau_cc: dict[str, UncertaintyInterval] = {}
        for (c1, l1), (c2, l2) in itertools.combinations(enumerate(labels), 2):
            if l1 <= l2:
                key = f"{l1}|{l2}"
                diff_col = mu_per_rep[:, c1] - mu_per_rep[:, c2]
            else:
                key = f"{l2}|{l1}"
                diff_col = mu_per_rep[:, c2] - mu_per_rep[:, c1]
            point = float(np.mean(diff_col))
            lo = float(np.quantile(diff_col, self._alpha / 2))
            hi = float(np.quantile(diff_col, 1 - self._alpha / 2))
            tau_cc[key] = UncertaintyInterval(point=point, lower=lo, upper=hi)

        # Ranking by μ_c point, respecting the higher_is_better flag.
        ranking = sorted(
            labels,
            key=lambda lbl: mu_c[lbl].point,
            reverse=self._higher_is_better,
        )

        # Overlap diagnostic — always-fit propensity (decision #3).
        from src.causal.estimators._propensity import (
            fit_propensity,
            overlap_from_propensity,
        )

        X_cohort = self._cohort.df[self._cohort.covariate_columns]
        T_cohort = self._cohort.df[self._cohort.treatment_column]
        propensity_model = fit_propensity(
            X_cohort, T_cohort, random_state=self._random_state,
        )
        overlap = overlap_from_propensity(propensity_model, X_cohort)

        diagnostics = DiagnosticReport(
            overlap=overlap,
            notes=[
                "Phase 8d — overlap populated via always-fit propensity "
                "(decision #3). Full diagnostics (balance, positivity, "
                "extrapolation, missingness) arrive in 8h.",
            ],
        )

        ledger = [
            AssumptionClaim(
                name="consistency", status="declared",
                detail="Phase 8d — consistency is caller-declared; 8h adds checks.",
            ),
            AssumptionClaim(
                name="ignorability", status="declared",
                detail="Phase 8d — propensity is fit but ignorability is not "
                       "verified; 8h adds unmeasured-confounder sensitivity.",
            ),
            AssumptionClaim(
                name="positivity", status="declared",
                detail="Phase 8d — overlap diagnostic populated; positivity "
                       "threshold check lands in 8h.",
            ),
            AssumptionClaim(
                name="sutva", status="declared",
                detail="Phase 8d — SUTVA is caller-declared; interference "
                       "checks are out of scope.",
            ),
        ]

        return CausalEffectResult(
            mu_c=mu_c,
            mu_c_k=mu_c_k,
            tau_cc_prime=tau_cc,
            ranking=ranking,
            diagnostics=diagnostics,
            mode="associative",
            assumption_ledger=ledger,
            uncertainty_kind="confidence",
            alpha=self._alpha,
            is_stub=False,
        )
