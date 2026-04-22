"""Metalearner adapters over econml (Phase 8d).

``TLearnerAdapter`` / ``SLearnerAdapter`` / ``XLearnerAdapter`` wrap
``econml.metalearners.{TLearner, SLearner, XLearner}`` into the thin
``CausalEstimator`` protocol defined at ``base.py``. The bootstrap
wrapper lives in ``base.py::BootstrapRunner`` and calls
``predict_mu_per_arm`` once per replicate.

Commit 2 of the 8d TDD trail (current): class shells registered in the
default registry. Bodies raise ``NotImplementedError``.
Commit 3: ``TLearnerAdapter`` + ``BootstrapRunner`` fleshed out.
Commit 4: ``SLearnerAdapter`` + ``XLearnerAdapter`` fleshed out.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TLearnerAdapter:
    """T-learner: one outcome model per arm. Registered as ``"t_learner"``.

    This is the default estimator selected when
    ``CompetencyQuestion.estimator_family`` is unset (see
    ``src/conversational/models.py:212``).

    Implementation: the "metalearner" aspect is trivial — just fit one
    sklearn/xgboost learner per arm via
    ``_base_learners.make_base_learner`` (decision #4: XGBoost for
    continuous). The heavier econml metalearners in 8g (DRLearner /
    CausalForestDML) will come in via ``econml.dml`` / ``econml.dr``.
    """

    key = "t_learner"
    supported_outcome_types = ("binary", "continuous", "ordinal")

    def __init__(
        self, cohort, random_state: int = 0, outcome_type: str = "continuous",
    ) -> None:
        self._cohort = cohort
        self._random_state = random_state
        self._outcome_type = outcome_type
        self._n_arms = len(cohort.intervention_labels)
        self._models: list | None = None

    def fit(self, cohort, outcome_name: str) -> None:
        from src.causal.estimators._base_learners import make_base_learner

        X = cohort.df[cohort.covariate_columns].to_numpy(dtype=float)
        T = cohort.df[cohort.treatment_column].to_numpy(dtype=int)
        Y = cohort.df[outcome_name].to_numpy()

        models: list = []
        for c in range(self._n_arms):
            mask = T == c
            if not mask.any():
                raise ValueError(
                    f"TLearnerAdapter.fit: arm {c} "
                    f"({cohort.intervention_labels[c]!r}) has no rows in the "
                    "cohort; per-arm fit impossible. Check cohort assembly / "
                    "bootstrap stratification."
                )
            learner = make_base_learner(
                self._outcome_type,
                random_state=self._random_state + c,
            )
            learner.fit(X[mask], Y[mask])
            models.append(learner)
        self._models = models

    def predict_mu_per_arm(self, X_df: pd.DataFrame) -> dict[int, np.ndarray]:
        if self._models is None:
            raise RuntimeError(
                "TLearnerAdapter.fit must be called before predict_mu_per_arm"
            )
        cols = self._cohort.covariate_columns
        X = X_df[cols].to_numpy(dtype=float)
        out: dict[int, np.ndarray] = {}
        for c, model in enumerate(self._models):
            if self._outcome_type == "binary" and hasattr(model, "predict_proba"):
                # μ_c = P(Y=1 | X, T=c) for binary outcomes.
                out[c] = model.predict_proba(X)[:, 1]
            else:
                out[c] = np.asarray(model.predict(X), dtype=float)
        return out

    @property
    def n_arms(self) -> int:
        return self._n_arms


class SLearnerAdapter:
    """S-learner: single outcome model with treatment as a feature.
    Registered as ``"s_learner"``.
    """

    key = "s_learner"
    supported_outcome_types = ("binary", "continuous", "ordinal")

    def __init__(self, cohort, random_state: int = 0) -> None:
        raise NotImplementedError(
            "SLearnerAdapter — fleshed out in commit 4 of the 8d TDD trail. "
            "See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
        )

    def fit(self, cohort, outcome_name: str) -> None:
        raise NotImplementedError

    def predict_mu_per_arm(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        raise NotImplementedError

    @property
    def n_arms(self) -> int:
        raise NotImplementedError


class XLearnerAdapter:
    """X-learner: bias-corrected T-learner with internal propensity.
    Registered as ``"x_learner"``.

    The internal propensity makes X-learner the only built-in that gets
    overlap diagnostics "for free" without calling ``_propensity.fit_propensity``
    — but per decision #3, we always fit propensity in ``run_causal``
    anyway so the diagnostic shape is consistent across estimator choice.
    """

    key = "x_learner"
    supported_outcome_types = ("binary", "continuous", "ordinal")

    def __init__(self, cohort, random_state: int = 0) -> None:
        raise NotImplementedError(
            "XLearnerAdapter — fleshed out in commit 4 of the 8d TDD trail. "
            "See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
        )

    def fit(self, cohort, outcome_name: str) -> None:
        raise NotImplementedError

    def predict_mu_per_arm(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        raise NotImplementedError

    @property
    def n_arms(self) -> int:
        raise NotImplementedError
