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

    Fits one augmented learner on ``[X | one_hot(T)]`` and predicts
    ``μ_c(X)`` by setting the T one-hot to arm ``c`` and calling
    ``predict``. Generalizes trivially to C ≥ 2. Hand-rolled for the
    same reason T-learner is — "one model with treatment as feature"
    needs no econml ceremony.
    """

    key = "s_learner"
    supported_outcome_types = ("binary", "continuous", "ordinal")

    def __init__(
        self, cohort, random_state: int = 0, outcome_type: str = "continuous",
    ) -> None:
        self._cohort = cohort
        self._random_state = random_state
        self._outcome_type = outcome_type
        self._n_arms = len(cohort.intervention_labels)
        self._model = None

    @staticmethod
    def _one_hot(T: np.ndarray, n_arms: int) -> np.ndarray:
        return np.eye(n_arms, dtype=float)[T]

    def fit(self, cohort, outcome_name: str) -> None:
        from src.causal.estimators._base_learners import make_base_learner

        X_base = cohort.df[cohort.covariate_columns].to_numpy(dtype=float)
        T = cohort.df[cohort.treatment_column].to_numpy(dtype=int)
        Y = cohort.df[outcome_name].to_numpy()

        X_aug = np.hstack([X_base, self._one_hot(T, self._n_arms)])
        learner = make_base_learner(
            self._outcome_type, random_state=self._random_state,
        )
        learner.fit(X_aug, Y)
        self._model = learner

    def predict_mu_per_arm(self, X_df: pd.DataFrame) -> dict[int, np.ndarray]:
        if self._model is None:
            raise RuntimeError(
                "SLearnerAdapter.fit must be called before predict_mu_per_arm"
            )
        X_base = X_df[self._cohort.covariate_columns].to_numpy(dtype=float)
        n = len(X_base)
        out: dict[int, np.ndarray] = {}
        for c in range(self._n_arms):
            # Counterfactual input for arm c: T one-hot set to c.
            T_forced = np.full(n, c, dtype=int)
            X_aug = np.hstack([X_base, self._one_hot(T_forced, self._n_arms)])
            if self._outcome_type == "binary" and hasattr(self._model, "predict_proba"):
                out[c] = self._model.predict_proba(X_aug)[:, 1]
            else:
                out[c] = np.asarray(self._model.predict(X_aug), dtype=float)
        return out

    @property
    def n_arms(self) -> int:
        return self._n_arms


class XLearnerAdapter:
    """X-learner: bias-corrected T-learner with internal propensity.
    Registered as ``"x_learner"``.

    Wraps ``econml.metalearners.XLearner`` rather than hand-rolling:
    the X-learner recipe (per-arm outcome regressors + cross-imputed
    pseudo-outcomes + propensity-weighted combination) is non-trivial
    for C ≥ 2 and econml already implements it correctly. This honors
    the plan's decision that "econml is the estimator backbone from
    8d onwards" without pulling ceremony into the trivial cases
    (T/S-learner).

    After fitting, ``self._xlearner.models`` holds the per-arm
    outcome regressors; ``predict_mu_per_arm`` calls each for μ_c(X).
    The bias-corrected CATEs live at ``self._xlearner.effect(X, T0, T1)``
    — available to 8g's richer diagnostics if needed.

    Note: X-learner's "outcome regressor must implement fit/predict" —
    XGBoost regressor for continuous works; LogisticRegression for
    binary works because sklearn classifiers also implement predict().
    """

    key = "x_learner"
    supported_outcome_types = ("binary", "continuous", "ordinal")

    def __init__(
        self, cohort, random_state: int = 0, outcome_type: str = "continuous",
    ) -> None:
        self._cohort = cohort
        self._random_state = random_state
        self._outcome_type = outcome_type
        self._n_arms = len(cohort.intervention_labels)
        self._xlearner = None

    def fit(self, cohort, outcome_name: str) -> None:
        from sklearn.linear_model import LogisticRegression

        from econml.metalearners import XLearner

        from src.causal.estimators._base_learners import make_base_learner

        X = cohort.df[cohort.covariate_columns].to_numpy(dtype=float)
        T = cohort.df[cohort.treatment_column].to_numpy(dtype=int)
        Y = cohort.df[outcome_name].to_numpy()

        # econml clones the template model per arm internally.
        outcome_template = make_base_learner(
            self._outcome_type, random_state=self._random_state,
        )
        cate_template = make_base_learner(
            "continuous", random_state=self._random_state + 1,
        )
        propensity_model = LogisticRegression(
            max_iter=1000, random_state=self._random_state,
        )

        self._xlearner = XLearner(
            models=outcome_template,
            cate_models=cate_template,
            propensity_model=propensity_model,
        )
        self._xlearner.fit(Y, T, X=X)

    def predict_mu_per_arm(self, X_df: pd.DataFrame) -> dict[int, np.ndarray]:
        if self._xlearner is None:
            raise RuntimeError(
                "XLearnerAdapter.fit must be called before predict_mu_per_arm"
            )
        cols = self._cohort.covariate_columns
        X = X_df[cols].to_numpy(dtype=float)
        out: dict[int, np.ndarray] = {}
        # econml.metalearners.XLearner stores per-arm outcome regressors at
        # ``self.models`` — a list indexed by arm. This is the same access
        # pattern econml itself uses in ``effect()``.
        per_arm_models = self._xlearner.models
        for c in range(self._n_arms):
            model = per_arm_models[c]
            if self._outcome_type == "binary" and hasattr(model, "predict_proba"):
                out[c] = model.predict_proba(X)[:, 1]
            else:
                out[c] = np.asarray(model.predict(X), dtype=float)
        return out

    @property
    def n_arms(self) -> int:
        return self._n_arms
