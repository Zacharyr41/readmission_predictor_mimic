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
    """

    key = "t_learner"
    supported_outcome_types = ("binary", "continuous", "ordinal")

    def __init__(self, cohort, random_state: int = 0) -> None:
        raise NotImplementedError(
            "TLearnerAdapter — fleshed out in commit 3 of the 8d TDD trail. "
            "See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
        )

    def fit(self, cohort, outcome_name: str) -> None:
        raise NotImplementedError

    def predict_mu_per_arm(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        raise NotImplementedError

    @property
    def n_arms(self) -> int:
        raise NotImplementedError


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
