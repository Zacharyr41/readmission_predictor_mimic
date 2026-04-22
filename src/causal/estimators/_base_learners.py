"""Outcome-type → base learner factory (Phase 8d).

Called by ``TLearnerAdapter`` / ``SLearnerAdapter`` / ``XLearnerAdapter``
in ``metalearners.py`` so the outcome_type → learner dispatch lives in
one place.

Decision #4 (plan file): continuous outcomes use ``xgboost.XGBRegressor``
— already a repo dep via ``src/prediction/model.py`` — not
``sklearn.ensemble.GradientBoostingRegressor``. Keeps the gradient-
boosting flavor consistent with existing predictive work.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from src.causal.estimators.base import SurvivalNotYetSupported


def make_base_learner(outcome_type: str, random_state: int = 0):
    """Return a fresh unfitted sklearn / xgboost estimator for
    ``outcome_type``.

    Args:
        outcome_type: one of ``binary``, ``continuous``, ``ordinal``,
            ``time_to_event``. The first three return a fitted-able
            sklearn / xgboost class instance; ``time_to_event`` raises
            ``SurvivalNotYetSupported`` (Phase 8g will add the
            survival branch).
        random_state: threaded through to the underlying learner for
            reproducibility. Matches ``CompetencyQuestion.random_state``
            when called from ``run_causal``.

    Raises:
        SurvivalNotYetSupported: on ``time_to_event`` (see Phase 8g).
        ValueError: on any other unknown ``outcome_type``.
    """
    if outcome_type == "binary":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=random_state)

    if outcome_type == "continuous":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=200,
            random_state=random_state,
            verbosity=0,
            n_jobs=1,  # deterministic; bootstrap parallelism lives above
        )

    if outcome_type == "ordinal":
        from sklearn.linear_model import LogisticRegression

        # sklearn 1.5+ picks multinomial automatically when classes > 2;
        # the explicit ``multi_class="multinomial"`` kwarg was deprecated.
        # Default behaviour is correct for ordinal-as-multinomial here.
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

    if outcome_type == "time_to_event":
        raise SurvivalNotYetSupported(
            "time_to_event outcomes require a survival-aware learner "
            "(Kaplan-Meier / Cox / RMST). Phase 8d ships scalar learners "
            "only — see Phase 8g."
        )

    raise ValueError(
        f"Unknown outcome_type: {outcome_type!r} "
        "(expected one of: binary, continuous, ordinal, time_to_event)"
    )
