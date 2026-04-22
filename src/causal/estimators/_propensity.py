"""Propensity model + overlap-diagnostic helper (Phase 8d, decision #3).

Decision #3 (plan file): ``run_causal`` always fits a multinomial
propensity model on the cohort — even when the chosen estimator family
(T/S-learner) doesn't need one internally — so
``DiagnosticReport.overlap`` has a consistent shape across every
estimator choice. The UI + 8h diagnostics can then treat overlap as a
uniform field rather than branching on ``estimator_family``.

Multi-class ``sklearn.linear_model.LogisticRegression`` generalizes to
C≥2 without a second code path; the X-learner adapter also reuses this
helper for its internal propensity.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_propensity(
    X: pd.DataFrame,
    T: pd.Series | np.ndarray,
    *,
    random_state: int = 0,
):
    """Fit P(T=c | X) as a multinomial logistic regression.

    Args:
        X: covariate DataFrame (numeric columns only — see
            ``src/causal/covariates.py`` for the build shape).
        T: integer arm index per row, matching ``X``.
        random_state: threaded through to sklearn.

    Returns:
        Fitted ``sklearn.linear_model.LogisticRegression``. Use
        ``model.predict_proba(X.values)`` to get the per-row per-arm
        propensity matrix; ``model.classes_`` gives the arm ordering.
    """
    from sklearn.linear_model import LogisticRegression

    # Note: sklearn >= 1.5 auto-selects multinomial for multiclass targets;
    # the explicit ``multi_class="multinomial"`` kwarg is deprecated there.
    # We rely on the default, which is correct across sklearn 1.3–1.7.
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X.values, np.asarray(T))
    return model


def overlap_from_propensity(
    model, X: pd.DataFrame,
) -> dict[str, float]:
    """Return the overlap diagnostic keys for ``DiagnosticReport.overlap``.

    Output shape: ``{arm_{c}_min_propensity, arm_{c}_max_propensity}``
    per arm ``c`` in ``model.classes_``. Values are per-row propensities
    extremized across the cohort — a low min_propensity flags that
    some covariate profile is essentially unrepresented in that arm
    (a positivity risk, fully diagnosed in 8h).
    """
    probs = model.predict_proba(X.values)  # (n_rows, n_arms)
    out: dict[str, float] = {}
    for i, arm in enumerate(model.classes_):
        out[f"arm_{int(arm)}_min_propensity"] = float(probs[:, i].min())
        out[f"arm_{int(arm)}_max_propensity"] = float(probs[:, i].max())
    return out
