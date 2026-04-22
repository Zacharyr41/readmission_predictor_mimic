"""Registry plumbing for Phase 8d estimator dispatch.

Mirrors the ``OperationRegistry`` pattern at
``src/conversational/operations.py:264-328`` — duplicate ``register``
rejections, ``get`` by key, ``describe_for_prompt`` for the LLM surface.
Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from src.causal.estimators import (
    CausalEstimator,
    EstimatorOutcomeTypeMismatch,
    EstimatorRegistry,
    get_default_registry,
)


class _DummyEstimator:
    """Minimal registry-compatible estimator for plumbing tests."""

    key = "dummy"
    supported_outcome_types = ("continuous",)

    def __init__(self, cohort, random_state: int = 0) -> None:
        self._cohort = cohort
        self._n_arms = len(cohort.intervention_labels)

    def fit(self, cohort, outcome_name: str) -> None:
        pass

    def predict_mu_per_arm(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        return {c: np.zeros(len(X)) for c in range(self._n_arms)}

    @property
    def n_arms(self) -> int:
        return self._n_arms


class TestEstimatorRegistry:
    def test_register_and_get(self):
        reg = EstimatorRegistry()
        reg.register(_DummyEstimator)
        assert reg.get("dummy") is _DummyEstimator

    def test_duplicate_registration_rejected(self):
        reg = EstimatorRegistry()
        reg.register(_DummyEstimator)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_DummyEstimator)

    def test_unknown_key_raises(self):
        reg = EstimatorRegistry()
        with pytest.raises(KeyError, match="nope"):
            reg.require("nope")

    def test_supported_keys_listed(self):
        reg = EstimatorRegistry()
        reg.register(_DummyEstimator)
        assert "dummy" in reg.supported_keys()

    def test_describe_for_prompt_contains_registered_keys(self):
        reg = EstimatorRegistry()
        reg.register(_DummyEstimator)
        rendered = reg.describe_for_prompt()
        assert "dummy" in rendered

    def test_default_registry_has_t_learner(self):
        reg = get_default_registry()
        assert reg.get("t_learner") is not None

    def test_default_registry_has_s_and_x_learner(self):
        reg = get_default_registry()
        assert reg.get("s_learner") is not None
        assert reg.get("x_learner") is not None


class TestOutcomeTypeMismatch:
    def test_dummy_estimator_rejects_unsupported_outcome(self, seeded_cohort_frame):
        reg = EstimatorRegistry()
        reg.register(_DummyEstimator)
        est_cls = reg.require("dummy")
        est = est_cls(seeded_cohort_frame)
        # Dummy supports only 'continuous'. A 'binary' outcome should
        # raise — the registry entry declares supported_outcome_types.
        with pytest.raises(EstimatorOutcomeTypeMismatch, match="binary"):
            reg.check_outcome_type(est_cls, "binary")


class TestCausalEstimatorProtocol:
    def test_dummy_satisfies_protocol(self, seeded_cohort_frame):
        est = _DummyEstimator(seeded_cohort_frame)
        # Structural check — Protocol membership is duck-typed.
        assert isinstance(est, CausalEstimator)
