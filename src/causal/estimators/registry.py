"""Estimator registry for Phase 8d+.

Mirrors ``OperationRegistry`` at
``src/conversational/operations.py:264-328`` — same
``register`` / ``get`` / ``require`` / ``describe_for_prompt`` trio,
same duplicate-rejection discipline. Keyed by a single string
(the estimator's unique key) rather than ``(kind, name)`` because
there's only one kind of entry in this registry.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from typing import Type

from src.causal.estimators.base import (
    CausalEstimator,
    EstimatorOutcomeTypeMismatch,
)


class EstimatorRegistry:
    """Lookup table for causal estimator classes.

    Registered classes must carry two class-level attributes:

      * ``key: str`` — unique lookup key (e.g. ``"t_learner"``).
      * ``supported_outcome_types: tuple[str, ...]`` — values of
        ``OutcomeSpec.outcome_type`` this estimator accepts.

    The registry enforces both on ``register()`` and on
    ``check_outcome_type()``.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Type[CausalEstimator]] = {}

    # -- mutation ---------------------------------------------------------

    def register(self, est_cls: Type[CausalEstimator]) -> None:
        """Register an estimator class. Refuses duplicates.

        Raises:
            ValueError: if ``est_cls.key`` is missing / empty, or if
                another class is already registered under the same key.
        """
        key = getattr(est_cls, "key", None)
        if not key:
            raise ValueError(
                f"Estimator class {est_cls.__name__} must define a "
                "class-level 'key: str' attribute"
            )
        if key in self._entries:
            raise ValueError(f"Estimator already registered: key={key!r}")
        self._entries[key] = est_cls

    # -- lookup -----------------------------------------------------------

    def get(self, key: str) -> Type[CausalEstimator] | None:
        return self._entries.get(key)

    def require(self, key: str) -> Type[CausalEstimator]:
        """Like ``get`` but raises ``KeyError`` for unknown keys.

        Used by ``run_causal`` when the CQ's ``estimator_family`` has
        already been validated — a loud failure here means someone's
        threading an unregistered string into the dispatch.
        """
        if key not in self._entries:
            raise KeyError(
                f"No estimator registered for key={key!r}. "
                f"Registered: {sorted(self._entries)}"
            )
        return self._entries[key]

    def supported_keys(self) -> frozenset[str]:
        return frozenset(self._entries)

    # -- prompt integration ----------------------------------------------

    def describe_for_prompt(self) -> str:
        """Render the registered estimators for LLM-visible prompt inclusion.

        The conversational prompt builder (``src/conversational/prompts.py``)
        calls this so adding a new estimator family later (8g) extends the
        LLM-visible surface without prompt edits — same pattern as
        ``OperationRegistry.describe_for_prompt()`` at
        ``src/conversational/operations.py:317``.
        """
        if not self._entries:
            return "  (none registered)"
        lines = []
        for key in sorted(self._entries):
            cls = self._entries[key]
            types = ", ".join(cls.supported_outcome_types)
            lines.append(f"  * {key} — supports outcome types: {types}")
        return "\n".join(lines)

    # -- enforcement ------------------------------------------------------

    def check_outcome_type(
        self, est_cls: Type[CausalEstimator], outcome_type: str,
    ) -> None:
        """Raise ``EstimatorOutcomeTypeMismatch`` if ``est_cls`` does
        not support ``outcome_type``. No-op on success.
        """
        supported = getattr(est_cls, "supported_outcome_types", ())
        if outcome_type not in supported:
            raise EstimatorOutcomeTypeMismatch(
                f"Estimator {est_cls.__name__} does not support "
                f"outcome_type={outcome_type!r}; supports {supported}"
            )


def get_default_registry() -> EstimatorRegistry:
    """Return a fresh registry pre-populated with the 8d built-ins.

    Metalearner imports are lazy so consumers of the base package
    (non-causal CQs) never pull econml into memory.
    """
    from src.causal.estimators.metalearners import (
        SLearnerAdapter,
        TLearnerAdapter,
        XLearnerAdapter,
    )

    reg = EstimatorRegistry()
    reg.register(TLearnerAdapter)
    reg.register(SLearnerAdapter)
    reg.register(XLearnerAdapter)
    return reg
