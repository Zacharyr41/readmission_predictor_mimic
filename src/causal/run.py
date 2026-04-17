"""Top-level entry point for the causal-inference pipeline.

Phase 8a: this is a **stub**. It validates that the incoming
``CompetencyQuestion`` has the shape the downstream estimator will need,
then returns a ``CausalEffectResult`` populated with obviously-stub
numbers (μ_c=NaN point estimates, empty diagnostics, mode="associative",
``is_stub=True``).

The point of landing the stub now is end-to-end plumbing:

  * the planner already routes causal CQs to ``QueryPlan.CAUSAL``;
  * the orchestrator calls into this module;
  * the tests can assert the full result contract (``CausalEffectResult``
    fields, key-encoding scheme, ranking shape) against a working
    pipeline instead of a schema-only mock.

Real estimators land in 8d (T-learner via econml), 8e (S/X-learner),
8g (AIPW / TMLE / causal forest). Treatment-assignment / cohort
construction lands in 8b; outcome extraction (including survival) in 8c.
This file is the one that grows — its signature stays stable, its
implementation evolves.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from src.causal.models import (
    AssumptionClaim,
    CausalEffectResult,
    DiagnosticReport,
    UncertaintyInterval,
)
from src.conversational.models import CompetencyQuestion


class CausalQuestionInvalid(ValueError):
    """Raised when a CQ reaches ``run_causal`` without the minimum
    shape needed for a causal estimate (scope mismatch, |I| < 2, no
    outcome_vector). The orchestrator catches this and falls back;
    direct callers see it so bugs surface early."""


def _validate_causal_cq(cq: CompetencyQuestion) -> tuple[list, list]:
    """Return (interventions, outcomes) or raise CausalQuestionInvalid.

    Centralised here so each 8d-h estimator doesn't repeat the check.
    The planner already guards ``|I| ≥ 2`` before routing here, but this
    re-check makes the entry-point safe to call from tests and future
    non-orchestrator callers.
    """
    if cq.scope != "causal_effect":
        raise CausalQuestionInvalid(
            f"run_causal requires scope='causal_effect'; got {cq.scope!r}"
        )
    interventions = cq.intervention_set or []
    if len(interventions) < 2:
        raise CausalQuestionInvalid(
            f"causal inference requires at least 2 interventions; got {len(interventions)}"
        )
    outcomes = cq.outcome_vector or []
    if not outcomes:
        raise CausalQuestionInvalid(
            "causal inference requires a non-empty outcome_vector"
        )
    # Labels used as result keys must be unique — otherwise mu_c entries
    # collide silently. Catch upfront.
    labels = [i.label for i in interventions]
    if len(set(labels)) != len(labels):
        raise CausalQuestionInvalid(
            f"intervention labels must be unique; got {labels!r}"
        )
    names = [o.name for o in outcomes]
    if len(set(names)) != len(names):
        raise CausalQuestionInvalid(
            f"outcome names must be unique; got {names!r}"
        )
    return interventions, outcomes


def _stub_interval() -> UncertaintyInterval:
    """A transparent placeholder. Using NaN rather than 0.0 so downstream
    consumers can't mistake stub output for a real estimate."""
    return UncertaintyInterval(
        point=float("nan"),
        lower=float("nan"),
        upper=float("nan"),
    )


def run_causal(
    cq: CompetencyQuestion,
    backend: Any | None = None,
    *,
    estimator_family: str | None = None,
) -> CausalEffectResult:
    """Compute causal estimands for ``cq``.

    **Phase 8a behaviour**: stub. Validates the CQ, then returns a
    well-shaped ``CausalEffectResult`` whose numeric fields are NaN and
    whose ``is_stub`` flag is ``True``. The shape exactly matches what
    8d–h will produce for real — downstream consumers (orchestrator, UI,
    tests) can be written now against the real contract.

    Args:
        cq: a ``CompetencyQuestion`` with ``scope="causal_effect"``,
            ``intervention_set`` of size ≥ 2, and a non-empty
            ``outcome_vector``.
        backend: database backend (DuckDB or BigQuery). Ignored by the
            stub; 8b's cohort construction will use it.
        estimator_family: registry key. Ignored by the stub; 8d's
            ``EstimatorRegistry`` will dispatch on it. Defaults to the
            CQ's ``estimator_family``.

    Returns:
        A ``CausalEffectResult`` with ``is_stub=True``.

    Raises:
        CausalQuestionInvalid: if the CQ shape is not a valid causal
            question (scope, arity, or uniqueness violation).
    """
    interventions, outcomes = _validate_causal_cq(cq)
    _ = backend  # explicitly consumed once 8b lands
    _ = estimator_family or cq.estimator_family  # dispatched on in 8d

    # §7.1 mu_c — one interval per intervention.
    mu_c: dict[str, UncertaintyInterval] = {
        i.label: _stub_interval() for i in interventions
    }

    # §7.2 mu_c_k — one interval per (intervention, outcome). Key
    # encoding "<intervention>|<outcome>" documented in
    # src/causal/models.py::CausalEffectResult.
    mu_c_k: dict[str, UncertaintyInterval] = {
        f"{i.label}|{o.name}": _stub_interval()
        for i in interventions
        for o in outcomes
    }

    # §7.3 tau_{c,c'} — C(|I|, 2) unordered pairs. Key encoding
    # "<c>|<c_prime>" with lexicographic order so pair identity is stable
    # regardless of which arm the caller called "c" vs. "c'".
    tau_cc_prime: dict[str, UncertaintyInterval] = {}
    for c, c_prime in itertools.combinations(
        sorted(i.label for i in interventions), 2
    ):
        tau_cc_prime[f"{c}|{c_prime}"] = _stub_interval()

    # §7.4 ranking — with NaN points we can't actually rank. For the
    # stub, preserve the input order so the UI can still render a
    # labelled list without implying an ordering claim.
    ranking = [i.label for i in interventions]

    # §7.6 assumption ledger — caller declared mode="causal" with no
    # diagnostic backing; the 8h phase will check assumptions against
    # the real estimator diagnostics and downgrade to associative when
    # any fail. For 8a we record the caller's declaration verbatim.
    ledger = [
        AssumptionClaim(
            name="consistency",
            status="declared",
            detail="8a stub — assumption ledger will be populated by 8h diagnostics.",
        ),
        AssumptionClaim(
            name="ignorability",
            status="declared",
            detail="8a stub — no propensity model yet; no unmeasured confounding check.",
        ),
        AssumptionClaim(
            name="positivity",
            status="declared",
            detail="8a stub — overlap diagnostic arrives in 8h.",
        ),
        AssumptionClaim(
            name="sutva",
            status="declared",
            detail="8a stub — SUTVA is declared by the caller; interference checks are out of scope.",
        ),
    ]

    return CausalEffectResult(
        mu_c=mu_c,
        mu_c_k=mu_c_k,
        tau_cc_prime=tau_cc_prime,
        ranking=ranking,
        diagnostics=DiagnosticReport(
            notes=[
                "Phase 8a stub — no real cohort, no estimator, no diagnostics. "
                "Fields exist so downstream consumers can be written against the final shape."
            ],
        ),
        # Stub can never claim "causal" because it hasn't actually checked
        # any assumption; "associative" is the honest downgrade per spec §5.
        mode="associative",
        assumption_ledger=ledger,
        uncertainty_kind="confidence",
        alpha=cq.alpha,
        is_stub=True,
    )


# Sanity: ensure NaN is importable via math so the module-level import
# graph stays self-contained. (math is a stdlib module; this line is
# purely for readers who want to follow the stub convention.)
_ = math.nan
