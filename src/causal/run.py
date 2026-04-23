"""Top-level entry point for the causal-inference pipeline.

Phase 8d rewrite: this module went from stub (8a) to real dispatch.
``run_causal(cq, backend)`` now builds a real cohort, picks the
registered estimator class, runs the stratified bootstrap once per
outcome, and returns a fully-populated ``CausalEffectResult`` with
real point estimates and bootstrap CIs — ``is_stub=False``.

Locked plan decisions reflected here
====================================

* **Decision #2** — ``query_patient_covariates=None`` returns the ATE
  (average per-row prediction across the entire cohort) rather than
  "prediction at mean X." For non-linear base learners like XGBoost
  the two numbers genuinely diverge; mean-of-predictions is the
  honest population estimand. Personalized estimates come from
  passing a covariate dict.

* **Decision #3** — overlap diagnostic is always populated. The
  ``BootstrapRunner`` fits a multinomial propensity on the full
  cohort as its last step, so ``DiagnosticReport.overlap`` has a
  consistent shape across every ``estimator_family`` choice.

Scope boundaries enforced with typed errors
===========================================

* ``SurvivalNotYetSupported`` — any ``OutcomeSpec`` with
  ``outcome_type == "time_to_event"`` is rejected with a pointer to
  Phase 8g. 8c still assembles survival columns correctly; this
  module simply doesn't fit on them yet.

* ``AggregationNotYetSupported`` — any ``AggregationSpec`` with
  ``kind != "identity"`` is rejected with a pointer to Phase 8f.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from typing import Any

from src.causal.cohort import build_cohort_frame
from src.causal.estimators import (
    AggregationNotYetSupported,
    EstimatorRegistry,
    SurvivalNotYetSupported,
    get_default_registry,
)
from src.causal.estimators.base import BootstrapRunner
from src.causal.models import (
    CausalEffectResult,
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

    Centralised so each downstream branch doesn't repeat the check.
    The planner already guards ``|I| ≥ 2`` before routing here, but
    this re-check makes the entry-point safe to call from tests and
    future non-orchestrator callers.
    """
    if cq.scope != "causal_effect":
        raise CausalQuestionInvalid(
            f"run_causal requires scope='causal_effect'; got {cq.scope!r}"
        )
    interventions = cq.intervention_set or []
    if len(interventions) < 2:
        raise CausalQuestionInvalid(
            f"causal inference requires at least 2 interventions; "
            f"got {len(interventions)}"
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


def run_causal(
    cq: CompetencyQuestion,
    backend: Any | None = None,
    *,
    estimator_family: str | None = None,
    registry: EstimatorRegistry | None = None,
    cohort_hadm_ids: list[int] | None = None,
    cohort_frame: Any | None = None,
) -> CausalEffectResult:
    """Compute causal estimands for a ``scope='causal_effect'`` CQ.

    Pipeline:

    1. ``_validate_causal_cq`` — structural checks on the CQ.
    2. Guard rails — reject non-identity aggregation (→ 8f) and
       survival outcomes (→ 8g) with typed exceptions.
    3. Resolve ``estimator_family`` → class via the registry; check
       every ``OutcomeSpec.outcome_type`` against the class's
       ``supported_outcome_types``.
    4. ``build_cohort_frame`` — 8c's entry-point; assembles the full
       per-admission DataFrame.
    5. For each outcome: spin a ``BootstrapRunner`` and ``.run()``.
       First outcome contributes ``μ_c`` / ``τ_{c,c'}`` / ranking /
       diagnostics / assumption ledger; every outcome contributes
       entries to the accumulated ``μ_{c,k}`` dict.
    6. Return a merged ``CausalEffectResult`` with
       ``is_stub=False``, ``mode='associative'`` (8h owns mode
       transitions), ``uncertainty_kind='confidence'``.

    Args:
        cq: a ``CompetencyQuestion`` with ``scope='causal_effect'``,
            ``intervention_set`` of size ≥ 2, and a non-empty
            ``outcome_vector``.
        backend: database backend (DuckDB or BigQuery adapter). Passed
            through to ``build_cohort_frame``.
        estimator_family: override the CQ's ``estimator_family``.
            ``None`` ⇒ use ``cq.estimator_family`` (default
            ``"t_learner"`` per ``src/conversational/models.py:212``).
        registry: override the default ``EstimatorRegistry``. Used by
            tests + future phases that want to swap estimator sets.

    Raises:
        CausalQuestionInvalid: CQ-shape violations.
        AggregationNotYetSupported: ``AggregationSpec.kind != 'identity'``.
        SurvivalNotYetSupported: any outcome has ``outcome_type='time_to_event'``.
        EstimatorOutcomeTypeMismatch: registered estimator doesn't
            support one of the CQ's outcome types.

    Returns:
        A real (non-stub) ``CausalEffectResult``. Multi-outcome CQs
        get a per-outcome ``μ_{c,k}`` grid; the ``μ_c`` / ``τ`` /
        ranking reflect the first outcome only until 8f lands
        proper aggregation.
    """
    interventions, outcomes = _validate_causal_cq(cq)

    # Guard: non-identity aggregation → 8f.
    agg = cq.aggregation_spec
    if agg is not None and agg.kind != "identity":
        raise AggregationNotYetSupported(
            f"AggregationSpec.kind={agg.kind!r} is not supported in "
            "Phase 8d. Multi-outcome composition (weighted_sum, "
            "dominant, utility) arrives in Phase 8f; for now pass "
            "AggregationSpec(kind='identity') and consume mu_c_k "
            "per-outcome manually."
        )

    # Guard: survival outcomes → 8g.
    for spec in outcomes:
        if spec.outcome_type == "time_to_event":
            raise SurvivalNotYetSupported(
                f"OutcomeSpec {spec.name!r} has outcome_type="
                "'time_to_event'; Phase 8d fits scalar learners only. "
                "The 8c cohort builder at src/causal/cohort.py assembles "
                "(time, event) columns correctly but the survival "
                "estimator (Kaplan-Meier / Cox / RMST) is Phase 8g."
            )

    # Registry lookup + pre-flight outcome-type check.
    reg = registry or get_default_registry()
    family = estimator_family or cq.estimator_family or "t_learner"
    est_cls = reg.require(family)
    for spec in outcomes:
        reg.check_outcome_type(est_cls, spec.outcome_type)

    # Phase 9: similarity-based cohort narrowing. When the CQ carries
    # a ``similarity_spec``, we compute similarity against the population
    # first, take the top-K hadm_ids, and pass them as the causal cohort.
    # The similarity summary is threaded into DiagnosticReport.notes so
    # the UI + investigator can trace cohort selection. When a caller
    # has already pre-built the cohort (``cohort_hadm_ids`` or
    # ``cohort_frame``), we DON'T re-run similarity — but we still emit
    # an audit note so reviewers can see the spec was honored.
    similarity_note: str | None = None
    if cq.similarity_spec is not None:
        spec = cq.similarity_spec
        anchor_desc = (
            f"hadm_id={spec.anchor_hadm_id}" if spec.anchor_hadm_id is not None
            else f"subject_id={spec.anchor_subject_id}" if spec.anchor_subject_id is not None
            else "template anchor"
        )
        if cohort_frame is None and cohort_hadm_ids is None:
            from src.similarity.run import run_similarity

            sim_result = run_similarity(spec, backend)
            cohort_hadm_ids = [s.hadm_id for s in sim_result.scores]
            similarity_note = (
                f"Phase 9 — cohort narrowed by similarity to "
                f"{sim_result.anchor_description} "
                f"(top_k={spec.top_k}, "
                f"n_pool={sim_result.n_pool}, n_returned={sim_result.n_returned})."
            )
        else:
            similarity_note = (
                f"Phase 9 — similarity_spec present (anchor: {anchor_desc}, "
                f"top_k={spec.top_k}) but the caller supplied a pre-built "
                "cohort; narrowing skipped."
            )

    # Assemble cohort via the 8c entry-point. Tests + callers with a
    # pre-computed cohort pass cohort_hadm_ids directly (mirrors the
    # escape-hatch at src/causal/cohort.py:125-128) so a thin backend
    # adapter that only implements .execute() suffices. Tests that want
    # to bypass the DB layer entirely (e.g. binary-outcome correctness
    # harnesses with a hand-built synthetic cohort) can pass
    # ``cohort_frame=`` directly and skip backend + build_cohort_frame.
    if cohort_frame is not None:
        cohort = cohort_frame
    else:
        cohort = build_cohort_frame(cq, backend, cohort_hadm_ids=cohort_hadm_ids)

    # Bootstrap once per outcome.
    B = getattr(cq, "uncertainty_reps", 200)
    random_state = getattr(cq, "random_state", 0)

    primary_result: CausalEffectResult | None = None
    mu_c_k_accum: dict[str, UncertaintyInterval] = {}

    for idx, spec in enumerate(outcomes):
        runner = BootstrapRunner(
            est_cls,
            cohort,
            outcome_name=spec.name,
            outcome_type=spec.outcome_type,
            B=B,
            random_state=random_state,
            alpha=cq.alpha,
            query_patient_covariates=cq.query_patient_covariates,
            higher_is_better=spec.higher_is_better,
        )
        result_i = runner.run()
        if idx == 0:
            primary_result = result_i
        for label, ui in result_i.mu_c.items():
            mu_c_k_accum[f"{label}|{spec.name}"] = ui

    assert primary_result is not None  # outcome_vector non-empty by validation

    # Merge: single-outcome CausalEffectResult from the primary runner
    # + the accumulated per-outcome μ_{c,k} grid.
    merged_notes = list(primary_result.diagnostics.notes)
    if similarity_note is not None:
        merged_notes.insert(0, similarity_note)
    if len(outcomes) > 1:
        merged_notes.append(
            f"Phase 8d — mu_c + tau + ranking reflect the first outcome "
            f"({outcomes[0].name!r}) only; multi-outcome composite "
            "ranking via AggregationSpec arrives in 8f."
        )
    merged_diagnostics = primary_result.diagnostics.model_copy(
        update={"notes": merged_notes}
    )

    return primary_result.model_copy(update={
        "mu_c_k": mu_c_k_accum,
        "diagnostics": merged_diagnostics,
    })
