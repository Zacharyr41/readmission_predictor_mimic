"""Cohort-frame assembly (Phase 8c).

Given a causal ``CompetencyQuestion`` and a database backend, produce
one DataFrame ready for the estimator stage (Phase 8d):

    [subject_id, hadm_id, T (int), intervention_label,
     *covariates, *outcomes]

The pieces assembled here are implemented elsewhere in the package —
this module just wires them together:

  * Population predicate φ_A → cohort hadm_ids
    (``src.conversational.extractor._get_filtered_hadm_ids``).
  * Intervention predicates → per-admission T
    (``src.causal.interventions.InterventionResolver`` +
    ``src.causal.treatment_assignment.assign_treatments``).
  * Covariates X → numeric columns
    (``src.causal.covariates.build_covariate_matrix``).
  * Outcomes Y_1 … Y_n → per-outcome columns
    (``src.causal.outcomes.get_default_registry``).

Row-drop policy
---------------

A cohort admission is retained in the returned frame iff it has:

  * a non-None ``T`` — so admissions matching zero or ≥2 arms drop
    (counted in ``n_unassigned`` / ``n_overlapping`` diagnostics);
  * a full set of covariates (no NaN);
  * a value for every outcome in the spec's outcome_vector (NaN
    biomarker values are allowed and carried through — the estimator
    decides how to handle them).

Dropped rows are counted in the returned ``CohortFrame.provenance``
so downstream diagnostics (8h) can audit exclusion reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from src.causal.covariates import (
    CovariateProfile,
    build_covariate_matrix,
)
from src.causal.interventions import InterventionResolver
from src.causal.outcomes import OutcomeRegistry, get_default_registry
from src.causal.treatment_assignment import TreatmentAssignment, assign_treatments
from src.conversational.models import CompetencyQuestion

logger = logging.getLogger(__name__)


class _SupportsExecute(Protocol):
    def execute(self, sql: str, params: list) -> list[tuple]: ...


class CausalCohortError(ValueError):
    """Raised when the CQ cannot be assembled into a valid cohort frame
    (no interventions, no outcomes, no cohort admissions, etc.)."""


@dataclass(frozen=True)
class CohortFrame:
    """Assembled cohort ready for the estimator stage.

    Attributes:
        df: one row per retained cohort admission. Columns:
            ``[subject_id, hadm_id, T, intervention_label,
            n_matching, *covariate_columns, *outcome_columns]``.
            Survival outcomes contribute two columns per outcome
            (``<name>_time``, ``<name>_event``).
        covariate_columns: the numeric X columns in ``df`` the
            estimator should use.
        outcome_columns: the Y columns (one per scalar outcome; two
            per survival outcome).
        treatment_column: always ``"T"`` for now; documented in case
            future phases need a different name.
        treatment_assignment: the full assignment result including
            diagnostic counts (n_assigned, n_unassigned, n_overlapping,
            per_arm_matched). Preserved so 8h's diagnostics can surface
            exclusion reasons.
        provenance: merge of every stage's provenance plus row-drop
            reasons — the auditable trace a reviewer needs.
    """

    df: pd.DataFrame
    covariate_columns: list[str]
    outcome_columns: list[str]
    treatment_column: str
    intervention_labels: list[str]
    treatment_assignment: TreatmentAssignment
    provenance: dict = field(default_factory=dict)


def build_cohort_frame(
    cq: CompetencyQuestion,
    backend: _SupportsExecute,
    *,
    covariate_profile: CovariateProfile = "demographics",
    resolver: InterventionResolver | None = None,
    outcome_registry: OutcomeRegistry | None = None,
    cohort_hadm_ids: list[int] | None = None,
) -> CohortFrame:
    """Assemble the full cohort DataFrame for a causal CQ.

    Args:
        cq: must have ``scope="causal_effect"`` with a non-empty
            ``intervention_set`` (|I| ≥ 2) and ``outcome_vector``.
        backend: database backend exposing ``execute(sql, params)``.
            When ``cohort_hadm_ids`` is supplied directly, any
            ``.execute(sql, params) -> list[tuple]`` shape works;
            otherwise the backend must be a
            ``src.conversational.extractor._DuckDBBackend`` or
            ``_BigQueryBackend`` (the filter-compilation path needs
            ``.table()`` / ``.ilike()`` / ``.random_fn()``).
        covariate_profile: registered profile name; defaults to
            ``"demographics"``. Future phases add ``"full"``.
        resolver / outcome_registry: dependency injection for tests.
            Production code passes ``None`` and takes the defaults.
        cohort_hadm_ids: bypass the filter-compilation step by supplying
            the cohort directly (used by tests + by upstream code that
            has already computed the cohort for other reasons).
    """
    interventions = cq.intervention_set or []
    outcomes = cq.outcome_vector or []
    if cq.scope != "causal_effect":
        raise CausalCohortError(
            f"build_cohort_frame requires scope='causal_effect'; got {cq.scope!r}"
        )
    if len(interventions) < 2:
        raise CausalCohortError(
            f"causal cohort requires ≥ 2 interventions; got {len(interventions)}"
        )
    if not outcomes:
        raise CausalCohortError("causal cohort requires a non-empty outcome_vector")

    resolver = resolver or InterventionResolver()
    outcome_registry = outcome_registry or get_default_registry()

    # 1. Population predicate → cohort hadm_ids.
    if cohort_hadm_ids is None:
        cohort_hadm_ids = _select_cohort_hadm_ids(cq, backend)
    if not cohort_hadm_ids:
        raise CausalCohortError(
            "population predicate (patient_filters) matched no admissions; "
            "cannot assemble a causal cohort"
        )

    # 2. Resolve interventions and assign treatments.
    resolved = [resolver.resolve(spec) for spec in interventions]
    assignment = assign_treatments(backend, resolved, cohort_hadm_ids)
    if assignment.n_assigned == 0:
        raise CausalCohortError(
            f"no cohort admission matched any intervention arm cleanly; "
            f"n_unassigned={assignment.n_unassigned}, "
            f"n_overlapping={assignment.n_overlapping}. "
            "Check interventions + cohort definition for mismatch."
        )

    # 3. Covariates for the admissions that actually got a T.
    retained_hadm_ids = [
        int(h) for h, t in zip(assignment.df["hadm_id"], assignment.df["T"]) if t is not None
    ]
    covariates = build_covariate_matrix(backend, retained_hadm_ids, profile=covariate_profile)
    covariate_columns = [c for c in covariates.columns if c != "hadm_id"]

    # 4. Outcomes.
    outcome_frames: list[pd.DataFrame] = []
    outcome_columns: list[str] = []
    outcome_provenance: list[dict[str, Any]] = []
    for spec in outcomes:
        df_y = outcome_registry.extract(spec, backend, retained_hadm_ids)
        if spec.outcome_type == "time_to_event":
            # Survival outcomes contribute two columns.
            df_y = df_y.rename(
                columns={"time": f"{spec.name}_time", "event": f"{spec.name}_event"}
            )
            outcome_columns.extend([f"{spec.name}_time", f"{spec.name}_event"])
        else:
            df_y = df_y.rename(columns={"value": spec.name})
            outcome_columns.append(spec.name)
        outcome_frames.append(df_y)
        outcome_provenance.append({
            "name": spec.name,
            "extractor_key": spec.extractor_key,
            "outcome_type": spec.outcome_type,
            "extractor_params": spec.extractor_params,
            "censoring_clock": spec.censoring_clock,
            "censoring_horizon_days": spec.censoring_horizon_days,
            "n_rows": len(df_y),
        })

    # 5. Merge on hadm_id. Start from the assignment frame so we carry
    # subject_id, intervention_label, and n_matching into the final output.
    df = assignment.df.copy()
    df = df[df["T"].notna()].reset_index(drop=True)
    df["hadm_id"] = df["hadm_id"].astype(int)
    df = df.merge(covariates, on="hadm_id", how="left")
    for f in outcome_frames:
        f["hadm_id"] = f["hadm_id"].astype(int)
        df = df.merge(f, on="hadm_id", how="left")

    # 6. Row-drop policy: drop rows with NaN in any covariate column
    # (this shouldn't happen because build_covariate_matrix raises on
    # missing age, but LEFT-join quirks can still leave NaN if a
    # hadm_id slipped through both directions).
    pre_drop = len(df)
    if covariate_columns:
        df = df.dropna(subset=covariate_columns).reset_index(drop=True)
    post_drop = len(df)
    dropped_for_covariates = pre_drop - post_drop

    provenance = {
        "cohort_hadm_ids_requested": len(cohort_hadm_ids),
        "treatment_assignment": assignment.provenance,
        "n_assigned": assignment.n_assigned,
        "n_unassigned": assignment.n_unassigned,
        "n_overlapping": assignment.n_overlapping,
        "n_dropped_for_covariate_missingness": dropped_for_covariates,
        "n_final_rows": post_drop,
        "covariate_profile": covariate_profile,
        "covariate_columns": covariate_columns,
        "outcomes": outcome_provenance,
        "resolver_version": "8c-2026-04-18",
    }

    return CohortFrame(
        df=df,
        covariate_columns=covariate_columns,
        outcome_columns=outcome_columns,
        treatment_column="T",
        intervention_labels=assignment.intervention_labels,
        treatment_assignment=assignment,
        provenance=provenance,
    )


def _select_cohort_hadm_ids(
    cq: CompetencyQuestion, backend: _SupportsExecute,
) -> list[int]:
    """Delegate to the existing filter-compilation path.

    Note: this path requires ``backend`` to conform to
    ``_DuckDBBackend`` / ``_BigQueryBackend`` — the filter-compilation
    uses ``.table()``, ``.ilike()``, ``.random_fn()`` which plain
    connection adapters don't expose. Callers that have already
    computed the cohort out-of-band (tests, upstream pipelines) should
    pass ``cohort_hadm_ids=`` directly to skip this.
    """
    from src.conversational.extractor import _get_filtered_hadm_ids

    return _get_filtered_hadm_ids(backend, cq.patient_filters or [])


__all__ = [
    "CausalCohortError",
    "CohortFrame",
    "build_cohort_frame",
]
