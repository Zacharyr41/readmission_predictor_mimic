"""Treatment-vector construction with mutual-exclusivity enforcement.

Given a cohort of admission IDs and a list of resolved interventions,
compute a per-admission treatment index ``T_i ∈ {0, …, C-1} ∪ {None}``
where ``None`` means either (a) the admission matched zero
interventions, or (b) it matched multiple interventions (overlap
violating SUTVA / mutual exclusivity). Both failure modes are distinct
reasons to exclude the admission from the causal analysis; the
diagnostic carries counts for each so the assumption ledger (8h) can
surface them.

For ``is_control=True`` interventions the ``EXISTS`` predicate is
negated — the control arm matches admissions that did NOT receive the
referenced ontology code. Binary shapes (drug vs not-drug) thus satisfy
mutual exclusivity by construction; N-ary shapes rely on the caller
picking non-overlapping interventions (or on the resolver correctly
expanding class concepts in 8c+).

BigQuery support is explicitly deferred — the fragment SQL uses ``?``
positional parameters which DuckDB accepts directly. 8d adds a param-
style adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import pandas as pd

from src.causal.interventions import ResolvedIntervention

logger = logging.getLogger(__name__)


class _SupportsExecute(Protocol):
    """Minimum backend surface used by the assigner.

    Matches ``_DuckDBBackend`` and ``_BigQueryBackend`` from
    ``src.conversational.extractor`` — we don't import those directly
    to keep this module testable with a plain DuckDB connection via a
    tiny adapter.
    """

    def execute(self, sql: str, params: list) -> list[tuple]:
        ...


@dataclass(frozen=True)
class TreatmentAssignment:
    """Per-admission treatment vector + diagnostic counts.

    Attributes:
        df: one row per cohort admission. Columns:
            ``subject_id``, ``hadm_id``, ``T`` (int index into
            ``intervention_labels`` or ``None``),
            ``intervention_label`` (string or ``None``),
            ``n_matching`` (int; 0 = no exposure, 1 = clean
            assignment, ≥2 = overlap).
        intervention_labels: labels in order; ``T=i`` ⇔ this label.
        n_assigned: rows with exactly one matching intervention.
        n_unassigned: rows with zero matching interventions.
        n_overlapping: rows with ≥2 matching interventions.
        n_cohort: total cohort size (``n_assigned + n_unassigned +
            n_overlapping``).
        per_arm_matched: number of admissions matched per arm, by
            label. Useful for overlap diagnostics and for sanity-
            checking small-sample arms.
        provenance: concatenation of each intervention's provenance
            dict plus the resolver version.
    """

    df: pd.DataFrame
    intervention_labels: list[str]
    n_assigned: int
    n_unassigned: int
    n_overlapping: int
    n_cohort: int
    per_arm_matched: dict[str, int]
    provenance: dict = field(default_factory=dict)


class InsufficientInterventionsError(ValueError):
    """Raised when fewer than two interventions are passed to the
    assigner. At least two arms are required for a causal contrast —
    this is a hard precondition, not a silent fall-through."""


def assign_treatments(
    backend: _SupportsExecute,
    resolved_interventions: list[ResolvedIntervention],
    cohort_hadm_ids: Iterable[int],
) -> TreatmentAssignment:
    """Compute the per-admission treatment vector for a cohort.

    Args:
        backend: a ``_DuckDBBackend``-shaped object (``execute(sql,
            params)`` returning row tuples).
        resolved_interventions: ordered list; ``T=i`` refers to
            ``resolved_interventions[i].label``. Must have length ≥ 2.
        cohort_hadm_ids: iterable of MIMIC hadm_id values defining
            the population cohort.

    Returns:
        A ``TreatmentAssignment`` whose ``df`` has exactly one row per
        distinct cohort admission that resolved to a valid
        ``subject_id`` via the ``admissions`` table. Admissions not
        present in the ``admissions`` table (shouldn't happen in
        practice — would indicate a cohort-construction bug upstream)
        are silently dropped.
    """
    if len(resolved_interventions) < 2:
        raise InsufficientInterventionsError(
            f"causal treatment assignment requires ≥2 arms; got "
            f"{len(resolved_interventions)}"
        )

    cohort_ids = list(cohort_hadm_ids)
    labels = [ri.label for ri in resolved_interventions]

    if not cohort_ids:
        return TreatmentAssignment(
            df=pd.DataFrame(
                columns=["subject_id", "hadm_id", "T", "intervention_label", "n_matching"]
            ),
            intervention_labels=labels,
            n_assigned=0,
            n_unassigned=0,
            n_overlapping=0,
            n_cohort=0,
            per_arm_matched=dict.fromkeys(labels, 0),
            provenance={
                "resolved_interventions": [ri.provenance for ri in resolved_interventions],
                "cohort_size": 0,
            },
        )

    hadm_placeholders = ",".join(["?"] * len(cohort_ids))

    # Step 1: for each intervention, get the set of matching hadm_ids
    # by running a single SQL query. Done one-at-a-time so each
    # provenance entry keeps its own diagnostic payload; can be
    # parallelised in 8d with Phase 7b's concurrency primitives.
    exposed_per_arm: dict[int, set[int]] = {}
    for idx, ri in enumerate(resolved_interventions):
        predicate = ri.sql_exists_fragment
        if ri.is_control:
            predicate = f"NOT ({predicate})"
        sql = (
            "SELECT DISTINCT a.hadm_id FROM admissions a "
            f"WHERE a.hadm_id IN ({hadm_placeholders}) AND {predicate}"
        )
        params = list(cohort_ids) + list(ri.params)
        rows = backend.execute(sql, params)
        exposed_per_arm[idx] = {int(r[0]) for r in rows}
        logger.debug(
            "assign_treatments: arm=%s matched %d / %d cohort admissions",
            ri.label, len(exposed_per_arm[idx]), len(cohort_ids),
        )

    # Step 2: fetch subject_id for every cohort admission in one round-trip.
    subject_rows = backend.execute(
        f"SELECT subject_id, hadm_id FROM admissions WHERE hadm_id IN ({hadm_placeholders})",
        list(cohort_ids),
    )

    # Step 3: combine into the per-admission record.
    records = []
    for subject_id, hadm_id in subject_rows:
        matching_indices = [
            idx for idx, exposed in exposed_per_arm.items() if int(hadm_id) in exposed
        ]
        n_matching = len(matching_indices)
        if n_matching == 1:
            T = matching_indices[0]
            label = labels[T]
        else:
            T = None
            label = None
        records.append({
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "T": T,
            "intervention_label": label,
            "n_matching": n_matching,
        })

    df = pd.DataFrame.from_records(
        records,
        columns=["subject_id", "hadm_id", "T", "intervention_label", "n_matching"],
    )
    n_assigned = int((df["n_matching"] == 1).sum())
    n_unassigned = int((df["n_matching"] == 0).sum())
    n_overlapping = int((df["n_matching"] >= 2).sum())

    per_arm_matched = {
        labels[idx]: len(exposed) for idx, exposed in exposed_per_arm.items()
    }

    return TreatmentAssignment(
        df=df,
        intervention_labels=labels,
        n_assigned=n_assigned,
        n_unassigned=n_unassigned,
        n_overlapping=n_overlapping,
        n_cohort=len(df),
        per_arm_matched=per_arm_matched,
        provenance={
            "resolved_interventions": [ri.provenance for ri in resolved_interventions],
            "cohort_size": len(cohort_ids),
            "unique_cohort_admissions_resolved": len(df),
        },
    )
