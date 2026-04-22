"""Fixtures for ``src.causal`` tests.

Extends ``tests/conftest.py::synthetic_duckdb_with_events`` with the
additional prescriptions / procedures_icd rows needed to exercise all
four intervention-resolution paths (RxNorm, SNOMED, ICD-10-PCS, LOINC)
plus the mutual-exclusivity edge cases in
``src.causal.treatment_assignment``.

The layout is intentional — each admission in the base fixture is
given a distinct exposure profile so a single assignment run can
observe: clean single-arm matches (n=1), no exposure (n=0), and
overlap (n≥2).

Phase 8d additions at the bottom: ``make_synthetic_cohort_frame`` +
``seeded_cohort_frame`` fixture — an in-memory ``CohortFrame`` with a
programmed ATE so estimator + bootstrap tests can assert correctness
without dragging in the full 8c cohort-assembly pipeline. The existing
6-row ``synthetic_duckdb_for_causal`` is untouched so 8a/8b/8c tests
keep their hand-tuned exposure profile.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.causal.cohort import CohortFrame
from src.causal.treatment_assignment import TreatmentAssignment


@pytest.fixture
def synthetic_duckdb_for_causal(
    synthetic_duckdb_with_events: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyConnection:
    """Extend the base synthetic fixture with causal-inference test data.

    Exposure profile by hadm_id:

      * 101 (subject 1) — Alteplase (tPA)  [+ Vancomycin already present]
      * 102 (subject 1) — Warfarin
      * 103 (subject 2) — (Ceftriaxone already present, no new drug)
      * 104 (subject 3) — NO drug exposure (tests n_matching=0)
      * 105 (subject 4) — Alteplase AND Warfarin (tests n_matching=2 overlap)
      * 106 (subject 5) — (Unchanged)

    Procedures added: 101, 102 carry an ICD-10-PCS thrombolytic code
    (3E03317) so the ICD-10-PCS resolver path has hits.
    """
    conn = synthetic_duckdb_with_events

    # Augment prescriptions. The base fixture inserted 2 rows (vanc→101,
    # ceftriaxone→103). We add: tPA→101, Warfarin→102, tPA+Warfarin→105.
    conn.execute("""
        INSERT INTO prescriptions VALUES
        (1, 101, '2150-01-15 09:00:00', '2150-01-15 11:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (1, 102, '2150-02-10 12:00:00', '2150-02-15 08:00:00', 'Warfarin', 5.0, 'mg', 'PO'),
        (4, 105, '2150-07-01 10:00:00', '2150-07-01 10:30:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (4, 105, '2150-07-02 08:00:00', '2150-07-05 08:00:00', 'Warfarin', 5.0, 'mg', 'PO')
    """)

    # procedures_icd is not created by synthetic_duckdb_with_events — create + populate.
    conn.execute("""
        CREATE TABLE procedures_icd (
            subject_id INTEGER,
            hadm_id INTEGER,
            seq_num INTEGER,
            icd_code VARCHAR,
            icd_version INTEGER
        )
    """)
    # 3E03317 = "Introduction of Other Thrombolytic into Peripheral Vein,
    # Percutaneous Approach" (canonical ICD-10-PCS for IV tPA).
    conn.execute("""
        INSERT INTO procedures_icd VALUES
        (1, 101, 1, '3E03317', 10),
        (1, 102, 1, '3E03317', 10),
        (4, 105, 1, '3E03317', 10)
    """)

    return conn


class DuckDBAdapter:
    """Thin ``backend`` adapter exposing just ``execute(sql, params)``.

    The real ``_DuckDBBackend`` in ``src.conversational.extractor`` is
    wired to a path + connection-per-thread. For causal tests we only
    need the ``execute(sql, params) -> list[tuple]`` contract, so this
    wraps the raw DuckDB connection from the fixture.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: list) -> list[tuple]:
        return self._conn.execute(sql, params).fetchall()


@pytest.fixture
def duckdb_backend(synthetic_duckdb_for_causal: duckdb.DuckDBPyConnection) -> DuckDBAdapter:
    return DuckDBAdapter(synthetic_duckdb_for_causal)


# ---------------------------------------------------------------------------
# Phase 8d — synthetic CohortFrame with a programmed ATE.
# ---------------------------------------------------------------------------


def make_synthetic_cohort_frame(
    n_per_arm: int = 100,
    n_arms: int = 2,
    ate: float = 2.0,
    seed: int = 0,
    outcome_names: list[str] | None = None,
) -> CohortFrame:
    """Build a ``CohortFrame`` with known ATE.

    Data-generating process: age + gender confound BOTH treatment
    assignment AND outcome. A T/S/X-learner that correctly conditions
    on age and gender should recover the true ATE; a marginal mean
    difference will be biased by the confounding. This is the
    correctness harness for Phase 8d's estimators.

    Args:
        n_per_arm: rows per arm (stratified bootstrap tests assume
            this is stable).
        n_arms: C — number of intervention arms.
        ate: true treatment effect per arm-step (arm c adds c*ate to
            the outcome before noise). For n_arms=2 this is the
            standard ATE; for n_arms>2 the τ_{c,c'} between arms c
            and c' is (c-c')*ate.
        seed: numpy RNG seed.
        outcome_names: if provided, adds one outcome column per name
            with the same programmed effect (used by the per-outcome
            breakdown test). Default: ``["Y"]``.

    Returns:
        A frozen ``CohortFrame`` with integer ``T`` in [0, n_arms),
        demographic covariates, and one outcome per ``outcome_names``.
    """
    outcome_names = outcome_names or ["Y"]
    rng = np.random.default_rng(seed)
    total = n_arms * n_per_arm

    # Stratified arm assignment so bootstrap tests can assert size
    # preservation deterministically.
    T = np.repeat(np.arange(n_arms), n_per_arm)
    # Shuffle so row order isn't a predictor.
    perm = rng.permutation(total)
    T = T[perm]

    # Confounders correlated with T (older + male likelier in higher arms).
    # The correlation is programmed in so unadjusted mean diffs are biased.
    age = rng.integers(30, 90, size=total) + (5.0 * T).astype(int)
    gender_M = (rng.random(size=total) < (0.35 + 0.15 * T / max(n_arms - 1, 1))).astype(int)
    gender_F = 1 - gender_M
    gender_unknown = np.zeros(total, dtype=int)

    hadm_id = np.arange(1_000_000, 1_000_000 + total, dtype=int)
    subject_id = np.arange(5_000_000, 5_000_000 + total, dtype=int)
    intervention_labels = [f"arm{c}" for c in range(n_arms)]

    df = pd.DataFrame({
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "T": T,
        "intervention_label": [intervention_labels[t] for t in T],
        "n_matching": np.ones(total, dtype=int),
        "age": age,
        "gender_M": gender_M,
        "gender_F": gender_F,
        "gender_unknown": gender_unknown,
    })
    # Outcome: Y = 0.05*age + 1.5*gender_M + ate*T + N(0, 1). True ATE
    # between arm c and arm 0 is c*ate (under ignorability given X).
    for name in outcome_names:
        df[name] = (
            0.05 * age
            + 1.5 * gender_M
            + ate * T
            + rng.normal(0.0, 1.0, size=total)
        )

    per_arm_matched = {label: int((df["T"] == c).sum()) for c, label in enumerate(intervention_labels)}
    assignment = TreatmentAssignment(
        df=df[["subject_id", "hadm_id", "T", "intervention_label", "n_matching"]].copy(),
        intervention_labels=intervention_labels,
        n_assigned=total,
        n_unassigned=0,
        n_overlapping=0,
        n_cohort=total,
        per_arm_matched=per_arm_matched,
        provenance={"source": "synthetic — make_synthetic_cohort_frame", "seed": seed},
    )

    return CohortFrame(
        df=df,
        covariate_columns=["age", "gender_M", "gender_F", "gender_unknown"],
        outcome_columns=list(outcome_names),
        treatment_column="T",
        intervention_labels=intervention_labels,
        treatment_assignment=assignment,
        provenance={"source": "synthetic", "seed": seed, "true_ate": ate},
    )


@pytest.fixture
def seeded_cohort_frame() -> CohortFrame:
    """Default synthetic cohort for estimator tests: C=2, n=1, ATE=2.0."""
    return make_synthetic_cohort_frame(n_per_arm=150, n_arms=2, ate=2.0, seed=0)
