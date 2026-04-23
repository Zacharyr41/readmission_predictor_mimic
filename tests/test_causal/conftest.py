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


@pytest.fixture
def synthetic_duckdb_for_causal_binary(
    synthetic_duckdb_for_causal: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyConnection:
    """Augment the base causal fixture with enough admissions for binary LR.

    The 6-admission base fixture has ``readmitted_30d`` collapsing to
    all-False in the no-tPA arm, which makes sklearn's LogisticRegression
    refuse to fit (ValueError: only one class in data). This fixture adds
    12 more admissions (hadm_ids 201–212) across 6 new subjects with a
    deliberately mixed readmission pattern so both arms have both classes:

      * tPA arm (rxnorm 8410 present):
          201 (subj 10) → readmit (follow-up 202 within 30d)
          203 (subj 11) → readmit (follow-up 204 within 30d)
          205 (subj 12) → no readmit
          206 (subj 13) → no readmit
      * no-tPA arm:
          207 (subj 14) → readmit (follow-up 208 within 30d)
          209 (subj 15) → readmit (follow-up 210 within 30d)
          211 (subj 16) → no readmit
          212 (subj 17) → no readmit

    Selection cohort for tests: hadm_ids 201, 203, 205, 206 (tPA arm) +
    207, 209, 211, 212 (no-tPA arm) = 8 admissions with 4-4-4-4 class
    balance. Follow-ups 202, 204, 208, 210 feed readmitted_30d but are
    not in the cohort (the outcome extractor looks them up by subject).
    """
    conn = synthetic_duckdb_for_causal

    # New subjects. patients schema: (subject_id, gender, anchor_age,
    # anchor_year, dod) per tests/conftest.py:40-46.
    conn.execute("""
        INSERT INTO patients VALUES
        (10, 'M', 65, 2150, NULL),
        (11, 'F', 70, 2150, NULL),
        (12, 'M', 55, 2151, NULL),
        (13, 'F', 60, 2151, NULL),
        (14, 'M', 72, 2150, NULL),
        (15, 'F', 68, 2150, NULL),
        (16, 'M', 58, 2151, NULL),
        (17, 'F', 62, 2151, NULL)
    """)

    # New admissions. Pairs within 30 days trigger readmitted_30d=True on
    # the first of each pair.
    conn.execute("""
        INSERT INTO admissions VALUES
        (201, 10, '2150-03-01 08:00:00', '2150-03-05 14:00:00', 'EMERGENCY', 'HOME', 0),
        (202, 10, '2150-03-20 09:00:00', '2150-03-24 10:00:00', 'EMERGENCY', 'HOME', 0),
        (203, 11, '2150-04-01 10:00:00', '2150-04-04 12:00:00', 'EMERGENCY', 'HOME', 0),
        (204, 11, '2150-04-20 11:00:00', '2150-04-23 13:00:00', 'EMERGENCY', 'HOME', 0),
        (205, 12, '2151-05-01 08:00:00', '2151-05-05 14:00:00', 'EMERGENCY', 'HOME', 0),
        (206, 13, '2151-06-01 10:00:00', '2151-06-04 12:00:00', 'EMERGENCY', 'HOME', 0),
        (207, 14, '2150-07-01 08:00:00', '2150-07-05 14:00:00', 'EMERGENCY', 'HOME', 0),
        (208, 14, '2150-07-20 09:00:00', '2150-07-24 10:00:00', 'EMERGENCY', 'HOME', 0),
        (209, 15, '2150-08-01 10:00:00', '2150-08-04 12:00:00', 'EMERGENCY', 'HOME', 0),
        (210, 15, '2150-08-20 11:00:00', '2150-08-23 13:00:00', 'EMERGENCY', 'HOME', 0),
        (211, 16, '2151-09-01 08:00:00', '2151-09-05 14:00:00', 'EMERGENCY', 'HOME', 0),
        (212, 17, '2151-10-01 10:00:00', '2151-10-04 12:00:00', 'EMERGENCY', 'HOME', 0)
    """)

    # Prescriptions: Alteplase for the tPA arm (201, 203, 205, 206 and
    # follow-ups 202, 204), nothing for the no-tPA arm.
    conn.execute("""
        INSERT INTO prescriptions VALUES
        (10, 201, '2150-03-01 09:00:00', '2150-03-01 11:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (10, 202, '2150-03-20 10:00:00', '2150-03-20 12:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (11, 203, '2150-04-01 11:00:00', '2150-04-01 13:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (11, 204, '2150-04-20 12:00:00', '2150-04-20 14:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (12, 205, '2151-05-01 09:00:00', '2151-05-01 11:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (13, 206, '2151-06-01 11:00:00', '2151-06-01 13:00:00', 'Alteplase', 90.0, 'mg', 'IV')
    """)

    return conn


@pytest.fixture
def duckdb_backend_binary(
    synthetic_duckdb_for_causal_binary: duckdb.DuckDBPyConnection,
) -> DuckDBAdapter:
    return DuckDBAdapter(synthetic_duckdb_for_causal_binary)


# ---------------------------------------------------------------------------
# Phase 8d — synthetic CohortFrame with a programmed ATE.
# ---------------------------------------------------------------------------


def make_synthetic_cohort_frame(
    n_per_arm: int = 100,
    n_arms: int = 2,
    ate: float = 2.0,
    seed: int = 0,
    outcome_names: list[str] | None = None,
    binary_outcome: bool = False,
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
    # When ``binary_outcome`` is True, pass the continuous score through
    # a logistic (centered so each arm has a usable class balance) and
    # sample 0/1 — gives LR enough per-arm variation to converge under
    # stratified bootstrap.
    for name in outcome_names:
        latent = (
            0.05 * age
            + 1.5 * gender_M
            + ate * T
            + rng.normal(0.0, 1.0, size=total)
        )
        if binary_outcome:
            p = 1.0 / (1.0 + np.exp(-(latent - latent.mean())))
            df[name] = (rng.random(total) < p).astype(int)
        else:
            df[name] = latent

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
