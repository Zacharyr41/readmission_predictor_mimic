"""Fixtures for ``src.similarity`` tests.

Two primary layers of fixtures:

1. **Unit-level** (no DB) — hand-built feature dicts / DataFrames +
   event-stream lists + bucket dicts. These isolate the scoring math
   (contextual distance functions, Jaccard + decay, bucket assignment)
   from the feature-extraction pipeline.

2. **End-to-end** — reuses ``synthetic_duckdb_with_events`` from
   ``tests/conftest.py`` via pytest conftest chaining. The
   ``similarity_backend`` fixture is a thin ``.execute()``-only
   adapter identical in shape to
   ``tests/test_causal/conftest.py::DuckDBAdapter``.

All synthetic data is programmed so the expected ranking is
deterministic — tests assert ordering + per-group contribution sign,
not absolute score floats.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Unit fixtures — pre-built feature rows + event sets.
# ---------------------------------------------------------------------------


def _feature_row(**overrides) -> dict:
    """Baseline feature dict: low-risk patient. Overrides per-test."""
    base = {
        "hadm_id": 0,
        "subject_id": 0,
        # demographics
        "age": 60,
        "gender_M": 0, "gender_F": 1, "gender_unknown": 0,
        "admission_type_EMERGENCY": 1,
        "admission_type_ELECTIVE": 0,
        "admission_type_URGENT": 0,
        "admission_type_other": 0,
        # comorbidity burden (Charlson)
        "charlson_index": 2,
        "charlson_myocardial_infarction": 0,
        "charlson_chf": 0,
        "charlson_pvd": 0,
        "charlson_cvd": 0,
        "charlson_dementia": 0,
        "charlson_copd": 0,
        "charlson_rheumatoid": 0,
        "charlson_pud": 0,
        "charlson_mild_liver": 0,
        "charlson_diabetes": 0,
        "charlson_diabetes_complications": 0,
        "charlson_hemiplegia": 0,
        "charlson_renal": 0,
        "charlson_malignancy": 0,
        "charlson_moderate_severe_liver": 0,
        "charlson_metastatic_tumor": 0,
        "charlson_aids_hiv": 0,
        # comorbidity set (SNOMED groups — presence flags)
        "snomed_group_I63": 0,   # stroke
        "snomed_group_I48": 0,   # afib
        "snomed_group_N18": 0,   # CKD
        "snomed_group_E11": 0,   # diabetes
        "snomed_group_J44": 0,   # COPD
        # severity
        "creatinine_max": 1.0,
        "sodium_mean": 140.0,
        "platelet_min": 220.0,
        "icu_los_hours": 12.0,
        # social
        "language_barrier": 0,
        "is_neuro_service": 0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def anchor_features() -> dict:
    """Anchor: 68yo F, afib + CKD, moderate Charlson."""
    return _feature_row(
        hadm_id=1001, subject_id=100,
        age=68, gender_F=1, gender_M=0,
        charlson_index=5, charlson_chf=1, charlson_renal=1,
        snomed_group_I48=1, snomed_group_N18=1,
        creatinine_max=2.1, icu_los_hours=36.0,
    )


@pytest.fixture
def candidate_features_df(anchor_features) -> pd.DataFrame:
    """Four candidates with a known similarity ordering relative to anchor.

    Under the default group weights (``comorbidity_burden=0.35`` +
    ``comorbidity_set=0.25`` ⇒ 60% weight on comorbidity), shared
    baseline comorbidities dominate single-admission severity. Expected
    rank:

      2001 — near-identical: demographics + comorbidities + similar severity
      2003 — chronic-profile match + acute severity gap (same demographics,
             same Charlson / SNOMED burden, abnormal labs)
      2002 — thinner comorbidity overlap (shared afib only, missing CKD
             + renal), closer labs than 2003 but insufficient to outrank
             it at default weights
      2004 — dissimilar across every group
    """
    rows = [
        _feature_row(
            hadm_id=2001, subject_id=200,
            age=70, gender_F=1, gender_M=0,
            charlson_index=5, charlson_chf=1, charlson_renal=1,
            snomed_group_I48=1, snomed_group_N18=1,
            creatinine_max=2.0, icu_los_hours=40.0,
        ),
        _feature_row(
            hadm_id=2002, subject_id=201,
            age=75, gender_F=1, gender_M=0,
            charlson_index=3, charlson_chf=1,
            snomed_group_I48=1,
            creatinine_max=1.3, icu_los_hours=24.0,
        ),
        _feature_row(
            hadm_id=2003, subject_id=202,
            age=68, gender_F=1, gender_M=0,
            charlson_index=5, charlson_chf=1, charlson_renal=1,
            snomed_group_I48=1, snomed_group_N18=1,
            creatinine_max=5.5, sodium_mean=125.0, platelet_min=45.0,
            icu_los_hours=168.0,
        ),
        _feature_row(
            hadm_id=2004, subject_id=203,
            age=32, gender_F=0, gender_M=1,
            charlson_index=0,
            creatinine_max=0.8, icu_los_hours=8.0,
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def anchor_buckets() -> dict[str, set[str]]:
    """Anchor temporal profile: 3 bucket days.

    d0 — antibiotic + vasopressor + lactate abnormal
    d1 — antibiotic + steroid
    d2 — antibiotic
    """
    return {
        "icu_day_0": {"snomed_drug:abx_broad", "snomed_drug:vasopressor", "snomed_lab:lactate_abn"},
        "icu_day_1": {"snomed_drug:abx_broad", "snomed_drug:steroid"},
        "icu_day_2": {"snomed_drug:abx_broad"},
    }


@pytest.fixture
def candidate_buckets_by_hadm() -> dict[int, dict[str, set[str]]]:
    """Candidates with known bucket-overlap profile vs the anchor fixture."""
    return {
        2001: {  # near-identical
            "icu_day_0": {"snomed_drug:abx_broad", "snomed_drug:vasopressor", "snomed_lab:lactate_abn"},
            "icu_day_1": {"snomed_drug:abx_broad", "snomed_drug:steroid"},
            "icu_day_2": {"snomed_drug:abx_broad"},
        },
        2002: {  # partial overlap
            "icu_day_0": {"snomed_drug:abx_broad", "snomed_drug:vasopressor"},
            "icu_day_1": {"snomed_drug:abx_broad"},
            "icu_day_2": set(),
        },
        2003: {  # different trajectory
            "icu_day_0": {"snomed_drug:diuretic", "snomed_drug:insulin"},
            "icu_day_1": {"snomed_drug:diuretic"},
        },
    }


# ---------------------------------------------------------------------------
# End-to-end fixture — thin DuckDB adapter over the top-level synthetic DB.
# ---------------------------------------------------------------------------


class _ExecuteAdapter:
    """Minimal ``.execute(sql, params) -> list[tuple]`` adapter. Mirrors
    tests/test_causal/conftest.py::DuckDBAdapter — same contract."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: list) -> list[tuple]:
        return self._conn.execute(sql, params).fetchall()


@pytest.fixture
def similarity_backend(synthetic_duckdb_with_events: duckdb.DuckDBPyConnection):
    """Adapter over the top-level 6-admission synthetic fixture. Suitable
    for a cohort-small end-to-end smoke; not sized for statistical
    claims."""
    return _ExecuteAdapter(synthetic_duckdb_with_events)
