"""Covariate-matrix extraction for the causal pipeline (Phase 8c).

The Neyman–Rubin spec (§1) treats covariates X as a vector of
pre-treatment patient characteristics — age, demographics, comorbidity
load, baseline labs, etc. The estimator in 8d needs these as numeric
columns to fit per-arm outcome models.

Phase 8c ships a minimal but correct covariate builder keyed by a
named ``profile``. Additional profiles (``demographics+labs``,
``full``) wire in existing ``FeatureBuilder`` machinery; they land in
future phases once the estimator is in place and the covariate set
demonstrably helps. For now ``"demographics"`` covers the I3/I5/I6
demo scenarios; the extension surface is straightforward.

Design notes
------------

* Covariates are produced as a ``pandas.DataFrame`` indexed by
  ``hadm_id`` with entirely numeric columns — so ``econml``
  estimators in 8d can consume them without additional encoding.
* Categorical columns (gender, admission_type) are one-hot-encoded.
  The encoding is stable across calls: column names include the
  category value (``gender_M``, ``admission_type_EMERGENCY``) so a
  reviewer can read the matrix.
* Missing demographics are rare in MIMIC but possible. The builder
  raises on missing age (an integer column; nullability would silently
  affect distance metrics in 8h's overlap diagnostics); for gender it
  falls back to a ``gender_unknown`` indicator so the row survives.

The covariate builder is *not* a ``FeatureBuilder`` replacement — it
intentionally ships a narrow set so 8c stays small. When the set
needs to grow, the integration is: (1) add a new profile here, (2)
call into ``src.feature_extraction.feature_builder`` for the extra
columns, (3) append them to the returned DataFrame.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


CovariateProfile = Literal["demographics", "demographics+admission"]


class _SupportsExecute(Protocol):
    def execute(self, sql: str, params: list) -> list[tuple]: ...


class UnknownCovariateProfileError(ValueError):
    """Raised when a profile name isn't recognised. Listed separately
    from generic ``ValueError`` so callers can distinguish
    configuration mistakes from other errors."""


def build_covariate_matrix(
    backend: _SupportsExecute,
    hadm_ids: list[int],
    profile: CovariateProfile = "demographics",
) -> pd.DataFrame:
    """Return a DataFrame keyed by ``hadm_id`` with numeric covariates.

    Args:
        backend: database backend exposing ``execute(sql, params)``.
        hadm_ids: cohort admissions whose covariates are needed.
        profile: ``"demographics"`` → age, gender (one-hot).
                 ``"demographics+admission"`` → adds admission_type
                 one-hot + hospital_los_days.

    Returns:
        DataFrame with ``hadm_id`` column plus numeric covariate
        columns. One row per admission in ``hadm_ids``.
    """
    if not hadm_ids:
        cols = _expected_columns(profile)
        return pd.DataFrame(columns=["hadm_id"] + cols)

    if profile == "demographics":
        return _demographics_only(backend, hadm_ids)
    if profile == "demographics+admission":
        demo = _demographics_only(backend, hadm_ids)
        adm = _admission_features(backend, hadm_ids)
        return demo.merge(adm, on="hadm_id", how="inner")

    raise UnknownCovariateProfileError(
        f"unknown covariate profile {profile!r}; valid: "
        "{'demographics', 'demographics+admission'}"
    )


def _expected_columns(profile: CovariateProfile) -> list[str]:
    """For empty-cohort short-circuit only — avoid building an empty
    DataFrame with no schema info (breaks downstream concat)."""
    if profile == "demographics":
        return ["age", "gender_M", "gender_F", "gender_unknown"]
    return [
        "age", "gender_M", "gender_F", "gender_unknown",
        "admission_type_EMERGENCY", "admission_type_ELECTIVE",
        "admission_type_URGENT", "admission_type_other",
        "hospital_los_days",
    ]


def _demographics_only(
    backend: _SupportsExecute, hadm_ids: list[int]
) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(hadm_ids))
    sql = (
        "SELECT a.hadm_id, p.anchor_age AS age, p.gender "
        "FROM admissions a JOIN patients p ON a.subject_id = p.subject_id "
        f"WHERE a.hadm_id IN ({placeholders})"
    )
    rows = backend.execute(sql, list(hadm_ids))
    df = pd.DataFrame(rows, columns=["hadm_id", "age", "gender"])
    if df["age"].isna().any():
        missing = df.loc[df["age"].isna(), "hadm_id"].tolist()
        raise ValueError(
            f"demographics profile: {len(missing)} admissions have NULL age "
            f"(hadm_id sample: {missing[:5]}); impute upstream or exclude them from the cohort"
        )
    df["age"] = df["age"].astype(int)
    # One-hot gender with explicit columns so schema is stable across
    # cohorts that happen to contain only one gender.
    df["gender_M"] = (df["gender"] == "M").astype(int)
    df["gender_F"] = (df["gender"] == "F").astype(int)
    df["gender_unknown"] = (~df["gender"].isin(["M", "F"])).astype(int)
    return df[["hadm_id", "age", "gender_M", "gender_F", "gender_unknown"]]


def _admission_features(
    backend: _SupportsExecute, hadm_ids: list[int]
) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(hadm_ids))
    sql = (
        "SELECT hadm_id, admission_type, admittime, dischtime "
        f"FROM admissions WHERE hadm_id IN ({placeholders})"
    )
    rows = backend.execute(sql, list(hadm_ids))
    df = pd.DataFrame(rows, columns=["hadm_id", "admission_type", "admittime", "dischtime"])
    df["admission_type_EMERGENCY"] = (df["admission_type"] == "EMERGENCY").astype(int)
    df["admission_type_ELECTIVE"] = (df["admission_type"] == "ELECTIVE").astype(int)
    df["admission_type_URGENT"] = (df["admission_type"] == "URGENT").astype(int)
    df["admission_type_other"] = (~df["admission_type"].isin(
        ["EMERGENCY", "ELECTIVE", "URGENT"]
    )).astype(int)
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
    df["hospital_los_days"] = (
        (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    )
    return df[[
        "hadm_id",
        "admission_type_EMERGENCY", "admission_type_ELECTIVE",
        "admission_type_URGENT", "admission_type_other",
        "hospital_los_days",
    ]]


__all__ = [
    "CovariateProfile",
    "UnknownCovariateProfileError",
    "build_covariate_matrix",
]
