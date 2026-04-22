"""Tests for ``src.causal.covariates`` — covariate matrix (Phase 8c)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.causal.covariates import (
    UnknownCovariateProfileError,
    build_covariate_matrix,
)


class TestDemographicsProfile:
    def test_produces_expected_columns(self, duckdb_backend):
        df = build_covariate_matrix(duckdb_backend, [101, 103, 106])
        assert set(df.columns) == {
            "hadm_id", "age", "gender_M", "gender_F", "gender_unknown"
        }

    def test_values_match_fixture(self, duckdb_backend):
        df = build_covariate_matrix(duckdb_backend, [101, 103, 106])
        by_hadm = df.set_index("hadm_id").to_dict("index")
        # 101 → subject 1 (M, 65)
        assert by_hadm[101]["age"] == 65
        assert by_hadm[101]["gender_M"] == 1
        assert by_hadm[101]["gender_F"] == 0
        # 103 → subject 2 (F, 72)
        assert by_hadm[103]["age"] == 72
        assert by_hadm[103]["gender_F"] == 1
        assert by_hadm[103]["gender_M"] == 0
        # 106 → subject 5 (M, 80)
        assert by_hadm[106]["age"] == 80
        assert by_hadm[106]["gender_M"] == 1
        # gender_unknown is always populated (as 0 when known)
        assert by_hadm[106]["gender_unknown"] == 0

    def test_empty_cohort_returns_empty_typed_frame(self, duckdb_backend):
        df = build_covariate_matrix(duckdb_backend, [])
        assert set(df.columns) >= {"hadm_id", "age", "gender_M", "gender_F", "gender_unknown"}
        assert len(df) == 0

    def test_columns_entirely_numeric(self, duckdb_backend):
        """Estimator in 8d requires numeric columns — confirm no strings
        leak through (e.g. raw gender)."""
        df = build_covariate_matrix(duckdb_backend, [101, 103, 106])
        for col in df.columns:
            if col == "hadm_id":
                continue
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


class TestAdmissionProfile:
    def test_adds_admission_type_and_los(self, duckdb_backend):
        df = build_covariate_matrix(duckdb_backend, [101, 103, 106], profile="demographics+admission")
        expected = {
            "hadm_id", "age", "gender_M", "gender_F", "gender_unknown",
            "admission_type_EMERGENCY", "admission_type_ELECTIVE",
            "admission_type_URGENT", "admission_type_other",
            "hospital_los_days",
        }
        assert set(df.columns) == expected

    def test_admission_type_values_match_fixture(self, duckdb_backend):
        df = build_covariate_matrix(
            duckdb_backend, [101, 103, 105], profile="demographics+admission",
        )
        by_hadm = df.set_index("hadm_id").to_dict("index")
        # 101 EMERGENCY, 103 ELECTIVE, 105 URGENT
        assert by_hadm[101]["admission_type_EMERGENCY"] == 1
        assert by_hadm[101]["admission_type_ELECTIVE"] == 0
        assert by_hadm[103]["admission_type_ELECTIVE"] == 1
        assert by_hadm[105]["admission_type_URGENT"] == 1

    def test_los_days_computed_correctly(self, duckdb_backend):
        df = build_covariate_matrix(duckdb_backend, [101], profile="demographics+admission")
        # 2150-01-15 08:00 → 2150-01-20 14:00 = 5.25 days
        assert abs(df.iloc[0]["hospital_los_days"] - 5.25) < 0.05


class TestErrors:
    def test_unknown_profile_raises(self, duckdb_backend):
        with pytest.raises(UnknownCovariateProfileError):
            build_covariate_matrix(duckdb_backend, [101], profile="nonsense")  # type: ignore[arg-type]
