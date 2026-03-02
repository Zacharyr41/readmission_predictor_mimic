"""Tests for TBI cohort selection and WLST label derivation."""

import pytest
import pandas as pd

from src.wlst.cohort import (
    create_wlst_labels,
    generate_cohort_summary,
    select_tbi_cohort,
)


class TestSelectTbiCohort:
    def test_returns_dataframe(self, wlst_duckdb):
        df = select_tbi_cohort(wlst_duckdb)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, wlst_duckdb):
        df = select_tbi_cohort(wlst_duckdb)
        expected = {"subject_id", "hadm_id", "stay_id", "intime", "outtime",
                    "first_careunit", "gcs_total", "gcs_eye", "gcs_verbal", "gcs_motor"}
        assert expected.issubset(set(df.columns))

    def test_gcs_threshold_filtering(self, wlst_duckdb):
        """All patients in cohort should have GCS <= 8."""
        df = select_tbi_cohort(wlst_duckdb, gcs_threshold=8)
        if len(df) > 0:
            assert df["gcs_total"].max() <= 8
            assert df["gcs_total"].min() > 0

    def test_strict_gcs_threshold(self, wlst_duckdb):
        """Lowering threshold should not increase cohort size."""
        df_8 = select_tbi_cohort(wlst_duckdb, gcs_threshold=8)
        df_5 = select_tbi_cohort(wlst_duckdb, gcs_threshold=5)
        assert len(df_5) <= len(df_8)

    def test_patients_limit(self, wlst_duckdb):
        df = select_tbi_cohort(wlst_duckdb, patients_limit=2)
        assert len(df) <= 2

    def test_only_tbi_diagnoses(self, wlst_duckdb):
        """All patients should have S06.x TBI diagnosis."""
        df = select_tbi_cohort(wlst_duckdb)
        # Verify by checking hadm_ids against diagnoses
        if len(df) > 0:
            hadm_ids = df["hadm_id"].tolist()
            dx = wlst_duckdb.execute(
                f"SELECT DISTINCT hadm_id FROM diagnoses_icd "
                f"WHERE icd_version = 10 AND icd_code LIKE 'S06%' "
                f"AND hadm_id IN ({','.join(str(h) for h in hadm_ids)})"
            ).fetchdf()
            assert set(hadm_ids).issubset(set(dx["hadm_id"].tolist()))

    def test_icu_type_filtering(self, wlst_duckdb):
        """All patients should be in neuro/trauma ICU."""
        df = select_tbi_cohort(wlst_duckdb)
        if len(df) > 0:
            valid_icus = {
                "Neuro Stepdown",
                "Neuro Surgical Intensive Care Unit (Neuro SICU)",
                "Trauma SICU (TSICU)",
            }
            assert set(df["first_careunit"]).issubset(valid_icus)


class TestCreateWlstLabels:
    def test_returns_dataframe(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        assert isinstance(labels, pd.DataFrame)

    def test_label_column_exists(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        assert "wlst_label" in labels.columns
        assert "outcome_category" in labels.columns

    def test_label_is_binary(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        assert set(labels["wlst_label"].unique()).issubset({0, 1})

    def test_hospice_discharge_is_wlst(self, wlst_duckdb):
        """Patient discharged to HOSPICE should be labeled WLST=1."""
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        hospice = labels[labels["discharge_location"] == "HOSPICE"]
        if len(hospice) > 0:
            assert (hospice["wlst_label"] == 1).all()

    def test_code_status_change_is_wlst(self, wlst_duckdb):
        """Patient with code status change from Full should be labeled WLST=1."""
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        with_code_change = labels[labels["code_status_value"].notna()]
        if len(with_code_change) > 0:
            assert (with_code_change["wlst_label"] == 1).all()

    def test_preserves_all_cohort_patients(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        assert set(cohort["stay_id"]) == set(labels["stay_id"])


class TestGenerateCohortSummary:
    def test_returns_markdown(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        summary = generate_cohort_summary(labels)
        assert isinstance(summary, str)
        assert "# WLST Cohort Summary" in summary

    def test_empty_cohort(self):
        summary = generate_cohort_summary(pd.DataFrame())
        assert "No patients" in summary

    def test_includes_label_distribution(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)
        summary = generate_cohort_summary(labels)
        assert "WLST positive" in summary or "label=1" in summary
