"""Tests for ``src.causal.outcomes`` — outcome extraction (Phase 8c).

Ground truth comes from the fixture admissions in tests/conftest.py:

    hadm_id  subject  gender  age  admittime    dischtime    mortality  dod
    101      1        M       65   2150-01-15   2150-01-20   0          NULL
    102      1        M       65   2150-02-10   2150-02-15   0          NULL
    103      2        F       72   2151-03-01   2151-03-10   0          NULL
    104      3        M       58   2152-05-20   2152-05-25   0          NULL
    105      4        F       45   2150-07-01   2150-07-05   0          NULL
    106      5        M       80   2151-04-10   2151-04-20   1          2151-06-15

Readmissions: subject 1 has 101→102 (disch→admit = 21 days) so 101 is
readmitted within 30d. Everyone else: no second admission, so 0.

ICU stays: 101 → 2.9 days, 103 → 6.2, 106 → 7.25. Others: no ICU.

Creatinine labs (d_labitems.label='Creatinine'): 101 → 1.2, 103 → 0.9,
106 → 1.5. Others: no creatinine lab.

Diagnoses: 101 I639, 102 I634, 103 I630, 104 G409, 105 I251, 106 I639.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.causal.outcomes import (
    OutcomeExtractionError,
    OutcomeExtractor,
    OutcomeRegistry,
    get_default_registry,
)
from src.conversational.models import OutcomeSpec


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


class TestOutcomeRegistry:
    def test_default_registry_contains_expected_keys(self):
        r = get_default_registry()
        assert "readmitted_30d" in r.keys()
        assert "readmitted_60d" in r.keys()
        assert "mortality_inhospital" in r.keys()
        assert "mortality_time_to_event" in r.keys()
        assert "icu_los_days" in r.keys()
        assert "hospital_los_days" in r.keys()
        assert "biomarker_peak" in r.keys()
        assert "biomarker_mean" in r.keys()
        assert "biomarker_min" in r.keys()
        assert "diagnosis_within_horizon" in r.keys()

    def test_unknown_key_raises(self):
        r = get_default_registry()
        with pytest.raises(OutcomeExtractionError, match="no outcome extractor"):
            r.get("does_not_exist")

    def test_missing_required_params_raises(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(
            name="peak_creatinine",
            outcome_type="continuous",
            extractor_key="biomarker_peak",
            # missing biomarker_label
        )
        with pytest.raises(OutcomeExtractionError, match="biomarker_label"):
            r.extract(spec, duckdb_backend, [101])

    def test_type_mismatch_raises(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(
            name="peak_creatinine",
            outcome_type="binary",  # WRONG — biomarker_peak is continuous
            extractor_key="biomarker_peak",
            extractor_params={"biomarker_label": "Creatinine"},
        )
        with pytest.raises(OutcomeExtractionError, match="type .*mismatch"):
            r.extract(spec, duckdb_backend, [101])

    def test_register_duplicate_key_raises(self):
        r = OutcomeRegistry()
        r.register(OutcomeExtractor(
            key="x", outcome_type="binary",
            extract_fn=lambda *a: pd.DataFrame({"hadm_id": [], "value": []}),
        ))
        with pytest.raises(ValueError, match="already registered"):
            r.register(OutcomeExtractor(
                key="x", outcome_type="binary",
                extract_fn=lambda *a: pd.DataFrame(),
            ))


# ---------------------------------------------------------------------------
# Readmitted
# ---------------------------------------------------------------------------


class TestReadmittedOutcomes:
    def test_readmitted_30d_flags_only_subject_1_admission_101(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(name="r30", outcome_type="binary", extractor_key="readmitted_30d")
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        assert by_hadm[101] == 1
        for h in (102, 103, 104, 105, 106):
            assert by_hadm[h] == 0

    def test_readmitted_60d_matches_30d_in_this_fixture(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(name="r60", outcome_type="binary", extractor_key="readmitted_60d")
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        # Same result since 21 days < 60 and no other cross-30-to-60 readmits.
        assert by_hadm[101] == 1
        for h in (102, 103, 104, 105, 106):
            assert by_hadm[h] == 0

    def test_empty_cohort_returns_empty_frame(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(name="r30", outcome_type="binary", extractor_key="readmitted_30d")
        df = r.extract(spec, duckdb_backend, [])
        assert list(df.columns) == ["hadm_id", "value"]
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Mortality
# ---------------------------------------------------------------------------


class TestMortalityOutcomes:
    def test_inhospital_mortality_from_hospital_expire_flag(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(name="m", outcome_type="binary", extractor_key="mortality_inhospital")
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        assert by_hadm[106] == 1
        for h in (101, 102, 103, 104, 105):
            assert by_hadm[h] == 0

    def test_time_to_event_horizon_90_finds_patient_5_death(self, duckdb_backend):
        """Patient 5 dies 2151-06-15; admitted 2151-04-10 → ~66 days.
        With horizon=90, event=1, time≈66."""
        r = get_default_registry()
        spec = OutcomeSpec(
            name="mort_90d",
            outcome_type="time_to_event",
            extractor_key="mortality_time_to_event",
            censoring_clock="admission",
            censoring_horizon_days=90,
        )
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id").to_dict("index")
        # Admission 106 — died 66 days after admit (within 90-day horizon)
        assert by_hadm[106]["event"] == 1
        assert 65 <= by_hadm[106]["time"] <= 67
        # Everyone else — censored at horizon.
        for h in (101, 102, 103, 104, 105):
            assert by_hadm[h]["event"] == 0
            assert by_hadm[h]["time"] == 90.0

    def test_time_to_event_horizon_30_censors_patient_5(self, duckdb_backend):
        """With horizon=30, patient 5's death at day 66 falls outside
        the window → event=0, time=30 (censored at horizon)."""
        r = get_default_registry()
        spec = OutcomeSpec(
            name="mort_30d",
            outcome_type="time_to_event",
            extractor_key="mortality_time_to_event",
            censoring_horizon_days=30,
        )
        df = r.extract(spec, duckdb_backend, [106])
        row = df.iloc[0]
        assert row["event"] == 0
        assert row["time"] == 30.0

    def test_time_to_event_requires_horizon(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(
            name="bad",
            outcome_type="time_to_event",
            extractor_key="mortality_time_to_event",
            # no censoring_horizon_days
        )
        with pytest.raises(OutcomeExtractionError, match="censoring_horizon_days"):
            r.extract(spec, duckdb_backend, [101])


# ---------------------------------------------------------------------------
# LOS outcomes
# ---------------------------------------------------------------------------


class TestLOSOutcomes:
    def test_icu_los_days(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(name="icu_los", outcome_type="continuous", extractor_key="icu_los_days")
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        assert abs(by_hadm[101] - 2.9) < 0.01
        assert abs(by_hadm[103] - 6.2) < 0.01
        assert abs(by_hadm[106] - 7.25) < 0.01
        # No ICU: zero (not NaN).
        for h in (102, 104, 105):
            assert by_hadm[h] == 0.0

    def test_hospital_los_days(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(name="los", outcome_type="continuous", extractor_key="hospital_los_days")
        df = r.extract(spec, duckdb_backend, [101, 103, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        # 101: 2150-01-15 08:00 → 2150-01-20 14:00 = 5.25 days
        assert abs(by_hadm[101] - 5.25) < 0.05
        # 103: 2151-03-01 06:00 → 2151-03-10 16:00 = ~9.42 days
        assert abs(by_hadm[103] - 9.42) < 0.05
        # 106: 2151-04-10 12:00 → 2151-04-20 08:00 = ~9.83 days
        assert abs(by_hadm[106] - 9.83) < 0.05


# ---------------------------------------------------------------------------
# Biomarker outcomes
# ---------------------------------------------------------------------------


class TestBiomarkerOutcomes:
    def test_peak_creatinine(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(
            name="peak_cr",
            outcome_type="continuous",
            extractor_key="biomarker_peak",
            extractor_params={"biomarker_label": "Creatinine"},
        )
        df = r.extract(spec, duckdb_backend, [101, 103, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        assert by_hadm[101] == 1.2
        assert by_hadm[103] == 0.9
        assert by_hadm[106] == 1.5

    def test_admission_without_creatinine_is_nan(self, duckdb_backend):
        """Admissions 102, 104, 105 have no creatinine labs → NaN, not 0.
        Downstream estimator decides to drop or impute."""
        r = get_default_registry()
        spec = OutcomeSpec(
            name="peak_cr",
            outcome_type="continuous",
            extractor_key="biomarker_peak",
            extractor_params={"biomarker_label": "Creatinine"},
        )
        df = r.extract(spec, duckdb_backend, [102, 104, 105])
        assert df["value"].isna().all()

    def test_biomarker_mean(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(
            name="mean_cr",
            outcome_type="continuous",
            extractor_key="biomarker_mean",
            extractor_params={"biomarker_label": "Creatinine"},
        )
        df = r.extract(spec, duckdb_backend, [101])
        # Only one measurement (value 1.2).
        assert df.iloc[0]["value"] == 1.2


# ---------------------------------------------------------------------------
# Diagnosis-based outcomes
# ---------------------------------------------------------------------------


class TestDiagnosisOutcomes:
    def test_index_admission_diagnosis_match_horizon_zero(self, duckdb_backend):
        """horizon_days=0 → diagnoses from the index admission itself."""
        r = get_default_registry()
        spec = OutcomeSpec(
            name="had_stroke",
            outcome_type="binary",
            extractor_key="diagnosis_within_horizon",
            extractor_params={"icd_prefixes": ["I63"], "horizon_days": 0},
        )
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        # I63x on 101, 102, 103, 106; others (G409, I251) don't match.
        assert by_hadm[101] == 1
        assert by_hadm[102] == 1
        assert by_hadm[103] == 1
        assert by_hadm[104] == 0
        assert by_hadm[105] == 0
        assert by_hadm[106] == 1

    def test_subsequent_admission_within_horizon(self, duckdb_backend):
        """horizon_days>0 → any subsequent admission's diagnoses."""
        r = get_default_registry()
        spec = OutcomeSpec(
            name="stroke_recurrence_30d",
            outcome_type="binary",
            extractor_key="diagnosis_within_horizon",
            extractor_params={"icd_prefixes": ["I63"], "horizon_days": 30},
        )
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        by_hadm = df.set_index("hadm_id")["value"].to_dict()
        # 101 discharged 2150-01-20 → next adm (102) at 2150-02-10 = 21d later,
        # carries I634 → stroke recurrence = 1.
        assert by_hadm[101] == 1
        # Everyone else: no subsequent admission within 30d of discharge.
        for h in (102, 103, 104, 105, 106):
            assert by_hadm[h] == 0

    def test_multiple_prefixes_or(self, duckdb_backend):
        """Bleeding example: intracranial (I60-I62) OR GI (K92)."""
        r = get_default_registry()
        spec = OutcomeSpec(
            name="bleeding",
            outcome_type="binary",
            extractor_key="diagnosis_within_horizon",
            extractor_params={"icd_prefixes": ["I60", "I61", "I62", "K92"], "horizon_days": 0},
        )
        df = r.extract(spec, duckdb_backend, [101, 102, 103, 104, 105, 106])
        # No bleeding in the fixture — all 0.
        assert df["value"].sum() == 0

    def test_requires_icd_prefixes(self, duckdb_backend):
        r = get_default_registry()
        spec = OutcomeSpec(
            name="bad",
            outcome_type="binary",
            extractor_key="diagnosis_within_horizon",
            extractor_params={"horizon_days": 30},
        )
        with pytest.raises(OutcomeExtractionError, match="icd_prefixes"):
            r.extract(spec, duckdb_backend, [101])
