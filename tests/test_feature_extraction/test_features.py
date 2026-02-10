"""Test suite for feature extraction from RDF graphs.

Tests feature extraction functions with a synthetic graph containing:
- Patient 1 (age 70, male): 2 admissions (101, 102)
- Patient 2 (age 55, female): 1 admission (201)
- Patient 3 (age 80, male): 2 admissions (301, 302)
"""

import pytest
import pandas as pd
from datetime import datetime
from rdflib import Graph

from src.graph_construction.ontology import initialize_graph, MIMIC_NS
from src.graph_construction.patient_writer import write_patient, write_admission
from src.graph_construction.event_writers import (
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_clinical_sign_event,
    write_prescription_event,
    write_diagnosis_event,
)
from src.graph_construction.temporal.allen_relations import compute_allen_relations
from src.feature_extraction import (
    extract_demographics,
    extract_stay_features,
    extract_lab_summary,
    extract_vital_summary,
    extract_medication_features,
    extract_diagnosis_features,
    extract_temporal_features,
    extract_graph_structure_features,
    build_feature_matrix,
)

from pathlib import Path


ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


@pytest.fixture
def synthetic_feature_graph() -> Graph:
    """Graph with known structure for feature extraction tests.

    Structure:
    - Patient 1 (age 70, male): 2 admissions
      - Admission 101: readmitted_30d=True, 1 ICU stay, 5 creatinine values, 3 HR vitals, 1 prescription (3 days), 2 diagnoses
      - Admission 102: readmitted_30d=False, 1 ICU stay, 3 biomarkers, 2 vitals
    - Patient 2 (age 55, female): 1 admission
      - Admission 201: readmitted_30d=False, 1 ICU stay, 4 biomarkers, 2 vitals, 0 prescriptions
    - Patient 3 (age 80, male): 2 admissions
      - Admission 301: readmitted_30d=True, 1 ICU stay, sparse events
      - Admission 302: readmitted_30d=False, 1 ICU stay, sparse events
    """
    g = initialize_graph(ONTOLOGY_DIR)

    # ==================== Patient 1 (age 70, male) ====================
    patient1 = {"subject_id": 1, "gender": "M", "anchor_age": 70}
    patient1_uri = write_patient(g, patient1)

    # Admission 101 (readmitted_30d=True)
    admission101 = {
        "hadm_id": 101,
        "subject_id": 1,
        "admittime": datetime(2150, 1, 1, 8, 0, 0),
        "dischtime": datetime(2150, 1, 5, 14, 0, 0),
        "admission_type": "EMERGENCY",
        "discharge_location": "HOME",
        "readmitted_30d": True,
        "readmitted_60d": True,
    }
    admission101_uri = write_admission(g, admission101, patient1_uri)

    icu_stay101 = {
        "stay_id": 1001,
        "hadm_id": 101,
        "subject_id": 1,
        "intime": datetime(2150, 1, 1, 10, 0, 0),
        "outtime": datetime(2150, 1, 4, 10, 0, 0),  # 3 days
        "los": 3.0,
    }
    icu_stay101_uri = write_icu_stay(g, icu_stay101, admission101_uri)
    icu_day_metadata101 = write_icu_days(g, icu_stay101, icu_stay101_uri)

    # 5 creatinine values: 1.0, 1.2, 1.5, 1.3, 1.1 -> mean = 1.22
    creatinine_values = [1.0, 1.2, 1.5, 1.3, 1.1]
    creatinine_times = [
        datetime(2150, 1, 1, 12, 0, 0),
        datetime(2150, 1, 1, 18, 0, 0),
        datetime(2150, 1, 2, 6, 0, 0),
        datetime(2150, 1, 2, 12, 0, 0),
        datetime(2150, 1, 2, 18, 0, 0),
    ]
    for i, val in enumerate(creatinine_values):
        biomarker = {
            "labevent_id": 10010 + i,
            "stay_id": 1001,
            "itemid": 50912,
            "charttime": creatinine_times[i],
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": val,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        write_biomarker_event(g, biomarker, icu_stay101_uri, icu_day_metadata101)

    # 3 HR vitals: 72, 85, 78 -> mean = 78.33
    hr_values = [72, 85, 78]
    hr_times = [
        datetime(2150, 1, 1, 10, 0, 0),
        datetime(2150, 1, 1, 18, 0, 0),
        datetime(2150, 1, 2, 10, 0, 0),
    ]
    for i, val in enumerate(hr_values):
        vital = {
            "stay_id": 1001,
            "itemid": 220045,
            "charttime": hr_times[i],
            "label": "Heart Rate",
            "category": "Routine Vital Signs",
            "valuenum": float(val),
        }
        write_clinical_sign_event(g, vital, icu_stay101_uri, icu_day_metadata101)

    # 1 prescription (3 days: Jan 1-4)
    rx101 = {
        "hadm_id": 101,
        "stay_id": 1001,
        "drug": "Vancomycin",
        "starttime": datetime(2150, 1, 1, 12, 0, 0),
        "stoptime": datetime(2150, 1, 4, 12, 0, 0),
        "dose_val_rx": 1000.0,
        "dose_unit_rx": "mg",
        "route": "IV",
    }
    write_prescription_event(g, rx101, icu_stay101_uri, icu_day_metadata101)

    # 2 diagnoses (I63.0, E11.9)
    dx101_1 = {
        "hadm_id": 101,
        "seq_num": 1,
        "icd_code": "I63.0",
        "icd_version": 10,
        "long_title": "Cerebral infarction due to thrombosis of precerebral arteries",
    }
    dx101_2 = {
        "hadm_id": 101,
        "seq_num": 2,
        "icd_code": "E11.9",
        "icd_version": 10,
        "long_title": "Type 2 diabetes mellitus without complications",
    }
    write_diagnosis_event(g, dx101_1, admission101_uri)
    write_diagnosis_event(g, dx101_2, admission101_uri)

    # Compute Allen relations for ICU stay 101
    compute_allen_relations(g, icu_stay101_uri)

    # Admission 102 (readmitted_30d=False)
    admission102 = {
        "hadm_id": 102,
        "subject_id": 1,
        "admittime": datetime(2150, 1, 20, 10, 0, 0),
        "dischtime": datetime(2150, 1, 25, 12, 0, 0),
        "admission_type": "URGENT",
        "discharge_location": "SNF",
        "readmitted_30d": False,
        "readmitted_60d": False,
    }
    admission102_uri = write_admission(g, admission102, patient1_uri)

    icu_stay102 = {
        "stay_id": 1002,
        "hadm_id": 102,
        "subject_id": 1,
        "intime": datetime(2150, 1, 20, 12, 0, 0),
        "outtime": datetime(2150, 1, 23, 12, 0, 0),  # 3 days
        "los": 3.0,
    }
    icu_stay102_uri = write_icu_stay(g, icu_stay102, admission102_uri)
    icu_day_metadata102 = write_icu_days(g, icu_stay102, icu_stay102_uri)

    # 3 biomarkers
    for i in range(3):
        biomarker = {
            "labevent_id": 10020 + i,
            "stay_id": 1002,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 21, 8 + i * 4, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 0.9 + i * 0.1,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        write_biomarker_event(g, biomarker, icu_stay102_uri, icu_day_metadata102)

    # 2 vitals
    for i in range(2):
        vital = {
            "stay_id": 1002,
            "itemid": 220045,
            "charttime": datetime(2150, 1, 21, 10 + i * 6, 0, 0),
            "label": "Heart Rate",
            "category": "Routine Vital Signs",
            "valuenum": 75.0 + i * 5,
        }
        write_clinical_sign_event(g, vital, icu_stay102_uri, icu_day_metadata102)

    compute_allen_relations(g, icu_stay102_uri)

    # ==================== Patient 2 (age 55, female) ====================
    patient2 = {"subject_id": 2, "gender": "F", "anchor_age": 55}
    patient2_uri = write_patient(g, patient2)

    # Admission 201 (readmitted_30d=False)
    admission201 = {
        "hadm_id": 201,
        "subject_id": 2,
        "admittime": datetime(2150, 2, 1, 6, 0, 0),
        "dischtime": datetime(2150, 2, 8, 12, 0, 0),
        "admission_type": "ELECTIVE",
        "discharge_location": "HOME",
        "readmitted_30d": False,
        "readmitted_60d": True,
    }
    admission201_uri = write_admission(g, admission201, patient2_uri)

    icu_stay201 = {
        "stay_id": 2001,
        "hadm_id": 201,
        "subject_id": 2,
        "intime": datetime(2150, 2, 1, 8, 0, 0),
        "outtime": datetime(2150, 2, 5, 8, 0, 0),  # 4 days
        "los": 4.0,
    }
    icu_stay201_uri = write_icu_stay(g, icu_stay201, admission201_uri)
    icu_day_metadata201 = write_icu_days(g, icu_stay201, icu_stay201_uri)

    # 4 biomarkers
    sodium_times = [
        datetime(2150, 2, 1, 10, 0, 0),
        datetime(2150, 2, 1, 16, 0, 0),
        datetime(2150, 2, 1, 22, 0, 0),
        datetime(2150, 2, 2, 4, 0, 0),
    ]
    for i in range(4):
        biomarker = {
            "labevent_id": 20010 + i,
            "stay_id": 2001,
            "itemid": 50971,
            "charttime": sodium_times[i],
            "label": "Sodium",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 138.0 + i,
            "valueuom": "mEq/L",
            "ref_range_lower": 136.0,
            "ref_range_upper": 145.0,
        }
        write_biomarker_event(g, biomarker, icu_stay201_uri, icu_day_metadata201)

    # 2 vitals
    for i in range(2):
        vital = {
            "stay_id": 2001,
            "itemid": 220179,
            "charttime": datetime(2150, 2, 2, 8 + i * 8, 0, 0),
            "label": "Non Invasive Blood Pressure systolic",
            "category": "Routine Vital Signs",
            "valuenum": 120.0 + i * 5,
        }
        write_clinical_sign_event(g, vital, icu_stay201_uri, icu_day_metadata201)

    # 1 diagnosis
    dx201 = {
        "hadm_id": 201,
        "seq_num": 1,
        "icd_code": "J18.9",
        "icd_version": 10,
        "long_title": "Pneumonia, unspecified organism",
    }
    write_diagnosis_event(g, dx201, admission201_uri)

    compute_allen_relations(g, icu_stay201_uri)

    # ==================== Patient 3 (age 80, male) ====================
    patient3 = {"subject_id": 3, "gender": "M", "anchor_age": 80}
    patient3_uri = write_patient(g, patient3)

    # Admission 301 (readmitted_30d=True)
    admission301 = {
        "hadm_id": 301,
        "subject_id": 3,
        "admittime": datetime(2150, 3, 1, 8, 0, 0),
        "dischtime": datetime(2150, 3, 5, 14, 0, 0),
        "admission_type": "EMERGENCY",
        "discharge_location": "HOME",
        "readmitted_30d": True,
        "readmitted_60d": True,
    }
    admission301_uri = write_admission(g, admission301, patient3_uri)

    icu_stay301 = {
        "stay_id": 3001,
        "hadm_id": 301,
        "subject_id": 3,
        "intime": datetime(2150, 3, 1, 10, 0, 0),
        "outtime": datetime(2150, 3, 3, 10, 0, 0),  # 2 days
        "los": 2.0,
    }
    icu_stay301_uri = write_icu_stay(g, icu_stay301, admission301_uri)
    icu_day_metadata301 = write_icu_days(g, icu_stay301, icu_stay301_uri)

    # 1 biomarker
    biomarker301 = {
        "labevent_id": 30010,
        "stay_id": 3001,
        "itemid": 50912,
        "charttime": datetime(2150, 3, 1, 12, 0, 0),
        "label": "Creatinine",
        "fluid": "Blood",
        "category": "Chemistry",
        "valuenum": 1.8,
        "valueuom": "mg/dL",
        "ref_range_lower": 0.7,
        "ref_range_upper": 1.3,
    }
    write_biomarker_event(g, biomarker301, icu_stay301_uri, icu_day_metadata301)

    # 1 vital
    vital301 = {
        "stay_id": 3001,
        "itemid": 220045,
        "charttime": datetime(2150, 3, 1, 14, 0, 0),
        "label": "Heart Rate",
        "category": "Routine Vital Signs",
        "valuenum": 92.0,
    }
    write_clinical_sign_event(g, vital301, icu_stay301_uri, icu_day_metadata301)

    dx301 = {
        "hadm_id": 301,
        "seq_num": 1,
        "icd_code": "N17.9",
        "icd_version": 10,
        "long_title": "Acute kidney failure, unspecified",
    }
    write_diagnosis_event(g, dx301, admission301_uri)

    compute_allen_relations(g, icu_stay301_uri)

    # Admission 302 (readmitted_30d=False)
    admission302 = {
        "hadm_id": 302,
        "subject_id": 3,
        "admittime": datetime(2150, 3, 20, 10, 0, 0),
        "dischtime": datetime(2150, 3, 25, 12, 0, 0),
        "admission_type": "URGENT",
        "discharge_location": "HOME",
        "readmitted_30d": False,
        "readmitted_60d": False,
    }
    admission302_uri = write_admission(g, admission302, patient3_uri)

    icu_stay302 = {
        "stay_id": 3002,
        "hadm_id": 302,
        "subject_id": 3,
        "intime": datetime(2150, 3, 20, 12, 0, 0),
        "outtime": datetime(2150, 3, 22, 12, 0, 0),  # 2 days
        "los": 2.0,
    }
    icu_stay302_uri = write_icu_stay(g, icu_stay302, admission302_uri)
    icu_day_metadata302 = write_icu_days(g, icu_stay302, icu_stay302_uri)

    # 2 biomarkers
    for i in range(2):
        biomarker = {
            "labevent_id": 30020 + i,
            "stay_id": 3002,
            "itemid": 50912,
            "charttime": datetime(2150, 3, 20, 14 + i * 6, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.4 + i * 0.1,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        write_biomarker_event(g, biomarker, icu_stay302_uri, icu_day_metadata302)

    # 1 vital
    vital302 = {
        "stay_id": 3002,
        "itemid": 220045,
        "charttime": datetime(2150, 3, 21, 10, 0, 0),
        "label": "Heart Rate",
        "category": "Routine Vital Signs",
        "valuenum": 88.0,
    }
    write_clinical_sign_event(g, vital302, icu_stay302_uri, icu_day_metadata302)

    dx302 = {
        "hadm_id": 302,
        "seq_num": 1,
        "icd_code": "N17.0",
        "icd_version": 10,
        "long_title": "Acute kidney failure with tubular necrosis",
    }
    write_diagnosis_event(g, dx302, admission302_uri)

    compute_allen_relations(g, icu_stay302_uri)

    return g


# ==================== Test Cases ====================


class TestExtractDemographics:
    """(a) test_extract_demographics"""

    def test_returns_dataframe_with_expected_columns(self, synthetic_feature_graph):
        """Returns DataFrame with hadm_id, age, gender_M, gender_F."""
        df = extract_demographics(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns
        assert "age" in df.columns
        assert "gender_M" in df.columns
        assert "gender_F" in df.columns

    def test_patient1_admission101_demographics(self, synthetic_feature_graph):
        """Patient 1 admission 101: age=70, gender_M=1, gender_F=0."""
        df = extract_demographics(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        assert row["age"] == 70
        assert row["gender_M"] == 1
        assert row["gender_F"] == 0

    def test_patient2_admission201_demographics(self, synthetic_feature_graph):
        """Patient 2 admission 201: age=55, gender_M=0, gender_F=1."""
        df = extract_demographics(synthetic_feature_graph)

        row = df[df["hadm_id"] == 201].iloc[0]
        assert row["age"] == 55
        assert row["gender_M"] == 0
        assert row["gender_F"] == 1


class TestExtractStayFeatures:
    """(b) test_extract_stay_features"""

    def test_returns_dataframe_with_expected_columns(self, synthetic_feature_graph):
        """Returns DataFrame with hadm_id, icu_los_hours, num_icu_days."""
        df = extract_stay_features(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns
        assert "icu_los_hours" in df.columns
        assert "num_icu_days" in df.columns

    def test_los_calculation_admission101(self, synthetic_feature_graph):
        """Admission 101: 3 days LOS = 72 hours."""
        df = extract_stay_features(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        assert row["icu_los_hours"] == pytest.approx(72.0, rel=0.01)

    def test_icu_days_count_admission201(self, synthetic_feature_graph):
        """Admission 201: 4 days LOS spans 5 calendar days."""
        df = extract_stay_features(synthetic_feature_graph)

        row = df[df["hadm_id"] == 201].iloc[0]
        # 4 day LOS = 96 hours
        assert row["icu_los_hours"] == pytest.approx(96.0, rel=0.01)


class TestExtractLabSummary:
    """(c) test_extract_lab_summary"""

    def test_returns_dataframe_with_aggregates(self, synthetic_feature_graph):
        """Returns DataFrame with per-biomarker aggregates."""
        df = extract_lab_summary(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns

    def test_creatinine_mean_admission101(self, synthetic_feature_graph):
        """Patient 1 admission 101: creatinine_mean = 1.22."""
        df = extract_lab_summary(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        # Mean of [1.0, 1.2, 1.5, 1.3, 1.1] = 6.1 / 5 = 1.22
        assert "Creatinine_mean" in df.columns
        assert row["Creatinine_mean"] == pytest.approx(1.22, rel=0.01)

    def test_lab_count_admission101(self, synthetic_feature_graph):
        """Admission 101 has 5 creatinine values."""
        df = extract_lab_summary(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        assert row["Creatinine_count"] == 5


class TestExtractVitalSummary:
    """(d) test_extract_vital_summary"""

    def test_returns_dataframe_with_aggregates(self, synthetic_feature_graph):
        """Returns DataFrame with per-vital aggregates."""
        df = extract_vital_summary(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns

    def test_hr_mean_admission101(self, synthetic_feature_graph):
        """Patient 1 admission 101: Heart Rate mean = 78.33."""
        df = extract_vital_summary(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        # Mean of [72, 85, 78] = 235 / 3 = 78.33
        assert "Heart Rate_mean" in df.columns
        assert row["Heart Rate_mean"] == pytest.approx(78.33, rel=0.01)


class TestExtractMedicationFeatures:
    """(e) test_extract_medication_features"""

    def test_returns_dataframe_with_expected_columns(self, synthetic_feature_graph):
        """Returns DataFrame with hadm_id, num_distinct_meds, total_prescription_days, has_prescription."""
        df = extract_medication_features(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns
        assert "num_distinct_meds" in df.columns
        assert "total_prescription_days" in df.columns
        assert "has_prescription" in df.columns

    def test_admission101_has_prescription(self, synthetic_feature_graph):
        """Admission 101: has_prescription=1, total_prescription_days=3."""
        df = extract_medication_features(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        assert row["has_prescription"] == 1
        assert row["total_prescription_days"] == pytest.approx(3.0, rel=0.01)

    def test_admission201_no_prescription(self, synthetic_feature_graph):
        """Admission 201: has_prescription=0."""
        df = extract_medication_features(synthetic_feature_graph)

        row = df[df["hadm_id"] == 201].iloc[0]
        assert row["has_prescription"] == 0
        assert row["total_prescription_days"] == 0


class TestExtractDiagnosisFeatures:
    """Test diagnosis feature extraction."""

    def test_returns_dataframe_with_counts(self, synthetic_feature_graph):
        """Returns DataFrame with diagnosis counts."""
        df = extract_diagnosis_features(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns
        assert "num_diagnoses" in df.columns

    def test_admission101_has_two_diagnoses(self, synthetic_feature_graph):
        """Admission 101 has 2 diagnoses."""
        df = extract_diagnosis_features(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        assert row["num_diagnoses"] == 2


class TestExtractTemporalFeatures:
    """(f) test_extract_temporal_features"""

    def test_returns_dataframe_with_expected_columns(self, synthetic_feature_graph):
        """Returns DataFrame with temporal relation counts."""
        df = extract_temporal_features(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns
        assert "total_temporal_edges" in df.columns

    def test_admission101_has_temporal_edges(self, synthetic_feature_graph):
        """Admission 101 should have temporal edges from Allen relations."""
        df = extract_temporal_features(synthetic_feature_graph)

        row = df[df["hadm_id"] == 101].iloc[0]
        # With 5 biomarkers + 3 vitals + 1 prescription = 9 events, there should be many before relations
        assert row["total_temporal_edges"] > 0


class TestExtractGraphStructureFeatures:
    """(g) test_extract_graph_structure_features"""

    def test_returns_dataframe_with_expected_columns(self, synthetic_feature_graph):
        """Returns DataFrame with graph structure metrics."""
        df = extract_graph_structure_features(synthetic_feature_graph)

        assert isinstance(df, pd.DataFrame)
        assert "hadm_id" in df.columns
        assert "patient_subgraph_nodes" in df.columns
        assert "patient_subgraph_edges" in df.columns
        assert "patient_subgraph_density" in df.columns
        assert "mean_node_degree" in df.columns
        assert "max_node_degree" in df.columns

    def test_admission_has_positive_node_count(self, synthetic_feature_graph):
        """Each admission should have positive node count."""
        df = extract_graph_structure_features(synthetic_feature_graph)

        for _, row in df.iterrows():
            assert row["patient_subgraph_nodes"] > 0
            assert row["patient_subgraph_edges"] > 0


class TestBuildFeatureMatrix:
    """(h) test_build_feature_matrix_combines_all"""

    def test_has_5_rows(self, synthetic_feature_graph):
        """Feature matrix has 5 rows (one per admission)."""
        df = build_feature_matrix(synthetic_feature_graph)

        assert len(df) == 5

    def test_has_more_than_20_columns(self, synthetic_feature_graph):
        """Feature matrix has >20 columns."""
        df = build_feature_matrix(synthetic_feature_graph)

        assert len(df.columns) > 20

    def test_includes_readmission_labels(self, synthetic_feature_graph):
        """Feature matrix includes readmitted_30d and readmitted_60d."""
        df = build_feature_matrix(synthetic_feature_graph)

        assert "readmitted_30d" in df.columns
        assert "readmitted_60d" in df.columns


class TestFeatureMatrixNoNullLabels:
    """(i) test_feature_matrix_no_nulls_in_labels"""

    def test_labels_have_no_nan(self, synthetic_feature_graph):
        """Label columns have no NaN values."""
        df = build_feature_matrix(synthetic_feature_graph)

        assert df["readmitted_30d"].notna().all()
        assert df["readmitted_60d"].notna().all()


class TestNoFutureLeakage:
    """(j) test_feature_matrix_no_future_leakage"""

    def test_no_discharge_diagnosis_feature(self, synthetic_feature_graph):
        """No discharge_diagnosis feature exists (documenting potential leakage)."""
        df = build_feature_matrix(synthetic_feature_graph)

        # We don't use discharge diagnosis text as a feature
        # The diagnosis features we use are from diagnoses_icd which is coded
        # at discharge - we document this potential leakage in the module
        leakage_columns = [col for col in df.columns if "discharge" in col.lower() and "diagnosis" in col.lower()]
        assert len(leakage_columns) == 0, f"Found potential leakage columns: {leakage_columns}"
