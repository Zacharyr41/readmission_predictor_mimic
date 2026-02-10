"""Tests for SNOMED-based feature extraction."""

import json
import pytest
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdflib import Graph

from src.graph_construction.ontology import initialize_graph
from src.graph_construction.patient_writer import write_patient, write_admission
from src.graph_construction.event_writers import (
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_prescription_event,
    write_diagnosis_event,
)
from src.graph_construction.terminology.snomed_mapper import SnomedMapper
from src.feature_extraction.tabular_features import (
    extract_snomed_diagnosis_features,
    extract_snomed_medication_features,
)


ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


@pytest.fixture
def snomed_mapper(tmp_path: Path) -> SnomedMapper:
    """Create a SnomedMapper with test data."""
    icd_data = {
        "_metadata": {"source": "test"},
        "I630": {"snomed_code": "432504007", "snomed_term": "Cerebral infarction due to thrombosis"},
        "I639": {"snomed_code": "432504007", "snomed_term": "Cerebral infarction, unspecified"},
        "E119": {"snomed_code": "44054006", "snomed_term": "Type 2 diabetes mellitus"},
    }
    drug_data = {
        "_metadata": {"source": "test"},
        "vancomycin": {"snomed_code": "372735009", "snomed_term": "Vancomycin", "rxcui": "11124"},
        "ceftriaxone": {"snomed_code": "372670001", "snomed_term": "Ceftriaxone", "rxcui": "2193"},
    }
    (tmp_path / "icd10cm_to_snomed.json").write_text(json.dumps(icd_data))
    (tmp_path / "drug_to_snomed.json").write_text(json.dumps(drug_data))
    # Empty files for other types (needed by mapper)
    for name in ["labitem_to_snomed.json", "chartitem_to_snomed.json",
                  "organism_to_snomed.json", "comorbidity_to_snomed.json"]:
        (tmp_path / name).write_text(json.dumps({"_metadata": {"source": "test"}}))
    return SnomedMapper(tmp_path)


@pytest.fixture
def snomed_enriched_graph(snomed_mapper: SnomedMapper) -> Graph:
    """Build a graph with SNOMED triples using the test mapper."""
    g = initialize_graph(ONTOLOGY_DIR)

    # Patient 1: 2 diagnoses (both stroke → same SNOMED), 1 prescription
    patient1 = {"subject_id": 1, "gender": "M", "anchor_age": 65}
    p1_uri = write_patient(g, patient1)
    adm1 = {
        "hadm_id": 1, "subject_id": 1,
        "admittime": datetime(2150, 1, 1, 8, 0, 0),
        "dischtime": datetime(2150, 1, 10, 14, 0, 0),
        "admission_type": "EMERGENCY", "discharge_location": "HOME",
        "readmitted_30d": False, "readmitted_60d": False,
    }
    a1_uri = write_admission(g, adm1, p1_uri)
    icu1 = {
        "stay_id": 1, "hadm_id": 1, "subject_id": 1,
        "intime": datetime(2150, 1, 1, 10, 0, 0),
        "outtime": datetime(2150, 1, 4, 10, 0, 0), "los": 3.0,
    }
    icu1_uri = write_icu_stay(g, icu1, a1_uri)
    icu1_days = write_icu_days(g, icu1, icu1_uri)

    write_diagnosis_event(g, {"hadm_id": 1, "seq_num": 1, "icd_code": "I630", "icd_version": 10, "long_title": "Test"}, a1_uri, snomed_mapper=snomed_mapper)
    write_diagnosis_event(g, {"hadm_id": 1, "seq_num": 2, "icd_code": "E119", "icd_version": 10, "long_title": "Test"}, a1_uri, snomed_mapper=snomed_mapper)
    write_prescription_event(g, {
        "hadm_id": 1, "drug": "Vancomycin",
        "starttime": datetime(2150, 1, 1, 10, 0, 0),
        "stoptime": datetime(2150, 1, 3, 10, 0, 0),
        "dose_val_rx": 1000.0, "dose_unit_rx": "mg", "route": "IV",
    }, icu1_uri, icu1_days, snomed_mapper=snomed_mapper)

    # Patient 2: 1 diagnosis, 1 different prescription
    patient2 = {"subject_id": 2, "gender": "F", "anchor_age": 55}
    p2_uri = write_patient(g, patient2)
    adm2 = {
        "hadm_id": 2, "subject_id": 2,
        "admittime": datetime(2150, 2, 1, 6, 0, 0),
        "dischtime": datetime(2150, 2, 10, 12, 0, 0),
        "admission_type": "ELECTIVE", "discharge_location": "SNF",
        "readmitted_30d": True, "readmitted_60d": True,
    }
    a2_uri = write_admission(g, adm2, p2_uri)
    icu2 = {
        "stay_id": 2, "hadm_id": 2, "subject_id": 2,
        "intime": datetime(2150, 2, 2, 8, 0, 0),
        "outtime": datetime(2150, 2, 5, 8, 0, 0), "los": 3.0,
    }
    icu2_uri = write_icu_stay(g, icu2, a2_uri)
    icu2_days = write_icu_days(g, icu2, icu2_uri)

    write_diagnosis_event(g, {"hadm_id": 2, "seq_num": 1, "icd_code": "I639", "icd_version": 10, "long_title": "Test"}, a2_uri, snomed_mapper=snomed_mapper)
    write_prescription_event(g, {
        "hadm_id": 2, "drug": "Ceftriaxone",
        "starttime": datetime(2150, 2, 2, 10, 0, 0),
        "stoptime": datetime(2150, 2, 4, 10, 0, 0),
        "dose_val_rx": 2000.0, "dose_unit_rx": "mg", "route": "IV",
    }, icu2_uri, icu2_days, snomed_mapper=snomed_mapper)

    return g


class TestSnomedDiagnosisFeatures:
    """Tests for extract_snomed_diagnosis_features."""

    def test_returns_expected_columns(self, snomed_enriched_graph: Graph) -> None:
        df = extract_snomed_diagnosis_features(snomed_enriched_graph)
        assert "hadm_id" in df.columns
        assert "num_snomed_mapped" in df.columns
        snomed_cols = [c for c in df.columns if c.startswith("snomed_group_")]
        assert len(snomed_cols) > 0

    def test_correct_counts(self, snomed_enriched_graph: Graph) -> None:
        df = extract_snomed_diagnosis_features(snomed_enriched_graph)
        # hadm_id 1: 2 diagnoses mapped
        row1 = df[df["hadm_id"] == 1].iloc[0]
        assert row1["num_snomed_mapped"] == 2
        # hadm_id 2: 1 diagnosis mapped
        row2 = df[df["hadm_id"] == 2].iloc[0]
        assert row2["num_snomed_mapped"] == 1

    def test_snomed_group_columns_are_binary(self, snomed_enriched_graph: Graph) -> None:
        df = extract_snomed_diagnosis_features(snomed_enriched_graph)
        snomed_cols = [c for c in df.columns if c.startswith("snomed_group_")]
        for col in snomed_cols:
            assert df[col].isin([0, 1]).all()

    def test_empty_on_non_snomed_graph(self) -> None:
        """Backward compatibility: returns empty DataFrame on non-SNOMED graph."""
        g = initialize_graph(ONTOLOGY_DIR)
        # Add a patient/admission without SNOMED
        p_uri = write_patient(g, {"subject_id": 99, "gender": "M", "anchor_age": 50})
        a_uri = write_admission(g, {
            "hadm_id": 99, "subject_id": 99,
            "admittime": datetime(2150, 1, 1, 8, 0, 0),
            "dischtime": datetime(2150, 1, 5, 14, 0, 0),
            "admission_type": "EMERGENCY", "discharge_location": "HOME",
            "readmitted_30d": False, "readmitted_60d": False,
        }, p_uri)
        write_diagnosis_event(g, {"hadm_id": 99, "seq_num": 1, "icd_code": "I630", "icd_version": 10, "long_title": "Test"}, a_uri)
        df = extract_snomed_diagnosis_features(g)
        assert "hadm_id" in df.columns
        assert len(df) == 0


class TestSnomedMedicationFeatures:
    """Tests for extract_snomed_medication_features."""

    def test_returns_expected_columns(self, snomed_enriched_graph: Graph) -> None:
        df = extract_snomed_medication_features(snomed_enriched_graph)
        assert "hadm_id" in df.columns
        assert "num_snomed_med_classes" in df.columns

    def test_correct_med_classes(self, snomed_enriched_graph: Graph) -> None:
        df = extract_snomed_medication_features(snomed_enriched_graph)
        # hadm_id 1: 1 drug (vancomycin → 372735009)
        row1 = df[df["hadm_id"] == 1].iloc[0]
        assert row1["num_snomed_med_classes"] == 1
        assert row1.get("snomed_med_372735009", 0) == 1
        # hadm_id 2: 1 drug (ceftriaxone → 372670001)
        row2 = df[df["hadm_id"] == 2].iloc[0]
        assert row2["num_snomed_med_classes"] == 1
        assert row2.get("snomed_med_372670001", 0) == 1

    def test_empty_on_non_snomed_graph(self) -> None:
        g = initialize_graph(ONTOLOGY_DIR)
        df = extract_snomed_medication_features(g)
        assert "hadm_id" in df.columns
        assert len(df) == 0
