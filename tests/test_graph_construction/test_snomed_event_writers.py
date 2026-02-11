"""Tests for SNOMED-CT enrichment in event writers."""

import json
import pytest
from datetime import datetime
from pathlib import Path

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, XSD

from src.graph_construction.ontology import initialize_graph, MIMIC_NS, SNOMED_NS
from src.graph_construction.patient_writer import write_patient, write_admission
from src.graph_construction.event_writers import (
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_clinical_sign_event,
    write_microbiology_event,
    write_prescription_event,
    write_diagnosis_event,
    write_comorbidity,
)
from src.graph_construction.terminology.snomed_mapper import SnomedMapper


ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


@pytest.fixture
def synthetic_mapper(tmp_path: Path) -> SnomedMapper:
    """Create a SnomedMapper with minimal test data."""
    icd_data = {
        "_metadata": {"source": "test"},
        "I630": {"snomed_code": "432504007", "snomed_term": "Cerebral infarction due to thrombosis"},
    }
    lab_data = {
        "_metadata": {"source": "test"},
        "50912": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement", "loinc": "2160-0"},
    }
    chart_data = {
        "_metadata": {"source": "test"},
        "220045": {"snomed_code": "364075005", "snomed_term": "Heart rate", "loinc": "8867-4"},
    }
    drug_data = {
        "_metadata": {"source": "test"},
        "vancomycin": {"snomed_code": "372735009", "snomed_term": "Vancomycin", "rxcui": "11124"},
    }
    org_data = {
        "_metadata": {"source": "test"},
        "STAPHYLOCOCCUS AUREUS": {"snomed_code": "3092008", "snomed_term": "Staphylococcus aureus"},
    }
    comorbidity_data = {
        "_metadata": {"source": "test"},
        "diabetes": {"snomed_code": "73211009", "snomed_term": "Diabetes mellitus"},
    }

    loinc_data = {
        "_metadata": {"source": "test"},
        "2160-0": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement"},
    }

    for name, data in [
        ("icd10cm_to_snomed.json", icd_data),
        ("labitem_to_snomed.json", lab_data),
        ("chartitem_to_snomed.json", chart_data),
        ("drug_to_snomed.json", drug_data),
        ("organism_to_snomed.json", org_data),
        ("comorbidity_to_snomed.json", comorbidity_data),
        ("loinc_to_snomed.json", loinc_data),
    ]:
        (tmp_path / name).write_text(json.dumps(data))

    return SnomedMapper(tmp_path)


@pytest.fixture
def graph_with_patient():
    """Graph with ontology, patient, admission, ICU stay and ICU days."""
    g = initialize_graph(ONTOLOGY_DIR)
    patient_data = {"subject_id": 100, "gender": "M", "anchor_age": 65}
    patient_uri = write_patient(g, patient_data)
    admission_data = {
        "hadm_id": 200, "subject_id": 100,
        "admittime": datetime(2150, 1, 1, 8, 0, 0),
        "dischtime": datetime(2150, 1, 10, 14, 0, 0),
        "admission_type": "EMERGENCY", "discharge_location": "HOME",
        "readmitted_30d": False, "readmitted_60d": False,
    }
    admission_uri = write_admission(g, admission_data, patient_uri)
    icu_stay_data = {
        "stay_id": 300, "hadm_id": 200, "subject_id": 100,
        "intime": datetime(2150, 1, 1, 8, 0, 0),
        "outtime": datetime(2150, 1, 4, 14, 0, 0), "los": 3.25,
    }
    icu_stay_uri = write_icu_stay(g, icu_stay_data, admission_uri)
    icu_day_metadata = write_icu_days(g, icu_stay_data, icu_stay_uri)
    return g, patient_uri, admission_uri, icu_stay_uri, icu_day_metadata


def _has_snomed_triples(graph: Graph, event_uri, expected_code: str) -> bool:
    """Check that SNOMED triples are present for an event."""
    has_code = (event_uri, MIMIC_NS.hasSnomedCode, Literal(expected_code, datatype=XSD.string)) in graph
    has_concept = (event_uri, MIMIC_NS.hasSnomedConcept, SNOMED_NS[expected_code]) in graph
    has_term = any(graph.triples((event_uri, MIMIC_NS.hasSnomedTerm, None)))
    return has_code and has_concept and has_term


def _count_snomed_triples(graph: Graph, event_uri) -> int:
    """Count SNOMED-related triples for an event."""
    count = 0
    count += len(list(graph.triples((event_uri, MIMIC_NS.hasSnomedCode, None))))
    count += len(list(graph.triples((event_uri, MIMIC_NS.hasSnomedTerm, None))))
    count += len(list(graph.triples((event_uri, MIMIC_NS.hasSnomedConcept, None))))
    return count


class TestDiagnosisSNOMED:
    """Tests for SNOMED enrichment on DiagnosisEvent."""

    def test_snomed_triples_present_with_mapper(self, graph_with_patient, synthetic_mapper):
        g, patient_uri, admission_uri, _, _ = graph_with_patient
        dx_data = {"hadm_id": 200, "seq_num": 1, "icd_code": "I630", "icd_version": 10, "long_title": "Test"}
        event_uri = write_diagnosis_event(g, dx_data, admission_uri, snomed_mapper=synthetic_mapper)
        assert _has_snomed_triples(g, event_uri, "432504007")

    def test_no_snomed_triples_without_mapper(self, graph_with_patient):
        g, _, admission_uri, _, _ = graph_with_patient
        dx_data = {"hadm_id": 200, "seq_num": 1, "icd_code": "I630", "icd_version": 10, "long_title": "Test"}
        event_uri = write_diagnosis_event(g, dx_data, admission_uri)
        assert _count_snomed_triples(g, event_uri) == 0

    def test_sparql_snomed_code(self, graph_with_patient, synthetic_mapper):
        g, _, admission_uri, _, _ = graph_with_patient
        dx_data = {"hadm_id": 200, "seq_num": 1, "icd_code": "I630", "icd_version": 10, "long_title": "Test"}
        write_diagnosis_event(g, dx_data, admission_uri, snomed_mapper=synthetic_mapper)
        result = bool(g.query(
            'ASK { ?event mimic:hasSnomedCode ?code . FILTER(?code = "432504007") }'
        ))
        assert result

    def test_snomed_concept_uri(self, graph_with_patient, synthetic_mapper):
        g, _, admission_uri, _, _ = graph_with_patient
        dx_data = {"hadm_id": 200, "seq_num": 1, "icd_code": "I630", "icd_version": 10, "long_title": "Test"}
        event_uri = write_diagnosis_event(g, dx_data, admission_uri, snomed_mapper=synthetic_mapper)
        expected_uri = SNOMED_NS["432504007"]
        assert (event_uri, MIMIC_NS.hasSnomedConcept, expected_uri) in g


class TestBiomarkerSNOMED:
    """Tests for SNOMED enrichment on BioMarkerEvent."""

    def test_snomed_triples_present(self, graph_with_patient, synthetic_mapper):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        lab_data = {
            "labevent_id": 1, "stay_id": 300, "itemid": 50912,
            "charttime": datetime(2150, 1, 2, 6, 0, 0), "label": "Creatinine",
            "fluid": "Blood", "category": "Chemistry", "valuenum": 1.2,
            "valueuom": "mg/dL", "ref_range_lower": 0.7, "ref_range_upper": 1.3,
        }
        event_uri = write_biomarker_event(g, lab_data, icu_stay_uri, icu_day_metadata, snomed_mapper=synthetic_mapper)
        assert _has_snomed_triples(g, event_uri, "70901006")

    def test_no_snomed_without_mapper(self, graph_with_patient):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        lab_data = {
            "labevent_id": 1, "stay_id": 300, "itemid": 50912,
            "charttime": datetime(2150, 1, 2, 6, 0, 0), "label": "Creatinine",
            "fluid": "Blood", "category": "Chemistry", "valuenum": 1.2,
            "valueuom": "mg/dL", "ref_range_lower": 0.7, "ref_range_upper": 1.3,
        }
        event_uri = write_biomarker_event(g, lab_data, icu_stay_uri, icu_day_metadata)
        assert _count_snomed_triples(g, event_uri) == 0


class TestClinicalSignSNOMED:
    """Tests for SNOMED enrichment on ClinicalSignEvent."""

    def test_snomed_triples_present(self, graph_with_patient, synthetic_mapper):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        vital_data = {
            "stay_id": 300, "itemid": 220045,
            "charttime": datetime(2150, 1, 1, 12, 0, 0),
            "label": "Heart Rate", "category": "Routine Vital Signs", "valuenum": 78.0,
        }
        event_uri = write_clinical_sign_event(g, vital_data, icu_stay_uri, icu_day_metadata, snomed_mapper=synthetic_mapper)
        assert _has_snomed_triples(g, event_uri, "364075005")

    def test_no_snomed_without_mapper(self, graph_with_patient):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        vital_data = {
            "stay_id": 300, "itemid": 220045,
            "charttime": datetime(2150, 1, 1, 12, 0, 0),
            "label": "Heart Rate", "category": "Routine Vital Signs", "valuenum": 78.0,
        }
        event_uri = write_clinical_sign_event(g, vital_data, icu_stay_uri, icu_day_metadata)
        assert _count_snomed_triples(g, event_uri) == 0


class TestPrescriptionSNOMED:
    """Tests for SNOMED enrichment on PrescriptionEvent."""

    def test_snomed_triples_present(self, graph_with_patient, synthetic_mapper):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        rx_data = {
            "hadm_id": 200, "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 10, 0, 0),
            "stoptime": datetime(2150, 1, 3, 10, 0, 0),
            "dose_val_rx": 1000.0, "dose_unit_rx": "mg", "route": "IV",
        }
        event_uri = write_prescription_event(g, rx_data, icu_stay_uri, icu_day_metadata, snomed_mapper=synthetic_mapper)
        assert _has_snomed_triples(g, event_uri, "372735009")

    def test_no_snomed_without_mapper(self, graph_with_patient):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        rx_data = {
            "hadm_id": 200, "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 10, 0, 0),
            "stoptime": datetime(2150, 1, 3, 10, 0, 0),
            "dose_val_rx": 1000.0, "dose_unit_rx": "mg", "route": "IV",
        }
        event_uri = write_prescription_event(g, rx_data, icu_stay_uri, icu_day_metadata)
        assert _count_snomed_triples(g, event_uri) == 0


class TestMicrobiologySNOMED:
    """Tests for SNOMED enrichment on MicrobiologyEvent."""

    def test_snomed_triples_present(self, graph_with_patient, synthetic_mapper):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        micro_data = {
            "microevent_id": 1, "stay_id": 300,
            "charttime": datetime(2150, 1, 2, 12, 0, 0),
            "spec_type_desc": "BLOOD CULTURE", "org_name": "STAPHYLOCOCCUS AUREUS",
        }
        event_uri = write_microbiology_event(g, micro_data, icu_stay_uri, icu_day_metadata, snomed_mapper=synthetic_mapper)
        assert _has_snomed_triples(g, event_uri, "3092008")

    def test_no_snomed_without_mapper(self, graph_with_patient):
        g, _, _, icu_stay_uri, icu_day_metadata = graph_with_patient
        micro_data = {
            "microevent_id": 1, "stay_id": 300,
            "charttime": datetime(2150, 1, 2, 12, 0, 0),
            "spec_type_desc": "BLOOD CULTURE", "org_name": "STAPHYLOCOCCUS AUREUS",
        }
        event_uri = write_microbiology_event(g, micro_data, icu_stay_uri, icu_day_metadata)
        assert _count_snomed_triples(g, event_uri) == 0


class TestComorbidity:
    """Tests for SNOMED enrichment on Comorbidity."""

    def test_snomed_triples_present(self, graph_with_patient, synthetic_mapper):
        g, patient_uri, _, _, _ = graph_with_patient
        comorbidity_data = {"subject_id": 100, "name": "diabetes", "value": True}
        comorbidity_uri = write_comorbidity(g, comorbidity_data, patient_uri, snomed_mapper=synthetic_mapper)
        assert _has_snomed_triples(g, comorbidity_uri, "73211009")

    def test_no_snomed_without_mapper(self, graph_with_patient):
        g, patient_uri, _, _, _ = graph_with_patient
        comorbidity_data = {"subject_id": 100, "name": "diabetes", "value": True}
        comorbidity_uri = write_comorbidity(g, comorbidity_data, patient_uri)
        assert _count_snomed_triples(g, comorbidity_uri) == 0
