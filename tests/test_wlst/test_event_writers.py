"""Tests for WLST event writers."""

import pytest
from datetime import datetime

from rdflib import Graph
from rdflib.namespace import RDF

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_construction.event_writers import write_icu_stay, write_icu_days
from src.wlst.event_writers import (
    write_gcs_event,
    write_icp_medication_event,
    write_map_event,
    write_neurosurgery_event,
    write_vasopressor_event,
    write_ventilation_event,
)


@pytest.fixture
def graph_with_icu():
    """Graph with a pre-existing ICU stay and day metadata."""
    g = Graph()
    stay_data = {
        "stay_id": 9001,
        "hadm_id": 501,
        "intime": datetime(2150, 1, 10, 10, 0),
        "outtime": datetime(2150, 1, 14, 10, 0),
        "los": 4.0,
    }
    admission_uri = MIMIC_NS["HA-501"]
    g.add((admission_uri, RDF.type, MIMIC_NS.HospitalAdmission))

    icu_stay_uri = write_icu_stay(g, stay_data, admission_uri)
    icu_day_metadata = write_icu_days(g, stay_data, icu_stay_uri)

    return g, icu_stay_uri, icu_day_metadata


class TestWriteGcsEvent:
    def test_creates_gcs_event(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        gcs_data = {
            "stay_id": 9001,
            "charttime": datetime(2150, 1, 10, 12, 0),
            "gcs_eye": 3,
            "gcs_verbal": 1,
            "gcs_motor": 4,
            "gcs_total": 8,
        }
        uri = write_gcs_event(g, gcs_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.GCSEvent) in g
        assert (uri, RDF.type, TIME_NS.Instant) in g

    def test_gcs_properties(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        gcs_data = {
            "stay_id": 9001,
            "charttime": datetime(2150, 1, 10, 12, 0),
            "gcs_eye": 2,
            "gcs_verbal": 1,
            "gcs_motor": 5,
            "gcs_total": 8,
        }
        uri = write_gcs_event(g, gcs_data, icu_stay_uri, icu_day_metadata)
        assert any(g.triples((uri, MIMIC_NS.hasGCSEye, None)))
        assert any(g.triples((uri, MIMIC_NS.hasGCSMotor, None)))
        assert any(g.triples((uri, MIMIC_NS.hasGCSTotal, None)))

    def test_linked_to_icu_stay(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        gcs_data = {
            "stay_id": 9001,
            "charttime": datetime(2150, 1, 10, 12, 0),
            "gcs_eye": 1, "gcs_verbal": 1, "gcs_motor": 1, "gcs_total": 3,
        }
        uri = write_gcs_event(g, gcs_data, icu_stay_uri, icu_day_metadata)
        assert (uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri) in g


class TestWriteVasopressorEvent:
    def test_creates_interval_event(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        vaso_data = {
            "stay_id": 9001,
            "starttime": datetime(2150, 1, 10, 14, 0),
            "endtime": datetime(2150, 1, 10, 18, 0),
            "label": "Norepinephrine",
            "rate": 0.1,
            "amount": 5.0,
            "rateuom": "mcg/kg/min",
            "amountuom": "mg",
        }
        uri = write_vasopressor_event(g, vaso_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.VasopressorEvent) in g
        assert (uri, RDF.type, TIME_NS.ProperInterval) in g
        assert any(g.triples((uri, MIMIC_NS.hasDrugName, None)))
        assert any(g.triples((uri, MIMIC_NS.hasRate, None)))


class TestWriteVentilationEvent:
    def test_creates_ventilation_event(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        vent_data = {
            "stay_id": 9001,
            "starttime": datetime(2150, 1, 10, 11, 0),
            "endtime": datetime(2150, 1, 12, 11, 0),
            "itemid": 225792,
            "label": "Invasive Ventilation",
        }
        uri = write_ventilation_event(g, vent_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.VentilationEvent) in g
        assert (uri, RDF.type, TIME_NS.ProperInterval) in g


class TestWriteNeurosurgeryEvent:
    def test_creates_with_interval(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        proc_data = {
            "stay_id": 9001,
            "starttime": datetime(2150, 1, 10, 16, 0),
            "endtime": datetime(2150, 1, 10, 20, 0),
            "itemid": 225752,
            "label": "Craniectomy",
        }
        uri = write_neurosurgery_event(g, proc_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.NeurosurgeryEvent) in g
        assert (uri, RDF.type, TIME_NS.ProperInterval) in g

    def test_creates_as_instant_without_endtime(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        proc_data = {
            "stay_id": 9001,
            "starttime": datetime(2150, 1, 11, 10, 0),
            "endtime": None,
            "itemid": 228114,
            "label": "ICP Monitor",
        }
        uri = write_neurosurgery_event(g, proc_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.NeurosurgeryEvent) in g
        assert (uri, RDF.type, TIME_NS.Instant) in g


class TestWriteIcpMedicationEvent:
    def test_creates_icp_med_event(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        med_data = {
            "stay_id": 9001,
            "starttime": datetime(2150, 1, 10, 15, 0),
            "endtime": datetime(2150, 1, 10, 16, 0),
            "label": "Mannitol",
            "amount": 100.0,
            "amountuom": "mL",
            "rate": 200.0,
            "rateuom": "mL/hr",
        }
        uri = write_icp_medication_event(g, med_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.ICPMedicationEvent) in g
        assert any(g.triples((uri, MIMIC_NS.hasDrugName, None)))


class TestWriteMapEvent:
    def test_creates_arterial_map(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        map_data = {
            "stay_id": 9001,
            "charttime": datetime(2150, 1, 10, 12, 0),
            "valuenum": 75.0,
            "itemid": 220052,
            "label": "Arterial BP Mean",
        }
        uri = write_map_event(g, map_data, icu_stay_uri, icu_day_metadata)
        assert (uri, RDF.type, MIMIC_NS.MAPEvent) in g
        assert (uri, RDF.type, TIME_NS.Instant) in g
        assert any(g.triples((uri, MIMIC_NS.hasMeasurementMethod, None)))

    def test_arterial_vs_cuff_method(self, graph_with_icu):
        g, icu_stay_uri, icu_day_metadata = graph_with_icu
        arterial = {
            "stay_id": 9001, "charttime": datetime(2150, 1, 10, 12, 0),
            "valuenum": 75.0, "itemid": 220052, "label": "Arterial",
        }
        cuff = {
            "stay_id": 9001, "charttime": datetime(2150, 1, 10, 13, 0),
            "valuenum": 70.0, "itemid": 220181, "label": "Cuff",
        }
        uri_a = write_map_event(g, arterial, icu_stay_uri, icu_day_metadata)
        uri_c = write_map_event(g, cuff, icu_stay_uri, icu_day_metadata)

        methods_a = list(g.objects(uri_a, MIMIC_NS.hasMeasurementMethod))
        methods_c = list(g.objects(uri_c, MIMIC_NS.hasMeasurementMethod))
        assert str(methods_a[0]) == "arterial"
        assert str(methods_c[0]) == "cuff"
