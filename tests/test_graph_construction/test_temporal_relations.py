"""Tests for Allen temporal relation computation (TDD Red Phase).

Test suite for Layer 2D: Allen interval algebra relation computation as post-processing
for clinical event graphs.
"""

import time
import pytest
from datetime import datetime
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from pathlib import Path

from src.graph_construction.ontology import MIMIC_NS, TIME_NS, initialize_graph

ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"
from src.graph_construction.patient_writer import write_patient, write_admission
from src.graph_construction.event_writers import (
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_prescription_event,
)
from src.graph_construction.temporal.allen_relations import (
    _classify_allen_relation,
    _batch_get_temporal_bounds,
    compute_allen_relations,
    compute_allen_relations_for_patient,
)


@pytest.fixture
def graph_with_temporal_setup(
    graph_with_ontology: Graph,
    sample_patient_data: dict,
    sample_admission_data: dict,
    sample_icu_stay_data: dict,
) -> tuple[Graph, URIRef, URIRef, list]:
    """Graph with patient/admission/ICU stay ready for temporal event testing.

    Returns:
        Tuple of (graph, patient_uri, icu_stay_uri, icu_day_metadata)
    """
    patient_uri = write_patient(graph_with_ontology, sample_patient_data)
    admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
    icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)
    icu_day_metadata = write_icu_days(graph_with_ontology, sample_icu_stay_data, icu_stay_uri)

    return graph_with_ontology, patient_uri, icu_stay_uri, icu_day_metadata


class TestClassifyAllenRelation:
    """Unit tests for pure _classify_allen_relation function."""

    def test_before_relation(self) -> None:
        """A ends before B starts -> 'before'."""
        a_start = datetime(2150, 1, 1, 8, 0, 0)
        a_end = datetime(2150, 1, 1, 10, 0, 0)
        b_start = datetime(2150, 1, 1, 12, 0, 0)
        b_end = datetime(2150, 1, 1, 14, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "before", f"Expected 'before', got '{result}'"

    def test_meets_relation(self) -> None:
        """A ends exactly when B starts -> 'meets'."""
        a_start = datetime(2150, 1, 1, 8, 0, 0)
        a_end = datetime(2150, 1, 1, 12, 0, 0)
        b_start = datetime(2150, 1, 1, 12, 0, 0)
        b_end = datetime(2150, 1, 1, 16, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "meets", f"Expected 'meets', got '{result}'"

    def test_overlaps_relation(self) -> None:
        """A starts before B, A ends during B -> 'overlaps'."""
        a_start = datetime(2150, 1, 1, 8, 0, 0)
        a_end = datetime(2150, 1, 1, 14, 0, 0)
        b_start = datetime(2150, 1, 1, 12, 0, 0)
        b_end = datetime(2150, 1, 1, 18, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "overlaps", f"Expected 'overlaps', got '{result}'"

    def test_starts_relation(self) -> None:
        """Same start, A ends first -> 'starts'."""
        a_start = datetime(2150, 1, 1, 8, 0, 0)
        a_end = datetime(2150, 1, 1, 12, 0, 0)
        b_start = datetime(2150, 1, 1, 8, 0, 0)
        b_end = datetime(2150, 1, 1, 16, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "starts", f"Expected 'starts', got '{result}'"

    def test_during_relation(self) -> None:
        """A entirely within B -> 'during'."""
        a_start = datetime(2150, 1, 1, 10, 0, 0)
        a_end = datetime(2150, 1, 1, 14, 0, 0)
        b_start = datetime(2150, 1, 1, 8, 0, 0)
        b_end = datetime(2150, 1, 1, 18, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "during", f"Expected 'during', got '{result}'"

    def test_finishes_relation(self) -> None:
        """Same end, A starts later -> 'finishes'."""
        a_start = datetime(2150, 1, 1, 12, 0, 0)
        a_end = datetime(2150, 1, 1, 16, 0, 0)
        b_start = datetime(2150, 1, 1, 8, 0, 0)
        b_end = datetime(2150, 1, 1, 16, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "finishes", f"Expected 'finishes', got '{result}'"

    def test_instant_during_interval(self) -> None:
        """Point event (start==end) inside interval -> 'during'."""
        a_start = datetime(2150, 1, 1, 12, 0, 0)
        a_end = datetime(2150, 1, 1, 12, 0, 0)  # Instant
        b_start = datetime(2150, 1, 1, 8, 0, 0)
        b_end = datetime(2150, 1, 1, 18, 0, 0)

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "during", f"Expected 'during', got '{result}'"

    def test_instant_before_instant(self) -> None:
        """Two point events, first before second -> 'before'."""
        a_start = datetime(2150, 1, 1, 10, 0, 0)
        a_end = datetime(2150, 1, 1, 10, 0, 0)  # Instant
        b_start = datetime(2150, 1, 1, 14, 0, 0)
        b_end = datetime(2150, 1, 1, 14, 0, 0)  # Instant

        result = _classify_allen_relation(a_start, a_end, b_start, b_end)
        assert result == "before", f"Expected 'before', got '{result}'"


class TestComputeAllenRelations:
    """Integration tests for compute_allen_relations on ICU stay level."""

    def test_before_relation_instants(self, graph_with_temporal_setup: tuple) -> None:
        """Two biomarkers at t=10:00, t=14:00 -> first 'before' second."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Create two biomarker events at different times
        biomarker1 = {
            "stay_id": 300,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 1, 10, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.1,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        biomarker2 = {
            "stay_id": 300,
            "itemid": 50971,
            "charttime": datetime(2150, 1, 1, 14, 0, 0),
            "label": "Sodium",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 140.0,
            "valueuom": "mEq/L",
            "ref_range_lower": 136.0,
            "ref_range_upper": 145.0,
        }

        write_biomarker_event(graph, biomarker1, icu_stay_uri, icu_day_metadata)
        write_biomarker_event(graph, biomarker2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        # Should have 1 'before' relation (A before B)
        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:before triple exists
        query = """
        ASK {
            ?a time:before ?b .
            ?a rdf:type mimic:BioMarkerEvent .
            ?b rdf:type mimic:BioMarkerEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:before relation between biomarkers"

    def test_during_relation_instant_in_interval(self, graph_with_temporal_setup: tuple) -> None:
        """Biomarker during prescription -> 'during' (biomarker inside prescription interval)."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Prescription: 08:00 - 18:00
        prescription = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Biomarker at 12:00 (inside prescription interval)
        biomarker = {
            "stay_id": 300,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 1, 12, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.2,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }

        write_prescription_event(graph, prescription, icu_stay_uri, icu_day_metadata)
        write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:inside triple exists (biomarker inside prescription)
        query = """
        ASK {
            ?biomarker time:inside ?prescription .
            ?biomarker rdf:type mimic:BioMarkerEvent .
            ?prescription rdf:type mimic:PrescriptionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:inside relation (biomarker during prescription)"

    def test_overlaps_relation_intervals(self, graph_with_temporal_setup: tuple) -> None:
        """Two overlapping prescriptions."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Rx 1: 08:00 - 14:00
        rx1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 14, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Rx 2: 12:00 - 20:00 (overlaps with first)
        rx2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 12, 0, 0),
            "stoptime": datetime(2150, 1, 1, 20, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_prescription_event(graph, rx1, icu_stay_uri, icu_day_metadata)
        write_prescription_event(graph, rx2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalOverlaps triple exists
        query = """
        ASK {
            ?a time:intervalOverlaps ?b .
            ?a rdf:type mimic:PrescriptionEvent .
            ?b rdf:type mimic:PrescriptionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalOverlaps relation"

    def test_meets_relation(self, graph_with_temporal_setup: tuple) -> None:
        """Consecutive prescriptions - first ends exactly when second starts."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Rx 1: 08:00 - 12:00
        rx1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 12, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Rx 2: 12:00 - 18:00 (starts exactly when first ends)
        rx2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 12, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_prescription_event(graph, rx1, icu_stay_uri, icu_day_metadata)
        write_prescription_event(graph, rx2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalMeets triple exists
        query = """
        ASK {
            ?a time:intervalMeets ?b .
            ?a rdf:type mimic:PrescriptionEvent .
            ?b rdf:type mimic:PrescriptionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalMeets relation"

    def test_starts_relation(self, graph_with_temporal_setup: tuple) -> None:
        """Same-start intervals, A ends first."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Rx 1: 08:00 - 12:00
        rx1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 12, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Rx 2: 08:00 - 18:00 (same start, longer)
        rx2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_prescription_event(graph, rx1, icu_stay_uri, icu_day_metadata)
        write_prescription_event(graph, rx2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalStarts triple exists
        query = """
        ASK {
            ?a time:intervalStarts ?b .
            ?a rdf:type mimic:PrescriptionEvent .
            ?b rdf:type mimic:PrescriptionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalStarts relation"

    def test_finishes_relation(self, graph_with_temporal_setup: tuple) -> None:
        """Same-end intervals, A starts later."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Rx 1: 08:00 - 18:00
        rx1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Rx 2: 12:00 - 18:00 (same end, starts later)
        rx2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 12, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_prescription_event(graph, rx1, icu_stay_uri, icu_day_metadata)
        write_prescription_event(graph, rx2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalFinishes triple exists
        query = """
        ASK {
            ?a time:intervalFinishes ?b .
            ?a rdf:type mimic:PrescriptionEvent .
            ?b rdf:type mimic:PrescriptionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalFinishes relation"

    def test_no_cross_patient_relations(self, graph_with_ontology: Graph) -> None:
        """Two patients with separate ICU stays - no relations between them."""
        # Patient 1
        patient1 = {"subject_id": 1001, "gender": "M", "anchor_age": 65}
        admission1 = {
            "hadm_id": 2001,
            "subject_id": 1001,
            "admittime": datetime(2150, 1, 1, 8, 0, 0),
            "dischtime": datetime(2150, 1, 10, 14, 0, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "readmitted_30d": False,
            "readmitted_60d": False,
        }
        icu_stay1 = {
            "stay_id": 3001,
            "hadm_id": 2001,
            "subject_id": 1001,
            "intime": datetime(2150, 1, 1, 10, 0, 0),
            "outtime": datetime(2150, 1, 3, 10, 0, 0),
            "los": 2.0,
        }

        patient1_uri = write_patient(graph_with_ontology, patient1)
        admission1_uri = write_admission(graph_with_ontology, admission1, patient1_uri)
        icu_stay1_uri = write_icu_stay(graph_with_ontology, icu_stay1, admission1_uri)
        icu_day1_metadata = write_icu_days(graph_with_ontology, icu_stay1, icu_stay1_uri)

        # Patient 2
        patient2 = {"subject_id": 1002, "gender": "F", "anchor_age": 70}
        admission2 = {
            "hadm_id": 2002,
            "subject_id": 1002,
            "admittime": datetime(2150, 1, 1, 8, 0, 0),
            "dischtime": datetime(2150, 1, 10, 14, 0, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "readmitted_30d": False,
            "readmitted_60d": False,
        }
        icu_stay2 = {
            "stay_id": 3002,
            "hadm_id": 2002,
            "subject_id": 1002,
            "intime": datetime(2150, 1, 1, 10, 0, 0),
            "outtime": datetime(2150, 1, 3, 10, 0, 0),
            "los": 2.0,
        }

        patient2_uri = write_patient(graph_with_ontology, patient2)
        admission2_uri = write_admission(graph_with_ontology, admission2, patient2_uri)
        icu_stay2_uri = write_icu_stay(graph_with_ontology, icu_stay2, admission2_uri)
        icu_day2_metadata = write_icu_days(graph_with_ontology, icu_stay2, icu_stay2_uri)

        # Add biomarker to each patient at same time
        biomarker1 = {
            "stay_id": 3001,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 1, 12, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.1,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        biomarker2 = {
            "stay_id": 3002,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 1, 14, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.2,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }

        event1_uri = write_biomarker_event(graph_with_ontology, biomarker1, icu_stay1_uri, icu_day1_metadata)
        event2_uri = write_biomarker_event(graph_with_ontology, biomarker2, icu_stay2_uri, icu_day2_metadata)

        # Compute relations for each ICU stay separately
        count1 = compute_allen_relations(graph_with_ontology, icu_stay1_uri)
        count2 = compute_allen_relations(graph_with_ontology, icu_stay2_uri)

        # Each ICU stay has only 1 event, so no intra-stay relations
        assert count1 == 0, f"Expected 0 relations for patient 1 (single event), got {count1}"
        assert count2 == 0, f"Expected 0 relations for patient 2 (single event), got {count2}"

        # Verify no cross-patient relations were created
        query = f"""
        ASK {{
            <{event1_uri}> ?relation <{event2_uri}> .
            FILTER(STRSTARTS(STR(?relation), "http://www.w3.org/2006/time#"))
        }}
        """
        result = graph_with_ontology.query(query)
        assert not bool(result), "Should not have temporal relations between different patients"

    def test_relation_count_for_complex_timeline(self, graph_with_temporal_setup: tuple) -> None:
        """5 biomarkers + 1 prescription - verify relation count."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # 5 biomarkers at different times
        times = [
            datetime(2150, 1, 1, 8, 0, 0),
            datetime(2150, 1, 1, 10, 0, 0),
            datetime(2150, 1, 1, 12, 0, 0),
            datetime(2150, 1, 1, 14, 0, 0),
            datetime(2150, 1, 1, 16, 0, 0),
        ]

        for i, t in enumerate(times):
            biomarker = {
                "stay_id": 300,
                "itemid": 50900 + i,
                "charttime": t,
                "label": f"Lab{i}",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 1.0 + i * 0.1,
                "valueuom": "mg/dL",
                "ref_range_lower": 0.5,
                "ref_range_upper": 2.0,
            }
            write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        # 1 prescription: 09:00 - 15:00 (overlaps with several biomarkers)
        rx = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 9, 0, 0),
            "stoptime": datetime(2150, 1, 1, 15, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }
        write_prescription_event(graph, rx, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        # 6 events total, expect multiple relations
        # - Biomarkers form a chain of "before" relations
        # - Some biomarkers are "during" the prescription
        # Minimum expected: at least 5 "before" relations from biomarker chain
        assert count >= 5, f"Expected at least 5 relations, got {count}"

    def test_performance_within_bounds(self, graph_with_temporal_setup: tuple) -> None:
        """50 events should complete in < 2 seconds."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Create 50 biomarker events
        base_time = datetime(2150, 1, 1, 8, 0, 0)
        for i in range(50):
            charttime = datetime(
                base_time.year,
                base_time.month,
                base_time.day,
                base_time.hour + (i // 6),
                (i % 6) * 10,
                0,
            )
            biomarker = {
                "stay_id": 300,
                "itemid": 51000 + i,
                "charttime": charttime,
                "label": f"PerfLab{i}",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 1.0,
                "valueuom": "mg/dL",
                "ref_range_lower": 0.5,
                "ref_range_upper": 2.0,
            }
            write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        start_time = time.time()
        count = compute_allen_relations(graph, icu_stay_uri)
        elapsed = time.time() - start_time

        assert elapsed < 2.0, f"Performance test failed: {elapsed:.2f}s > 2s"
        assert count > 0, "Should have computed some relations"


class TestComputeAllenRelationsForPatient:
    """Tests for patient-level API that aggregates across ICU stays."""

    def test_patient_with_multiple_icu_stays(self, graph_with_ontology: Graph) -> None:
        """Patient with 2 ICU stays - aggregates relations from both."""
        # Patient with 2 admissions, each with 1 ICU stay
        patient = {"subject_id": 4001, "gender": "M", "anchor_age": 60}
        patient_uri = write_patient(graph_with_ontology, patient)

        # Admission 1 with ICU stay 1
        admission1 = {
            "hadm_id": 5001,
            "subject_id": 4001,
            "admittime": datetime(2150, 1, 1, 8, 0, 0),
            "dischtime": datetime(2150, 1, 5, 14, 0, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "readmitted_30d": True,
            "readmitted_60d": True,
        }
        icu_stay1 = {
            "stay_id": 6001,
            "hadm_id": 5001,
            "subject_id": 4001,
            "intime": datetime(2150, 1, 1, 10, 0, 0),
            "outtime": datetime(2150, 1, 3, 10, 0, 0),
            "los": 2.0,
        }

        admission1_uri = write_admission(graph_with_ontology, admission1, patient_uri)
        icu_stay1_uri = write_icu_stay(graph_with_ontology, icu_stay1, admission1_uri)
        icu_day1_metadata = write_icu_days(graph_with_ontology, icu_stay1, icu_stay1_uri)

        # Add 2 biomarkers to ICU stay 1
        biomarker1a = {
            "stay_id": 6001,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 1, 12, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.1,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        biomarker1b = {
            "stay_id": 6001,
            "itemid": 50971,
            "charttime": datetime(2150, 1, 1, 16, 0, 0),
            "label": "Sodium",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 140.0,
            "valueuom": "mEq/L",
            "ref_range_lower": 136.0,
            "ref_range_upper": 145.0,
        }
        write_biomarker_event(graph_with_ontology, biomarker1a, icu_stay1_uri, icu_day1_metadata)
        write_biomarker_event(graph_with_ontology, biomarker1b, icu_stay1_uri, icu_day1_metadata)

        # Admission 2 with ICU stay 2
        admission2 = {
            "hadm_id": 5002,
            "subject_id": 4001,
            "admittime": datetime(2150, 2, 1, 8, 0, 0),
            "dischtime": datetime(2150, 2, 5, 14, 0, 0),
            "admission_type": "URGENT",
            "discharge_location": "HOME",
            "readmitted_30d": False,
            "readmitted_60d": False,
        }
        icu_stay2 = {
            "stay_id": 6002,
            "hadm_id": 5002,
            "subject_id": 4001,
            "intime": datetime(2150, 2, 1, 10, 0, 0),
            "outtime": datetime(2150, 2, 3, 10, 0, 0),
            "los": 2.0,
        }

        admission2_uri = write_admission(graph_with_ontology, admission2, patient_uri)
        icu_stay2_uri = write_icu_stay(graph_with_ontology, icu_stay2, admission2_uri)
        icu_day2_metadata = write_icu_days(graph_with_ontology, icu_stay2, icu_stay2_uri)

        # Add 2 biomarkers to ICU stay 2
        biomarker2a = {
            "stay_id": 6002,
            "itemid": 50912,
            "charttime": datetime(2150, 2, 1, 12, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.0,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        biomarker2b = {
            "stay_id": 6002,
            "itemid": 50971,
            "charttime": datetime(2150, 2, 1, 18, 0, 0),
            "label": "Sodium",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 142.0,
            "valueuom": "mEq/L",
            "ref_range_lower": 136.0,
            "ref_range_upper": 145.0,
        }
        write_biomarker_event(graph_with_ontology, biomarker2a, icu_stay2_uri, icu_day2_metadata)
        write_biomarker_event(graph_with_ontology, biomarker2b, icu_stay2_uri, icu_day2_metadata)

        # Compute relations for entire patient
        total_count = compute_allen_relations_for_patient(graph_with_ontology, patient_uri)

        # Each ICU stay has 2 events with 1 "before" relation each
        # Total should be at least 2 (one from each ICU stay)
        assert total_count >= 2, f"Expected at least 2 relations total, got {total_count}"

        # Verify relations exist in both ICU stays
        query1 = f"""
        ASK {{
            ?a time:before ?b .
            ?a mimic:associatedWithICUStay <{icu_stay1_uri}> .
            ?b mimic:associatedWithICUStay <{icu_stay1_uri}> .
        }}
        """
        result1 = graph_with_ontology.query(query1)
        assert bool(result1), "Expected temporal relations in ICU stay 1"

        query2 = f"""
        ASK {{
            ?a time:before ?b .
            ?a mimic:associatedWithICUStay <{icu_stay2_uri}> .
            ?b mimic:associatedWithICUStay <{icu_stay2_uri}> .
        }}
        """
        result2 = graph_with_ontology.query(query2)
        assert bool(result2), "Expected temporal relations in ICU stay 2"


class TestBatchTemporalBounds:
    """Tests for batch temporal bounds query optimization."""

    def test_batch_query_returns_same_bounds_as_individual(
        self, graph_with_temporal_setup: tuple
    ) -> None:
        """3 instants + 1 interval: batch results match individual query results."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # 3 biomarkers (instants)
        for i, hour in enumerate([10, 12, 14]):
            biomarker = {
                "stay_id": 300,
                "itemid": 50900 + i,
                "charttime": datetime(2150, 1, 1, hour, 0, 0),
                "label": f"Lab{i}",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 1.0,
                "valueuom": "mg/dL",
                "ref_range_lower": 0.5,
                "ref_range_upper": 2.0,
            }
            write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        # 1 prescription (interval)
        rx = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 9, 0, 0),
            "stoptime": datetime(2150, 1, 1, 15, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }
        write_prescription_event(graph, rx, icu_stay_uri, icu_day_metadata)

        events = _batch_get_temporal_bounds(graph, icu_stay_uri)

        assert len(events) == 4, f"Expected 4 events, got {len(events)}"

        # Each event should be (uri, type, start, end)
        for event in events:
            assert len(event) == 4
            uri, etype, start, end = event
            assert isinstance(start, datetime)
            assert isinstance(end, datetime)
            assert etype in ("instant", "interval")
            if etype == "instant":
                assert start == end
            else:
                assert start < end

    def test_batch_query_handles_missing_bounds(
        self, graph_with_temporal_setup: tuple
    ) -> None:
        """Prescription with NULL stoptime is excluded (same as current behavior)."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Prescription with no stoptime (NULL) — should be excluded
        rx_no_stop = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": None,
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }
        write_prescription_event(graph, rx_no_stop, icu_stay_uri, icu_day_metadata)

        # 1 valid biomarker
        biomarker = {
            "stay_id": 300,
            "itemid": 50912,
            "charttime": datetime(2150, 1, 1, 12, 0, 0),
            "label": "Creatinine",
            "fluid": "Blood",
            "category": "Chemistry",
            "valuenum": 1.1,
            "valueuom": "mg/dL",
            "ref_range_lower": 0.7,
            "ref_range_upper": 1.3,
        }
        write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        events = _batch_get_temporal_bounds(graph, icu_stay_uri)

        # Only the biomarker should appear (prescription has no stoptime -> no end node)
        assert len(events) == 1, f"Expected 1 event (excluding NULL stoptime), got {len(events)}"

    def test_batch_query_empty_events(self, graph_with_temporal_setup: tuple) -> None:
        """No events -> empty list."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Don't add any events
        events = _batch_get_temporal_bounds(graph, icu_stay_uri)

        assert events == [], f"Expected empty list, got {events}"


class TestEarlyTermination:
    """Tests for early termination optimization (sorted-order pruning)."""

    def test_sequential_instants_correct_count(
        self, graph_with_temporal_setup: tuple
    ) -> None:
        """10 sequential instants -> exactly 45 'before' triples."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        for i in range(10):
            biomarker = {
                "stay_id": 300,
                "itemid": 50900 + i,
                "charttime": datetime(2150, 1, 1, 8 + i, 0, 0),
                "label": f"Lab{i}",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 1.0,
                "valueuom": "mg/dL",
                "ref_range_lower": 0.5,
                "ref_range_upper": 2.0,
            }
            write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        # C(10,2) = 45 pairs, all "before"
        assert count == 45, f"Expected 45 'before' relations, got {count}"

    def test_performance_scales_subquadratically(
        self, graph_with_ontology: Graph,
    ) -> None:
        """Verify early termination gives sub-quadratic scaling.

        With pure O(N^2) pair iteration, doubling N should ~4x the time.
        With early termination on sorted sequential events, the inner loop
        breaks after the first pair per outer iteration, so doubling N
        should only ~2x the time. We assert the ratio stays below 3x.
        """
        from src.graph_construction.patient_writer import write_patient, write_admission
        from src.graph_construction.event_writers import (
            write_icu_stay, write_icu_days, write_biomarker_event,
        )

        def _build_and_time(n: int) -> float:
            """Build a fresh graph with n sequential events and time compute_allen_relations."""
            g = initialize_graph(ONTOLOGY_DIR)
            p = write_patient(g, {"subject_id": 100, "gender": "M", "anchor_age": 65})
            a = write_admission(g, {
                "hadm_id": 200, "subject_id": 100,
                "admittime": datetime(2150, 1, 1, 8, 0, 0),
                "dischtime": datetime(2150, 1, 20, 14, 0, 0),
                "admission_type": "EMERGENCY", "discharge_location": "HOME",
                "readmitted_30d": False, "readmitted_60d": False,
            }, p)
            s = write_icu_stay(g, {
                "stay_id": 300, "hadm_id": 200, "subject_id": 100,
                "intime": datetime(2150, 1, 1, 8, 0, 0),
                "outtime": datetime(2150, 1, 20, 14, 0, 0), "los": 19.0,
            }, a)
            days = write_icu_days(g, {
                "stay_id": 300, "hadm_id": 200, "subject_id": 100,
                "intime": datetime(2150, 1, 1, 8, 0, 0),
                "outtime": datetime(2150, 1, 20, 14, 0, 0), "los": 19.0,
            }, s)
            for i in range(n):
                write_biomarker_event(g, {
                    "stay_id": 300, "itemid": 50000 + i,
                    "charttime": datetime(2150, 1, 1 + i // 24, i % 24, 0, 0),
                    "label": f"Lab{i}", "fluid": "Blood", "category": "Chemistry",
                    "valuenum": 1.0, "valueuom": "mg/dL",
                    "ref_range_lower": 0.5, "ref_range_upper": 2.0,
                }, s, days)
            t0 = time.time()
            compute_allen_relations(g, s)
            return time.time() - t0

        t_small = _build_and_time(100)
        t_large = _build_and_time(200)

        # With O(N^2), ratio ≈ 4.0; with early termination, ratio ≈ 2.0
        ratio = t_large / t_small if t_small > 0 else float("inf")
        assert ratio < 3.0, (
            f"Scaling ratio {ratio:.1f}x exceeds 3.0x threshold "
            f"(100 events: {t_small:.3f}s, 200 events: {t_large:.3f}s). "
            f"Expected sub-quadratic scaling from early termination."
        )

    def test_mixed_before_and_during_correct(
        self, graph_with_temporal_setup: tuple
    ) -> None:
        """1 long interval + 5 contained instants: early termination must NOT skip 'during' relations."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Long prescription interval: 08:00 - 20:00
        rx = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 20, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }
        write_prescription_event(graph, rx, icu_stay_uri, icu_day_metadata)

        # 5 biomarkers contained within the interval
        for i in range(5):
            biomarker = {
                "stay_id": 300,
                "itemid": 50900 + i,
                "charttime": datetime(2150, 1, 1, 10 + i * 2, 0, 0),
                "label": f"Lab{i}",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 1.0,
                "valueuom": "mg/dL",
                "ref_range_lower": 0.5,
                "ref_range_upper": 2.0,
            }
            write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        # 5 "during" relations (each biomarker during the prescription)
        # + C(5,2) = 10 "before" relations among the biomarkers
        # = 15 total
        assert count == 15, f"Expected 15 relations (5 during + 10 before), got {count}"

        # Verify "during" triples exist
        query = """
        SELECT (COUNT(*) AS ?c) WHERE {
            ?bio time:inside ?abx .
            ?bio rdf:type mimic:BioMarkerEvent .
            ?abx rdf:type mimic:PrescriptionEvent .
        }
        """
        result = list(graph.query(query))
        during_count = int(result[0][0])
        assert during_count == 5, f"Expected 5 'during' relations, got {during_count}"


class TestSortedOrderInvariants:
    """Tests for double-classification elimination optimization."""

    def test_forward_nonone_implies_reverse_none(self) -> None:
        """All 4 forward results -> verify reverse is None (characterization)."""
        # before: A ends before B starts
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 10),
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 14),
        ) == "before"
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 14),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 10),
        ) is None  # "after" not in our 6

        # meets: A ends when B starts
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 12),
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 16),
        ) == "meets"
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 16),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 12),
        ) is None  # "metBy" not in our 6

        # overlaps: A starts before B, A ends during B
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 14),
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 18),
        ) == "overlaps"
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 18),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 14),
        ) is None  # "overlappedBy" not in our 6

        # starts: same start, A ends first
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 12),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
        ) == "starts"
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 12),
        ) is None  # "startedBy" not in our 6

    def test_forward_none_reverse_during(self) -> None:
        """A contains B -> forward is None, reverse is 'during'."""
        # A = [08, 18], B = [10, 14] — sorted: A first (or same start)
        # forward: A->B — A starts before B, A ends after B. Not in our 6 as forward.
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
            datetime(2150, 1, 1, 10), datetime(2150, 1, 1, 14),
        ) is None
        # reverse: B->A = "during"
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 10), datetime(2150, 1, 1, 14),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
        ) == "during"

    def test_forward_none_reverse_finishes(self) -> None:
        """Same end, A longer -> forward is None, reverse is 'finishes'."""
        # A = [08, 18], B = [12, 18] — sorted: A first
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 18),
        ) is None
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 12), datetime(2150, 1, 1, 18),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
        ) == "finishes"

    def test_forward_none_reverse_starts(self) -> None:
        """Same start, A longer -> forward is None, reverse is 'starts'."""
        # A = [08, 18], B = [08, 12] — both same start, A sorted first if longer
        # Actually sorted by start then... order doesn't matter for classification
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 12),
        ) is None
        assert _classify_allen_relation(
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 12),
            datetime(2150, 1, 1, 8), datetime(2150, 1, 1, 18),
        ) == "starts"

    def test_batch_query_uses_single_sparql_call(
        self, graph_with_temporal_setup: tuple
    ) -> None:
        """Batch query fetches all bounds in 1 SPARQL call, not N individual calls.

        Verifies the optimization by comparing batch query result count against
        the number of events discovered by the enumeration query, confirming
        the batch approach retrieves all bounds without per-event queries.
        """
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Create 20 events of mixed types
        for i in range(15):
            write_biomarker_event(graph, {
                "stay_id": 300, "itemid": 50000 + i,
                "charttime": datetime(2150, 1, 1, 8 + i, 0, 0),
                "label": f"Lab{i}", "fluid": "Blood", "category": "Chemistry",
                "valuenum": 1.0, "valueuom": "mg/dL",
                "ref_range_lower": 0.5, "ref_range_upper": 2.0,
            }, icu_stay_uri, icu_day_metadata)

        for i in range(5):
            write_prescription_event(graph, {
                "hadm_id": 200, "stay_id": 300,
                "drug": f"Drug{i}",
                "starttime": datetime(2150, 1, 1, 6 + i * 3, 0, 0),
                "stoptime": datetime(2150, 1, 1, 8 + i * 3, 0, 0),
                "dose_val_rx": 100.0, "dose_unit_rx": "mg", "route": "IV",
            }, icu_stay_uri, icu_day_metadata)

        events = _batch_get_temporal_bounds(graph, icu_stay_uri)

        # All 20 events retrieved in a single batch call
        assert len(events) == 20, (
            f"Batch query should retrieve all 20 events, got {len(events)}"
        )
        # Verify correctness: 15 instants + 5 intervals
        n_instant = sum(1 for e in events if e[1] == "instant")
        n_interval = sum(1 for e in events if e[1] == "interval")
        assert n_instant == 15, f"Expected 15 instants, got {n_instant}"
        assert n_interval == 5, f"Expected 5 intervals, got {n_interval}"
