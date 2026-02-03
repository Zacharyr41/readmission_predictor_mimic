"""Tests for Allen temporal relation computation (TDD Red Phase).

Test suite for Layer 2D: Allen interval algebra relation computation as post-processing
for clinical event graphs.
"""

import time
import pytest
from datetime import datetime
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_construction.patient_writer import write_patient, write_admission
from src.graph_construction.event_writers import (
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_antibiotic_event,
)
from src.graph_construction.temporal.allen_relations import (
    _classify_allen_relation,
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
        """Biomarker during antibiotic -> 'during' (biomarker inside antibiotic interval)."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Antibiotic: 08:00 - 18:00
        antibiotic = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Biomarker at 12:00 (inside antibiotic interval)
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

        write_antibiotic_event(graph, antibiotic, icu_stay_uri, icu_day_metadata)
        write_biomarker_event(graph, biomarker, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:inside triple exists (biomarker inside antibiotic)
        query = """
        ASK {
            ?biomarker time:inside ?antibiotic .
            ?biomarker rdf:type mimic:BioMarkerEvent .
            ?antibiotic rdf:type mimic:AntibioticAdmissionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:inside relation (biomarker during antibiotic)"

    def test_overlaps_relation_intervals(self, graph_with_temporal_setup: tuple) -> None:
        """Two overlapping antibiotics."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Antibiotic 1: 08:00 - 14:00
        antibiotic1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 14, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Antibiotic 2: 12:00 - 20:00 (overlaps with first)
        antibiotic2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 12, 0, 0),
            "stoptime": datetime(2150, 1, 1, 20, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_antibiotic_event(graph, antibiotic1, icu_stay_uri, icu_day_metadata)
        write_antibiotic_event(graph, antibiotic2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalOverlaps triple exists
        query = """
        ASK {
            ?a time:intervalOverlaps ?b .
            ?a rdf:type mimic:AntibioticAdmissionEvent .
            ?b rdf:type mimic:AntibioticAdmissionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalOverlaps relation"

    def test_meets_relation(self, graph_with_temporal_setup: tuple) -> None:
        """Consecutive antibiotics - first ends exactly when second starts."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Antibiotic 1: 08:00 - 12:00
        antibiotic1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 12, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Antibiotic 2: 12:00 - 18:00 (starts exactly when first ends)
        antibiotic2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 12, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_antibiotic_event(graph, antibiotic1, icu_stay_uri, icu_day_metadata)
        write_antibiotic_event(graph, antibiotic2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalMeets triple exists
        query = """
        ASK {
            ?a time:intervalMeets ?b .
            ?a rdf:type mimic:AntibioticAdmissionEvent .
            ?b rdf:type mimic:AntibioticAdmissionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalMeets relation"

    def test_starts_relation(self, graph_with_temporal_setup: tuple) -> None:
        """Same-start intervals, A ends first."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Antibiotic 1: 08:00 - 12:00
        antibiotic1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 12, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Antibiotic 2: 08:00 - 18:00 (same start, longer)
        antibiotic2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_antibiotic_event(graph, antibiotic1, icu_stay_uri, icu_day_metadata)
        write_antibiotic_event(graph, antibiotic2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalStarts triple exists
        query = """
        ASK {
            ?a time:intervalStarts ?b .
            ?a rdf:type mimic:AntibioticAdmissionEvent .
            ?b rdf:type mimic:AntibioticAdmissionEvent .
        }
        """
        result = graph.query(query)
        assert bool(result), "Expected time:intervalStarts relation"

    def test_finishes_relation(self, graph_with_temporal_setup: tuple) -> None:
        """Same-end intervals, A starts later."""
        graph, patient_uri, icu_stay_uri, icu_day_metadata = graph_with_temporal_setup

        # Antibiotic 1: 08:00 - 18:00
        antibiotic1 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 8, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        # Antibiotic 2: 12:00 - 18:00 (same end, starts later)
        antibiotic2 = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Ceftriaxone",
            "starttime": datetime(2150, 1, 1, 12, 0, 0),
            "stoptime": datetime(2150, 1, 1, 18, 0, 0),
            "dose_val_rx": 2000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }

        write_antibiotic_event(graph, antibiotic1, icu_stay_uri, icu_day_metadata)
        write_antibiotic_event(graph, antibiotic2, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        assert count >= 1, f"Expected at least 1 relation, got {count}"

        # Verify time:intervalFinishes triple exists
        query = """
        ASK {
            ?a time:intervalFinishes ?b .
            ?a rdf:type mimic:AntibioticAdmissionEvent .
            ?b rdf:type mimic:AntibioticAdmissionEvent .
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
        """5 biomarkers + 1 antibiotic - verify relation count."""
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

        # 1 antibiotic: 09:00 - 15:00 (overlaps with several biomarkers)
        antibiotic = {
            "hadm_id": 200,
            "stay_id": 300,
            "drug": "Vancomycin",
            "starttime": datetime(2150, 1, 1, 9, 0, 0),
            "stoptime": datetime(2150, 1, 1, 15, 0, 0),
            "dose_val_rx": 1000.0,
            "dose_unit_rx": "mg",
            "route": "IV",
        }
        write_antibiotic_event(graph, antibiotic, icu_stay_uri, icu_day_metadata)

        count = compute_allen_relations(graph, icu_stay_uri)

        # 6 events total, expect multiple relations
        # - Biomarkers form a chain of "before" relations
        # - Some biomarkers are "during" the antibiotic
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
