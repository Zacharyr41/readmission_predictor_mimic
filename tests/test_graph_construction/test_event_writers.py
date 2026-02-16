"""Tests for clinical event graph writers (TDD Red Phase).

Test suite for Layer 2C: Clinical Event Graph Writers.
Tests cover ICU stays, ICU days, biomarkers, vitals, prescriptions, diagnoses, and comorbidities.
"""

import pytest
from datetime import datetime
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_construction.patient_writer import write_patient, write_admission
from src.graph_construction.event_writers import (
    _assign_event_to_icu_day,
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_clinical_sign_event,
    write_prescription_event,
    write_diagnosis_event,
    write_comorbidity,
)


class TestAssignEventToICUDay:
    """Tests for _assign_event_to_icu_day with None/edge-case inputs."""

    @pytest.fixture()
    def icu_day_metadata(self):
        """Standard 2-day ICU stay metadata."""
        return [
            (URIRef("urn:day1"), datetime(2150, 1, 1, 8, 0), datetime(2150, 1, 2, 0, 0)),
            (URIRef("urn:day2"), datetime(2150, 1, 2, 0, 0), datetime(2150, 1, 2, 14, 0)),
        ]

    def test_normal_assignment(self, icu_day_metadata):
        """Event within Day 1 boundaries returns Day 1 URI."""
        result = _assign_event_to_icu_day(datetime(2150, 1, 1, 12, 0), icu_day_metadata)
        assert result == URIRef("urn:day1")

    def test_none_charttime_returns_none(self, icu_day_metadata):
        """None charttime (NULL starttime in MIMIC-IV) returns None without crashing."""
        result = _assign_event_to_icu_day(None, icu_day_metadata)
        assert result is None

    def test_none_begin_boundary_skipped(self):
        """ICU day with None begin boundary is skipped."""
        metadata = [
            (URIRef("urn:bad"), None, datetime(2150, 1, 2, 0, 0)),
            (URIRef("urn:good"), datetime(2150, 1, 2, 0, 0), datetime(2150, 1, 3, 0, 0)),
        ]
        result = _assign_event_to_icu_day(datetime(2150, 1, 2, 12, 0), metadata)
        assert result == URIRef("urn:good")

    def test_none_end_boundary_skipped(self):
        """ICU day with None end boundary is skipped."""
        metadata = [
            (URIRef("urn:good"), datetime(2150, 1, 1, 0, 0), datetime(2150, 1, 2, 0, 0)),
            (URIRef("urn:bad"), datetime(2150, 1, 2, 0, 0), None),
        ]
        result = _assign_event_to_icu_day(datetime(2150, 1, 1, 12, 0), metadata)
        assert result == URIRef("urn:good")

    def test_all_none_boundaries_returns_none(self):
        """All ICU days with None boundaries returns None."""
        metadata = [
            (URIRef("urn:bad1"), None, None),
            (URIRef("urn:bad2"), None, datetime(2150, 1, 2, 0, 0)),
        ]
        result = _assign_event_to_icu_day(datetime(2150, 1, 1, 12, 0), metadata)
        assert result is None

    def test_event_outside_icu_stay(self, icu_day_metadata):
        """Event outside all ICU day boundaries returns None."""
        result = _assign_event_to_icu_day(datetime(2150, 6, 1, 0, 0), icu_day_metadata)
        assert result is None

    def test_event_at_exact_outtime(self, icu_day_metadata):
        """Event at exact outtime is assigned to the last day."""
        result = _assign_event_to_icu_day(datetime(2150, 1, 2, 14, 0), icu_day_metadata)
        assert result == URIRef("urn:day2")

    def test_empty_metadata_returns_none(self):
        """Empty ICU day metadata returns None."""
        result = _assign_event_to_icu_day(datetime(2150, 1, 1, 12, 0), [])
        assert result is None


class TestWriteICUStay:
    """Tests for writing ICUStay individuals to the graph."""

    def test_write_icu_stay_as_interval(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_icu_stay_data: dict,
    ) -> None:
        """ICUStay created with time:hasBeginning/hasEnd Instants and hasStayId/duration."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
        icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)

        # SPARQL ASK for ICU stay as interval
        query = """
        ASK {
            ?stay rdf:type mimic:ICUStay ;
                  mimic:hasStayId ?stayId ;
                  time:hasBeginning ?begin ;
                  time:hasEnd ?end ;
                  time:hasDuration ?duration .
            ?begin rdf:type time:Instant ;
                   time:inXSDDateTimeStamp ?beginTime .
            ?end rdf:type time:Instant ;
                 time:inXSDDateTimeStamp ?endTime .
            ?duration time:numericDuration ?durationValue .
            FILTER (?stay = <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-300>)
            FILTER (?stayId = 300)
            FILTER (datatype(?beginTime) = xsd:dateTimeStamp)
            FILTER (datatype(?endTime) = xsd:dateTimeStamp)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "ICUStay should be created as interval with temporal bounds and duration"
        assert icu_stay_uri == MIMIC_NS["IS-300"], "Should return correct ICU stay URI"


class TestWriteICUDays:
    """Tests for partitioning ICU stay into ICU days."""

    def test_write_icu_days_partition_stay(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_icu_stay_data: dict,
    ) -> None:
        """3.25-day stay produces 4 ICUDay entities with correct boundaries."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
        icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)
        icu_day_metadata = write_icu_days(graph_with_ontology, sample_icu_stay_data, icu_stay_uri)

        # Should produce 4 days for 3.25-day stay
        assert len(icu_day_metadata) == 4, f"Expected 4 ICU days, got {len(icu_day_metadata)}"

        # SPARQL query to count ICU days linked to stay
        query = """
        SELECT (COUNT(?day) AS ?count)
        WHERE {
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-300> mimic:hasICUDay ?day .
            ?day rdf:type mimic:ICUDay ;
                 mimic:partOf <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-300> ;
                 mimic:hasDayNumber ?dayNum ;
                 time:hasBeginning ?begin ;
                 time:hasEnd ?end .
        }
        """
        results = list(graph_with_ontology.query(query))
        assert int(results[0][0]) == 4, "ICU stay should have 4 linked ICU days"

        # Verify day boundaries
        # Day 1: 2150-01-01 08:00 -> 2150-01-02 00:00
        # Day 2: 2150-01-02 00:00 -> 2150-01-03 00:00
        # Day 3: 2150-01-03 00:00 -> 2150-01-04 00:00
        # Day 4: 2150-01-04 00:00 -> 2150-01-04 14:00
        day1_uri, day1_begin, day1_end = icu_day_metadata[0]
        assert day1_begin == datetime(2150, 1, 1, 8, 0, 0), "Day 1 should start at intime"
        assert day1_end == datetime(2150, 1, 2, 0, 0, 0), "Day 1 should end at midnight"

        day4_uri, day4_begin, day4_end = icu_day_metadata[3]
        assert day4_begin == datetime(2150, 1, 4, 0, 0, 0), "Day 4 should start at midnight"
        assert day4_end == datetime(2150, 1, 4, 14, 0, 0), "Day 4 should end at outtime"


class TestWriteBiomarkerEvent:
    """Tests for writing BioMarkerEvent individuals to the graph."""

    def test_write_biomarker_event_as_instant(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_icu_stay_data: dict,
        sample_biomarker_data: dict,
    ) -> None:
        """BioMarkerEvent created as time:Instant with all properties."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
        icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)
        icu_day_metadata = write_icu_days(graph_with_ontology, sample_icu_stay_data, icu_stay_uri)

        event_uri = write_biomarker_event(
            graph_with_ontology, sample_biomarker_data, icu_stay_uri, icu_day_metadata
        )

        # SPARQL ASK for biomarker event properties
        query = """
        ASK {
            ?event rdf:type mimic:BioMarkerEvent ;
                   rdf:type time:Instant ;
                   time:inXSDDateTimeStamp ?timestamp ;
                   mimic:hasItemId ?itemId ;
                   mimic:hasBiomarkerType ?label ;
                   mimic:hasFluid ?fluid ;
                   mimic:hasCategory ?category ;
                   mimic:hasValue ?value ;
                   mimic:hasUnit ?unit ;
                   mimic:hasRefRangeLower ?lower ;
                   mimic:hasRefRangeUpper ?upper .
            FILTER (?itemId = 50912)
            FILTER (?label = "Creatinine")
            FILTER (?fluid = "Blood")
            FILTER (ABS(?value - 1.2) < 0.001)
            FILTER (?unit = "mg/dL")
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "BioMarkerEvent should have all required properties"

    def test_biomarker_linked_to_icu_stay_and_day(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_icu_stay_data: dict,
        sample_biomarker_data: dict,
    ) -> None:
        """Biomarker event linked to ICU stay and correct ICU day (Day 2)."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
        icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)
        icu_day_metadata = write_icu_days(graph_with_ontology, sample_icu_stay_data, icu_stay_uri)

        write_biomarker_event(
            graph_with_ontology, sample_biomarker_data, icu_stay_uri, icu_day_metadata
        )

        # Biomarker at 2150-01-02 06:00 should be in Day 2 (2150-01-02 00:00 - 2150-01-03 00:00)
        query = """
        ASK {
            ?event rdf:type mimic:BioMarkerEvent ;
                   mimic:associatedWithICUStay <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-300> ;
                   mimic:associatedWithICUDay ?day .
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-300> mimic:hasICUStayEvent ?event .
            ?day mimic:hasICUDayEvent ?event ;
                 mimic:hasDayNumber 2 .
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Biomarker should be linked to ICU stay and Day 2"


class TestWriteClinicalSignEvent:
    """Tests for writing ClinicalSignEvent individuals to the graph."""

    def test_write_clinical_sign_event(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_icu_stay_data: dict,
        sample_clinical_sign_data: dict,
    ) -> None:
        """ClinicalSignEvent created as time:Instant with all properties."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
        icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)
        icu_day_metadata = write_icu_days(graph_with_ontology, sample_icu_stay_data, icu_stay_uri)

        event_uri = write_clinical_sign_event(
            graph_with_ontology, sample_clinical_sign_data, icu_stay_uri, icu_day_metadata
        )

        # SPARQL ASK for clinical sign event
        query = """
        ASK {
            ?event rdf:type mimic:ClinicalSignEvent ;
                   rdf:type time:Instant ;
                   time:inXSDDateTimeStamp ?timestamp ;
                   mimic:hasItemId ?itemId ;
                   mimic:hasClinicalSignName ?name ;
                   mimic:hasCategory ?category ;
                   mimic:hasValue ?value .
            FILTER (?itemId = 220045)
            FILTER (?name = "Heart Rate")
            FILTER (?category = "Routine Vital Signs")
            FILTER (?value = 78.0)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "ClinicalSignEvent should have all required properties"


class TestWritePrescriptionEvent:
    """Tests for writing PrescriptionEvent individuals to the graph."""

    def test_write_prescription_event_as_interval(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_icu_stay_data: dict,
        sample_prescription_data: dict,
    ) -> None:
        """PrescriptionEvent created as time:ProperInterval with start/stop times."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)
        icu_stay_uri = write_icu_stay(graph_with_ontology, sample_icu_stay_data, admission_uri)
        icu_day_metadata = write_icu_days(graph_with_ontology, sample_icu_stay_data, icu_stay_uri)

        event_uri = write_prescription_event(
            graph_with_ontology, sample_prescription_data, icu_stay_uri, icu_day_metadata
        )

        # SPARQL ASK for prescription event as interval
        query = """
        ASK {
            ?event rdf:type mimic:PrescriptionEvent ;
                   rdf:type time:ProperInterval ;
                   time:hasBeginning ?begin ;
                   time:hasEnd ?end ;
                   mimic:hasDrugName ?drug ;
                   mimic:hasDoseValue ?dose ;
                   mimic:hasDoseUnit ?unit ;
                   mimic:hasRoute ?route .
            ?begin rdf:type time:Instant ;
                   time:inXSDDateTimeStamp ?beginTime .
            ?end rdf:type time:Instant ;
                 time:inXSDDateTimeStamp ?endTime .
            FILTER (?drug = "Vancomycin")
            FILTER (?dose = 1000.0)
            FILTER (?unit = "mg")
            FILTER (?route = "IV")
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "PrescriptionEvent should be interval with all properties"


class TestWriteDiagnosisEvent:
    """Tests for writing DiagnosisEvent individuals to the graph."""

    def test_write_diagnosis_event(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_admission_data: dict,
        sample_diagnosis_data: dict,
    ) -> None:
        """DiagnosisEvent created with ICD code, version, title, and linked to admission."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)

        event_uri = write_diagnosis_event(
            graph_with_ontology, sample_diagnosis_data, admission_uri
        )

        # SPARQL ASK for diagnosis event
        query = """
        ASK {
            ?event rdf:type mimic:DiagnosisEvent ;
                   mimic:hasIcdCode ?code ;
                   mimic:hasIcdVersion ?version ;
                   mimic:hasLongTitle ?title ;
                   mimic:hasSequenceNumber ?seq ;
                   mimic:diagnosisOf <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200> .
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200> mimic:hasDiagnosis ?event .
            FILTER (?code = "I63.0")
            FILTER (?version = 10)
            FILTER (?seq = 1)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "DiagnosisEvent should have all properties and be linked to admission"


class TestWriteComorbidity:
    """Tests for writing Comorbidity individuals to the graph."""

    def test_write_comorbidity(
        self,
        graph_with_ontology: Graph,
        sample_patient_data: dict,
        sample_comorbidity_data: dict,
    ) -> None:
        """Comorbidity created and linked to patient."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)

        comorbidity_uri = write_comorbidity(
            graph_with_ontology, sample_comorbidity_data, patient_uri
        )

        # SPARQL ASK for comorbidity
        query = """
        ASK {
            ?comorbidity rdf:type mimic:Comorbidity ;
                        mimic:hasComorbidityName ?name ;
                        mimic:hasComorbidityValue ?value .
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#PA-100> mimic:hasComorbidity ?comorbidity .
            FILTER (?name = "diabetes")
            FILTER (?value = true)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Comorbidity should have name/value and be linked to patient"


class TestFullPatientGraph:
    """Integration tests for complete patient graph with all event types."""

    def test_full_patient_graph_sparql(
        self,
        graph_with_ontology: Graph,
        full_patient_with_events: dict,
    ) -> None:
        """Complete patient graph allows SPARQL query for events during ICU Day 1."""
        data = full_patient_with_events

        # Create patient and admission
        patient_uri = write_patient(graph_with_ontology, data["patient"])
        admission_uri = write_admission(graph_with_ontology, data["admission"], patient_uri)

        # Create ICU stay and days
        icu_stay_uri = write_icu_stay(graph_with_ontology, data["icu_stay"], admission_uri)
        icu_day_metadata = write_icu_days(graph_with_ontology, data["icu_stay"], icu_stay_uri)

        # Create events
        for biomarker in data["biomarkers"]:
            write_biomarker_event(graph_with_ontology, biomarker, icu_stay_uri, icu_day_metadata)

        for vital in data["vitals"]:
            write_clinical_sign_event(graph_with_ontology, vital, icu_stay_uri, icu_day_metadata)

        for prescription in data["prescriptions"]:
            write_prescription_event(graph_with_ontology, prescription, icu_stay_uri, icu_day_metadata)

        for diagnosis in data["diagnoses"]:
            write_diagnosis_event(graph_with_ontology, diagnosis, admission_uri)

        # SPARQL: Count events during ICU Day 1
        # Day 1: 2150-03-01 08:00 -> 2150-03-02 00:00
        # Events on Day 1: 1 biomarker (10:00), 1 vital (12:00), 1 prescription (starts 10:00)
        # Prescription starts at 10:00 on Day 1 so it's linked to Day 1
        query = """
        SELECT (COUNT(?event) AS ?count)
        WHERE {
            ?day rdf:type mimic:ICUDay ;
                 mimic:hasDayNumber 1 ;
                 mimic:partOf <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-502> ;
                 mimic:hasICUDayEvent ?event .
        }
        """
        results = list(graph_with_ontology.query(query))
        event_count = int(results[0][0])

        # Day 1 should have 3 events (1 biomarker + 1 vital + 1 prescription)
        assert event_count == 3, f"Expected 3 events on ICU Day 1, got {event_count}"

        # Verify biomarker on Day 2
        query_day2 = """
        SELECT (COUNT(?event) AS ?count)
        WHERE {
            ?day rdf:type mimic:ICUDay ;
                 mimic:hasDayNumber 2 ;
                 mimic:partOf <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#IS-502> ;
                 mimic:hasICUDayEvent ?event .
            ?event rdf:type mimic:BioMarkerEvent .
        }
        """
        results_day2 = list(graph_with_ontology.query(query_day2))
        day2_biomarker_count = int(results_day2[0][0])
        assert day2_biomarker_count == 1, f"Expected 1 biomarker on ICU Day 2, got {day2_biomarker_count}"
