"""Tests for patient and admission graph writer (TDD Red Phase).

Test suite for Layer 2B: Patient and HospitalAdmission Graph Writer.
"""

import pytest
from datetime import datetime
from rdflib import Graph, Namespace, RDF, XSD, Literal, URIRef

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_construction.patient_writer import (
    write_patient,
    write_admission,
    link_sequential_admissions,
)


class TestWritePatient:
    """Tests for writing Patient individuals to the graph."""

    def test_write_patient_creates_individual(
        self, graph_with_ontology: Graph, sample_patient_data: dict
    ) -> None:
        """SPARQL ASK confirms <mimic:PA-100> rdf:type <mimic:Patient>."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)

        # SPARQL ASK query
        query = """
        ASK {
            ?patient rdf:type mimic:Patient .
            FILTER (?patient = <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#PA-100>)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Patient individual should exist with correct type"
        assert patient_uri == MIMIC_NS["PA-100"], "Should return correct patient URI"

    def test_write_patient_has_demographics(
        self, graph_with_ontology: Graph, sample_patient_data: dict
    ) -> None:
        """Patient has hasAge (xsd:integer), hasGender (xsd:string), hasSubjectId (xsd:integer)."""
        write_patient(graph_with_ontology, sample_patient_data)

        # SPARQL ASK for demographics
        query = """
        ASK {
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#PA-100> mimic:hasSubjectId ?subjectId ;
                                                          mimic:hasGender ?gender ;
                                                          mimic:hasAge ?age .
            FILTER (datatype(?subjectId) = xsd:integer)
            FILTER (datatype(?gender) = xsd:string)
            FILTER (datatype(?age) = xsd:integer)
            FILTER (?subjectId = 100)
            FILTER (?gender = "M")
            FILTER (?age = 65)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Patient should have correct demographics with proper datatypes"


class TestWriteAdmission:
    """Tests for writing HospitalAdmission individuals to the graph."""

    def test_write_admission_creates_interval(
        self, graph_with_ontology: Graph, sample_patient_data: dict, sample_admission_data: dict
    ) -> None:
        """HospitalAdmission individual created as time:Interval subclass."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        admission_uri = write_admission(graph_with_ontology, sample_admission_data, patient_uri)

        # SPARQL ASK query
        query = """
        ASK {
            ?admission rdf:type mimic:HospitalAdmission .
            FILTER (?admission = <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200>)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "HospitalAdmission individual should exist"
        assert admission_uri == MIMIC_NS["HA-200"], "Should return correct admission URI"

    def test_write_admission_has_temporal_bounds(
        self, graph_with_ontology: Graph, sample_patient_data: dict, sample_admission_data: dict
    ) -> None:
        """Admission has time:hasBeginning/hasEnd with time:Instant and time:inXSDDateTimeStamp."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        write_admission(graph_with_ontology, sample_admission_data, patient_uri)

        # SPARQL ASK for temporal bounds
        query = """
        ASK {
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200> time:hasBeginning ?begin ;
                                                          time:hasEnd ?end .
            ?begin rdf:type time:Instant ;
                   time:inXSDDateTimeStamp ?beginTime .
            ?end rdf:type time:Instant ;
                 time:inXSDDateTimeStamp ?endTime .
            FILTER (datatype(?beginTime) = xsd:dateTimeStamp)
            FILTER (datatype(?endTime) = xsd:dateTimeStamp)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Admission should have proper temporal bounds with Instants"

    def test_write_admission_has_readmission_labels(
        self, graph_with_ontology: Graph, sample_patient_data: dict, sample_admission_data: dict
    ) -> None:
        """Admission has readmittedWithin30Days/60Days (xsd:boolean)."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        write_admission(graph_with_ontology, sample_admission_data, patient_uri)

        # SPARQL ASK for readmission labels
        query = """
        ASK {
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200> mimic:readmittedWithin30Days ?r30 ;
                                                          mimic:readmittedWithin60Days ?r60 .
            FILTER (datatype(?r30) = xsd:boolean)
            FILTER (datatype(?r60) = xsd:boolean)
            FILTER (?r30 = true)
            FILTER (?r60 = true)
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Admission should have readmission labels as booleans"

    def test_write_admission_linked_to_patient(
        self, graph_with_ontology: Graph, sample_patient_data: dict, sample_admission_data: dict
    ) -> None:
        """Patient hasAdmission Admission, Admission admissionOf Patient."""
        patient_uri = write_patient(graph_with_ontology, sample_patient_data)
        write_admission(graph_with_ontology, sample_admission_data, patient_uri)

        # SPARQL ASK for bidirectional link
        query = """
        ASK {
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#PA-100> mimic:hasAdmission <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200> .
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-200> mimic:admissionOf <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#PA-100> .
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "Patient and admission should be bidirectionally linked"


class TestLinkSequentialAdmissions:
    """Tests for linking sequential admissions with followedBy."""

    def test_write_multiple_admissions_linked_by_followedby(
        self,
        graph_with_ontology: Graph,
        patient_with_multiple_admissions: tuple[dict, list[dict]],
    ) -> None:
        """Sequential admissions linked via followedBy property."""
        patient_data, admissions_data = patient_with_multiple_admissions

        patient_uri = write_patient(graph_with_ontology, patient_data)
        admission_uris = [
            write_admission(graph_with_ontology, adm, patient_uri) for adm in admissions_data
        ]
        link_sequential_admissions(graph_with_ontology, admission_uris)

        # SPARQL ASK for followedBy link
        query = """
        ASK {
            <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-301> mimic:followedBy <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#HA-302> .
        }
        """
        result = graph_with_ontology.query(query)
        assert bool(result), "First admission should be linked to second via followedBy"


class TestPatientRoundtrip:
    """Tests for complete patient trajectory retrieval via SPARQL."""

    def test_write_patient_roundtrip_sparql(
        self,
        graph_with_ontology: Graph,
        patient_with_multiple_admissions: tuple[dict, list[dict]],
    ) -> None:
        """Full SPARQL SELECT query verifying complete patient trajectory."""
        patient_data, admissions_data = patient_with_multiple_admissions

        patient_uri = write_patient(graph_with_ontology, patient_data)
        admission_uris = [
            write_admission(graph_with_ontology, adm, patient_uri) for adm in admissions_data
        ]
        link_sequential_admissions(graph_with_ontology, admission_uris)

        # SPARQL SELECT for full trajectory
        query = """
        SELECT ?patient ?patientGender ?patientAge ?admission ?admitTime ?dischTime ?readmit30
        WHERE {
            ?patient rdf:type mimic:Patient ;
                     mimic:hasSubjectId ?subjectId ;
                     mimic:hasGender ?patientGender ;
                     mimic:hasAge ?patientAge ;
                     mimic:hasAdmission ?admission .
            ?admission rdf:type mimic:HospitalAdmission ;
                       time:hasBeginning ?begin ;
                       time:hasEnd ?end ;
                       mimic:readmittedWithin30Days ?readmit30 .
            ?begin time:inXSDDateTimeStamp ?admitTime .
            ?end time:inXSDDateTimeStamp ?dischTime .
            FILTER (?subjectId = 300)
        }
        ORDER BY ?admitTime
        """
        results = list(graph_with_ontology.query(query))

        # Should have 2 admissions for patient 300
        assert len(results) == 2, f"Expected 2 admissions, got {len(results)}"

        # Verify first admission
        first = results[0]
        assert str(first.patientGender) == "F"
        assert int(first.patientAge) == 55
        assert first.readmit30.toPython() is True

        # Verify second admission
        second = results[1]
        assert second.readmit30.toPython() is False
