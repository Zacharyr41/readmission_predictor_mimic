"""Patient and admission graph writer for hospital readmission prediction.

This module provides functions to convert patient and admission data into RDF triples
following the extended MIMIC ontology with OWL-Time temporal modeling.
"""

from datetime import datetime

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.ontology import MIMIC_NS, TIME_NS


def write_patient(graph: Graph, patient_data: dict) -> URIRef:
    """Create Patient individual with demographics.

    Args:
        graph: RDF graph to write to.
        patient_data: Dictionary with keys: subject_id, gender, anchor_age.

    Returns:
        URIRef for the created patient.
    """
    subject_id = patient_data["subject_id"]
    patient_uri = MIMIC_NS[f"PA-{subject_id}"]

    # Type assertion
    graph.add((patient_uri, RDF.type, MIMIC_NS.Patient))

    # Demographics
    graph.add((patient_uri, MIMIC_NS.hasSubjectId, Literal(subject_id, datatype=XSD.integer)))
    graph.add((patient_uri, MIMIC_NS.hasGender, Literal(patient_data["gender"], datatype=XSD.string)))
    graph.add((patient_uri, MIMIC_NS.hasAge, Literal(patient_data["anchor_age"], datatype=XSD.integer)))

    return patient_uri


def _write_instant(graph: Graph, instant_uri: URIRef, timestamp: datetime) -> URIRef:
    """Create time:Instant with time:inXSDDateTimeStamp.

    Args:
        graph: RDF graph to write to.
        instant_uri: URI for the instant.
        timestamp: Datetime value.

    Returns:
        URIRef for the created instant.
    """
    graph.add((instant_uri, RDF.type, TIME_NS.Instant))

    # Format as xsd:dateTimeStamp (requires timezone, use UTC)
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((instant_uri, TIME_NS.inXSDDateTimeStamp, Literal(timestamp_str, datatype=XSD.dateTimeStamp)))

    return instant_uri


def write_admission(graph: Graph, admission_data: dict, patient_uri: URIRef) -> URIRef:
    """Create HospitalAdmission as time:Interval.

    Args:
        graph: RDF graph to write to.
        admission_data: Dictionary with keys: hadm_id, subject_id, admittime, dischtime,
                       admission_type, discharge_location, readmitted_30d, readmitted_60d.
        patient_uri: URIRef for the patient.

    Returns:
        URIRef for the created admission.
    """
    hadm_id = admission_data["hadm_id"]
    admission_uri = MIMIC_NS[f"HA-{hadm_id}"]

    # Type assertion
    graph.add((admission_uri, RDF.type, MIMIC_NS.HospitalAdmission))

    # Basic properties
    graph.add((admission_uri, MIMIC_NS.hasAdmissionId, Literal(hadm_id, datatype=XSD.integer)))
    graph.add((admission_uri, MIMIC_NS.hasAdmissionType, Literal(admission_data["admission_type"], datatype=XSD.string)))
    graph.add((admission_uri, MIMIC_NS.hasDischargeLocation, Literal(admission_data["discharge_location"], datatype=XSD.string)))

    # Temporal bounds (time:Interval pattern)
    begin_uri = MIMIC_NS[f"AdmissionBegin_{hadm_id}"]
    end_uri = MIMIC_NS[f"AdmissionEnd_{hadm_id}"]

    _write_instant(graph, begin_uri, admission_data["admittime"])
    _write_instant(graph, end_uri, admission_data["dischtime"])

    graph.add((admission_uri, TIME_NS.hasBeginning, begin_uri))
    graph.add((admission_uri, TIME_NS.hasEnd, end_uri))

    # Readmission labels
    graph.add((admission_uri, MIMIC_NS.readmittedWithin30Days, Literal(admission_data["readmitted_30d"], datatype=XSD.boolean)))
    graph.add((admission_uri, MIMIC_NS.readmittedWithin60Days, Literal(admission_data["readmitted_60d"], datatype=XSD.boolean)))

    # Bidirectional patient-admission links
    graph.add((patient_uri, MIMIC_NS.hasAdmission, admission_uri))
    graph.add((admission_uri, MIMIC_NS.admissionOf, patient_uri))

    return admission_uri


def link_sequential_admissions(graph: Graph, admission_uris: list[URIRef]) -> None:
    """Link consecutive admissions with followedBy property.

    Args:
        graph: RDF graph to write to.
        admission_uris: List of admission URIs in chronological order.
    """
    for i in range(len(admission_uris) - 1):
        current_admission = admission_uris[i]
        next_admission = admission_uris[i + 1]
        graph.add((current_admission, MIMIC_NS.followedBy, next_admission))
