"""Clinical event graph writers for hospital readmission prediction.

This module provides functions to convert clinical events (ICU stays, labs, vitals,
medications, diagnoses, comorbidities) into RDF triples following the MIMIC ontology
with OWL-Time temporal modeling.
"""

from datetime import datetime, timedelta
import re

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.ontology import MIMIC_NS, TIME_NS, SNOMED_NS


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
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((instant_uri, TIME_NS.inXSDDateTimeStamp, Literal(timestamp_str, datatype=XSD.dateTimeStamp)))
    return instant_uri


def _write_duration(graph: Graph, duration_uri: URIRef, days: float) -> URIRef:
    """Create time:Duration with time:numericDuration.

    Args:
        graph: RDF graph to write to.
        duration_uri: URI for the duration.
        days: Duration in days.

    Returns:
        URIRef for the created duration.
    """
    graph.add((duration_uri, RDF.type, TIME_NS.Duration))
    graph.add((duration_uri, TIME_NS.numericDuration, Literal(days, datatype=XSD.decimal)))
    graph.add((duration_uri, TIME_NS.unitType, TIME_NS.unitDay))
    return duration_uri


def _assign_event_to_icu_day(
    charttime: datetime, icu_day_metadata: list[tuple[URIRef, datetime, datetime]]
) -> URIRef | None:
    """Assign an event to the appropriate ICU day based on its charttime.

    Args:
        charttime: The timestamp of the event.
        icu_day_metadata: List of (day_uri, begin, end) tuples from write_icu_days.

    Returns:
        URIRef of the ICU day containing the event, or None if outside ICU stay.
    """
    for uri, begin, end in icu_day_metadata:
        if begin is None or end is None:
            continue
        if begin <= charttime < end:
            return uri
        # Include events at exact outtime in the last day
        if charttime == end and uri == icu_day_metadata[-1][0]:
            return uri
    return None


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _timestamp_slug(timestamp: datetime) -> str:
    """Convert timestamp to compact string for URI."""
    return timestamp.strftime("%Y%m%d%H%M%S")


def _add_snomed_triples(graph: Graph, event_uri: URIRef, snomed_concept) -> None:
    """Add SNOMED-CT triples to an event node if concept is not None."""
    if snomed_concept is None:
        return
    graph.add((event_uri, MIMIC_NS.hasSnomedCode, Literal(snomed_concept.code, datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasSnomedTerm, Literal(snomed_concept.term, datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasSnomedConcept, SNOMED_NS[snomed_concept.code]))


# ==================== ICU Structure ====================


def write_icu_stay(graph: Graph, stay_data: dict, admission_uri: URIRef) -> URIRef:
    """Create ICUStay individual as time:Interval.

    Args:
        graph: RDF graph to write to.
        stay_data: Dictionary with keys: stay_id, hadm_id, intime, outtime, los.
        admission_uri: URIRef for the hospital admission.

    Returns:
        URIRef for the created ICU stay.
    """
    stay_id = stay_data["stay_id"]
    icu_stay_uri = MIMIC_NS[f"IS-{stay_id}"]

    # Type assertion
    graph.add((icu_stay_uri, RDF.type, MIMIC_NS.ICUStay))

    # Stay ID
    graph.add((icu_stay_uri, MIMIC_NS.hasStayId, Literal(stay_id, datatype=XSD.integer)))

    # Temporal bounds
    begin_uri = MIMIC_NS[f"ICUStayBegin_{stay_id}"]
    end_uri = MIMIC_NS[f"ICUStayEnd_{stay_id}"]

    _write_instant(graph, begin_uri, stay_data["intime"])
    _write_instant(graph, end_uri, stay_data["outtime"])

    graph.add((icu_stay_uri, TIME_NS.hasBeginning, begin_uri))
    graph.add((icu_stay_uri, TIME_NS.hasEnd, end_uri))

    # Duration
    duration_uri = MIMIC_NS[f"ICUStayDuration_{stay_id}"]
    _write_duration(graph, duration_uri, stay_data["los"])
    graph.add((icu_stay_uri, TIME_NS.hasDuration, duration_uri))

    # Link to admission
    graph.add((admission_uri, MIMIC_NS.containsICUStay, icu_stay_uri))

    return icu_stay_uri


def write_icu_days(
    graph: Graph, stay_data: dict, icu_stay_uri: URIRef
) -> list[tuple[URIRef, datetime, datetime]]:
    """Partition ICU stay into ICU days.

    Partitioning algorithm:
    - Day 1: starts at intime, ends at midnight
    - Day 2..N-1: starts at midnight, ends at midnight
    - Day N: starts at midnight, ends at outtime

    Args:
        graph: RDF graph to write to.
        stay_data: Dictionary with keys: stay_id, intime, outtime.
        icu_stay_uri: URIRef for the ICU stay.

    Returns:
        List of (day_uri, begin_datetime, end_datetime) tuples.
    """
    stay_id = stay_data["stay_id"]
    intime = stay_data["intime"]
    outtime = stay_data["outtime"]

    icu_day_metadata = []
    day_num = 1
    current_start = intime

    while current_start < outtime:
        # End of current day is midnight of next calendar day
        next_midnight = datetime(
            current_start.year, current_start.month, current_start.day
        ) + timedelta(days=1)

        # If next midnight is after outtime, this is the last day
        if next_midnight >= outtime:
            current_end = outtime
        else:
            current_end = next_midnight

        # Create ICU day
        day_uri = MIMIC_NS[f"ID-{stay_id}-D{day_num}"]
        graph.add((day_uri, RDF.type, MIMIC_NS.ICUDay))
        graph.add((day_uri, MIMIC_NS.hasDayNumber, Literal(day_num, datatype=XSD.integer)))

        # Temporal bounds
        begin_uri = MIMIC_NS[f"ICUDayBegin_{stay_id}_D{day_num}"]
        end_uri = MIMIC_NS[f"ICUDayEnd_{stay_id}_D{day_num}"]
        _write_instant(graph, begin_uri, current_start)
        _write_instant(graph, end_uri, current_end)

        graph.add((day_uri, TIME_NS.hasBeginning, begin_uri))
        graph.add((day_uri, TIME_NS.hasEnd, end_uri))

        # Bidirectional links
        graph.add((icu_stay_uri, MIMIC_NS.hasICUDay, day_uri))
        graph.add((day_uri, MIMIC_NS.partOf, icu_stay_uri))

        icu_day_metadata.append((day_uri, current_start, current_end))

        # Move to next day
        current_start = current_end
        day_num += 1

    return icu_day_metadata


# ==================== Events (Instant-based) ====================


def write_biomarker_event(
    graph: Graph,
    lab_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
    snomed_mapper=None,
) -> URIRef:
    """Create BioMarkerEvent as time:Instant.

    Args:
        graph: RDF graph to write to.
        lab_data: Dictionary with keys: stay_id, itemid, charttime, label, fluid,
                  category, valuenum, valueuom, ref_range_lower, ref_range_upper.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = lab_data["stay_id"]
    itemid = lab_data["itemid"]
    charttime = lab_data["charttime"]
    timestamp_str = _timestamp_slug(charttime)

    event_uri = MIMIC_NS[f"BME-{stay_id}-{itemid}-{timestamp_str}"]

    # Type assertions
    graph.add((event_uri, RDF.type, MIMIC_NS.BioMarkerEvent))
    graph.add((event_uri, RDF.type, TIME_NS.Instant))

    # Timestamp
    timestamp_literal = charttime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((event_uri, TIME_NS.inXSDDateTimeStamp, Literal(timestamp_literal, datatype=XSD.dateTimeStamp)))

    # Properties
    graph.add((event_uri, MIMIC_NS.hasItemId, Literal(itemid, datatype=XSD.integer)))
    graph.add((event_uri, MIMIC_NS.hasBiomarkerType, Literal(lab_data["label"], datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasFluid, Literal(lab_data["fluid"], datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasCategory, Literal(lab_data["category"], datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasValue, Literal(lab_data["valuenum"], datatype=XSD.decimal)))
    graph.add((event_uri, MIMIC_NS.hasUnit, Literal(lab_data["valueuom"], datatype=XSD.string)))

    if lab_data.get("ref_range_lower") is not None:
        graph.add((event_uri, MIMIC_NS.hasRefRangeLower, Literal(lab_data["ref_range_lower"], datatype=XSD.decimal)))
    if lab_data.get("ref_range_upper") is not None:
        graph.add((event_uri, MIMIC_NS.hasRefRangeUpper, Literal(lab_data["ref_range_upper"], datatype=XSD.decimal)))

    # SNOMED-CT mapping
    if snomed_mapper is not None:
        _add_snomed_triples(graph, event_uri, snomed_mapper.get_snomed_for_labitem(itemid))

    # Link to ICU stay (bidirectional)
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    # Link to ICU day (bidirectional)
    icu_day_uri = _assign_event_to_icu_day(charttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_clinical_sign_event(
    graph: Graph,
    vital_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
    snomed_mapper=None,
) -> URIRef:
    """Create ClinicalSignEvent as time:Instant.

    Args:
        graph: RDF graph to write to.
        vital_data: Dictionary with keys: stay_id, itemid, charttime, label, category, valuenum.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = vital_data["stay_id"]
    itemid = vital_data["itemid"]
    charttime = vital_data["charttime"]
    timestamp_str = _timestamp_slug(charttime)

    event_uri = MIMIC_NS[f"CSE-{stay_id}-{itemid}-{timestamp_str}"]

    # Type assertions
    graph.add((event_uri, RDF.type, MIMIC_NS.ClinicalSignEvent))
    graph.add((event_uri, RDF.type, TIME_NS.Instant))

    # Timestamp
    timestamp_literal = charttime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((event_uri, TIME_NS.inXSDDateTimeStamp, Literal(timestamp_literal, datatype=XSD.dateTimeStamp)))

    # Properties
    graph.add((event_uri, MIMIC_NS.hasItemId, Literal(itemid, datatype=XSD.integer)))
    graph.add((event_uri, MIMIC_NS.hasClinicalSignName, Literal(vital_data["label"], datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasCategory, Literal(vital_data["category"], datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasValue, Literal(vital_data["valuenum"], datatype=XSD.decimal)))

    # SNOMED-CT mapping
    if snomed_mapper is not None:
        _add_snomed_triples(graph, event_uri, snomed_mapper.get_snomed_for_chartitem(itemid))

    # Link to ICU stay (bidirectional)
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    # Link to ICU day (bidirectional)
    icu_day_uri = _assign_event_to_icu_day(charttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_microbiology_event(
    graph: Graph,
    micro_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
    snomed_mapper=None,
) -> URIRef:
    """Create MicrobiologyEvent as time:Instant.

    Args:
        graph: RDF graph to write to.
        micro_data: Dictionary with keys: microevent_id, stay_id, charttime, spec_type_desc, org_name.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    microevent_id = micro_data["microevent_id"]
    charttime = micro_data["charttime"]

    event_uri = MIMIC_NS[f"MBE-{microevent_id}"]

    # Type assertions
    graph.add((event_uri, RDF.type, MIMIC_NS.MicrobiologyEvent))
    graph.add((event_uri, RDF.type, TIME_NS.Instant))

    # Timestamp
    timestamp_literal = charttime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((event_uri, TIME_NS.inXSDDateTimeStamp, Literal(timestamp_literal, datatype=XSD.dateTimeStamp)))

    # Properties
    if micro_data.get("spec_type_desc"):
        graph.add((event_uri, MIMIC_NS.hasSpecimenType, Literal(micro_data["spec_type_desc"], datatype=XSD.string)))
    if micro_data.get("org_name"):
        graph.add((event_uri, MIMIC_NS.hasOrganismName, Literal(micro_data["org_name"], datatype=XSD.string)))

    # SNOMED-CT mapping (via organism name)
    if snomed_mapper is not None and micro_data.get("org_name"):
        _add_snomed_triples(graph, event_uri, snomed_mapper.get_snomed_for_organism(micro_data["org_name"]))

    # Link to ICU stay (bidirectional)
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    # Link to ICU day (bidirectional)
    icu_day_uri = _assign_event_to_icu_day(charttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


# ==================== Events (Interval-based) ====================


def write_prescription_event(
    graph: Graph,
    rx_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
    snomed_mapper=None,
) -> URIRef:
    """Create PrescriptionEvent as time:ProperInterval.

    Args:
        graph: RDF graph to write to.
        rx_data: Dictionary with keys: hadm_id, drug, starttime, stoptime,
                 dose_val_rx, dose_unit_rx, route.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    hadm_id = rx_data["hadm_id"]
    drug = rx_data["drug"]
    drug_slug = _slugify(drug)
    starttime = rx_data["starttime"]

    event_uri = MIMIC_NS[f"RXE-{hadm_id}-{drug_slug}"]

    # Type assertions
    graph.add((event_uri, RDF.type, MIMIC_NS.PrescriptionEvent))
    graph.add((event_uri, RDF.type, TIME_NS.ProperInterval))

    # Temporal bounds (starttime may be NULL for some prescriptions)
    if rx_data.get("starttime") is not None:
        begin_uri = MIMIC_NS[f"RXEBegin_{hadm_id}_{drug_slug}"]
        _write_instant(graph, begin_uri, rx_data["starttime"])
        graph.add((event_uri, TIME_NS.hasBeginning, begin_uri))

    # End time may be NULL for ongoing prescriptions
    if rx_data.get("stoptime") is not None:
        end_uri = MIMIC_NS[f"RXEEnd_{hadm_id}_{drug_slug}"]
        _write_instant(graph, end_uri, rx_data["stoptime"])
        graph.add((event_uri, TIME_NS.hasEnd, end_uri))

    # Properties
    graph.add((event_uri, MIMIC_NS.hasDrugName, Literal(drug, datatype=XSD.string)))

    # SNOMED-CT mapping
    if snomed_mapper is not None:
        _add_snomed_triples(graph, event_uri, snomed_mapper.get_snomed_for_drug(drug))

    if rx_data.get("dose_val_rx") is not None:
        graph.add((event_uri, MIMIC_NS.hasDoseValue, Literal(rx_data["dose_val_rx"], datatype=XSD.decimal)))
    if rx_data.get("dose_unit_rx"):
        graph.add((event_uri, MIMIC_NS.hasDoseUnit, Literal(rx_data["dose_unit_rx"], datatype=XSD.string)))
    if rx_data.get("route"):
        graph.add((event_uri, MIMIC_NS.hasRoute, Literal(rx_data["route"], datatype=XSD.string)))

    # Link to ICU stay (bidirectional)
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    # Link to ICU day based on start time (bidirectional)
    icu_day_uri = _assign_event_to_icu_day(starttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


# ==================== Non-temporal ====================


def write_diagnosis_event(graph: Graph, dx_data: dict, admission_uri: URIRef, snomed_mapper=None) -> URIRef:
    """Create DiagnosisEvent linked to admission.

    Args:
        graph: RDF graph to write to.
        dx_data: Dictionary with keys: hadm_id, seq_num, icd_code, icd_version, long_title.
        admission_uri: URIRef for the hospital admission.

    Returns:
        URIRef for the created diagnosis event.
    """
    hadm_id = dx_data["hadm_id"]
    seq_num = dx_data["seq_num"]

    event_uri = MIMIC_NS[f"DXE-{hadm_id}-{seq_num}"]

    # Type assertion
    graph.add((event_uri, RDF.type, MIMIC_NS.DiagnosisEvent))

    # Properties
    graph.add((event_uri, MIMIC_NS.hasIcdCode, Literal(dx_data["icd_code"], datatype=XSD.string)))
    graph.add((event_uri, MIMIC_NS.hasIcdVersion, Literal(dx_data["icd_version"], datatype=XSD.integer)))
    graph.add((event_uri, MIMIC_NS.hasSequenceNumber, Literal(seq_num, datatype=XSD.integer)))

    if dx_data.get("long_title"):
        graph.add((event_uri, MIMIC_NS.hasLongTitle, Literal(dx_data["long_title"], datatype=XSD.string)))

    # SNOMED-CT mapping
    if snomed_mapper is not None:
        _add_snomed_triples(
            graph, event_uri,
            snomed_mapper.get_snomed_for_icd(dx_data["icd_code"], dx_data["icd_version"]),
        )

    # Bidirectional link to admission
    graph.add((admission_uri, MIMIC_NS.hasDiagnosis, event_uri))
    graph.add((event_uri, MIMIC_NS.diagnosisOf, admission_uri))

    return event_uri


def write_comorbidity(graph: Graph, comorbidity_data: dict, patient_uri: URIRef, snomed_mapper=None) -> URIRef:
    """Create Comorbidity individual linked to patient.

    Args:
        graph: RDF graph to write to.
        comorbidity_data: Dictionary with keys: subject_id, name, value.
        patient_uri: URIRef for the patient.

    Returns:
        URIRef for the created comorbidity.
    """
    subject_id = comorbidity_data["subject_id"]
    name = comorbidity_data["name"]

    comorbidity_uri = MIMIC_NS[f"CM-{subject_id}-{name}"]

    # Type assertion
    graph.add((comorbidity_uri, RDF.type, MIMIC_NS.Comorbidity))

    # Properties
    graph.add((comorbidity_uri, MIMIC_NS.hasComorbidityName, Literal(name, datatype=XSD.string)))
    graph.add((comorbidity_uri, MIMIC_NS.hasComorbidityValue, Literal(comorbidity_data["value"], datatype=XSD.boolean)))

    # SNOMED-CT mapping
    if snomed_mapper is not None:
        _add_snomed_triples(graph, comorbidity_uri, snomed_mapper.get_snomed_for_comorbidity(name))

    # Link to patient
    graph.add((patient_uri, MIMIC_NS.hasComorbidity, comorbidity_uri))

    return comorbidity_uri
