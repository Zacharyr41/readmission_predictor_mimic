"""WLST-specific event graph writers.

New RDF event types for TBI/WLST clinical data: GCS series, vasopressor
events, ventilation events, neurosurgery procedures, ICP medications,
and MAP (mean arterial pressure) measurements.
"""

from datetime import datetime

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_construction.event_writers import (
    _write_instant,
    _assign_event_to_icu_day,
    _slugify,
    _timestamp_slug,
)


def write_gcs_event(
    graph: Graph,
    gcs_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
) -> URIRef:
    """Create GCSEvent as time:Instant.

    Args:
        graph: RDF graph to write to.
        gcs_data: Dictionary with keys: stay_id, charttime, gcs_eye, gcs_verbal,
                  gcs_motor, gcs_total.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = gcs_data["stay_id"]
    charttime = gcs_data["charttime"]
    timestamp_str = _timestamp_slug(charttime)

    event_uri = MIMIC_NS[f"GCS-{stay_id}-{timestamp_str}"]

    graph.add((event_uri, RDF.type, MIMIC_NS.GCSEvent))
    graph.add((event_uri, RDF.type, TIME_NS.Instant))

    ts_literal = charttime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((event_uri, TIME_NS.inXSDDateTimeStamp, Literal(ts_literal, datatype=XSD.dateTimeStamp)))

    if gcs_data.get("gcs_eye") is not None:
        graph.add((event_uri, MIMIC_NS.hasGCSEye, Literal(gcs_data["gcs_eye"], datatype=XSD.decimal)))
    if gcs_data.get("gcs_verbal") is not None:
        graph.add((event_uri, MIMIC_NS.hasGCSVerbal, Literal(gcs_data["gcs_verbal"], datatype=XSD.decimal)))
    if gcs_data.get("gcs_motor") is not None:
        graph.add((event_uri, MIMIC_NS.hasGCSMotor, Literal(gcs_data["gcs_motor"], datatype=XSD.decimal)))
    if gcs_data.get("gcs_total") is not None:
        graph.add((event_uri, MIMIC_NS.hasGCSTotal, Literal(gcs_data["gcs_total"], datatype=XSD.decimal)))

    # Link to ICU stay
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    # Link to ICU day
    icu_day_uri = _assign_event_to_icu_day(charttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_vasopressor_event(
    graph: Graph,
    vaso_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
) -> URIRef:
    """Create VasopressorEvent as time:ProperInterval.

    Args:
        graph: RDF graph to write to.
        vaso_data: Dictionary with keys: stay_id, starttime, endtime, label,
                   rate, amount, amountuom, rateuom.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = vaso_data["stay_id"]
    starttime = vaso_data["starttime"]
    drug_slug = _slugify(vaso_data["label"])
    timestamp_str = _timestamp_slug(starttime)

    event_uri = MIMIC_NS[f"VASO-{stay_id}-{drug_slug}-{timestamp_str}"]

    graph.add((event_uri, RDF.type, MIMIC_NS.VasopressorEvent))
    graph.add((event_uri, RDF.type, TIME_NS.ProperInterval))

    # Temporal bounds
    begin_uri = MIMIC_NS[f"VASOBegin_{stay_id}_{drug_slug}_{timestamp_str}"]
    _write_instant(graph, begin_uri, starttime)
    graph.add((event_uri, TIME_NS.hasBeginning, begin_uri))

    if vaso_data.get("endtime") is not None:
        end_uri = MIMIC_NS[f"VASOEnd_{stay_id}_{drug_slug}_{timestamp_str}"]
        _write_instant(graph, end_uri, vaso_data["endtime"])
        graph.add((event_uri, TIME_NS.hasEnd, end_uri))

    # Properties
    graph.add((event_uri, MIMIC_NS.hasDrugName, Literal(vaso_data["label"], datatype=XSD.string)))
    if vaso_data.get("rate") is not None:
        graph.add((event_uri, MIMIC_NS.hasRate, Literal(vaso_data["rate"], datatype=XSD.decimal)))
    if vaso_data.get("amount") is not None:
        graph.add((event_uri, MIMIC_NS.hasAmount, Literal(vaso_data["amount"], datatype=XSD.decimal)))
    if vaso_data.get("rateuom"):
        graph.add((event_uri, MIMIC_NS.hasRateUnit, Literal(vaso_data["rateuom"], datatype=XSD.string)))
    if vaso_data.get("amountuom"):
        graph.add((event_uri, MIMIC_NS.hasAmountUnit, Literal(vaso_data["amountuom"], datatype=XSD.string)))

    # Link to ICU stay
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    # Link to ICU day
    icu_day_uri = _assign_event_to_icu_day(starttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_ventilation_event(
    graph: Graph,
    vent_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
) -> URIRef:
    """Create VentilationEvent as time:ProperInterval.

    Args:
        graph: RDF graph to write to.
        vent_data: Dictionary with keys: stay_id, starttime, endtime, itemid, label.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = vent_data["stay_id"]
    starttime = vent_data["starttime"]
    timestamp_str = _timestamp_slug(starttime)

    event_uri = MIMIC_NS[f"VENT-{stay_id}-{timestamp_str}"]

    graph.add((event_uri, RDF.type, MIMIC_NS.VentilationEvent))
    graph.add((event_uri, RDF.type, TIME_NS.ProperInterval))

    # Temporal bounds
    begin_uri = MIMIC_NS[f"VENTBegin_{stay_id}_{timestamp_str}"]
    _write_instant(graph, begin_uri, starttime)
    graph.add((event_uri, TIME_NS.hasBeginning, begin_uri))

    if vent_data.get("endtime") is not None:
        end_uri = MIMIC_NS[f"VENTEnd_{stay_id}_{timestamp_str}"]
        _write_instant(graph, end_uri, vent_data["endtime"])
        graph.add((event_uri, TIME_NS.hasEnd, end_uri))

    graph.add((event_uri, MIMIC_NS.hasItemId, Literal(vent_data["itemid"], datatype=XSD.integer)))
    if vent_data.get("label"):
        graph.add((event_uri, MIMIC_NS.hasProcedureName, Literal(vent_data["label"], datatype=XSD.string)))

    # Link to ICU stay
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    icu_day_uri = _assign_event_to_icu_day(starttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_neurosurgery_event(
    graph: Graph,
    proc_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
) -> URIRef:
    """Create NeurosurgeryEvent as time:Instant or time:ProperInterval.

    Args:
        graph: RDF graph to write to.
        proc_data: Dictionary with keys: stay_id, starttime, endtime, itemid, label.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = proc_data["stay_id"]
    starttime = proc_data["starttime"]
    timestamp_str = _timestamp_slug(starttime)
    proc_slug = _slugify(proc_data.get("label", str(proc_data["itemid"])))

    event_uri = MIMIC_NS[f"NSURG-{stay_id}-{proc_slug}-{timestamp_str}"]

    graph.add((event_uri, RDF.type, MIMIC_NS.NeurosurgeryEvent))

    # Use interval if endtime exists, instant otherwise
    if proc_data.get("endtime") is not None:
        graph.add((event_uri, RDF.type, TIME_NS.ProperInterval))
        begin_uri = MIMIC_NS[f"NSURGBegin_{stay_id}_{proc_slug}_{timestamp_str}"]
        _write_instant(graph, begin_uri, starttime)
        graph.add((event_uri, TIME_NS.hasBeginning, begin_uri))

        end_uri = MIMIC_NS[f"NSURGEnd_{stay_id}_{proc_slug}_{timestamp_str}"]
        _write_instant(graph, end_uri, proc_data["endtime"])
        graph.add((event_uri, TIME_NS.hasEnd, end_uri))
    else:
        graph.add((event_uri, RDF.type, TIME_NS.Instant))
        ts_literal = starttime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        graph.add((event_uri, TIME_NS.inXSDDateTimeStamp, Literal(ts_literal, datatype=XSD.dateTimeStamp)))

    graph.add((event_uri, MIMIC_NS.hasItemId, Literal(proc_data["itemid"], datatype=XSD.integer)))
    if proc_data.get("label"):
        graph.add((event_uri, MIMIC_NS.hasProcedureName, Literal(proc_data["label"], datatype=XSD.string)))

    # Link to ICU stay
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    icu_day_uri = _assign_event_to_icu_day(starttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_icp_medication_event(
    graph: Graph,
    med_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
) -> URIRef:
    """Create ICPMedicationEvent as time:ProperInterval.

    Args:
        graph: RDF graph to write to.
        med_data: Dictionary with keys: stay_id, starttime, endtime, label,
                  amount, amountuom, rate, rateuom.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = med_data["stay_id"]
    starttime = med_data["starttime"]
    drug_slug = _slugify(med_data["label"])
    timestamp_str = _timestamp_slug(starttime)

    event_uri = MIMIC_NS[f"ICPMED-{stay_id}-{drug_slug}-{timestamp_str}"]

    graph.add((event_uri, RDF.type, MIMIC_NS.ICPMedicationEvent))
    graph.add((event_uri, RDF.type, TIME_NS.ProperInterval))

    # Temporal bounds
    begin_uri = MIMIC_NS[f"ICPMEDBegin_{stay_id}_{drug_slug}_{timestamp_str}"]
    _write_instant(graph, begin_uri, starttime)
    graph.add((event_uri, TIME_NS.hasBeginning, begin_uri))

    if med_data.get("endtime") is not None:
        end_uri = MIMIC_NS[f"ICPMEDEnd_{stay_id}_{drug_slug}_{timestamp_str}"]
        _write_instant(graph, end_uri, med_data["endtime"])
        graph.add((event_uri, TIME_NS.hasEnd, end_uri))

    graph.add((event_uri, MIMIC_NS.hasDrugName, Literal(med_data["label"], datatype=XSD.string)))
    if med_data.get("amount") is not None:
        graph.add((event_uri, MIMIC_NS.hasAmount, Literal(med_data["amount"], datatype=XSD.decimal)))
    if med_data.get("rate") is not None:
        graph.add((event_uri, MIMIC_NS.hasRate, Literal(med_data["rate"], datatype=XSD.decimal)))
    if med_data.get("amountuom"):
        graph.add((event_uri, MIMIC_NS.hasAmountUnit, Literal(med_data["amountuom"], datatype=XSD.string)))
    if med_data.get("rateuom"):
        graph.add((event_uri, MIMIC_NS.hasRateUnit, Literal(med_data["rateuom"], datatype=XSD.string)))

    # Link to ICU stay
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    icu_day_uri = _assign_event_to_icu_day(starttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri


def write_map_event(
    graph: Graph,
    map_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[tuple[URIRef, datetime, datetime]],
) -> URIRef:
    """Create MAPEvent (Mean Arterial Pressure) as time:Instant.

    Args:
        graph: RDF graph to write to.
        map_data: Dictionary with keys: stay_id, charttime, valuenum, itemid, label.
        icu_stay_uri: URIRef for the ICU stay.
        icu_day_metadata: List of (day_uri, begin, end) tuples.

    Returns:
        URIRef for the created event.
    """
    stay_id = map_data["stay_id"]
    charttime = map_data["charttime"]
    timestamp_str = _timestamp_slug(charttime)
    itemid = map_data["itemid"]

    event_uri = MIMIC_NS[f"MAP-{stay_id}-{itemid}-{timestamp_str}"]

    graph.add((event_uri, RDF.type, MIMIC_NS.MAPEvent))
    graph.add((event_uri, RDF.type, TIME_NS.Instant))

    ts_literal = charttime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    graph.add((event_uri, TIME_NS.inXSDDateTimeStamp, Literal(ts_literal, datatype=XSD.dateTimeStamp)))

    graph.add((event_uri, MIMIC_NS.hasValue, Literal(map_data["valuenum"], datatype=XSD.decimal)))
    graph.add((event_uri, MIMIC_NS.hasItemId, Literal(itemid, datatype=XSD.integer)))

    # Method attribute: arterial vs cuff
    method = "arterial" if itemid == 220052 else "cuff"
    graph.add((event_uri, MIMIC_NS.hasMeasurementMethod, Literal(method, datatype=XSD.string)))

    # Link to ICU stay
    graph.add((event_uri, MIMIC_NS.associatedWithICUStay, icu_stay_uri))
    graph.add((icu_stay_uri, MIMIC_NS.hasICUStayEvent, event_uri))

    icu_day_uri = _assign_event_to_icu_day(charttime, icu_day_metadata)
    if icu_day_uri:
        graph.add((event_uri, MIMIC_NS.associatedWithICUDay, icu_day_uri))
        graph.add((icu_day_uri, MIMIC_NS.hasICUDayEvent, event_uri))

    return event_uri
