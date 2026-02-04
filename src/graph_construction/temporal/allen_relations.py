"""Allen temporal relation computation for clinical event graphs.

This module provides functions to compute Allen interval algebra relations
between temporal events within ICU stays and add them as RDF triples.
"""

from datetime import datetime

from rdflib import Graph, URIRef

from src.graph_construction.ontology import MIMIC_NS, TIME_NS


# OWL-Time predicate mapping for Allen relations
ALLEN_PREDICATES = {
    "before": TIME_NS.before,
    "during": TIME_NS.inside,
    "overlaps": TIME_NS.intervalOverlaps,
    "meets": TIME_NS.intervalMeets,
    "starts": TIME_NS.intervalStarts,
    "finishes": TIME_NS.intervalFinishes,
}


def _classify_allen_relation(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> str | None:
    """Pure function: classify Allen relation between two intervals.

    Allen's interval algebra defines 13 possible relations between two temporal
    intervals. This function implements the 6 most clinically relevant relations
    for modeling event sequences in ICU stays.

    Relation diagrams (time flows left to right):
        before:   A---|     |---B      A completely precedes B
        meets:    A---|B--------       A ends exactly when B starts
        overlaps: A------|             A starts before B, ends during B
                     |---B----
        starts:   |A---|               Same start, A ends before B
                  |----B----|
        during:      |--A--|           A entirely contained within B
                  |----B----|
        finishes:        |---A|        Same end, A starts after B
                  |----B----|

    For point events (instants), start == end, and the function treats them
    as zero-duration intervals.

    Args:
        a_start: Start time of interval A.
        a_end: End time of interval A (equals a_start for instants).
        b_start: Start time of interval B.
        b_end: End time of interval B (equals b_start for instants).

    Returns:
        One of: "before", "during", "overlaps", "meets", "starts", "finishes",
        or None if no supported relation applies (e.g., A equals B, or
        inverse relations like "after" which are handled by swapping A and B).

    Example:
        >>> from datetime import datetime
        >>> t1 = datetime(2024, 1, 1, 10, 0)
        >>> t2 = datetime(2024, 1, 1, 11, 0)
        >>> t3 = datetime(2024, 1, 1, 12, 0)
        >>> _classify_allen_relation(t1, t2, t2, t3)  # A meets B
        'meets'
    """
    # A ends before B starts -> before
    # Timeline: A---|     |---B
    if a_end < b_start:
        return "before"

    # A ends exactly when B starts (and A is a proper interval) -> meets
    # Timeline: A---|B--------
    # Note: Requires A to be a proper interval (not instant) to avoid ambiguity
    if a_end == b_start and a_start < b_start:
        return "meets"

    # A starts before B, A ends during B -> overlaps
    # Timeline: A starts, then B starts, then A ends, then B ends
    if a_start < b_start < a_end < b_end:
        return "overlaps"

    # Same start, A ends first -> starts
    # Both intervals begin at the same moment, but A finishes earlier
    if a_start == b_start and a_end < b_end:
        return "starts"

    # A entirely within B (proper intervals) -> during
    # A starts after B starts AND A ends before B ends
    if b_start < a_start and a_end < b_end:
        return "during"

    # A is an instant occurring within interval B -> during
    # Special case for point events (e.g., a lab result during an ICU day)
    if a_start == a_end and b_start < a_start < b_end:
        return "during"

    # Same end, A starts later -> finishes
    # Both intervals end at the same moment, but A began later
    if a_end == b_end and a_start > b_start:
        return "finishes"

    # No supported relation (e.g., equals, or inverse relations)
    # Inverse relations are handled by calling this function with swapped args
    return None


def _get_temporal_bounds(
    graph: Graph, event_uri: URIRef, event_type: str
) -> tuple[datetime, datetime] | None:
    """Extract start and end times from a temporal entity.

    Args:
        graph: RDF graph to query.
        event_uri: URI of the temporal event.
        event_type: "instant" or "interval".

    Returns:
        Tuple of (start_datetime, end_datetime) or None if not found.
        For instants, start == end.
    """
    if event_type == "instant":
        # Query for instant timestamp
        query = f"""
        SELECT ?timestamp
        WHERE {{
            <{event_uri}> time:inXSDDateTimeStamp ?timestamp .
        }}
        """
        results = list(graph.query(query))
        if results:
            timestamp_str = str(results[0][0])
            # Parse ISO timestamp (remove trailing Z if present)
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1]
            dt = datetime.fromisoformat(timestamp_str)
            return (dt, dt)
    else:
        # Query for interval bounds
        query = f"""
        SELECT ?beginTime ?endTime
        WHERE {{
            <{event_uri}> time:hasBeginning ?begin ;
                          time:hasEnd ?end .
            ?begin time:inXSDDateTimeStamp ?beginTime .
            ?end time:inXSDDateTimeStamp ?endTime .
        }}
        """
        results = list(graph.query(query))
        if results:
            begin_str = str(results[0][0])
            end_str = str(results[0][1])
            if begin_str.endswith("Z"):
                begin_str = begin_str[:-1]
            if end_str.endswith("Z"):
                end_str = end_str[:-1]
            begin_dt = datetime.fromisoformat(begin_str)
            end_dt = datetime.fromisoformat(end_str)
            return (begin_dt, end_dt)

    return None


def compute_allen_relations(graph: Graph, icu_stay_uri: URIRef) -> int:
    """Compute Allen temporal relations for events in a single ICU stay.

    Queries all temporal entities (time:Instant and time:ProperInterval) associated
    with the given ICU stay, then computes and adds Allen relation triples for
    each pair of events.

    Args:
        graph: RDF graph containing the events.
        icu_stay_uri: URI of the ICU stay to process.

    Returns:
        Count of relation triples added.
    """
    # Query all temporal entities for this ICU stay
    query = f"""
    SELECT ?event ?eventType
    WHERE {{
        ?event mimic:associatedWithICUStay <{icu_stay_uri}> .
        {{
            ?event rdf:type time:Instant .
            BIND("instant" AS ?eventType)
        }}
        UNION
        {{
            ?event rdf:type time:ProperInterval .
            BIND("interval" AS ?eventType)
        }}
    }}
    """

    results = list(graph.query(query))

    # Build list of (uri, type, start, end)
    events = []
    for row in results:
        event_uri = row[0]
        event_type = str(row[1])
        bounds = _get_temporal_bounds(graph, event_uri, event_type)
        if bounds:
            events.append((event_uri, event_type, bounds[0], bounds[1]))

    # Sort by start time
    events.sort(key=lambda x: x[2])

    # Compute relations for each pair
    count = 0
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a_uri, a_type, a_start, a_end = events[i]
            b_uri, b_type, b_start, b_end = events[j]

            # Classify A -> B relation
            relation_ab = _classify_allen_relation(a_start, a_end, b_start, b_end)
            if relation_ab and relation_ab in ALLEN_PREDICATES:
                predicate = ALLEN_PREDICATES[relation_ab]
                graph.add((a_uri, predicate, b_uri))
                count += 1

            # Classify B -> A relation (inverse)
            relation_ba = _classify_allen_relation(b_start, b_end, a_start, a_end)
            if relation_ba and relation_ba in ALLEN_PREDICATES:
                predicate = ALLEN_PREDICATES[relation_ba]
                graph.add((b_uri, predicate, a_uri))
                count += 1

    return count


def compute_allen_relations_for_patient(graph: Graph, patient_uri: URIRef) -> int:
    """Compute Allen relations for all ICU stays of a patient.

    Iterates over all ICU stays linked to the patient through their admissions
    and calls compute_allen_relations for each.

    Args:
        graph: RDF graph containing the patient data.
        patient_uri: URI of the patient.

    Returns:
        Total count of relation triples added across all ICU stays.
    """
    # Query all ICU stays for this patient
    query = f"""
    SELECT ?stay
    WHERE {{
        <{patient_uri}> mimic:hasAdmission ?adm .
        ?adm mimic:containsICUStay ?stay .
    }}
    """

    results = list(graph.query(query))

    total_count = 0
    for row in results:
        icu_stay_uri = row[0]
        count = compute_allen_relations(graph, icu_stay_uri)
        total_count += count

    return total_count
