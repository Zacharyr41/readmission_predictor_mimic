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


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse an ISO timestamp string, stripping trailing 'Z' if present."""
    s = str(ts_str)
    if s.endswith("Z"):
        s = s[:-1]
    return datetime.fromisoformat(s)


def _batch_get_temporal_bounds(
    graph: Graph, icu_stay_uri: URIRef
) -> list[tuple[URIRef, str, datetime, datetime]]:
    """Fetch all event URIs and their temporal bounds in a single SPARQL query.

    Replaces N individual _get_temporal_bounds() calls with one UNION query.

    Args:
        graph: RDF graph to query.
        icu_stay_uri: URI of the ICU stay.

    Returns:
        List of (event_uri, event_type, start_dt, end_dt) tuples.
        Events with missing bounds are excluded.
    """
    query = f"""
    SELECT ?event ?eventType ?timestamp ?beginTime ?endTime
    WHERE {{
        ?event mimic:associatedWithICUStay <{icu_stay_uri}> .
        {{
            ?event rdf:type time:Instant .
            BIND("instant" AS ?eventType)
            ?event time:inXSDDateTimeStamp ?timestamp .
        }}
        UNION
        {{
            ?event rdf:type time:ProperInterval .
            BIND("interval" AS ?eventType)
            ?event time:hasBeginning ?begin ;
                   time:hasEnd ?end .
            ?begin time:inXSDDateTimeStamp ?beginTime .
            ?end time:inXSDDateTimeStamp ?endTime .
        }}
    }}
    """

    results = list(graph.query(query))
    events = []
    for row in results:
        event_uri = row[0]
        event_type = str(row[1])
        if event_type == "instant":
            dt = _parse_timestamp(row[2])
            events.append((event_uri, event_type, dt, dt))
        else:
            begin_dt = _parse_timestamp(row[3])
            end_dt = _parse_timestamp(row[4])
            events.append((event_uri, event_type, begin_dt, end_dt))

    return events


def compute_allen_relations(graph: Graph, icu_stay_uri: URIRef) -> int:
    """Compute Allen temporal relations for events in a single ICU stay.

    Uses three optimizations:
    A) Batch temporal bounds query (single SPARQL instead of N queries)
    B) Early termination for "before" chains (sorted-order pruning)
    C) Skip reverse classification when forward is non-None

    Args:
        graph: RDF graph containing the events.
        icu_stay_uri: URI of the ICU stay to process.

    Returns:
        Count of relation triples added.
    """
    # Optimization A: single batch query
    events = _batch_get_temporal_bounds(graph, icu_stay_uri)

    # Sort by start time
    events.sort(key=lambda x: x[2])

    before_predicate = ALLEN_PREDICATES["before"]

    # Compute relations for each pair
    count = 0
    for i in range(len(events)):
        a_uri, a_type, a_start, a_end = events[i]

        for j in range(i + 1, len(events)):
            b_uri, b_type, b_start, b_end = events[j]

            # Classify A -> B relation
            relation_ab = _classify_allen_relation(a_start, a_end, b_start, b_end)

            if relation_ab:
                graph.add((a_uri, ALLEN_PREDICATES[relation_ab], b_uri))
                count += 1

                # Optimization B: early termination for "before"
                # All remaining events k > j also have start >= b_start > a_end,
                # so they are all "before" A too.
                if relation_ab == "before":
                    for k in range(j + 1, len(events)):
                        graph.add((a_uri, before_predicate, events[k][0]))
                    count += len(events) - j - 1
                    break

                # Optimization C: if forward is non-None, reverse is guaranteed
                # None for sorted pairs (before, meets, overlaps, starts).
            else:
                # Only check reverse when forward is None
                relation_ba = _classify_allen_relation(
                    b_start, b_end, a_start, a_end
                )
                if relation_ba and relation_ba in ALLEN_PREDICATES:
                    graph.add((b_uri, ALLEN_PREDICATES[relation_ba], a_uri))
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
