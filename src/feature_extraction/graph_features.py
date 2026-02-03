"""Graph structure feature extraction from RDF graphs.

This module extracts features from the temporal and structural properties
of the clinical knowledge graph, including temporal relation counts and
graph topology metrics.
"""

import pandas as pd
import numpy as np
import networkx as nx
from rdflib import Graph, URIRef

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_analysis.rdf_to_networkx import rdf_to_networkx


# Allen temporal relation predicates from OWL-Time
TEMPORAL_PREDICATES = [
    str(TIME_NS.before),
    str(TIME_NS.inside),
    str(TIME_NS.intervalOverlaps),
    str(TIME_NS.intervalMeets),
    str(TIME_NS.intervalStarts),
    str(TIME_NS.intervalFinishes),
]


def extract_temporal_features(graph: Graph) -> pd.DataFrame:
    """Extract temporal relation features for each admission.

    Counts Allen temporal relations between events within each ICU stay.

    Args:
        graph: RDF graph containing temporal events and relations.

    Returns:
        DataFrame with columns: hadm_id, events_per_icu_day, num_before_relations,
                                num_during_relations, total_temporal_edges
    """
    # Get all admissions and their ICU stays
    admission_query = """
    SELECT ?hadmId ?icuStay
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:containsICUStay ?icuStay .
    }
    """

    admission_results = list(graph.query(admission_query))

    if not admission_results:
        return pd.DataFrame(columns=[
            "hadm_id", "events_per_icu_day", "num_before_relations",
            "num_during_relations", "total_temporal_edges"
        ])

    # Build mapping of admission to ICU stays
    admission_stays = {}
    for row in admission_results:
        hadm_id = int(row[0])
        icu_stay = row[1]
        if hadm_id not in admission_stays:
            admission_stays[hadm_id] = []
        admission_stays[hadm_id].append(icu_stay)

    # Count temporal relations per admission
    data = []
    for hadm_id, icu_stays in admission_stays.items():
        total_before = 0
        total_during = 0
        total_temporal = 0
        total_events = 0
        total_icu_days = 0

        for icu_stay_uri in icu_stays:
            # Count events for this ICU stay
            event_query = f"""
            SELECT (COUNT(?event) AS ?count)
            WHERE {{
                ?event mimic:associatedWithICUStay <{icu_stay_uri}> .
            }}
            """
            event_results = list(graph.query(event_query))
            if event_results:
                total_events += int(event_results[0][0])

            # Count ICU days
            day_query = f"""
            SELECT (COUNT(?day) AS ?count)
            WHERE {{
                <{icu_stay_uri}> mimic:hasICUDay ?day .
            }}
            """
            day_results = list(graph.query(day_query))
            if day_results:
                total_icu_days += int(day_results[0][0])

            # Count temporal relations
            for predicate in TEMPORAL_PREDICATES:
                rel_query = f"""
                SELECT (COUNT(*) AS ?count)
                WHERE {{
                    ?eventA mimic:associatedWithICUStay <{icu_stay_uri}> ;
                            <{predicate}> ?eventB .
                    ?eventB mimic:associatedWithICUStay <{icu_stay_uri}> .
                }}
                """
                rel_results = list(graph.query(rel_query))
                if rel_results:
                    count = int(rel_results[0][0])
                    total_temporal += count

                    if predicate == str(TIME_NS.before):
                        total_before += count
                    elif predicate == str(TIME_NS.inside):
                        total_during += count

        events_per_day = total_events / total_icu_days if total_icu_days > 0 else 0.0

        data.append({
            "hadm_id": hadm_id,
            "events_per_icu_day": events_per_day,
            "num_before_relations": total_before,
            "num_during_relations": total_during,
            "total_temporal_edges": total_temporal,
        })

    return pd.DataFrame(data)


def _get_patient_subgraph(
    nx_graph: nx.DiGraph,
    admission_node: str
) -> nx.DiGraph:
    """Extract subgraph for a single admission.

    Uses BFS from the admission node to find all connected nodes
    within 3 hops (admission -> ICU stay -> events -> temporal relations).

    Args:
        nx_graph: NetworkX graph of the full RDF graph.
        admission_node: Node ID for the admission (e.g., "HA-101").

    Returns:
        Subgraph containing the admission and all connected nodes.
    """
    if admission_node not in nx_graph:
        return nx.DiGraph()

    # Find all nodes within 3 hops of the admission
    nodes_to_include = {admission_node}

    # BFS to find connected nodes
    frontier = {admission_node}
    for _ in range(3):  # 3 hops
        new_frontier = set()
        for node in frontier:
            # Get successors and predecessors
            if node in nx_graph:
                new_frontier.update(nx_graph.successors(node))
                new_frontier.update(nx_graph.predecessors(node))
        nodes_to_include.update(new_frontier)
        frontier = new_frontier - nodes_to_include

    return nx_graph.subgraph(nodes_to_include).copy()


def extract_graph_structure_features(graph: Graph) -> pd.DataFrame:
    """Extract graph structure features for each admission.

    Converts the RDF graph to NetworkX and computes topology metrics
    for each patient's admission subgraph.

    Args:
        graph: RDF graph containing the clinical data.

    Returns:
        DataFrame with columns: hadm_id, patient_subgraph_nodes,
                                patient_subgraph_edges, patient_subgraph_density,
                                mean_node_degree, max_node_degree
    """
    # Convert full graph to NetworkX
    nx_graph = rdf_to_networkx(graph, include_literals=False)

    # Get all admissions
    admission_query = """
    SELECT ?hadmId
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId .
    }
    """

    admission_results = list(graph.query(admission_query))

    if not admission_results:
        return pd.DataFrame(columns=[
            "hadm_id", "patient_subgraph_nodes", "patient_subgraph_edges",
            "patient_subgraph_density", "mean_node_degree", "max_node_degree"
        ])

    data = []
    for row in admission_results:
        hadm_id = int(row[0])
        admission_node = f"HA-{hadm_id}"

        # Get subgraph for this admission
        subgraph = _get_patient_subgraph(nx_graph, admission_node)

        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        if num_nodes > 1:
            # Density for directed graph
            density = num_edges / (num_nodes * (num_nodes - 1))
            degrees = [d for n, d in subgraph.degree()]
            mean_degree = np.mean(degrees)
            max_degree = np.max(degrees)
        else:
            density = 0.0
            mean_degree = 0.0
            max_degree = 0.0

        data.append({
            "hadm_id": hadm_id,
            "patient_subgraph_nodes": num_nodes,
            "patient_subgraph_edges": num_edges,
            "patient_subgraph_density": density,
            "mean_node_degree": mean_degree,
            "max_node_degree": max_degree,
        })

    return pd.DataFrame(data)
