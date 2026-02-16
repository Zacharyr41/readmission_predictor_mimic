"""Graph structure analysis functions for clinical knowledge graphs.

This module provides functions to analyze RDF graph structure using SPARQL queries
and NetworkX graph metrics.
"""

from pathlib import Path
from statistics import mean, median, stdev

import networkx as nx
from rdflib import Graph, RDF

from src.graph_construction.ontology import MIMIC_NS
from src.graph_analysis.rdf_to_networkx import rdf_to_networkx, _get_local_name


def count_nodes_by_type(rdf_graph: Graph) -> dict[str, int]:
    """Count nodes grouped by RDF type using SPARQL.

    Only counts types from the MIMIC namespace.

    Args:
        rdf_graph: An rdflib Graph to analyze.

    Returns:
        Dictionary mapping type local names to counts.
    """
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>

    SELECT ?type (COUNT(DISTINCT ?node) AS ?count)
    WHERE {
        ?node rdf:type ?type .
        FILTER(STRSTARTS(STR(?type), STR(mimic:)))
    }
    GROUP BY ?type
    ORDER BY DESC(?count)
    """

    results = rdf_graph.query(query)
    counts = {}
    for row in results:
        type_local = _get_local_name(row.type)
        counts[type_local] = int(row["count"])

    return counts


def count_edges_by_type(rdf_graph: Graph) -> dict[str, int]:
    """Count edges grouped by predicate using SPARQL.

    Excludes rdf:type predicates.

    Args:
        rdf_graph: An rdflib Graph to analyze.

    Returns:
        Dictionary mapping predicate local names to counts.
    """
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX time: <http://www.w3.org/2006/time#>

    SELECT ?predicate (COUNT(*) AS ?count)
    WHERE {
        ?s ?predicate ?o .
        FILTER(?predicate != rdf:type)
        FILTER(
            STRSTARTS(STR(?predicate), STR(mimic:)) ||
            STRSTARTS(STR(?predicate), STR(time:))
        )
    }
    GROUP BY ?predicate
    ORDER BY DESC(?count)
    """

    results = rdf_graph.query(query)
    counts = {}
    for row in results:
        pred_local = _get_local_name(row.predicate)
        counts[pred_local] = int(row["count"])

    return counts


def degree_distribution(
    rdf_graph: Graph, nx_graph: nx.DiGraph | None = None
) -> dict[str, float]:
    """Compute degree distribution statistics.

    Args:
        rdf_graph: An rdflib Graph to analyze.
        nx_graph: Optional pre-built NetworkX graph to avoid redundant conversion.

    Returns:
        Dictionary with keys: mean, median, max, std.
    """
    if nx_graph is None:
        nx_graph = rdf_to_networkx(rdf_graph)

    if nx_graph.number_of_nodes() == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0, "std": 0.0}

    # Get degree for each node (in + out for directed graph)
    degrees = [d for _, d in nx_graph.degree()]

    if len(degrees) == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0, "std": 0.0}

    if len(degrees) == 1:
        return {
            "mean": float(degrees[0]),
            "median": float(degrees[0]),
            "max": degrees[0],
            "std": 0.0,
        }

    return {
        "mean": mean(degrees),
        "median": median(degrees),
        "max": max(degrees),
        "std": stdev(degrees),
    }


def connected_components(
    rdf_graph: Graph, nx_graph: nx.DiGraph | None = None
) -> int:
    """Count the number of weakly connected components.

    Args:
        rdf_graph: An rdflib Graph to analyze.
        nx_graph: Optional pre-built NetworkX graph to avoid redundant conversion.

    Returns:
        Number of weakly connected components.
    """
    if nx_graph is None:
        nx_graph = rdf_to_networkx(rdf_graph)

    if nx_graph.number_of_nodes() == 0:
        return 0

    return nx.number_weakly_connected_components(nx_graph)


def temporal_density(rdf_graph: Graph) -> float:
    """Compute average events per ICU day.

    Counts events (BioMarkerEvent, ClinicalSignEvent, etc.) linked to ICU days
    and divides by the number of ICU days.

    Args:
        rdf_graph: An rdflib Graph to analyze.

    Returns:
        Average number of events per ICU day.
    """
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>

    SELECT ?day (COUNT(?event) AS ?eventCount)
    WHERE {
        ?day rdf:type mimic:ICUDay .
        OPTIONAL { ?day mimic:hasICUDayEvent ?event . }
    }
    GROUP BY ?day
    """

    results = list(rdf_graph.query(query))

    if not results:
        return 0.0

    total_events = sum(int(row["eventCount"]) for row in results)
    num_days = len(results)

    if num_days == 0:
        return 0.0

    return total_events / num_days


def generate_analysis_report(
    rdf_graph: Graph, output_path: Path | None = None
) -> tuple[str, nx.DiGraph]:
    """Generate a markdown report with graph analysis.

    Args:
        rdf_graph: An rdflib Graph to analyze.
        output_path: Optional path to write the report. If None, only returns string.

    Returns:
        Tuple of (markdown report string, NetworkX DiGraph built during analysis).
    """
    lines = []

    # Header
    lines.append("# Graph Analysis Report")
    lines.append("")

    # Summary — build nx_graph once, reuse for degree_distribution & connected_components
    nx_graph = rdf_to_networkx(rdf_graph)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Nodes**: {nx_graph.number_of_nodes()}")
    lines.append(f"- **Total Edges**: {nx_graph.number_of_edges()}")
    lines.append(f"- **Total RDF Triples**: {len(rdf_graph)}")
    lines.append("")

    # Node Counts
    lines.append("## Node Counts")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|------|-------|")
    node_counts = count_nodes_by_type(rdf_graph)
    for type_name, count in sorted(node_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {type_name} | {count} |")
    lines.append("")

    # Edge Counts
    lines.append("## Edge Counts")
    lines.append("")
    lines.append("| Predicate | Count |")
    lines.append("|-----------|-------|")
    edge_counts = count_edges_by_type(rdf_graph)
    for pred_name, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {pred_name} | {count} |")
    lines.append("")

    # Degree Distribution — pass cached nx_graph
    lines.append("## Degree Distribution")
    lines.append("")
    deg_stats = degree_distribution(rdf_graph, nx_graph=nx_graph)
    lines.append(f"- **Mean**: {deg_stats['mean']:.2f}")
    lines.append(f"- **Median**: {deg_stats['median']:.2f}")
    lines.append(f"- **Max**: {deg_stats['max']}")
    lines.append(f"- **Std Dev**: {deg_stats['std']:.2f}")
    lines.append("")

    # Connected Components — pass cached nx_graph
    lines.append("## Connected Components")
    lines.append("")
    num_components = connected_components(rdf_graph, nx_graph=nx_graph)
    lines.append(f"- **Weakly Connected Components**: {num_components}")
    lines.append("")

    # Temporal Density
    lines.append("## Temporal Density")
    lines.append("")
    density = temporal_density(rdf_graph)
    lines.append(f"- **Events per ICU Day**: {density:.2f}")
    lines.append("")

    report = "\n".join(lines)

    # Write to file if path provided
    if output_path:
        output_path.write_text(report)

    return report, nx_graph
