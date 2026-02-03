"""Graph analysis module for clinical knowledge graphs.

This module provides tools for:
- Converting RDF graphs to NetworkX for structural analysis
- Computing graph metrics (node/edge counts, degree distribution, connected components)
- Temporal density analysis for ICU events
- Optional Neo4j integration for graph database import and Cypher queries
"""

from src.graph_analysis.rdf_to_networkx import rdf_to_networkx
from src.graph_analysis.analysis import (
    count_nodes_by_type,
    count_edges_by_type,
    degree_distribution,
    connected_components,
    temporal_density,
    generate_analysis_report,
)
from src.graph_analysis.neo4j_import import (
    check_neo4j_connection,
    import_rdf_to_neo4j,
    query_neo4j,
    clear_neo4j_database,
)

__all__ = [
    # RDF to NetworkX conversion
    "rdf_to_networkx",
    # Analysis functions
    "count_nodes_by_type",
    "count_edges_by_type",
    "degree_distribution",
    "connected_components",
    "temporal_density",
    "generate_analysis_report",
    # Neo4j integration
    "check_neo4j_connection",
    "import_rdf_to_neo4j",
    "query_neo4j",
    "clear_neo4j_database",
]
