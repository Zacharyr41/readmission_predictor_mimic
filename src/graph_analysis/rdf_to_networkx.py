"""RDF to NetworkX conversion for graph structure analysis.

This module provides functions to convert RDF graphs to NetworkX directed graphs
for structural analysis (degree distribution, connected components, etc.).
"""

from collections import defaultdict

import networkx as nx
from rdflib import Graph, RDF, URIRef, Literal


def _get_local_name(uri: URIRef) -> str:
    """Extract the local name from a URI.

    Args:
        uri: An RDF URIRef.

    Returns:
        The local name (fragment or last path component).
    """
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.rsplit("/", 1)[-1]


def rdf_to_networkx(rdf_graph: Graph, include_literals: bool = False) -> nx.DiGraph:
    """Convert RDF triples to NetworkX directed graph.

    Nodes are created from URI subjects/objects. Literals are excluded by default.
    The rdf:type predicate is used to populate node type attributes, not edges.

    Args:
        rdf_graph: An rdflib Graph to convert.
        include_literals: If True, include literal values as nodes. Default False.

    Returns:
        A NetworkX DiGraph with:
        - Nodes: URI local names (e.g., PA-100, HA-200)
        - Node attrs: uri (full URI), types (list of type local names), node_kind
        - Edges: predicates (excluding rdf:type)
        - Edge attrs: predicate (full URI), predicate_local (local name)
    """
    nx_graph = nx.DiGraph()

    # First pass: collect all node types
    node_types: dict[str, list[str]] = defaultdict(list)
    for s, p, o in rdf_graph.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            subject_local = _get_local_name(s)
            type_local = _get_local_name(o)
            if type_local not in node_types[subject_local]:
                node_types[subject_local].append(type_local)

    # Second pass: create nodes and edges
    for s, p, o in rdf_graph:
        # Skip rdf:type triples (already processed for node types)
        if p == RDF.type:
            continue

        # Skip if subject is not a URI
        if not isinstance(s, URIRef):
            continue

        subject_local = _get_local_name(s)

        # Add subject node if not exists
        if subject_local not in nx_graph:
            nx_graph.add_node(
                subject_local,
                uri=str(s),
                types=node_types.get(subject_local, []),
                node_kind="uri",
            )

        # Handle object based on type
        if isinstance(o, URIRef):
            object_local = _get_local_name(o)

            # Add object node if not exists
            if object_local not in nx_graph:
                nx_graph.add_node(
                    object_local,
                    uri=str(o),
                    types=node_types.get(object_local, []),
                    node_kind="uri",
                )

            # Add edge
            predicate_local = _get_local_name(p)
            nx_graph.add_edge(
                subject_local,
                object_local,
                predicate=str(p),
                predicate_local=predicate_local,
            )

        elif isinstance(o, Literal) and include_literals:
            # Create a unique literal node identifier
            literal_id = f"lit_{hash(str(o))}"

            # Add literal node if not exists
            if literal_id not in nx_graph:
                nx_graph.add_node(
                    literal_id,
                    uri=None,
                    types=[],
                    node_kind="literal",
                    value=str(o),
                    datatype=str(o.datatype) if o.datatype else None,
                )

            # Add edge to literal
            predicate_local = _get_local_name(p)
            nx_graph.add_edge(
                subject_local,
                literal_id,
                predicate=str(p),
                predicate_local=predicate_local,
            )

    return nx_graph
