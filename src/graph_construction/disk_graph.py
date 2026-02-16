"""Disk-backed RDF graph via Oxigraph.

Provides utilities to open, close, and manage Oxigraph-backed rdflib graphs
that store triples on disk (RocksDB) instead of in memory, enabling graphs
with hundreds of millions of triples without OOM.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef

from src.graph_construction.ontology import MIMIC_NS, TIME_NS, SNOMED_NS

# Fixed graph identifier so that reopening the same store finds previously
# added triples (Oxigraph is a quad store; rdflib uses the Graph.identifier
# as the named-graph context).
_DEFAULT_GRAPH_ID = URIRef("urn:x-rdflib:default")


def open_disk_graph(store_path: Path | str, create: bool = True) -> Graph:
    """Open an Oxigraph-backed rdflib graph stored on disk.

    Args:
        store_path: Directory for the RocksDB store.
        create: If True, create the directory if it doesn't exist.

    Returns:
        An rdflib Graph backed by an Oxigraph store.
    """
    store_path = Path(store_path)
    if create:
        store_path.mkdir(parents=True, exist_ok=True)
    graph = Graph(store="Oxigraph", identifier=_DEFAULT_GRAPH_ID)
    graph.open(str(store_path))
    return graph


def close_disk_graph(graph: Graph) -> None:
    """Close an Oxigraph-backed graph, releasing resources.

    Args:
        graph: An rdflib Graph with an Oxigraph store.
    """
    graph.close()


@contextmanager
def disk_graph(
    store_path: Path | str | None = None,
) -> Generator[Graph, None, None]:
    """Context manager for a temporary or explicit disk-backed graph.

    If *store_path* is ``None``, a temporary directory is created and cleaned up
    on exit.  If a path is given, it persists after exit but the store is still
    closed.

    Args:
        store_path: Optional directory for the store.  ``None`` → temp dir.

    Yields:
        An open Oxigraph-backed rdflib Graph.
    """
    tmp_dir = None
    if store_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="oxigraph_"))
        store_path = tmp_dir
    else:
        store_path = Path(store_path)

    graph = open_disk_graph(store_path)
    try:
        yield graph
    finally:
        close_disk_graph(graph)
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def bind_namespaces(graph: Graph) -> None:
    """Bind standard project namespaces to the graph.

    NTriples format does not preserve namespace bindings, so this must be
    called after opening / parsing into a disk graph if SPARQL queries with
    prefixed names are needed.

    Args:
        graph: An rdflib Graph (any backend).
    """
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("owl", OWL)
    graph.bind("xsd", XSD)
    graph.bind("time", TIME_NS)
    graph.bind("mimic", MIMIC_NS)
    graph.bind("sct", SNOMED_NS)
