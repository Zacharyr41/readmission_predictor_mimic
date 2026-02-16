"""Tests for the Oxigraph disk-backed graph utilities."""

from pathlib import Path

import pytest
from rdflib import RDF, RDFS, Literal, URIRef

from src.graph_construction.disk_graph import (
    bind_namespaces,
    close_disk_graph,
    disk_graph,
    open_disk_graph,
)
from src.graph_construction.ontology import MIMIC_NS, SNOMED_NS, TIME_NS


EX = URIRef("http://example.org/")


def _triple(i: int = 0):
    """Return a deterministic test triple."""
    return (URIRef(f"{EX}s{i}"), URIRef(f"{EX}p{i}"), URIRef(f"{EX}o{i}"))


# ── TestOpenDiskGraph ────────────────────────────────────────────────────────


class TestOpenDiskGraph:
    def test_creates_store_directory(self, tmp_path: Path):
        store = tmp_path / "store"
        g = open_disk_graph(store)
        try:
            assert store.exists()
            assert store.is_dir()
        finally:
            close_disk_graph(g)

    def test_add_and_retrieve_triple(self, tmp_path: Path):
        g = open_disk_graph(tmp_path / "store")
        try:
            t = _triple()
            g.add(t)
            assert t in g
            assert len(g) == 1
        finally:
            close_disk_graph(g)

    def test_data_persists_after_close_reopen(self, tmp_path: Path):
        store = tmp_path / "store"
        t = _triple()

        g = open_disk_graph(store)
        g.add(t)
        close_disk_graph(g)

        g2 = open_disk_graph(store, create=False)
        try:
            assert t in g2
            assert len(g2) == 1
        finally:
            close_disk_graph(g2)

    def test_len_returns_triple_count(self, tmp_path: Path):
        g = open_disk_graph(tmp_path / "store")
        try:
            for i in range(100):
                g.add(_triple(i))
            assert len(g) == 100
        finally:
            close_disk_graph(g)


# ── TestDiskGraphContextManager ──────────────────────────────────────────────


class TestDiskGraphContextManager:
    def test_explicit_path_creates_store(self, tmp_path: Path):
        store = tmp_path / "ctx_store"
        with disk_graph(store) as g:
            g.add(_triple())
        assert store.exists()

    def test_none_path_uses_temp_directory(self):
        with disk_graph() as g:
            g.add(_triple())
            assert len(g) == 1
        # No error — temp dir cleaned up automatically

    def test_graph_closed_after_exit(self, tmp_path: Path):
        store = tmp_path / "ctx_store"
        with disk_graph(store) as g:
            g.add(_triple())

        # Re-open to verify data persists
        g2 = open_disk_graph(store, create=False)
        try:
            assert _triple() in g2
        finally:
            close_disk_graph(g2)


# ── TestBindNamespaces ───────────────────────────────────────────────────────


class TestBindNamespaces:
    def test_binds_all_standard_prefixes(self, tmp_path: Path):
        g = open_disk_graph(tmp_path / "ns_store")
        try:
            bind_namespaces(g)
            ns_map = dict(g.namespaces())
            assert "mimic" in ns_map
            assert "time" in ns_map
            assert "sct" in ns_map
        finally:
            close_disk_graph(g)

    def test_sparql_prefix_resolution_after_bind(self, tmp_path: Path):
        g = open_disk_graph(tmp_path / "ns_store")
        try:
            bind_namespaces(g)
            patient = MIMIC_NS["Patient_1"]
            g.add((patient, RDF.type, MIMIC_NS.Patient))
            results = list(
                g.query("SELECT ?p WHERE { ?p rdf:type mimic:Patient }")
            )
            assert len(results) == 1
            assert results[0][0] == patient
        finally:
            close_disk_graph(g)


# ── TestOxigraphApiCompatibility ─────────────────────────────────────────────


class TestOxigraphApiCompatibility:
    @pytest.fixture()
    def g(self, tmp_path: Path):
        graph = open_disk_graph(tmp_path / "compat_store")
        bind_namespaces(graph)
        yield graph
        close_disk_graph(graph)

    def test_triples_pattern_iteration(self, g):
        p1 = MIMIC_NS["Patient_1"]
        p2 = MIMIC_NS["Patient_2"]
        g.add((p1, RDF.type, MIMIC_NS.Patient))
        g.add((p2, RDF.type, MIMIC_NS.Patient))
        g.add((p1, MIMIC_NS.hasSubjectId, Literal(1)))

        matches = list(g.triples((None, RDF.type, None)))
        assert len(matches) == 2

    def test_full_iteration(self, g):
        for i in range(5):
            g.add(_triple(i))
        all_triples = list(g)
        assert len(all_triples) == 5

    def test_sparql_select(self, g):
        g.add((MIMIC_NS["Patient_1"], RDF.type, MIMIC_NS.Patient))
        g.add((MIMIC_NS["Patient_1"], MIMIC_NS.hasSubjectId, Literal(42)))

        results = list(g.query("""
            SELECT ?sid WHERE {
                ?p rdf:type mimic:Patient ;
                   mimic:hasSubjectId ?sid .
            }
        """))
        assert len(results) == 1
        assert int(results[0][0]) == 42

    def test_sparql_ask(self, g):
        g.add((MIMIC_NS["Patient_1"], RDF.type, MIMIC_NS.Patient))
        result = bool(g.query("ASK { ?p rdf:type mimic:Patient }"))
        assert result is True

    def test_sparql_group_by_count(self, g):
        for i in range(3):
            g.add((MIMIC_NS[f"Patient_{i}"], RDF.type, MIMIC_NS.Patient))
        results = list(g.query("""
            SELECT ?type (COUNT(?e) AS ?cnt)
            WHERE { ?e rdf:type ?type }
            GROUP BY ?type
        """))
        assert len(results) == 1
        assert int(results[0][1]) == 3

    def test_serialize_and_parse_ntriples(self, g):
        t = (MIMIC_NS["Patient_1"], RDF.type, MIMIC_NS.Patient)
        g.add(t)

        nt_data = g.serialize(format="nt")

        from rdflib import Graph as MemGraph

        g2 = MemGraph()
        g2.parse(data=nt_data, format="nt")
        assert t in g2
