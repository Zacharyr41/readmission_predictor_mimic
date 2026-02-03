"""Tests for graph structure analysis with NetworkX and optional Neo4j integration.

These tests follow TDD - they define expected behavior for Layer 3 graph analysis.
"""

import pytest
from rdflib import Graph

from src.graph_analysis import (
    rdf_to_networkx,
    count_nodes_by_type,
    count_edges_by_type,
    degree_distribution,
    connected_components,
    temporal_density,
    generate_analysis_report,
    check_neo4j_connection,
    import_rdf_to_neo4j,
    query_neo4j,
)


class TestRdfToNetworkx:
    """Tests for RDF to NetworkX conversion."""

    def test_rdf_to_networkx_conversion(self, synthetic_analysis_graph: Graph) -> None:
        """Test that rdf_to_networkx returns a DiGraph with correct node/edge counts."""
        nx_graph = rdf_to_networkx(synthetic_analysis_graph)

        # Should be a directed graph
        import networkx as nx

        assert isinstance(nx_graph, nx.DiGraph)

        # Should have nodes (URIs become nodes, literals excluded by default)
        assert nx_graph.number_of_nodes() > 0

        # Should have edges (predicates become edges, excluding rdf:type)
        assert nx_graph.number_of_edges() > 0

        # Nodes should have attributes
        for node in list(nx_graph.nodes())[:1]:
            node_data = nx_graph.nodes[node]
            assert "uri" in node_data
            assert "types" in node_data
            assert "node_kind" in node_data

    def test_rdf_to_networkx_node_attributes(self, synthetic_analysis_graph: Graph) -> None:
        """Test that nodes have correct attributes."""
        nx_graph = rdf_to_networkx(synthetic_analysis_graph)

        # Find a Patient node
        patient_nodes = [
            n for n in nx_graph.nodes() if "PA-" in n
        ]
        assert len(patient_nodes) >= 2

        # Check patient node attributes
        patient_data = nx_graph.nodes[patient_nodes[0]]
        assert "Patient" in patient_data["types"]

    def test_rdf_to_networkx_edge_attributes(self, synthetic_analysis_graph: Graph) -> None:
        """Test that edges have correct attributes."""
        nx_graph = rdf_to_networkx(synthetic_analysis_graph)

        # Should have edges with predicate info
        for u, v, data in list(nx_graph.edges(data=True))[:1]:
            assert "predicate" in data
            assert "predicate_local" in data


class TestNodeCounts:
    """Tests for counting nodes by type."""

    def test_node_count_by_type(self, synthetic_analysis_graph: Graph) -> None:
        """Test counting nodes grouped by RDF type."""
        counts = count_nodes_by_type(synthetic_analysis_graph)

        # Should return a dictionary
        assert isinstance(counts, dict)

        # Should have expected types
        assert "Patient" in counts
        assert "HospitalAdmission" in counts
        assert "ICUStay" in counts
        assert "ICUDay" in counts
        assert "BioMarkerEvent" in counts
        assert "ClinicalSignEvent" in counts

        # Verify expected counts from synthetic data
        assert counts["Patient"] == 2
        assert counts["HospitalAdmission"] == 2
        assert counts["ICUStay"] == 2
        # ICU days span calendar days, so 3-day LOS starting mid-day = 4 calendar days
        assert counts["ICUDay"] == 8  # 4 days per stay * 2 stays
        assert counts["BioMarkerEvent"] == 4  # 2 per stay * 2 stays
        assert counts["ClinicalSignEvent"] == 2  # 1 per stay * 2 stays


class TestEdgeCounts:
    """Tests for counting edges by predicate."""

    def test_edge_count_by_type(self, synthetic_analysis_graph: Graph) -> None:
        """Test counting edges grouped by predicate."""
        counts = count_edges_by_type(synthetic_analysis_graph)

        # Should return a dictionary
        assert isinstance(counts, dict)

        # Should have expected predicates (excluding rdf:type)
        assert "hasAdmission" in counts
        assert "containsICUStay" in counts
        assert "hasICUDay" in counts
        assert "hasICUDayEvent" in counts

        # Verify some expected counts
        assert counts["hasAdmission"] == 2  # 2 patients, 1 admission each
        assert counts["containsICUStay"] == 2  # 2 admissions with 1 ICU stay each
        assert counts["hasICUDay"] == 8  # 2 stays * 4 calendar days each


class TestDegreeDistribution:
    """Tests for degree distribution analysis."""

    def test_degree_distribution(self, synthetic_analysis_graph: Graph) -> None:
        """Test computing degree distribution statistics."""
        stats = degree_distribution(synthetic_analysis_graph)

        # Should return a dict with expected keys
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "median" in stats
        assert "max" in stats
        assert "std" in stats

        # All values should be numeric
        assert isinstance(stats["mean"], (int, float))
        assert isinstance(stats["median"], (int, float))
        assert isinstance(stats["max"], (int, float))
        assert isinstance(stats["std"], (int, float))

        # Max degree should be positive
        assert stats["max"] > 0


class TestConnectedComponents:
    """Tests for connected component analysis."""

    def test_connected_components(self, synthetic_analysis_graph: Graph) -> None:
        """Test counting weakly connected components."""
        num_components = connected_components(synthetic_analysis_graph)

        # Should return an integer
        assert isinstance(num_components, int)

        # With 2 independent patient subgraphs, expect >= 2 components
        # (may have more due to ontology triples)
        assert num_components >= 2


class TestTemporalDensity:
    """Tests for temporal density analysis."""

    def test_temporal_density(self, synthetic_analysis_graph: Graph) -> None:
        """Test computing events per ICU day."""
        density = temporal_density(synthetic_analysis_graph)

        # Should return a float
        assert isinstance(density, float)

        # With 6 events over 8 ICU days, expect ~0.75 events/day
        assert 0.5 <= density <= 1.0


class TestAnalysisReport:
    """Tests for markdown report generation."""

    def test_analysis_report_generates_markdown(self, synthetic_analysis_graph: Graph) -> None:
        """Test that generate_analysis_report returns valid markdown."""
        report = generate_analysis_report(synthetic_analysis_graph)

        # Should return a string
        assert isinstance(report, str)

        # Should contain markdown headers
        assert "# " in report
        assert "## Node Counts" in report
        assert "## Edge Counts" in report

        # Should contain some data
        assert "Patient" in report
        assert "HospitalAdmission" in report

    def test_analysis_report_writes_to_file(
        self, synthetic_analysis_graph: Graph, tmp_path
    ) -> None:
        """Test that report can be written to file."""
        from pathlib import Path

        output_path = tmp_path / "analysis_report.md"
        report = generate_analysis_report(synthetic_analysis_graph, output_path=output_path)

        # Should return the report string
        assert isinstance(report, str)

        # File should exist and contain the report
        assert output_path.exists()
        assert output_path.read_text() == report


@pytest.mark.integration
class TestNeo4jIntegration:
    """Integration tests for Neo4j import (skipped if Neo4j unavailable)."""

    @pytest.fixture(autouse=True)
    def check_neo4j(self):
        """Skip tests if Neo4j is not available."""
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"

        if not check_neo4j_connection(uri, user, password):
            pytest.skip("Neo4j not available")

    def test_neo4j_import(
        self, synthetic_analysis_graph: Graph, tmp_path
    ) -> None:
        """Test importing RDF to Neo4j."""
        # Write graph to temp file
        rdf_path = tmp_path / "test_graph.ttl"
        synthetic_analysis_graph.serialize(destination=str(rdf_path), format="turtle")

        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"

        result = import_rdf_to_neo4j(rdf_path, uri, user, password)

        # Should return import statistics
        assert isinstance(result, dict)
        assert "nodes_created" in result or "triples_loaded" in result

    def test_neo4j_cypher_query(self, synthetic_analysis_graph: Graph, tmp_path) -> None:
        """Test executing Cypher queries."""
        # First import the graph
        rdf_path = tmp_path / "test_graph.ttl"
        synthetic_analysis_graph.serialize(destination=str(rdf_path), format="turtle")

        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"

        import_rdf_to_neo4j(rdf_path, uri, user, password)

        # Query for hospital admissions
        cypher = "MATCH (n:HospitalAdmission) RETURN count(n) AS count"
        results = query_neo4j(cypher, uri, user, password)

        # Should return list of dicts
        assert isinstance(results, list)
        if results:
            assert "count" in results[0]
            assert results[0]["count"] == 2
