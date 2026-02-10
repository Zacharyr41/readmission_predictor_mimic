"""Tests for ontology loading and extension (TDD Red Phase).

Test suite for Layer 2A: Ontology Extension and RDF Graph Initialization.
"""

import pytest
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD


# Namespaces matching the base ontology
MIMIC_NS = Namespace("http://www.cnam.fr/MIMIC4-ICU-BSI/V1#")
TIME_NS = Namespace("http://www.w3.org/2006/time#")

# Path to ontology files
ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


class TestBaseOntology:
    """Tests for loading the base ontology."""

    def test_base_ontology_loads(self) -> None:
        """Parse base_ontology.rdf succeeds with >50 triples."""
        base_path = ONTOLOGY_DIR / "base_ontology.rdf"
        assert base_path.exists(), f"Base ontology not found at {base_path}"

        g = Graph()
        g.parse(base_path)

        # Base ontology should have substantial content
        triple_count = len(g)
        assert triple_count > 50, f"Expected >50 triples, got {triple_count}"


class TestExtendedOntology:
    """Tests for the extended ontology with HospitalAdmission class."""

    @pytest.fixture
    def extended_graph(self) -> Graph:
        """Load extended ontology into a graph."""
        extended_path = ONTOLOGY_DIR / "extended_ontology.rdf"
        assert extended_path.exists(), f"Extended ontology not found at {extended_path}"

        g = Graph()
        g.parse(extended_path)
        return g

    def test_extended_ontology_has_admission_class(self, extended_graph: Graph) -> None:
        """HospitalAdmission is owl:Class and rdfs:subClassOf time:Interval."""
        # Check HospitalAdmission is an OWL class
        is_class = (MIMIC_NS.HospitalAdmission, RDF.type, OWL.Class) in extended_graph
        assert is_class, "HospitalAdmission should be an owl:Class"

        # Check HospitalAdmission is subclass of time:Interval
        is_subclass = (
            MIMIC_NS.HospitalAdmission,
            RDFS.subClassOf,
            TIME_NS.Interval,
        ) in extended_graph
        assert is_subclass, "HospitalAdmission should be rdfs:subClassOf time:Interval"

    def test_extended_ontology_has_readmission_properties(
        self, extended_graph: Graph
    ) -> None:
        """Extended ontology has readmission-related DatatypeProperties."""
        # Required datatype properties with their expected ranges
        required_props = {
            MIMIC_NS.readmittedWithin30Days: XSD.boolean,
            MIMIC_NS.readmittedWithin60Days: XSD.boolean,
            MIMIC_NS.hasDischargeLocation: XSD.string,
            MIMIC_NS.hasAdmissionType: XSD.string,
        }

        for prop, expected_range in required_props.items():
            # Check property exists and is a DatatypeProperty
            is_datatype_prop = (prop, RDF.type, OWL.DatatypeProperty) in extended_graph
            assert is_datatype_prop, f"{prop} should be an owl:DatatypeProperty"

            # Check range
            has_range = (prop, RDFS.range, expected_range) in extended_graph
            assert has_range, f"{prop} should have rdfs:range {expected_range}"

    def test_extended_ontology_has_followedby(self, extended_graph: Graph) -> None:
        """ObjectProperty followedBy with domain/range HospitalAdmission."""
        prop = MIMIC_NS.followedBy

        # Check it's an ObjectProperty
        is_obj_prop = (prop, RDF.type, OWL.ObjectProperty) in extended_graph
        assert is_obj_prop, "followedBy should be an owl:ObjectProperty"

        # Check domain is HospitalAdmission
        has_domain = (prop, RDFS.domain, MIMIC_NS.HospitalAdmission) in extended_graph
        assert has_domain, "followedBy should have domain HospitalAdmission"

        # Check range is HospitalAdmission
        has_range = (prop, RDFS.range, MIMIC_NS.HospitalAdmission) in extended_graph
        assert has_range, "followedBy should have range HospitalAdmission"


class TestGraphInitializer:
    """Tests for the graph initialization function."""

    def test_graph_initializer_creates_bound_graph(self) -> None:
        """initialize_graph() returns a Graph with required namespaces bound."""
        from src.graph_construction.ontology import initialize_graph

        g = initialize_graph(ONTOLOGY_DIR)

        # Check required namespaces are bound
        namespaces = dict(g.namespaces())

        required_ns = ["time", "xsd", "owl", "rdf", "rdfs", "mimic"]
        for ns in required_ns:
            assert ns in namespaces, f"Namespace '{ns}' should be bound"

    def test_graph_initializer_loads_ontology(self) -> None:
        """initialize_graph() loads triples from both base and extended ontologies."""
        from src.graph_construction.ontology import initialize_graph

        g = initialize_graph(ONTOLOGY_DIR)

        # Should have content from base ontology (e.g., Patient class)
        has_patient = (MIMIC_NS.Patient, RDF.type, OWL.Class) in g
        assert has_patient, "Graph should contain Patient class from base ontology"

        # Should have content from extended ontology (e.g., HospitalAdmission)
        has_admission = (MIMIC_NS.HospitalAdmission, RDF.type, OWL.Class) in g
        assert (
            has_admission
        ), "Graph should contain HospitalAdmission from extended ontology"

        # Check total triple count is substantial
        triple_count = len(g)
        assert (
            triple_count > 50
        ), f"Combined graph should have >50 triples, got {triple_count}"

    def test_snomed_namespace_bound(self) -> None:
        """initialize_graph() binds the SNOMED-CT namespace as 'sct'."""
        from src.graph_construction.ontology import initialize_graph

        g = initialize_graph(ONTOLOGY_DIR)
        namespaces = dict(g.namespaces())
        assert "sct" in namespaces, "SNOMED namespace 'sct' should be bound"
        assert str(namespaces["sct"]) == "http://snomed.info/id/"
