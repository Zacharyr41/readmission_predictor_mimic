"""Ontology loading and graph initialization for hospital readmission prediction.

This module provides namespace constants and functions for loading the MIMIC-IV
temporal ontology extended with HospitalAdmission and readmission properties.
"""

from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD


# Namespace constants matching the base and extended ontologies
MIMIC_NS = Namespace("http://www.cnam.fr/MIMIC4-ICU-BSI/V1#")
TIME_NS = Namespace("http://www.w3.org/2006/time#")
SNOMED_NS = Namespace("http://snomed.info/id/")


def initialize_graph(ontology_dir: Path) -> Graph:
    """Load base and extended ontologies, bind namespaces, and return the Graph.

    Args:
        ontology_dir: Path to directory containing base_ontology.rdf and
                      extended_ontology.rdf files.

    Returns:
        An rdflib Graph with both ontologies loaded and standard namespaces bound.

    Raises:
        FileNotFoundError: If ontology files are not found.
    """
    g = Graph()

    # Bind standard namespaces
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("time", TIME_NS)
    g.bind("mimic", MIMIC_NS)
    g.bind("sct", SNOMED_NS)

    # Load base ontology
    base_path = ontology_dir / "base_ontology.rdf"
    if not base_path.exists():
        raise FileNotFoundError(f"Base ontology not found at {base_path}")
    g.parse(base_path)

    # Load extended ontology
    extended_path = ontology_dir / "extended_ontology.rdf"
    if not extended_path.exists():
        raise FileNotFoundError(f"Extended ontology not found at {extended_path}")
    g.parse(extended_path)

    return g
