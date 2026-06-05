"""Tests for the Charlson comorbidity builder (plan I-B).

Wires the previously-dead ``write_comorbidity`` writer into graph construction:
a patient's ICD-10 diagnoses are prefix-matched against the Quan et al. 2005
Charlson mapping (``data/mappings/icd10_to_charlson.json`` — the same
ontology-grounded mapping the WLST feature builder uses, no hand-curated
synonym lists) and each present category becomes a ``mimic:Comorbidity`` node
linked to the patient and grounded to SNOMED-CT.
"""

from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.comorbidity_builder import (
    derive_charlson_categories,
    write_patient_comorbidities,
)
from src.graph_construction.ontology import MIMIC_NS
from src.graph_construction.terminology.snomed_mapper import SnomedMapper

_MAPPINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "mappings"


class TestDeriveCharlsonCategories:
    """The pure derivation: ICD-10 codes -> Charlson category names."""

    def test_chf_icd10_maps_to_congestive_heart_failure(self):
        dx = [{"icd_code": "I509", "icd_version": 10}]
        assert derive_charlson_categories(dx) == ["congestive_heart_failure"]

    def test_dotted_code_is_normalized(self):
        # Real MIMIC stores dotless codes, but some upstreams carry the decimal
        # point; either form must match the dotless Charlson prefixes.
        dx = [{"icd_code": "I50.9", "icd_version": 10}]
        assert "congestive_heart_failure" in derive_charlson_categories(dx)

    def test_icd9_codes_are_ignored(self):
        # The mapping is ICD-10; an ICD-9 CHF code (428.0 -> "4280") must not
        # match an ICD-10 prefix by coincidence.
        dx = [{"icd_code": "4280", "icd_version": 9}]
        assert derive_charlson_categories(dx) == []

    def test_unmatched_code_yields_no_category(self):
        dx = [{"icd_code": "Z9999", "icd_version": 10}]
        assert derive_charlson_categories(dx) == []

    def test_multiple_codes_dedupe_category_and_collect_distinct(self):
        # Two CHF codes collapse to one category; a cerebrovascular code adds a
        # second. Category order is stable (mapping order) for a deterministic
        # graph.
        dx = [
            {"icd_code": "I509", "icd_version": 10},
            {"icd_code": "I5082", "icd_version": 10},
            {"icd_code": "I630", "icd_version": 10},
        ]
        cats = derive_charlson_categories(dx)
        assert cats.count("congestive_heart_failure") == 1
        assert "cerebrovascular_disease" in cats

    def test_empty_diagnoses_yield_no_categories(self):
        assert derive_charlson_categories([]) == []


class TestWritePatientComorbidities:
    """Writing the derived categories as ``mimic:Comorbidity`` nodes."""

    def test_writes_comorbidity_node_linked_to_patient(self):
        graph = Graph()
        patient_uri = MIMIC_NS["Patient-100"]
        dx = [{"icd_code": "I509", "icd_version": 10}]

        uris = write_patient_comorbidities(graph, dx, patient_uri, subject_id=100)

        assert len(uris) == 1
        node = uris[0]
        assert (node, RDF.type, MIMIC_NS.Comorbidity) in graph
        assert (
            node,
            MIMIC_NS.hasComorbidityName,
            Literal("congestive_heart_failure", datatype=XSD.string),
        ) in graph
        assert (patient_uri, MIMIC_NS.hasComorbidity, node) in graph

    def test_snomed_grounding_when_mapper_supplied(self):
        # congestive_heart_failure -> SNOMED 42343007 via comorbidity_to_snomed.
        graph = Graph()
        patient_uri = MIMIC_NS["Patient-100"]
        dx = [{"icd_code": "I509", "icd_version": 10}]

        write_patient_comorbidities(
            graph, dx, patient_uri, subject_id=100,
            snomed_mapper=SnomedMapper(_MAPPINGS_DIR),
        )

        codes = {
            str(o) for o in graph.objects(None, MIMIC_NS.hasSnomedCode)
        }
        assert "42343007" in codes

    def test_no_diagnoses_writes_nothing(self):
        graph = Graph()
        patient_uri = MIMIC_NS["Patient-100"]
        uris = write_patient_comorbidities(graph, [], patient_uri, subject_id=100)
        assert uris == []
        assert (None, RDF.type, MIMIC_NS.Comorbidity) not in graph
