"""Tests for the SNOMED-based concept resolver."""

from pathlib import Path

import pytest

from src.conversational.concept_resolver import ConceptResolver
from src.conversational.models import ClinicalConcept


MAPPINGS_DIR = Path(__file__).parent.parent.parent / "data" / "mappings"


@pytest.fixture
def resolver():
    """ConceptResolver using the real category_to_snomed.json."""
    return ConceptResolver(mappings_dir=MAPPINGS_DIR)


class TestConceptResolver:
    def test_specific_drug_passes_through(self, resolver):
        """A specific drug name returns unchanged."""
        concept = ClinicalConcept(name="vancomycin", concept_type="drug")
        names = resolver.resolve(concept)
        assert names == ["vancomycin"]

    def test_category_resolves_to_specifics(self, resolver):
        """'antibiotics' resolves to multiple specific drug names."""
        concept = ClinicalConcept(name="antibiotics", concept_type="drug")
        names = resolver.resolve(concept)
        assert len(names) > 5
        assert "vancomycin" in names
        assert "ceftriaxone" in names
        assert "meropenem" in names

    def test_unknown_category_passes_through(self, resolver):
        """Unknown category name returns [name] unchanged."""
        concept = ClinicalConcept(name="mystery_drug_xyz", concept_type="drug")
        names = resolver.resolve(concept)
        assert names == ["mystery_drug_xyz"]

    def test_resolves_lab_category(self, resolver):
        """'liver function tests' resolves to lab names."""
        concept = ClinicalConcept(name="liver function tests", concept_type="biomarker")
        names = resolver.resolve(concept)
        assert len(names) > 3
        assert any("aminotransferase" in n for n in names)

    def test_resolves_vasopressors(self, resolver):
        concept = ClinicalConcept(name="vasopressors", concept_type="drug")
        names = resolver.resolve(concept)
        assert "norepinephrine" in names
        assert "vasopressin" in names

    def test_resolves_electrolytes(self, resolver):
        concept = ClinicalConcept(name="electrolytes", concept_type="biomarker")
        names = resolver.resolve(concept)
        assert "sodium" in names
        assert "potassium" in names

    def test_case_insensitive_lookup(self, resolver):
        """Category lookup should be case-insensitive."""
        concept = ClinicalConcept(name="Antibiotics", concept_type="drug")
        names = resolver.resolve(concept)
        assert len(names) > 5

    def test_no_mappings_dir_degrades(self, tmp_path):
        """Missing mappings dir → always pass through."""
        resolver = ConceptResolver(mappings_dir=tmp_path)
        concept = ClinicalConcept(name="antibiotics", concept_type="drug")
        names = resolver.resolve(concept)
        assert names == ["antibiotics"]
