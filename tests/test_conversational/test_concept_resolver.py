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


# ---------------------------------------------------------------------------
# Phase 5 — SNOMED hierarchy fallback
# ---------------------------------------------------------------------------


_TINY_HIERARCHY = (
    Path(__file__).parent / "fixtures" / "snomed" / "tiny_hierarchy.json"
)


@pytest.fixture
def tiny_hierarchy():
    from src.graph_construction.terminology.snomed_hierarchy import SnomedHierarchy

    return SnomedHierarchy(_TINY_HIERARCHY)


@pytest.fixture
def mini_mappings(tmp_path):
    """Tiny mappings dir. Drug index has vancomycin / ceftriaxone / cefepime /
    piperacillin — deliberately missing '999999999' so the reverse-lookup
    filter has something to discard. ``category_to_snomed`` is empty so the
    fallback path actually fires (we force a category-map miss)."""
    import json

    (tmp_path / "category_to_snomed.json").write_text(json.dumps({}))
    (tmp_path / "drug_to_snomed.json").write_text(json.dumps({
        "vancomycin": {
            "rxcui": "11124", "drug_name": "Vancomycin",
            "snomed_code": "372735009", "snomed_term": "Vancomycin",
        },
        "ceftriaxone": {
            "rxcui": "2193", "drug_name": "Ceftriaxone",
            "snomed_code": "396065004", "snomed_term": "Ceftriaxone",
        },
        "cefepime": {
            "rxcui": "25033", "drug_name": "Cefepime",
            "snomed_code": "395985004", "snomed_term": "Cefepime",
        },
        "piperacillin": {
            "rxcui": "8347", "drug_name": "Piperacillin",
            "snomed_code": "387467008", "snomed_term": "Piperacillin",
        },
        # NB: no entry for "cephalosporin" (372756006) — the concept to
        # resolve; the fallback should look it up via SCTID after we seed
        # that in the test.
        "cephalosporin": {
            "rxcui": "2191", "drug_name": "Cephalosporin",
            "snomed_code": "372756006", "snomed_term": "Cephalosporin",
        },
    }))
    return tmp_path


class TestSnomedHierarchyFallback:
    def test_hierarchy_none_passes_through(self, tmp_path):
        """When no hierarchy is provided, unknown-category concepts fall back
        to ``[concept.name]`` — Phase 5 must not regress existing behaviour."""
        (tmp_path / "category_to_snomed.json").write_text("{}")
        resolver = ConceptResolver(mappings_dir=tmp_path, hierarchy=None)
        names = resolver.resolve(
            ClinicalConcept(name="cephalosporin", concept_type="drug")
        )
        assert names == ["cephalosporin"]

    def test_category_map_hit_wins_over_hierarchy(self, tmp_path, tiny_hierarchy):
        """If category_to_snomed already has an entry with ``members``, we use
        it — the hierarchy fallback is not consulted. Keeps curated lists
        authoritative; hierarchy is only a safety net."""
        import json

        (tmp_path / "category_to_snomed.json").write_text(json.dumps({
            "cephalosporin": {
                "snomed_code": "372756006",
                "members": ["curated_a", "curated_b"],
            },
        }))
        resolver = ConceptResolver(mappings_dir=tmp_path, hierarchy=tiny_hierarchy)
        names = resolver.resolve(
            ClinicalConcept(name="cephalosporin", concept_type="drug")
        )
        assert names == ["curated_a", "curated_b"]

    def test_hierarchy_fallback_expands_category_to_mimic_names(
        self, mini_mappings, tiny_hierarchy,
    ):
        """Category miss + concept name maps to an SCTID with ≥2 descendants
        that reverse-lookup to MIMIC-known names → return those names.
        Cephalosporin has ceftriaxone + cefepime descendants in our fixture;
        both are in drug_to_snomed, so both come back."""
        resolver = ConceptResolver(
            mappings_dir=mini_mappings, hierarchy=tiny_hierarchy,
        )
        names = resolver.resolve(
            ClinicalConcept(name="cephalosporin", concept_type="drug")
        )
        assert sorted(names) == ["cefepime", "ceftriaxone"]

    def test_hierarchy_fallback_excludes_sctids_not_in_mimic_mappings(
        self, mini_mappings, tiny_hierarchy,
    ):
        """Descendants whose SCTIDs don't appear in any mapping file are
        excluded. The fixture includes SCTID 999999999 as a child of
        antibiotics that has no MIMIC reverse-lookup — it must not surface."""
        resolver = ConceptResolver(
            mappings_dir=mini_mappings, hierarchy=tiny_hierarchy,
        )
        names = resolver.resolve(
            ClinicalConcept(name="cephalosporin", concept_type="drug")
        )
        # Only the descendants that round-trip through drug_to_snomed.
        assert "999999999" not in names
        assert "not in mimic" not in [n.lower() for n in names]

    def test_hierarchy_fallback_single_descendant_treated_as_specific(
        self, mini_mappings, tiny_hierarchy,
    ):
        """If the concept resolves to a SCTID with zero or one descendant in
        MIMIC, it's not a category — pass through unchanged. "vancomycin"
        is a specific drug in our fixture (no children), so it should pass
        through."""
        resolver = ConceptResolver(
            mappings_dir=mini_mappings, hierarchy=tiny_hierarchy,
        )
        names = resolver.resolve(
            ClinicalConcept(name="vancomycin", concept_type="drug")
        )
        assert names == ["vancomycin"]

    def test_hierarchy_fallback_concept_unknown_in_any_mapping(
        self, mini_mappings, tiny_hierarchy,
    ):
        """If the concept name isn't in any forward mapping, we have no
        SCTID to start from — return ``[concept.name]`` unchanged."""
        resolver = ConceptResolver(
            mappings_dir=mini_mappings, hierarchy=tiny_hierarchy,
        )
        names = resolver.resolve(
            ClinicalConcept(name="mystery_drug_xyz", concept_type="drug")
        )
        assert names == ["mystery_drug_xyz"]

    def test_hierarchy_file_missing_degrades_to_passthrough(
        self, mini_mappings, tmp_path,
    ):
        """If the hierarchy JSON file is absent, SnomedHierarchy returns an
        empty data dict and the fallback produces zero descendants — so we
        pass through cleanly, no exception."""
        from src.graph_construction.terminology.snomed_hierarchy import (
            SnomedHierarchy,
        )

        missing = tmp_path / "does_not_exist.json"
        hierarchy = SnomedHierarchy(missing)
        resolver = ConceptResolver(
            mappings_dir=mini_mappings, hierarchy=hierarchy,
        )
        names = resolver.resolve(
            ClinicalConcept(name="cephalosporin", concept_type="drug")
        )
        assert names == ["cephalosporin"]


# ---------------------------------------------------------------------------
# Phase 5 — Data quality scan of production mappings
# ---------------------------------------------------------------------------


class TestMappingsDataQuality:
    """Guard against accidental corruption of the curated mapping JSONs.
    These run against the REAL files in data/mappings/ — they're the canonical
    dataset the resolver relies on."""

    @pytest.mark.parametrize("filename", [
        "category_to_snomed.json",
        "drug_to_snomed.json",
        "labitem_to_snomed.json",
        "comorbidity_to_snomed.json",
        "chartitem_to_snomed.json",
        "organism_to_snomed.json",
    ])
    def test_every_mapping_file_is_valid_json(self, filename: str):
        import json

        path = MAPPINGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not present in this repo")
        # Must parse — a corrupted JSON would silently degrade the resolver.
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    @pytest.mark.parametrize("filename", [
        "drug_to_snomed.json",
        "labitem_to_snomed.json",
        "comorbidity_to_snomed.json",
    ])
    def test_no_case_insensitive_duplicate_keys(self, filename: str):
        """Two entries like 'Vancomycin' and 'vancomycin' would make the
        resolver nondeterministic — catch it here."""
        import json

        path = MAPPINGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not present in this repo")
        data = json.loads(path.read_text())
        keys = [k for k in data.keys() if k != "_metadata"]
        lowered = [k.lower() for k in keys]
        dupes = {k for k in lowered if lowered.count(k) > 1}
        assert not dupes, f"{filename}: case-insensitive duplicates: {dupes}"
