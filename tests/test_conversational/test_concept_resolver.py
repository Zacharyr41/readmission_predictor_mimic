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
# Lab-resolver fix — LOINC-grounded biomarker resolution
# ---------------------------------------------------------------------------
#
# These tests pin the contract for the lab-resolver fix. ``resolve_biomarker``
# is a new method on ConceptResolver that takes a ClinicalConcept (which may
# carry a ``loinc_code``) and returns a ``BiomarkerResolution`` describing
# either an itemid set (success) or a name passthrough (fallback).
#
# The fallback distinction matters: ``loinc_code is None`` is *normal* (uncommon
# labs, novel terms) and gets silent LIKE behavior. When a LOINC was supplied
# but couldn't be grounded (cases 2 + 3 below), ``fallback_reason`` is set so
# the orchestrator can surface a visible warning to the user.


class TestBiomarkerLoincResolution:
    """LOINC-grounded resolution path. ``resolve_biomarker`` is the new
    method; it does not exist on ConceptResolver yet, so most of these
    tests fail until Phase 4.1 lands the implementation. The schema-level
    tests pass on Phase 2's ClinicalConcept addition alone.
    """

    # -- Schema (Phase 2 production change) ----------------------------------

    def test_clinical_concept_accepts_loinc_code(self):
        c = ClinicalConcept(
            name="creatinine", concept_type="biomarker", loinc_code="2160-0",
        )
        assert c.loinc_code == "2160-0"

    def test_clinical_concept_loinc_code_optional(self):
        c = ClinicalConcept(name="rare_lab", concept_type="biomarker")
        assert c.loinc_code is None

    # -- Successful LOINC grounding ------------------------------------------

    def test_resolve_biomarker_with_loinc_returns_itemids(self, resolver):
        """LOINC 2160-0 (serum creatinine) resolves via SNOMED 113075003 to
        MIMIC itemid 50912 (and any sibling itemids that also map to the
        same SNOMED term, e.g. 51081). Urine creatinine (51082) maps to
        a *different* LOINC (2161-8) and must be excluded."""
        concept = ClinicalConcept(
            name="creatinine", concept_type="biomarker", loinc_code="2160-0",
        )
        result = resolver.resolve_biomarker(concept)
        assert result.itemids is not None
        assert 50912 in result.itemids
        assert 51082 not in result.itemids
        assert result.loinc_code == "2160-0"
        assert result.snomed_code == "113075003"
        assert result.fallback_reason is None
        # ``names`` is always populated — used by the LIKE fallback path.
        assert "creatinine" in result.names

    # -- No LOINC: silent passthrough (Case 1) -------------------------------

    def test_resolve_biomarker_without_loinc_returns_name_passthrough(self, resolver):
        """When no LOINC is supplied, ``itemids`` is None and ``fallback_reason``
        is also None — this is the normal path for uncommon labs and shouldn't
        produce a warning. ``names`` carries through whatever the existing
        ``resolve()`` would return."""
        concept = ClinicalConcept(name="creatinine", concept_type="biomarker")
        result = resolver.resolve_biomarker(concept)
        assert result.itemids is None
        assert result.loinc_code is None
        assert result.fallback_reason is None  # no grounding attempted
        assert result.names == ["creatinine"]

    # -- LOINC supplied but missing from loinc_to_snomed (Case 2) ------------

    def test_resolve_biomarker_unknown_loinc_falls_back_with_reason(self, resolver):
        """LOINC '99999-9' is not in loinc_to_snomed.json. Resolution falls
        back to name passthrough AND sets ``fallback_reason`` so the
        orchestrator can surface a visible warning."""
        concept = ClinicalConcept(
            name="some_lab", concept_type="biomarker", loinc_code="99999-9",
        )
        result = resolver.resolve_biomarker(concept)
        assert result.itemids is None
        assert result.fallback_reason is not None
        assert "99999-9" in result.fallback_reason
        assert result.names == ["some_lab"]

    # -- Latent forward-index bug (Phase 4.5) --------------------------------

    def test_forward_index_resolves_lab_name_to_sctid(self, resolver):
        """Latent bug: ``labitem_to_snomed.json`` and ``chartitem_to_snomed.json``
        are keyed by MIMIC itemid (e.g. "50912"), not by lab name. The forward
        index in ``_build_sctid_indices`` keyed every entry by its outer key,
        so ``forward["creatinine"]`` always missed and the SNOMED-hierarchy
        fallback was structurally broken for any lab term.

        Fix: for lab/chartitem files, key the forward index by ``entry['label']``
        instead of the outer key. ``forward["creatinine"]`` should now return
        the SNOMED concept id 113075003.
        """
        # Force the index to build by calling resolve on something innocuous.
        forward, _ = resolver._build_sctid_indices()
        assert "creatinine" in forward, (
            "Forward index missed lab name 'creatinine' — the file is keyed "
            "by itemid but the forward index should also accept lab names."
        )
        assert forward["creatinine"] == "113075003"

    # -- LOINC valid, but no MIMIC labitem maps to its SNOMED (Case 3) -------

    def test_resolve_biomarker_loinc_with_no_mimic_coverage(self, resolver):
        """LOINC 10331-7 ('Rh type') is in loinc_to_snomed.json with SNOMED
        371154000, but no entry in labitem_to_snomed.json maps to that SNOMED
        (verified by data-mapping audit). The resolver should detect this
        and fall back with a fallback_reason naming the SNOMED gap."""
        concept = ClinicalConcept(
            name="rh_type", concept_type="biomarker", loinc_code="10331-7",
        )
        result = resolver.resolve_biomarker(concept)
        assert result.itemids is None
        assert result.fallback_reason is not None
        # The reason should mention the SNOMED code so it's diagnosable.
        assert "371154000" in result.fallback_reason or "10331-7" in result.fallback_reason
        assert result.names == ["rh_type"]


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
