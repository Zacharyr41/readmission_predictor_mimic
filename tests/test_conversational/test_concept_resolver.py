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


# ---------------------------------------------------------------------------
# Front-half OMOPHub grounding — DiagnosisResolution + resolve_diagnosis
# ---------------------------------------------------------------------------


class TestDiagnosisResolution:
    """resolve_diagnosis mirrors resolve_biomarker but for diagnosis concepts.

    Three terminal cases (mirroring BiomarkerResolution):
      1. concept.icd_codes pre-populated → grounded path; **no MCP call**.
      2. concept.icd_codes None and enable_mcp_grounding=False → silent
         fallback (icd_codes=None, fallback_reason=None).
      3. concept.icd_codes None and enable_mcp_grounding=True → delegated to
         _ground_via_icd_autocode (Inc 3); for now this test class only
         exercises (1) and (2).
    """

    def test_resolve_diagnosis_uses_concept_icd_codes_when_supplied(
        self, resolver,
    ):
        """When the concept already carries icd_codes, the resolver returns
        them directly. No MCP call. This is the test-friendly bypass that
        also handles the (currently hypothetical) case where the LLM
        pre-populates icd_codes itself."""
        concept = ClinicalConcept(
            name="sepsis", concept_type="diagnosis",
            icd_codes=["A41.9", "R65.21"],
        )
        result = resolver.resolve_diagnosis(concept)
        assert result.icd_codes == ["A41.9", "R65.21"]
        # names is always populated for parallel-OR LIKE fallback in SQL.
        assert result.names  # non-empty
        assert result.fallback_reason is None
        assert result.confidence_floor is None

    def test_resolve_diagnosis_silent_fallback_when_mcp_disabled(
        self, resolver,
    ):
        """No icd_codes on the concept AND grounding disabled → silent
        fallback. icd_codes=None signals the SQL emitter to use LIKE-only.
        fallback_reason=None signals 'we didn't try' (no warning surfaced)."""
        concept = ClinicalConcept(
            name="sepsis", concept_type="diagnosis",
        )
        # Default constructor has enable_mcp_grounding=False.
        result = resolver.resolve_diagnosis(concept)
        assert result.icd_codes is None
        assert result.fallback_reason is None
        assert result.names == ["sepsis"]

    def test_resolve_diagnosis_returns_category_names_for_categories(
        self, resolver,
    ):
        """When the concept is a curated category (e.g. 'liver disease' if
        present), `names` should reflect the resolved members, not just the
        category name. This shares the existing resolve() logic."""
        # Use a category that's known to exist in category_to_snomed.json.
        concept = ClinicalConcept(
            name="electrolytes", concept_type="diagnosis",
        )
        result = resolver.resolve_diagnosis(concept)
        # `names` should contain the category members from resolve(),
        # since electrolytes IS a curated category (even though it's
        # canonically a biomarker concept, resolve() doesn't gate on type).
        assert "sodium" in result.names
        assert "potassium" in result.names

    def test_diagnosis_resolution_is_immutable(self):
        """DiagnosisResolution should be a frozen dataclass like
        BiomarkerResolution — prevents downstream mutation surprises."""
        from dataclasses import FrozenInstanceError
        from src.conversational.concept_resolver import DiagnosisResolution

        r = DiagnosisResolution(
            icd_codes=["A41.9"], names=["sepsis"],
            fallback_reason=None, confidence_floor=None,
        )
        with pytest.raises(FrozenInstanceError):
            r.icd_codes = ["X"]  # type: ignore[misc]

    def test_resolve_diagnosis_does_not_call_icd_autocode_when_disabled(
        self, resolver, monkeypatch,
    ):
        """Belt-and-suspenders: even if the MCP module is patched to track
        calls, with grounding disabled we must not invoke it."""
        from src.conversational import concept_resolver as cr

        called: list = []
        def fake_autocode(*args, **kwargs):
            called.append((args, kwargs))
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        resolver.resolve_diagnosis(concept)
        assert called == []


class TestGroundViaIcdAutocode:
    """Inc 3 — the cached MCP-grounded path inside resolve_diagnosis.

    These tests construct a resolver with ``enable_mcp_grounding=True``
    and monkeypatch the module-level ``icd_autocode`` symbol. The cache
    must be cleared between tests so prior memoization doesn't leak.
    """

    @pytest.fixture
    def grounded_resolver(self, monkeypatch):
        """Resolver with MCP grounding enabled + cache cleared."""
        from src.conversational import concept_resolver as cr
        # Clear the lru_cache so cross-test pollution doesn't bite.
        cr._cached_icd_autocode.cache_clear()
        # No real backoff sleeps when the retry path fires in these tests.
        monkeypatch.setattr(cr, "_ICD_AUTOCODE_RETRY_BACKOFF", 0.0, raising=False)
        return ConceptResolver(
            mappings_dir=MAPPINGS_DIR, enable_mcp_grounding=True,
        )

    def test_filters_by_confidence_threshold(
        self, grounded_resolver, monkeypatch,
    ):
        """Results with confidence < 0.5 are dropped; None confidence is
        accepted (OMOPHub semantic_search often omits scores)."""
        from src.conversational import concept_resolver as cr

        def fake_autocode(text, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"code": "A41.9", "title": "Sepsis", "confidence": 0.92},
                    {"code": "R65.21", "title": "Severe sepsis", "confidence": 0.71},
                    {"code": "L08.9", "title": "Local skin infection", "confidence": 0.31},  # drop
                    {"code": "B99",    "title": "Other infectious", "confidence": None},      # keep
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        result = grounded_resolver.resolve_diagnosis(concept)
        assert result.icd_codes is not None
        # A41.9, R65.21 (above threshold) and B99 (None confidence) — kept.
        # L08.9 (below threshold) — dropped.
        assert "A41.9" in result.icd_codes
        assert "R65.21" in result.icd_codes
        assert "B99" in result.icd_codes
        assert "L08.9" not in result.icd_codes
        # Min confidence among accepted (None doesn't count).
        assert result.confidence_floor == 0.71
        assert result.fallback_reason is None

    def test_returns_loud_fallback_when_unavailable(
        self, grounded_resolver, monkeypatch,
    ):
        """OMOPHub returns unavailable → loud fallback with reason."""
        from src.conversational import concept_resolver as cr

        def fake_autocode(text, **kwargs):
            return {"status": "unavailable", "error": "MCP timeout"}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        result = grounded_resolver.resolve_diagnosis(concept)
        assert result.icd_codes is None
        assert result.fallback_reason is not None
        assert "icd autocoding" in result.fallback_reason.lower()
        # User-visible warning should mention the analyte name.
        assert "sepsis" in result.fallback_reason.lower()

    def test_returns_loud_fallback_when_all_low_confidence(
        self, grounded_resolver, monkeypatch,
    ):
        """All candidates below 0.5 → loud fallback citing max confidence."""
        from src.conversational import concept_resolver as cr

        def fake_autocode(text, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"code": "X1", "title": "Low1", "confidence": 0.31},
                    {"code": "X2", "title": "Low2", "confidence": 0.42},
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        result = grounded_resolver.resolve_diagnosis(concept)
        assert result.icd_codes is None
        assert result.fallback_reason is not None
        # Max-confidence value should appear in the message for telemetry.
        assert "0.42" in result.fallback_reason

    def test_caches_repeat_lookups(
        self, grounded_resolver, monkeypatch,
    ):
        """Process-wide lru_cache: identical (text, version) hits OMOPHub
        only once across multiple resolver calls."""
        from src.conversational import concept_resolver as cr

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {
                "status": "ok",
                "results": [{"code": "A41.9", "title": "Sepsis", "confidence": 0.9}],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        grounded_resolver.resolve_diagnosis(concept)
        grounded_resolver.resolve_diagnosis(concept)
        grounded_resolver.resolve_diagnosis(concept)
        assert call_count[0] == 1

    def test_does_not_cache_unavailable(
        self, grounded_resolver, monkeypatch,
    ):
        """A *persistent* failure must not be cached — transient OMOPHub
        failures shouldn't poison the cache for the rest of the process.
        Implemented via sentinel-raise: the cached function raises LookupError
        once its bounded retries are exhausted, so nothing is memoized and a
        later call hits the MCP again. (A *single* transient miss is instead
        absorbed by the retry — see ``TestIcdAutocodeRetryResilience``.)"""
        from src.conversational import concept_resolver as cr

        # Exhaust the first resolve's whole retry budget so it falls back,
        # then succeed — proving the failure was never cached.
        max_attempts = cr._ICD_AUTOCODE_MAX_ATTEMPTS
        call_count = [0]
        responses = (
            [{"status": "unavailable", "error": "transient"}] * max_attempts
            + [{"status": "ok", "results": [
                {"code": "A41.9", "title": "Sepsis", "confidence": 0.9},
            ]}]
        )
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return responses[min(call_count[0] - 1, len(responses) - 1)]
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        first = grounded_resolver.resolve_diagnosis(concept)
        second = grounded_resolver.resolve_diagnosis(concept)
        assert first.icd_codes is None  # retries exhausted → fallback
        assert second.icd_codes == ["A41.9"]  # not cached → hits MCP again
        # max_attempts failed tries on the first resolve + 1 success on the second.
        assert call_count[0] == max_attempts + 1

    def test_caches_keyed_by_lowercased_name(
        self, grounded_resolver, monkeypatch,
    ):
        """Cache key uses lowered name so 'Sepsis' and 'sepsis' share the
        same MCP call. Avoids redundant lookups when the LLM varies casing."""
        from src.conversational import concept_resolver as cr

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {
                "status": "ok",
                "results": [{"code": "A41.9", "title": "Sepsis", "confidence": 0.9}],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        a = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        b = ClinicalConcept(name="Sepsis", concept_type="diagnosis")
        grounded_resolver.resolve_diagnosis(a)
        grounded_resolver.resolve_diagnosis(b)
        assert call_count[0] == 1


class TestIcdAutocodeRetryResilience:
    """``icd_autocode`` grounding must survive a *transient* MCP hiccup.

    Bug reproduced here: under cumulative full-suite load the OMOPHub
    ``icd_autocode`` MCP intermittently returns ``unavailable`` (or a transient
    empty result). The old code called the tool exactly once, so a single miss
    fell straight through to the title-LIKE fallback — which is empty for
    colloquial terms ("diabetic ketoacidosis" matches 0 MIMIC ``long_title``
    rows, since they read "diabetes mellitus *with* ketoacidosis"). The result
    was an empty cohort and a spurious "no data" answer for DKA / cirrhosis
    biomarker questions, intermittently.

    Fix: ``_cached_icd_autocode`` retries on a transient/empty result (bounded,
    with backoff) and never caches a failed lookup — so a single hiccup is
    absorbed and grounding still succeeds.

    These tests inject the transient deterministically (mock ``icd_autocode``
    to miss then succeed), so they fail RELIABLY without the retry and pass with
    it — no dependence on real MCP load.
    """

    @pytest.fixture
    def grounded_resolver(self, monkeypatch):
        from src.conversational import concept_resolver as cr
        cr._cached_icd_autocode.cache_clear()
        # No real backoff sleeps in tests (constant added by the fix; tolerate
        # its absence so this also runs pre-fix to prove the failure).
        monkeypatch.setattr(cr, "_ICD_AUTOCODE_RETRY_BACKOFF", 0.0, raising=False)
        return ConceptResolver(mappings_dir=MAPPINGS_DIR, enable_mcp_grounding=True)

    @staticmethod
    def _sequenced_autocode(responses, calls):
        def fake_autocode(text, **kwargs):
            calls[0] += 1
            return responses[min(calls[0] - 1, len(responses) - 1)]
        return fake_autocode

    def test_retries_transient_unavailable_then_succeeds(
        self, grounded_resolver, monkeypatch,
    ):
        """One transient ``unavailable`` is retried; grounding still succeeds.

        Pre-fix (single call, no retry): the lone ``unavailable`` falls back →
        ``icd_codes is None`` → this assertion FAILS. That is the bug."""
        from src.conversational import concept_resolver as cr
        calls = [0]
        responses = [
            {"status": "unavailable", "error": "transient MCP timeout"},
            {"status": "ok", "results": [
                {"code": "E11.1", "title": "T2DM with ketoacidosis", "confidence": 0.9},
            ]},
        ]
        monkeypatch.setattr(
            cr, "icd_autocode", self._sequenced_autocode(responses, calls),
            raising=False,
        )
        concept = ClinicalConcept(
            name="diabetic ketoacidosis", concept_type="diagnosis",
        )
        result = grounded_resolver.resolve_diagnosis(concept)
        assert result.icd_codes == ["E11.1"], (
            "transient MCP failure was not retried — grounding fell back "
            f"(icd_codes={result.icd_codes!r})"
        )
        assert calls[0] >= 2, "expected a retry after the transient failure"

    def test_retries_transient_empty_result_then_succeeds(
        self, grounded_resolver, monkeypatch,
    ):
        """A transient ``status=ok`` but empty result is retried, not accepted.

        Pre-fix: the empty result is taken as final (cached as ``()``) →
        fallback → ``icd_codes is None`` → FAILS."""
        from src.conversational import concept_resolver as cr
        calls = [0]
        responses = [
            {"status": "ok", "results": []},  # transient empty
            {"status": "ok", "results": [
                {"code": "K74.6", "title": "Cirrhosis of liver", "confidence": 0.88},
            ]},
        ]
        monkeypatch.setattr(
            cr, "icd_autocode", self._sequenced_autocode(responses, calls),
            raising=False,
        )
        concept = ClinicalConcept(name="cirrhosis", concept_type="diagnosis")
        result = grounded_resolver.resolve_diagnosis(concept)
        assert result.icd_codes == ["K74.6"], (
            "transient empty result was not retried — grounding fell back "
            f"(icd_codes={result.icd_codes!r})"
        )
        assert calls[0] >= 2, "expected a retry after the transient empty result"

    def test_persistent_failure_falls_back_after_bounded_retries(
        self, grounded_resolver, monkeypatch,
    ):
        """A *persistent* failure still falls back (no infinite loop), and the
        retry count is bounded by ``_ICD_AUTOCODE_MAX_ATTEMPTS``.

        Pre-fix: only 1 attempt is made, so ``calls[0] == MAX`` FAILS (MAX>1)."""
        from src.conversational import concept_resolver as cr
        calls = [0]
        responses = [{"status": "unavailable", "error": "OMOPHub down"}]
        monkeypatch.setattr(
            cr, "icd_autocode", self._sequenced_autocode(responses, calls),
            raising=False,
        )
        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        result = grounded_resolver.resolve_diagnosis(concept)
        assert result.icd_codes is None, "persistent failure should fall back"
        assert result.fallback_reason is not None
        max_attempts = getattr(cr, "_ICD_AUTOCODE_MAX_ATTEMPTS", 3)
        assert max_attempts > 1, "retry budget must allow at least one retry"
        assert calls[0] == max_attempts, (
            f"expected exactly {max_attempts} bounded attempts, got {calls[0]}"
        )


# ---------------------------------------------------------------------------
# Inc 7 — biomarker mimic_itemid_search fallback when LOINC mapping misses
# ---------------------------------------------------------------------------


class TestBiomarkerMimicItemidFallback:
    """When the local LOINC→itemid index misses (LOINC unknown OR no MIMIC
    coverage), call OMOPHub-backed ``mimic_itemid_search`` to find a live
    match against MIMIC's d_labitems / d_items. Restricts results to
    ``table='labevents'`` for biomarker concepts; chartevents-only results
    fall through to the existing loud-fallback (don't risk pulling
    vital-sign itemids into a lab query)."""

    @pytest.fixture
    def grounded_resolver(self):
        """Resolver with MCP grounding enabled + cache cleared."""
        from src.conversational import concept_resolver as cr
        cr._cached_mimic_itemid_search.cache_clear()
        return ConceptResolver(
            mappings_dir=MAPPINGS_DIR, enable_mcp_grounding=True,
        )

    def test_falls_back_to_mimic_itemid_search_when_loinc_unknown(
        self, grounded_resolver, monkeypatch,
    ):
        """LOINC absent from local index → call mimic_itemid_search;
        recover labevents itemids for the analyte."""
        from src.conversational import concept_resolver as cr

        def fake_search(query, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"itemid": 50813, "label": "Lactate", "table": "labevents", "loinc": "32693-4"},
                    {"itemid": 52442, "label": "Lactate, ABG", "table": "labevents", "loinc": "2518-9"},
                ],
            }
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        concept = ClinicalConcept(
            name="lactate", concept_type="biomarker",
            loinc_code="99999-9",  # not in local index
        )
        result = grounded_resolver.resolve_biomarker(concept)
        assert result.itemids == [50813, 52442]
        assert result.fallback_reason is None

    def test_unchanged_when_local_loinc_index_hits(
        self, grounded_resolver, monkeypatch,
    ):
        """Local LOINC mapping wins; mimic_itemid_search is NOT called.
        Existing fast-path stays fast and avoids needless MCP traffic."""
        from src.conversational import concept_resolver as cr

        call_count = [0]
        def fake_search(query, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        # 2160-0 (serum creatinine) is in the local index.
        concept = ClinicalConcept(
            name="creatinine", concept_type="biomarker", loinc_code="2160-0",
        )
        result = grounded_resolver.resolve_biomarker(concept)
        assert result.itemids and 50912 in result.itemids
        assert call_count[0] == 0  # MCP never called

    def test_filters_to_labevents_table(
        self, grounded_resolver, monkeypatch,
    ):
        """Mixed-table results: only labevents itemids end up in resolution.
        Chartevents itemids represent vitals/charts — pulling them into a
        biomarker query would over-include unit-incompatible signals."""
        from src.conversational import concept_resolver as cr

        def fake_search(query, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"itemid": 50813, "label": "Lactate", "table": "labevents"},
                    {"itemid": 220045, "label": "Heart rate", "table": "chartevents"},  # drop
                    {"itemid": 52442, "label": "Lactate ABG", "table": "labevents"},
                ],
            }
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        concept = ClinicalConcept(
            name="lactate", concept_type="biomarker", loinc_code="99999-9",
        )
        result = grounded_resolver.resolve_biomarker(concept)
        assert result.itemids == [50813, 52442]
        assert 220045 not in (result.itemids or [])

    def test_chartevents_only_returns_loud_fallback(
        self, grounded_resolver, monkeypatch,
    ):
        """If mimic_itemid_search returns ONLY chartevents (no labevents),
        we don't ground at all — better to surface a visible warning than
        risk pulling vitals into a lab query. itemids stays None and
        fallback_reason is set."""
        from src.conversational import concept_resolver as cr

        def fake_search(query, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"itemid": 220045, "label": "Heart rate", "table": "chartevents"},
                    {"itemid": 220180, "label": "BP", "table": "chartevents"},
                ],
            }
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        concept = ClinicalConcept(
            name="something_oddly_named", concept_type="biomarker",
            loinc_code="99999-9",
        )
        result = grounded_resolver.resolve_biomarker(concept)
        assert result.itemids is None
        assert result.fallback_reason is not None
        assert "label match" in result.fallback_reason.lower()

    def test_caches_repeat_lookups(
        self, grounded_resolver, monkeypatch,
    ):
        """Same (loinc, name) pair cached across repeated resolve_biomarker
        calls."""
        from src.conversational import concept_resolver as cr

        call_count = [0]
        def fake_search(query, **kwargs):
            call_count[0] += 1
            return {
                "status": "ok",
                "results": [
                    {"itemid": 50813, "label": "Lactate", "table": "labevents"},
                ],
            }
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        concept = ClinicalConcept(
            name="lactate", concept_type="biomarker", loinc_code="99999-9",
        )
        grounded_resolver.resolve_biomarker(concept)
        grounded_resolver.resolve_biomarker(concept)
        grounded_resolver.resolve_biomarker(concept)
        assert call_count[0] == 1

    def test_does_not_run_when_grounding_disabled(
        self, monkeypatch,
    ):
        """enable_mcp_grounding=False (the test default) → no MCP call,
        even if the LOINC isn't in the local index. Existing loud-fallback
        path is unchanged."""
        from src.conversational import concept_resolver as cr

        call_count = [0]
        def fake_search(query, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        offline_resolver = ConceptResolver(
            mappings_dir=MAPPINGS_DIR, enable_mcp_grounding=False,
        )
        concept = ClinicalConcept(
            name="lactate", concept_type="biomarker", loinc_code="99999-9",
        )
        result = offline_resolver.resolve_biomarker(concept)
        assert call_count[0] == 0
        # Existing loud-fallback path still fires.
        assert result.itemids is None
        assert result.fallback_reason is not None

    def test_falls_back_to_loud_fallback_when_search_unavailable(
        self, grounded_resolver, monkeypatch,
    ):
        """MCP unavailable → loud-fallback. Existing fallback_reason
        message preserved (so test snapshots don't churn)."""
        from src.conversational import concept_resolver as cr

        def fake_search(query, **kwargs):
            return {"status": "unavailable", "error": "MCP timeout"}
        monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

        concept = ClinicalConcept(
            name="lactate", concept_type="biomarker", loinc_code="99999-9",
        )
        result = grounded_resolver.resolve_biomarker(concept)
        assert result.itemids is None
        assert result.fallback_reason is not None


# ---------------------------------------------------------------------------
# Inc 8 — Live integration tests (gated by env var; skipped in CI)
#
# These hit the real OMOPHub hosted MCP. Skipped unless RUN_LIVE_OMOPHUB=1.
# Useful before shipping — proves the front-half plumbing actually grounds
# real diagnoses against the live endpoint.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("RUN_LIVE_OMOPHUB"),
    reason="Set RUN_LIVE_OMOPHUB=1 + OMOPHUB_API_KEY to run live integration",
)
class TestLiveOmophubFrontHalfGrounding:
    """End-to-end against real OMOPHub. Confirms that with the real MCP,
    resolve_diagnosis actually grounds 'sepsis' to ICD codes that match
    the SQL-fastpath compiler's expected shape."""

    @pytest.fixture
    def live_resolver(self):
        from src.conversational import concept_resolver as cr
        cr._cached_icd_autocode.cache_clear()
        cr._cached_mimic_itemid_search.cache_clear()
        return ConceptResolver(
            mappings_dir=MAPPINGS_DIR, enable_mcp_grounding=True,
        )

    def test_live_grounds_sepsis_to_icd_codes(self, live_resolver):
        """Real OMOPHub call: 'sepsis' should ground to A41/R65/A40 family."""
        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        result = live_resolver.resolve_diagnosis(concept)
        assert result.icd_codes is not None
        # Sepsis-family codes start with A41, R65, A40 in ICD-10-CM.
        assert any(
            c.startswith(("A41", "R65", "A40", "A42"))
            for c in result.icd_codes
        ), f"expected sepsis codes; got {result.icd_codes!r}"
        assert result.fallback_reason is None

    def test_live_compile_sql_emits_in_list_for_sepsis(self, live_resolver):
        """Threading through compile_sql: the grounded codes show up as an
        IN-list in the generated WHERE clause. This is the smoking-gun
        end-to-end check — what users see in 'Query Details' on the dash."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        from src.conversational.models import CompetencyQuestion

        concept = ClinicalConcept(name="sepsis", concept_type="diagnosis")
        diag = live_resolver.resolve_diagnosis(concept)

        cq = CompetencyQuestion(
            original_question="how many sepsis patients?",
            clinical_concepts=[concept],
            aggregation="count", scope="cohort",
        )
        # Use a minimal stub backend just for SQL emission (we don't
        # execute against a DB — just check the emitted SQL string).
        from tests.test_conversational.test_sql_fastpath import _ConnBackend
        import duckdb
        backend = _ConnBackend(duckdb.connect(":memory:"))

        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=diag.names,
            resolved_icd_codes=diag.icd_codes,
        )
        assert "di.icd_code IN (" in query.sql
        # At least one sepsis-family code in the params.
        assert any(
            isinstance(p, str) and p.startswith(("A41", "R65", "A40", "A42"))
            for p in query.params
        ), f"expected sepsis-family ICD code in params; got {query.params!r}"

    def test_live_compile_sql_grounds_filter_side_for_lactate_in_sepsis(
        self, live_resolver,
    ):
        """Inc 9 smoking-gun verification: the originally-failing query
        ('mean lactate in sepsis cohort') decomposes to a biomarker CQ
        with diagnosis as a patient_filter. With Inc 9 wiring,
        compile_sql(enable_mcp_grounding=True) routes the filter through
        icd_autocode and the emitted SQL contains ``di.icd_code IN (...)``
        in the cohort WHERE clause."""
        from src.conversational.models import CompetencyQuestion, PatientFilter
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        from tests.test_conversational.test_sql_fastpath import _ConnBackend
        import duckdb

        cq = CompetencyQuestion(
            original_question="mean lactate in sepsis",
            clinical_concepts=[
                ClinicalConcept(name="lactate", concept_type="biomarker"),
            ],
            patient_filters=[
                PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
            ],
            aggregation="mean", scope="cohort",
        )
        backend = _ConnBackend(duckdb.connect(":memory:"))

        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["lactate"],
            enable_mcp_grounding=True,
        )
        # Filter side now ICD-grounded.
        assert "di.icd_code IN (" in query.sql
        assert any(
            isinstance(p, str) and p.startswith(("A41", "R65", "A40", "A42"))
            for p in query.params
        ), f"expected sepsis-family ICD codes in params; got {query.params!r}"
