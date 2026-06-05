"""Tests for build-time drug-category resolution (plan I-A).

A MIMIC ``prescriptions.drug`` string is resolved to one or more canonical
drug *categories* (vasopressors, antibiotics, ...) so the cohort feature
extractor (plan III-A) can select, e.g., "all vasopressor administrations
for this stay" and compute a dose trend.

The category vocabulary is the ontology-grounded set in
``data/mappings/category_to_snomed.json`` (each category carries a SNOMED
root). Membership comes from the curated ``members`` bootstrap; brand and
combination names that miss the bootstrap are resolved through an
RxNorm-ingredient escape hatch (no hand-curated synonym lists). Only
categories tagged ``kind: "drug"`` participate, so a human-albumin infusion
is never mislabeled a "liver function test".
"""

from pathlib import Path

from src.graph_construction.terminology.drug_category import (
    DrugCategory,
    DrugCategoryResolver,
)

_MAPPINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "mappings"


def _names(cats: list[DrugCategory]) -> set[str]:
    return {c.name for c in cats}


class TestCuratedResolution:
    """Offline reverse-lookup over the curated ``members`` bootstrap."""

    def test_vasopressor_drug_maps_to_vasopressors(self):
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        cats = resolver.resolve("Norepinephrine")
        assert "vasopressors" in _names(cats)
        vp = next(c for c in cats if c.name == "vasopressors")
        assert vp.snomed_code == "372881000"
        assert vp.source == "curated"

    def test_antibiotic_drug_maps_to_antibiotics(self):
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        assert "antibiotics" in _names(resolver.resolve("Vancomycin"))

    def test_case_insensitive(self):
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        assert "antibiotics" in _names(resolver.resolve("VANCOMYCIN"))

    def test_trailing_dose_is_stripped(self):
        # MIMIC carries dose/strength on the drug string; normalization must
        # strip it so "Norepinephrine 4 mg/250 mL" still matches.
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        assert "vasopressors" in _names(resolver.resolve("Norepinephrine 4 mg/250 mL"))

    def test_combination_salt_name_matches_by_token(self):
        # "Norepinephrine Bitartrate" should match the ingredient token; the
        # match is word-boundary, not prefix (so "ph" never matches
        # "phenylephrine").
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        assert "vasopressors" in _names(resolver.resolve("Norepinephrine Bitartrate"))

    def test_drug_in_two_categories_returns_both(self):
        # Fentanyl is both a sedative and an opioid in the bootstrap; both
        # categories are returned (multi-valued).
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        names = _names(resolver.resolve("Fentanyl"))
        assert {"sedatives", "opioids"} <= names

    def test_unknown_drug_yields_no_category(self):
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        assert resolver.resolve("Zzznonexistentdrug") == []

    def test_non_drug_category_member_is_not_a_drug_category(self):
        # "albumin" is a member of the (lab) "liver function tests" category,
        # but a human-albumin infusion must NOT be tagged a liver function
        # test: only kind:"drug" categories participate.
        resolver = DrugCategoryResolver(_MAPPINGS_DIR)
        assert "liver function tests" not in _names(resolver.resolve("Albumin"))


class TestRxNormEscapeHatch:
    """Brand / combination names that miss the bootstrap are grounded via the
    RxNorm-ingredient escape hatch — injected here so tests never hit network.
    """

    def test_brand_name_resolved_via_rxnorm_ingredient(self):
        # "Levophed" is the brand for norepinephrine; it is not in the
        # bootstrap. rxnorm_lookup -> rxcui -> ingredient "norepinephrine"
        # re-matches the curated vasopressors category.
        calls = {"rxnorm": 0, "ingredient": 0}

        def fake_rxnorm_lookup(drug_name, max_results=5):
            calls["rxnorm"] += 1
            return {"status": "ok", "results": [{"rxcui": "7512", "name": "Levophed"}]}

        def fake_ingredient(rxcui):
            calls["ingredient"] += 1
            return "Norepinephrine" if rxcui == "7512" else None

        resolver = DrugCategoryResolver(
            _MAPPINGS_DIR,
            enable_mcp=True,
            rxnorm_lookup=fake_rxnorm_lookup,
            ingredient_resolver=fake_ingredient,
        )
        cats = resolver.resolve("Levophed")
        assert "vasopressors" in _names(cats)
        assert next(c for c in cats if c.name == "vasopressors").source == "rxnorm"
        assert calls["rxnorm"] == 1

    def test_escape_hatch_off_by_default(self):
        # Without enable_mcp, a bootstrap miss returns nothing and makes no
        # MCP call (deterministic, offline) — the unit-test default.
        called = {"n": 0}

        def fake_rxnorm_lookup(drug_name, max_results=5):
            called["n"] += 1
            return {"status": "ok", "results": []}

        resolver = DrugCategoryResolver(
            _MAPPINGS_DIR, rxnorm_lookup=fake_rxnorm_lookup,
        )
        assert resolver.resolve("Levophed") == []
        assert called["n"] == 0

    def test_unavailable_mcp_degrades_to_empty(self):
        def fake_rxnorm_lookup(drug_name, max_results=5):
            return {"status": "unavailable", "error": "OMOPHub not configured"}

        resolver = DrugCategoryResolver(
            _MAPPINGS_DIR, enable_mcp=True, rxnorm_lookup=fake_rxnorm_lookup,
        )
        assert resolver.resolve("Levophed") == []

    def test_curated_hit_skips_escape_hatch(self):
        # A bootstrap hit must not fire the (network) escape hatch.
        called = {"n": 0}

        def fake_rxnorm_lookup(drug_name, max_results=5):
            called["n"] += 1
            return {"status": "ok", "results": []}

        resolver = DrugCategoryResolver(
            _MAPPINGS_DIR, enable_mcp=True, rxnorm_lookup=fake_rxnorm_lookup,
        )
        assert "vasopressors" in _names(resolver.resolve("Norepinephrine"))
        assert called["n"] == 0


class TestMemoization:
    """Repeated administrations of the same drug must not re-call the escape
    hatch — a cohort has many rows per distinct drug.
    """

    def test_resolve_is_memoized_per_drug(self):
        calls = {"n": 0}

        def fake_rxnorm_lookup(drug_name, max_results=5):
            calls["n"] += 1
            return {"status": "ok", "results": [{"rxcui": "7512", "name": "x"}]}

        def fake_ingredient(rxcui):
            return "Norepinephrine"

        resolver = DrugCategoryResolver(
            _MAPPINGS_DIR,
            enable_mcp=True,
            rxnorm_lookup=fake_rxnorm_lookup,
            ingredient_resolver=fake_ingredient,
        )
        resolver.resolve("Levophed")
        resolver.resolve("Levophed")
        assert calls["n"] == 1
