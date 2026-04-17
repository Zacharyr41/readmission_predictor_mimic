"""Tests for ``src.causal.interventions`` — ontology-grounded
intervention-predicate resolution (Phase 8b).

Covers all four resolver paths (RxNorm / SNOMED / ICD-10-PCS / LOINC),
the ``is_control`` semantic, the no-curation correctness guard (failing
loudly instead of silently matching nothing), and the rxnav fallback
behaviour with HTTP mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.causal._rxnav import RxNavClient, RxNavError
from src.causal.interventions import (
    InterventionResolutionError,
    InterventionResolver,
    ResolvedIntervention,
)
from src.conversational.models import InterventionSpec


# ---------------------------------------------------------------------------
# RxNorm path
# ---------------------------------------------------------------------------


class TestRxNormResolution:
    """RxCUI → ``drug_to_snomed.json`` local cache → MIMIC drug name(s).

    Every drug in the repo's registry already carries an RxCUI, so the
    common path never hits rxnav. The slow path (rxnav fallback) is
    exercised separately with HTTP mocked.
    """

    def test_local_cache_hit_for_alteplase(self):
        """RxCUI 8410 (alteplase / tPA) resolves purely from the local
        registry — no rxnav needed."""
        resolver = InterventionResolver()
        spec = InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410")
        resolved = resolver.resolve(spec)
        assert isinstance(resolved, ResolvedIntervention)
        assert resolved.label == "tPA"
        assert resolved.is_control is False
        assert "alteplase" in resolved.params
        assert "prescriptions" in resolved.sql_exists_fragment
        assert resolved.provenance["ontology"] == "RxNorm"
        assert resolved.provenance["resolved_via"] == "local drug_to_snomed.json"
        assert resolved.provenance["rxnav_used"] is False
        assert resolved.provenance["target_rxcui"] == "8410"

    def test_local_cache_hit_for_vancomycin(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(label="vanc", kind="drug", rxnorm_ingredient="11124")
        resolved = resolver.resolve(spec)
        assert "vancomycin" in resolved.params

    def test_unknown_rxcui_falls_through_to_rxnav(self):
        """A RxCUI not in the local registry triggers an rxnav lookup.
        We mock rxnav so the test is offline-safe."""
        fake_rxnav = MagicMock(spec=RxNavClient)
        # Simulate rxnav returning two descendants, one of which is a MIMIC-
        # registered drug name ("acetaminophen" is in drug_to_snomed.json).
        fake_rxnav.get_related_rxcuis.return_value = [
            {"rxcui": "12345", "name": "Acetaminophen 500 MG Oral Tablet", "tty": "SCD"},
            {"rxcui": "67890", "name": "Acetaminophen", "tty": "SBD"},
        ]
        resolver = InterventionResolver(rxnav_client=fake_rxnav)
        spec = InterventionSpec(
            label="analgesic",
            kind="drug",
            rxnorm_ingredient="999999",  # fake RxCUI not in local cache
        )
        resolved = resolver.resolve(spec)
        # At least the normalised "acetaminophen" key should have matched.
        assert "acetaminophen" in resolved.params
        assert resolved.provenance["rxnav_used"] is True
        assert resolved.provenance["rxnav_descendants_returned"] == 2
        assert resolved.provenance["resolved_via"] == "rxnav /related.json"
        fake_rxnav.get_related_rxcuis.assert_called_once_with("999999")

    def test_rxnav_error_surfaces_loudly(self):
        """If rxnav is unreachable AND the RxCUI isn't cached locally,
        the resolver raises rather than producing an empty predicate
        (correctness-first)."""
        failing_rxnav = MagicMock(spec=RxNavClient)
        failing_rxnav.get_related_rxcuis.side_effect = RxNavError("connection timeout")
        resolver = InterventionResolver(rxnav_client=failing_rxnav)
        spec = InterventionSpec(label="unknown", kind="drug", rxnorm_ingredient="999999")
        with pytest.raises(InterventionResolutionError, match="could not be resolved"):
            resolver.resolve(spec)

    def test_rxnav_no_descendants_raises(self):
        empty_rxnav = MagicMock(spec=RxNavClient)
        empty_rxnav.get_related_rxcuis.return_value = []
        resolver = InterventionResolver(rxnav_client=empty_rxnav)
        spec = InterventionSpec(label="nothing", kind="drug", rxnorm_ingredient="999999")
        with pytest.raises(InterventionResolutionError, match="rxnav returned 0 descendants"):
            resolver.resolve(spec)


# ---------------------------------------------------------------------------
# SNOMED path (PRIMARY for drug interventions per user decision)
# ---------------------------------------------------------------------------


class TestSnomedResolution:
    """SNOMED concept → reverse-index over ``drug_to_snomed.json`` →
    MIMIC drug names.

    Without a ``SnomedHierarchy`` JSON the resolver uses just the target
    concept (no descendant expansion); this is a degraded-but-safe
    mode and provenance flags it.
    """

    def test_snomed_vancomycin_resolves_without_hierarchy(self):
        resolver = InterventionResolver()  # no hierarchy JSON
        spec = InterventionSpec(
            label="vanc",
            kind="drug",
            snomed_concept_id="372735009",
        )
        resolved = resolver.resolve(spec)
        assert "vancomycin" in resolved.params
        assert resolved.provenance["ontology"] == "SNOMED-CT"
        assert resolved.provenance["hierarchy_loaded"] is False
        assert resolved.provenance["descendants_expanded"] == 0
        assert resolved.provenance["target_concept_id"] == "372735009"

    def test_snomed_alteplase(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="tPA", kind="drug", snomed_concept_id="387152000",
        )
        resolved = resolver.resolve(spec)
        assert "alteplase" in resolved.params

    def test_snomed_with_hierarchy_expands_descendants(self):
        """A live hierarchy object expands the concept. We stub the
        hierarchy to avoid depending on an on-disk SNOMED dump."""
        stub_hierarchy = MagicMock()
        # Pretend SNOMED class concept 'Anticoagulant therapy' has
        # warfarin as a descendant.
        stub_hierarchy.get_descendants.return_value = ["372756006"]  # warfarin

        resolver = InterventionResolver(hierarchy=stub_hierarchy)
        spec = InterventionSpec(
            label="anticoag_class",
            kind="drug",
            snomed_concept_id="81839001",  # placeholder class concept id
        )
        resolved = resolver.resolve(spec)
        assert "warfarin" in resolved.params
        assert resolved.provenance["hierarchy_loaded"] is True
        assert resolved.provenance["descendants_expanded"] == 1
        stub_hierarchy.get_descendants.assert_called_once_with("81839001")

    def test_unknown_snomed_raises(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="nothing",
            kind="drug",
            snomed_concept_id="99999999",  # no registry entry
        )
        with pytest.raises(InterventionResolutionError, match="no entries"):
            resolver.resolve(spec)


# ---------------------------------------------------------------------------
# ICD-10-PCS path
# ---------------------------------------------------------------------------


class TestIcd10PcsResolution:
    """ICD-10-PCS codes match directly against ``procedures_icd`` —
    no ontology lookup is needed because the code structure is already
    hierarchical (prefix match ≡ subtree match)."""

    def test_exact_code_fragment(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="tpa_iv",
            kind="procedure",
            icd10pcs_code="3E03317",
        )
        resolved = resolver.resolve(spec)
        assert "procedures_icd" in resolved.sql_exists_fragment
        assert "icd_version = 10" in resolved.sql_exists_fragment
        assert resolved.params == ("3E03317", "3E03317%")
        assert resolved.provenance["ontology"] == "ICD-10-PCS"
        assert resolved.provenance["target_code"] == "3E03317"

    def test_dotted_and_lowercase_code_normalised(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="proc",
            kind="procedure",
            icd10pcs_code="3e0.3317",
        )
        resolved = resolver.resolve(spec)
        assert resolved.params == ("3E03317", "3E03317%")


# ---------------------------------------------------------------------------
# LOINC path
# ---------------------------------------------------------------------------


class TestLoincResolution:
    """LOINC → SNOMED (via loinc_to_snomed.json) → MIMIC itemids (via
    labitem_to_snomed.json). Targets a measurement-based intervention
    (lab-driven treatment rule — e.g. "received renal replacement
    therapy for creatinine > X")."""

    def test_loinc_creatinine_resolves_to_labitem(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="had_creatinine_lab",
            kind="measurement",
            loinc_code="2160-0",
        )
        resolved = resolver.resolve(spec)
        assert 50912 in resolved.params  # MIMIC itemid for creatinine
        assert "labevents" in resolved.sql_exists_fragment
        assert resolved.provenance["ontology"] == "LOINC"
        assert resolved.provenance["resolved_via_snomed"] == "113075003"

    def test_unknown_loinc_raises(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="unknown_lab",
            kind="measurement",
            loinc_code="99999-9",
        )
        with pytest.raises(InterventionResolutionError, match="no SNOMED mapping"):
            resolver.resolve(spec)


# ---------------------------------------------------------------------------
# is_control semantics
# ---------------------------------------------------------------------------


class TestControlArmSemantics:
    """``is_control=True`` threads through to the resolved intervention
    unchanged; ``assign_treatments`` is what negates the predicate.
    Here we check only that the flag propagates — assignment behaviour
    is covered in ``test_treatment_assignment.py``."""

    def test_control_flag_propagates(self):
        resolver = InterventionResolver()
        control = InterventionSpec(
            label="no_tPA",
            kind="drug",
            rxnorm_ingredient="8410",
            is_control=True,
        )
        resolved = resolver.resolve(control)
        assert resolved.is_control is True
        # Predicate itself is the positive fragment; negation applied later.
        assert resolved.sql_exists_fragment.startswith("EXISTS")


# ---------------------------------------------------------------------------
# End-to-end: resolved predicate executes on DuckDB
# ---------------------------------------------------------------------------


class TestPredicateExecutesOnDuckDB:
    """Sanity check: the resolved SQL fragment plus the outer-alias
    template produces a runnable query on the synthetic fixture. This
    catches shape bugs (missing alias, wrong param count) that a unit
    test on the fragment alone would miss."""

    def test_snomed_vancomycin_matches_fixture_admission(self, duckdb_backend):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="vanc", kind="drug", snomed_concept_id="372735009",
        )
        resolved = resolver.resolve(spec)
        sql = (
            "SELECT DISTINCT a.hadm_id FROM admissions a "
            f"WHERE {resolved.sql_exists_fragment}"
        )
        rows = duckdb_backend.execute(sql, list(resolved.params))
        hadm_ids = {r[0] for r in rows}
        # Fixture has vancomycin on hadm_id 101 only.
        assert hadm_ids == {101}

    def test_rxnorm_alteplase_matches_fixture_admissions(self, duckdb_backend):
        resolver = InterventionResolver()
        spec = InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410")
        resolved = resolver.resolve(spec)
        sql = (
            "SELECT DISTINCT a.hadm_id FROM admissions a "
            f"WHERE {resolved.sql_exists_fragment}"
        )
        rows = duckdb_backend.execute(sql, list(resolved.params))
        hadm_ids = {r[0] for r in rows}
        # Fixture has alteplase on 101 and 105.
        assert hadm_ids == {101, 105}

    def test_icd10pcs_matches_fixture_procedures(self, duckdb_backend):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="tpa_iv", kind="procedure", icd10pcs_code="3E03317"
        )
        resolved = resolver.resolve(spec)
        sql = (
            "SELECT DISTINCT a.hadm_id FROM admissions a "
            f"WHERE {resolved.sql_exists_fragment}"
        )
        rows = duckdb_backend.execute(sql, list(resolved.params))
        assert {r[0] for r in rows} == {101, 102, 105}

    def test_loinc_creatinine_matches_fixture_labevents(self, duckdb_backend):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="had_creatinine_lab", kind="measurement", loinc_code="2160-0",
        )
        resolved = resolver.resolve(spec)
        sql = (
            "SELECT DISTINCT a.hadm_id FROM admissions a "
            f"WHERE {resolved.sql_exists_fragment}"
        )
        rows = duckdb_backend.execute(sql, list(resolved.params))
        # Fixture has creatinine labs on 101, 103, 106.
        assert {r[0] for r in rows} == {101, 103, 106}


# ---------------------------------------------------------------------------
# Provenance auditability
# ---------------------------------------------------------------------------


class TestProvenanceAuditable:
    """Every resolution records the ontology, version, and every code
    it matched — so a reviewer can reproduce the treatment-assignment
    step without re-running the resolver."""

    def test_rxnorm_provenance_carries_caller_metadata_too(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(
            label="tPA", kind="drug", rxnorm_ingredient="8410",
            provenance={"study_id": "EMR-2026-001", "analyst": "zr"},
        )
        resolved = resolver.resolve(spec)
        assert resolved.provenance["study_id"] == "EMR-2026-001"
        assert resolved.provenance["analyst"] == "zr"
        assert resolved.provenance["ontology"] == "RxNorm"
        assert resolved.provenance["resolver_version"] == "8b-2026-04-17"

    def test_snomed_provenance_lists_every_matched_drug_name(self):
        resolver = InterventionResolver()
        spec = InterventionSpec(label="vanc", kind="drug", snomed_concept_id="372735009")
        resolved = resolver.resolve(spec)
        # The matched-names list is populated so a reviewer can see
        # exactly which MIMIC drug strings made it into the arm.
        assert "vancomycin" in resolved.provenance["matched_drug_names"]
