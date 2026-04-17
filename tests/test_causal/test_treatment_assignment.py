"""Tests for ``src.causal.treatment_assignment`` — per-admission
treatment vector with mutual-exclusivity enforcement (Phase 8b).

Covers: clean binary assignment (tPA vs no-tPA), N-ary arms, overlap
detection and exclusion, the empty-cohort edge case, and the
"fewer than two arms" precondition.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.causal.interventions import InterventionResolver
from src.causal.treatment_assignment import (
    InsufficientInterventionsError,
    TreatmentAssignment,
    assign_treatments,
)
from src.conversational.models import InterventionSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_all(
    resolver: InterventionResolver, specs: list[InterventionSpec]
) -> list:
    return [resolver.resolve(s) for s in specs]


# ---------------------------------------------------------------------------
# Binary: tPA vs no-tPA with perfect mutual exclusivity
# ---------------------------------------------------------------------------


class TestBinaryAssignment:
    """The canonical I3 shape. Every cohort admission matches EXACTLY
    one arm by construction (since no_tPA is the negation of tPA)."""

    def test_binary_is_control_yields_perfect_mutual_exclusivity(self, duckdb_backend):
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA",
                kind="drug",
                rxnorm_ingredient="8410",
                is_control=True,
            ),
        ]
        resolved = _resolve_all(resolver, specs)
        cohort = [101, 102, 103, 104, 105, 106]  # full fixture
        assignment = assign_treatments(duckdb_backend, resolved, cohort)

        assert isinstance(assignment, TreatmentAssignment)
        assert assignment.intervention_labels == ["tPA", "no_tPA"]
        assert assignment.n_cohort == len(cohort)
        # Every admission matches exactly one arm — the negation shape
        # guarantees this; overlap count must be zero.
        assert assignment.n_overlapping == 0
        assert assignment.n_unassigned == 0
        assert assignment.n_assigned == len(cohort)

    def test_binary_assigns_tpa_only_to_exposed_admissions(self, duckdb_backend):
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA",
                kind="drug",
                rxnorm_ingredient="8410",
                is_control=True,
            ),
        ]
        resolved = _resolve_all(resolver, specs)
        assignment = assign_treatments(
            duckdb_backend, resolved,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        by_hadm = assignment.df.set_index("hadm_id")["intervention_label"].to_dict()
        # Fixture exposure: alteplase on 101 and 105.
        assert by_hadm[101] == "tPA"
        assert by_hadm[105] == "tPA"
        # Others are negations.
        for hadm in (102, 103, 104, 106):
            assert by_hadm[hadm] == "no_tPA"


# ---------------------------------------------------------------------------
# N-ary: three positive arms; overlap detected and excluded
# ---------------------------------------------------------------------------


class TestNaryAssignment:
    def test_three_positive_arms_capture_overlap(self, duckdb_backend):
        """Three positive-exposure arms (no is_control): admissions
        that received >1 arm are correctly excluded for overlap."""
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(label="alteplase", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(label="warfarin", kind="drug", rxnorm_ingredient="11289"),
            InterventionSpec(label="vancomycin", kind="drug", rxnorm_ingredient="11124"),
        ]
        resolved = _resolve_all(resolver, specs)
        cohort = [101, 102, 103, 104, 105, 106]
        assignment = assign_treatments(duckdb_backend, resolved, cohort)

        by_hadm = {row["hadm_id"]: row for _, row in assignment.df.iterrows()}
        # 101: alteplase + vancomycin → 2 matches → overlap
        assert by_hadm[101]["n_matching"] == 2
        assert by_hadm[101]["intervention_label"] is None
        # 102: warfarin only
        assert by_hadm[102]["intervention_label"] == "warfarin"
        # 103: vancomycin only (ceftriaxone isn't in our arm list)
        assert by_hadm[103]["intervention_label"] is None  # ceftriaxone → 0 matches
        assert by_hadm[103]["n_matching"] == 0
        # 104: no drugs at all → unassigned
        assert by_hadm[104]["n_matching"] == 0
        # 105: alteplase + warfarin → overlap
        assert by_hadm[105]["n_matching"] == 2
        # 106: ceftriaxone → 0 of our arms
        assert by_hadm[106]["n_matching"] == 0

        # Aggregate counts
        assert assignment.n_overlapping == 2  # 101 + 105
        assert assignment.n_unassigned == 3  # 103 (ceftriaxone), 104, 106
        assert assignment.n_assigned == 1  # 102 only

    def test_per_arm_match_counts(self, duckdb_backend):
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(label="alteplase", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(label="warfarin", kind="drug", rxnorm_ingredient="11289"),
        ]
        resolved = _resolve_all(resolver, specs)
        assignment = assign_treatments(
            duckdb_backend, resolved, cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        # Pre-exclusion per-arm counts (including admissions that will
        # be excluded for overlap — so alteplase has 2, warfarin has 2).
        assert assignment.per_arm_matched == {"alteplase": 2, "warfarin": 2}


# ---------------------------------------------------------------------------
# ICD-10-PCS + drug combination — procedure-anchored arms
# ---------------------------------------------------------------------------


class TestProcedureArmAssignment:
    def test_icd10pcs_arm_plus_control(self, duckdb_backend):
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(
                label="had_thrombolytic_iv",
                kind="procedure",
                icd10pcs_code="3E03317",
            ),
            InterventionSpec(
                label="no_thrombolytic_iv",
                kind="procedure",
                icd10pcs_code="3E03317",
                is_control=True,
            ),
        ]
        resolved = _resolve_all(resolver, specs)
        assignment = assign_treatments(
            duckdb_backend, resolved, cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        by_hadm = assignment.df.set_index("hadm_id")["intervention_label"].to_dict()
        # Fixture has thrombolytic_iv on 101, 102, 105.
        for hadm in (101, 102, 105):
            assert by_hadm[hadm] == "had_thrombolytic_iv"
        for hadm in (103, 104, 106):
            assert by_hadm[hadm] == "no_thrombolytic_iv"


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


class TestPreconditions:
    def test_fewer_than_two_arms_raises(self, duckdb_backend):
        resolver = InterventionResolver()
        spec = InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410")
        resolved = [resolver.resolve(spec)]
        with pytest.raises(InsufficientInterventionsError, match="≥2 arms"):
            assign_treatments(duckdb_backend, resolved, [101])

    def test_empty_cohort_returns_empty_frame(self, duckdb_backend):
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ]
        resolved = _resolve_all(resolver, specs)
        assignment = assign_treatments(duckdb_backend, resolved, cohort_hadm_ids=[])
        assert len(assignment.df) == 0
        assert assignment.n_assigned == 0
        assert assignment.n_overlapping == 0
        assert assignment.n_cohort == 0
        # Labels still reported so downstream UI can show empty arms.
        assert assignment.intervention_labels == ["tPA", "no_tPA"]

    def test_cohort_restricted_by_hadm_id_filter(self, duckdb_backend):
        """Cohort subset: only admissions 101 and 102 are in scope. The
        assignment must not include other fixture admissions."""
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ]
        resolved = _resolve_all(resolver, specs)
        assignment = assign_treatments(
            duckdb_backend, resolved, cohort_hadm_ids=[101, 102],
        )
        assert set(assignment.df["hadm_id"]) == {101, 102}
        assert assignment.n_cohort == 2


# ---------------------------------------------------------------------------
# Provenance round-trip
# ---------------------------------------------------------------------------


class TestProvenanceRoundTrip:
    def test_assignment_carries_every_resolved_provenance(self, duckdb_backend):
        resolver = InterventionResolver()
        specs = [
            InterventionSpec(
                label="tPA", kind="drug", rxnorm_ingredient="8410",
                provenance={"study_id": "CAUSAL-TEST-001"},
            ),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410",
                is_control=True,
                provenance={"study_id": "CAUSAL-TEST-001"},
            ),
        ]
        resolved = _resolve_all(resolver, specs)
        assignment = assign_treatments(
            duckdb_backend, resolved, cohort_hadm_ids=[101, 102],
        )
        resolved_provs = assignment.provenance["resolved_interventions"]
        assert len(resolved_provs) == 2
        for prov in resolved_provs:
            assert prov["study_id"] == "CAUSAL-TEST-001"
            assert prov["ontology"] == "RxNorm"
            assert prov["resolver_version"] == "8b-2026-04-17"
