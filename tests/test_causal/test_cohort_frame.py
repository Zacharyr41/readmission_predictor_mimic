"""Tests for ``src.causal.cohort.build_cohort_frame`` (Phase 8c).

End-to-end assembly: a causal ``CompetencyQuestion`` plus a backend
plus a cohort produces a fully-merged DataFrame ready for the 8d
estimator.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.causal.cohort import CausalCohortError, CohortFrame, build_cohort_frame
from src.causal.interventions import InterventionResolver
from src.causal.outcomes import get_default_registry
from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_tpa_cq() -> CompetencyQuestion:
    """I3 shape."""
    return CompetencyQuestion(
        original_question="I3",
        scope="causal_effect",
        intervention_set=[
            InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ],
        outcome_vector=[
            OutcomeSpec(name="readmitted_30d", outcome_type="binary",
                        extractor_key="readmitted_30d"),
        ],
        aggregation_spec=AggregationSpec(kind="identity"),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestBinaryCohortAssembly:
    def test_returns_cohort_frame_with_expected_shape(self, duckdb_backend):
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        assert isinstance(cf, CohortFrame)
        assert cf.treatment_column == "T"
        assert cf.intervention_labels == ["tPA", "no_tPA"]
        # Binary negation gives clean mutual exclusivity → all 6 retained.
        assert len(cf.df) == 6

    def test_covariate_columns_present(self, duckdb_backend):
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        for col in ("age", "gender_M", "gender_F", "gender_unknown"):
            assert col in cf.df.columns
            assert col in cf.covariate_columns

    def test_outcome_columns_present(self, duckdb_backend):
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        assert "readmitted_30d" in cf.df.columns
        assert cf.outcome_columns == ["readmitted_30d"]

    def test_T_values_match_treatment_assignment(self, duckdb_backend):
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        by_hadm = cf.df.set_index("hadm_id")["intervention_label"].to_dict()
        # Fixture alteplase exposure: 101 and 105.
        assert by_hadm[101] == "tPA"
        assert by_hadm[105] == "tPA"
        for h in (102, 103, 104, 106):
            assert by_hadm[h] == "no_tPA"

    def test_outcomes_filled_correctly(self, duckdb_backend):
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        by_hadm = cf.df.set_index("hadm_id")["readmitted_30d"].to_dict()
        # Subject 1's 101→102 gap = 21 days → 101 is readmitted_30d=1
        assert by_hadm[101] == 1
        for h in (102, 103, 104, 105, 106):
            assert by_hadm[h] == 0

    def test_provenance_lists_resolver_and_outcome_metadata(self, duckdb_backend):
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        prov = cf.provenance
        assert prov["cohort_hadm_ids_requested"] == 6
        assert prov["n_final_rows"] == 6
        assert prov["covariate_profile"] == "demographics"
        assert prov["outcomes"][0]["name"] == "readmitted_30d"
        assert prov["outcomes"][0]["extractor_key"] == "readmitted_30d"
        assert "resolver_version" in prov
        # Treatment-assignment counts are propagated.
        assert prov["n_assigned"] == 6
        assert prov["n_unassigned"] == 0
        assert prov["n_overlapping"] == 0


# ---------------------------------------------------------------------------
# N-ary (I5 shape, 3 positive arms with overlap)
# ---------------------------------------------------------------------------


class TestNaryCohortAssembly:
    def test_overlap_rows_dropped_not_assigned(self, duckdb_backend):
        """When three positive arms are used, admissions matching ≥2 of
        them drop from the final frame (n_matching captured in
        treatment_assignment)."""
        cq = CompetencyQuestion(
            original_question="three-arm",
            scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="alteplase", kind="drug", rxnorm_ingredient="8410"),
                InterventionSpec(label="warfarin", kind="drug", rxnorm_ingredient="11289"),
                InterventionSpec(label="vancomycin", kind="drug", rxnorm_ingredient="11124"),
            ],
            outcome_vector=[
                OutcomeSpec(name="readmitted_30d", outcome_type="binary",
                            extractor_key="readmitted_30d"),
            ],
        )
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        # Fixture exposure:
        #   101: vanc + alteplase (2 → overlap, dropped)
        #   102: warfarin only (retained)
        #   103: ceftriaxone (0 → dropped as unassigned)
        #   104: nothing
        #   105: alteplase + warfarin (2 → overlap, dropped)
        #   106: ceftriaxone (0 → dropped)
        assert set(cf.df["hadm_id"]) == {102}
        assert cf.provenance["n_overlapping"] == 2  # 101, 105
        assert cf.provenance["n_unassigned"] == 3   # 103, 104, 106


# ---------------------------------------------------------------------------
# Multi-outcome (I6 shape)
# ---------------------------------------------------------------------------


class TestMultiOutcomeCohortAssembly:
    def test_survival_outcome_contributes_time_and_event_columns(self, duckdb_backend):
        """I6's 90-day mortality is time_to_event — cohort frame carries
        two columns (<name>_time, <name>_event) for that outcome."""
        cq = CompetencyQuestion(
            original_question="I6 shape",
            scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
                InterventionSpec(
                    label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
                ),
            ],
            outcome_vector=[
                OutcomeSpec(name="readmitted_30d", outcome_type="binary",
                            extractor_key="readmitted_30d"),
                OutcomeSpec(name="mortality_90d",
                            outcome_type="time_to_event",
                            extractor_key="mortality_time_to_event",
                            censoring_horizon_days=90),
            ],
            aggregation_spec=AggregationSpec(
                kind="weighted_sum",
                weights={"readmitted_30d": 1.0, "mortality_90d": 2.0},
            ),
        )
        cf = build_cohort_frame(
            cq, duckdb_backend,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        assert "readmitted_30d" in cf.df.columns
        assert "mortality_90d_time" in cf.df.columns
        assert "mortality_90d_event" in cf.df.columns
        assert set(cf.outcome_columns) == {
            "readmitted_30d", "mortality_90d_time", "mortality_90d_event"
        }
        # Admission 106 — died within 90 days.
        mask = cf.df["hadm_id"] == 106
        assert int(cf.df.loc[mask, "mortality_90d_event"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# Preconditions / error paths
# ---------------------------------------------------------------------------


class TestPreconditions:
    def test_non_causal_scope_rejected(self, duckdb_backend):
        cq = CompetencyQuestion(
            original_question="wrong scope",
            scope="cohort",
            intervention_set=[
                InterventionSpec(label="a", kind="drug", rxnorm_ingredient="8410"),
                InterventionSpec(label="b", kind="drug", rxnorm_ingredient="11289"),
            ],
            outcome_vector=[
                OutcomeSpec(name="r30", outcome_type="binary", extractor_key="readmitted_30d"),
            ],
        )
        with pytest.raises(CausalCohortError, match="causal_effect"):
            build_cohort_frame(cq, duckdb_backend, cohort_hadm_ids=[101])

    def test_single_intervention_rejected(self, duckdb_backend):
        cq = CompetencyQuestion(
            original_question="degenerate",
            scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="a", kind="drug", rxnorm_ingredient="8410"),
            ],
            outcome_vector=[
                OutcomeSpec(name="r30", outcome_type="binary", extractor_key="readmitted_30d"),
            ],
        )
        with pytest.raises(CausalCohortError, match="≥ 2 interventions"):
            build_cohort_frame(cq, duckdb_backend, cohort_hadm_ids=[101])

    def test_empty_outcome_vector_rejected(self, duckdb_backend):
        cq = CompetencyQuestion(
            original_question="no outcomes",
            scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="a", kind="drug", rxnorm_ingredient="8410"),
                InterventionSpec(
                    label="b", kind="drug", rxnorm_ingredient="8410", is_control=True,
                ),
            ],
        )
        with pytest.raises(CausalCohortError, match="non-empty outcome_vector"):
            build_cohort_frame(cq, duckdb_backend, cohort_hadm_ids=[101])

    def test_empty_cohort_rejected(self, duckdb_backend):
        cq = _binary_tpa_cq()
        with pytest.raises(CausalCohortError, match="no admissions"):
            build_cohort_frame(cq, duckdb_backend, cohort_hadm_ids=[])

    def test_zero_assigned_rejected(self, duckdb_backend):
        """Cohort admission 103 has only ceftriaxone — doesn't match
        either alteplase-or-warfarin positive arm, doesn't match either
        is_control negation (since both controls target the opposite
        ontology code). Actually both controls do match for 103 — it
        didn't receive alteplase AND didn't receive warfarin — so this
        is actually an overlap case. Pick a more specific cohort to
        force n_assigned=0."""
        cq = CompetencyQuestion(
            original_question="unassigned",
            scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="alteplase_only", kind="drug", rxnorm_ingredient="8410"),
                InterventionSpec(label="warfarin_only", kind="drug", rxnorm_ingredient="11289"),
            ],
            outcome_vector=[
                OutcomeSpec(name="r30", outcome_type="binary", extractor_key="readmitted_30d"),
            ],
        )
        # Admission 104 has no drugs at all → n_matching=0 for both arms.
        with pytest.raises(CausalCohortError, match="no cohort admission matched"):
            build_cohort_frame(cq, duckdb_backend, cohort_hadm_ids=[104])


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


class TestDependencyInjection:
    def test_custom_resolver_is_used(self, duckdb_backend):
        custom = InterventionResolver()
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            resolver=custom,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        assert len(cf.df) == 6

    def test_custom_outcome_registry_is_used(self, duckdb_backend):
        from src.causal.outcomes import OutcomeExtractor, OutcomeRegistry

        # A registry that defines only readmitted_30d; build_cohort_frame
        # should use it without consulting the default.
        custom = OutcomeRegistry()
        default = get_default_registry()
        custom.register(default.get("readmitted_30d"))
        cq = _binary_tpa_cq()
        cf = build_cohort_frame(
            cq, duckdb_backend,
            outcome_registry=custom,
            cohort_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        assert "readmitted_30d" in cf.outcome_columns
