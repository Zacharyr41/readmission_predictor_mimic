"""Tests for clinician intervention-effect questions (I1–I6 in the smoke test).

Phase history:

  * T1 (2026-04-17): RED-stage, xfail-strict scaffold pinning the target
    behaviour against the formal Neyman–Rubin spec. 4 tests passed
    today (baseline comorbidity phenotype + Q6-shape workaround); 12
    xfailed targeting Phase 8a-h sub-phases.

  * Phase 8a (2026-04-17): schema + planner route + stub dispatch landed.
    9 of the xfails flip to plain asserts here; 3 remain xfailed for
    Phase 8b (end-to-end N-ary), 8c (end-to-end multi-outcome), 8h
    (diagnostics + associative-mode downgrade).

The ``xfail`` markers that remain use ``strict=True`` so the day an
implementation lands they turn into hard failures unless the marker is
removed — each Phase 8 sub-phase PR is forced to flip them explicitly.

Spec section references in class docstrings refer to the formal spec
saved in ``memory/project_phase8_causal_spec.md``.
"""

from __future__ import annotations

import math

import pytest

from src.causal.models import (
    AssumptionClaim,
    CausalEffectResult,
    DiagnosticReport,
    UncertaintyInterval,
)
from src.causal.run import CausalQuestionInvalid, run_causal
from src.conversational.models import (
    AggregationSpec,
    ClinicalConcept,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
    PatientFilter,
    TemporalConstraint,
)


# ---------------------------------------------------------------------------
# Fixture: deterministic CQ builder (mirrors tests/test_conversational/test_planner.py:32)
# ---------------------------------------------------------------------------


def _cq(
    *,
    concepts: list[tuple[str, str]] | None = None,
    filters: list[tuple[str, str, str]] | None = None,
    aggregation: str | None = None,
    scope: str = "cohort",
    comparison_field: str | None = None,
    temporal: list[tuple[str, str, str | None]] | None = None,
    return_type: str = "text_and_table",
    intervention_set: list[InterventionSpec] | None = None,
    outcome_vector: list[OutcomeSpec] | None = None,
    aggregation_spec: AggregationSpec | None = None,
    alpha: float = 0.05,
) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="test",
        clinical_concepts=[
            ClinicalConcept(name=n, concept_type=t) for n, t in (concepts or [])
        ],
        patient_filters=[
            PatientFilter(field=f, operator=o, value=v)
            for f, o, v in (filters or [])
        ],
        temporal_constraints=[
            TemporalConstraint(relation=r, reference_event=ev, time_window=tw)
            for r, ev, tw in (temporal or [])
        ],
        aggregation=aggregation,
        scope=scope,
        comparison_field=comparison_field,
        return_type=return_type,
        intervention_set=intervention_set,
        outcome_vector=outcome_vector,
        aggregation_spec=aggregation_spec,
        alpha=alpha,
    )


# Canonical InterventionSpec builders used across the causal tests. Kept
# tiny on purpose — the ontology code-resolution pipeline lands in 8b; for
# 8a we only care that the schema accepts a valid ontology-coded spec.

def _ispec_tpa() -> InterventionSpec:
    return InterventionSpec(
        label="tPA",
        kind="drug",
        rxnorm_ingredient="8410",  # RxCUI for alteplase — example, real lookup in 8b
        provenance={"ontology": "RxNorm", "resolved_via": "test-fixture"},
    )


def _ispec_no_tpa() -> InterventionSpec:
    return InterventionSpec(
        label="no_tPA",
        kind="drug",
        rxnorm_ingredient="8410",
        is_control=True,
        provenance={"ontology": "RxNorm", "resolved_via": "test-fixture", "inverted": True},
    )


def _ispec_vanc_ptz() -> InterventionSpec:
    return InterventionSpec(
        label="vanc+pip-tazo",
        kind="drug",
        rxnorm_ingredient="11124",  # vancomycin — illustrative
        provenance={"ontology": "RxNorm", "resolved_via": "test-fixture"},
    )


def _ispec_vanc_cefepime() -> InterventionSpec:
    return InterventionSpec(
        label="vanc+cefepime",
        kind="drug",
        rxnorm_ingredient="2191",  # cefepime — illustrative
        provenance={"ontology": "RxNorm", "resolved_via": "test-fixture"},
    )


def _ispec_vanc_meropenem() -> InterventionSpec:
    return InterventionSpec(
        label="vanc+meropenem",
        kind="drug",
        rxnorm_ingredient="29561",  # meropenem — illustrative
        provenance={"ontology": "RxNorm", "resolved_via": "test-fixture"},
    )


def _ispec_warfarin() -> InterventionSpec:
    return InterventionSpec(label="warfarin", kind="drug", rxnorm_ingredient="11289")


def _ispec_doac() -> InterventionSpec:
    return InterventionSpec(label="doac", kind="drug", rxnorm_ingredient="1037042")


def _ispec_no_anticoag() -> InterventionSpec:
    return InterventionSpec(
        label="no_anticoag",
        kind="drug",
        rxnorm_ingredient="11289",
        is_control=True,
    )


def _outcome_readmit30() -> OutcomeSpec:
    return OutcomeSpec(
        name="readmitted_30d",
        outcome_type="binary",
        extractor_key="readmitted_30d",
        higher_is_better=False,
    )


def _outcome_mortality28d() -> OutcomeSpec:
    return OutcomeSpec(
        name="mortality_28d",
        outcome_type="binary",
        extractor_key="mortality_28d",
        higher_is_better=False,
    )


# ---------------------------------------------------------------------------
# 1. Comorbidity phenotype composition — passes today (spec §5 population predicate φ_A)
# ---------------------------------------------------------------------------


class TestComorbidityPhenotypeRouting:
    """Multi-diagnosis cohort composition (I1) routes SQL_FAST.

    Corresponds to smoke question I1: "Among sepsis patients over 65 with a
    history of diabetes, what was the mean peak creatinine?" Stacked
    ``diagnosis`` filters + age filter + biomarker aggregate.

    This is the Phase 8-independent baseline — the formal spec's population
    predicate φ_A composes from the existing filter machinery with no new
    code.
    """

    def test_two_diagnoses_plus_age_plus_biomarker_routes_sql_fast(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[
                ("diagnosis", "contains", "sepsis"),
                ("diagnosis", "contains", "diabetes"),
                ("age", ">", "65"),
            ],
            aggregation="mean",
            scope="cohort",
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.SQL_FAST, (
            f"I1 comorbidity phenotype must route SQL_FAST, got {plan}"
        )

    def test_three_diagnoses_still_routes_sql_fast(self):
        """I6's cohort predicate: AFib + stroke + CKD. The planner is
        unaware of how many diagnoses the clinician stacked — it just
        routes on shape."""
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[
                ("diagnosis", "contains", "atrial fibrillation"),
                ("diagnosis", "contains", "stroke"),
                ("diagnosis", "contains", "chronic kidney disease"),
            ],
            aggregation="mean",
            scope="cohort",
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.SQL_FAST


# ---------------------------------------------------------------------------
# 2. Binary intervention via 2-CQ workaround — positive leg passes today
# ---------------------------------------------------------------------------


class TestInterventionWorkaroundRouting:
    """The documented 2-CQ workaround's positive leg routes SQL_FAST.

    Corresponds to smoke question I2 leg-a: "How many stroke patients who
    received tPA were readmitted within 30 days?" — the existing
    `drug` concept + `diagnosis` filter + `readmitted_30d` filter + count.

    The negative leg ("who did NOT receive tPA") has no representation in
    the current schema — the clarifying-question path handles it. Phase 8a
    (I3) replaces this workaround with a single `CAUSAL` CQ.
    """

    def test_drug_count_with_diagnosis_and_readmitted_filters_routes_sql_fast(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = _cq(
            concepts=[("tPA", "drug")],
            filters=[
                ("diagnosis", "contains", "stroke"),
                ("readmitted_30d", "=", "1"),
            ],
            aggregation="count",
            scope="cohort",
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.SQL_FAST

    def test_antibiotic_count_in_sepsis_cohort_routes_sql_fast(self):
        """I5's workaround leg: drug count within a diagnosis cohort."""
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = _cq(
            concepts=[("vancomycin", "drug")],
            filters=[("diagnosis", "contains", "septic shock")],
            aggregation="count",
            scope="cohort",
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.SQL_FAST


# ---------------------------------------------------------------------------
# 3. Causal CQ schema — PHASE 8a LANDED (spec §§2, 3, 4, 6)
# ---------------------------------------------------------------------------


class TestCausalCQSchema:
    """The ``CompetencyQuestion`` schema carries the spec's causal inputs.

    Spec §6 inputs: cohort D, intervention set I with |I| ≥ 2, aggregation
    g, query point x, population predicate φ_A, significance level α,
    declared assumption set, optional data-source flag.

    Phase 8a added the scope Literal entry ``"causal_effect"`` plus the
    new fields. These tests were xfailed in the RED stage; they now pass.
    """

    def test_binary_intervention_single_outcome_identity_aggregation(self):
        """I3 shape: (C=2, n=1, g=id)."""
        cq = CompetencyQuestion(
            original_question="Compare 30-day readmission between ischemic-stroke "
            "patients who received tPA and those who didn't.",
            scope="causal_effect",
            clinical_concepts=[
                ClinicalConcept(name="readmitted_30d", concept_type="outcome"),
            ],
            patient_filters=[
                PatientFilter(field="diagnosis", operator="contains", value="ischemic stroke"),
            ],
            intervention_set=[_ispec_tpa(), _ispec_no_tpa()],
            outcome_vector=[_outcome_readmit30()],
            aggregation_spec=AggregationSpec(kind="identity"),
            alpha=0.05,
        )
        assert cq.scope == "causal_effect"
        assert len(cq.intervention_set) == 2
        assert len(cq.outcome_vector) == 1
        assert cq.aggregation_spec.kind == "identity"
        assert cq.alpha == 0.05
        assert cq.mode == "associative"  # default
        assert cq.estimator_family == "t_learner"
        assert cq.uncertainty_method == "bootstrap"

    def test_nary_intervention_schema(self):
        """I5 shape: (C=3, n=1, g=id)."""
        cq = CompetencyQuestion(
            original_question="Compare 28-day mortality across three empiric "
            "antibiotic regimens in septic shock.",
            scope="causal_effect",
            clinical_concepts=[ClinicalConcept(name="mortality_28d", concept_type="outcome")],
            patient_filters=[PatientFilter(field="diagnosis", operator="contains", value="septic shock")],
            intervention_set=[_ispec_vanc_ptz(), _ispec_vanc_cefepime(), _ispec_vanc_meropenem()],
            outcome_vector=[_outcome_mortality28d()],
            aggregation_spec=AggregationSpec(kind="identity"),
        )
        assert len(cq.intervention_set) == 3
        labels = [i.label for i in cq.intervention_set]
        assert labels == ["vanc+pip-tazo", "vanc+cefepime", "vanc+meropenem"]

    def test_multi_outcome_composite_schema(self):
        """I6 shape: (C=3, n=3, g=weighted)."""
        outcomes = [
            OutcomeSpec(name="stroke_recurrence_30d", outcome_type="binary",
                        extractor_key="stroke_recurrence_30d"),
            OutcomeSpec(name="major_bleeding", outcome_type="binary",
                        extractor_key="major_bleeding"),
            OutcomeSpec(name="mortality_90d", outcome_type="time_to_event",
                        extractor_key="mortality_90d",
                        censoring_clock="admission", censoring_horizon_days=90),
        ]
        cq = CompetencyQuestion(
            original_question="Compare warfarin vs DOAC vs no-anticoagulation "
            "across a composite of 30-day stroke recurrence, major bleeding, "
            "and 90-day mortality for a 72yo with AFib + new stroke + CKD.",
            scope="causal_effect",
            intervention_set=[_ispec_warfarin(), _ispec_doac(), _ispec_no_anticoag()],
            outcome_vector=outcomes,
            aggregation_spec=AggregationSpec(
                kind="weighted_sum",
                weights={
                    "stroke_recurrence_30d": 1.0,
                    "major_bleeding": 1.0,
                    "mortality_90d": 2.0,
                },
            ),
        )
        assert len(cq.outcome_vector) == 3
        assert cq.aggregation_spec is not None
        assert cq.aggregation_spec.kind == "weighted_sum"
        # Survival outcome correctly typed
        assert cq.outcome_vector[2].outcome_type == "time_to_event"
        assert cq.outcome_vector[2].censoring_horizon_days == 90


# ---------------------------------------------------------------------------
# 4. InterventionSpec ontology-code validation — PHASE 8a (spec §1; no-curation rule)
# ---------------------------------------------------------------------------


class TestInterventionSpecValidation:
    """The no-curation rule (``feedback_correctness_no_curation.md``) is
    enforced at the schema level: every intervention must carry exactly
    one ontology code, never a free-text synonym list."""

    def test_zero_ontology_codes_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="exactly one ontology code"):
            InterventionSpec(label="tPA", kind="drug")

    def test_multiple_ontology_codes_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="exactly one ontology code"):
            InterventionSpec(
                label="tPA", kind="drug",
                rxnorm_ingredient="8410",
                snomed_concept_id="387367007",
            )

    def test_rxnorm_only_accepted(self):
        spec = InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410")
        assert spec.rxnorm_ingredient == "8410"
        assert spec.snomed_concept_id is None

    def test_snomed_only_accepted(self):
        spec = InterventionSpec(label="tPA", kind="drug", snomed_concept_id="387367007")
        assert spec.snomed_concept_id == "387367007"

    def test_icd10pcs_only_accepted(self):
        spec = InterventionSpec(
            label="thrombolytic-iv",
            kind="procedure",
            icd10pcs_code="3E03317",
        )
        assert spec.icd10pcs_code == "3E03317"

    def test_provenance_is_free_form_dict(self):
        spec = InterventionSpec(
            label="tPA", kind="drug", rxnorm_ingredient="8410",
            provenance={"ontology": "RxNorm", "version": "2024-07",
                        "resolved_via": "rxnav", "descendants": 37},
        )
        assert spec.provenance["ontology"] == "RxNorm"
        assert spec.provenance["descendants"] == 37


# ---------------------------------------------------------------------------
# 5. AggregationSpec validation — PHASE 8a (spec §3)
# ---------------------------------------------------------------------------


class TestAggregationSpecValidation:
    def test_identity_default(self):
        spec = AggregationSpec()
        assert spec.kind == "identity"

    def test_weighted_sum_requires_weights(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="weights"):
            AggregationSpec(kind="weighted_sum")

    def test_dominant_requires_priority(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="priority"):
            AggregationSpec(kind="dominant")

    def test_identity_rejects_spurious_params(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="must not carry"):
            AggregationSpec(kind="identity", weights={"x": 1.0})


# ---------------------------------------------------------------------------
# 6. Causal planner routing — PHASE 8a LANDED (spec §6 input routing)
# ---------------------------------------------------------------------------


class TestCausalPlannerRouting:
    """A well-formed causal CQ routes to ``QueryPlan.CAUSAL``.

    Well-formed means: ``scope="causal_effect"`` and ``|I| ≥ 2``.
    Degenerate causal CQs (|I| < 2) fall back to SQL_FAST / GRAPH so the
    system still answers something instead of erroring.
    """

    def test_binary_intervention_routes_causal(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = CompetencyQuestion(
            original_question="I3: tPA vs no-tPA, 30-day readmission in ischemic stroke",
            scope="causal_effect",
            intervention_set=[_ispec_tpa(), _ispec_no_tpa()],
            outcome_vector=[_outcome_readmit30()],
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.CAUSAL

    def test_nary_intervention_routes_causal(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = CompetencyQuestion(
            original_question="I5: three antibiotic regimens, 28-day mortality",
            scope="causal_effect",
            intervention_set=[_ispec_vanc_ptz(), _ispec_vanc_cefepime(), _ispec_vanc_meropenem()],
            outcome_vector=[_outcome_mortality28d()],
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.CAUSAL

    def test_multi_outcome_composite_routes_causal(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = CompetencyQuestion(
            original_question="I6: AFib + stroke + CKD → warfarin vs DOAC vs none; composite",
            scope="causal_effect",
            intervention_set=[_ispec_warfarin(), _ispec_doac(), _ispec_no_anticoag()],
            outcome_vector=[_outcome_readmit30()],  # shape-only for 8a
            aggregation_spec=AggregationSpec(
                kind="weighted_sum",
                weights={"readmitted_30d": 1.0},
            ),
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.CAUSAL

    def test_degenerate_single_arm_falls_back(self):
        """A causal-scoped CQ with |I|=1 is degenerate — the planner drops
        through to the non-causal classifier so the system still answers."""
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = CompetencyQuestion(
            original_question="Degenerate: one 'intervention' only",
            scope="causal_effect",
            intervention_set=[_ispec_tpa()],
            outcome_vector=[_outcome_readmit30()],
        )
        plan = QueryPlanner().classify(cq)
        assert plan in (QueryPlan.SQL_FAST, QueryPlan.GRAPH)

    def test_empty_intervention_set_falls_back(self):
        """A causal-scoped CQ with no interventions at all also drops
        through. This is the shape the LLM might emit if extraction
        misfires — must not crash the pipeline."""
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = CompetencyQuestion(
            original_question="No interventions declared",
            scope="causal_effect",
        )
        plan = QueryPlanner().classify(cq)
        assert plan in (QueryPlan.SQL_FAST, QueryPlan.GRAPH)


# ---------------------------------------------------------------------------
# 7. Causal module is importable + result contract — PHASE 8a LANDED (spec §7)
# ---------------------------------------------------------------------------


class TestCausalEstimandOutputShape:
    """``CausalEffectResult`` exposes the spec §7 output contract and
    ``run_causal`` (stub in 8a, real estimators in 8d+) returns instances
    that conform to it."""

    def test_causal_module_is_importable(self):
        import src.causal  # noqa: F401

    def test_result_contract_fields_present(self):
        """The spec §7 fields must exist on the pydantic model."""
        fields = set(CausalEffectResult.model_fields.keys())
        required = {
            "mu_c", "mu_c_k", "tau_cc_prime",
            "ranking", "diagnostics",
            "mode", "assumption_ledger",
        }
        missing = required - fields
        assert not missing, f"CausalEffectResult missing spec §7 fields: {missing}"

    def test_stub_returns_well_shaped_result_for_binary(self):
        """I3 shape: 2 interventions × 1 outcome ⇒ |mu_c|=2, |mu_c_k|=2,
        |tau|=1 (C(2,2)=1). Stub uses NaN points but shape must be right."""
        cq = CompetencyQuestion(
            original_question="I3 stub",
            scope="causal_effect",
            intervention_set=[_ispec_tpa(), _ispec_no_tpa()],
            outcome_vector=[_outcome_readmit30()],
        )
        result = run_causal(cq)
        assert isinstance(result, CausalEffectResult)
        assert result.is_stub is True
        assert result.mode == "associative"
        assert set(result.mu_c.keys()) == {"tPA", "no_tPA"}
        assert set(result.mu_c_k.keys()) == {"tPA|readmitted_30d", "no_tPA|readmitted_30d"}
        assert set(result.tau_cc_prime.keys()) == {"no_tPA|tPA"}  # lexicographic
        assert math.isnan(result.mu_c["tPA"].point)
        # Assumption ledger carries a row per spec §5 assumption — all
        # declared in 8a because no real diagnostic ran.
        names = {a.name for a in result.assumption_ledger}
        assert names == {"consistency", "ignorability", "positivity", "sutva"}
        assert all(a.status == "declared" for a in result.assumption_ledger)

    def test_stub_tau_key_encoding_uses_lexicographic_order(self):
        """Pair identity is stable: whether the caller thinks of an arm as
        'c' or 'c_prime', the tau key is the same string."""
        cq1 = CompetencyQuestion(
            original_question="ordering A",
            scope="causal_effect",
            intervention_set=[_ispec_tpa(), _ispec_no_tpa()],
            outcome_vector=[_outcome_readmit30()],
        )
        cq2 = CompetencyQuestion(
            original_question="ordering B",
            scope="causal_effect",
            intervention_set=[_ispec_no_tpa(), _ispec_tpa()],  # reversed
            outcome_vector=[_outcome_readmit30()],
        )
        result1 = run_causal(cq1)
        result2 = run_causal(cq2)
        assert set(result1.tau_cc_prime.keys()) == set(result2.tau_cc_prime.keys())

    def test_run_causal_rejects_non_causal_scope(self):
        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            aggregation="mean",
            scope="cohort",
        )
        with pytest.raises(CausalQuestionInvalid, match="scope='causal_effect'"):
            run_causal(cq)

    def test_run_causal_rejects_single_intervention(self):
        cq = CompetencyQuestion(
            original_question="degenerate",
            scope="causal_effect",
            intervention_set=[_ispec_tpa()],
            outcome_vector=[_outcome_readmit30()],
        )
        with pytest.raises(CausalQuestionInvalid, match="at least 2 interventions"):
            run_causal(cq)

    def test_run_causal_rejects_empty_outcome_vector(self):
        cq = CompetencyQuestion(
            original_question="no outcomes",
            scope="causal_effect",
            intervention_set=[_ispec_tpa(), _ispec_no_tpa()],
        )
        with pytest.raises(CausalQuestionInvalid, match="non-empty outcome_vector"):
            run_causal(cq)

    def test_run_causal_rejects_duplicate_intervention_labels(self):
        dup1 = InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410")
        dup2 = InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="1234")
        cq = CompetencyQuestion(
            original_question="dup labels",
            scope="causal_effect",
            intervention_set=[dup1, dup2],
            outcome_vector=[_outcome_readmit30()],
        )
        with pytest.raises(CausalQuestionInvalid, match="labels must be unique"):
            run_causal(cq)

    # ---- Remaining xfails for later sub-phases ------------------------

    @pytest.mark.xfail(
        strict=True,
        reason="Phase 8b — end-to-end real estimates for N-ary (stub returns NaN)",
        raises=AssertionError,
    )
    def test_run_causal_returns_real_n_arms_for_c_equals_3(self):
        """I5 shape: three antibiotic regimens with real (non-NaN) estimates."""
        cq = CompetencyQuestion(
            original_question="I5",
            scope="causal_effect",
            intervention_set=[_ispec_vanc_ptz(), _ispec_vanc_cefepime(), _ispec_vanc_meropenem()],
            outcome_vector=[_outcome_mortality28d()],
        )
        result = run_causal(cq)
        assert len(result.mu_c) == 3
        assert len(result.tau_cc_prime) == 3  # C(3,2)
        # Real estimator: no NaNs.
        for ui in result.mu_c.values():
            assert not math.isnan(ui.point)

    @pytest.mark.xfail(
        strict=True,
        reason="Phase 8c — per-outcome breakdown μ_{c,k} with real estimates (stub returns NaN)",
        raises=AssertionError,
    )
    def test_run_causal_returns_real_per_outcome_breakdown_for_n_equals_3(self):
        """I6 shape: (C=3, n=3) — μ_{c,k} has 3×3=9 real entries."""
        outcomes = [
            OutcomeSpec(name="stroke_recurrence_30d", outcome_type="binary",
                        extractor_key="stroke_recurrence_30d"),
            OutcomeSpec(name="major_bleeding", outcome_type="binary",
                        extractor_key="major_bleeding"),
            OutcomeSpec(name="mortality_90d", outcome_type="time_to_event",
                        extractor_key="mortality_90d",
                        censoring_clock="admission", censoring_horizon_days=90),
        ]
        cq = CompetencyQuestion(
            original_question="I6",
            scope="causal_effect",
            intervention_set=[_ispec_warfarin(), _ispec_doac(), _ispec_no_anticoag()],
            outcome_vector=outcomes,
            aggregation_spec=AggregationSpec(
                kind="weighted_sum",
                weights={"stroke_recurrence_30d": 1.0, "major_bleeding": 1.0, "mortality_90d": 2.0},
            ),
        )
        result = run_causal(cq)
        assert len(result.mu_c_k) == 9
        for ui in result.mu_c_k.values():
            assert not math.isnan(ui.point)

    @pytest.mark.xfail(
        strict=True,
        reason="Phase 8h — causal/associative mode + assumption-ledger diagnostics (stub declares all)",
        raises=AssertionError,
    )
    def test_positivity_failure_triggers_associative_mode(self):
        """Diagnostics catch poor overlap and downgrade mode to associative,
        populating the assumption ledger with the failed assumption."""
        cq = CompetencyQuestion(
            original_question="poor-overlap case",
            scope="causal_effect",
            intervention_set=[_ispec_tpa(), _ispec_no_tpa()],
            outcome_vector=[_outcome_readmit30()],
            mode="causal",  # caller asserts but diagnostics must reject
        )
        result = run_causal(cq)
        # After 8h: mode must end up "associative" because the real
        # positivity check fails. The stub returns "associative"
        # unconditionally, so the interesting assertion is on ledger
        # statuses.
        assert result.mode == "associative"
        positivity_claim = next(a for a in result.assumption_ledger if a.name == "positivity")
        assert positivity_claim.status == "failed"  # stub has "declared" → test xfails
