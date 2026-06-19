"""Tests for QueryPlanner — routes CQs between SQL-fast-path and graph-path.

Phase 7a: the planner is the single decision point deciding whether a CQ can
be answered by a direct SQL aggregate (skipping extract+graph+reason) or
needs the RDF graph.

These tests are behavioural: they assert the plan decision for a table of
CQ shapes, and they rely on the default ``OperationRegistry`` for what
counts as a "supported aggregate" and "supported comparison axis". Adding
new aggregates / axes to the registry should widen the fast-path
automatically without edits here.
"""

from __future__ import annotations

import pytest

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
    ReturnType,
    TemporalConstraint,
)


# ---------------------------------------------------------------------------
# Fixture: deterministic CQ builder
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
    )


# ---------------------------------------------------------------------------
# 1. QueryPlan enum exists
# ---------------------------------------------------------------------------


class TestQueryPlanEnum:
    def test_plan_has_sql_fast_and_graph(self):
        from src.conversational.planner import QueryPlan

        assert QueryPlan.SQL_FAST
        assert QueryPlan.GRAPH
        assert QueryPlan.SQL_FAST != QueryPlan.GRAPH


# ---------------------------------------------------------------------------
# 2. Classification table — fast-path cases
# ---------------------------------------------------------------------------


_FAST_PATH_CASES = [
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            filters=[("age", ">", "65")],
            aggregation="mean", scope="cohort"),
        id="mean_biomarker_with_age_filter",
    ),
    pytest.param(
        _cq(concepts=[("heart rate", "vital")],
            aggregation="max", scope="cohort"),
        id="max_vital_no_filter",
    ),
    pytest.param(
        _cq(concepts=[("lactate", "biomarker")],
            aggregation="min", scope="cohort"),
        id="min_biomarker",
    ),
    pytest.param(
        _cq(concepts=[("sepsis", "diagnosis")],
            aggregation="count", scope="cohort"),
        id="count_diagnosis",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            aggregation="mean", scope="comparison",
            comparison_field="gender"),
        id="comparison_mean_by_gender",
    ),
    pytest.param(
        _cq(concepts=[("lactate", "biomarker")],
            aggregation="mean", scope="comparison",
            comparison_field="readmitted_30d"),
        id="comparison_mean_by_readmitted_30d",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            filters=[("subject_id", "=", "12345")],
            aggregation="mean", scope="single_patient"),
        id="single_patient_aggregate",
    ),
    pytest.param(
        _cq(concepts=[("mortality", "outcome")],
            aggregation="count", scope="cohort"),
        id="mortality_count",
    ),
    pytest.param(
        _cq(concepts=[("propofol", "drug")],
            aggregation="count", scope="cohort"),
        id="drug_count",
    ),
    pytest.param(
        _cq(concepts=[("MRSA", "microbiology")],
            aggregation="count", scope="cohort"),
        id="microbiology_count",
    ),
    # Part A: window/anchor temporal constraints are SQL-bound-able. "during
    # the ICU stay" is a charttime WHERE on icustays.intime/outtime, not Allen
    # reasoning — it must NOT veto the fast-path.
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("during", "ICU stay", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_during_icu_window",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("during", "hospital admission", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_during_admission_window",
    ),
    pytest.param(
        _cq(concepts=[("lactate", "biomarker")],
            temporal=[("within", "admission", "24h")],
            aggregation="mean", scope="cohort"),
        id="temporal_within_24h_admission",
    ),
    pytest.param(
        _cq(concepts=[("heart rate", "vital")],
            temporal=[("during", "ICU stay", None)],
            aggregation="max", scope="cohort"),
        id="temporal_window_vital",
    ),
    pytest.param(
        _cq(concepts=[("vancomycin", "drug")],
            temporal=[("during", "ICU stay", None)],
            aggregation="count", scope="cohort"),
        id="temporal_window_drug_count",
    ),
    pytest.param(
        _cq(concepts=[("MRSA", "microbiology")],
            temporal=[("during", "ICU stay", None)],
            aggregation="count", scope="cohort"),
        id="temporal_window_microbiology_count",
    ),
]


@pytest.mark.parametrize("cq", _FAST_PATH_CASES)
def test_fast_path_cases_route_to_sql_fast(cq):
    from src.conversational.planner import QueryPlan, QueryPlanner

    plan = QueryPlanner().classify(cq)
    assert plan == QueryPlan.SQL_FAST, (
        f"expected SQL_FAST, got {plan}. CQ: {cq.model_dump()}"
    )


# ---------------------------------------------------------------------------
# 3. Classification table — graph-path cases
# ---------------------------------------------------------------------------


_GRAPH_PATH_CASES = [
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            aggregation="median", scope="cohort"),
        id="median_needs_python_postprocessor",
    ),
    # Part A: relational/Allen constraints (before/after an arbitrary clinical
    # *event*, not a structural interval) genuinely need the graph and stay.
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("before", "intubation", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_before_event_relational",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("after", "dialysis", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_after_event_relational",
    ),
    # A window constraint mixed with a relational one cannot be split — the
    # whole CQ falls to the graph.
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("during", "ICU stay", None), ("before", "intubation", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_mixed_window_and_relational",
    ),
    # diagnosis/outcome have no temporal bounding in the extractor, so a
    # window constraint on them must stay on the graph (else the SQL path
    # would silently ignore the constraint — parity break).
    pytest.param(
        _cq(concepts=[("sepsis", "diagnosis")],
            temporal=[("during", "ICU stay", None)],
            aggregation="count", scope="cohort"),
        id="temporal_window_on_diagnosis_stays_graph",
    ),
    # A window constraint over >1 concept still can't fast-path (rule 6).
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker"), ("lactate", "biomarker")],
            temporal=[("during", "ICU stay", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_window_multi_concept",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker"), ("lactate", "biomarker")],
            aggregation="mean", scope="cohort"),
        id="multiple_concepts",
    ),
    pytest.param(
        _cq(concepts=[("lactate", "biomarker")],
            return_type="visualization", scope="single_patient",
            filters=[("subject_id", "=", "123")]),
        id="visualization_without_aggregate",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            aggregation="mean", scope="comparison",
            comparison_field="some_unregistered_axis"),
        id="comparison_with_unregistered_axis",
    ),
    pytest.param(
        _cq(concepts=[("lactate", "biomarker")],
            aggregation="sum", scope="cohort"),
        id="sum_aggregate_today_has_no_sql_fn",
    ),
]


@pytest.mark.parametrize("cq", _GRAPH_PATH_CASES)
def test_graph_path_cases_route_to_graph(cq):
    from src.conversational.planner import QueryPlan, QueryPlanner

    plan = QueryPlanner().classify(cq)
    assert plan == QueryPlan.GRAPH, (
        f"expected GRAPH, got {plan}. CQ: {cq.model_dump()}"
    )


# ---------------------------------------------------------------------------
# 2b. classify() is a faithful plan-only view of explain()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cq", _FAST_PATH_CASES + _GRAPH_PATH_CASES)
def test_classify_equals_explain_plan(cq):
    """``classify`` is a thin wrapper that returns ``explain(cq).plan``. Pin
    that contract so a future edit to ``explain`` can't silently desync the two
    (the decision log + audit read ``explain``; the orchestrator reads its
    ``.plan``). Also confirm the decision carries a reason whose ``rule`` is a
    real §4.1 row."""
    from src.conversational.planner import QueryPlanner, RoutingReason

    planner = QueryPlanner()
    decision = planner.explain(cq)
    assert planner.classify(cq) == decision.plan
    assert isinstance(decision.reason, RoutingReason)
    assert 1 <= decision.rule <= 14
    assert decision.detail


# ---------------------------------------------------------------------------
# 3b. Part A — the temporal veto is split by kind of constraint
# ---------------------------------------------------------------------------


class TestTemporalVetoSplit:
    """Rule 4 no longer vetoes *every* temporal constraint. Window/anchor
    constraints (ICU stay / hospital admission) fall through to the normal
    single-concept SQL legs; only relational/Allen constraints veto to GRAPH."""

    def test_window_constraint_falls_through_to_sql(self):
        from src.conversational.planner import (
            QueryPlan,
            QueryPlanner,
            RoutingReason,
        )

        cq = _cq(concepts=[("creatinine", "biomarker")],
                 temporal=[("during", "ICU stay", None)],
                 aggregation="mean", scope="cohort")
        decision = QueryPlanner().explain(cq)
        assert decision.plan == QueryPlan.SQL_FAST
        # It reaches the generic fall-through, not a temporal-specific leg.
        assert decision.reason == RoutingReason.FALLTHROUGH_SQL_FAST

    def test_relational_constraint_still_vetoes(self):
        from src.conversational.planner import (
            QueryPlan,
            QueryPlanner,
            RoutingReason,
        )

        cq = _cq(concepts=[("creatinine", "biomarker")],
                 temporal=[("before", "intubation", None)],
                 aggregation="mean", scope="cohort")
        decision = QueryPlanner().explain(cq)
        assert decision.plan == QueryPlan.GRAPH
        assert decision.reason == RoutingReason.TEMPORAL_CONSTRAINTS_PRESENT


# ---------------------------------------------------------------------------
# 4. Registry-driven: coverage expands with new registrations
# ---------------------------------------------------------------------------


class TestRegistryDriven:
    def test_every_registered_aggregate_with_sql_fn_is_fast_path(self):
        """An aggregate that declares ``sql_fn`` must route to SQL_FAST
        for a minimal biomarker-only CQ. This is the widening contract:
        add a new aggregate with sql_fn, and it automatically gains
        fast-path eligibility."""
        from src.conversational.operations import get_default_registry
        from src.conversational.planner import QueryPlan, QueryPlanner

        registry = get_default_registry()
        planner = QueryPlanner()
        for name in registry.supported_names("aggregate"):
            op = registry.require("aggregate", name)
            if getattr(op, "sql_fn", None) is None:
                continue  # graph-only aggregates (median, etc.)
            cq = _cq(
                concepts=[("creatinine", "biomarker")],
                aggregation=name,
                scope="cohort",
            )
            assert planner.classify(cq) == QueryPlan.SQL_FAST, (
                f"aggregate {name!r} declares sql_fn but planner routed to GRAPH"
            )

    def test_every_comparison_axis_with_sql_descriptor_is_fast_path(self):
        """A comparison axis that declares a SQL GROUP BY descriptor must
        route to SQL_FAST for a minimal comparison CQ."""
        from src.conversational.operations import get_default_registry
        from src.conversational.planner import QueryPlan, QueryPlanner

        registry = get_default_registry()
        planner = QueryPlanner()
        for name in registry.supported_names("comparison_axis"):
            op = registry.require("comparison_axis", name)
            if getattr(op, "sql_group_by", None) is None:
                continue
            cq = _cq(
                concepts=[("creatinine", "biomarker")],
                aggregation="mean",
                scope="comparison",
                comparison_field=name,
            )
            assert planner.classify(cq) == QueryPlan.SQL_FAST, (
                f"comparison axis {name!r} declares sql_group_by "
                f"but planner routed to GRAPH"
            )


# ---------------------------------------------------------------------------
# 5. classify() is deterministic on every decomposer_cases fixture
# ---------------------------------------------------------------------------


class TestClassificationIsDeterministic:
    """Every fixture CQ (from the Phase 6 regression suite) must produce a
    plan decision without raising. We don't assert a specific plan — we
    only check determinism. The plan fields are added later as explicit
    ``expected_plan`` annotations on fixtures that want them pinned."""

    def test_every_decomposer_case_classifies_cleanly(self):
        from tests.test_conversational.conftest import load_decomposer_cases
        from src.conversational.planner import QueryPlanner

        planner = QueryPlanner()
        for param in load_decomposer_cases():
            case = param.values[0]
            expected_cq_dict = case["expected_cq"]
            cq = CompetencyQuestion.model_validate(expected_cq_dict)
            # Two calls should agree; classify() must be pure.
            assert planner.classify(cq) == planner.classify(cq)

    def test_decomposer_cases_with_expected_plan_are_pinned(self):
        """Decomposer fixtures may carry an optional ``expected_plan`` key that
        pins the route their post-processed CQ must take. This exercises the
        decomposer→planner seam with *real* decomposer output (vs. the
        hand-built CQs in the routing corpus). Fixtures without the key are
        unaffected (the loader ignores unknown keys)."""
        from tests.test_conversational.conftest import load_decomposer_cases
        from src.conversational.planner import QueryPlan, QueryPlanner

        planner = QueryPlanner()
        pinned = 0
        for param in load_decomposer_cases():
            case = param.values[0]
            if "expected_plan" not in case:
                continue
            cq = CompetencyQuestion.model_validate(case["expected_cq"])
            expected = QueryPlan(case["expected_plan"])
            assert planner.classify(cq) == expected, (
                f"{case.get('name')!r}: expected_plan={expected}, "
                f"got {planner.classify(cq)}"
            )
            pinned += 1
        assert pinned >= 1, "no decomposer case carries expected_plan yet"


# ---------------------------------------------------------------------------
# New SQL-fast-path routes: split-by-condition comparison + event_ordering
# ---------------------------------------------------------------------------


class TestSplitByConditionRouting:
    """``comparison_field='condition'`` carries no fixed ``sql_group_by`` (the
    GROUP BY is built from ``split_condition`` at compile time), so the planner
    must route it to SQL_FAST when a split_condition is present, and to GRAPH
    when it's missing (underspecified)."""

    def _cq(self, *, split_condition):
        return CompetencyQuestion(
            original_question="test",
            clinical_concepts=[
                ClinicalConcept(name="in-hospital mortality", concept_type="outcome")
            ],
            aggregation="count",
            scope="comparison",
            comparison_field="condition",
            split_condition=split_condition,
        )

    def test_condition_with_split_routes_to_sql_fast(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = self._cq(split_condition=PatientFilter(
            field="diagnosis", operator="contains",
            value="chronic anticoagulant use",
        ))
        assert QueryPlanner().classify(cq) == QueryPlan.SQL_FAST

    def test_condition_without_split_routes_to_graph(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = self._cq(split_condition=None)
        assert QueryPlanner().classify(cq) == QueryPlan.GRAPH


class TestEventOrderingRouting:
    """``aggregation='event_ordering'`` is a multi-event SQL-fast-path op; it must
    route to SQL_FAST with ≥2 concepts (despite having no ``sql_fn`` and >1
    concept, which the generic guards reject)."""

    def _cq(self, concepts):
        return CompetencyQuestion(
            original_question="test",
            clinical_concepts=[
                ClinicalConcept(name=n, concept_type=t) for n, t in concepts
            ],
            aggregation="event_ordering",
            scope="cohort",
        )

    def test_event_ordering_with_two_events_routes_to_sql_fast(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = self._cq([("intubation", "procedure"), ("mannitol", "drug")])
        assert QueryPlanner().classify(cq) == QueryPlan.SQL_FAST

    def test_event_ordering_with_one_event_routes_to_graph(self):
        from src.conversational.planner import QueryPlan, QueryPlanner

        cq = self._cq([("intubation", "procedure")])
        assert QueryPlanner().classify(cq) == QueryPlan.GRAPH
