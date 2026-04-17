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
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("during", "ICU stay", None)],
            aggregation="mean", scope="cohort"),
        id="temporal_during_constraint",
    ),
    pytest.param(
        _cq(concepts=[("creatinine", "biomarker")],
            temporal=[("before", "intubation", None)],
            scope="cohort"),
        id="temporal_before_constraint",
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
