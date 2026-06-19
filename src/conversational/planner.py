"""Query planner — routes a CompetencyQuestion between the SQL fast-path
and the graph-path.

Phase 7a: the motivator for this module is cohort-scale BigQuery latency.
For a question like *"average creatinine for patients over 65"* the old
pipeline ran the full extract→build-graph→SPARQL sequence, which is
dominated by hundreds of sequential BigQuery round-trips and a graph
build that touches every event. For purely-aggregate questions the answer
is a single SQL query with ``AVG/MAX/MIN/COUNT`` — the graph adds no
value.

The planner decides once, per CompetencyQuestion, which path to take. A
CQ is eligible for the SQL fast-path if:

  - Its aggregate declares a portable ``sql_fn`` (AVG/MAX/MIN/COUNT).
    Median is excluded — it needs Python post-processing.
  - There is exactly one clinical concept of a type the fast-path knows
    how to resolve directly in SQL (biomarker, vital, drug, diagnosis,
    microbiology, outcome).
  - Any ``temporal_constraints`` are *window/anchor* constraints — bounded by
    the ICU-stay or hospital-admission interval the database records, so they
    compile to a ``charttime`` ``WHERE`` bound (Part A). *Relational/Allen*
    constraints (``before``/``after`` an arbitrary clinical *event*) still
    require the graph; the split is decided by
    ``src.conversational.temporal.is_sql_window``.
  - If the scope is "comparison", the axis is a registered
    ``comparison_axis`` operation whose ``sql_group_by`` is set.

A separate class of metadata-only CQs (patient_demographics,
admission_details, icu_length_of_stay, mortality_count, diagnosis lists
that are really just SELECTs over admissions/diagnoses) is also SQL-fast
eligible — these are handled by the fast-path compiler directly without
needing an aggregate at all.

Everything else falls through to the graph path, which is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.conversational.models import CompetencyQuestion
from src.conversational.operations import OperationRegistry, get_default_registry
from src.conversational.temporal import (
    WINDOW_BOUNDABLE_CONCEPT_TYPES,
    is_sql_window,
)


class QueryPlan(Enum):
    """The execution paths a CQ can take."""

    SQL_FAST = "sql_fast"
    """Answered by a single SQL query. Skip extract + graph + SPARQL."""

    GRAPH = "graph"
    """Requires the RDF knowledge graph (Allen relations, median, time-series,
    multi-event reasoning, or an aggregate/axis that isn't SQL-compilable)."""

    CAUSAL = "causal"
    """Phase 8: routed to ``src.causal.run_causal``. A well-formed causal CQ
    carries ``scope="causal_effect"`` plus an intervention set of size
    |I| ≥ 2. Degenerate shapes (no intervention set, |I| < 2) fall back to
    the SQL_FAST / GRAPH dispatch above."""

    SIMILARITY = "similarity"
    """Phase 9: routed to ``src.similarity.run.run_similarity``. A CQ with
    ``scope="patient_similarity"`` and a populated ``similarity_spec``
    produces a ranked list of similar patients with contextual + temporal
    explanations. A causal CQ with ``similarity_spec`` STAYS on
    ``CAUSAL`` — ``run_causal`` then consumes the spec as a cohort-
    narrowing directive rather than a standalone query."""


# Concept types whose events live in a single MIMIC table the fast-path can
# aggregate over directly. Extending this needs a matching branch in
# ``sql_fastpath.compile_sql``.
_SQL_FAST_CONCEPT_TYPES: frozenset[str] = frozenset({
    "biomarker", "vital", "drug", "diagnosis", "microbiology", "outcome",
})


class RoutingReason(Enum):
    """*Why* the planner chose a plan — one member per distinct ``return`` leg of
    the :meth:`QueryPlanner.explain` cascade.

    These map 1:1 to the rule table in ``querytriagesystem.md`` §4.1 (the
    ``rule`` attribute is that table's row number). Note that rule 3
    (``event_ordering``) and rule 12 (``condition`` axis) each have *two* legs —
    an SQL_FAST leg and a GRAPH leg — so there are 16 reasons over 14 rules.

    The enum *value* is a stable snake_case string code, safe to serialize into
    the decision log; ``RoutingReason.X.rule`` is the numeric rule.
    """

    SCOPE_PATIENT_SIMILARITY = ("scope_patient_similarity", 1)
    CAUSAL_WELL_FORMED = ("causal_well_formed", 2)
    EVENT_ORDERING_MULTI_CONCEPT = ("event_ordering_multi_concept", 3)
    EVENT_ORDERING_UNDERSPECIFIED = ("event_ordering_underspecified", 3)
    TEMPORAL_CONSTRAINTS_PRESENT = ("temporal_constraints_present", 4)
    NO_CLINICAL_CONCEPTS = ("no_clinical_concepts", 5)
    MULTI_CONCEPT = ("multi_concept", 6)
    CONCEPT_TYPE_UNSUPPORTED = ("concept_type_unsupported", 7)
    DIAGNOSIS_LIST = ("diagnosis_list", 8)
    RAW_VALUE_NON_DIAGNOSIS = ("raw_value_non_diagnosis", 9)
    AGGREGATE_NO_SQL_FN = ("aggregate_no_sql_fn", 10)
    COMPARISON_NO_FIELD = ("comparison_no_field", 11)
    CONDITION_SPLIT_MISSING = ("condition_split_missing", 12)
    CONDITION_SPLIT_PRESENT = ("condition_split_present", 12)
    COMPARISON_AXIS_UNREGISTERED = ("comparison_axis_unregistered", 13)
    FALLTHROUGH_SQL_FAST = ("fallthrough_sql_fast", 14)

    rule: int

    def __new__(cls, code: str, rule: int) -> "RoutingReason":
        obj = object.__new__(cls)
        obj._value_ = code
        obj.rule = rule
        return obj


@dataclass(frozen=True)
class RoutingDecision:
    """The full result of classifying a CQ: the chosen ``plan`` plus *why*.

    :meth:`QueryPlanner.classify` returns only ``.plan`` for backward
    compatibility; callers that want the rationale (the decision log, the routing
    audit) call :meth:`QueryPlanner.explain` and read ``.reason`` / ``.rule`` /
    ``.detail``. ``rule`` is derived from ``reason`` so the two can never drift.
    """

    plan: QueryPlan
    reason: RoutingReason
    detail: str = ""

    @property
    def rule(self) -> int:
        """The ``querytriagesystem.md`` §4.1 rule-table number that fired."""
        return self.reason.rule


class QueryPlanner:
    """Stateless classifier. Inject a custom ``OperationRegistry`` in tests
    that want to observe the widening contract (register a new aggregate →
    fast-path coverage expands automatically)."""

    def __init__(self, registry: OperationRegistry | None = None) -> None:
        self._registry = registry or get_default_registry()

    def classify(self, cq: CompetencyQuestion) -> QueryPlan:
        """Return the plan decision for ``cq``.

        Thin plan-only wrapper over :meth:`explain` (which also carries the
        reason). Pure; does not mutate state. Called once per sub-CQ in the
        orchestrator.
        """
        return self.explain(cq).plan

    def explain(self, cq: CompetencyQuestion) -> RoutingDecision:
        """Return the full routing decision for ``cq`` — the chosen plan *and*
        the reason it was chosen.

        This is the routing cascade itself; :meth:`classify` is the plan-only
        wrapper over it. Pure; does not mutate state. The first matching rule
        wins, so the rule order below is load-bearing (e.g. ``event_ordering``
        must short-circuit before the temporal veto).
        """
        # Phase 9: standalone similarity CQ. Takes priority over the
        # causal check because a similarity-only CQ has no intervention
        # set — the causal branch would fall through anyway. A CAUSAL
        # CQ that also carries a similarity_spec stays on CAUSAL
        # (cohort narrowing there, not standalone).
        if cq.scope == "patient_similarity":
            return RoutingDecision(
                QueryPlan.SIMILARITY,
                RoutingReason.SCOPE_PATIENT_SIMILARITY,
                "scope=patient_similarity",
            )

        # Phase 8a: causal route takes priority when the CQ is well-formed
        # for causal inference. A causal CQ with |I| < 2 is degenerate
        # (only one "intervention") — fall through to the non-causal path
        # so the system still answers *something* instead of erroring.
        if cq.scope == "causal_effect":
            interventions = cq.intervention_set or []
            if len(interventions) >= 2:
                return RoutingDecision(
                    QueryPlan.CAUSAL,
                    RoutingReason.CAUSAL_WELL_FORMED,
                    f"scope=causal_effect with {len(interventions)} interventions",
                )
            # Degenerate: drop through to the legacy classifier so the
            # existing aggregate / graph branches can still handle it.

        # event_ordering is a multi-event SQL-fast-path operation: it carries
        # ≥2 clinical_concepts and asks for their temporal ORDER, so it must be
        # routed BEFORE the single-concept / aggregate guards below (which would
        # otherwise send it to the graph). The compiler's dedicated
        # ``_compile_event_ordering`` branch handles it; it needs ≥2 concepts.
        if cq.aggregation == "event_ordering":
            if len(cq.clinical_concepts) >= 2:
                return RoutingDecision(
                    QueryPlan.SQL_FAST,
                    RoutingReason.EVENT_ORDERING_MULTI_CONCEPT,
                    f"event_ordering over {len(cq.clinical_concepts)} concepts",
                )
            return RoutingDecision(
                QueryPlan.GRAPH,
                RoutingReason.EVENT_ORDERING_UNDERSPECIFIED,
                "event_ordering needs ≥2 concepts",
            )

        # Part A: split the temporal veto by *kind* of constraint. A
        # window/anchor constraint ("during the ICU stay", "first 24h of
        # admission") is a plain ``charttime`` bound against a structural
        # interval the database records — the fast-path compiles it via the
        # SAME generator the graph extractor uses, so parity holds. Only
        # relational/Allen constraints (``before``/``after`` an arbitrary
        # clinical *event*) still need the graph.
        #
        # Fall through ONLY when *every* constraint is a recognized window AND
        # the single concept is an event-valued type the compiler bounds; any
        # relational constraint, a mixed list, a non-bound-able concept type, or
        # >1 concept vetoes to the graph (the relational one can't be split out,
        # and diagnosis/outcome have no temporal bound — see
        # ``WINDOW_BOUNDABLE_CONCEPT_TYPES``).
        if cq.temporal_constraints:
            all_window = all(is_sql_window(tc) for tc in cq.temporal_constraints)
            concept_ok = (
                len(cq.clinical_concepts) == 1
                and cq.clinical_concepts[0].concept_type
                in WINDOW_BOUNDABLE_CONCEPT_TYPES
            )
            if not (all_window and concept_ok):
                rels = ", ".join(tc.relation for tc in cq.temporal_constraints)
                return RoutingDecision(
                    QueryPlan.GRAPH,
                    RoutingReason.TEMPORAL_CONSTRAINTS_PRESENT,
                    f"temporal_constraints present ({rels})",
                )
            # else: window-bounded event aggregate — fall through to the normal
            # single-concept SQL eligibility checks below.

        # Metadata-only CQs (mortality_count etc.) have no clinical concepts
        # and no aggregation — the fast-path still handles them via dedicated
        # branches in the compiler. These are narrow; keep them on graph-path
        # for now and widen in a follow-up.
        if not cq.clinical_concepts:
            return RoutingDecision(
                QueryPlan.GRAPH,
                RoutingReason.NO_CLINICAL_CONCEPTS,
                "no clinical_concepts (metadata-only)",
            )

        # Fast-path requires exactly one concept of a supported type.
        if len(cq.clinical_concepts) != 1:
            return RoutingDecision(
                QueryPlan.GRAPH,
                RoutingReason.MULTI_CONCEPT,
                f"{len(cq.clinical_concepts)} concepts (fast-path needs exactly 1)",
            )
        if cq.clinical_concepts[0].concept_type not in _SQL_FAST_CONCEPT_TYPES:
            return RoutingDecision(
                QueryPlan.GRAPH,
                RoutingReason.CONCEPT_TYPE_UNSUPPORTED,
                f"concept_type={cq.clinical_concepts[0].concept_type!r} "
                "not in fast-path set",
            )

        # Aggregation must be SQL-compilable. None aggregation + diagnosis
        # concept is a plain patient-list query — also fast-path compilable.
        if cq.aggregation is None:
            if cq.clinical_concepts[0].concept_type == "diagnosis":
                # Bare "list patients with X" — handled by the fast-path's
                # diagnosis-list branch.
                return RoutingDecision(
                    QueryPlan.SQL_FAST,
                    RoutingReason.DIAGNOSIS_LIST,
                    "bare diagnosis list (no aggregation)",
                )
            # Raw-values query: could be SQL too, but the answerer/UI
            # pattern for raw values expects the graph's timestamp handling.
            # Keep on graph-path for now.
            return RoutingDecision(
                QueryPlan.GRAPH,
                RoutingReason.RAW_VALUE_NON_DIAGNOSIS,
                "raw-value lookup (no aggregation) kept on graph",
            )

        agg_op = self._registry.get("aggregate", cq.aggregation)
        if agg_op is None or getattr(agg_op, "sql_fn", None) is None:
            return RoutingDecision(
                QueryPlan.GRAPH,
                RoutingReason.AGGREGATE_NO_SQL_FN,
                f"aggregate={cq.aggregation!r} has no sql_fn",
            )

        # Comparison scope: axis must be a registered comparison_axis with
        # a SQL group-by descriptor.
        if cq.scope == "comparison":
            if not cq.comparison_field:
                return RoutingDecision(
                    QueryPlan.GRAPH,
                    RoutingReason.COMPARISON_NO_FIELD,
                    "comparison scope with no comparison_field",
                )
            # The dynamic ``condition`` axis carries no fixed ``sql_group_by``
            # (the GROUP BY column is built from ``split_condition`` at compile
            # time), so it's SQL-fast-path-eligible as long as a split_condition
            # is supplied. Without one it's underspecified → graph path.
            if cq.comparison_field == "condition":
                if cq.split_condition is None:
                    return RoutingDecision(
                        QueryPlan.GRAPH,
                        RoutingReason.CONDITION_SPLIT_MISSING,
                        "condition axis with no split_condition",
                    )
                return RoutingDecision(
                    QueryPlan.SQL_FAST,
                    RoutingReason.CONDITION_SPLIT_PRESENT,
                    "condition axis with split_condition",
                )
            axis_op = self._registry.get("comparison_axis", cq.comparison_field)
            if axis_op is None or getattr(axis_op, "sql_group_by", None) is None:
                return RoutingDecision(
                    QueryPlan.GRAPH,
                    RoutingReason.COMPARISON_AXIS_UNREGISTERED,
                    f"comparison axis {cq.comparison_field!r} not SQL-compilable",
                )

        return RoutingDecision(
            QueryPlan.SQL_FAST,
            RoutingReason.FALLTHROUGH_SQL_FAST,
            "single-concept SQL aggregate",
        )
