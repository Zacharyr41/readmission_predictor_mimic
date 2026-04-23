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
  - There are no ``temporal_constraints``. Allen relations live in the
    graph; lifting them into SQL would require explicit interval joins
    and lose the semantic intent.
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

from enum import Enum

from src.conversational.models import CompetencyQuestion
from src.conversational.operations import OperationRegistry, get_default_registry


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


class QueryPlanner:
    """Stateless classifier. Inject a custom ``OperationRegistry`` in tests
    that want to observe the widening contract (register a new aggregate →
    fast-path coverage expands automatically)."""

    def __init__(self, registry: OperationRegistry | None = None) -> None:
        self._registry = registry or get_default_registry()

    def classify(self, cq: CompetencyQuestion) -> QueryPlan:
        """Return the plan decision for ``cq``.

        Pure; does not mutate state. Called once per sub-CQ in the
        orchestrator.
        """
        # Phase 9: standalone similarity CQ. Takes priority over the
        # causal check because a similarity-only CQ has no intervention
        # set — the causal branch would fall through anyway. A CAUSAL
        # CQ that also carries a similarity_spec stays on CAUSAL
        # (cohort narrowing there, not standalone).
        if cq.scope == "patient_similarity":
            return QueryPlan.SIMILARITY

        # Phase 8a: causal route takes priority when the CQ is well-formed
        # for causal inference. A causal CQ with |I| < 2 is degenerate
        # (only one "intervention") — fall through to the non-causal path
        # so the system still answers *something* instead of erroring.
        if cq.scope == "causal_effect":
            interventions = cq.intervention_set or []
            if len(interventions) >= 2:
                return QueryPlan.CAUSAL
            # Degenerate: drop through to the legacy classifier so the
            # existing aggregate / graph branches can still handle it.

        # Temporal constraints always require the graph.
        if cq.temporal_constraints:
            return QueryPlan.GRAPH

        # Metadata-only CQs (mortality_count etc.) have no clinical concepts
        # and no aggregation — the fast-path still handles them via dedicated
        # branches in the compiler. These are narrow; keep them on graph-path
        # for now and widen in a follow-up.
        if not cq.clinical_concepts:
            return QueryPlan.GRAPH

        # Fast-path requires exactly one concept of a supported type.
        if len(cq.clinical_concepts) != 1:
            return QueryPlan.GRAPH
        if cq.clinical_concepts[0].concept_type not in _SQL_FAST_CONCEPT_TYPES:
            return QueryPlan.GRAPH

        # Aggregation must be SQL-compilable. None aggregation + diagnosis
        # concept is a plain patient-list query — also fast-path compilable.
        if cq.aggregation is None:
            if cq.clinical_concepts[0].concept_type == "diagnosis":
                # Bare "list patients with X" — handled by the fast-path's
                # diagnosis-list branch.
                return QueryPlan.SQL_FAST
            # Raw-values query: could be SQL too, but the answerer/UI
            # pattern for raw values expects the graph's timestamp handling.
            # Keep on graph-path for now.
            return QueryPlan.GRAPH

        agg_op = self._registry.get("aggregate", cq.aggregation)
        if agg_op is None or getattr(agg_op, "sql_fn", None) is None:
            return QueryPlan.GRAPH

        # Comparison scope: axis must be a registered comparison_axis with
        # a SQL group-by descriptor.
        if cq.scope == "comparison":
            if not cq.comparison_field:
                return QueryPlan.GRAPH
            axis_op = self._registry.get("comparison_axis", cq.comparison_field)
            if axis_op is None or getattr(axis_op, "sql_group_by", None) is None:
                return QueryPlan.GRAPH

        return QueryPlan.SQL_FAST
