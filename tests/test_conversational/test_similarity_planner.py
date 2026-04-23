"""Planner dispatch for similarity CQs (Phase 9).

``scope == "patient_similarity"`` → ``QueryPlan.SIMILARITY``.
``similarity_spec`` + ``scope == "causal_effect"`` → stays on
``QueryPlan.CAUSAL`` (cohort narrowing, not standalone).
"""

from __future__ import annotations

from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)
from src.conversational.planner import QueryPlan, QueryPlanner
from src.similarity.models import SimilaritySpec


class TestSimilarityScopeRouting:
    def test_patient_similarity_scope_routes_to_similarity_plan(self):
        cq = CompetencyQuestion(
            original_question="Show me patients similar to hadm 101",
            scope="patient_similarity",
            similarity_spec=SimilaritySpec(anchor_hadm_id=101, top_k=20),
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.SIMILARITY

    def test_patient_similarity_with_template_anchor_routes_to_similarity(self):
        cq = CompetencyQuestion(
            original_question="Find patients like a 68yo F with afib",
            scope="patient_similarity",
            similarity_spec=SimilaritySpec(
                anchor_template={"age": 68, "gender_F": 1, "snomed_group_I48": 1},
            ),
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.SIMILARITY


class TestCausalPlusSimilarityStaysCausal:
    def test_similarity_spec_with_causal_scope_stays_causal(self):
        cq = CompetencyQuestion(
            original_question="Compare tPA effect among patients similar to hadm 101",
            scope="causal_effect",
            similarity_spec=SimilaritySpec(anchor_hadm_id=101, top_k=30),
            intervention_set=[
                InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
                InterventionSpec(
                    label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
                ),
            ],
            outcome_vector=[
                OutcomeSpec(
                    name="readmitted_30d", outcome_type="binary",
                    extractor_key="readmitted_30d",
                ),
            ],
            aggregation_spec=AggregationSpec(kind="identity"),
        )
        plan = QueryPlanner().classify(cq)
        assert plan == QueryPlan.CAUSAL
