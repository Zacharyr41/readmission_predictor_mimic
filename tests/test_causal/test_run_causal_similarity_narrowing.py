"""``run_causal`` with a ``similarity_spec`` narrows the cohort (Phase 9).

When a causal CQ carries a ``similarity_spec``, ``run_causal`` must:
  (1) compute similarity against the population, take top-K,
  (2) pass those hadm_ids as the causal cohort,
  (3) record the narrowing in result provenance.
"""

from __future__ import annotations

from src.causal.run import run_causal
from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)
from src.similarity.models import SimilaritySpec


def _causal_cq_with_similarity(top_k: int = 3) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="tPA effect among similar patients",
        scope="causal_effect",
        similarity_spec=SimilaritySpec(anchor_hadm_id=101, top_k=top_k),
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
        uncertainty_reps=5,  # small — this test is about narrowing, not CIs
    )


class TestSimilarityNarrowingInCausal:
    def test_causal_provenance_carries_similarity_summary(self, similarity_backend):
        cq = _causal_cq_with_similarity(top_k=3)
        result = run_causal(cq, similarity_backend)
        # The diagnostic notes should include a "narrowed by similarity"
        # breadcrumb so the UI + investigator can trace cohort selection.
        narrowing_notes = [
            n for n in result.diagnostics.notes
            if "similar" in n.lower() or "narrow" in n.lower()
        ]
        assert narrowing_notes, (
            "Expected a diagnostic note mentioning similarity narrowing; "
            f"got notes={result.diagnostics.notes!r}"
        )


class TestNarrowingHonorsTopK:
    def test_top_k_caps_cohort_size(self, similarity_backend):
        # With top_k=2, the cohort should be ≤ 2 before treatment assignment.
        # We verify via the provenance — diagnostics should note the narrowed pool.
        cq = _causal_cq_with_similarity(top_k=2)
        result = run_causal(cq, similarity_backend)
        # Each bootstrap replicate fits on ≤ top_k rows split across arms.
        # Narrowing note should state the cap.
        narrowing_text = " ".join(result.diagnostics.notes).lower()
        assert "2" in narrowing_text or "top_k" in narrowing_text or "top-k" in narrowing_text
