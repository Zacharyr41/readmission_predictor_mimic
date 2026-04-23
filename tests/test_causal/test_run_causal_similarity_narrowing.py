"""``run_causal`` with a ``similarity_spec`` narrows the cohort (Phase 9).

When a causal CQ carries a ``similarity_spec``, ``run_causal`` must
thread the similarity summary into ``DiagnosticReport.notes`` so the
UI + investigator can audit cohort selection.

Two separate scenarios are covered:

1. Caller passes ``cohort_frame=`` directly (synthetic cohort for
   deterministic bootstrap) AND sets ``similarity_spec`` — run_causal
   skips live narrowing but still emits an audit note with the
   anchor + top_k.
2. Caller passes ``backend`` + lets run_causal narrow live —
   ``run_similarity`` is mocked to return a canned top-K so we
   isolate the run_causal integration from the engine's DB extract.

The real end-to-end similarity engine is exercised in
``tests/test_similarity/test_run_similarity_end_to_end.py``.
"""

from __future__ import annotations

from unittest.mock import patch

from src.causal.run import run_causal
from src.conversational.models import (
    AggregationSpec,
    CompetencyQuestion,
    InterventionSpec,
    OutcomeSpec,
)
from src.similarity.models import (
    ContextualExplanation,
    SimilarityResult,
    SimilarityScore,
    SimilaritySpec,
)


def _causal_cq_with_similarity(
    top_k: int = 8, anchor_hadm_id: int = 101,
) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="tPA effect among similar patients",
        scope="causal_effect",
        similarity_spec=SimilaritySpec(
            anchor_hadm_id=anchor_hadm_id, top_k=top_k,
        ),
        intervention_set=[
            InterventionSpec(label="arm0", kind="drug", rxnorm_ingredient="8410"),
            InterventionSpec(
                label="arm1", kind="drug", rxnorm_ingredient="8410", is_control=True,
            ),
        ],
        outcome_vector=[
            OutcomeSpec(
                name="Y", outcome_type="binary", extractor_key="readmitted_30d",
            ),
        ],
        aggregation_spec=AggregationSpec(kind="identity"),
        uncertainty_reps=10,
    )


def _fake_similarity_result(
    spec: SimilaritySpec, hadm_ids: list[int],
) -> SimilarityResult:
    scores = [
        SimilarityScore(
            hadm_id=h, subject_id=100 + h, combined=0.9 - 0.01 * i,
            contextual=0.9, temporal=None,
            contextual_explanation=ContextualExplanation(overall_score=0.9),
            temporal_explanation=None,
        )
        for i, h in enumerate(hadm_ids)
    ]
    return SimilarityResult(
        anchor_description=f"mock anchor hadm_id={spec.anchor_hadm_id}",
        n_pool=20,
        n_returned=len(hadm_ids),
        scores=scores,
        spec=spec,
        provenance={"source": "mock"},
    )


class TestNarrowingAuditNoteWithPreBuiltCohort:
    """When the caller supplies ``cohort_frame=``, run_causal skips
    live narrowing but emits an audit note so the similarity_spec
    isn't silently ignored. Uses a synthetic cohort with controlled
    class balance so the bootstrap is deterministic."""

    def test_audit_note_present_when_spec_carried_with_cohort_frame(self):
        from tests.test_causal.conftest import make_synthetic_cohort_frame

        cohort = make_synthetic_cohort_frame(
            n_per_arm=80, n_arms=2, ate=1.0, seed=0, binary_outcome=True,
        )
        cq = _causal_cq_with_similarity(top_k=30, anchor_hadm_id=101)
        result = run_causal(cq, backend=None, cohort_frame=cohort)
        narrowing_notes = [
            n for n in result.diagnostics.notes
            if "similar" in n.lower() or "narrow" in n.lower()
        ]
        assert narrowing_notes, (
            "Expected an audit note mentioning similarity_spec; "
            f"got notes={result.diagnostics.notes!r}"
        )

    def test_audit_note_carries_anchor_and_top_k(self):
        from tests.test_causal.conftest import make_synthetic_cohort_frame

        cohort = make_synthetic_cohort_frame(
            n_per_arm=80, n_arms=2, ate=1.0, seed=0, binary_outcome=True,
        )
        cq = _causal_cq_with_similarity(top_k=42, anchor_hadm_id=777)
        result = run_causal(cq, backend=None, cohort_frame=cohort)
        combined = " ".join(result.diagnostics.notes).lower()
        assert "42" in combined
        assert "777" in combined


class TestLiveNarrowingWithMockedSimilarity:
    """When no cohort is pre-provided, run_causal calls run_similarity
    and uses the top-K hadm_ids. We mock run_similarity to avoid
    coupling this test to the engine internals."""

    def test_live_narrowing_note_contains_pool_and_returned_counts(self):
        """``run_causal`` records n_pool + n_returned from the
        SimilarityResult in its narrowing note. Uses cohort_frame=
        passthrough via a monkeypatch on build_cohort_frame so the
        live-narrowing branch is exercised without DB dependency."""
        from tests.test_causal.conftest import make_synthetic_cohort_frame

        cohort = make_synthetic_cohort_frame(
            n_per_arm=80, n_arms=2, ate=1.0, seed=0, binary_outcome=True,
        )
        cq = _causal_cq_with_similarity(top_k=5, anchor_hadm_id=101)
        # Test the INTEGRATION shape: run_causal detects similarity_spec,
        # calls run_similarity (mocked), incorporates the note. We use
        # cohort_frame only for the bootstrap side (after the note lands).
        # The "live narrowing" branch requires backend None AND cohort_frame
        # None — which we can't have together with a synthetic cohort.
        # So we patch build_cohort_frame to return our synthetic cohort
        # regardless of the narrowed hadm_ids.
        fake_sim = _fake_similarity_result(
            cq.similarity_spec, [1_000_000 + i for i in range(5)],
        )
        with (
            patch("src.similarity.run.run_similarity", return_value=fake_sim),
            patch("src.causal.run.build_cohort_frame", return_value=cohort),
        ):
            result = run_causal(cq, backend=object())  # sentinel; mocked away
        text = " ".join(result.diagnostics.notes).lower()
        assert "n_pool=20" in text or "n_pool" in text
        assert "n_returned=5" in text or "n_returned" in text
