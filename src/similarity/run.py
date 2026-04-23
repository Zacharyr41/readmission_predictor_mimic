"""Top-level entry point for the patient-similarity engine (Phase 9).

``run_similarity(spec, backend)`` is the orchestrator-facing function:

1. Resolve the anchor — real patient (fetch features + event stream)
   or template (use covariate dict, skip temporal).
2. Pull the candidate pool — starts from every admission, narrows by
   ``spec.candidate_filters`` if any.
3. Build the contextual feature matrix for the pool via
   ``src.feature_extraction.feature_builder.build_feature_matrix``.
4. Build the per-candidate bucketed event sets via
   ``src.similarity.bucketing.assign_buckets``.
5. Score contextual + temporal + combined per candidate.
6. Rank, apply ``min_similarity`` / ``top_k``, package into a
   ``SimilarityResult`` with provenance.

Commit 5 of the Phase 9 TDD trail fills this in; commit 2 stubs the
surface so the planner / orchestrator / causal narrowing paths can
import it and the end-to-end test can target the real signature.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from typing import Any

from src.similarity.models import SimilarityResult, SimilaritySpec


def run_similarity(
    spec: SimilaritySpec,
    backend: Any,
) -> SimilarityResult:
    """Compute a ranked similarity result.

    Raises:
        ValueError: if the anchor resolves to no known admission /
            subject (real-patient anchor paths).
    """
    raise NotImplementedError(
        "run_similarity — ships in commit 5 of the Phase 9 TDD trail. "
        "See /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md"
    )


__all__ = ["run_similarity"]
