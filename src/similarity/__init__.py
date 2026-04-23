"""Patient-similarity engine (temporal + contextual) — Phase 9.

Delivers an interpretable, deterministic two-axis similarity score +
rich explanations for both standalone chat queries ("show me patients
similar to X") and causal cohort narrowing ("effect of T on Y among
patients similar to X").

Public surface:

    from src.similarity import (
        SimilaritySpec, SimilarityResult, SimilarityScore,
        ContextualExplanation, TemporalExplanation,
        run_similarity,
    )

Design decisions are locked in the plan file at
``/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md``.
"""

from src.similarity.models import (
    ContextualExplanation,
    SimilarityResult,
    SimilarityScore,
    SimilaritySpec,
    TemporalExplanation,
)
from src.similarity.run import run_similarity

__all__ = [
    "ContextualExplanation",
    "SimilarityResult",
    "SimilarityScore",
    "SimilaritySpec",
    "TemporalExplanation",
    "run_similarity",
]
