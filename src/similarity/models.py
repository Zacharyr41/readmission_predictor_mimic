"""Pydantic schemas for the patient-similarity engine (Phase 9).

This module owns the input + output contract. Input side:
``SimilaritySpec`` — carried on ``CompetencyQuestion.similarity_spec``
for both standalone similarity CQs and as a cohort-narrowing spec on
causal CQs. Output side: ``SimilarityResult`` with ranked
``SimilarityScore`` rows, each carrying both contextual and temporal
explanation payloads.

The four locked decisions are reflected directly in the schema:

  * Decision #1 (anchor = real patient OR template) — ``SimilaritySpec``
    accepts ``anchor_hadm_id`` XOR ``anchor_subject_id`` XOR
    ``anchor_template``. The validator enforces exactly one.
  * Decision #2 (hybrid temporal bucketing) — bucket labels on
    ``TemporalExplanation.per_bucket`` follow the two conventions
    (``icu_day_N`` / ``pre_icu_*`` / ``h_0_24`` / ``h_24_48`` /
    ``day_N``) without the schema enforcing which was used.
  * Decision #3 (both cohort-narrowing AND standalone CQ) — the
    spec is shape-identical in both cases; the caller (planner /
    run_causal) decides dispatch.
  * Decision #4 (full chat/LLM/UI integration) — explanations carry
    enough detail (per-group scores, top contributors / detractors,
    shared-events breakdown) to render the Streamlit panel + the
    plain-English chat summary.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# NOTE: ``PatientFilter`` (used by ``SimilaritySpec.candidate_filters``) is
# imported at the BOTTOM of this module, not here. See the forward-reference
# resolution block at the end for why deferring it makes this module's import
# order-independent w.r.t. ``src.conversational.models``.


# ---------------------------------------------------------------------------
# Explanation payloads.
# ---------------------------------------------------------------------------


class ContextualExplanation(BaseModel):
    """Per-candidate contextual-similarity breakdown.

    ``per_group`` carries the 5-group decomposition — the dict keys
    are ``demographics``, ``comorbidity_burden``, ``comorbidity_set``,
    ``severity``, ``social``. Each value is a 0–1 group-similarity
    score. ``top_contributors`` surfaces features that raised the
    overall score (positive signed contribution); ``top_detractors``
    surfaces features that lowered it (negative signed contribution).
    """

    model_config = {"extra": "forbid"}

    overall_score: float
    per_group: dict[str, float] = Field(default_factory=dict)
    top_contributors: list[tuple[str, float]] = Field(default_factory=list)
    top_detractors: list[tuple[str, float]] = Field(default_factory=list)


class TemporalExplanation(BaseModel):
    """Per-candidate temporal-similarity breakdown.

    ``per_bucket`` is a time-ordered list of ``(bucket_label, jaccard)``.
    Bucket labels follow the hybrid-bucketing scheme from decision #2:
    ``icu_day_N`` / ``pre_icu_*`` for ICU admissions, or ``h_0_24`` /
    ``h_24_48`` / ``day_N`` for floor-only admissions.

    ``shared_events`` / ``anchor_only`` / ``candidate_only`` partition
    the event sets across both patients, keyed by ``(bucket, event_code)``
    tuples so the UI can group per bucket.

    ``los_gap_days`` is the difference in trajectory length between
    anchor and candidate; large gaps flag mismatched stays (the UI
    surfaces a warning pill).

    ``temporal_available=False`` is set when the anchor is a covariate
    template (no trajectory to compare against) — ``combine_scores``
    then falls back to contextual-only regardless of ``temporal_weight``.
    """

    model_config = {"extra": "forbid"}

    overall_score: float
    per_bucket: list[tuple[str, float]] = Field(default_factory=list)
    shared_events: list[tuple[str, str]] = Field(default_factory=list)
    anchor_only: list[tuple[str, str]] = Field(default_factory=list)
    candidate_only: list[tuple[str, str]] = Field(default_factory=list)
    los_gap_days: int = 0
    temporal_available: bool = True


# ---------------------------------------------------------------------------
# Scores + result container.
# ---------------------------------------------------------------------------


class SimilarityScore(BaseModel):
    """One row of the ranked similarity result.

    Carries both numeric scores and the full explanation payloads so
    the UI + LLM can surface WHY this candidate was ranked where it
    was. ``temporal`` is ``None`` when the anchor is a template (no
    trajectory); the combine step then uses ``contextual`` only.
    """

    model_config = {"extra": "forbid"}

    hadm_id: int
    subject_id: int
    combined: float
    contextual: float
    temporal: float | None = None
    contextual_explanation: ContextualExplanation
    temporal_explanation: TemporalExplanation | None = None


class SimilarityResult(BaseModel):
    """Output of ``run_similarity``.

    ``scores`` is ranked descending by ``combined`` — the caller can
    rely on ``scores[0]`` being the most similar candidate.
    ``anchor_description`` is human-readable (referenced by the chat
    response). ``provenance`` captures the bucketing mode, feature-
    matrix version, weights used, filter narrowing — anything a
    downstream reader needs to reproduce the ranking.
    """

    model_config = {"extra": "forbid"}

    anchor_description: str
    n_pool: int
    n_returned: int
    scores: list[SimilarityScore] = Field(default_factory=list)
    spec: "SimilaritySpec"
    provenance: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input spec.
# ---------------------------------------------------------------------------


class SimilaritySpec(BaseModel):
    """Input specification for a similarity query.

    Carried on ``CompetencyQuestion.similarity_spec`` (additive; the
    conversational schema extension lands in commit 6). One anchor —
    exactly one of ``anchor_hadm_id`` / ``anchor_subject_id`` /
    ``anchor_template`` — must be set.

    Weight defaults follow the plan's locked decisions. Group
    overrides for ``contextual_weights`` are partial by convention
    (callers typically tweak one or two groups); the compute layer
    in ``contextual.py`` validates that the effective weight vector
    sums to 1.
    """

    model_config = {"extra": "forbid"}

    # Anchor — exactly one of these three must be set (enforced in
    # the model validator below).
    anchor_hadm_id: int | None = None
    anchor_subject_id: int | None = None
    anchor_template: dict | None = None

    # Weights — defaults per plan.
    temporal_weight: float = 0.5                      # α
    contextual_weights: dict[str, float] | None = None
    temporal_decay: float = 0.9

    # Selection.
    top_k: int | None = 30
    min_similarity: float | None = None

    # Candidate pool — pre-narrow before scoring. Reuses the existing
    # PatientFilter schema at src/conversational/models.py:24-28.
    candidate_filters: list[PatientFilter] = Field(default_factory=list)

    # Reserved for future UX detail control.
    explanation_depth: Literal["summary", "full"] = "summary"

    @model_validator(mode="after")
    def _validate_spec(self) -> "SimilaritySpec":
        # Exactly one anchor.
        provided = sum(
            1
            for a in (self.anchor_hadm_id, self.anchor_subject_id, self.anchor_template)
            if a is not None
        )
        if provided != 1:
            raise ValueError(
                "SimilaritySpec must carry exactly one anchor "
                "(anchor_hadm_id | anchor_subject_id | anchor_template); "
                f"got {provided}"
            )
        # Weight bounds.
        if not (0.0 <= self.temporal_weight <= 1.0):
            raise ValueError(
                f"temporal_weight must be in [0, 1]; got {self.temporal_weight}"
            )
        if not (0.0 < self.temporal_decay <= 1.0):
            raise ValueError(
                f"temporal_decay must be in (0, 1]; got {self.temporal_decay}"
            )
        # Selection bounds.
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive or None; got {self.top_k}")
        if self.min_similarity is not None and not (0.0 <= self.min_similarity <= 1.0):
            raise ValueError(
                f"min_similarity must be in [0, 1] or None; got {self.min_similarity}"
            )
        return self


# ---------------------------------------------------------------------------
# Forward-reference resolution.
#
# ``PatientFilter`` (referenced by ``SimilaritySpec.candidate_filters``) is
# imported HERE, after every class in this module is defined, rather than at
# module top. That keeps this module free of any ``src.conversational`` import
# during its class-definition phase, so the two modules may be imported in
# EITHER order without the partial-initialization cycle breaking
# (``src.conversational.models`` imports ``SimilaritySpec`` from here at its own
# bottom). ``from __future__ import annotations`` makes every annotation a
# string, so no class needs ``PatientFilter`` resolved until the rebuilds below.
#
# Previously this import lived at module top, which only worked when
# ``src.conversational.models`` happened to be imported first; importing
# ``src.similarity.*`` first raised ImportError (e.g. running
# ``pytest tests/test_similarity/`` in isolation).
# ---------------------------------------------------------------------------
from src.conversational.models import (  # noqa: E402
    CompetencyQuestion,
    PatientFilter,  # noqa: F401  (resolved from globals by model_rebuild)
)

SimilaritySpec.model_rebuild()
SimilarityResult.model_rebuild()
CompetencyQuestion.model_rebuild(
    _types_namespace={"SimilaritySpec": SimilaritySpec},
    force=True,
)
