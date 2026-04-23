"""``SimilaritySpec`` pydantic validation (Phase 9).

Anchor must be exactly one of {hadm_id, subject_id, template}. Weights
+ top_k + thresholds must make sense. See
``/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md``.
"""

from __future__ import annotations

import pytest

from src.conversational.models import PatientFilter
from src.similarity.models import (
    ContextualExplanation,
    SimilarityResult,
    SimilarityScore,
    SimilaritySpec,
    TemporalExplanation,
)


class TestAnchorExactlyOne:
    def test_hadm_id_anchor_ok(self):
        spec = SimilaritySpec(anchor_hadm_id=101)
        assert spec.anchor_hadm_id == 101

    def test_subject_id_anchor_ok(self):
        spec = SimilaritySpec(anchor_subject_id=42)
        assert spec.anchor_subject_id == 42

    def test_template_anchor_ok(self):
        spec = SimilaritySpec(anchor_template={"age": 68, "gender_F": 1})
        assert spec.anchor_template["age"] == 68

    def test_no_anchor_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            SimilaritySpec()

    def test_multiple_anchors_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            SimilaritySpec(anchor_hadm_id=101, anchor_subject_id=42)

    def test_hadm_and_template_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            SimilaritySpec(anchor_hadm_id=101, anchor_template={"age": 68})


class TestWeightBounds:
    def test_temporal_weight_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, temporal_weight=-0.1)
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, temporal_weight=1.1)

    def test_temporal_decay_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, temporal_decay=0.0)
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, temporal_decay=1.5)

    def test_contextual_weights_default_none(self):
        spec = SimilaritySpec(anchor_hadm_id=101)
        assert spec.contextual_weights is None

    def test_contextual_weights_accepts_partial_override(self):
        spec = SimilaritySpec(
            anchor_hadm_id=101,
            contextual_weights={"demographics": 0.5, "comorbidity_burden": 0.5},
        )
        assert spec.contextual_weights["demographics"] == 0.5


class TestSelectionParams:
    def test_top_k_positive(self):
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, top_k=0)
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, top_k=-5)

    def test_top_k_none_allowed(self):
        spec = SimilaritySpec(anchor_hadm_id=101, top_k=None)
        assert spec.top_k is None

    def test_min_similarity_in_unit_interval(self):
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, min_similarity=-0.1)
        with pytest.raises(ValueError):
            SimilaritySpec(anchor_hadm_id=101, min_similarity=1.5)


class TestCandidateFilters:
    def test_accepts_patient_filter_list(self):
        spec = SimilaritySpec(
            anchor_hadm_id=101,
            candidate_filters=[
                PatientFilter(field="age", operator=">=", value="60"),
                PatientFilter(field="diagnosis", operator="contains", value="I48"),
            ],
        )
        assert len(spec.candidate_filters) == 2
        assert spec.candidate_filters[0].field == "age"


class TestExplanationSchemas:
    def test_contextual_explanation_shape(self):
        exp = ContextualExplanation(
            overall_score=0.82,
            per_group={"demographics": 0.9, "comorbidity_burden": 0.75},
            top_contributors=[("shared_chf", 0.15)],
            top_detractors=[("age_diff_18y", -0.12)],
        )
        assert exp.overall_score == 0.82
        assert "demographics" in exp.per_group

    def test_temporal_explanation_shape(self):
        exp = TemporalExplanation(
            overall_score=0.71,
            per_bucket=[("icu_day_0", 0.9), ("icu_day_1", 0.5)],
            shared_events=[("icu_day_0", "snomed_drug:vasopressor")],
            anchor_only=[],
            candidate_only=[("icu_day_1", "snomed_drug:diuretic")],
            los_gap_days=2,
            temporal_available=True,
        )
        assert exp.temporal_available is True
        assert exp.los_gap_days == 2


class TestResultContainer:
    def test_similarity_result_sorted_by_combined(self):
        # Caller can build any ordering; result spec doesn't enforce
        # sort — but the ``run_similarity`` path does.
        spec = SimilaritySpec(anchor_hadm_id=101)
        result = SimilarityResult(
            anchor_description="hadm 101",
            n_pool=10,
            n_returned=2,
            scores=[
                SimilarityScore(
                    hadm_id=201, subject_id=2001, combined=0.9,
                    contextual=0.85, temporal=0.95,
                    contextual_explanation=ContextualExplanation(
                        overall_score=0.85, per_group={}, top_contributors=[], top_detractors=[]),
                    temporal_explanation=TemporalExplanation(
                        overall_score=0.95, per_bucket=[], shared_events=[],
                        anchor_only=[], candidate_only=[], los_gap_days=0, temporal_available=True),
                ),
                SimilarityScore(
                    hadm_id=202, subject_id=2002, combined=0.6,
                    contextual=0.7, temporal=0.5,
                    contextual_explanation=ContextualExplanation(
                        overall_score=0.7, per_group={}, top_contributors=[], top_detractors=[]),
                    temporal_explanation=TemporalExplanation(
                        overall_score=0.5, per_bucket=[], shared_events=[],
                        anchor_only=[], candidate_only=[], los_gap_days=1, temporal_available=True),
                ),
            ],
            spec=spec,
            provenance={"source": "test"},
        )
        assert result.n_returned == 2
        assert result.scores[0].combined > result.scores[1].combined
