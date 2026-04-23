"""Explanation formatter output (Phase 9).

The explanation module converts ``ContextualExplanation`` /
``TemporalExplanation`` payloads into Streamlit-ready structures and
plain-text chat snippets. Tests assert shape + content, not formatting.
"""

from __future__ import annotations

from src.similarity.explanations import (
    format_contextual_text,
    format_similarity_text,
    format_temporal_text,
)
from src.similarity.models import (
    ContextualExplanation,
    SimilarityScore,
    TemporalExplanation,
)


def _score(hadm_id: int = 2001) -> SimilarityScore:
    ctx = ContextualExplanation(
        overall_score=0.82,
        per_group={
            "demographics": 0.95, "comorbidity_burden": 0.78,
            "comorbidity_set": 0.90, "severity": 0.70, "social": 1.0,
        },
        top_contributors=[("charlson_chf", 0.12), ("snomed_group_I48", 0.10)],
        top_detractors=[("creatinine_max", -0.06)],
    )
    temp = TemporalExplanation(
        overall_score=0.71,
        per_bucket=[("icu_day_0", 1.0), ("icu_day_1", 0.5), ("icu_day_2", 0.33)],
        shared_events=[("icu_day_0", "snomed_drug:abx_broad"),
                       ("icu_day_0", "snomed_drug:vasopressor")],
        anchor_only=[("icu_day_2", "snomed_drug:steroid")],
        candidate_only=[("icu_day_1", "snomed_drug:diuretic")],
        los_gap_days=1,
        temporal_available=True,
    )
    return SimilarityScore(
        hadm_id=hadm_id, subject_id=200 + hadm_id,
        combined=0.77, contextual=0.82, temporal=0.71,
        contextual_explanation=ctx, temporal_explanation=temp,
    )


class TestContextualTextFormatter:
    def test_mentions_matching_group(self):
        text = format_contextual_text(_score().contextual_explanation)
        assert "demographics" in text.lower()

    def test_mentions_top_contributor(self):
        text = format_contextual_text(_score().contextual_explanation)
        # CHF is the largest contributor in the fixture — should appear.
        assert "chf" in text.lower()


class TestTemporalTextFormatter:
    def test_mentions_shared_event(self):
        text = format_temporal_text(_score().temporal_explanation)
        assert "abx" in text.lower() or "antibiotic" in text.lower() \
            or "vasopressor" in text.lower()

    def test_flags_los_gap(self):
        text = format_temporal_text(_score().temporal_explanation)
        assert "1" in text  # los_gap_days=1

    def test_unavailable_temporal_has_explanatory_text(self):
        from src.similarity.models import TemporalExplanation

        unavailable = TemporalExplanation(
            overall_score=0.0,
            per_bucket=[], shared_events=[], anchor_only=[], candidate_only=[],
            los_gap_days=0, temporal_available=False,
        )
        text = format_temporal_text(unavailable)
        # Text must make clear temporal was skipped — e.g., "template"
        # or "unavailable" or "contextual-only".
        assert any(k in text.lower() for k in ("template", "unavailable", "contextual"))


class TestSimilarityScoreTextFormatter:
    def test_combines_contextual_and_temporal(self):
        text = format_similarity_text(_score())
        # Should mention both axes.
        assert any(k in text.lower() for k in ("contextual", "demograph"))
        assert any(k in text.lower() for k in ("temporal", "trajectory", "day"))

    def test_contains_overall_score_reference(self):
        text = format_similarity_text(_score())
        # Overall combined of 0.77 → some representation.
        assert "0.77" in text or "77" in text
