"""End-to-end ``run_similarity(spec, backend)`` (Phase 9).

Wires contextual + temporal + combined together, backed by the
top-level ``synthetic_duckdb_with_events`` fixture. This is a
shape-and-ordering smoke, not a statistical claim — the 6-admission
fixture is too small for absolute-score precision.
"""

from __future__ import annotations

from src.similarity.models import SimilarityResult, SimilaritySpec
from src.similarity.run import run_similarity


class TestEndToEndWithBackend:
    def test_returns_similarity_result(self, similarity_backend):
        spec = SimilaritySpec(anchor_hadm_id=101, top_k=5)
        result = run_similarity(spec, similarity_backend)
        assert isinstance(result, SimilarityResult)

    def test_anchor_excluded_from_ranked_results(self, similarity_backend):
        spec = SimilaritySpec(anchor_hadm_id=101, top_k=5)
        result = run_similarity(spec, similarity_backend)
        # The anchor admission itself must not be among the "similar
        # candidates" — otherwise every query self-matches at 1.0.
        for score in result.scores:
            assert score.hadm_id != 101

    def test_top_k_respected(self, similarity_backend):
        spec = SimilaritySpec(anchor_hadm_id=101, top_k=3)
        result = run_similarity(spec, similarity_backend)
        assert len(result.scores) <= 3

    def test_scores_sorted_descending_by_combined(self, similarity_backend):
        spec = SimilaritySpec(anchor_hadm_id=101, top_k=5)
        result = run_similarity(spec, similarity_backend)
        combineds = [s.combined for s in result.scores]
        assert combineds == sorted(combineds, reverse=True)


class TestTemplateAnchor:
    def test_template_anchor_yields_contextual_only(self, similarity_backend):
        spec = SimilaritySpec(
            anchor_template={
                "age": 65, "gender_F": 1, "gender_M": 0, "gender_unknown": 0,
                "admission_type_EMERGENCY": 1,
                "charlson_chf": 1, "snomed_group_I48": 1,
            },
            top_k=5,
        )
        result = run_similarity(spec, similarity_backend)
        assert len(result.scores) > 0
        # Template anchor ⇒ no trajectory ⇒ temporal score is None.
        for score in result.scores:
            assert score.temporal is None
            assert score.temporal_explanation is None or \
                score.temporal_explanation.temporal_available is False


class TestResultProvenance:
    def test_provenance_carries_bucketing_mode_and_pool_size(
        self, similarity_backend,
    ):
        spec = SimilaritySpec(anchor_hadm_id=101, top_k=5)
        result = run_similarity(spec, similarity_backend)
        assert "n_pool" in result.provenance or result.n_pool > 0
        # Anchor description should reference the anchor hadm_id.
        assert "101" in result.anchor_description


class TestMinSimilarityThreshold:
    def test_min_similarity_filters_low_scorers(self, similarity_backend):
        spec_no_floor = SimilaritySpec(anchor_hadm_id=101, top_k=10)
        spec_with_floor = SimilaritySpec(
            anchor_hadm_id=101, top_k=10, min_similarity=0.99,
        )
        result_no_floor = run_similarity(spec_no_floor, similarity_backend)
        result_with_floor = run_similarity(spec_with_floor, similarity_backend)
        assert result_with_floor.n_returned <= result_no_floor.n_returned
        for score in result_with_floor.scores:
            assert score.combined >= 0.99
