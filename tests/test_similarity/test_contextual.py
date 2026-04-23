"""Contextual similarity scoring (Phase 9).

Verifies the five-group Gower-like distance on hand-built feature
DataFrames. Ranks candidates whose features are programmed to produce
a known ordering relative to the anchor.
"""

from __future__ import annotations

from src.similarity.contextual import compute_contextual_similarity


class TestContextualRanking:
    def test_near_identical_candidate_scores_highest(
        self, anchor_features, candidate_features_df,
    ):
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        # 2001 is programmed to be very similar; should be top-scored.
        best_hadm = max(out, key=lambda h: out[h].overall_score)
        assert best_hadm == 2001

    def test_programmed_ranking_is_monotonic(
        self, anchor_features, candidate_features_df,
    ):
        """Fixture is programmed so rank is 2001 > 2002 > 2003 > 2004."""
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        ordered = sorted(out.items(), key=lambda kv: -kv[1].overall_score)
        hadm_ids = [h for h, _ in ordered]
        assert hadm_ids == [2001, 2002, 2003, 2004]


class TestContextualGroupScores:
    def test_all_groups_present_in_explanation(
        self, anchor_features, candidate_features_df,
    ):
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        exp = out[2001]
        expected_groups = {
            "demographics", "comorbidity_burden", "comorbidity_set",
            "severity", "social",
        }
        assert set(exp.per_group.keys()) == expected_groups

    def test_group_scores_in_unit_interval(
        self, anchor_features, candidate_features_df,
    ):
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        for exp in out.values():
            for group, score in exp.per_group.items():
                assert 0.0 <= score <= 1.0, f"group {group} score out of [0,1]: {score}"

    def test_overall_score_in_unit_interval(
        self, anchor_features, candidate_features_df,
    ):
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        for exp in out.values():
            assert 0.0 <= exp.overall_score <= 1.0


class TestContextualWeightOverride:
    def test_custom_weights_change_ranking(
        self, anchor_features, candidate_features_df,
    ):
        """Severity-dominant weights should push 2003 (same demographics
        + comorbidities, very different labs) below the default ranking."""
        default = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        severity_dominant = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
            weights={
                "demographics": 0.05,
                "comorbidity_burden": 0.05,
                "comorbidity_set": 0.05,
                "severity": 0.80,
                "social": 0.05,
            },
        )
        # 2003 has severely abnormal labs; under severity-dominant
        # weights its score should drop below the default.
        assert severity_dominant[2003].overall_score < default[2003].overall_score

    def test_weights_must_sum_to_one(
        self, anchor_features, candidate_features_df,
    ):
        import pytest

        with pytest.raises(ValueError, match="sum to 1"):
            compute_contextual_similarity(
                anchor_features=anchor_features,
                candidate_features_df=candidate_features_df,
                weights={"demographics": 0.5, "severity": 0.2},  # sums to 0.7
            )


class TestContextualContributorsAndDetractors:
    def test_shared_comorbidity_appears_in_contributors(
        self, anchor_features, candidate_features_df,
    ):
        """For candidate 2001 (shared afib + CKD), the shared comorbidity
        flags should surface as top contributors."""
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        contributors = {name for name, _ in out[2001].top_contributors}
        # At least one shared comorbidity feature pulls the score up.
        assert any(
            c.startswith("charlson_chf") or c.startswith("charlson_renal")
            or c.startswith("snomed_group_I48") or c.startswith("snomed_group_N18")
            for c in contributors
        )

    def test_detractor_contributions_are_negative(
        self, anchor_features, candidate_features_df,
    ):
        """Detractors pull score DOWN — contribution magnitude is negative."""
        out = compute_contextual_similarity(
            anchor_features=anchor_features,
            candidate_features_df=candidate_features_df,
        )
        # 2004 has big age / gender / comorbidity gaps — should have
        # visible detractors.
        for _, contribution in out[2004].top_detractors:
            assert contribution < 0
