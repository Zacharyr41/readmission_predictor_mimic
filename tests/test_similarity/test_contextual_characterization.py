"""Characterization (golden) tests for contextual similarity.

These freeze the EXACT current outputs of ``compute_contextual_similarity``
on the ``conftest`` fixtures so the pygower refactor (plan task II-B) is
provably behavior-preserving down to per-feature contribution ordering and
float values. The product-critical ranking is already pinned by
``test_contextual.py``; this file additionally pins the explanation payload
(overall score, per-group scores, and the top-5 contributor/detractor lists
including their tie-broken order).

If a future change intentionally alters the contextual math, regenerate the
goldens below rather than loosening the assertions.
"""

from __future__ import annotations

import pytest

from src.similarity.contextual import compute_contextual_similarity

# Per-group scores are weight-independent (only ``overall`` blends them), so
# they are shared across the default and severity-dominant scenarios.
PER_GROUP = {
    2001: {"demographics": 0.9866666666666667, "comorbidity_burden": 1.0,
           "comorbidity_set": 1.0, "severity": 0.9822222222222223, "social": 1.0},
    2002: {"demographics": 0.9533333333333333, "comorbidity_burden": 0.5666666666666667,
           "comorbidity_set": 0.5, "severity": 0.9133333333333333, "social": 1.0},
    2003: {"demographics": 1.0, "comorbidity_burden": 1.0,
           "comorbidity_set": 1.0, "severity": 0.25, "social": 1.0},
    2004: {"demographics": 0.4266666666666667, "comorbidity_burden": 0.25,
           "comorbidity_set": 0.0, "severity": 0.8355555555555554, "social": 1.0},
}

GOLDENS_DEFAULT = {
    2001: {
        "overall": 0.9953333333333333,
        "contributors": [("charlson_index", 0.175), ("snomed_group_I48", 0.05),
                         ("snomed_group_N18", 0.05), ("language_barrier", 0.05),
                         ("is_neuro_service", 0.05)],
        "detractors": [],
    },
    2002: {
        "overall": 0.7033333333333333,
        "contributors": [("charlson_index", 0.10500000000000001),
                         ("snomed_group_I48", 0.05), ("language_barrier", 0.05),
                         ("is_neuro_service", 0.05), ("gender", 0.049999999999999996)],
        "detractors": [("snomed_group_N18", -0.05),
                       ("charlson_renal", -0.010294117647058823)],
    },
    2003: {
        "overall": 0.8875,
        "contributors": [("charlson_index", 0.175), ("snomed_group_I48", 0.05),
                         ("snomed_group_N18", 0.05), ("language_barrier", 0.05),
                         ("is_neuro_service", 0.05)],
        "detractors": [("icu_los_hours", -0.03), ("platelet_min", -0.03),
                       ("creatinine_max", -0.03), ("sodium_mean", -0.015)],
    },
    2004: {
        "overall": 0.37683333333333324,
        "contributors": [("language_barrier", 0.05), ("is_neuro_service", 0.05),
                         ("admission_type", 0.049999999999999996),
                         ("sodium_mean", 0.03), ("platelet_min", 0.03)],
        "detractors": [("snomed_group_N18", -0.05), ("snomed_group_I48", -0.05),
                       ("gender", -0.049999999999999996),
                       ("age", -0.021999999999999995),
                       ("charlson_renal", -0.010294117647058823)],
    },
}

SEVERITY_WEIGHTS = {
    "demographics": 0.05, "comorbidity_burden": 0.05, "comorbidity_set": 0.05,
    "severity": 0.80, "social": 0.05,
}

GOLDENS_SEVERITY = {
    2001: {
        "overall": 0.9851111111111113,
        "contributors": [("sodium_mean", 0.16000000000000003),
                         ("platelet_min", 0.16000000000000003),
                         ("admission_type_EMERGENCY", 0.16000000000000003),
                         ("creatinine_max", 0.14933333333333335),
                         ("icu_los_hours", 0.14222222222222222)],
        "detractors": [],
    },
    2002: {
        "overall": 0.8816666666666667,
        "contributors": [("sodium_mean", 0.16000000000000003),
                         ("platelet_min", 0.16000000000000003),
                         ("admission_type_EMERGENCY", 0.16000000000000003),
                         ("icu_los_hours", 0.10666666666666669),
                         ("creatinine_max", 0.07466666666666669)],
        "detractors": [("snomed_group_N18", -0.010000000000000002),
                       ("charlson_renal", -0.0014705882352941176)],
    },
    2003: {
        "overall": 0.4,
        "contributors": [("admission_type_EMERGENCY", 0.16000000000000003),
                         ("charlson_index", 0.025), ("language_barrier", 0.025),
                         ("is_neuro_service", 0.025),
                         ("age", 0.016666666666666666)],
        "detractors": [("icu_los_hours", -0.16000000000000003),
                       ("platelet_min", -0.16000000000000003),
                       ("creatinine_max", -0.16000000000000003),
                       ("sodium_mean", -0.08000000000000002)],
    },
    2004: {
        "overall": 0.7522777777777778,
        "contributors": [("sodium_mean", 0.16000000000000003),
                         ("platelet_min", 0.16000000000000003),
                         ("admission_type_EMERGENCY", 0.16000000000000003),
                         ("icu_los_hours", 0.035555555555555576),
                         ("language_barrier", 0.025)],
        "detractors": [("gender", -0.016666666666666666),
                       ("snomed_group_N18", -0.010000000000000002),
                       ("snomed_group_I48", -0.010000000000000002),
                       ("age", -0.007333333333333332),
                       ("charlson_renal", -0.0014705882352941176)],
    },
}


def _assert_pairs(actual, golden):
    """Names must match exactly (this pins tie-broken order); values approx."""
    assert [n for n, _ in actual] == [n for n, _ in golden]
    for (an, av), (gn, gv) in zip(actual, golden):
        assert av == pytest.approx(gv, abs=1e-12), f"{an}: {av} != {gv}"


@pytest.mark.parametrize("weights,goldens", [
    (None, GOLDENS_DEFAULT),
    (SEVERITY_WEIGHTS, GOLDENS_SEVERITY),
])
def test_contextual_payload_is_byte_stable(
    anchor_features, candidate_features_df, weights, goldens,
):
    out = compute_contextual_similarity(
        anchor_features, candidate_features_df, weights=weights,
    )
    assert set(out) == set(goldens)
    for hadm, g in goldens.items():
        exp = out[hadm]
        assert exp.overall_score == pytest.approx(g["overall"], abs=1e-12)
        for group, score in PER_GROUP[hadm].items():
            assert exp.per_group[group] == pytest.approx(score, abs=1e-12)
        _assert_pairs(exp.top_contributors, g["contributors"])
        _assert_pairs(exp.top_detractors, g["detractors"])
