"""Feature-group definitions for contextual similarity (Phase 9).

The plan defines five feature groups used by the Gower-like distance
in ``src.similarity.contextual``. This module holds the group member-
ship map and the default per-group weights so both ``contextual`` and
``explanations`` can read from a single source.

Commit 2: shape + constants only. Commit 3 of the Phase 9 TDD trail
fills in the per-feature distance functions.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

# Default group weights. Sum must equal 1.0.
DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "demographics": 0.15,
    "comorbidity_burden": 0.35,
    "comorbidity_set": 0.25,
    "severity": 0.15,
    "social": 0.10,
}

# Feature membership per group. Keys must match columns in the feature
# matrix produced by ``src.feature_extraction.feature_builder.build_feature_matrix``.
# Prefixes (``charlson_``, ``snomed_group_``) are expanded at distance-
# compute time so new Charlson flags / SNOMED groups flow through
# without a code change.
GROUP_FEATURES: dict[str, dict] = {
    "demographics": {
        "numeric": ["age"],
        "categorical": [
            "gender_M", "gender_F", "gender_unknown",
            "admission_type_EMERGENCY", "admission_type_ELECTIVE",
            "admission_type_URGENT", "admission_type_other",
        ],
    },
    "comorbidity_burden": {
        "numeric": ["charlson_index"],
        "prefix_flags": ["charlson_"],  # expanded at runtime
    },
    "comorbidity_set": {
        "prefix_flags": ["snomed_group_"],  # expanded at runtime
    },
    "severity": {
        "numeric": [
            "creatinine_max", "sodium_mean", "platelet_min", "icu_los_hours",
        ],
        "categorical": ["admission_type_EMERGENCY"],
    },
    "social": {
        "categorical": ["language_barrier", "is_neuro_service"],
    },
}


__all__ = [
    "DEFAULT_GROUP_WEIGHTS",
    "GROUP_FEATURES",
]
