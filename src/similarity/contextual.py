"""Contextual similarity — grouped Gower-like distance (Phase 9).

Five feature groups (defined in ``feature_groups.py``), each computing
a 0-1 group-similarity score plus per-feature signed contributions.
Overall contextual similarity is a weighted mean across groups; group
weights are user-overrideable via ``SimilaritySpec.contextual_weights``
and validated here to sum to 1.0 across the provided keys.

Distance conventions (from the plan):

  * demographics: weighted L1 — age ``|Δ|/50`` clipped to [0, 1];
    gender / admission_type 0/1 flag match
  * comorbidity_burden: weighted Jaccard on Charlson flags (weights
    from ``data/mappings/icd10_to_charlson.json``) + normalized ``|Δ|``
    on ``charlson_index``
  * comorbidity_set: Jaccard on SNOMED group set-valued flags
  * severity: per-feature ``|Δ|/scale`` clipped (fixed scales per
    variable for interpretability; no cohort-dependent z-score)
  * social: 0/1 flag match

Per-feature contributions (for ``top_contributors`` /
``top_detractors``) are computed as
``weight × (2 × feature_similarity − 1)`` so a perfect match (sim=1)
contributes +weight, a perfect mismatch (sim=0) contributes -weight,
and a neutral 0.5 contributes 0.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

import pandas as pd

from src.similarity.feature_groups import DEFAULT_GROUP_WEIGHTS, GROUP_FEATURES
from src.similarity.models import ContextualExplanation

# Charlson per-category weights (matches ``data/mappings/icd10_to_charlson.json``
# weighting). Used for weighted Jaccard in the comorbidity_burden group.
CHARLSON_WEIGHTS: dict[str, int] = {
    "charlson_myocardial_infarction": 1,
    "charlson_chf": 1,
    "charlson_pvd": 1,
    "charlson_cvd": 1,
    "charlson_dementia": 1,
    "charlson_copd": 1,
    "charlson_rheumatoid": 1,
    "charlson_pud": 1,
    "charlson_mild_liver": 1,
    "charlson_diabetes": 1,
    "charlson_diabetes_complications": 2,
    "charlson_hemiplegia": 2,
    "charlson_renal": 2,
    "charlson_malignancy": 2,
    "charlson_moderate_severe_liver": 3,
    "charlson_metastatic_tumor": 6,
    "charlson_aids_hiv": 6,
}

# Fixed clinical scales for continuous severity features. Using fixed
# rather than cohort-dependent scaling keeps the distance metric
# interpretable + reproducible across runs.
SEVERITY_SCALES: dict[str, float] = {
    "creatinine_max": 3.0,   # normal ~1, severe AKI 5+
    "sodium_mean": 20.0,      # clinical range ~120-160
    "platelet_min": 100.0,    # normal 150-400; severe < 50
    "icu_los_hours": 72.0,    # short stay ~24h, extended ~1 wk
}


def _pairwise_similarity(a: float, c: float, scale: float) -> float:
    """Normalised distance → similarity, clipped to [0, 1]."""
    if pd.isna(a) or pd.isna(c):
        return 0.0
    return max(0.0, 1.0 - min(abs(a - c) / scale, 1.0))


def _flag_match(a: int, c: int) -> float:
    """Flag-valued 0/1 similarity (1 if both match, 0 otherwise)."""
    if pd.isna(a) or pd.isna(c):
        return 0.0
    return 1.0 if int(a) == int(c) else 0.0


def _score_demographics(anchor: dict, candidate: dict) -> tuple[float, list[tuple[str, float]]]:
    """demographics group: age + gender + admission_type."""
    age_sim = _pairwise_similarity(
        anchor.get("age"), candidate.get("age"), scale=50.0,
    )

    gender_cols = GROUP_FEATURES["demographics"]["categorical"][:3]  # first 3 are gender
    gender_sim = 1.0 if any(
        int(anchor.get(c, 0)) == 1 and int(candidate.get(c, 0)) == 1
        for c in gender_cols
    ) else 0.0

    adm_cols = GROUP_FEATURES["demographics"]["categorical"][3:]
    adm_sim = 1.0 if any(
        int(anchor.get(c, 0)) == 1 and int(candidate.get(c, 0)) == 1
        for c in adm_cols
    ) else 0.0

    per_feature = [
        ("age", age_sim),
        ("gender", gender_sim),
        ("admission_type", adm_sim),
    ]
    group_score = sum(s for _, s in per_feature) / len(per_feature)
    contributions = [
        (name, (2 * s - 1) / len(per_feature)) for name, s in per_feature
    ]
    return group_score, contributions


def _score_comorbidity_burden(
    anchor: dict, candidate: dict, all_columns: list[str],
) -> tuple[float, list[tuple[str, float]]]:
    """comorbidity_burden group: Charlson index + weighted Jaccard on flags."""
    # Charlson index similarity.
    idx_sim = _pairwise_similarity(
        anchor.get("charlson_index"), candidate.get("charlson_index"), scale=10.0,
    )

    # Weighted Jaccard on Charlson flags present in the data.
    flag_cols = [
        c for c in all_columns
        if c.startswith("charlson_") and c != "charlson_index"
    ]
    shared_weight = 0.0
    union_weight = 0.0
    shared_flags: list[str] = []
    unique_flags: list[str] = []
    for col in flag_cols:
        w = CHARLSON_WEIGHTS.get(col, 1)
        a = int(anchor.get(col, 0) or 0)
        c = int(candidate.get(col, 0) or 0)
        if a == 1 and c == 1:
            shared_weight += w
            union_weight += w
            shared_flags.append(col)
        elif a == 1 or c == 1:
            union_weight += w
            unique_flags.append(col)
    flag_sim = shared_weight / union_weight if union_weight > 0 else 1.0

    # Group score: half index, half flags.
    group_score = 0.5 * idx_sim + 0.5 * flag_sim

    # Per-feature contributions.
    contributions: list[tuple[str, float]] = [("charlson_index", (2 * idx_sim - 1) * 0.5)]
    for col in shared_flags:
        contributions.append((col, 0.5 / max(len(flag_cols), 1)))
    for col in unique_flags:
        contributions.append((col, -0.5 / max(len(flag_cols), 1)))
    return group_score, contributions


def _score_comorbidity_set(
    anchor: dict, candidate: dict, all_columns: list[str],
) -> tuple[float, list[tuple[str, float]]]:
    """comorbidity_set group: Jaccard over SNOMED group flags."""
    flag_cols = [c for c in all_columns if c.startswith("snomed_group_")]
    shared = 0
    union = 0
    shared_flags: list[str] = []
    unique_flags: list[str] = []
    for col in flag_cols:
        a = int(anchor.get(col, 0) or 0)
        c = int(candidate.get(col, 0) or 0)
        if a == 1 and c == 1:
            shared += 1
            union += 1
            shared_flags.append(col)
        elif a == 1 or c == 1:
            union += 1
            unique_flags.append(col)
    group_score = shared / union if union > 0 else 1.0

    contributions: list[tuple[str, float]] = []
    if flag_cols:
        per_flag = 1.0 / len(flag_cols)
        for col in shared_flags:
            contributions.append((col, +per_flag))
        for col in unique_flags:
            contributions.append((col, -per_flag))
    return group_score, contributions


def _score_severity(anchor: dict, candidate: dict) -> tuple[float, list[tuple[str, float]]]:
    """severity group: continuous labs + LOS + emergency flag."""
    per_feature: list[tuple[str, float]] = []
    for name, scale in SEVERITY_SCALES.items():
        s = _pairwise_similarity(anchor.get(name), candidate.get(name), scale=scale)
        per_feature.append((name, s))
    # admission_type_EMERGENCY contributes as a binary match.
    emer_sim = _flag_match(
        anchor.get("admission_type_EMERGENCY", 0),
        candidate.get("admission_type_EMERGENCY", 0),
    )
    per_feature.append(("admission_type_EMERGENCY", emer_sim))

    group_score = sum(s for _, s in per_feature) / len(per_feature)
    contributions = [
        (name, (2 * s - 1) / len(per_feature)) for name, s in per_feature
    ]
    return group_score, contributions


def _score_social(anchor: dict, candidate: dict) -> tuple[float, list[tuple[str, float]]]:
    """social group: language_barrier, is_neuro_service."""
    per_feature: list[tuple[str, float]] = []
    for name in GROUP_FEATURES["social"]["categorical"]:
        s = _flag_match(anchor.get(name, 0), candidate.get(name, 0))
        per_feature.append((name, s))
    group_score = sum(s for _, s in per_feature) / max(len(per_feature), 1)
    contributions = [
        (name, (2 * s - 1) / len(per_feature)) for name, s in per_feature
    ] if per_feature else []
    return group_score, contributions


def _resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Validate caller-provided weights; fall back to defaults if None."""
    if weights is None:
        return dict(DEFAULT_GROUP_WEIGHTS)
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"contextual_weights must sum to 1 (got {total:.6f}); "
            f"weights={weights!r}"
        )
    # Missing groups get weight 0 (caller explicitly excluded them).
    resolved = {g: weights.get(g, 0.0) for g in DEFAULT_GROUP_WEIGHTS}
    return resolved


def compute_contextual_similarity(
    anchor_features: dict,
    candidate_features_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict[int, ContextualExplanation]:
    """Score each candidate's contextual similarity to the anchor.

    See module docstring + plan file for the five-group decomposition
    and distance conventions.
    """
    group_weights = _resolve_weights(weights)
    all_columns = list(candidate_features_df.columns)
    out: dict[int, ContextualExplanation] = {}

    for _, row in candidate_features_df.iterrows():
        candidate = row.to_dict()
        hadm_id = int(candidate["hadm_id"])

        per_group_scores: dict[str, float] = {}
        all_contributions: list[tuple[str, float]] = []

        # Each scorer returns (group_score, [(feature, signed_contribution), ...]).
        # Signed contribution is relative to the feature's slot in its group;
        # we scale by the group weight to get its contribution to the OVERALL score.
        scorers = [
            ("demographics", _score_demographics(anchor_features, candidate)),
            ("comorbidity_burden",
             _score_comorbidity_burden(anchor_features, candidate, all_columns)),
            ("comorbidity_set",
             _score_comorbidity_set(anchor_features, candidate, all_columns)),
            ("severity", _score_severity(anchor_features, candidate)),
            ("social", _score_social(anchor_features, candidate)),
        ]
        for group_name, (score, contributions) in scorers:
            per_group_scores[group_name] = float(max(0.0, min(1.0, score)))
            weight = group_weights[group_name]
            for feat, contrib in contributions:
                all_contributions.append((feat, contrib * weight))

        overall = sum(per_group_scores[g] * group_weights[g] for g in per_group_scores)
        overall = float(max(0.0, min(1.0, overall)))

        # Sort contributions by signed magnitude; top 5 each direction.
        all_contributions.sort(key=lambda kv: -kv[1])
        positives = [(f, c) for f, c in all_contributions if c > 1e-9][:5]
        negatives = [(f, c) for f, c in reversed(all_contributions) if c < -1e-9][:5]

        out[hadm_id] = ContextualExplanation(
            overall_score=overall,
            per_group=per_group_scores,
            top_contributors=positives,
            top_detractors=negatives,
        )
    return out


__all__ = ["compute_contextual_similarity"]
