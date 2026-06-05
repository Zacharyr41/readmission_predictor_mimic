"""The SQL-extractable trait vocabulary for the contextual cohort path.

A ``CohortDefinition`` (plan II-D) may only name traits the cohort runner can
actually pull. For ``source="sql"`` traits that means a column produced by
``src.similarity.run._fetch_admission_features`` — otherwise ``run_cohort``'s
missing-column guard rejects the whole definition. This module is the single
source of truth the definition builder uses to (a) tell the LLM which feature
names are legal and (b) reject/repair any it invents, exactly the way the
question decomposer constrains ``patient_filter`` fields to a registry.

This is a *schema* contract, not a curated synonym list: the names mirror the
extractor's output columns one-for-one, and the quantitative set is exactly the
keys of the frozen reference-ranges artifact (plan II-E), so a quantitative
trait always has a population scale to normalize against. Keep the three in
sync — extractor columns, frozen-ranges keys, and this catalog — if the
extractor's feature set changes. ``graph_temporal`` traits are deliberately
NOT listed here; they are graph-derived (plan III-A) and exempt from this guard.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pygower import Kind
from src.similarity.categorical_domains import load_categorical_domains


@dataclass(frozen=True)
class FeatureInfo:
    """One legal cohort trait: its Gower ``kind`` and (for nominal) categories."""

    name: str
    kind: Kind
    description: str
    categories: tuple[str, ...] | None = None


# Quantitative features — these names are simultaneously columns of
# ``_fetch_admission_features`` AND keys of the frozen reference-ranges artifact
# (``data/mappings/similarity_reference_ranges.json``), so each carries a
# population p1/p99 scale for the Gower kernel.
_QUANTITATIVE: dict[str, str] = {
    "age": "patient age in years at admission",
    "icu_los_hours": "total ICU length of stay across stays, in hours",
    "creatinine_max": "peak serum creatinine during the admission (mg/dL)",
    "sodium_mean": "mean serum sodium during the admission (mEq/L)",
    "platelet_min": "minimum platelet count during the admission (K/uL)",
}

# Nominal features — raw categorical columns the extractor also emits. Only the
# human description lives here; the legal category VALUES are NOT hardcoded but
# sourced from the frozen schema-grounded categorical-domain artifact
# (``data/mappings/similarity_categorical_domains.json``), so the prompt teaches
# the LLM the real MIMIC-IV vocabulary (``EW EMER.``/``DIRECT EMER.``/…) instead
# of stale MIMIC-III literals (``EMERGENCY``) that match nothing. Keep these
# names in sync with the extractor columns and the artifact.
_NOMINAL: dict[str, str] = {
    "gender": "recorded sex",
    "admission_type": "admission type",
}


def cohort_feature_catalog() -> dict[str, FeatureInfo]:
    """Return ``{name: FeatureInfo}`` for every legal ``source="sql"`` trait.

    Nominal categories are sourced from the frozen categorical-domain artifact
    (schema-grounded), so they always reflect the live MIMIC-IV vocabulary. A
    missing artifact degrades gracefully — the trait stays legal, just with no
    enumerated categories surfaced in the prompt.
    """
    domains = load_categorical_domains()
    catalog: dict[str, FeatureInfo] = {}
    for name, desc in _QUANTITATIVE.items():
        catalog[name] = FeatureInfo(name, Kind.QUANTITATIVE, desc)
    for name, desc in _NOMINAL.items():
        cats = domains.get(name)
        catalog[name] = FeatureInfo(
            name, Kind.NOMINAL, desc, categories=cats or None
        )
    return catalog


def catalog_feature_names() -> set[str]:
    """The set of legal ``source="sql"`` trait names."""
    return set(_QUANTITATIVE) | set(_NOMINAL)
