"""The cohort feature catalog sources nominal categories from the schema artifact.

Before the schema-grounding fix, ``admission_type``'s legal categories were a
hardcoded MIMIC-III tuple (``EMERGENCY``/``ELECTIVE``/``URGENT``/``other``) — the
literals that caused the "0 of 0 candidates" bug because MIMIC-IV uses a
different vocabulary. These tests pin the new contract: the catalog reads the
nominal value set from :mod:`src.similarity.categorical_domains` (the frozen,
data-derived artifact), so a fixture/artifact change flows straight through to
the prompt the LLM sees, and no stale literal is baked into code.
"""

from __future__ import annotations

import json

import pytest

from src.pygower import Kind
from src.similarity.feature_catalog import (
    catalog_feature_names,
    cohort_feature_catalog,
)


@pytest.fixture
def patched_domains(tmp_path, monkeypatch):
    """Point the categorical-domain loader at a controlled artifact."""
    import src.similarity.categorical_domains as cd

    artifact = {
        "version": "1",
        "source": "test",
        "domains": {
            "admission_type": {
                "values": ["EW EMER.", "DIRECT EMER.", "URGENT"],
                "counts": {"EW EMER.": 3, "DIRECT EMER.": 2, "URGENT": 1},
                "n": 6,
            },
            "gender": {"values": ["F", "M"], "counts": {"F": 1, "M": 1}, "n": 2},
        },
    }
    path = tmp_path / "domains.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(cd, "CATEGORICAL_DOMAINS_PATH", path)
    return artifact


class TestCohortFeatureCatalog:
    def test_admission_type_categories_reflect_artifact(self, patched_domains):
        cat = cohort_feature_catalog()
        assert cat["admission_type"].kind == Kind.NOMINAL
        # Categories come from the (monkeypatched) artifact, in its order.
        assert cat["admission_type"].categories == ("EW EMER.", "DIRECT EMER.", "URGENT")

    def test_no_stale_mimic3_literal_in_categories(self, patched_domains):
        cat = cohort_feature_catalog()
        assert "EMERGENCY" not in cat["admission_type"].categories

    def test_gender_categories_reflect_artifact(self, patched_domains):
        cat = cohort_feature_catalog()
        assert cat["gender"].categories == ("F", "M")

    def test_quantitative_features_unaffected(self, patched_domains):
        cat = cohort_feature_catalog()
        assert cat["age"].kind == Kind.QUANTITATIVE
        assert cat["age"].categories is None

    def test_missing_artifact_degrades_to_no_categories(self, tmp_path, monkeypatch):
        # A missing artifact must not break the catalog: the nominal trait stays
        # legal (its name is still a feature), just with no enumerated categories.
        import src.similarity.categorical_domains as cd

        monkeypatch.setattr(cd, "CATEGORICAL_DOMAINS_PATH", tmp_path / "absent.json")
        cat = cohort_feature_catalog()
        assert cat["admission_type"].kind == Kind.NOMINAL
        assert cat["admission_type"].categories is None
        assert "admission_type" in catalog_feature_names()
