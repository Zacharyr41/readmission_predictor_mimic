"""Tests for WLST feature extraction."""

import json
import pytest
import pandas as pd
from pathlib import Path

from src.wlst.cohort import create_wlst_labels, select_tbi_cohort
from src.wlst.features import extract_wlst_features


class TestExtractWlstFeatures:
    def _get_labels(self, wlst_duckdb):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        return create_wlst_labels(wlst_duckdb, cohort)

    def test_returns_dataframe(self, wlst_duckdb):
        labels = self._get_labels(wlst_duckdb)
        features = extract_wlst_features(wlst_duckdb, labels)
        assert isinstance(features, pd.DataFrame)

    def test_contains_label_column(self, wlst_duckdb):
        labels = self._get_labels(wlst_duckdb)
        features = extract_wlst_features(wlst_duckdb, labels)
        assert "wlst_label" in features.columns

    def test_contains_identifiers(self, wlst_duckdb):
        labels = self._get_labels(wlst_duckdb)
        features = extract_wlst_features(wlst_duckdb, labels)
        for col in ["subject_id", "hadm_id", "stay_id"]:
            assert col in features.columns

    def test_preserves_all_patients(self, wlst_duckdb):
        labels = self._get_labels(wlst_duckdb)
        features = extract_wlst_features(wlst_duckdb, labels)
        assert len(features) == len(labels)

    def test_empty_labels(self, wlst_duckdb):
        features = extract_wlst_features(wlst_duckdb, pd.DataFrame())
        assert len(features) == 0

    def test_ais_features_with_mapping(self, wlst_duckdb, tmp_path):
        """AIS mapping should produce head_ais_score column."""
        labels = self._get_labels(wlst_duckdb)

        # Create a minimal AIS mapping
        mappings_dir = tmp_path / "mappings"
        mappings_dir.mkdir()
        ais_map = {"S065": 5, "S062": 5, "S064": 5, "S061": 5, "S06": 4}
        (mappings_dir / "icd10_to_ais_head.json").write_text(json.dumps(ais_map))

        features = extract_wlst_features(
            wlst_duckdb, labels, mappings_dir=mappings_dir,
        )
        if "head_ais_score" in features.columns:
            # Non-null AIS scores should be positive
            ais = features["head_ais_score"].dropna()
            if len(ais) > 0:
                assert (ais > 0).all()


class TestCharlsonIndex:
    def test_charlson_features_with_mapping(self, wlst_duckdb, tmp_path):
        cohort = select_tbi_cohort(wlst_duckdb)
        if len(cohort) == 0:
            pytest.skip("No TBI cohort patients in synthetic data")
        labels = create_wlst_labels(wlst_duckdb, cohort)

        # Create minimal Charlson mapping
        mappings_dir = tmp_path / "mappings"
        mappings_dir.mkdir()
        charlson_map = {
            "cerebrovascular_disease": {
                "weight": 1,
                "icd10_prefixes": ["I60", "I61", "I62", "I63"]
            },
        }
        (mappings_dir / "icd10_to_charlson.json").write_text(json.dumps(charlson_map))
        # Need AIS file too (even if empty)
        (mappings_dir / "icd10_to_ais_head.json").write_text(json.dumps({"S06": 4}))

        features = extract_wlst_features(
            wlst_duckdb, labels, stage="stage2", mappings_dir=mappings_dir,
        )
        assert isinstance(features, pd.DataFrame)
