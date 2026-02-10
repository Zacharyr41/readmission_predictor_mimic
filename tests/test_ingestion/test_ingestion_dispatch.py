"""Tests for load_mimic_data dispatch routing."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config.settings import Settings
from src.ingestion import load_mimic_data


@pytest.fixture
def local_settings(tmp_path: Path) -> Settings:
    return Settings(
        mimic_iv_path=tmp_path / "mimic",
        duckdb_path=tmp_path / "test.duckdb",
        clinical_tkg_repo=tmp_path / "tkg",
        data_source="local",
    )


@pytest.fixture
def bigquery_settings(tmp_path: Path) -> Settings:
    return Settings(
        mimic_iv_path=tmp_path / "mimic",
        duckdb_path=tmp_path / "test.duckdb",
        clinical_tkg_repo=tmp_path / "tkg",
        data_source="bigquery",
        bigquery_project="my-proj",
    )


class TestLoadMimicDataDispatch:
    """Tests for the load_mimic_data dispatch function."""

    @patch("src.ingestion.mimic_loader.load_mimic_to_duckdb")
    def test_local_calls_csv_loader(self, mock_loader, local_settings):
        """data_source='local' should call load_mimic_to_duckdb."""
        mock_conn = MagicMock()
        mock_loader.return_value = mock_conn

        result = load_mimic_data(local_settings)

        mock_loader.assert_called_once_with(
            local_settings.mimic_iv_path,
            local_settings.duckdb_path,
        )
        assert result is mock_conn

    @patch("src.ingestion.bigquery_loader.load_mimic_from_bigquery")
    def test_bigquery_calls_bigquery_loader(self, mock_loader, bigquery_settings):
        """data_source='bigquery' should call load_mimic_from_bigquery."""
        mock_conn = MagicMock()
        mock_loader.return_value = mock_conn

        result = load_mimic_data(bigquery_settings)

        mock_loader.assert_called_once_with(
            bigquery_project="my-proj",
            db_path=bigquery_settings.duckdb_path,
            cohort_icd_codes=bigquery_settings.cohort_icd_codes,
            patients_limit=bigquery_settings.patients_limit,
        )
        assert result is mock_conn

    def test_unknown_data_source_raises(self, tmp_path):
        """Unknown data_source should raise ValueError."""
        settings = Settings(
            mimic_iv_path=tmp_path,
            duckdb_path=tmp_path / "test.duckdb",
            clinical_tkg_repo=tmp_path,
            data_source="local",
        )
        # Manually override to simulate an unexpected value
        object.__setattr__(settings, "data_source", "foo")

        with pytest.raises(ValueError, match="Unknown data_source"):
            load_mimic_data(settings)
