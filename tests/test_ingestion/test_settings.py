"""Tests for data_source and bigquery_project configuration fields."""

import pytest
from pydantic import ValidationError

from config.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch, tmp_path):
    """Prevent .env file from leaking into settings tests."""
    monkeypatch.delenv("BIGQUERY_PROJECT", raising=False)
    monkeypatch.delenv("DATA_SOURCE", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in tmp_path


class TestDataSourceConfig:
    """Tests for data_source and bigquery_project settings."""

    def test_data_source_defaults_to_bigquery(self, tmp_path):
        """data_source should default to 'bigquery' when bigquery_project is set."""
        s = Settings(
            mimic_iv_path=tmp_path,
            duckdb_path=tmp_path / "test.duckdb",
            clinical_tkg_repo=tmp_path,
            bigquery_project="my-proj",
        )
        assert s.data_source == "bigquery"

    def test_local_source_succeeds_without_bigquery_project(self, tmp_path):
        """data_source='local' should not require bigquery_project."""
        s = Settings(
            mimic_iv_path=tmp_path,
            duckdb_path=tmp_path / "test.duckdb",
            clinical_tkg_repo=tmp_path,
            data_source="local",
        )
        assert s.data_source == "local"
        assert s.bigquery_project is None

    def test_bigquery_source_without_project_raises(self, tmp_path):
        """data_source='bigquery' without bigquery_project must raise."""
        with pytest.raises(ValidationError, match="BIGQUERY_PROJECT is required"):
            Settings(
                mimic_iv_path=tmp_path,
                duckdb_path=tmp_path / "test.duckdb",
                clinical_tkg_repo=tmp_path,
                data_source="bigquery",
            )

    def test_bigquery_source_with_project_succeeds(self, tmp_path):
        """data_source='bigquery' with bigquery_project should work."""
        s = Settings(
            mimic_iv_path=tmp_path,
            duckdb_path=tmp_path / "test.duckdb",
            clinical_tkg_repo=tmp_path,
            data_source="bigquery",
            bigquery_project="my-proj",
        )
        assert s.data_source == "bigquery"
        assert s.bigquery_project == "my-proj"

    def test_invalid_data_source_rejected(self, tmp_path):
        """Invalid data_source values should be rejected by Literal validation."""
        with pytest.raises(ValidationError):
            Settings(
                mimic_iv_path=tmp_path,
                duckdb_path=tmp_path / "test.duckdb",
                clinical_tkg_repo=tmp_path,
                data_source="foo",
            )
