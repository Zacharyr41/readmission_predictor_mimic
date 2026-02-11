"""Tests for LOINC→SNOMED mapping sources (StaticMappingSource, UMLSCrosswalkSource)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.graph_construction.terminology.mapping_sources import (
    StaticMappingSource,
    UMLSCrosswalkSource,
)


# ── StaticMappingSource ──────────────────────────────────────────────────────


class TestStaticMappingSource:

    @pytest.fixture
    def loinc_json(self, tmp_path: Path) -> Path:
        data = {
            "_metadata": {"source": "test"},
            "2160-0": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement"},
            "2951-2": {"snomed_code": "104934005", "snomed_term": "Sodium measurement"},
        }
        p = tmp_path / "loinc_to_snomed.json"
        p.write_text(json.dumps(data))
        return p

    def test_lookup_known_code(self, loinc_json: Path) -> None:
        src = StaticMappingSource(loinc_json)
        result = src.lookup("2160-0")
        assert result is not None
        assert result["snomed_code"] == "70901006"
        assert result["snomed_term"] == "Creatinine measurement"

    def test_lookup_unknown_code(self, loinc_json: Path) -> None:
        src = StaticMappingSource(loinc_json)
        assert src.lookup("99999-9") is None

    def test_rejects_cui_format_code(self, tmp_path: Path) -> None:
        """Entry whose snomed_code is a CUI like 'C0201985' should be rejected."""
        data = {
            "2160-0": {"snomed_code": "C0201985", "snomed_term": "Fake CUI"},
        }
        p = tmp_path / "loinc.json"
        p.write_text(json.dumps(data))
        src = StaticMappingSource(p)
        assert src.lookup("2160-0") is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        src = StaticMappingSource(tmp_path / "does_not_exist.json")
        assert src.lookup("2160-0") is None

    def test_lookup_batch(self, loinc_json: Path) -> None:
        src = StaticMappingSource(loinc_json)
        results = src.lookup_batch(["2160-0", "2951-2", "99999-9"])
        assert set(results.keys()) == {"2160-0", "2951-2"}

    def test_metadata_key_stripped(self, loinc_json: Path) -> None:
        """_metadata should not be treated as a mapping entry."""
        src = StaticMappingSource(loinc_json)
        assert src.lookup("_metadata") is None

    def test_name(self, loinc_json: Path) -> None:
        src = StaticMappingSource(loinc_json)
        assert src.name == "static"


# ── UMLSCrosswalkSource ─────────────────────────────────────────────────────


class TestUMLSCrosswalkSource:

    @pytest.fixture
    def cache_path(self, tmp_path: Path) -> Path:
        return tmp_path / "loinc_crosswalk_cache.json"

    @pytest.fixture
    def pre_cached(self, cache_path: Path) -> Path:
        cache_path.write_text(json.dumps({
            "2160-0": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement"},
        }))
        return cache_path

    def test_lookup_from_cache_no_api_call(self, pre_cached: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=pre_cached)
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            result = src.lookup("2160-0")
            mock_req.Session.assert_not_called()
        assert result is not None
        assert result["snomed_code"] == "70901006"

    def test_cache_miss_calls_api(self, cache_path: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [{"ui": "70901006", "name": "Creatinine measurement", "obsolete": False}]
        }
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_req.Session.return_value = session
            result = src.lookup("2160-0")
        assert result is not None
        assert result["snomed_code"] == "70901006"
        session.get.assert_called_once()
        # Verify crosswalk URL contains LNC
        call_args = session.get.call_args
        assert "LNC/2160-0" in call_args[0][0]

    def test_api_returns_valid_sctid(self, cache_path: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [{"ui": "104934005", "name": "Sodium measurement", "obsolete": False}]
        }
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_req.Session.return_value = session
            result = src.lookup("2951-2")
        assert result is not None
        assert result["snomed_code"] == "104934005"

    def test_api_returns_cui_rejected(self, cache_path: Path) -> None:
        """Non-numeric ui field (CUI) should be filtered out."""
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [{"ui": "C0201985", "name": "Fake CUI", "obsolete": False}]
        }
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_req.Session.return_value = session
            result = src.lookup("2160-0")
        assert result is None

    def test_api_404_returns_none(self, cache_path: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_req.Session.return_value = session
            result = src.lookup("99999-9")
        assert result is None

    def test_api_429_retries(self, cache_path: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "result": [{"ui": "70901006", "name": "Creatinine", "obsolete": False}]
        }
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.side_effect = [resp_429, resp_ok]
            mock_req.Session.return_value = session
            with patch("src.graph_construction.terminology.mapping_sources.time.sleep"):
                result = src.lookup("2160-0")
        assert result is not None
        assert session.get.call_count == 2

    def test_cache_file_written_after_batch(self, cache_path: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [{"ui": "70901006", "name": "Creatinine", "obsolete": False}]
        }
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_req.Session.return_value = session
            with patch("src.graph_construction.terminology.mapping_sources.time.sleep"):
                src.lookup_batch(["2160-0"])
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert "2160-0" in data

    def test_lookup_batch_splits_cached_uncached(self, pre_cached: Path) -> None:
        """Cached codes skip the API; uncached codes hit it."""
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=pre_cached)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [{"ui": "104934005", "name": "Sodium", "obsolete": False}]
        }
        with patch("src.graph_construction.terminology.mapping_sources.requests") as mock_req:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_req.Session.return_value = session
            with patch("src.graph_construction.terminology.mapping_sources.time.sleep"):
                results = src.lookup_batch(["2160-0", "2951-2"])
        # 2160-0 from cache, 2951-2 from API
        assert "2160-0" in results
        assert "2951-2" in results
        # API should only be called for the uncached code
        assert session.get.call_count == 1

    def test_name(self, cache_path: Path) -> None:
        src = UMLSCrosswalkSource(api_key="fake-key", cache_path=cache_path)
        assert src.name == "umls_crosswalk"
