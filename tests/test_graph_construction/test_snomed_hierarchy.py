"""Tests for the SNOMED IS-A hierarchy provider."""

import json
from pathlib import Path

import pytest

from src.graph_construction.terminology.snomed_hierarchy import SnomedHierarchy


@pytest.fixture
def hierarchy_path(tmp_path):
    """Create a small synthetic hierarchy JSON for testing."""
    # Simplified drug hierarchy:
    #   373873005 (Pharmaceutical product)
    #     └─ 372532007 (Antibiotic)
    #          ├─ 372735009 (Vancomycin)
    #          └─ 372623008 (Ceftriaxone)
    #     └─ 372881000 (Vasopressor)
    #          └─ 372719009 (Norepinephrine)
    data = {
        "_metadata": {"source": "test", "concepts": 5},
        "372735009": {
            "term": "Vancomycin",
            "parents": ["372532007"],
            "ancestors": ["372532007", "373873005"],
        },
        "372623008": {
            "term": "Ceftriaxone",
            "parents": ["372532007"],
            "ancestors": ["372532007", "373873005"],
        },
        "372719009": {
            "term": "Norepinephrine",
            "parents": ["372881000"],
            "ancestors": ["372881000", "373873005"],
        },
        "372532007": {
            "term": "Antibiotic",
            "parents": ["373873005"],
            "ancestors": ["373873005"],
        },
        "372881000": {
            "term": "Vasopressor agent",
            "parents": ["373873005"],
            "ancestors": ["373873005"],
        },
        "373873005": {
            "term": "Pharmaceutical / biologic product",
            "parents": [],
            "ancestors": [],
        },
    }
    path = tmp_path / "snomed_hierarchy.json"
    path.write_text(json.dumps(data))
    return path


class TestSnomedHierarchy:
    def test_load_from_json(self, hierarchy_path):
        h = SnomedHierarchy(hierarchy_path)
        assert h.get_term("372735009") == "Vancomycin"

    def test_get_descendants(self, hierarchy_path):
        h = SnomedHierarchy(hierarchy_path)
        # Antibiotic (372532007) has two descendants: Vancomycin, Ceftriaxone
        descendants = h.get_descendants("372532007")
        assert set(descendants) == {"372735009", "372623008"}

    def test_get_descendants_transitive(self, hierarchy_path):
        """Pharmaceutical root should find all drugs."""
        h = SnomedHierarchy(hierarchy_path)
        descendants = h.get_descendants("373873005")
        assert len(descendants) == 5  # antibiotic, ceftriaxone, vancomycin, vasopressor, norepinephrine

    def test_is_a(self, hierarchy_path):
        h = SnomedHierarchy(hierarchy_path)
        assert h.is_a("372735009", "372532007") is True  # Vancomycin IS-A Antibiotic
        assert h.is_a("372735009", "373873005") is True  # Vancomycin IS-A Pharmaceutical
        assert h.is_a("372735009", "372881000") is False  # Vancomycin IS NOT A Vasopressor

    def test_missing_code_returns_empty(self, hierarchy_path):
        h = SnomedHierarchy(hierarchy_path)
        assert h.get_descendants("999999999") == []
        assert h.is_a("999999999", "372532007") is False

    def test_get_ancestors(self, hierarchy_path):
        h = SnomedHierarchy(hierarchy_path)
        ancestors = h.get_ancestors("372735009")
        assert "372532007" in ancestors  # Antibiotic
        assert "373873005" in ancestors  # Pharmaceutical

    def test_no_hierarchy_file_degrades_gracefully(self, tmp_path):
        """Missing file → empty hierarchy, no crash."""
        h = SnomedHierarchy(tmp_path / "nonexistent.json")
        assert h.get_descendants("372532007") == []
        assert h.is_a("372735009", "372532007") is False
