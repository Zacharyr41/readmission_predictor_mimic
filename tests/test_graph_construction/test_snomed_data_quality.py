"""Data-quality tests for SNOMED-CT mapping files.

These tests read the real mapping files in data/mappings/ and verify
that SNOMED codes are valid SCTIDs (numeric, plausible length),
that CUI codes have been eliminated, and that coverage thresholds
are met.
"""

import json
import re
from pathlib import Path

import pytest

MAPPINGS_DIR = Path(__file__).parent.parent.parent / "data" / "mappings"

# ---- helpers ----

_SCTID_RE = re.compile(r"^\d{5,18}$")
_LOINC_RE = re.compile(r"^\d+-\d+$")


def _load_map(filename: str) -> dict:
    path = MAPPINGS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found")
    with open(path) as f:
        data = json.load(f)
    data.pop("_metadata", None)
    return data


# ==================== labitem_to_snomed.json ====================


class TestLabitemMapQuality:
    """Validate labitem_to_snomed.json on disk."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.data = _load_map("labitem_to_snomed.json")

    def test_no_cui_codes_in_labitem_map(self):
        """No snomed_code should be a UMLS CUI (e.g. C0201985)."""
        cuis = {
            k: v["snomed_code"]
            for k, v in self.data.items()
            if v.get("snomed_code", "").startswith("C")
        }
        assert len(cuis) == 0, f"{len(cuis)} CUI codes found, e.g. {dict(list(cuis.items())[:5])}"

    def test_snomed_codes_are_plausible_length(self):
        """All snomed_codes must be pure-numeric 5-18 digit SCTIDs."""
        bad = {
            k: v["snomed_code"]
            for k, v in self.data.items()
            if v.get("snomed_code") and not _SCTID_RE.match(str(v["snomed_code"]))
        }
        assert len(bad) == 0, f"{len(bad)} invalid SCTIDs, e.g. {dict(list(bad.items())[:5])}"

    def test_at_least_60_percent_coverage(self):
        """At least 60% of lab items should have a valid SNOMED SCTID."""
        total = len(self.data)
        valid = sum(
            1 for v in self.data.values()
            if v.get("snomed_code") and _SCTID_RE.match(str(v["snomed_code"]))
        )
        pct = valid / total * 100 if total else 0
        assert pct >= 60, f"Only {pct:.1f}% ({valid}/{total}) have valid SCTIDs"

    def test_common_labs_are_mapped(self):
        """Key clinical lab items must have valid SNOMED SCTIDs."""
        required = {
            "50912": "Creatinine",
            "50983": "Sodium",
            "50809": "Glucose",
            "51265": "Platelet Count",
            "51222": "Hemoglobin",
            "50971": "Potassium",
        }
        missing = []
        for itemid, label in required.items():
            entry = self.data.get(itemid, {})
            code = entry.get("snomed_code", "")
            if not _SCTID_RE.match(str(code)):
                missing.append(f"{itemid} ({label}): got '{code}'")
        assert not missing, f"Missing valid SCTID for: {missing}"


# ==================== loinc_to_snomed.json ====================


class TestLoincMapQuality:
    """Validate loinc_to_snomed.json on disk."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.data = _load_map("loinc_to_snomed.json")

    def test_loinc_file_exists(self):
        path = MAPPINGS_DIR / "loinc_to_snomed.json"
        assert path.exists(), "loinc_to_snomed.json does not exist"

    def test_loinc_file_has_minimum_entries(self):
        # Base mapping covers ~500; RF2 package expands to ~41,000
        assert len(self.data) >= 500, f"Only {len(self.data)} entries (need >= 500)"

    def test_loinc_codes_are_valid_format(self):
        """All keys should match the LOINC pattern \\d+-\\d+."""
        bad = [k for k in self.data if not _LOINC_RE.match(k)]
        assert not bad, f"{len(bad)} invalid LOINC keys, e.g. {bad[:5]}"

    def test_loinc_snomed_codes_are_numeric(self):
        """All SNOMED codes in LOINC map must be numeric SCTIDs."""
        bad = {
            k: v.get("snomed_code")
            for k, v in self.data.items()
            if not _SCTID_RE.match(str(v.get("snomed_code", "")))
        }
        assert not bad, f"{len(bad)} non-numeric SCTIDs, e.g. {dict(list(bad.items())[:5])}"

    def test_common_loinc_codes_mapped(self):
        """Key LOINC codes used in MIMIC labs must be present."""
        required = {
            "2160-0": "Creatinine",
            "2951-2": "Sodium",
            "2345-7": "Glucose",
            "777-3": "Platelets",
            "718-7": "Hemoglobin",
            "2823-3": "Potassium",
        }
        missing = [f"{code} ({name})" for code, name in required.items() if code not in self.data]
        assert not missing, f"Missing LOINC codes: {missing}"
