"""SNOMED-CT concept mapper for MIMIC-IV clinical identifiers.

Loads pre-generated JSON mapping files and resolves MIMIC identifiers
(ICD codes, lab item IDs, drug names, etc.) to SNOMED-CT concepts.
Gracefully degrades when mappings are unavailable — unmapped concepts
return None, never raise.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnomedConcept:
    """A SNOMED-CT concept with code and preferred term."""

    code: str
    term: str


class SnomedMapper:
    """Resolve MIMIC-IV identifiers to SNOMED-CT concepts.

    Loads JSON mapping files lazily from a configurable directory.
    All lookup methods return ``SnomedConcept | None`` and never raise
    on unmapped identifiers.

    Args:
        mappings_dir: Path to directory containing ``*_to_snomed.json`` files.
    """

    def __init__(self, mappings_dir: Path) -> None:
        self._dir = mappings_dir
        self._icd: dict | None = None
        self._labitem: dict | None = None
        self._chartitem: dict | None = None
        self._drug: dict | None = None
        self._organism: dict | None = None
        self._comorbidity: dict | None = None

    # ---- lazy loaders ----

    def _load(self, filename: str) -> dict:
        path = self._dir / filename
        if not path.exists():
            logger.warning("Mapping file not found: %s", path)
            return {}
        with open(path) as f:
            data = json.load(f)
        # Strip metadata key
        data.pop("_metadata", None)
        return data

    @property
    def _icd_map(self) -> dict:
        if self._icd is None:
            self._icd = self._load("icd10cm_to_snomed.json")
        return self._icd

    @property
    def _labitem_map(self) -> dict:
        if self._labitem is None:
            self._labitem = self._load("labitem_to_snomed.json")
        return self._labitem

    @property
    def _chartitem_map(self) -> dict:
        if self._chartitem is None:
            self._chartitem = self._load("chartitem_to_snomed.json")
        return self._chartitem

    @property
    def _drug_map(self) -> dict:
        if self._drug is None:
            self._drug = self._load("drug_to_snomed.json")
        return self._drug

    @property
    def _organism_map(self) -> dict:
        if self._organism is None:
            self._organism = self._load("organism_to_snomed.json")
        return self._organism

    @property
    def _comorbidity_map(self) -> dict:
        if self._comorbidity is None:
            self._comorbidity = self._load("comorbidity_to_snomed.json")
        return self._comorbidity

    # ---- helpers ----

    @staticmethod
    def _normalize_icd(code: str) -> str:
        """Normalize ICD code to MIMIC undotted format (uppercase, no dots)."""
        return code.replace(".", "").strip().upper()

    @staticmethod
    def _normalize_drug(name: str) -> str:
        """Normalize drug name: lowercase, strip trailing dosage info."""
        name = name.lower().strip()
        # Strip common trailing dosage patterns like "500mg", "0.9%", "(1000 mg)"
        name = re.sub(r"\s*\(.*\)\s*$", "", name)
        name = re.sub(r"\s+\d+[\d.]*\s*(mg|g|ml|mcg|%|units?|meq).*$", "", name, flags=re.IGNORECASE)
        return name.strip()

    @staticmethod
    def _to_concept(entry: dict | None) -> SnomedConcept | None:
        if entry is None:
            return None
        code = entry.get("snomed_code")
        term = entry.get("snomed_term")
        if code and term:
            return SnomedConcept(code=str(code), term=str(term))
        return None

    # ---- public API ----

    def get_snomed_for_icd(self, icd_code: str, icd_version: int = 10) -> SnomedConcept | None:
        """Look up SNOMED concept for an ICD code.

        Only ICD-10-CM is supported. ICD-9 codes return None.
        Handles both dotted ("I63.9") and undotted ("I639") formats.
        Falls back to 3-character prefix if exact code not found.
        """
        if icd_version != 10:
            return None
        normalized = self._normalize_icd(icd_code)
        # Try exact match
        entry = self._icd_map.get(normalized)
        if entry:
            return self._to_concept(entry)
        # Try 3-char prefix fallback
        if len(normalized) > 3:
            entry = self._icd_map.get(normalized[:3])
            if entry:
                return self._to_concept(entry)
        return None

    def get_snomed_for_labitem(self, itemid: int) -> SnomedConcept | None:
        """Look up SNOMED concept for a MIMIC lab item ID."""
        entry = self._labitem_map.get(str(itemid))
        return self._to_concept(entry)

    def get_snomed_for_chartitem(self, itemid: int) -> SnomedConcept | None:
        """Look up SNOMED concept for a MIMIC chart event item ID."""
        entry = self._chartitem_map.get(str(itemid))
        return self._to_concept(entry)

    def get_snomed_for_drug(self, drug_name: str) -> SnomedConcept | None:
        """Look up SNOMED concept for a drug name (case-insensitive)."""
        normalized = self._normalize_drug(drug_name)
        # Try exact match
        entry = self._drug_map.get(normalized)
        if entry:
            return self._to_concept(entry)
        # Try matching against stored drug names (handles slight variations)
        for key, val in self._drug_map.items():
            if normalized.startswith(key) or key.startswith(normalized):
                return self._to_concept(val)
        return None

    def get_snomed_for_organism(self, org_name: str) -> SnomedConcept | None:
        """Look up SNOMED concept for an organism name (case-insensitive)."""
        entry = self._organism_map.get(org_name.upper().strip())
        return self._to_concept(entry)

    def get_snomed_for_comorbidity(self, name: str) -> SnomedConcept | None:
        """Look up SNOMED concept for a comorbidity name."""
        entry = self._comorbidity_map.get(name.lower().strip())
        return self._to_concept(entry)

    def coverage_stats(self) -> dict[str, int]:
        """Return count of loaded mappings per category."""
        return {
            "icd": len(self._icd_map),
            "labitem": len(self._labitem_map),
            "chartitem": len(self._chartitem_map),
            "drug": len(self._drug_map),
            "organism": len(self._organism_map),
            "comorbidity": len(self._comorbidity_map),
        }
