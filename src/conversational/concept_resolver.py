"""Resolve category-level clinical concepts to specific MIMIC identifiers.

When the decomposer outputs a category like "antibiotics", this module
expands it to specific drug names (vancomycin, ceftriaxone, ...) using
a curated category-to-SNOMED mapping with known MIMIC members.

Falls back to the SNOMED IS-A hierarchy (if available) for categories
not in the curated map.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.conversational.models import ClinicalConcept

logger = logging.getLogger(__name__)


class ConceptResolver:
    """Resolve clinical concept categories to specific MIMIC names.

    Args:
        mappings_dir: Path to ``data/mappings/`` directory.
        hierarchy: Optional ``SnomedHierarchy`` for full hierarchy resolution.
    """

    def __init__(
        self,
        mappings_dir: Path,
        hierarchy: object | None = None,
    ) -> None:
        self._mappings_dir = mappings_dir
        self._hierarchy = hierarchy
        self._category_map: dict | None = None

    def _load_category_map(self) -> dict:
        if self._category_map is not None:
            return self._category_map

        path = self._mappings_dir / "category_to_snomed.json"
        if not path.exists():
            logger.warning("Category map not found: %s", path)
            self._category_map = {}
            return self._category_map

        with open(path) as f:
            raw = json.load(f)

        # Build lowercase-keyed map for case-insensitive lookup
        self._category_map = {
            k.lower(): v
            for k, v in raw.items()
            if k != "_metadata"
        }
        return self._category_map

    def resolve(self, concept: ClinicalConcept) -> list[str]:
        """Resolve a concept to specific MIMIC names.

        If the concept name matches a known category (e.g. "antibiotics"),
        returns a list of specific member names. Otherwise returns
        ``[concept.name]`` unchanged.
        """
        category_map = self._load_category_map()
        entry = category_map.get(concept.name.lower())

        if entry and "members" in entry:
            members = entry["members"]
            logger.info(
                "Resolved category '%s' to %d specific names.",
                concept.name, len(members),
            )
            return members

        # TODO: Fall back to SnomedHierarchy.get_descendants() when available

        return [concept.name]
