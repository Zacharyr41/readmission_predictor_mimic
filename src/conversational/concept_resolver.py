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


_FALLBACK_MAPPING_FILES = (
    "drug_to_snomed.json",
    "labitem_to_snomed.json",
    "comorbidity_to_snomed.json",
    "organism_to_snomed.json",
    "chartitem_to_snomed.json",
)
"""Forward-index files consulted when ``category_to_snomed`` misses.

Each file is expected to be ``{name: {snomed_code: str, ...}}``; both the
forward name→SCTID lookup and the reverse SCTID→[names] index used by the
SNOMED fallback come from these files."""


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
        # Forward (name → SCTID) and reverse (SCTID → [names]) indices for
        # the SNOMED fallback path. Built lazily from the mapping files.
        self._forward_sctid_index: dict[str, str] | None = None
        self._reverse_sctid_index: dict[str, list[str]] | None = None

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

    def _build_sctid_indices(self) -> tuple[dict[str, str], dict[str, list[str]]]:
        """Build forward/reverse indices over every mapping file.

        Forward: ``name_lower -> sctid`` (used to look up a starting SCTID
        for the concept the user asked about).
        Reverse: ``sctid -> [name_lower]`` (used to reverse-map hierarchy
        descendants back into MIMIC-known names). A single SCTID can
        reverse-map to multiple names — keep them all.
        """
        if self._forward_sctid_index is not None and self._reverse_sctid_index is not None:
            return self._forward_sctid_index, self._reverse_sctid_index

        forward: dict[str, str] = {}
        reverse: dict[str, list[str]] = {}
        for filename in _FALLBACK_MAPPING_FILES:
            path = self._mappings_dir / filename
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load mapping %s: %s", filename, exc)
                continue
            for name, entry in data.items():
                if name == "_metadata" or not isinstance(entry, dict):
                    continue
                sctid = entry.get("snomed_code")
                if not sctid:
                    continue
                sctid = str(sctid)
                name_lower = name.lower()
                # Prefer the first forward entry we saw — curated categories
                # take priority over later mapping files for the same name.
                forward.setdefault(name_lower, sctid)
                reverse.setdefault(sctid, []).append(name_lower)

        self._forward_sctid_index = forward
        self._reverse_sctid_index = reverse
        return forward, reverse

    def resolve(self, concept: ClinicalConcept) -> list[str]:
        """Resolve a concept to specific MIMIC names.

        Resolution order:
          1. ``category_to_snomed`` hit with curated ``members`` → authoritative.
          2. SNOMED hierarchy fallback (if ``_hierarchy`` is supplied and the
             concept's name resolves to a SCTID in one of the mapping files):
             take the SCTID's descendants, reverse-map each to MIMIC-known
             names, and return them if there are at least two. Fewer than
             two means the concept is effectively specific, not categorical.
          3. Pass through ``[concept.name]`` unchanged.

        Curated categories always win: the hierarchy is a safety net for
        terms clinicians use that our curated list doesn't cover.
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

        # Phase 5: SNOMED hierarchy fallback. Conservative — only return
        # expanded names when we find at least two MIMIC-known descendants,
        # which is the signal that the concept is really a category.
        if self._hierarchy is not None:
            expanded = self._resolve_via_hierarchy(concept)
            if expanded is not None:
                return expanded

        return [concept.name]

    def _resolve_via_hierarchy(
        self, concept: ClinicalConcept,
    ) -> list[str] | None:
        """Attempt SNOMED-hierarchy expansion; ``None`` means fall through.

        Kept as a separate method so the happy path of ``resolve`` stays
        linear and the fallback's control flow is explicit.
        """
        forward, reverse = self._build_sctid_indices()
        sctid = forward.get(concept.name.lower())
        if sctid is None:
            return None

        # ``_hierarchy`` is duck-typed to avoid a hard import dependency on
        # the graph-construction layer from the conversational layer.
        try:
            descendants = self._hierarchy.get_descendants(sctid)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover — hierarchy shouldn't raise
            logger.warning("SNOMED hierarchy call failed: %s", exc)
            return None

        # Reverse-map to MIMIC-known names; dedup preserving sorted order.
        expanded_names: set[str] = set()
        for desc_sctid in descendants:
            for name in reverse.get(str(desc_sctid), []):
                expanded_names.add(name)

        if len(expanded_names) < 2:
            # Zero or one descendant in MIMIC → the concept is effectively
            # specific; don't fan out. Pass-through via returning None.
            return None

        result = sorted(expanded_names)
        logger.info(
            "Resolved '%s' via SNOMED hierarchy to %d MIMIC names.",
            concept.name, len(result),
        )
        return result
