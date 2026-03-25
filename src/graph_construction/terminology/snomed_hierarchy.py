"""SNOMED-CT IS-A hierarchy provider.

Loads a pre-generated hierarchy JSON (built by ``scripts/build_snomed_hierarchy.py``)
and supports ancestor/descendant queries for concept resolution.
Degrades gracefully when no hierarchy file is available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SnomedHierarchy:
    """Lazy-loaded SNOMED IS-A hierarchy for concept resolution.

    Args:
        hierarchy_path: Path to ``snomed_hierarchy.json``.
    """

    def __init__(self, hierarchy_path: Path) -> None:
        self._path = hierarchy_path
        self._data: dict[str, dict] | None = None
        self._children_index: dict[str, list[str]] | None = None

    def _load(self) -> dict[str, dict]:
        if self._data is not None:
            return self._data

        if not self._path.exists():
            logger.warning("SNOMED hierarchy file not found: %s", self._path)
            self._data = {}
            return self._data

        with open(self._path) as f:
            raw = json.load(f)

        # Strip metadata key
        self._data = {k: v for k, v in raw.items() if k != "_metadata"}
        return self._data

    def _build_children_index(self) -> dict[str, list[str]]:
        """Build a reverse index: parent → list of children."""
        if self._children_index is not None:
            return self._children_index

        data = self._load()
        index: dict[str, list[str]] = {}
        for sctid, entry in data.items():
            for parent in entry.get("parents", []):
                index.setdefault(parent, []).append(sctid)
        self._children_index = index
        return self._children_index

    def get_term(self, sctid: str) -> str | None:
        """Return the preferred term for a SNOMED code."""
        entry = self._load().get(sctid)
        return entry["term"] if entry else None

    def get_ancestors(self, sctid: str) -> list[str]:
        """Return all ancestor SNOMED codes (transitive IS-A)."""
        entry = self._load().get(sctid)
        if not entry:
            return []
        return list(entry.get("ancestors", []))

    def get_descendants(self, sctid: str) -> list[str]:
        """Return all descendant SNOMED codes (transitive)."""
        children_index = self._build_children_index()
        result: list[str] = []
        queue = list(children_index.get(sctid, []))
        visited: set[str] = set()

        while queue:
            child = queue.pop(0)
            if child in visited:
                continue
            visited.add(child)
            result.append(child)
            queue.extend(children_index.get(child, []))

        return result

    def is_a(self, child_sctid: str, parent_sctid: str) -> bool:
        """Check if child IS-A parent (directly or transitively)."""
        entry = self._load().get(child_sctid)
        if not entry:
            return False
        ancestors = entry.get("ancestors", [])
        parents = entry.get("parents", [])
        return parent_sctid in ancestors or parent_sctid in parents
