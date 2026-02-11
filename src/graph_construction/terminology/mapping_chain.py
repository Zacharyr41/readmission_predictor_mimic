"""Ordered waterfall of LOINC→SNOMED mapping sources.

``MappingChain`` tries each ``MappingSource`` in order and returns the
first hit. For batch resolution, each successive source only receives
codes that are still unresolved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph_construction.terminology.mapping_sources import MappingSource

logger = logging.getLogger(__name__)


class MappingChain:
    """Try an ordered list of mapping sources; first hit wins."""

    def __init__(self, sources: list[MappingSource]) -> None:
        self._sources = sources

    def resolve(self, loinc_code: str) -> dict | None:
        for source in self._sources:
            result = source.lookup(loinc_code)
            if result is not None:
                return result
        return None

    def resolve_batch(self, codes: list[str]) -> dict[str, dict]:
        resolved: dict[str, dict] = {}
        remaining = list(codes)
        for source in self._sources:
            if not remaining:
                break
            hits = source.lookup_batch(remaining)
            resolved.update(hits)
            remaining = [c for c in remaining if c not in resolved]
        return resolved
