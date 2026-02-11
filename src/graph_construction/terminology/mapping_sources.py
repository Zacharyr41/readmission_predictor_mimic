"""Pluggable LOINC→SNOMED mapping sources.

Provides a ``MappingSource`` protocol with two concrete implementations:

* ``StaticMappingSource`` — reads from a pre-generated JSON file.
* ``UMLSCrosswalkSource`` — queries the NLM UMLS REST crosswalk endpoint,
  with a lazy disk-cache so repeat builds incur zero API calls.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
SCTID_RE = re.compile(r"^\d{5,18}$")


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class MappingSource(Protocol):
    """Interface for a LOINC→SNOMED mapping source."""

    @property
    def name(self) -> str: ...

    def lookup(self, loinc_code: str) -> dict | None:
        """Return ``{"snomed_code": ..., "snomed_term": ...}`` or ``None``."""
        ...

    def lookup_batch(self, codes: list[str]) -> dict[str, dict]:
        """Return ``{loinc_code: {"snomed_code": ..., "snomed_term": ...}}``."""
        ...


# ── StaticMappingSource ──────────────────────────────────────────────────────


class StaticMappingSource:
    """Load LOINC→SNOMED mappings from a pre-generated JSON file.

    Lazy-loads the file on first access. Entries whose ``snomed_code`` does
    not match the 5–18 digit SCTID pattern are silently rejected.
    """

    def __init__(self, json_path: Path) -> None:
        self._path = json_path
        self._data: dict | None = None

    @property
    def name(self) -> str:
        return "static"

    def _ensure_loaded(self) -> dict:
        if self._data is None:
            if not self._path.exists():
                logger.warning("Static mapping file not found: %s", self._path)
                self._data = {}
            else:
                with open(self._path) as f:
                    raw = json.load(f)
                raw.pop("_metadata", None)
                self._data = raw
        return self._data

    def lookup(self, loinc_code: str) -> dict | None:
        data = self._ensure_loaded()
        entry = data.get(loinc_code)
        if entry is None:
            return None
        code = entry.get("snomed_code", "")
        if not SCTID_RE.match(str(code)):
            return None
        return {"snomed_code": str(code), "snomed_term": entry.get("snomed_term", "")}

    def lookup_batch(self, codes: list[str]) -> dict[str, dict]:
        results = {}
        for code in codes:
            hit = self.lookup(code)
            if hit is not None:
                results[code] = hit
        return results


# ── UMLSCrosswalkSource ─────────────────────────────────────────────────────


class UMLSCrosswalkSource:
    """Query the NLM UMLS REST crosswalk (LNC→SNOMEDCT_US) with disk caching.

    On first call the cache file is loaded (if it exists). Cache hits are
    returned immediately; cache misses trigger an API request. After
    ``lookup_batch()`` completes, the cache is persisted to disk so that
    subsequent builds incur zero API calls for already-resolved codes.
    """

    def __init__(self, api_key: str, cache_path: Path) -> None:
        self._api_key = api_key
        self._cache_path = cache_path
        self._cache: dict | None = None

    @property
    def name(self) -> str:
        return "umls_crosswalk"

    def _ensure_cache(self) -> dict:
        if self._cache is None:
            if self._cache_path.exists():
                with open(self._cache_path) as f:
                    self._cache = json.load(f)
                logger.info(
                    "Loaded %d entries from crosswalk cache %s",
                    len(self._cache),
                    self._cache_path.name,
                )
            else:
                self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2))

    def _crosswalk_single(self, session, loinc_code: str) -> dict | None:
        """Query UMLS crosswalk for one LOINC code. Returns dict or None."""
        url = f"{UMLS_BASE}/crosswalk/current/source/LNC/{loinc_code}"
        params = {"targetSource": "SNOMEDCT_US", "apiKey": self._api_key}
        for attempt in range(3):
            try:
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                if resp.status_code in (404, 400):
                    return None
                resp.raise_for_status()
                results = resp.json().get("result", [])
                if not results:
                    return None
                non_obsolete = [r for r in results if not r.get("obsolete", False)]
                best = non_obsolete[0] if non_obsolete else results[0]
                sctid = best.get("ui", "")
                if SCTID_RE.match(sctid):
                    return {"snomed_code": sctid, "snomed_term": best.get("name", "")}
                return None
            except Exception:
                time.sleep(2 ** attempt)
        return None

    def lookup(self, loinc_code: str) -> dict | None:
        cache = self._ensure_cache()
        if loinc_code in cache:
            return cache[loinc_code]
        # API call
        session = requests.Session()
        result = self._crosswalk_single(session, loinc_code)
        if result is not None:
            cache[loinc_code] = result
        return result

    def lookup_batch(self, codes: list[str]) -> dict[str, dict]:
        cache = self._ensure_cache()

        # Split cached vs uncached
        results = {}
        uncached = []
        for code in codes:
            if code in cache:
                results[code] = cache[code]
            else:
                uncached.append(code)

        if not uncached:
            self._save_cache()
            return results

        # Query uncached codes concurrently
        session = requests.Session()
        total = len(uncached)
        done = 0
        start = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._crosswalk_single, session, code): code
                for code in uncached
            }
            for future in as_completed(futures):
                code = futures[future]
                done += 1
                try:
                    result = future.result()
                    if result is not None:
                        results[code] = result
                        cache[code] = result
                except Exception as e:
                    logger.debug("Crosswalk error for %s: %s", code, e)
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  crosswalk: %d/%d (%d mapped, %.1f/sec)",
                        done, total, len(results), rate,
                    )

        self._save_cache()
        return results
