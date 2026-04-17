"""RxNav REST client with disk-backed cache.

The user's 2026-04-17 decision for Phase 8b picks ``rxnav`` HTTP at
query time as the RxNorm resolution mechanism. The correctness-first
rule (``memory/feedback_correctness_no_curation.md``) means we cannot
fall back to string matching on free-text drug names when an
``InterventionSpec`` carries an RxCUI — the resolver must either
resolve it through an authoritative ontology path, or raise.

This module encapsulates the HTTP dependency:

  * one ``requests.Session`` per client instance, reused across calls;
  * a disk-backed JSON cache keyed by ``<rxcui>:<tty-list>`` so the
    same RxCUI never triggers a second network round-trip for the
    same term-type query within or across runs;
  * explicit ``RxNavError`` on transport / HTTP failures so callers can
    decide whether to raise upward or degrade to a different path.

Tests monkey-patch ``RxNavClient.get_related_rxcuis`` to avoid network
I/O; production code uses the real client.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class RxNavError(RuntimeError):
    """Raised on any unrecoverable rxnav HTTP failure."""


class RxNavClient:
    """Thin wrapper around the rxnav REST API.

    All public methods cache by ``(rxcui, *args)`` tuple. Cache hits are
    returned from memory if already loaded, else from disk. On a cache
    miss the HTTP request runs, the result is memoised and written to
    disk before returning.

    The cache file is a single JSON document; the first call that
    populates it creates the parent directory. Concurrent writers are
    not protected (if that becomes a problem in 8d+ parallel cohort
    builds we can swap to an on-disk SQLite / DuckDB cache).
    """

    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    # Clinical-drug term types that actually appear in MIMIC
    # ``prescriptions.drug`` strings. SCD/SBD are dose-form-branded
    # products; GPCK/BPCK are packs. "IN" (ingredient) is the root you
    # typically pass in so we don't include it in the default descendant
    # fetch. See https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html.
    DEFAULT_TTY = ("SCD", "SBD", "GPCK", "BPCK", "SCDC", "SCDF")

    def __init__(
        self,
        cache_path: Path | None = None,
        *,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        repo_root = Path(__file__).parent.parent.parent
        self._cache_path = cache_path or (
            repo_root / "data" / "ontology_cache" / "rxnav_cache.json"
        )
        self._timeout = timeout
        self._session = session or requests.Session()
        self._cache: dict[str, Any] = {}
        self._load_cache()

    # ---- cache I/O -------------------------------------------------------

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with self._cache_path.open() as f:
                self._cache = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            # Corrupt cache file is not fatal — we just re-populate.
            logger.warning("rxnav cache at %s is unreadable (%s); ignoring", self._cache_path, e)
            self._cache = {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(self._cache, f, indent=2, sort_keys=True)
        tmp.replace(self._cache_path)

    # ---- public API ------------------------------------------------------

    def get_related_rxcuis(
        self,
        rxcui: str,
        tty: tuple[str, ...] = DEFAULT_TTY,
    ) -> list[dict[str, str]]:
        """Return RxCUIs of ``tty`` term types related to ``rxcui``.

        Each returned dict has keys ``{"rxcui", "name", "tty"}``. The
        rxnav endpoint used is
        ``/rxcui/{rxcui}/related.json?tty=<tty1>+<tty2>+…``.

        Raises ``RxNavError`` on transport / HTTP failure — the caller
        is responsible for deciding whether to propagate or fall back.
        Ingredient RxCUIs with no descendants of the requested term
        types return an empty list (not an error).
        """
        cache_key = f"{rxcui}:{','.join(tty)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.BASE_URL}/rxcui/{rxcui}/related.json"
        params = {"tty": "+".join(tty)}
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            raise RxNavError(
                f"rxnav related.json failed for rxcui={rxcui} tty={tty}: {e}"
            ) from e
        except ValueError as e:  # json decode error
            raise RxNavError(
                f"rxnav returned non-JSON for rxcui={rxcui}: {e}"
            ) from e

        related = []
        group = payload.get("relatedGroup", {}).get("conceptGroup") or []
        for arm in group:
            arm_tty = arm.get("tty")
            for concept in arm.get("conceptProperties") or []:
                related.append({
                    "rxcui": str(concept.get("rxcui", "")),
                    "name": str(concept.get("name", "")),
                    "tty": str(arm_tty or ""),
                })

        self._cache[cache_key] = related
        self._save_cache()
        return related

    def get_ingredient_name(self, rxcui: str) -> str | None:
        """Return the canonical ingredient name for ``rxcui`` via
        ``/rxcui/{rxcui}/properties.json``. ``None`` if not found.

        Used as a cross-check when the local drug-to-snomed cache has
        no entry for an RxCUI — we can still log the canonical name in
        the provenance trace.
        """
        cache_key = f"{rxcui}:properties"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.BASE_URL}/rxcui/{rxcui}/properties.json"
        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            raise RxNavError(f"rxnav properties.json failed for rxcui={rxcui}: {e}") from e
        except ValueError as e:
            raise RxNavError(f"rxnav returned non-JSON for rxcui={rxcui}: {e}") from e

        name = payload.get("properties", {}).get("name")
        name_str = str(name) if name else None
        self._cache[cache_key] = name_str
        self._save_cache()
        return name_str
