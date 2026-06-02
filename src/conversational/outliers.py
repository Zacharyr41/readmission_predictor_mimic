"""Biological-possibility limit resolution for outlier screening.

Resolves an absolute ``[low, high]`` envelope per analyte — the bounds beyond
which a measurement is physiologically impossible / a data-entry error. These
are DELIBERATELY WIDER than the normal reference ranges in
``loinc_reference_ranges.json``: a sepsis lactate of 12 mmol/L is high but real
(kept), while a lactate of 1,000,000 is impossible (removed). The bounds feed
``sql_fastpath.OutlierScreen`` as a cheap constant ``BETWEEN`` predicate.

Layered, no-curation-as-mechanism:

1. **Derived-limits cache** — ``data/ontology_cache/biological_limits.json``,
   keyed by LOINC code and analyte alias. Ships with a literature-grounded seed
   for the common analytes; auditable, fast, offline.
2. **Grounded derivation (escape hatch)** — on a cache miss, reuse the critic's
   :class:`EvidenceAgent` (Sonnet/Opus — never Haiku, per the standing rule for
   judgment tasks) with a focused prompt, validate, and write the answer back to
   the cache with its provenance. Runs at most once per analyte.
3. **Graceful skip** — no cache hit and no derivation ⇒ return ``None``; the
   caller skips screening for that query. Never invent a bound; never
   false-remove.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.conversational.health_evidence import EvidenceAgent
from src.conversational.health_evidence.tool_defs import ALL_TOOL_DEFS
from src.conversational.health_evidence.tools import (  # noqa: F401  (test/dispatch hooks)
    loinc_reference_range,
    pubmed_search,
)
from src.conversational.models import ClinicalConcept

logger = logging.getLogger(__name__)

# Judgment task → Sonnet (never Haiku). Opus is also acceptable for depth; the
# critic uses Opus, this defaults to the cheaper-but-still-capable Sonnet.
_DEFAULT_JUDGMENT_MODEL = "claude-sonnet-4-6"
_DERIVATION_MAX_TOKENS = 500
_DERIVATION_MAX_ITERATIONS = 3
_DERIVATION_TIMEOUT_SECONDS = 30.0

# Grounding tools the derivation agent may call: LOINC for units/orientation,
# PubMed for the physiological-extremes literature.
_GROUNDING_TOOLS: tuple[str, ...] = ("loinc_reference_range", "pubmed_search")

_DERIVATION_SYSTEM_PROMPT = (
    "You are a clinical laboratory expert. Given an analyte (lab biomarker or "
    "vital sign), return the ABSOLUTE BIOLOGICAL POSSIBILITY LIMITS — the "
    "values beyond which a measurement is physiologically impossible in a "
    "living human and therefore must be a data-entry error. These limits are "
    "DELIBERATELY WIDER than the normal reference range: pathological extremes "
    "(severe sepsis, DKA, renal failure, leukemia) must fall INSIDE the "
    "envelope and be kept. Only physically-impossible values fall outside. Use "
    "the loinc_reference_range tool to confirm units/orientation and "
    "pubmed_search for documented physiological extremes if unsure. Respond "
    "with ONLY a JSON object: {\"low\": <number>, \"high\": <number>, "
    "\"units\": \"<string>\"}. No prose."
)


@dataclass
class BiologicalLimits:
    """Absolute biological-possibility envelope for one analyte.

    ``source`` records provenance for the UI/traceability: e.g.
    ``"seed:literature"`` for a seeded entry or ``"derived"`` for one the
    EvidenceAgent produced.
    """

    low: float
    high: float
    units: str | None
    source: str


class BiologicalLimitsResolver:
    """Resolve a :class:`BiologicalLimits` envelope for a clinical concept.

    Cache-first, with an optional grounded-derivation escape hatch. Construct
    once and reuse — the cache index is built at construction.
    """

    def __init__(
        self,
        *,
        cache_path: Path | str,
        client: Any | None = None,
        enable_derivation: bool = True,
        model: str = _DEFAULT_JUDGMENT_MODEL,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.client = client
        self.enable_derivation = enable_derivation
        self.model = model
        self._by_loinc, self._by_name = self._load_cache()

    # -- public ----------------------------------------------------------

    def resolve(self, concept: ClinicalConcept) -> BiologicalLimits | None:
        """Return the envelope for ``concept`` or ``None`` if no bound exists.

        Tries the cache (LOINC code first, then analyte name), then a single
        grounded derivation when enabled and a client is available. ``None``
        means the caller should skip screening (never a guessed bound).
        """
        hit = self._lookup(concept)
        if hit is not None:
            return hit
        if self.enable_derivation and self.client is not None:
            return self._derive_and_cache(concept)
        return None

    # -- cache -----------------------------------------------------------

    def _load_cache(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """Load the seed/derived cache into ``(by_loinc, by_name)`` indexes.

        Missing or malformed files degrade to empty indexes — the resolver
        then either derives (if enabled) or skips, never crashes.
        """
        by_loinc: dict[str, dict] = {}
        by_name: dict[str, dict] = {}
        if not self.cache_path.exists():
            return by_loinc, by_name
        try:
            raw = json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("biological_limits cache unreadable (%s); ignoring", exc)
            return by_loinc, by_name
        for entry in raw.get("analytes", []) or []:
            if not isinstance(entry, dict) or "low" not in entry or "high" not in entry:
                continue
            loinc = entry.get("loinc")
            if loinc:
                by_loinc[str(loinc)] = entry
            for alias in entry.get("aliases", []) or []:
                by_name[str(alias).strip().lower()] = entry
        return by_loinc, by_name

    def _lookup(self, concept: ClinicalConcept) -> BiologicalLimits | None:
        entry: dict | None = None
        loinc = getattr(concept, "loinc_code", None)
        if loinc and str(loinc) in self._by_loinc:
            entry = self._by_loinc[str(loinc)]
        elif concept.name:
            entry = self._by_name.get(concept.name.strip().lower())
        if entry is None:
            return None
        return BiologicalLimits(
            low=float(entry["low"]),
            high=float(entry["high"]),
            units=entry.get("units"),
            source=entry.get("source", "seed"),
        )

    # -- grounded derivation ---------------------------------------------

    def _derive_and_cache(self, concept: ClinicalConcept) -> BiologicalLimits | None:
        """Derive limits via the EvidenceAgent, validate, and write back.

        Returns ``None`` (skip screening) on any failure — malformed JSON,
        non-numeric or inverted bounds, or an empty agent result.
        """
        agent = EvidenceAgent(
            self.client,
            model=self.model,
            max_tokens=_DERIVATION_MAX_TOKENS,
            max_iterations=_DERIVATION_MAX_ITERATIONS,
            timeout=_DERIVATION_TIMEOUT_SECONDS,
            tools=list(_grounding_tool_defs()),
            tool_dispatch=_grounding_tool_dispatch(),
        )
        loinc = getattr(concept, "loinc_code", None)
        user_prompt = (
            f"Analyte: {concept.name}. "
            f"LOINC: {loinc or 'unknown'}. "
            "Give the absolute biological possibility limits as JSON."
        )
        result = agent.consult(_DERIVATION_SYSTEM_PROMPT, user_prompt)
        limits = self._validate(result.parsed_json)
        if limits is None:
            logger.info(
                "biological-limits derivation produced no usable bound for %r",
                concept.name,
            )
            return None
        self._write_back(concept, limits, result)
        return limits

    @staticmethod
    def _validate(payload: Any) -> BiologicalLimits | None:
        if not isinstance(payload, dict):
            return None
        try:
            low = float(payload["low"])
            high = float(payload["high"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (low < high):
            return None
        units = payload.get("units")
        return BiologicalLimits(
            low=low, high=high,
            units=str(units) if units else None,
            source="derived",
        )

    def _write_back(
        self, concept: ClinicalConcept, limits: BiologicalLimits, result: Any,
    ) -> None:
        """Append the derived entry to the cache file and refresh the indexes.

        Best-effort: a write failure logs and leaves the in-memory result
        usable for this run (the next run simply re-derives).
        """
        citations = [
            {"source": getattr(c, "source", None), "id": getattr(c, "id", None)}
            for c in (getattr(result, "observed_citations", None) or [])
        ]
        entry = {
            "loinc": getattr(concept, "loinc_code", None),
            "aliases": [concept.name.strip().lower()] if concept.name else [],
            "low": limits.low,
            "high": limits.high,
            "units": limits.units,
            "source": limits.source,
            "derived_at": date.today().isoformat(),
            "citations": citations,
        }
        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text())
            else:
                data = {"analytes": []}
            data.setdefault("analytes", []).append(entry)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(data, indent=2) + "\n")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to write biological_limits cache (%s)", exc)
            return
        # Refresh indexes so a second resolve() in the same process hits cache.
        loinc = entry["loinc"]
        if loinc:
            self._by_loinc[str(loinc)] = entry
        for alias in entry["aliases"]:
            self._by_name[alias] = entry


# ---------------------------------------------------------------------------
# Tool wiring — mirrors critic._critic_tool_dispatch so tests can monkeypatch
# ``src.conversational.outliers.<tool_name>`` to inject canned tool responses.
# ---------------------------------------------------------------------------


def _grounding_tool_defs() -> list[dict[str, Any]]:
    return [d for d in ALL_TOOL_DEFS if d["name"] in _GROUNDING_TOOLS]


def _grounding_tool_dispatch() -> dict[str, Any]:
    """Resolve tool names through this module's globals at call time, so a
    test that monkeypatches ``outliers.loinc_reference_range`` takes effect.

    The ``_n=name`` default-arg trick avoids Python's late-binding closure
    pitfall (every lambda would otherwise close over the last ``name``)."""
    module_dict = sys.modules[__name__].__dict__
    return {
        name: (lambda _n=name, **kw: module_dict[_n](**kw))
        for name in _GROUNDING_TOOLS
    }
