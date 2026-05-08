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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.conversational.health_evidence.tools import (
    icd_autocode,
    mimic_itemid_search,
)
from src.conversational.models import ClinicalConcept

logger = logging.getLogger(__name__)


# Confidence threshold: keep results where ``confidence >= 0.5`` OR
# ``confidence is None``. OMOPHub's semantic_search often omits scores;
# rejecting None would silently disable grounding for most queries.
_ICD_CONFIDENCE_THRESHOLD = 0.5

# Cap on results accepted from a single icd_autocode call. The MCP tool's
# default top-K is 10; this is a defensive secondary cap so we don't fan
# out to dozens of long-tail codes if a future tweak raises that default.
_ICD_MAX_RESULTS = 10


@lru_cache(maxsize=256)
def _cached_mimic_itemid_search(
    text_lower: str,
) -> tuple[tuple[int, str, str], ...]:
    """Process-wide cache of mimic_itemid_search results.

    Cache key is ``text_lower``. Returns a tuple of
    ``(itemid, label, table)`` triples — table is one of
    ``"labevents"`` or ``"chartevents"`` (or empty string if missing).

    Same negative-result-bypass as ``_cached_icd_autocode``: raises
    ``LookupError`` on ``unavailable`` so the cache never stores
    transient failures.
    """
    envelope = mimic_itemid_search(text_lower, max_results=20)
    if envelope.get("status") != "ok":
        raise LookupError(envelope.get("error") or "mimic_itemid_search unavailable")
    results = envelope.get("results") or []
    out: list[tuple[int, str, str]] = []
    for r in results:
        try:
            itemid = int(r.get("itemid"))
        except (TypeError, ValueError):
            continue
        label = str(r.get("label", ""))
        table = str(r.get("table", ""))
        out.append((itemid, label, table))
    return tuple(out)


@lru_cache(maxsize=256)
def _cached_icd_autocode(
    text_lower: str, version: str,
) -> tuple[tuple[str, float | None], ...]:
    """Process-wide cache of icd_autocode results.

    Cache key is ``(text_lower, version)`` so 'Sepsis' and 'sepsis' share
    the same MCP call. Returns a tuple of ``(code, confidence)`` pairs
    so the cache stores hashable values.

    **Negative-result bypass:** when the underlying tool returns
    ``unavailable``, this function raises ``LookupError`` so the cache
    never stores the failure. Transient OMOPHub failures shouldn't
    poison the cache for the rest of the process. The outer
    ``_ground_via_icd_autocode`` catches LookupError and falls back.
    """
    envelope = icd_autocode(text_lower, version=version, max_results=_ICD_MAX_RESULTS)
    if envelope.get("status") != "ok":
        # Sentinel-raise: never cache negative results.
        raise LookupError(envelope.get("error") or "icd_autocode unavailable")
    results = envelope.get("results") or []
    return tuple(
        (str(r.get("code", "")), r.get("confidence"))
        for r in results
        if r.get("code")
    )


@dataclass(frozen=True)
class DiagnosisResolution:
    """Result of resolving a diagnosis concept to ICD codes for SQL emission.

    Mirrors :class:`BiomarkerResolution` but narrower: ICD codes are the
    grounding output, no LOINC/SNOMED layer.

    Three terminal shapes:

    * **Grounded:** ``icd_codes`` is a non-empty list. The SQL compiler
      emits ``WHERE di.icd_code IN (...) OR <existing LIKE>`` — IN-list
      runs as a parallel OR with LIKE so ICD-9 admissions still match.
      ``fallback_reason`` is ``None``.
    * **Silent fallback:** ``icd_codes`` is ``None``, ``fallback_reason``
      is ``None``. Compiler uses ``names`` for LIKE-only filter. This is
      the path when grounding is disabled or the concept is mid-resolve.
    * **Loud fallback:** ``icd_codes`` is ``None``, ``fallback_reason``
      describes why grounding failed (MCP unavailable, all candidates
      below confidence threshold, etc.). Surfaced to the user via the
      orchestrator's ``fallback_warning`` channel.

    ``names`` is always populated for the parallel-OR LIKE clause that
    catches ICD-9 admissions whose codes aren't in OMOPHub's ICD10CM-only
    coverage.

    ``confidence_floor`` records the minimum confidence among accepted
    codes, for provenance/debugging. ``None`` when no MCP call was made
    (concept-supplied codes path) or all returned confidences were
    ``None``.
    """

    icd_codes: list[str] | None
    names: list[str]
    fallback_reason: str | None
    confidence_floor: float | None


@dataclass(frozen=True)
class BiomarkerResolution:
    """Result of resolving a biomarker concept to MIMIC labitem identifiers.

    There are three terminal shapes:

    * **Grounded:** ``itemids`` is a non-empty list. The compiler emits
      ``WHERE l.itemid IN (?, ?, …)``. ``loinc_code`` and ``snomed_code``
      are populated; ``fallback_reason`` is ``None``.
    * **Silent fallback (no LOINC):** ``itemids`` is ``None``,
      ``loinc_code`` is ``None``, ``fallback_reason`` is ``None``. Compiler
      uses ``names`` for a LIKE filter. This is the *normal* path for
      uncommon labs where the LLM didn't emit a LOINC.
    * **Loud fallback (LOINC supplied but couldn't be grounded):**
      ``itemids`` is ``None``, ``loinc_code`` is set, ``fallback_reason``
      describes why grounding failed (LOINC absent from mapping, or SNOMED
      has no MIMIC labitem coverage). The orchestrator surfaces a
      user-visible warning so the user knows the answer may pool variants.

    ``names`` is always populated and used by the LIKE fallback path.
    """

    itemids: list[int] | None
    names: list[str]
    loinc_code: str | None
    snomed_code: str | None
    fallback_reason: str | None


_FALLBACK_MAPPING_FILES = (
    "drug_to_snomed.json",
    "labitem_to_snomed.json",
    "comorbidity_to_snomed.json",
    "organism_to_snomed.json",
    "chartitem_to_snomed.json",
)
"""Forward-index files consulted when ``category_to_snomed`` misses.

Files keyed by name (drug, comorbidity, organism) use the outer key
directly; files keyed by MIMIC numeric identifier (labitem, chartitem)
use ``entry["label"]`` as the human-readable name. See
``_LABEL_KEYED_FILES`` for the latter set."""


_LABEL_KEYED_FILES = frozenset({
    "labitem_to_snomed.json",
    "chartitem_to_snomed.json",
})
"""Files whose outer key is a MIMIC ``itemid`` rather than a human-readable
name. The actual concept name lives in ``entry["label"]`` for these.

Without this distinction, the forward ``name → SCTID`` index would be
populated with itemid strings (``"50912" → "113075003"``) and a user-facing
lookup like ``forward["creatinine"]`` would always miss — which silently
broke the SNOMED-hierarchy fallback for any lab or chart term."""


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
        *,
        enable_mcp_grounding: bool = False,
    ) -> None:
        self._mappings_dir = mappings_dir
        self._hierarchy = hierarchy
        # When True, ``resolve_diagnosis`` calls the OMOPHub-backed
        # ``icd_autocode`` MCP tool to ground diagnosis concepts to ICD
        # code lists. Default is False so existing test fixtures keep
        # offline-safe behavior; production wiring opts in via the
        # orchestrator's ConceptResolver(...) construction.
        self._enable_mcp_grounding = enable_mcp_grounding
        self._category_map: dict | None = None
        # Forward (name → SCTID) and reverse (SCTID → [names]) indices for
        # the SNOMED fallback path. Built lazily from the mapping files.
        self._forward_sctid_index: dict[str, str] | None = None
        self._reverse_sctid_index: dict[str, list[str]] | None = None
        # LOINC → [itemid] reverse index for biomarker resolution. Goes
        # directly LOINC → itemid rather than LOINC → SNOMED → itemid: the
        # SNOMED layer collapses unit-distinct LOINCs (e.g., serum vs urine
        # creatinine both share SNOMED 113075003), which would defeat the
        # whole point of grounding. Built lazily from labitem_to_snomed.json
        # on the first ``resolve_biomarker`` call that supplies a LOINC.
        self._snomed_mapper: object | None = None
        self._loinc_to_labitem: dict[str, list[int]] | None = None

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
            label_keyed = filename in _LABEL_KEYED_FILES
            for outer_key, entry in data.items():
                if outer_key == "_metadata" or not isinstance(entry, dict):
                    continue
                sctid = entry.get("snomed_code")
                if not sctid:
                    continue
                sctid = str(sctid)
                # For label-keyed files (labitem / chartitem), the outer key
                # is a numeric itemid string and the human-readable name is
                # in ``entry["label"]``. For all other files the outer key
                # IS the name. Skip silently if a label-keyed entry is
                # missing its label rather than indexing by the itemid.
                if label_keyed:
                    name = entry.get("label")
                    if not name:
                        continue
                else:
                    name = outer_key
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

    # -- LOINC-grounded biomarker resolution -------------------------------

    def _get_snomed_mapper(self):
        """Lazy SnomedMapper construction — only built when a biomarker
        concept actually carries a LOINC. Construction itself is cheap
        (no file I/O), but the import drag is paid only on demand."""
        if self._snomed_mapper is None:
            from src.graph_construction.terminology.snomed_mapper import (
                SnomedMapper,
            )
            self._snomed_mapper = SnomedMapper(self._mappings_dir)
        return self._snomed_mapper

    def _build_loinc_to_labitem_index(self) -> None:
        """Build ``{loinc → [itemid]}`` directly from labitem_to_snomed.json.

        Going LOINC-direct (rather than LOINC → SNOMED → itemid) preserves
        specimen-level distinctions: serum creatinine (LOINC 2160-0) and
        urine creatinine (LOINC 2161-8) both share SNOMED 113075003, so a
        SNOMED-mediated lookup would over-include. Building the index
        directly from each labitem entry's ``loinc`` field keeps them
        separate. Constructed lazily on the first biomarker LOINC query.
        """
        if self._loinc_to_labitem is not None:
            return
        mapper = self._get_snomed_mapper()
        reverse: dict[str, list[int]] = {}
        for itemid_key, entry in mapper._labitem_map.items():
            if not isinstance(entry, dict):
                continue
            loinc = entry.get("loinc")
            if loinc:
                try:
                    reverse.setdefault(str(loinc), []).append(int(itemid_key))
                except ValueError:
                    continue
        self._loinc_to_labitem = reverse

    def resolve_biomarker(
        self, concept: ClinicalConcept,
    ) -> BiomarkerResolution:
        """Resolve a biomarker concept to MIMIC labitem identifiers via LOINC.

        Resolution path:
          LOINC code → ``loinc_to_labitem`` index → ``[itemid, …]``

        SNOMED is consulted only for *provenance* (populating ``snomed_code``
        on the result) and to distinguish a "LOINC unknown" failure from a
        "LOINC valid but no MIMIC coverage" failure — the actual resolution
        runs LOINC-direct so unit-distinct LOINCs stay separate.

        Three terminal cases:
          1. No LOINC supplied → silent fallback (names only, no warning).
          2. LOINC supplied but absent from loinc_to_snomed.json → loud
             fallback with ``fallback_reason`` for user-visible warning.
          3. LOINC valid (in loinc_to_snomed.json) but no MIMIC labitem
             carries that LOINC → loud fallback with ``fallback_reason``.

        On success, returns a ``BiomarkerResolution`` with ``itemids``
        populated and ``fallback_reason=None``.
        """
        names = self.resolve(concept)  # always populated; LIKE-fallback names

        if not concept.loinc_code:
            return BiomarkerResolution(
                itemids=None,
                names=names,
                loinc_code=None,
                snomed_code=None,
                fallback_reason=None,
            )

        self._build_loinc_to_labitem_index()
        assert self._loinc_to_labitem is not None
        itemids = sorted(self._loinc_to_labitem.get(concept.loinc_code, []))

        if itemids:
            # Successful grounding. Look up SNOMED for provenance only;
            # the result is correct even if SNOMED lookup happens to fail.
            mapper = self._get_snomed_mapper()
            sn = mapper.get_snomed_for_loinc(concept.loinc_code)
            logger.info(
                "Resolved biomarker %r via LOINC %s → %d itemids: %s",
                concept.name, concept.loinc_code, len(itemids), itemids,
            )
            return BiomarkerResolution(
                itemids=itemids,
                names=names,
                loinc_code=concept.loinc_code,
                snomed_code=sn.code if sn is not None else None,
                fallback_reason=None,
            )

        # Local LOINC index missed. Before falling back to LIKE, try
        # OMOPHub-backed mimic_itemid_search against MIMIC's d_labitems
        # to find labevents itemids matching the analyte name. Restricts
        # to ``table='labevents'`` so we never pull chartevents (vital
        # signs) into a lab query.
        if self._enable_mcp_grounding:
            mcp_itemids = self._ground_via_mimic_itemid_search(concept.name)
            if mcp_itemids is not None:
                mapper = self._get_snomed_mapper()
                sn = mapper.get_snomed_for_loinc(concept.loinc_code)
                logger.info(
                    "Resolved biomarker %r via mimic_itemid_search → %d "
                    "labevents itemids: %s",
                    concept.name, len(mcp_itemids), mcp_itemids,
                )
                return BiomarkerResolution(
                    itemids=mcp_itemids,
                    names=names,
                    loinc_code=concept.loinc_code,
                    snomed_code=sn.code if sn is not None else None,
                    fallback_reason=None,
                )

        # No itemids — distinguish "LOINC unknown" (case 2) from "LOINC
        # known but no MIMIC coverage" (case 3) so the warning text is
        # diagnosable.
        mapper = self._get_snomed_mapper()
        sn = mapper.get_snomed_for_loinc(concept.loinc_code)
        if sn is None:
            return BiomarkerResolution(
                itemids=None,
                names=names,
                loinc_code=concept.loinc_code,
                snomed_code=None,
                fallback_reason=(
                    f"LOINC {concept.loinc_code!r} not found in mapping table "
                    "— falling back to label match; result may pool "
                    "unit-incompatible variants."
                ),
            )
        return BiomarkerResolution(
            itemids=None,
            names=names,
            loinc_code=concept.loinc_code,
            snomed_code=sn.code,
            fallback_reason=(
                f"LOINC {concept.loinc_code!r} → SNOMED {sn.code} has no "
                "MIMIC labitem coverage — falling back to label match; "
                "result may pool unit-incompatible variants."
            ),
        )

    def _ground_via_mimic_itemid_search(
        self, name: str,
    ) -> list[int] | None:
        """Query OMOPHub-backed mimic_itemid_search for labevents itemids.

        Returns a sorted list of itemids whose ``table`` is
        ``'labevents'`` (chartevents results are intentionally dropped —
        they represent vital-sign / chart entries that don't belong in
        a biomarker query). Returns ``None`` when:
          - The MCP returns ``unavailable`` (caught from sentinel raise).
          - No labevents results in the response (chartevents-only or
            empty).

        ``None`` signals the caller to fall through to the existing
        loud-fallback path.
        """
        try:
            cached = _cached_mimic_itemid_search(name.lower())
        except LookupError as exc:
            logger.info(
                "mimic_itemid_search unavailable for %r: %s — falling back",
                name, exc,
            )
            return None
        labevents_ids = sorted(
            {itemid for itemid, _label, table in cached if table == "labevents"}
        )
        if not labevents_ids:
            return None
        return labevents_ids

    # -- ICD-grounded diagnosis resolution ---------------------------------

    def resolve_diagnosis(
        self, concept: ClinicalConcept,
    ) -> DiagnosisResolution:
        """Resolve a diagnosis concept to ICD codes for parallel-OR SQL emission.

        Resolution order:
          1. If ``concept.icd_codes`` is pre-populated (LLM-grounded or
             test-supplied), use them directly. **No MCP call.**
          2. If ``enable_mcp_grounding`` is False (default for tests),
             return silent fallback so the SQL emitter falls back to LIKE
             with no user-visible warning.
          3. (Inc 3) Otherwise, call ``_ground_via_icd_autocode`` to query
             OMOPHub's ``icd_autocode`` MCP tool with the concept name.

        ``names`` is always populated via the existing ``resolve(...)``
        category-expansion path; SQL uses it as the parallel-OR LIKE
        clause that catches ICD-9 admissions whose codes aren't in
        OMOPHub's ICD10CM-only coverage.
        """
        names = self.resolve(concept)  # always populated; LIKE-fallback names

        if concept.icd_codes:
            return DiagnosisResolution(
                icd_codes=list(concept.icd_codes),
                names=names,
                fallback_reason=None,
                confidence_floor=None,
            )

        if not self._enable_mcp_grounding:
            return DiagnosisResolution(
                icd_codes=None,
                names=names,
                fallback_reason=None,
                confidence_floor=None,
            )

        return self._ground_via_icd_autocode(concept.name, names)

    def _ground_via_icd_autocode(
        self, name: str, names: list[str],
    ) -> DiagnosisResolution:
        """Ground a diagnosis name via OMOPHub's icd_autocode MCP tool.

        Returns a grounded ``DiagnosisResolution`` when the MCP returns at
        least one above-threshold candidate. Returns a loud-fallback
        resolution (with user-visible ``fallback_reason``) when the MCP is
        unavailable or all candidates are below the confidence threshold.
        """
        try:
            cached = _cached_icd_autocode(name.lower(), "10")
        except LookupError as exc:
            logger.info(
                "icd_autocode unavailable for %r: %s — falling back to LIKE",
                name, exc,
            )
            return DiagnosisResolution(
                icd_codes=None,
                names=names,
                fallback_reason=(
                    f"ICD autocoding for {name!r} unavailable "
                    f"({exc}); falling back to title LIKE — result may "
                    "pool unrelated diagnoses."
                ),
                confidence_floor=None,
            )

        # Filter by confidence threshold; keep None-confidence entries.
        accepted: list[tuple[str, float | None]] = []
        max_seen: float | None = None
        for code, conf in cached:
            if conf is None:
                accepted.append((code, conf))
                continue
            if max_seen is None or conf > max_seen:
                max_seen = conf
            if conf >= _ICD_CONFIDENCE_THRESHOLD:
                accepted.append((code, conf))

        if not accepted:
            return DiagnosisResolution(
                icd_codes=None,
                names=names,
                fallback_reason=(
                    f"ICD autocoding for {name!r} returned only "
                    f"low-confidence candidates "
                    f"(max={max_seen if max_seen is not None else 'N/A'}); "
                    "falling back to title LIKE — result may pool "
                    "unrelated diagnoses."
                ),
                confidence_floor=max_seen,
            )

        codes = [c for c, _ in accepted]
        # Confidence floor: minimum NON-None confidence among accepted.
        # All-None case → None.
        confs = [c for _, c in accepted if c is not None]
        floor = min(confs) if confs else None
        logger.info(
            "Grounded diagnosis %r via icd_autocode → %d codes (floor=%s): %s",
            name, len(codes), floor, codes,
        )
        return DiagnosisResolution(
            icd_codes=codes,
            names=names,
            fallback_reason=None,
            confidence_floor=floor,
        )
