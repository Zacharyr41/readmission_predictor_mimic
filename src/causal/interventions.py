"""Ontology-grounded intervention resolution (Phase 8b).

The user's 2026-04-17 correctness-first decision (see
``memory/feedback_correctness_no_curation.md``) forbids hand-maintained
synonym lists for intervention-predicate derivation. Every
``InterventionSpec`` (defined in ``src.conversational.models``) carries
exactly one ontology code; this module is responsible for turning that
code into an executable SQL ``EXISTS`` predicate over the MIMIC tables,
and for producing an auditable provenance trail.

Resolution paths
----------------

* ``snomed_concept_id`` (PRIMARY for drug interventions per user
  decision): expand the concept via ``SnomedHierarchy.get_descendants``
  when the hierarchy JSON is present; reverse-map each (including the
  root) against the existing ``data/mappings/drug_to_snomed.json`` to
  obtain MIMIC-known drug names; emit a predicate over
  ``prescriptions.drug``. Degrades gracefully when
  ``SnomedHierarchy`` has no on-disk file — the root concept alone is
  used, and provenance records the degraded state.
* ``rxnorm_ingredient``: first check whether
  ``drug_to_snomed.json`` already contains this RxCUI (the MIMIC-
  specific drug registry already carries RxCUI per entry); if yes,
  resolve purely locally. If not, call ``RxNavClient.get_related_rxcuis``
  and match the returned names against the normalised keys in
  ``drug_to_snomed.json``. This keeps network I/O out of the common
  path but still honours rxnav as the canonical source when needed.
* ``icd10pcs_code``: direct prefix match against ``procedures_icd``
  (``icd_version = 10``). No ontology lookup necessary; ICD-10-PCS
  codes are already structured.
* ``loinc_code``: resolve LOINC → SNOMED via
  ``SnomedMapper.get_snomed_for_loinc``; reverse-map the SCTID to
  MIMIC ``labitem`` itemids via ``data/mappings/labitem_to_snomed.json``
  and emit a predicate over ``labevents``.

Control arms
------------

``is_control=True`` negates the resolved predicate (``NOT EXISTS(…)``).
For single-code interventions this is clean — "received tPA" vs "did
not receive tPA" is exact mutual exclusivity on the alteplase RxCUI.
For class-level controls ("no anticoagulant"), the caller must pick a
SNOMED class concept whose descendants cover the full class, and
depends on ``SnomedHierarchy`` being loaded. This is a known
limitation that 8h's diagnostics surface in the assumption ledger.

SQL predicate contract
----------------------

Each ``ResolvedIntervention`` exposes an ``sql_exists_fragment`` that
is a valid DuckDB predicate referencing an outer ``a`` alias bound to
``admissions``. Callers compose it into

    SELECT … FROM admissions a WHERE a.hadm_id IN (?) AND <fragment>

BigQuery parameter binding (``@p0`` style) is not threaded through in
8b — resolver tests use DuckDB only. 8d adds a backend adapter that
translates param styles transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.causal._rxnav import RxNavClient, RxNavError
from src.conversational.models import InterventionSpec
from src.graph_construction.terminology.snomed_hierarchy import SnomedHierarchy
from src.graph_construction.terminology.snomed_mapper import SnomedMapper

logger = logging.getLogger(__name__)


_RESOLVER_VERSION = "8b-2026-04-17"


class InterventionResolutionError(RuntimeError):
    """Raised when an ``InterventionSpec`` cannot be resolved to a
    non-empty MIMIC predicate. Fails loudly per the correctness-first
    rule — a silent empty-predicate match would produce an empty
    cohort arm and misleadingly valid-looking downstream estimates."""


@dataclass(frozen=True)
class ResolvedIntervention:
    """Executable representation of a validated ``InterventionSpec``.

    Instances are frozen so the treatment-assignment pipeline can
    assume the predicate and provenance won't mutate mid-run.

    Attributes:
        label: carried through from the input spec; used as the
            ``mu_c`` key in the downstream ``CausalEffectResult``.
        is_control: if True, the predicate is negated at assignment
            time (``NOT (<fragment>)``).
        sql_exists_fragment: DuckDB-compatible SQL EXISTS clause; must
            reference an outer ``a`` alias bound to ``admissions``.
        params: positional parameters for the fragment.
        provenance: auditable resolution trace — ontology source,
            version, descendants expanded, MIMIC codes matched.
            Every field a reviewer would need to reproduce the
            treatment-assignment step.
    """

    label: str
    is_control: bool
    sql_exists_fragment: str
    params: tuple[Any, ...]
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class InterventionResolver:
    """Resolve ``InterventionSpec`` → ``ResolvedIntervention``.

    One resolver instance can be reused across many interventions; it
    caches the ``SnomedMapper`` + ``RxNavClient`` + reverse-index
    lookups internally.
    """

    def __init__(
        self,
        mappings_dir: Path | None = None,
        *,
        hierarchy: SnomedHierarchy | None = None,
        rxnav_client: RxNavClient | None = None,
    ) -> None:
        if mappings_dir is None:
            repo_root = Path(__file__).parent.parent.parent
            mappings_dir = repo_root / "data" / "mappings"
        self._mappings_dir = mappings_dir
        self._mapper = SnomedMapper(mappings_dir)
        self._hierarchy = hierarchy
        self._rxnav = rxnav_client  # Lazy: only constructed on a true rxnav miss
        # Lazy reverse indices keyed by attribute in drug_to_snomed:
        #   {snomed_code: [normalised_drug_name, ...]}
        #   {rxcui:       [normalised_drug_name, ...]}
        #   {labitem_snomed_code: [itemid, ...]}
        self._snomed_to_drug: dict[str, list[str]] | None = None
        self._rxcui_to_drug: dict[str, list[str]] | None = None
        self._snomed_to_labitem: dict[str, list[int]] | None = None

    # ---- reverse-index builders ----

    def _build_drug_reverse_indices(self) -> None:
        if self._snomed_to_drug is not None:
            return
        snomed_to_drug: dict[str, list[str]] = {}
        rxcui_to_drug: dict[str, list[str]] = {}
        for drug_name, entry in self._mapper._drug_map.items():
            if not isinstance(entry, dict):
                continue
            sctid = entry.get("snomed_code")
            rxcui = entry.get("rxcui")
            if sctid:
                snomed_to_drug.setdefault(str(sctid), []).append(drug_name)
            if rxcui:
                rxcui_to_drug.setdefault(str(rxcui), []).append(drug_name)
        self._snomed_to_drug = snomed_to_drug
        self._rxcui_to_drug = rxcui_to_drug

    def _build_labitem_reverse_index(self) -> None:
        if self._snomed_to_labitem is not None:
            return
        reverse: dict[str, list[int]] = {}
        for itemid, entry in self._mapper._labitem_map.items():
            if not isinstance(entry, dict):
                continue
            sctid = entry.get("snomed_code")
            if sctid:
                try:
                    reverse.setdefault(str(sctid), []).append(int(itemid))
                except ValueError:
                    continue
        self._snomed_to_labitem = reverse

    # ---- dispatch ----

    def resolve(self, spec: InterventionSpec) -> ResolvedIntervention:
        """Dispatch on which ontology code the spec carries."""
        if spec.snomed_concept_id:
            return self._resolve_snomed(spec)
        if spec.rxnorm_ingredient:
            return self._resolve_rxnorm(spec)
        if spec.icd10pcs_code:
            return self._resolve_icd10pcs(spec)
        if spec.loinc_code:
            return self._resolve_loinc(spec)
        # The InterventionSpec validator prevents zero-code specs from
        # being constructed, so reaching here means the schema drifted.
        raise InterventionResolutionError(
            f"intervention {spec.label!r} carries no ontology code; "
            "schema validation should have rejected this upstream"
        )

    # ---- per-ontology resolution ----

    def _resolve_snomed(self, spec: InterventionSpec) -> ResolvedIntervention:
        target = spec.snomed_concept_id
        assert target is not None  # dispatch guarantees this

        # Expand via hierarchy if available. When the JSON file isn't
        # present we still proceed with just the target concept — but
        # we record the degraded state in provenance so a reviewer can
        # tell whether the cohort used class-level expansion or just a
        # single SCTID.
        targets: set[str] = {target}
        hierarchy_used = False
        descendants_added = 0
        if self._hierarchy is not None:
            descendants = self._hierarchy.get_descendants(target)
            if descendants:
                targets.update(descendants)
                descendants_added = len(descendants)
                hierarchy_used = True

        self._build_drug_reverse_indices()
        assert self._snomed_to_drug is not None
        drug_names = sorted({
            name
            for sctid in targets
            for name in self._snomed_to_drug.get(sctid, [])
        })

        if not drug_names:
            raise InterventionResolutionError(
                f"SNOMED concept {target!r} (and its descendants, n="
                f"{descendants_added}) have no entries in "
                f"data/mappings/drug_to_snomed.json. Either pick a more "
                "specific concept whose descendants are MIMIC-registered, "
                "or extend the drug_to_snomed registry."
            )

        provenance = {
            "ontology": "SNOMED-CT",
            "target_concept_id": target,
            "hierarchy_loaded": hierarchy_used,
            "descendants_expanded": descendants_added,
            "mimic_drug_names_matched": len(drug_names),
            "matched_drug_names": drug_names,
            "resolver_version": _RESOLVER_VERSION,
            **(spec.provenance or {}),
        }
        return ResolvedIntervention(
            label=spec.label,
            is_control=spec.is_control,
            sql_exists_fragment=_drug_exists_fragment(len(drug_names)),
            params=tuple(drug_names),
            provenance=provenance,
        )

    def _resolve_rxnorm(self, spec: InterventionSpec) -> ResolvedIntervention:
        rxcui = spec.rxnorm_ingredient
        assert rxcui is not None

        self._build_drug_reverse_indices()
        assert self._rxcui_to_drug is not None

        # Fast path: local drug registry already tags this RxCUI.
        local_names = sorted(self._rxcui_to_drug.get(str(rxcui), []))

        # Slow path: hit rxnav, fan out the ingredient to its descendant
        # clinical-drug products, then match product names against the
        # MIMIC-registered drug keys (normalised lowercase).
        descendants: list[dict[str, str]] = []
        rxnav_names: list[str] = []
        used_rxnav = False
        rxnav_error: str | None = None
        if not local_names:
            client = self._rxnav or RxNavClient()
            try:
                descendants = client.get_related_rxcuis(str(rxcui))
                used_rxnav = True
            except RxNavError as e:
                rxnav_error = str(e)
                descendants = []
            if descendants:
                registered_keys = set(self._mapper._drug_map.keys())
                rxnav_names = sorted({
                    SnomedMapper._normalize_drug(d.get("name", ""))
                    for d in descendants
                } & registered_keys)

        drug_names = local_names or rxnav_names
        if not drug_names:
            # Correctness-first: refuse to silently produce an empty
            # cohort arm. The caller must either (a) register the
            # RxCUI in drug_to_snomed.json with its MIMIC drug name(s),
            # or (b) use the SNOMED concept path instead.
            trace = (
                f"no local registry entry for rxcui={rxcui}; "
                f"rxnav returned {len(descendants)} descendants"
            )
            if rxnav_error is not None:
                trace += f"; rxnav error: {rxnav_error}"
            raise InterventionResolutionError(
                f"RxCUI {rxcui!r} could not be resolved to any MIMIC drug name. "
                f"{trace}. See memory/feedback_correctness_no_curation.md."
            )

        provenance = {
            "ontology": "RxNorm",
            "target_rxcui": rxcui,
            "resolved_via": "local drug_to_snomed.json" if local_names else "rxnav /related.json",
            "rxnav_used": used_rxnav,
            "rxnav_descendants_returned": len(descendants),
            "mimic_drug_names_matched": len(drug_names),
            "matched_drug_names": drug_names,
            "resolver_version": _RESOLVER_VERSION,
            **(spec.provenance or {}),
        }
        return ResolvedIntervention(
            label=spec.label,
            is_control=spec.is_control,
            sql_exists_fragment=_drug_exists_fragment(len(drug_names)),
            params=tuple(drug_names),
            provenance=provenance,
        )

    def _resolve_icd10pcs(self, spec: InterventionSpec) -> ResolvedIntervention:
        raw_code = spec.icd10pcs_code
        assert raw_code is not None
        # MIMIC stores ICD codes undotted, uppercased.
        code = raw_code.replace(".", "").strip().upper()
        if not code:
            raise InterventionResolutionError(
                f"intervention {spec.label!r}: icd10pcs_code is empty after normalisation"
            )
        # ICD-10-PCS codes are hierarchical — matching both the exact
        # code and its prefix covers the "root + all descendant codes"
        # case without needing an external ontology table.
        fragment = (
            "EXISTS (SELECT 1 FROM procedures_icd pi "
            "WHERE pi.hadm_id = a.hadm_id "
            "AND pi.icd_version = 10 "
            "AND (pi.icd_code = ? OR pi.icd_code LIKE ?))"
        )
        params = (code, f"{code}%")
        provenance = {
            "ontology": "ICD-10-PCS",
            "target_code": code,
            "match_policy": "exact OR prefix",
            "resolver_version": _RESOLVER_VERSION,
            **(spec.provenance or {}),
        }
        return ResolvedIntervention(
            label=spec.label,
            is_control=spec.is_control,
            sql_exists_fragment=fragment,
            params=params,
            provenance=provenance,
        )

    def _resolve_loinc(self, spec: InterventionSpec) -> ResolvedIntervention:
        loinc = spec.loinc_code
        assert loinc is not None

        concept = self._mapper.get_snomed_for_loinc(loinc)
        if concept is None:
            raise InterventionResolutionError(
                f"LOINC {loinc!r} has no SNOMED mapping in "
                "data/mappings/loinc_to_snomed.json (or its cache)."
            )
        self._build_labitem_reverse_index()
        assert self._snomed_to_labitem is not None
        itemids = sorted(self._snomed_to_labitem.get(concept.code, []))
        if not itemids:
            raise InterventionResolutionError(
                f"LOINC {loinc!r} (SNOMED {concept.code}) has no mapped "
                "MIMIC lab itemids — labitem_to_snomed.json lacks an entry "
                "for any itemid under this concept."
            )
        placeholders = ",".join(["?"] * len(itemids))
        fragment = (
            f"EXISTS (SELECT 1 FROM labevents le WHERE le.hadm_id = a.hadm_id "
            f"AND le.itemid IN ({placeholders}))"
        )
        provenance = {
            "ontology": "LOINC",
            "target_loinc": loinc,
            "resolved_via_snomed": concept.code,
            "mimic_labitems_matched": len(itemids),
            "matched_itemids": itemids,
            "resolver_version": _RESOLVER_VERSION,
            **(spec.provenance or {}),
        }
        return ResolvedIntervention(
            label=spec.label,
            is_control=spec.is_control,
            sql_exists_fragment=fragment,
            params=tuple(itemids),
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# SQL fragment helpers
# ---------------------------------------------------------------------------


def _drug_exists_fragment(n_names: int) -> str:
    """EXISTS predicate over prescriptions.drug, case-insensitive.

    MIMIC's ``prescriptions.drug`` is a free-text clinical name.
    ``drug_to_snomed.json`` keys are lowercase-normalised; we lower
    the column at match time to cover case drift.
    """
    placeholders = ",".join(["?"] * n_names)
    return (
        "EXISTS (SELECT 1 FROM prescriptions p WHERE p.hadm_id = a.hadm_id "
        f"AND LOWER(p.drug) IN ({placeholders}))"
    )
