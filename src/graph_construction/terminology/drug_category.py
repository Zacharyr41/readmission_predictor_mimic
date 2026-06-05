"""Resolve a MIMIC drug name to canonical drug *categories* (plan I-A).

A ``prescriptions.drug`` string ("Norepinephrine 4 mg/250 mL") is mapped to
one or more canonical categories ("vasopressors") so the cohort feature
extractor (plan III-A) can select all administrations of a class for a stay
and compute a dose trend ("escalating pressors").

The category vocabulary is the ontology-grounded set in
``data/mappings/category_to_snomed.json`` — each category carries a SNOMED
root, and only those tagged ``kind: "drug"`` participate (so a human-albumin
infusion is never tagged the lab category "liver function tests"). Membership
comes from the curated ``members`` bootstrap. Brand / combination names that
miss the bootstrap are grounded through an RxNorm-ingredient escape hatch:
``rxnorm_lookup`` → RxCUI → canonical ingredient (RxNav) → re-match. That
keeps the *emitted vocabulary canonical* (no ATC-name drift) while resolving
the long tail without hand-curated synonym lists. The escape hatch is
injectable and off by default, so unit tests are deterministic and offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.graph_construction.terminology.snomed_mapper import SnomedMapper

logger = logging.getLogger(__name__)

_DEFAULT_MAPPINGS_DIR = Path(__file__).resolve().parents[3] / "data" / "mappings"


@dataclass(frozen=True)
class DrugCategory:
    """A canonical drug category a prescription belongs to.

    ``snomed_code`` is the category's SNOMED root (the ontology grounding);
    ``source`` records how membership was established (``"curated"`` bootstrap
    or ``"rxnorm"`` ingredient escape hatch).
    """

    name: str
    snomed_code: str | None
    source: str


class DrugCategoryResolver:
    """Resolve drug names to canonical drug categories.

    Parameters
    ----------
    mappings_dir:
        Directory holding ``category_to_snomed.json``. Defaults to
        ``data/mappings``.
    enable_mcp:
        When True, a bootstrap miss falls back to the RxNorm-ingredient
        escape hatch. Default False keeps resolution offline and
        deterministic (the unit-test default).
    rxnorm_lookup:
        Injectable ``drug_name -> envelope`` callable (the OMOPHub MCP tool by
        default, resolved lazily). Tests pass a fake to avoid network.
    ingredient_resolver:
        Injectable ``rxcui -> ingredient_name | None`` callable (RxNav by
        default, resolved lazily).
    """

    def __init__(
        self,
        mappings_dir: Path | None = None,
        *,
        enable_mcp: bool = False,
        rxnorm_lookup: Callable[..., dict] | None = None,
        ingredient_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        self._mappings_dir = Path(mappings_dir) if mappings_dir else _DEFAULT_MAPPINGS_DIR
        self._enable_mcp = enable_mcp
        self._rxnorm_lookup = rxnorm_lookup
        self._ingredient_resolver = ingredient_resolver
        self._rxnav = None  # lazily constructed RxNavClient for the default path
        self._cache: dict[str, list[DrugCategory]] = {}
        # (category_name, snomed_code, [members_lowercased]) in mapping order.
        self._drug_categories = self._load_drug_categories()

    # -- loading --------------------------------------------------------------

    def _load_drug_categories(self) -> list[tuple[str, str | None, list[str]]]:
        path = self._mappings_dir / "category_to_snomed.json"
        if not path.exists():
            logger.warning("Category map not found: %s", path)
            return []
        with open(path) as fh:
            raw = json.load(fh)
        out: list[tuple[str, str | None, list[str]]] = []
        for name, info in raw.items():
            if name == "_metadata" or not isinstance(info, dict):
                continue
            if info.get("kind") != "drug":
                continue
            members = [str(m).lower().strip() for m in info.get("members", [])]
            out.append((name, info.get("snomed_code"), members))
        return out

    # -- matching -------------------------------------------------------------

    @staticmethod
    def _matches(member: str, normalized: str, tokens: set[str]) -> bool:
        """Word-boundary membership: a single-word member must be a whole token
        (so "ph" never matches "phenylephrine"); a multi-word member matches as
        a substring.
        """
        if " " in member:
            return member in normalized
        return member in tokens

    def _match(self, drug_name: str, source: str) -> list[DrugCategory]:
        normalized = SnomedMapper._normalize_drug(drug_name)
        tokens = set(normalized.split())
        out: list[DrugCategory] = []
        for name, sctid, members in self._drug_categories:
            if any(self._matches(m, normalized, tokens) for m in members):
                out.append(DrugCategory(name, sctid, source))
        return out

    # -- resolution -----------------------------------------------------------

    def resolve(self, drug_name: str) -> list[DrugCategory]:
        """Return the canonical drug categories ``drug_name`` belongs to.

        Memoized per normalized drug name: a cohort has many administrations
        per distinct drug, and the escape hatch must not re-call the network.
        """
        if not drug_name:
            return []
        key = SnomedMapper._normalize_drug(drug_name)
        if key in self._cache:
            return self._cache[key]

        cats = self._match(drug_name, "curated")
        if not cats and self._enable_mcp:
            cats = self._resolve_via_rxnorm(drug_name)

        self._cache[key] = cats
        return cats

    def _resolve_via_rxnorm(self, drug_name: str) -> list[DrugCategory]:
        """Escape hatch: resolve the canonical ingredient via RxNorm and
        re-match the bootstrap, so brand / combination names land in the same
        canonical category vocabulary.
        """
        lookup = self._rxnorm_lookup or _default_rxnorm_lookup
        try:
            envelope = lookup(drug_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("rxnorm_lookup failed for %r: %s", drug_name, exc)
            return []
        if not isinstance(envelope, dict) or envelope.get("status") != "ok":
            return []
        for rec in envelope.get("results", []) or []:
            if not isinstance(rec, dict):
                continue
            candidate = self._ingredient_name(rec.get("rxcui")) or rec.get("name")
            if not candidate:
                continue
            cats = self._match(str(candidate), "rxnorm")
            if cats:
                return cats
        return []

    def _ingredient_name(self, rxcui) -> str | None:
        if not rxcui:
            return None
        resolver = self._ingredient_resolver or self._default_ingredient
        try:
            return resolver(str(rxcui))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ingredient lookup failed for rxcui=%s: %s", rxcui, exc)
            return None

    def _default_ingredient(self, rxcui: str) -> str | None:
        if self._rxnav is None:
            from src.causal._rxnav import RxNavClient

            self._rxnav = RxNavClient()
        return self._rxnav.get_ingredient_name(rxcui)


def _default_rxnorm_lookup(drug_name: str) -> dict:
    from src.conversational.health_evidence.tools import rxnorm_lookup

    return rxnorm_lookup(drug_name)
