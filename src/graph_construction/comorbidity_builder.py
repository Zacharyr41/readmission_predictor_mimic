"""Derive Charlson comorbidity flags from a patient's diagnoses (plan I-B).

Wires the previously-dead ``write_comorbidity`` writer into graph construction.
A patient's ICD-10 diagnoses are prefix-matched against the Quan et al. 2005
Charlson mapping (``data/mappings/icd10_to_charlson.json``) — the same
ontology-grounded mapping the WLST feature builder uses, so there is no
hand-curated synonym list — and each *present* category becomes a
``mimic:Comorbidity`` node linked to the patient and grounded to SNOMED-CT.

Only present comorbidities are emitted (absence is the lack of a node), which
matches the cohort path's presence-feature semantics. ICD-9 codes are ignored:
the mapping is ICD-10, and matching a dotless ICD-9 code against an ICD-10
prefix would be a coincidental false positive.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from rdflib import Graph, URIRef

from src.graph_construction.event_writers import write_comorbidity

logger = logging.getLogger(__name__)

_DEFAULT_MAPPINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "mappings"


@lru_cache(maxsize=4)
def _load_charlson_categories(mappings_dir: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Load ``(category, icd10_prefixes)`` pairs, dropping non-category metadata.

    Returns a tuple (hashable, cache-friendly) in mapping order so derivation is
    deterministic. Entries without ``icd10_prefixes`` (e.g. ``_metadata``) are
    skipped.
    """
    path = Path(mappings_dir) / "icd10_to_charlson.json"
    if not path.exists():
        logger.warning("Charlson mapping not found: %s", path)
        return ()
    with open(path) as fh:
        raw = json.load(fh)
    return tuple(
        (cat, tuple(info["icd10_prefixes"]))
        for cat, info in raw.items()
        if isinstance(info, dict) and info.get("icd10_prefixes")
    )


def derive_charlson_categories(
    diagnoses: list[dict], *, mappings_dir: Path | None = None,
) -> list[str]:
    """Charlson category names present in a patient's ICD-10 diagnoses.

    Each ICD-10 code is prefix-matched against the mapping; ICD-9 codes are
    ignored. The decimal point is stripped before matching so both dotted
    (``I50.9``) and dotless (``I509``, as MIMIC stores them) forms work.
    Categories are returned in mapping order, de-duplicated.
    """
    categories = _load_charlson_categories(mappings_dir or _DEFAULT_MAPPINGS_DIR)
    if not categories:
        return []
    codes = [
        str(dx.get("icd_code", "")).replace(".", "").strip()
        for dx in diagnoses
        if dx.get("icd_version") == 10 and dx.get("icd_code")
    ]
    codes = [c for c in codes if c]
    present: list[str] = []
    for category, prefixes in categories:
        if any(code.startswith(prefixes) for code in codes):
            present.append(category)
    return present


def write_patient_comorbidities(
    graph: Graph,
    diagnoses: list[dict],
    patient_uri: URIRef,
    subject_id: int,
    *,
    snomed_mapper=None,
    mappings_dir: Path | None = None,
) -> list[URIRef]:
    """Derive Charlson comorbidities from ``diagnoses`` and write one
    ``mimic:Comorbidity`` node per present category, linked to the patient.

    Returns the list of created comorbidity URIs (possibly empty).
    """
    uris: list[URIRef] = []
    for category in derive_charlson_categories(diagnoses, mappings_dir=mappings_dir):
        uri = write_comorbidity(
            graph,
            {"subject_id": subject_id, "name": category, "value": True},
            patient_uri,
            snomed_mapper=snomed_mapper,
        )
        uris.append(uri)
    return uris
