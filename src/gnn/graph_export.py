"""RDF knowledge graph to PyG HeteroData conversion.

Converts the full RDF knowledge graph (built from MIMIC-IV clinical data) into
a PyTorch Geometric ``HeteroData`` object preserving all 9 node types and their
relationships, ready for GNN training.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

import pandas as pd
import rdflib
import torch
from rdflib import URIRef
from torch_geometric.data import HeteroData

from src.graph_construction.ontology import MIMIC_NS, SNOMED_NS, TIME_NS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NODE_TYPES = [
    "patient", "admission", "icu_stay", "icu_day",
    "drug", "diagnosis", "lab", "vital", "microbiology",
]
EMBEDDING_DIM = 768

# Gender encoding: Male → 1.0, Female → 0.0
GENDER_ENCODING = {"M": 1.0, "F": 0.0}

# Map clinical event RDF class → (node_type, fallback_id_predicate, label_predicate)
_CLINICAL_TYPE_CONFIG = {
    "lab": (MIMIC_NS.BioMarkerEvent, MIMIC_NS.hasItemId, MIMIC_NS.hasBiomarkerType),
    "vital": (MIMIC_NS.ClinicalSignEvent, MIMIC_NS.hasItemId, MIMIC_NS.hasClinicalSignName),
    "drug": (MIMIC_NS.PrescriptionEvent, MIMIC_NS.hasDrugName, MIMIC_NS.hasDrugName),
    "microbiology": (MIMIC_NS.MicrobiologyEvent, MIMIC_NS.hasOrganismName, MIMIC_NS.hasOrganismName),
    "diagnosis": (MIMIC_NS.DiagnosisEvent, MIMIC_NS.hasIcdCode, MIMIC_NS.hasLongTitle),
}


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp parsing (mirrors allen_relations._parse_timestamp)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_timestamp(ts_str: str) -> datetime:
    """Parse an ISO timestamp string, stripping trailing 'Z' if present."""
    s = str(ts_str)
    if s.endswith("Z"):
        s = s[:-1]
    return datetime.fromisoformat(s)


# ──────────────────────────────────────────────────────────────────────────────
# A. Node ID Mapping
# ──────────────────────────────────────────────────────────────────────────────

_STRUCTURAL_TYPES = {
    "patient": MIMIC_NS.Patient,
    "admission": MIMIC_NS.HospitalAdmission,
    "icu_stay": MIMIC_NS.ICUStay,
    "icu_day": MIMIC_NS.ICUDay,
}


def _build_structural_node_maps(
    g: rdflib.Graph,
) -> dict[str, dict[URIRef, int]]:
    """Build deterministic URI→int maps for structural node types."""
    maps: dict[str, dict[URIRef, int]] = {}
    for ntype, rdf_class in _STRUCTURAL_TYPES.items():
        query = f"""
        SELECT ?node WHERE {{ ?node rdf:type <{rdf_class}> }}
        """
        uris = sorted(str(row[0]) for row in g.query(query))
        maps[ntype] = {URIRef(uri): idx for idx, uri in enumerate(uris)}
    return maps


class ClinicalNodeMaps(NamedTuple):
    """Return type for :func:`_build_clinical_node_maps`.

    Attributes:
        concept_id_map: Per-type mapping from concept key (SNOMED URI string
            or ``"unmapped:{type}:{fallback}"``) to contiguous integer ID.
        event_to_concept: Per-type mapping from RDF event URI to its
            concept key (used to resolve edges through deduplication).
        concept_labels: Per-type mapping from concept key to a human-readable
            label string (used for embedding unmapped terms).
    """

    concept_id_map: dict[str, dict[str, int]]
    event_to_concept: dict[str, dict[URIRef, str]]
    concept_labels: dict[str, dict[str, str]]


def _build_clinical_node_maps(
    g: rdflib.Graph,
) -> ClinicalNodeMaps:
    """Build deduplicated concept-level maps for clinical event types.

    Events sharing the same SNOMED concept URI are collapsed into a single
    node.  Events without a SNOMED mapping get a synthetic key of the form
    ``"unmapped:{type}:{fallback_id}"``.
    """
    concept_id_map: dict[str, dict[str, int]] = {}
    event_to_concept: dict[str, dict[URIRef, str]] = {}
    concept_labels: dict[str, dict[str, str]] = {}

    for ntype, (rdf_class, fallback_pred, label_pred) in _CLINICAL_TYPE_CONFIG.items():
        query = f"""
        SELECT ?event ?snomed ?fallback ?label WHERE {{
            ?event rdf:type <{rdf_class}> .
            OPTIONAL {{ ?event <{MIMIC_NS.hasSnomedConcept}> ?snomed }}
            OPTIONAL {{ ?event <{fallback_pred}> ?fallback }}
            OPTIONAL {{ ?event <{label_pred}> ?label }}
        }}
        """
        concepts: dict[str, str] = {}  # concept_key → label
        evt_map: dict[URIRef, str] = {}

        for row in g.query(query):
            event_uri = row[0]
            snomed_uri = row[1]
            fallback = row[2]
            label = row[3]

            if snomed_uri is not None:
                concept_key = str(snomed_uri)
            else:
                concept_key = f"unmapped:{ntype}:{fallback}"

            label_str = str(label) if label is not None else str(fallback) if fallback is not None else "unknown"

            evt_map[event_uri] = concept_key
            if concept_key not in concepts:
                concepts[concept_key] = label_str

        sorted_keys = sorted(concepts.keys())
        concept_id_map[ntype] = {k: i for i, k in enumerate(sorted_keys)}
        event_to_concept[ntype] = evt_map
        concept_labels[ntype] = {k: concepts[k] for k in sorted_keys}

    return ClinicalNodeMaps(concept_id_map, event_to_concept, concept_labels)


def _save_node_mappings(
    structural_maps: dict[str, dict[URIRef, int]],
    concept_maps: dict[str, dict[str, int]],
    path: Path,
) -> None:
    """Persist node mappings as JSON for reproducibility."""
    data = {
        "structural": {
            ntype: {str(uri): idx for uri, idx in m.items()}
            for ntype, m in structural_maps.items()
        },
        "clinical": concept_maps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved node mappings to %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# B. Node Features
# ──────────────────────────────────────────────────────────────────────────────

def _build_patient_features(
    g: rdflib.Graph, node_map: dict[URIRef, int]
) -> torch.Tensor:
    """Patient features: [age, gender_encoded] shape (N, 2)."""
    n = len(node_map)
    feats = torch.zeros(n, 2)

    query = """
    SELECT ?patient ?age ?gender WHERE {
        ?patient a mimic:Patient ;
                 mimic:hasAge ?age ;
                 mimic:hasGender ?gender .
    }
    """
    for row in g.query(query):
        uri = row[0]
        if uri in node_map:
            idx = node_map[uri]
            feats[idx, 0] = float(row[1])
            feats[idx, 1] = GENDER_ENCODING.get(str(row[2]), 0.0)
    return feats


def _build_admission_features(
    node_map: dict[URIRef, int],
    feature_matrix_path: Path,
) -> torch.Tensor:
    """Load parquet feature matrix aligned to admission node ordering."""
    df = pd.read_parquet(feature_matrix_path)

    drop_cols = {"hadm_id", "subject_id", "readmitted_30d", "readmitted_60d"}
    feat_cols = sorted(c for c in df.columns if c not in drop_cols)

    n = len(node_map)
    feat_dim = len(feat_cols)
    feats = torch.zeros(n, feat_dim)

    for _, row in df.iterrows():
        hadm_id = int(row["hadm_id"])
        uri = MIMIC_NS[f"HA-{hadm_id}"]
        if uri in node_map:
            idx = node_map[uri]
            feats[idx] = torch.tensor([float(row[c]) for c in feat_cols])

    return feats


def _build_icu_stay_features(
    g: rdflib.Graph, node_map: dict[URIRef, int]
) -> torch.Tensor:
    """ICU stay features: [duration_hours, num_icu_days] shape (N, 2)."""
    n = len(node_map)
    feats = torch.zeros(n, 2)

    query = """
    SELECT ?stay ?dur (COUNT(?day) AS ?ndays) WHERE {
        ?stay a mimic:ICUStay .
        ?stay time:hasDuration ?durNode .
        ?durNode time:numericDuration ?dur .
        OPTIONAL { ?stay mimic:hasICUDay ?day }
    }
    GROUP BY ?stay ?dur
    """
    for row in g.query(query):
        uri = row[0]
        if uri in node_map:
            idx = node_map[uri]
            feats[idx, 0] = float(row[1]) * 24.0  # days → hours
            feats[idx, 1] = float(row[2])
    return feats


def _build_icu_day_features(
    g: rdflib.Graph, node_map: dict[URIRef, int]
) -> torch.Tensor:
    """ICU day features: [day_number, num_events] shape (N, 2)."""
    n = len(node_map)
    feats = torch.zeros(n, 2)

    query = """
    SELECT ?day ?dayNum (COUNT(?evt) AS ?nevts) WHERE {
        ?day a mimic:ICUDay .
        ?day mimic:hasDayNumber ?dayNum .
        OPTIONAL { ?day mimic:hasICUDayEvent ?evt }
    }
    GROUP BY ?day ?dayNum
    """
    for row in g.query(query):
        uri = row[0]
        if uri in node_map:
            idx = node_map[uri]
            feats[idx, 0] = float(row[1])
            feats[idx, 1] = float(row[2])
    return feats


def _build_clinical_node_features(
    g: rdflib.Graph,
    concept_id_map: dict[str, int],
    event_to_concept: dict[URIRef, str],
    concept_labels: dict[str, str],
    concept_embeddings_path: Path,
    node_type: str,
    embed_unmapped_fn: Callable | None = None,
) -> torch.Tensor:
    """Build feature matrix for a clinical node type.

    SNOMED-mapped concepts get their pre-computed embedding.  Unmapped concepts
    get SapBERT embeddings via ``embed_unmapped_fn``.  Lab and vital nodes are
    augmented with population statistics (mean, std, min, max, abnormal_rate)
    yielding 773 dimensions; other types yield 768.
    """
    from src.gnn.embeddings import embed_unmapped_terms as _default_embed

    if embed_unmapped_fn is None:
        embed_unmapped_fn = _default_embed

    n = len(concept_id_map)
    if n == 0:
        return torch.empty(0, EMBEDDING_DIM)

    concept_embs = torch.load(concept_embeddings_path, weights_only=True)

    # Separate SNOMED-mapped vs unmapped concepts
    snomed_prefix = str(SNOMED_NS)
    embeddings: dict[str, torch.Tensor] = {}

    unmapped_labels: list[str] = []
    unmapped_keys: list[str] = []

    for key in concept_id_map:
        if key.startswith(snomed_prefix):
            code = key[len(snomed_prefix):]
            if code in concept_embs:
                embeddings[key] = concept_embs[code]
            else:
                # SNOMED code not in cache — treat as unmapped
                unmapped_keys.append(key)
                unmapped_labels.append(concept_labels.get(key, "unknown"))
        else:
            unmapped_keys.append(key)
            unmapped_labels.append(concept_labels.get(key, "unknown"))

    if unmapped_labels:
        unmapped_embs = embed_unmapped_fn(unmapped_labels)
        for key, label in zip(unmapped_keys, unmapped_labels):
            if label in unmapped_embs:
                embeddings[key] = unmapped_embs[label]
            else:
                embeddings[key] = torch.zeros(EMBEDDING_DIM)

    # Determine if we need population stats (lab/vital only)
    has_stats = node_type in ("lab", "vital")
    feat_dim = EMBEDDING_DIM + 5 if has_stats else EMBEDDING_DIM
    feats = torch.zeros(n, feat_dim)

    # Fill embedding columns
    for key, idx in concept_id_map.items():
        if key in embeddings:
            feats[idx, :EMBEDDING_DIM] = embeddings[key]

    # Compute population stats for lab/vital
    if has_stats:
        rdf_class = _CLINICAL_TYPE_CONFIG[node_type][0]

        has_ref = node_type == "lab"

        if has_ref:
            query = f"""
            SELECT ?event ?val ?lo ?hi WHERE {{
                ?event a <{rdf_class}> .
                ?event <{MIMIC_NS.hasValue}> ?val .
                OPTIONAL {{ ?event <{MIMIC_NS.hasRefRangeLower}> ?lo }}
                OPTIONAL {{ ?event <{MIMIC_NS.hasRefRangeUpper}> ?hi }}
            }}
            """
        else:
            query = f"""
            SELECT ?event ?val WHERE {{
                ?event a <{rdf_class}> .
                ?event <{MIMIC_NS.hasValue}> ?val .
            }}
            """

        # Collect values per concept
        concept_values: dict[str, list[float]] = {k: [] for k in concept_id_map}
        concept_abnormal: dict[str, list[bool]] = {k: [] for k in concept_id_map}

        for row in g.query(query):
            event_uri = row[0]
            if event_uri not in event_to_concept:
                continue
            ckey = event_to_concept[event_uri]
            val = float(row[1])
            concept_values[ckey].append(val)

            if has_ref:
                lo = float(row[2]) if row[2] is not None else None
                hi = float(row[3]) if row[3] is not None else None
                abnormal = False
                if lo is not None and val < lo:
                    abnormal = True
                elif hi is not None and val > hi:
                    abnormal = True
                concept_abnormal[ckey].append(abnormal)
            else:
                concept_abnormal[ckey].append(False)

        for key, idx in concept_id_map.items():
            vals = concept_values[key]
            if vals:
                t = torch.tensor(vals)
                feats[idx, EMBEDDING_DIM] = t.mean()
                feats[idx, EMBEDDING_DIM + 1] = t.std() if len(vals) > 1 else 0.0
                feats[idx, EMBEDDING_DIM + 2] = t.min()
                feats[idx, EMBEDDING_DIM + 3] = t.max()
                ab_list = concept_abnormal[key]
                feats[idx, EMBEDDING_DIM + 4] = sum(ab_list) / len(ab_list) if ab_list else 0.0

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# C. Edge Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _build_admission_start_lookup(
    g: rdflib.Graph,
) -> dict[URIRef, datetime]:
    """Map admission URI → admit datetime."""
    query = """
    SELECT ?adm ?ts WHERE {
        ?adm a mimic:HospitalAdmission ;
             time:hasBeginning ?begin .
        ?begin time:inXSDDateTimeStamp ?ts .
    }
    """
    return {row[0]: _parse_timestamp(row[1]) for row in g.query(query)}


def _build_event_timestamp_lookup(
    g: rdflib.Graph,
) -> dict[URIRef, datetime]:
    """Map event URI → timestamp for instant and interval events."""
    lookup: dict[URIRef, datetime] = {}

    # Instant events
    for ntype in ("lab", "vital", "microbiology"):
        rdf_class = _CLINICAL_TYPE_CONFIG[ntype][0]
        query = f"""
        SELECT ?event ?ts WHERE {{
            ?event a <{rdf_class}> ;
                   <{TIME_NS.inXSDDateTimeStamp}> ?ts .
        }}
        """
        for row in g.query(query):
            lookup[row[0]] = _parse_timestamp(row[1])

    # Interval events (prescriptions) — use beginning timestamp
    query = f"""
    SELECT ?event ?ts WHERE {{
        ?event a <{MIMIC_NS.PrescriptionEvent}> ;
               <{TIME_NS.hasBeginning}> ?begin .
        ?begin <{TIME_NS.inXSDDateTimeStamp}> ?ts .
    }}
    """
    for row in g.query(query):
        lookup[row[0]] = _parse_timestamp(row[1])

    return lookup


def _build_event_admission_lookup(
    g: rdflib.Graph,
) -> dict[URIRef, URIRef]:
    """Map event URI → admission URI via ICU day chain or diagnosis link.

    ICU-linked events are resolved through a 3-hop traversal::

        event → associatedWithICUDay → day
        day   → partOf               → icu_stay
        adm   → containsICUStay      → icu_stay  (matched in reverse)

    Diagnosis events have a direct ``diagnosisOf`` link to the admission.
    """
    lookup: dict[URIRef, URIRef] = {}

    # 3-hop chain: event→day→stay, then find admission owning that stay
    query = """
    SELECT ?event ?adm WHERE {
        ?event mimic:associatedWithICUDay ?day .
        ?day mimic:partOf ?stay .
        ?adm mimic:containsICUStay ?stay .
    }
    """
    for row in g.query(query):
        lookup[row[0]] = row[1]

    # Diagnosis events linked directly
    query = """
    SELECT ?event ?adm WHERE {
        ?event mimic:diagnosisOf ?adm .
    }
    """
    for row in g.query(query):
        lookup[row[0]] = row[1]

    return lookup


def _extract_structural_edges(
    g: rdflib.Graph,
    structural_maps: dict[str, dict[URIRef, int]],
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor | None]]:
    """Extract patient→admission, admission→icu_stay, icu_stay→icu_day edges."""
    edges: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor | None]] = {}

    # patient → has_admission → admission
    query = """
    SELECT ?patient ?adm WHERE {
        ?patient mimic:hasAdmission ?adm .
    }
    """
    src_ids, dst_ids = [], []
    for row in g.query(query):
        if row[0] in structural_maps["patient"] and row[1] in structural_maps["admission"]:
            src_ids.append(structural_maps["patient"][row[0]])
            dst_ids.append(structural_maps["admission"][row[1]])
    if src_ids:
        edges[("patient", "has_admission", "admission")] = (
            torch.tensor([src_ids, dst_ids], dtype=torch.long), None
        )

    # admission → contains_icu_stay → icu_stay
    query = """
    SELECT ?adm ?stay WHERE {
        ?adm mimic:containsICUStay ?stay .
    }
    """
    src_ids, dst_ids = [], []
    for row in g.query(query):
        if row[0] in structural_maps["admission"] and row[1] in structural_maps["icu_stay"]:
            src_ids.append(structural_maps["admission"][row[0]])
            dst_ids.append(structural_maps["icu_stay"][row[1]])
    if src_ids:
        edges[("admission", "contains_icu_stay", "icu_stay")] = (
            torch.tensor([src_ids, dst_ids], dtype=torch.long), None
        )

    # icu_stay → has_icu_day → icu_day
    query = """
    SELECT ?stay ?day WHERE {
        ?stay mimic:hasICUDay ?day .
    }
    """
    src_ids, dst_ids = [], []
    for row in g.query(query):
        if row[0] in structural_maps["icu_stay"] and row[1] in structural_maps["icu_day"]:
            src_ids.append(structural_maps["icu_stay"][row[0]])
            dst_ids.append(structural_maps["icu_day"][row[1]])
    if src_ids:
        edges[("icu_stay", "has_icu_day", "icu_day")] = (
            torch.tensor([src_ids, dst_ids], dtype=torch.long), None
        )

    return edges


def _extract_clinical_event_edges(
    g: rdflib.Graph,
    structural_maps: dict[str, dict[URIRef, int]],
    concept_id_map: dict[str, dict[str, int]],
    event_to_concept: dict[str, dict[URIRef, str]],
    adm_start_lookup: dict[URIRef, datetime],
    event_ts_lookup: dict[URIRef, datetime],
    event_adm_lookup: dict[URIRef, URIRef],
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor | None]]:
    """Extract icu_day→event edges with temporal delta attributes."""
    edges: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor | None]] = {}

    for ntype in ("lab", "vital", "drug", "microbiology"):
        rdf_class = _CLINICAL_TYPE_CONFIG[ntype][0]
        query = f"""
        SELECT ?day ?event WHERE {{
            ?day <{MIMIC_NS.hasICUDayEvent}> ?event .
            ?event a <{rdf_class}> .
        }}
        """
        src_ids, dst_ids, deltas = [], [], []
        for row in g.query(query):
            day_uri, event_uri = row[0], row[1]
            if day_uri not in structural_maps["icu_day"]:
                continue
            if event_uri not in event_to_concept[ntype]:
                continue
            ckey = event_to_concept[ntype][event_uri]
            if ckey not in concept_id_map[ntype]:
                continue

            src_ids.append(structural_maps["icu_day"][day_uri])
            dst_ids.append(concept_id_map[ntype][ckey])

            # Compute Δt
            dt = 0.0
            if event_uri in event_ts_lookup and event_uri in event_adm_lookup:
                adm_uri = event_adm_lookup[event_uri]
                if adm_uri in adm_start_lookup:
                    delta = event_ts_lookup[event_uri] - adm_start_lookup[adm_uri]
                    dt = delta.total_seconds() / 86400.0
            deltas.append(dt)

        if src_ids:
            edge_attr = torch.tensor(deltas, dtype=torch.float).unsqueeze(1)
            edges[("icu_day", "has_event", ntype)] = (
                torch.tensor([src_ids, dst_ids], dtype=torch.long), edge_attr
            )

    return edges


def _extract_diagnosis_edges(
    g: rdflib.Graph,
    structural_maps: dict[str, dict[URIRef, int]],
    concept_id_map: dict[str, dict[str, int]],
    event_to_concept: dict[str, dict[URIRef, str]],
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor | None]]:
    """Extract admission→diagnosis edges."""
    query = """
    SELECT ?adm ?event WHERE {
        ?adm mimic:hasDiagnosis ?event .
    }
    """
    src_ids, dst_ids = [], []
    for row in g.query(query):
        adm_uri, event_uri = row[0], row[1]
        if adm_uri not in structural_maps["admission"]:
            continue
        if event_uri not in event_to_concept["diagnosis"]:
            continue
        ckey = event_to_concept["diagnosis"][event_uri]
        if ckey not in concept_id_map["diagnosis"]:
            continue
        src_ids.append(structural_maps["admission"][adm_uri])
        dst_ids.append(concept_id_map["diagnosis"][ckey])

    if src_ids:
        return {
            ("admission", "has_diagnosis", "diagnosis"): (
                torch.tensor([src_ids, dst_ids], dtype=torch.long), None
            )
        }
    return {}


def _extract_followed_by_edges(
    g: rdflib.Graph,
    structural_maps: dict[str, dict[URIRef, int]],
    adm_start_lookup: dict[URIRef, datetime],
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor | None]]:
    """Extract admission→followed_by→admission edges with gap-day attribute."""
    query = """
    SELECT ?adm1 ?adm2 WHERE {
        ?adm1 mimic:followedBy ?adm2 .
    }
    """
    src_ids, dst_ids, gaps = [], [], []
    for row in g.query(query):
        a1, a2 = row[0], row[1]
        if a1 not in structural_maps["admission"] or a2 not in structural_maps["admission"]:
            continue
        src_ids.append(structural_maps["admission"][a1])
        dst_ids.append(structural_maps["admission"][a2])

        gap = 0.0
        if a1 in adm_start_lookup and a2 in adm_start_lookup:
            gap = (adm_start_lookup[a2] - adm_start_lookup[a1]).total_seconds() / 86400.0
        gaps.append(gap)

    if src_ids:
        return {
            ("admission", "followed_by", "admission"): (
                torch.tensor([src_ids, dst_ids], dtype=torch.long),
                torch.tensor(gaps, dtype=torch.float).unsqueeze(1),
            )
        }
    return {}


def _add_reverse_edges(data: HeteroData) -> None:
    """Add reverse edges for every existing edge type."""
    edge_types = list(data.edge_types)
    for src, rel, dst in edge_types:
        ei = data[src, rel, dst].edge_index
        rev_ei = torch.stack([ei[1], ei[0]], dim=0)
        rev_rel = f"rev_{rel}"
        data[dst, rev_rel, src].edge_index = rev_ei
        if hasattr(data[src, rel, dst], "edge_attr") and data[src, rel, dst].edge_attr is not None:
            data[dst, rev_rel, src].edge_attr = data[src, rel, dst].edge_attr.clone()


# ──────────────────────────────────────────────────────────────────────────────
# D. Labels and Masks
# ──────────────────────────────────────────────────────────────────────────────

def _build_labels(
    g: rdflib.Graph,
    admission_map: dict[URIRef, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build readmission label tensors aligned to admission node ordering."""
    n = len(admission_map)
    y_30d = torch.zeros(n, dtype=torch.float)
    y_60d = torch.zeros(n, dtype=torch.float)

    query = """
    SELECT ?adm ?r30 ?r60 WHERE {
        ?adm a mimic:HospitalAdmission ;
             mimic:readmittedWithin30Days ?r30 ;
             mimic:readmittedWithin60Days ?r60 .
    }
    """
    for row in g.query(query):
        uri = row[0]
        if uri in admission_map:
            idx = admission_map[uri]
            y_30d[idx] = 1.0 if str(row[1]).lower() == "true" else 0.0
            y_60d[idx] = 1.0 if str(row[2]).lower() == "true" else 0.0

    return y_30d, y_60d


def _build_masks(
    g: rdflib.Graph,
    admission_map: dict[URIRef, int],
    split_fn: Callable | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build train/val/test boolean masks over admission nodes."""
    n = len(admission_map)

    if split_fn is None:
        return torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)

    # Build a DataFrame with subject_id, hadm_id, readmitted_30d
    query = """
    SELECT ?adm ?sid ?hadm ?r30 WHERE {
        ?adm a mimic:HospitalAdmission ;
             mimic:hasAdmissionId ?hadm .
        ?patient mimic:hasAdmission ?adm ;
                 mimic:hasSubjectId ?sid .
        ?adm mimic:readmittedWithin30Days ?r30 .
    }
    """
    rows = []
    for row in g.query(query):
        adm_uri = row[0]
        if adm_uri not in admission_map:
            continue
        rows.append({
            "adm_uri": str(adm_uri),
            "subject_id": int(row[1]),
            "hadm_id": int(row[2]),
            "readmitted_30d": 1 if str(row[3]).lower() == "true" else 0,
        })

    df = pd.DataFrame(rows)
    train_df, val_df, test_df = split_fn(df, "readmitted_30d")

    # Map back to boolean masks
    uri_to_idx = {str(uri): idx for uri, idx in admission_map.items()}

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for _, row in train_df.iterrows():
        if row["adm_uri"] in uri_to_idx:
            train_mask[uri_to_idx[row["adm_uri"]]] = True
    for _, row in val_df.iterrows():
        if row["adm_uri"] in uri_to_idx:
            val_mask[uri_to_idx[row["adm_uri"]]] = True
    for _, row in test_df.iterrows():
        if row["adm_uri"] in uri_to_idx:
            test_mask[uri_to_idx[row["adm_uri"]]] = True

    return train_mask, val_mask, test_mask


# ──────────────────────────────────────────────────────────────────────────────
# E. Validation and Save
# ──────────────────────────────────────────────────────────────────────────────

def _validate_heterodata(data: HeteroData) -> None:
    """Validate the HeteroData for common issues."""
    for ntype in data.node_types:
        if hasattr(data[ntype], "x") and data[ntype].x is not None:
            if torch.isnan(data[ntype].x).any():
                raise ValueError(f"NaN detected in '{ntype}' node features")

    for src, rel, dst in data.edge_types:
        ei = data[src, rel, dst].edge_index
        n_src = data[src].num_nodes
        n_dst = data[dst].num_nodes
        if ei.numel() > 0:
            if ei[0].max() >= n_src:
                raise ValueError(
                    f"Edge index out of bounds for ({src},{rel},{dst}): "
                    f"max src {ei[0].max()} >= {n_src}"
                )
            if ei[1].max() >= n_dst:
                raise ValueError(
                    f"Edge index out of bounds for ({src},{rel},{dst}): "
                    f"max dst {ei[1].max()} >= {n_dst}"
                )

    if hasattr(data["admission"], "y"):
        if data["admission"].y.sum() == 0:
            logger.warning("All labels are zero — check label extraction")

    if (
        hasattr(data["admission"], "train_mask")
        and hasattr(data["admission"], "val_mask")
        and hasattr(data["admission"], "test_mask")
    ):
        tm = data["admission"].train_mask
        vm = data["admission"].val_mask
        tsm = data["admission"].test_mask
        if (tm & vm).any() or (tm & tsm).any() or (vm & tsm).any():
            raise ValueError("Masks are not disjoint")
        if not (tm | vm | tsm).all():
            raise ValueError("Masks do not cover all admission nodes")


def _save_metadata(
    data: HeteroData, output_path: Path,
) -> None:
    """Save metadata JSON alongside the graph file."""
    meta = {
        "node_counts": {ntype: data[ntype].num_nodes for ntype in data.node_types},
        "edge_counts": {
            f"({s},{r},{d})": data[s, r, d].edge_index.shape[1]
            for s, r, d in data.edge_types
        },
        "label_distribution": {},
        "timestamp": datetime.now().isoformat(),
    }
    if hasattr(data["admission"], "y"):
        y = data["admission"].y
        meta["label_distribution"] = {
            "positive": int(y.sum().item()),
            "negative": int((y == 0).sum().item()),
            "total": int(y.numel()),
        }

    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved metadata to %s", meta_path)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def export_rdf_to_heterodata(
    rdf_graph: rdflib.Graph,
    feature_matrix_path: Path,
    concept_embeddings_path: Path,
    split_fn: Callable | None = None,
    output_path: Path = Path("data/processed/full_hetero_graph.pt"),
    embed_unmapped_fn: Callable | None = None,
) -> HeteroData:
    """Convert an RDF knowledge graph to a PyG HeteroData object.

    Args:
        rdf_graph: Populated rdflib Graph with MIMIC-IV clinical data.
        feature_matrix_path: Path to parquet with admission-level features.
        concept_embeddings_path: Path to ``.pt`` file with SNOMED concept embeddings.
        split_fn: Callable ``(df, target_col) -> (train, val, test)`` for
            patient-level splitting.  If ``None``, all admissions are assigned
            to train_mask.
        output_path: Where to save the ``.pt`` file.
        embed_unmapped_fn: Optional override for unmapped term embedding
            (useful for testing without loading SapBERT).

    Returns:
        ``HeteroData`` with the following schema:

        **Node types and features** (``data[type].x``):

        ============  ====  =========================================
        Type          Dim   Description
        ============  ====  =========================================
        patient       2     [age, gender (M=1/F=0)]
        admission     F     Parquet features (F = non-ID columns)
        icu_stay      2     [duration_hours, num_icu_days]
        icu_day       2     [day_number, num_events]
        lab           773   [SapBERT(768), mean, std, min, max, abnormal_rate]
        vital         773   [SapBERT(768), mean, std, min, max, abnormal_rate]
        drug          768   SapBERT embedding
        diagnosis     768   SapBERT embedding
        microbiology  768   SapBERT embedding
        ============  ====  =========================================

        **Edge types** (``data[src, rel, dst].edge_index``):

        - ``(patient, has_admission, admission)``
        - ``(admission, contains_icu_stay, icu_stay)``
        - ``(icu_stay, has_icu_day, icu_day)``
        - ``(icu_day, has_event, lab|vital|drug|microbiology)`` — edge_attr: Δt days
        - ``(admission, has_diagnosis, diagnosis)``
        - ``(admission, followed_by, admission)`` — edge_attr: gap days
        - All of the above with ``rev_*`` reverse counterparts.

        **Labels** (on admission nodes):

        - ``data['admission'].y``: 30-day readmission (float, 0/1)
        - ``data['admission'].y_60d``: 60-day readmission (float, 0/1)
        - ``data['admission'].train_mask / val_mask / test_mask``: boolean
    """
    data = HeteroData()

    # ── A. Node maps ──────────────────────────────────────────────────────
    logger.info("Building node ID maps...")
    structural_maps = _build_structural_node_maps(rdf_graph)
    concept_id_map, event_to_concept, concept_labels = _build_clinical_node_maps(rdf_graph)

    _save_node_mappings(
        structural_maps, concept_id_map,
        output_path.with_suffix(".mappings.json"),
    )

    # Set num_nodes for all types
    for ntype, m in structural_maps.items():
        data[ntype].num_nodes = len(m)
    for ntype, m in concept_id_map.items():
        data[ntype].num_nodes = len(m)

    # ── B. Node features ──────────────────────────────────────────────────
    logger.info("Building node features...")
    data["patient"].x = _build_patient_features(rdf_graph, structural_maps["patient"])
    data["admission"].x = _build_admission_features(structural_maps["admission"], feature_matrix_path)
    data["icu_stay"].x = _build_icu_stay_features(rdf_graph, structural_maps["icu_stay"])
    data["icu_day"].x = _build_icu_day_features(rdf_graph, structural_maps["icu_day"])

    for ntype in ("lab", "vital", "drug", "diagnosis", "microbiology"):
        data[ntype].x = _build_clinical_node_features(
            rdf_graph,
            concept_id_map[ntype],
            event_to_concept[ntype],
            concept_labels[ntype],
            concept_embeddings_path,
            ntype,
            embed_unmapped_fn=embed_unmapped_fn,
        )

    # ── C. Edges ──────────────────────────────────────────────────────────
    logger.info("Extracting edges...")
    adm_start_lookup = _build_admission_start_lookup(rdf_graph)
    event_ts_lookup = _build_event_timestamp_lookup(rdf_graph)
    event_adm_lookup = _build_event_admission_lookup(rdf_graph)

    # Structural edges
    for edge_type, (ei, attr) in _extract_structural_edges(rdf_graph, structural_maps).items():
        data[edge_type].edge_index = ei
        if attr is not None:
            data[edge_type].edge_attr = attr

    # Clinical event edges (icu_day → event nodes)
    for edge_type, (ei, attr) in _extract_clinical_event_edges(
        rdf_graph, structural_maps, concept_id_map, event_to_concept,
        adm_start_lookup, event_ts_lookup, event_adm_lookup,
    ).items():
        data[edge_type].edge_index = ei
        if attr is not None:
            data[edge_type].edge_attr = attr

    # Diagnosis edges (admission → diagnosis)
    for edge_type, (ei, attr) in _extract_diagnosis_edges(
        rdf_graph, structural_maps, concept_id_map, event_to_concept,
    ).items():
        data[edge_type].edge_index = ei
        if attr is not None:
            data[edge_type].edge_attr = attr

    # Followed-by edges
    for edge_type, (ei, attr) in _extract_followed_by_edges(
        rdf_graph, structural_maps, adm_start_lookup,
    ).items():
        data[edge_type].edge_index = ei
        if attr is not None:
            data[edge_type].edge_attr = attr

    # Reverse edges
    _add_reverse_edges(data)

    # ── D. Labels & masks ─────────────────────────────────────────────────
    logger.info("Building labels and masks...")
    y_30d, y_60d = _build_labels(rdf_graph, structural_maps["admission"])
    data["admission"].y = y_30d
    data["admission"].y_60d = y_60d

    train_mask, val_mask, test_mask = _build_masks(
        rdf_graph, structural_maps["admission"], split_fn,
    )
    data["admission"].train_mask = train_mask
    data["admission"].val_mask = val_mask
    data["admission"].test_mask = test_mask

    # ── E. Validate & save ────────────────────────────────────────────────
    logger.info("Validating HeteroData...")
    _validate_heterodata(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    logger.info("Saved HeteroData to %s", output_path)

    _save_metadata(data, output_path)

    return data
