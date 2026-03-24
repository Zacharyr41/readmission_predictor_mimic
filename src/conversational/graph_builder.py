"""Build an RDF knowledge graph from a conversational ExtractionResult.

Bridges the conversational extractor output to the existing graph
construction writer functions — no new graph construction logic.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rdflib import Graph, URIRef

from src.conversational.models import ExtractionResult
from src.graph_construction.event_writers import (
    write_biomarker_event,
    write_clinical_sign_event,
    write_diagnosis_event,
    write_icu_days,
    write_icu_stay,
    write_microbiology_event,
    write_prescription_event,
)
from src.graph_construction.ontology import initialize_graph
from src.graph_construction.patient_writer import (
    link_sequential_admissions,
    write_admission,
    write_patient,
)
from src.graph_construction.temporal.allen_relations import (
    compute_allen_relations_for_patient,
)


def build_query_graph(
    ontology_dir: Path,
    extraction: ExtractionResult,
) -> tuple[Graph, dict]:
    """Build an RDF graph from a conversational extraction result.

    Parameters
    ----------
    ontology_dir:
        Path to the directory containing ``base_ontology.rdf`` and
        ``extended_ontology.rdf``.
    extraction:
        Structured extraction result from the conversational extractor.

    Returns
    -------
    tuple[Graph, dict]
        The RDF graph and a stats dictionary with entity counts.
    """
    graph = initialize_graph(ontology_dir)
    stats = {
        "patients": 0,
        "admissions": 0,
        "icu_stays": 0,
        "icu_days": 0,
        "biomarkers": 0,
        "vitals": 0,
        "prescriptions": 0,
        "diagnoses": 0,
        "microbiology": 0,
        "allen_relations": 0,
    }

    if not extraction.patients:
        return graph, stats

    # -- Lookup indices -------------------------------------------------------
    hadm_stay_index = _build_hadm_stay_index(extraction.icu_stays)
    adm_by_subject: dict[int, list[dict]] = defaultdict(list)
    for adm in extraction.admissions:
        adm_by_subject[adm["subject_id"]].append(adm)

    hadm_to_admission_uri: dict[int, URIRef] = {}
    stay_meta: dict[int, tuple[URIRef, list]] = {}
    patient_uris: list[URIRef] = []

    # -- Patients, admissions, ICU stays --------------------------------------
    for patient in extraction.patients:
        patient_uri = write_patient(graph, patient)
        patient_uris.append(patient_uri)
        stats["patients"] += 1

        admissions = sorted(
            adm_by_subject.get(patient["subject_id"], []),
            key=lambda a: a["admittime"],
        )
        admission_uris: list[URIRef] = []
        for adm in admissions:
            adm_uri = write_admission(graph, _augment_admission(adm), patient_uri)
            admission_uris.append(adm_uri)
            hadm_to_admission_uri[adm["hadm_id"]] = adm_uri
            stats["admissions"] += 1

        if len(admission_uris) > 1:
            link_sequential_admissions(graph, admission_uris)

        # ICU stays for this patient's admissions
        patient_hadm_ids = {a["hadm_id"] for a in admissions}
        for stay in extraction.icu_stays:
            if stay["hadm_id"] not in patient_hadm_ids:
                continue
            adm_uri = hadm_to_admission_uri[stay["hadm_id"]]
            icu_stay_uri = write_icu_stay(graph, stay, adm_uri)
            icu_day_metadata = write_icu_days(graph, stay, icu_stay_uri)
            stay_meta[stay["stay_id"]] = (icu_stay_uri, icu_day_metadata)
            stats["icu_stays"] += 1
            stats["icu_days"] += len(icu_day_metadata)

    # -- Events ---------------------------------------------------------------
    for event in extraction.events.get("biomarker", []):
        resolved = _resolve_event_to_stay(
            event, "charttime", hadm_stay_index, stay_meta,
        )
        if resolved is None:
            continue
        stay_id, icu_stay_uri, icu_day_meta = resolved
        lab_data = dict(event)
        lab_data["stay_id"] = stay_id
        write_biomarker_event(graph, lab_data, icu_stay_uri, icu_day_meta)
        stats["biomarkers"] += 1

    for event in extraction.events.get("vital", []):
        sid = event.get("stay_id")
        if sid is None or sid not in stay_meta:
            continue
        icu_stay_uri, icu_day_meta = stay_meta[sid]
        write_clinical_sign_event(graph, event, icu_stay_uri, icu_day_meta)
        stats["vitals"] += 1

    for event in extraction.events.get("drug", []):
        if event.get("starttime") is None:
            continue
        resolved = _resolve_event_to_stay(
            event, "starttime", hadm_stay_index, stay_meta,
        )
        if resolved is None:
            continue
        _, icu_stay_uri, icu_day_meta = resolved
        write_prescription_event(graph, event, icu_stay_uri, icu_day_meta)
        stats["prescriptions"] += 1

    for event in extraction.events.get("diagnosis", []):
        hadm_id = event.get("hadm_id")
        if hadm_id not in hadm_to_admission_uri:
            continue
        write_diagnosis_event(graph, event, hadm_to_admission_uri[hadm_id])
        stats["diagnoses"] += 1

    for event in extraction.events.get("microbiology", []):
        resolved = _resolve_event_to_stay(
            event, "charttime", hadm_stay_index, stay_meta,
        )
        if resolved is None:
            continue
        _, icu_stay_uri, icu_day_meta = resolved
        write_microbiology_event(graph, event, icu_stay_uri, icu_day_meta)
        stats["microbiology"] += 1

    # -- Allen temporal relations ---------------------------------------------
    for patient_uri in patient_uris:
        stats["allen_relations"] += compute_allen_relations_for_patient(
            graph, patient_uri,
        )

    return graph, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_hadm_stay_index(icu_stays: list[dict]) -> dict[int, list[dict]]:
    """Group ICU stays by ``hadm_id``."""
    index: dict[int, list[dict]] = defaultdict(list)
    for stay in icu_stays:
        index[stay["hadm_id"]].append(stay)
    return index


def _resolve_event_to_stay(
    event: dict,
    charttime_key: str,
    hadm_stay_index: dict[int, list[dict]],
    stay_meta: dict[int, tuple[URIRef, list]],
) -> tuple[int, URIRef, list] | None:
    """Find the ICU stay containing the event's timestamp.

    Returns ``(stay_id, icu_stay_uri, icu_day_metadata)`` or ``None`` if
    no matching ICU stay is found.
    """
    hadm_id = event.get("hadm_id")
    charttime = event.get(charttime_key)
    if hadm_id is None or charttime is None:
        return None
    for stay in hadm_stay_index.get(hadm_id, []):
        if stay["intime"] <= charttime <= stay["outtime"]:
            stay_id = stay["stay_id"]
            if stay_id in stay_meta:
                uri, meta = stay_meta[stay_id]
                return stay_id, uri, meta
    return None


def _augment_admission(admission: dict) -> dict:
    """Add default readmission labels for query-scoped graphs."""
    aug = dict(admission)
    aug.setdefault("readmitted_30d", False)
    aug.setdefault("readmitted_60d", False)
    return aug
