"""WLST graph construction pipeline.

Builds 48h-windowed RDF knowledge graphs for TBI patients. Adapts the
existing graph construction infrastructure with WLST-specific cohort,
labels, event types, and time windowing.
"""

import logging
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.graph_construction.disk_graph import (
    bind_namespaces,
    close_disk_graph,
    open_disk_graph,
)
from src.graph_construction.event_writers import (
    write_biomarker_event,
    write_diagnosis_event,
    write_icu_days,
    write_icu_stay,
)
from src.graph_construction.ontology import MIMIC_NS, initialize_graph
from src.graph_construction.patient_writer import (
    write_admission,
    write_patient,
)
from src.graph_construction.temporal.allen_relations import (
    compute_allen_relations_for_patient,
)
from src.graph_construction.terminology import SnomedMapper
from src.ingestion.derived_tables import create_age_table
from src.wlst.cohort import create_wlst_labels, select_tbi_cohort
from src.wlst.event_writers import (
    write_gcs_event,
    write_icp_medication_event,
    write_map_event,
    write_neurosurgery_event,
    write_vasopressor_event,
    write_ventilation_event,
)

logger = logging.getLogger(__name__)

# Code status itemid — EXCLUDED from features to prevent label leakage
CODE_STATUS_ITEMID = 223758

# GCS component itemids
GCS_EYE_ITEMID = 220739
GCS_VERBAL_ITEMID = 223900
GCS_MOTOR_ITEMID = 223901

# MAP itemids
MAP_ARTERIAL_ITEMID = 220052
MAP_CUFF_ITEMID = 220181

# Vasopressor drug names (lowercased for matching)
VASOPRESSOR_DRUGS = {
    "norepinephrine", "phenylephrine", "vasopressin",
    "epinephrine", "dopamine",
}

# ICP medication keywords (lowercased for matching)
ICP_MED_KEYWORDS = {"mannitol", "hypertonic saline", "levetiracetam"}

# WLST-relevant lab itemids
WLST_LAB_ITEMIDS = {
    50983,  # Sodium
    50813,  # Lactate
    50931,  # Glucose
    51237,  # INR
    50912,  # Creatinine
}

# Neurosurgery procedure itemids
NEUROSURGERY_ITEMIDS = {
    225752,  # Craniectomy
    228114,  # ICP monitor placement
    227190,  # EVD/Ventriculostomy
}

# Ventilation procedure itemid
VENTILATION_ITEMID = 225792


def build_wlst_graph(
    db_path: Path,
    ontology_dir: Path,
    output_path: Path,
    icd_prefixes: list[str] | None = None,
    gcs_threshold: int = 8,
    icu_types: list[str] | None = None,
    observation_window_hours: int = 48,
    patients_limit: int = 0,
    skip_allen_relations: bool = False,
    snomed_mappings_dir: Path | None = None,
    umls_api_key: str | None = None,
    n_workers: int | None = None,
    stage: str = "stage1",
) -> tuple[Graph, pd.DataFrame]:
    """Build RDF graph for WLST prediction from MIMIC-IV DuckDB.

    Orchestrates:
    1. TBI cohort selection + GCS filtering
    2. WLST label derivation
    3. 48h-windowed graph construction with WLST-specific event types
    4. Allen temporal relations (optional)

    Args:
        db_path: Path to DuckDB database file.
        ontology_dir: Path to directory containing ontology files.
        output_path: Path to write RDF output file.
        icd_prefixes: ICD-10 prefixes for TBI (default: ["S06"]).
        gcs_threshold: Maximum GCS total for inclusion.
        icu_types: ICU care unit types to include.
        observation_window_hours: Feature window in hours.
        patients_limit: Maximum patients to process (0 = no limit).
        skip_allen_relations: If True, skip Allen relation computation.
        snomed_mappings_dir: Path to SNOMED mapping files.
        umls_api_key: Optional UMLS API key.
        n_workers: Number of parallel workers. None = auto.
        stage: "stage1" (clinical only) or "stage2" (add non-clinical confounders).

    Returns:
        Tuple of (RDF graph, WLST labels DataFrame).
    """
    # Create derived tables
    write_conn = duckdb.connect(str(db_path))
    _ensure_age_table(write_conn)

    # Select cohort and derive labels
    cohort_df = select_tbi_cohort(
        write_conn, icd_prefixes, gcs_threshold, icu_types,
        observation_window_hours, patients_limit,
    )

    if len(cohort_df) == 0:
        logger.warning("No patients in TBI cohort. Graph will be empty.")
        write_conn.close()
        graph = Graph()
        return graph, pd.DataFrame()

    labels_df = create_wlst_labels(write_conn, cohort_df)

    # Store labels in DuckDB for pipeline queries
    write_conn.execute(
        "CREATE OR REPLACE TABLE wlst_labels AS SELECT * FROM labels_df"
    )
    write_conn.close()

    # Read-only connection for graph construction
    conn = duckdb.connect(str(db_path), read_only=True)

    unique_patients = labels_df["subject_id"].unique().tolist()
    total_patients = len(unique_patients)
    logger.info(f"Building WLST graph for {total_patients} patients (stage={stage})")

    # Initialize disk-backed graph
    store_path = output_path.parent / "wlst_oxigraph_store"
    graph = open_disk_graph(store_path)
    bind_namespaces(graph)

    # Copy ontology triples
    ontology_graph = initialize_graph(ontology_dir)
    for triple in ontology_graph:
        graph.add(triple)

    # Initialize SNOMED mapper
    snomed_mapper = None
    if snomed_mappings_dir is not None and snomed_mappings_dir.exists():
        snomed_mapper = SnomedMapper(snomed_mappings_dir, umls_api_key=umls_api_key)

    stats = _empty_stats()

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4)

    if n_workers > 1 and total_patients > n_workers:
        conn.close()
        logger.info(f"Using {n_workers} worker processes")

        worker_args = [
            (subject_id, labels_df, observation_window_hours,
             skip_allen_relations, stage)
            for subject_id in unique_patients
        ]
        init_args = (db_path, snomed_mappings_dir, umls_api_key)
        nt_paths: list[Path] = []

        with Pool(n_workers, _worker_init, init_args) as pool:
            for idx, (nt_path, patient_stats) in enumerate(
                pool.imap_unordered(_process_single_patient, worker_args), 1
            ):
                logger.info(
                    f"Patient {idx}/{total_patients} complete "
                    f"(+{sum(patient_stats.values())} entities)"
                )
                nt_paths.append(Path(nt_path))
                for key, value in patient_stats.items():
                    stats[key] = stats.get(key, 0) + value

        # Merge sub-graphs
        logger.info(f"Merging {len(nt_paths)} patient sub-graphs...")
        for idx, nt_path in enumerate(nt_paths, 1):
            graph.parse(str(nt_path), format="nt")
            nt_path.unlink(missing_ok=True)
            if idx % 25 == 0 or idx == len(nt_paths):
                logger.info(f"  Merged {idx}/{len(nt_paths)} files ({len(graph)} triples)")
    else:
        for idx, subject_id in enumerate(unique_patients, 1):
            logger.info(f"Processing patient {idx}/{total_patients} (subject_id={subject_id})")
            patient_stats = _process_patient(
                conn=conn,
                graph=graph,
                subject_id=subject_id,
                labels_df=labels_df,
                observation_window_hours=observation_window_hours,
                skip_allen_relations=skip_allen_relations,
                snomed_mapper=snomed_mapper,
                stage=stage,
            )
            for key, value in patient_stats.items():
                stats[key] = stats.get(key, 0) + value
        conn.close()

    # Serialize
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing {len(graph)} triples to {output_path}...")
    graph.serialize(destination=str(output_path), format="nt")

    _log_stats(stats)
    return graph, labels_df


def _empty_stats() -> dict[str, int]:
    return {
        "patients": 0, "admissions": 0, "icu_stays": 0, "icu_days": 0,
        "gcs_events": 0, "map_events": 0, "vasopressor_events": 0,
        "ventilation_events": 0, "neurosurgery_events": 0,
        "icp_medication_events": 0, "biomarkers": 0, "diagnoses": 0,
        "allen_relations": 0,
    }


def _log_stats(stats: dict[str, int]) -> None:
    logger.info("WLST graph construction complete:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def _ensure_age_table(conn: duckdb.DuckDBPyConnection) -> None:
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'age'"
    ).fetchone()
    if result[0] == 0:
        create_age_table(conn)


# ==================== Worker pool support ====================

_worker_conn = None
_worker_mapper = None


def _worker_init(db_path, snomed_mappings_dir, umls_api_key):
    global _worker_conn, _worker_mapper  # noqa: PLW0603
    _worker_conn = duckdb.connect(str(db_path), read_only=True)
    if snomed_mappings_dir is not None and Path(snomed_mappings_dir).exists():
        _worker_mapper = SnomedMapper(
            Path(snomed_mappings_dir), umls_api_key=umls_api_key
        )


def _process_single_patient(args: tuple) -> tuple[str, dict[str, int]]:
    (subject_id, labels_df, observation_window_hours,
     skip_allen_relations, stage) = args

    graph = Graph()
    patient_stats = _process_patient(
        conn=_worker_conn,
        graph=graph,
        subject_id=subject_id,
        labels_df=labels_df,
        observation_window_hours=observation_window_hours,
        skip_allen_relations=skip_allen_relations,
        snomed_mapper=_worker_mapper,
        stage=stage,
    )

    fd, nt_path = tempfile.mkstemp(suffix=".nt", prefix="wlst_patient_")
    os.close(fd)
    graph.serialize(destination=nt_path, format="nt")
    return nt_path, patient_stats


# ==================== Per-patient processing ====================


def _process_patient(
    conn: duckdb.DuckDBPyConnection,
    graph: Graph,
    subject_id: int,
    labels_df: pd.DataFrame,
    observation_window_hours: int,
    skip_allen_relations: bool = False,
    snomed_mapper=None,
    stage: str = "stage1",
) -> dict[str, int]:
    """Process a single patient for the WLST graph."""
    stats = _empty_stats()

    patient_data = _query_patient_demographics(conn, subject_id)
    if patient_data is None:
        return stats

    patient_uri = write_patient(graph, patient_data)
    stats["patients"] = 1

    # Get this patient's rows from labels_df
    patient_labels = labels_df[labels_df["subject_id"] == subject_id]

    for _, row in patient_labels.iterrows():
        hadm_id = row["hadm_id"]
        stay_id = row["stay_id"]
        intime = row["intime"]
        wlst_label = row["wlst_label"]

        # Write admission with WLST label
        adm_data = _query_admission(conn, hadm_id)
        if adm_data is None:
            continue
        adm_data["wlst_label"] = int(wlst_label)
        admission_uri = _write_wlst_admission(graph, adm_data, patient_uri)
        stats["admissions"] += 1

        # Write ICU stay and days
        stay_data = _query_icu_stay(conn, stay_id)
        if stay_data is None:
            continue
        icu_stay_uri = write_icu_stay(graph, stay_data, admission_uri)
        stats["icu_stays"] += 1

        icu_day_metadata = write_icu_days(graph, stay_data, icu_stay_uri)
        stats["icu_days"] += len(icu_day_metadata)

        # 48h window boundary
        window_end = intime + pd.Timedelta(hours=observation_window_hours)

        # Write WLST-specific events (all within 48h window)
        stats["gcs_events"] += _write_gcs_series(
            conn, graph, stay_id, intime, window_end,
            icu_stay_uri, icu_day_metadata,
        )
        stats["map_events"] += _write_map_series(
            conn, graph, stay_id, intime, window_end,
            icu_stay_uri, icu_day_metadata,
        )
        stats["vasopressor_events"] += _write_vasopressor_events(
            conn, graph, stay_id, intime, window_end,
            icu_stay_uri, icu_day_metadata,
        )
        stats["ventilation_events"] += _write_ventilation_events(
            conn, graph, stay_id, intime, window_end,
            icu_stay_uri, icu_day_metadata,
        )
        stats["neurosurgery_events"] += _write_neurosurgery_events(
            conn, graph, stay_id, intime, window_end,
            icu_stay_uri, icu_day_metadata,
        )
        stats["icp_medication_events"] += _write_icp_medication_events(
            conn, graph, stay_id, intime, window_end,
            icu_stay_uri, icu_day_metadata,
        )
        stats["biomarkers"] += _write_lab_events(
            conn, graph, stay_data, intime, window_end,
            icu_stay_uri, icu_day_metadata, snomed_mapper,
        )

        # Diagnoses (admission-level, no time window)
        stats["diagnoses"] += _write_diagnoses(
            conn, graph, hadm_id, admission_uri, snomed_mapper,
        )

        # Stage 2: add non-clinical confounders as admission attributes
        if stage == "stage2":
            _write_stage2_attributes(conn, graph, admission_uri, hadm_id, subject_id)

    # Allen temporal relations
    if not skip_allen_relations:
        allen_count = compute_allen_relations_for_patient(graph, patient_uri)
        stats["allen_relations"] = allen_count

    return stats


# ==================== Admission writer (WLST-specific) ====================


def _write_wlst_admission(
    graph: Graph, adm_data: dict, patient_uri: URIRef
) -> URIRef:
    """Write admission node with WLST label instead of readmission labels."""
    # Use base writer for structure
    adm_data_for_base = {
        **adm_data,
        "readmitted_30d": False,
        "readmitted_60d": False,
    }
    admission_uri = write_admission(graph, adm_data_for_base, patient_uri)

    # Add WLST label
    graph.add((
        admission_uri,
        MIMIC_NS.hasWLSTLabel,
        Literal(adm_data["wlst_label"], datatype=XSD.integer),
    ))

    return admission_uri


# ==================== Event query + write helpers ====================


def _write_gcs_series(
    conn, graph, stay_id, intime, window_end,
    icu_stay_uri, icu_day_metadata,
) -> int:
    """Query and write GCS events within 48h window (excluding code status)."""
    rows = conn.execute(
        """
        SELECT ce.stay_id, ce.charttime,
            MAX(CASE WHEN ce.itemid = ? THEN ce.valuenum END) AS gcs_eye,
            MAX(CASE WHEN ce.itemid = ? THEN ce.valuenum END) AS gcs_verbal,
            MAX(CASE WHEN ce.itemid = ? THEN ce.valuenum END) AS gcs_motor
        FROM chartevents ce
        WHERE ce.stay_id = ?
          AND ce.itemid IN (?, ?, ?)
          AND ce.charttime >= ? AND ce.charttime <= ?
        GROUP BY ce.stay_id, ce.charttime
        ORDER BY ce.charttime
        """,
        [GCS_EYE_ITEMID, GCS_VERBAL_ITEMID, GCS_MOTOR_ITEMID,
         stay_id,
         GCS_EYE_ITEMID, GCS_VERBAL_ITEMID, GCS_MOTOR_ITEMID,
         intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        gcs_eye = row[2]
        gcs_verbal = row[3]
        gcs_motor = row[4]
        gcs_total = sum(v for v in [gcs_eye, gcs_verbal, gcs_motor] if v is not None)
        if gcs_total == 0:
            continue

        gcs_data = {
            "stay_id": row[0],
            "charttime": row[1],
            "gcs_eye": gcs_eye,
            "gcs_verbal": gcs_verbal,
            "gcs_motor": gcs_motor,
            "gcs_total": gcs_total,
        }
        write_gcs_event(graph, gcs_data, icu_stay_uri, icu_day_metadata)
        count += 1
    return count


def _write_map_series(
    conn, graph, stay_id, intime, window_end,
    icu_stay_uri, icu_day_metadata,
) -> int:
    """Query and write MAP events within 48h window."""
    rows = conn.execute(
        """
        SELECT stay_id, charttime, valuenum, itemid,
            CASE WHEN itemid = ? THEN 'Arterial BP Mean'
                 ELSE 'Non Invasive BP Mean' END AS label
        FROM chartevents
        WHERE stay_id = ?
          AND itemid IN (?, ?)
          AND valuenum IS NOT NULL
          AND charttime >= ? AND charttime <= ?
        ORDER BY charttime
        """,
        [MAP_ARTERIAL_ITEMID, stay_id,
         MAP_ARTERIAL_ITEMID, MAP_CUFF_ITEMID,
         intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        map_data = {
            "stay_id": row[0], "charttime": row[1], "valuenum": row[2],
            "itemid": row[3], "label": row[4],
        }
        write_map_event(graph, map_data, icu_stay_uri, icu_day_metadata)
        count += 1
    return count


def _write_vasopressor_events(
    conn, graph, stay_id, intime, window_end,
    icu_stay_uri, icu_day_metadata,
) -> int:
    """Query and write vasopressor events from inputevents within 48h window."""
    try:
        conn.execute("SELECT 1 FROM inputevents LIMIT 0")
    except duckdb.CatalogException:
        return 0

    drug_list = ", ".join(f"'{d}'" for d in VASOPRESSOR_DRUGS)
    rows = conn.execute(
        f"""
        SELECT stay_id, starttime, endtime, label, rate, amount, rateuom, amountuom
        FROM inputevents
        WHERE stay_id = ?
          AND LOWER(label) IN ({drug_list})
          AND starttime >= ? AND starttime <= ?
        ORDER BY starttime
        """,
        [stay_id, intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        vaso_data = {
            "stay_id": row[0], "starttime": row[1], "endtime": row[2],
            "label": row[3], "rate": row[4], "amount": row[5],
            "rateuom": row[6], "amountuom": row[7],
        }
        write_vasopressor_event(graph, vaso_data, icu_stay_uri, icu_day_metadata)
        count += 1
    return count


def _write_ventilation_events(
    conn, graph, stay_id, intime, window_end,
    icu_stay_uri, icu_day_metadata,
) -> int:
    """Query and write ventilation events from procedureevents within 48h window."""
    try:
        conn.execute("SELECT 1 FROM procedureevents LIMIT 0")
    except duckdb.CatalogException:
        return 0

    rows = conn.execute(
        """
        SELECT stay_id, starttime, endtime, itemid, label
        FROM procedureevents
        WHERE stay_id = ?
          AND itemid = ?
          AND starttime >= ? AND starttime <= ?
        ORDER BY starttime
        """,
        [stay_id, VENTILATION_ITEMID, intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        vent_data = {
            "stay_id": row[0], "starttime": row[1], "endtime": row[2],
            "itemid": row[3], "label": row[4],
        }
        write_ventilation_event(graph, vent_data, icu_stay_uri, icu_day_metadata)
        count += 1
    return count


def _write_neurosurgery_events(
    conn, graph, stay_id, intime, window_end,
    icu_stay_uri, icu_day_metadata,
) -> int:
    """Query and write neurosurgery events from procedureevents within 48h window."""
    try:
        conn.execute("SELECT 1 FROM procedureevents LIMIT 0")
    except duckdb.CatalogException:
        return 0

    itemid_list = ", ".join(str(i) for i in NEUROSURGERY_ITEMIDS)
    rows = conn.execute(
        f"""
        SELECT stay_id, starttime, endtime, itemid, label
        FROM procedureevents
        WHERE stay_id = ?
          AND itemid IN ({itemid_list})
          AND starttime >= ? AND starttime <= ?
        ORDER BY starttime
        """,
        [stay_id, intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        proc_data = {
            "stay_id": row[0], "starttime": row[1], "endtime": row[2],
            "itemid": row[3], "label": row[4],
        }
        write_neurosurgery_event(graph, proc_data, icu_stay_uri, icu_day_metadata)
        count += 1
    return count


def _write_icp_medication_events(
    conn, graph, stay_id, intime, window_end,
    icu_stay_uri, icu_day_metadata,
) -> int:
    """Query and write ICP medication events from inputevents within 48h window."""
    try:
        conn.execute("SELECT 1 FROM inputevents LIMIT 0")
    except duckdb.CatalogException:
        return 0

    # Build LIKE clauses for ICP medication keywords
    like_clauses = " OR ".join(
        f"LOWER(label) LIKE '%{kw}%'" for kw in ICP_MED_KEYWORDS
    )
    rows = conn.execute(
        f"""
        SELECT stay_id, starttime, endtime, label, amount, amountuom, rate, rateuom
        FROM inputevents
        WHERE stay_id = ?
          AND ({like_clauses})
          AND starttime >= ? AND starttime <= ?
        ORDER BY starttime
        """,
        [stay_id, intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        med_data = {
            "stay_id": row[0], "starttime": row[1], "endtime": row[2],
            "label": row[3], "amount": row[4], "amountuom": row[5],
            "rate": row[6], "rateuom": row[7],
        }
        write_icp_medication_event(graph, med_data, icu_stay_uri, icu_day_metadata)
        count += 1
    return count


def _write_lab_events(
    conn, graph, stay_data, intime, window_end,
    icu_stay_uri, icu_day_metadata, snomed_mapper,
) -> int:
    """Query and write lab events within 48h window (WLST-relevant labs only)."""
    itemid_list = ", ".join(str(i) for i in WLST_LAB_ITEMIDS)
    hadm_id = stay_data["hadm_id"]
    stay_id = stay_data["stay_id"]

    rows = conn.execute(
        f"""
        SELECT
            l.labevent_id, l.itemid, l.charttime,
            d.label, d.fluid, d.category,
            l.valuenum, l.valueuom, l.ref_range_lower, l.ref_range_upper
        FROM labevents l
        JOIN d_labitems d ON l.itemid = d.itemid
        WHERE l.hadm_id = ?
          AND l.itemid IN ({itemid_list})
          AND l.charttime >= ? AND l.charttime <= ?
          AND l.valuenum IS NOT NULL
        ORDER BY l.charttime
        """,
        [hadm_id, intime, window_end],
    ).fetchall()

    count = 0
    for row in rows:
        lab_data = {
            "labevent_id": row[0], "stay_id": stay_id, "itemid": row[1],
            "charttime": row[2], "label": row[3], "fluid": row[4],
            "category": row[5], "valuenum": row[6], "valueuom": row[7],
            "ref_range_lower": row[8], "ref_range_upper": row[9],
        }
        write_biomarker_event(
            graph, lab_data, icu_stay_uri, icu_day_metadata,
            snomed_mapper=snomed_mapper,
        )
        count += 1
    return count


def _write_diagnoses(
    conn, graph, hadm_id, admission_uri, snomed_mapper,
) -> int:
    """Write diagnosis events for an admission."""
    rows = conn.execute(
        """
        SELECT di.hadm_id, di.seq_num, di.icd_code, di.icd_version, d.long_title
        FROM diagnoses_icd di
        LEFT JOIN d_icd_diagnoses d
            ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version
        WHERE di.hadm_id = ?
        ORDER BY di.seq_num
        """,
        [hadm_id],
    ).fetchall()

    count = 0
    for row in rows:
        dx_data = {
            "hadm_id": row[0], "seq_num": row[1], "icd_code": row[2],
            "icd_version": row[3], "long_title": row[4],
        }
        write_diagnosis_event(graph, dx_data, admission_uri, snomed_mapper=snomed_mapper)
        count += 1
    return count


# ==================== Stage 2 attributes ====================


def _write_stage2_attributes(
    conn, graph, admission_uri, hadm_id, subject_id,
) -> None:
    """Add non-clinical confounder attributes to admission node (Stage 2)."""
    # Language
    row = conn.execute(
        "SELECT language FROM admissions WHERE hadm_id = ?", [hadm_id]
    ).fetchone()
    if row and row[0]:
        graph.add((
            admission_uri, MIMIC_NS.hasLanguage,
            Literal(row[0], datatype=XSD.string),
        ))

    # Hospital service
    try:
        rows = conn.execute(
            """
            SELECT curr_service FROM services
            WHERE hadm_id = ? ORDER BY transfertime LIMIT 1
            """,
            [hadm_id],
        ).fetchone()
        if rows and rows[0]:
            graph.add((
                admission_uri, MIMIC_NS.hasHospitalService,
                Literal(rows[0], datatype=XSD.string),
            ))
    except duckdb.CatalogException:
        pass

    # Transfer count
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM transfers WHERE hadm_id = ?", [hadm_id]
        ).fetchone()
        if row:
            graph.add((
                admission_uri, MIMIC_NS.hasTransferCount,
                Literal(row[0], datatype=XSD.integer),
            ))
    except duckdb.CatalogException:
        pass


# ==================== Query helpers ====================


def _query_patient_demographics(conn, subject_id):
    result = conn.execute(
        "SELECT subject_id, gender, anchor_age FROM patients WHERE subject_id = ?",
        [subject_id],
    ).fetchone()
    if result is None:
        return None
    return {"subject_id": result[0], "gender": result[1], "anchor_age": result[2]}


def _query_admission(conn, hadm_id):
    result = conn.execute(
        """
        SELECT hadm_id, subject_id, admittime, dischtime,
               admission_type, discharge_location
        FROM admissions WHERE hadm_id = ?
        """,
        [hadm_id],
    ).fetchone()
    if result is None:
        return None
    return {
        "hadm_id": result[0], "subject_id": result[1],
        "admittime": result[2], "dischtime": result[3],
        "admission_type": result[4], "discharge_location": result[5],
    }


def _query_icu_stay(conn, stay_id):
    result = conn.execute(
        """
        SELECT stay_id, hadm_id, subject_id, intime, outtime, los
        FROM icustays WHERE stay_id = ?
        """,
        [stay_id],
    ).fetchone()
    if result is None:
        return None
    return {
        "stay_id": result[0], "hadm_id": result[1], "subject_id": result[2],
        "intime": result[3], "outtime": result[4], "los": result[5],
    }
