"""Graph construction pipeline orchestrator for hospital readmission prediction.

This module provides the main build_graph() function that orchestrates the complete
DuckDB → RDF/OWL conversion for the neurology cohort.
"""

import logging
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
from rdflib import Graph

from src.graph_construction.disk_graph import (
    bind_namespaces,
    close_disk_graph,
    disk_graph,
    open_disk_graph,
)
from src.graph_construction.ontology import initialize_graph, MIMIC_NS
from src.graph_construction.patient_writer import (
    write_patient,
    write_admission,
    link_sequential_admissions,
)
from src.graph_construction.event_writers import (
    write_icu_stay,
    write_icu_days,
    write_biomarker_event,
    write_clinical_sign_event,
    write_microbiology_event,
    write_prescription_event,
    write_diagnosis_event,
)
from src.graph_construction.temporal.allen_relations import compute_allen_relations_for_patient
from src.graph_construction.terminology import SnomedMapper
from src.ingestion.derived_tables import (
    create_age_table,
    create_readmission_labels,
    select_neurology_cohort,
)


logger = logging.getLogger(__name__)


def build_graph(
    db_path: Path,
    ontology_dir: Path,
    output_path: Path,
    icd_prefixes: list[str],
    patients_limit: int = 0,
    biomarkers_limit: int = 0,
    vitals_limit: int = 0,
    microbiology_limit: int = 0,
    prescriptions_limit: int = 0,
    diagnoses_limit: int = 0,
    skip_allen_relations: bool = False,
    snomed_mappings_dir: Path | None = None,
    umls_api_key: str | None = None,
    n_workers: int | None = None,
) -> Graph:
    """Build RDF graph from MIMIC-IV DuckDB database.

    Orchestrates the complete pipeline:
    1. Connect to DuckDB
    2. Create derived tables (age, readmission_labels)
    3. Select cohort based on ICD prefixes
    4. Initialize graph with ontologies
    5. Process each patient with their admissions, ICU stays, and events
    6. Compute Allen temporal relations (optional)
    7. Serialize graph to output file

    Args:
        db_path: Path to DuckDB database file.
        ontology_dir: Path to directory containing ontology files.
        output_path: Path to write RDF/XML output file.
        icd_prefixes: List of ICD-10 prefixes to filter cohort (e.g., ["I63", "I61"]).
        patients_limit: Maximum number of patients to process (0 = no limit).
        biomarkers_limit: Maximum biomarker events per ICU stay (0 = no limit).
        vitals_limit: Maximum vital events per ICU stay (0 = no limit).
        microbiology_limit: Maximum microbiology events per ICU stay (0 = no limit).
        prescriptions_limit: Maximum prescriptions per admission (0 = no limit).
        diagnoses_limit: Maximum diagnoses per admission (0 = no limit).
        skip_allen_relations: If True, skip Allen relation computation (faster).
        n_workers: Number of parallel worker processes. None = auto-detect (min of
            cpu_count, 4). Set to 1 to force sequential processing.

    Returns:
        The constructed RDF graph.
    """
    # Create derived tables first (requires writable connection)
    write_conn = duckdb.connect(str(db_path))
    _ensure_derived_tables(write_conn)
    write_conn.close()

    # Now connect read-only for the rest of processing
    conn = duckdb.connect(str(db_path), read_only=True)

    # Select cohort
    cohort_df = select_neurology_cohort(conn, icd_prefixes)
    unique_patients = cohort_df["subject_id"].unique().tolist()

    # Apply patients limit
    if patients_limit > 0:
        unique_patients = unique_patients[:patients_limit]

    total_patients = len(unique_patients)
    logger.info(f"Processing {total_patients} patients from cohort")

    # Initialize disk-backed graph with ontologies
    store_path = output_path.parent / "oxigraph_store"
    graph = open_disk_graph(store_path)
    bind_namespaces(graph)

    # Copy ontology triples from the small in-memory graph
    ontology_graph = initialize_graph(ontology_dir)
    for triple in ontology_graph:
        graph.add(triple)

    # Initialize SNOMED mapper (optional)
    snomed_mapper = None
    if snomed_mappings_dir is not None and snomed_mappings_dir.exists():
        snomed_mapper = SnomedMapper(snomed_mappings_dir, umls_api_key=umls_api_key)
        stats_info = snomed_mapper.coverage_stats()
        logger.info(f"SNOMED mapper loaded: {stats_info}")

        # Ensure LOINC coverage via UMLS crosswalk (lazy-cached)
        if umls_api_key:
            all_loinc = _collect_loinc_codes(conn)
            if all_loinc:
                newly_resolved = snomed_mapper.ensure_loinc_coverage(all_loinc)
                logger.info(f"LOINC crosswalk: {newly_resolved} codes newly resolved")
    elif snomed_mappings_dir is not None:
        logger.warning(f"SNOMED mappings directory not found: {snomed_mappings_dir}")

    # Statistics
    stats = {
        "patients": 0,
        "admissions": 0,
        "icu_stays": 0,
        "icu_days": 0,
        "biomarkers": 0,
        "vitals": 0,
        "microbiology": 0,
        "prescriptions": 0,
        "diagnoses": 0,
        "allen_relations": 0,
    }

    # Determine number of workers
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4)

    if n_workers > 1 and total_patients > n_workers:
        # Parallel processing — per-patient tasks with main-process logging
        conn.close()  # Workers open their own connections
        logger.info(
            f"Using {n_workers} worker processes for {total_patients} patients"
        )

        worker_args = [
            (
                db_path, subject_id, cohort_df,
                biomarkers_limit, vitals_limit, microbiology_limit,
                prescriptions_limit, diagnoses_limit,
                skip_allen_relations, snomed_mappings_dir, umls_api_key,
            )
            for subject_id in unique_patients
        ]

        nt_paths: list[Path] = []
        with Pool(n_workers) as pool:
            for idx, (nt_path, patient_stats) in enumerate(
                pool.imap_unordered(_process_single_patient, worker_args), 1
            ):
                elapsed_entities = sum(patient_stats.values())
                logger.info(
                    f"Patient {idx}/{total_patients} complete "
                    f"(+{elapsed_entities} entities, "
                    f"nt={Path(nt_path).stat().st_size / 1024:.0f} KB)"
                )
                nt_paths.append(Path(nt_path))
                for key, value in patient_stats.items():
                    stats[key] = stats.get(key, 0) + value

        # Merge all per-patient NTriples files into the main graph
        logger.info(f"Merging {len(nt_paths)} patient sub-graphs...")
        merge_start = time.monotonic()
        for idx, nt_path in enumerate(nt_paths, 1):
            graph.parse(str(nt_path), format="nt")
            nt_path.unlink(missing_ok=True)
            if idx % 25 == 0 or idx == len(nt_paths):
                logger.info(
                    f"  Merged {idx}/{len(nt_paths)} files "
                    f"({len(graph)} triples, "
                    f"{time.monotonic() - merge_start:.1f}s)"
                )
        logger.info(f"All sub-graphs merged into main graph")
    else:
        # Sequential processing (single worker or few patients)
        if n_workers <= 1:
            logger.info(f"Sequential processing for {total_patients} patients")
        for idx, subject_id in enumerate(unique_patients, 1):
            logger.info(
                f"Processing patient {idx}/{total_patients} "
                f"(subject_id={subject_id})"
            )

            patient_stats = _process_patient(
                conn=conn,
                graph=graph,
                subject_id=subject_id,
                cohort_df=cohort_df,
                biomarkers_limit=biomarkers_limit,
                vitals_limit=vitals_limit,
                microbiology_limit=microbiology_limit,
                prescriptions_limit=prescriptions_limit,
                diagnoses_limit=diagnoses_limit,
                skip_allen_relations=skip_allen_relations,
                snomed_mapper=snomed_mapper,
            )

            for key, value in patient_stats.items():
                stats[key] = stats.get(key, 0) + value

        conn.close()

    # Serialize to output file (NTriples — fastest, grep-able)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing {len(graph)} triples to {output_path}...")
    t0 = time.monotonic()
    graph.serialize(destination=str(output_path), format="nt")
    ser_elapsed = time.monotonic() - t0
    output_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Serialized in {ser_elapsed:.1f}s ({output_mb:.1f} MB)")

    # Log summary
    logger.info(f"Graph construction complete: {len(graph)} triples")
    logger.info(f"  Patients: {stats['patients']}")
    logger.info(f"  Admissions: {stats['admissions']}")
    logger.info(f"  ICU Stays: {stats['icu_stays']}")
    logger.info(f"  ICU Days: {stats['icu_days']}")
    logger.info(f"  BioMarker Events: {stats['biomarkers']}")
    logger.info(f"  Clinical Sign Events: {stats['vitals']}")
    logger.info(f"  Microbiology Events: {stats['microbiology']}")
    logger.info(f"  Prescription Events: {stats['prescriptions']}")
    logger.info(f"  Diagnosis Events: {stats['diagnoses']}")
    logger.info(f"  Allen Relations: {stats['allen_relations']}")

    return graph


def _ensure_derived_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create derived tables if they don't exist.

    Args:
        conn: Writable DuckDB connection.
    """
    # Check if age table exists
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'age'"
    ).fetchone()
    if result[0] == 0:
        create_age_table(conn)

    # Check if readmission_labels table exists
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'readmission_labels'"
    ).fetchone()
    if result[0] == 0:
        create_readmission_labels(conn)


def _collect_loinc_codes(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Collect unique LOINC codes from d_labitems via the labitem mapping."""
    try:
        import json
        from pathlib import Path

        # Try to get LOINC codes from the labitem mapping file
        # This avoids needing a LOINC column in d_labitems (MIMIC doesn't have one)
        mapping_path = Path("data/mappings/labitem_to_snomed.json")
        if mapping_path.exists():
            with open(mapping_path) as f:
                data = json.load(f)
            data.pop("_metadata", None)
            codes = sorted({
                v["loinc"] for v in data.values()
                if isinstance(v, dict) and v.get("loinc")
            })
            return codes
    except Exception as e:
        logger.debug("Could not collect LOINC codes: %s", e)
    return []



def _process_single_patient(args: tuple) -> tuple[str, dict[str, int]]:
    """Process one patient in a worker process.

    Opens its own DuckDB connection and disk-backed graph, processes the
    patient, serializes to a temp NTriples file, and returns the path + stats.

    Returns:
        Tuple of (path to NTriples temp file, patient statistics dict).
    """
    (
        db_path, subject_id, cohort_df,
        biomarkers_limit, vitals_limit, microbiology_limit,
        prescriptions_limit, diagnoses_limit,
        skip_allen_relations, snomed_mappings_dir, umls_api_key,
    ) = args

    conn = duckdb.connect(str(db_path), read_only=True)

    snomed_mapper = None
    if snomed_mappings_dir is not None and Path(snomed_mappings_dir).exists():
        snomed_mapper = SnomedMapper(Path(snomed_mappings_dir), umls_api_key=umls_api_key)

    with disk_graph() as graph:
        bind_namespaces(graph)

        patient_stats = _process_patient(
            conn=conn,
            graph=graph,
            subject_id=subject_id,
            cohort_df=cohort_df,
            biomarkers_limit=biomarkers_limit,
            vitals_limit=vitals_limit,
            microbiology_limit=microbiology_limit,
            prescriptions_limit=prescriptions_limit,
            diagnoses_limit=diagnoses_limit,
            skip_allen_relations=skip_allen_relations,
            snomed_mapper=snomed_mapper,
        )

        fd, nt_path = tempfile.mkstemp(suffix=".nt", prefix="patient_")
        os.close(fd)
        graph.serialize(destination=nt_path, format="nt")

    conn.close()
    return nt_path, patient_stats


def _process_patient(
    conn: duckdb.DuckDBPyConnection,
    graph: Graph,
    subject_id: int,
    cohort_df,
    biomarkers_limit: int,
    vitals_limit: int,
    microbiology_limit: int,
    prescriptions_limit: int,
    diagnoses_limit: int,
    skip_allen_relations: bool = False,
    snomed_mapper=None,
) -> dict[str, int]:
    """Process a single patient and their associated data.

    Args:
        conn: DuckDB connection.
        graph: RDF graph to write to.
        subject_id: Patient subject ID.
        cohort_df: DataFrame with cohort membership (subject_id, hadm_id, stay_id).
        biomarkers_limit: Maximum biomarkers per ICU stay.
        vitals_limit: Maximum vitals per ICU stay.
        microbiology_limit: Maximum microbiology events per ICU stay.
        prescriptions_limit: Maximum prescriptions per admission.
        diagnoses_limit: Maximum diagnoses per admission.
        skip_allen_relations: If True, skip Allen relation computation (faster).

    Returns:
        Dictionary with counts of created entities.
    """
    stats = {
        "patients": 0,
        "admissions": 0,
        "icu_stays": 0,
        "icu_days": 0,
        "biomarkers": 0,
        "vitals": 0,
        "microbiology": 0,
        "prescriptions": 0,
        "diagnoses": 0,
        "allen_relations": 0,
    }

    # Get patient demographics
    patient_data = _query_patient_demographics(conn, subject_id)
    if patient_data is None:
        return stats

    # Write patient
    patient_uri = write_patient(graph, patient_data)
    stats["patients"] = 1

    # Get admissions with readmission labels
    admissions = _query_admissions_with_labels(conn, subject_id)
    admission_uris = []

    # Get eligible hadm_ids from cohort
    eligible_hadm_ids = set(
        cohort_df[cohort_df["subject_id"] == subject_id]["hadm_id"].tolist()
    )

    for adm_data in admissions:
        # Only process admissions in cohort
        if adm_data["hadm_id"] not in eligible_hadm_ids:
            continue

        # Write admission
        admission_uri = write_admission(graph, adm_data, patient_uri)
        admission_uris.append(admission_uri)
        stats["admissions"] += 1

        # Get ICU stays for this admission
        hadm_id = adm_data["hadm_id"]
        icu_stays = _query_icu_stays(conn, hadm_id)

        for stay_data in icu_stays:
            stay_id = stay_data["stay_id"]

            # Write ICU stay and days
            icu_stay_uri = write_icu_stay(graph, stay_data, admission_uri)
            stats["icu_stays"] += 1

            icu_day_metadata = write_icu_days(graph, stay_data, icu_stay_uri)
            stats["icu_days"] += len(icu_day_metadata)

            # Write lab events (biomarkers)
            lab_events = _query_lab_events(conn, stay_data, biomarkers_limit)
            for lab_data in lab_events:
                write_biomarker_event(graph, lab_data, icu_stay_uri, icu_day_metadata, snomed_mapper=snomed_mapper)
                stats["biomarkers"] += 1

            # Write vital events (clinical signs)
            vital_events = _query_vital_events(conn, stay_data, vitals_limit)
            for vital_data in vital_events:
                write_clinical_sign_event(graph, vital_data, icu_stay_uri, icu_day_metadata, snomed_mapper=snomed_mapper)
                stats["vitals"] += 1

            # Write microbiology events
            micro_events = _query_microbiology_events(conn, stay_data, microbiology_limit)
            for micro_data in micro_events:
                write_microbiology_event(graph, micro_data, icu_stay_uri, icu_day_metadata, snomed_mapper=snomed_mapper)
                stats["microbiology"] += 1

            # Write prescriptions
            prescriptions = _query_prescriptions(conn, hadm_id, prescriptions_limit)
            for rx_data in prescriptions:
                write_prescription_event(graph, rx_data, icu_stay_uri, icu_day_metadata, snomed_mapper=snomed_mapper)
                stats["prescriptions"] += 1

        # Write diagnoses for this admission
        diagnoses = _query_diagnoses(conn, hadm_id, diagnoses_limit)
        for dx_data in diagnoses:
            write_diagnosis_event(graph, dx_data, admission_uri, snomed_mapper=snomed_mapper)
            stats["diagnoses"] += 1

    # Link sequential admissions
    if len(admission_uris) > 1:
        link_sequential_admissions(graph, admission_uris)

    # Compute Allen relations for this patient (optional - can be slow for many events)
    if not skip_allen_relations:
        allen_count = compute_allen_relations_for_patient(graph, patient_uri)
        stats["allen_relations"] = allen_count

    return stats


def _query_patient_demographics(
    conn: duckdb.DuckDBPyConnection, subject_id: int
) -> dict | None:
    """Query patient demographics.

    Args:
        conn: DuckDB connection.
        subject_id: Patient subject ID.

    Returns:
        Dictionary with patient data or None if not found.
    """
    result = conn.execute(
        """
        SELECT subject_id, gender, anchor_age
        FROM patients
        WHERE subject_id = ?
        """,
        [subject_id],
    ).fetchone()

    if result is None:
        return None

    return {
        "subject_id": result[0],
        "gender": result[1],
        "anchor_age": result[2],
    }


def _query_admissions_with_labels(
    conn: duckdb.DuckDBPyConnection, subject_id: int
) -> list[dict]:
    """Query admissions with readmission labels for a patient.

    Args:
        conn: DuckDB connection.
        subject_id: Patient subject ID.

    Returns:
        List of admission dictionaries.
    """
    result = conn.execute(
        """
        SELECT
            a.hadm_id,
            a.subject_id,
            a.admittime,
            a.dischtime,
            a.admission_type,
            a.discharge_location,
            COALESCE(r.readmitted_30d, 0) AS readmitted_30d,
            COALESCE(r.readmitted_60d, 0) AS readmitted_60d
        FROM admissions a
        LEFT JOIN readmission_labels r ON a.hadm_id = r.hadm_id
        WHERE a.subject_id = ?
        ORDER BY a.admittime
        """,
        [subject_id],
    ).fetchall()

    admissions = []
    for row in result:
        admissions.append({
            "hadm_id": row[0],
            "subject_id": row[1],
            "admittime": row[2],
            "dischtime": row[3],
            "admission_type": row[4],
            "discharge_location": row[5],
            "readmitted_30d": bool(row[6]),
            "readmitted_60d": bool(row[7]),
        })

    return admissions


def _query_icu_stays(conn: duckdb.DuckDBPyConnection, hadm_id: int) -> list[dict]:
    """Query ICU stays for an admission.

    Args:
        conn: DuckDB connection.
        hadm_id: Hospital admission ID.

    Returns:
        List of ICU stay dictionaries.
    """
    result = conn.execute(
        """
        SELECT stay_id, hadm_id, subject_id, intime, outtime, los
        FROM icustays
        WHERE hadm_id = ?
        ORDER BY intime
        """,
        [hadm_id],
    ).fetchall()

    stays = []
    for row in result:
        stays.append({
            "stay_id": row[0],
            "hadm_id": row[1],
            "subject_id": row[2],
            "intime": row[3],
            "outtime": row[4],
            "los": row[5],
        })

    return stays


def _query_lab_events(
    conn: duckdb.DuckDBPyConnection, stay_data: dict, limit: int
) -> list[dict]:
    """Query lab events for an ICU stay.

    Filters labevents by hadm_id and charttime within ICU stay window.
    Note: MIMIC-IV labevents doesn't have stay_id, so we filter by time window.

    Args:
        conn: DuckDB connection.
        stay_data: ICU stay dictionary with hadm_id, stay_id, intime, outtime.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of lab event dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""
    hadm_id = stay_data["hadm_id"]
    stay_id = stay_data["stay_id"]
    intime = stay_data["intime"]
    outtime = stay_data["outtime"]

    result = conn.execute(
        f"""
        SELECT
            l.labevent_id,
            l.itemid,
            l.charttime,
            d.label,
            d.fluid,
            d.category,
            l.valuenum,
            l.valueuom,
            l.ref_range_lower,
            l.ref_range_upper
        FROM labevents l
        JOIN d_labitems d ON l.itemid = d.itemid
        WHERE l.hadm_id = ?
          AND l.charttime >= ?
          AND l.charttime <= ?
          AND l.valuenum IS NOT NULL
        ORDER BY l.charttime
        {limit_clause}
        """,
        [hadm_id, intime, outtime],
    ).fetchall()

    events = []
    for row in result:
        events.append({
            "labevent_id": row[0],
            "stay_id": stay_id,  # Add stay_id for downstream use
            "itemid": row[1],
            "charttime": row[2],
            "label": row[3],
            "fluid": row[4],
            "category": row[5],
            "valuenum": row[6],
            "valueuom": row[7],
            "ref_range_lower": row[8],
            "ref_range_upper": row[9],
        })

    return events


def _query_vital_events(
    conn: duckdb.DuckDBPyConnection, stay_data: dict, limit: int
) -> list[dict]:
    """Query vital sign events for an ICU stay.

    Args:
        conn: DuckDB connection.
        stay_data: ICU stay dictionary with stay_id.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of vital event dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""
    stay_id = stay_data["stay_id"]

    result = conn.execute(
        f"""
        SELECT
            c.stay_id,
            c.itemid,
            c.charttime,
            d.label,
            d.category,
            c.valuenum
        FROM chartevents c
        JOIN d_items d ON c.itemid = d.itemid
        WHERE c.stay_id = ?
          AND c.valuenum IS NOT NULL
        ORDER BY c.charttime
        {limit_clause}
        """,
        [stay_id],
    ).fetchall()

    events = []
    for row in result:
        events.append({
            "stay_id": row[0],
            "itemid": row[1],
            "charttime": row[2],
            "label": row[3],
            "category": row[4],
            "valuenum": row[5],
        })

    return events


def _query_microbiology_events(
    conn: duckdb.DuckDBPyConnection, stay_data: dict, limit: int
) -> list[dict]:
    """Query microbiology events for an ICU stay.

    Filters microbiologyevents by hadm_id and charttime within ICU stay window.
    Note: MIMIC-IV microbiologyevents doesn't have stay_id.

    Args:
        conn: DuckDB connection.
        stay_data: ICU stay dictionary with hadm_id, stay_id, intime, outtime.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of microbiology event dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""
    hadm_id = stay_data["hadm_id"]
    stay_id = stay_data["stay_id"]
    intime = stay_data["intime"]
    outtime = stay_data["outtime"]

    result = conn.execute(
        f"""
        SELECT
            microevent_id,
            charttime,
            spec_type_desc,
            org_name
        FROM microbiologyevents
        WHERE hadm_id = ?
          AND charttime >= ?
          AND charttime <= ?
        ORDER BY charttime
        {limit_clause}
        """,
        [hadm_id, intime, outtime],
    ).fetchall()

    events = []
    for row in result:
        events.append({
            "microevent_id": row[0],
            "stay_id": stay_id,  # Add stay_id for downstream use
            "charttime": row[1],
            "spec_type_desc": row[2],
            "org_name": row[3],
        })

    return events


def _query_prescriptions(
    conn: duckdb.DuckDBPyConnection, hadm_id: int, limit: int
) -> list[dict]:
    """Query prescriptions for an admission.

    Args:
        conn: DuckDB connection.
        hadm_id: Hospital admission ID.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of prescription dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    result = conn.execute(
        f"""
        SELECT
            hadm_id,
            drug,
            starttime,
            stoptime,
            dose_val_rx,
            dose_unit_rx,
            route
        FROM prescriptions
        WHERE hadm_id = ?
        ORDER BY starttime
        {limit_clause}
        """,
        [hadm_id],
    ).fetchall()

    prescriptions = []
    for row in result:
        prescriptions.append({
            "hadm_id": row[0],
            "drug": row[1],
            "starttime": row[2],
            "stoptime": row[3],
            "dose_val_rx": row[4],
            "dose_unit_rx": row[5],
            "route": row[6],
        })

    return prescriptions


def _query_diagnoses(
    conn: duckdb.DuckDBPyConnection, hadm_id: int, limit: int
) -> list[dict]:
    """Query diagnoses for an admission.

    Args:
        conn: DuckDB connection.
        hadm_id: Hospital admission ID.
        limit: Maximum number of diagnoses (0 = no limit).

    Returns:
        List of diagnosis dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    result = conn.execute(
        f"""
        SELECT
            di.hadm_id,
            di.seq_num,
            di.icd_code,
            di.icd_version,
            d.long_title
        FROM diagnoses_icd di
        LEFT JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version
        WHERE di.hadm_id = ?
        ORDER BY di.seq_num
        {limit_clause}
        """,
        [hadm_id],
    ).fetchall()

    diagnoses = []
    for row in result:
        diagnoses.append({
            "hadm_id": row[0],
            "seq_num": row[1],
            "icd_code": row[2],
            "icd_version": row[3],
            "long_title": row[4],
        })

    return diagnoses
