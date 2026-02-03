"""Graph construction pipeline orchestrator for hospital readmission prediction.

This module provides the main build_graph() function that orchestrates the complete
DuckDB → RDF/OWL conversion for the neurology cohort.
"""

import logging
from pathlib import Path

import duckdb
from rdflib import Graph

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
    write_antibiotic_event,
    write_diagnosis_event,
)
from src.graph_construction.temporal.allen_relations import compute_allen_relations_for_patient
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
) -> Graph:
    """Build RDF graph from MIMIC-IV DuckDB database.

    Orchestrates the complete pipeline:
    1. Connect to DuckDB
    2. Create derived tables (age, readmission_labels)
    3. Select cohort based on ICD prefixes
    4. Initialize graph with ontologies
    5. Process each patient with their admissions, ICU stays, and events
    6. Compute Allen temporal relations
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

    # Initialize graph with ontologies
    graph = initialize_graph(ontology_dir)

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

    # Process each patient
    for idx, subject_id in enumerate(unique_patients, 1):
        logger.info(f"Processing patient {idx}/{total_patients} (subject_id={subject_id})")

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
        )

        # Update statistics
        for key, value in patient_stats.items():
            stats[key] = stats.get(key, 0) + value

    conn.close()

    # Serialize to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="xml")

    # Log summary
    logger.info(f"Graph construction complete: {len(graph)} triples")
    logger.info(f"  Patients: {stats['patients']}")
    logger.info(f"  Admissions: {stats['admissions']}")
    logger.info(f"  ICU Stays: {stats['icu_stays']}")
    logger.info(f"  ICU Days: {stats['icu_days']}")
    logger.info(f"  BioMarker Events: {stats['biomarkers']}")
    logger.info(f"  Clinical Sign Events: {stats['vitals']}")
    logger.info(f"  Microbiology Events: {stats['microbiology']}")
    logger.info(f"  Antibiotic Events: {stats['prescriptions']}")
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
            lab_events = _query_lab_events(conn, stay_id, biomarkers_limit)
            for lab_data in lab_events:
                write_biomarker_event(graph, lab_data, icu_stay_uri, icu_day_metadata)
                stats["biomarkers"] += 1

            # Write vital events (clinical signs)
            vital_events = _query_vital_events(conn, stay_id, vitals_limit)
            for vital_data in vital_events:
                write_clinical_sign_event(graph, vital_data, icu_stay_uri, icu_day_metadata)
                stats["vitals"] += 1

            # Write microbiology events
            micro_events = _query_microbiology_events(conn, stay_id, microbiology_limit)
            for micro_data in micro_events:
                write_microbiology_event(graph, micro_data, icu_stay_uri, icu_day_metadata)
                stats["microbiology"] += 1

            # Write prescriptions (antibiotics)
            prescriptions = _query_prescriptions(conn, hadm_id, prescriptions_limit)
            for rx_data in prescriptions:
                write_antibiotic_event(graph, rx_data, icu_stay_uri, icu_day_metadata)
                stats["prescriptions"] += 1

        # Write diagnoses for this admission
        diagnoses = _query_diagnoses(conn, hadm_id, diagnoses_limit)
        for dx_data in diagnoses:
            write_diagnosis_event(graph, dx_data, admission_uri)
            stats["diagnoses"] += 1

    # Link sequential admissions
    if len(admission_uris) > 1:
        link_sequential_admissions(graph, admission_uris)

    # Compute Allen relations for this patient
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
    conn: duckdb.DuckDBPyConnection, stay_id: int, limit: int
) -> list[dict]:
    """Query lab events for an ICU stay.

    Args:
        conn: DuckDB connection.
        stay_id: ICU stay ID.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of lab event dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    result = conn.execute(
        f"""
        SELECT
            l.labevent_id,
            l.stay_id,
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
        WHERE l.stay_id = ?
        ORDER BY l.charttime
        {limit_clause}
        """,
        [stay_id],
    ).fetchall()

    events = []
    for row in result:
        events.append({
            "labevent_id": row[0],
            "stay_id": row[1],
            "itemid": row[2],
            "charttime": row[3],
            "label": row[4],
            "fluid": row[5],
            "category": row[6],
            "valuenum": row[7],
            "valueuom": row[8],
            "ref_range_lower": row[9],
            "ref_range_upper": row[10],
        })

    return events


def _query_vital_events(
    conn: duckdb.DuckDBPyConnection, stay_id: int, limit: int
) -> list[dict]:
    """Query vital sign events for an ICU stay.

    Args:
        conn: DuckDB connection.
        stay_id: ICU stay ID.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of vital event dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

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
    conn: duckdb.DuckDBPyConnection, stay_id: int, limit: int
) -> list[dict]:
    """Query microbiology events for an ICU stay.

    Args:
        conn: DuckDB connection.
        stay_id: ICU stay ID.
        limit: Maximum number of events (0 = no limit).

    Returns:
        List of microbiology event dictionaries.
    """
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    result = conn.execute(
        f"""
        SELECT
            microevent_id,
            stay_id,
            charttime,
            spec_type_desc,
            org_name
        FROM microbiologyevents
        WHERE stay_id = ?
        ORDER BY charttime
        {limit_clause}
        """,
        [stay_id],
    ).fetchall()

    events = []
    for row in result:
        events.append({
            "microevent_id": row[0],
            "stay_id": row[1],
            "charttime": row[2],
            "spec_type_desc": row[3],
            "org_name": row[4],
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
