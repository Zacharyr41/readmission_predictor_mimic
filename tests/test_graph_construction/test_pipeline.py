"""Tests for the graph construction pipeline orchestrator."""

import logging
from pathlib import Path

import duckdb
import pytest
from rdflib import Graph

from src.graph_construction.pipeline import build_graph
from src.graph_construction.ontology import MIMIC_NS, TIME_NS


# Path to ontology files
ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    """Temporary output path for RDF files."""
    return tmp_path / "test_output.rdf"


@pytest.fixture
def pipeline_db_path(tmp_path: Path) -> Path:
    """Create a DuckDB file with all required tables for pipeline testing."""
    db_path = tmp_path / "pipeline_test.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create patients table
    conn.execute("""
        CREATE TABLE patients (
            subject_id INTEGER PRIMARY KEY,
            gender VARCHAR,
            anchor_age INTEGER,
            anchor_year INTEGER,
            dod DATE
        )
    """)
    conn.execute("""
        INSERT INTO patients VALUES
        (1, 'M', 65, 2150, NULL),
        (2, 'F', 72, 2151, NULL),
        (3, 'M', 58, 2152, NULL),
        (4, 'F', 45, 2150, NULL),
        (5, 'M', 80, 2151, '2151-06-15')
    """)

    # Create admissions table
    conn.execute("""
        CREATE TABLE admissions (
            hadm_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            admittime TIMESTAMP,
            dischtime TIMESTAMP,
            admission_type VARCHAR,
            discharge_location VARCHAR,
            hospital_expire_flag INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO admissions VALUES
        (101, 1, '2150-01-15 08:00:00', '2150-01-20 14:00:00', 'EMERGENCY', 'HOME', 0),
        (102, 1, '2150-02-10 10:00:00', '2150-02-15 12:00:00', 'EMERGENCY', 'HOME', 0),
        (103, 2, '2151-03-01 06:00:00', '2151-03-10 16:00:00', 'ELECTIVE', 'SNF', 0),
        (104, 3, '2152-05-20 14:00:00', '2152-05-25 10:00:00', 'EMERGENCY', 'HOME', 0),
        (105, 4, '2150-07-01 09:00:00', '2150-07-05 11:00:00', 'URGENT', 'HOME', 0),
        (106, 5, '2151-04-10 12:00:00', '2151-04-20 08:00:00', 'EMERGENCY', 'HOSPICE', 1)
    """)

    # Create icustays table with longer stays to pass 24-hour filter
    conn.execute("""
        CREATE TABLE icustays (
            stay_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            hadm_id INTEGER,
            intime TIMESTAMP,
            outtime TIMESTAMP,
            los DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO icustays VALUES
        (1001, 1, 101, '2150-01-15 10:00:00', '2150-01-18 08:00:00', 2.9),
        (1002, 2, 103, '2151-03-02 08:00:00', '2151-03-08 12:00:00', 6.2),
        (1003, 5, 106, '2151-04-11 00:00:00', '2151-04-18 06:00:00', 7.25)
    """)

    # Create diagnoses_icd table
    conn.execute("""
        CREATE TABLE diagnoses_icd (
            subject_id INTEGER,
            hadm_id INTEGER,
            seq_num INTEGER,
            icd_code VARCHAR,
            icd_version INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO diagnoses_icd VALUES
        (1, 101, 1, 'I639', 10),
        (1, 102, 1, 'I634', 10),
        (2, 103, 1, 'I630', 10),
        (3, 104, 1, 'G409', 10),
        (4, 105, 1, 'I251', 10),
        (5, 106, 1, 'I639', 10)
    """)

    # d_labitems - lab item definitions
    conn.execute("""
        CREATE TABLE d_labitems (
            itemid INTEGER PRIMARY KEY,
            label VARCHAR,
            fluid VARCHAR,
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO d_labitems VALUES
        (50912, 'Creatinine', 'Blood', 'Chemistry'),
        (50971, 'Sodium', 'Blood', 'Chemistry'),
        (51265, 'Platelet Count', 'Blood', 'Hematology')
    """)

    # labevents - lab results
    conn.execute("""
        CREATE TABLE labevents (
            labevent_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            hadm_id INTEGER,
            stay_id INTEGER,
            itemid INTEGER,
            charttime TIMESTAMP,
            valuenum DOUBLE,
            valueuom VARCHAR,
            ref_range_lower DOUBLE,
            ref_range_upper DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO labevents VALUES
        (1, 1, 101, 1001, 50912, '2150-01-16 06:00:00', 1.2, 'mg/dL', 0.7, 1.3),
        (2, 1, 101, 1001, 50971, '2150-01-17 06:00:00', 140.0, 'mEq/L', 136.0, 145.0),
        (3, 2, 103, 1002, 50912, '2151-03-03 08:00:00', 0.9, 'mg/dL', 0.7, 1.3),
        (4, 5, 106, 1003, 50912, '2151-04-12 10:00:00', 1.5, 'mg/dL', 0.7, 1.3)
    """)

    # d_items - chartevents item definitions
    conn.execute("""
        CREATE TABLE d_items (
            itemid INTEGER PRIMARY KEY,
            label VARCHAR,
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO d_items VALUES
        (220045, 'Heart Rate', 'Routine Vital Signs'),
        (220179, 'Non Invasive Blood Pressure systolic', 'Routine Vital Signs'),
        (220180, 'Non Invasive Blood Pressure diastolic', 'Routine Vital Signs')
    """)

    # chartevents - vital signs
    conn.execute("""
        CREATE TABLE chartevents (
            subject_id INTEGER,
            hadm_id INTEGER,
            stay_id INTEGER,
            itemid INTEGER,
            charttime TIMESTAMP,
            valuenum DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO chartevents VALUES
        (1, 101, 1001, 220045, '2150-01-16 08:00:00', 78.0),
        (1, 101, 1001, 220179, '2150-01-16 08:00:00', 120.0),
        (2, 103, 1002, 220045, '2151-03-03 10:00:00', 82.0),
        (5, 106, 1003, 220045, '2151-04-12 12:00:00', 95.0)
    """)

    # microbiologyevents - culture results
    conn.execute("""
        CREATE TABLE microbiologyevents (
            microevent_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            hadm_id INTEGER,
            stay_id INTEGER,
            charttime TIMESTAMP,
            spec_type_desc VARCHAR,
            org_name VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO microbiologyevents VALUES
        (1, 1, 101, 1001, '2150-01-16 12:00:00', 'BLOOD CULTURE', 'STAPHYLOCOCCUS AUREUS'),
        (2, 2, 103, 1002, '2151-03-04 14:00:00', 'URINE', 'ESCHERICHIA COLI')
    """)

    # prescriptions - medications
    conn.execute("""
        CREATE TABLE prescriptions (
            subject_id INTEGER,
            hadm_id INTEGER,
            starttime TIMESTAMP,
            stoptime TIMESTAMP,
            drug VARCHAR,
            dose_val_rx DOUBLE,
            dose_unit_rx VARCHAR,
            route VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO prescriptions VALUES
        (1, 101, '2150-01-15 12:00:00', '2150-01-18 12:00:00', 'Vancomycin', 1000.0, 'mg', 'IV'),
        (2, 103, '2151-03-02 10:00:00', '2151-03-07 10:00:00', 'Ceftriaxone', 2000.0, 'mg', 'IV')
    """)

    # d_icd_diagnoses - diagnosis code descriptions
    conn.execute("""
        CREATE TABLE d_icd_diagnoses (
            icd_code VARCHAR,
            icd_version INTEGER,
            long_title VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO d_icd_diagnoses VALUES
        ('I639', 10, 'Cerebral infarction, unspecified'),
        ('I634', 10, 'Cerebral infarction due to embolism of cerebral arteries'),
        ('I630', 10, 'Cerebral infarction due to thrombosis of precerebral arteries'),
        ('G409', 10, 'Epilepsy, unspecified'),
        ('I251', 10, 'Atherosclerotic heart disease of native coronary artery')
    """)

    conn.close()
    return db_path


class TestBuildGraph:
    """Tests for the build_graph pipeline orchestrator."""

    def test_pipeline_produces_rdf_file(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """build_graph() creates an .rdf file at output_path."""
        graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
        )

        assert output_path.exists(), "RDF file should be created"
        assert output_path.stat().st_size > 0, "RDF file should not be empty"
        assert len(graph) > 0, "Graph should contain triples"

    def test_pipeline_filters_to_cohort(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """Only stroke patients (I63x) included, not epilepsy/cardiac."""
        graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
        )

        # Query for all patients in the graph
        query = """
        SELECT ?patient ?subject_id
        WHERE {
            ?patient rdf:type mimic:Patient ;
                     mimic:hasSubjectId ?subject_id .
        }
        """
        results = list(graph.query(query))
        subject_ids = {int(row[1]) for row in results}

        # Patients 1, 2 have stroke (I63x) diagnoses with ICU stays meeting criteria
        # Patient 3 has epilepsy (G40), Patient 4 has cardiac (I25) - should be excluded
        # Patient 5 has stroke but died in hospital (excluded from readmission labels)
        assert 1 in subject_ids or 2 in subject_ids, "At least one stroke patient should be included"
        assert 3 not in subject_ids, "Epilepsy patient should be excluded"
        assert 4 not in subject_ids, "Cardiac patient should be excluded"

    def test_pipeline_creates_all_entity_types(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """SPARQL finds Patient, HospitalAdmission, ICUStay, ICUDay, and event types."""
        graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
        )

        # Check for required entity types
        entity_types = [
            ("Patient", MIMIC_NS.Patient),
            ("HospitalAdmission", MIMIC_NS.HospitalAdmission),
            ("ICUStay", MIMIC_NS.ICUStay),
            ("ICUDay", MIMIC_NS.ICUDay),
            ("BioMarkerEvent", MIMIC_NS.BioMarkerEvent),
            ("ClinicalSignEvent", MIMIC_NS.ClinicalSignEvent),
            ("DiagnosisEvent", MIMIC_NS.DiagnosisEvent),
        ]

        for name, entity_type in entity_types:
            query = f"""
            SELECT (COUNT(?entity) AS ?count)
            WHERE {{
                ?entity rdf:type <{entity_type}> .
            }}
            """
            results = list(graph.query(query))
            count = int(results[0][0])
            assert count > 0, f"Graph should contain at least one {name}"

    def test_pipeline_includes_readmission_labels(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """Admissions have readmittedWithin30Days property."""
        graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
        )

        # Check for readmission labels
        query = """
        SELECT ?admission ?readmitted30
        WHERE {
            ?admission rdf:type mimic:HospitalAdmission ;
                       mimic:readmittedWithin30Days ?readmitted30 .
        }
        """
        results = list(graph.query(query))
        assert len(results) > 0, "Admissions should have readmittedWithin30Days property"

    def test_pipeline_includes_temporal_relations(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """Graph contains time:before triples."""
        graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
        )

        # Check for Allen temporal relations
        query = f"""
        SELECT (COUNT(*) AS ?count)
        WHERE {{
            ?a <{TIME_NS.before}> ?b .
        }}
        """
        results = list(graph.query(query))
        count = int(results[0][0])
        assert count > 0, "Graph should contain time:before relations"

    def test_pipeline_serialization_roundtrip(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """Serialize -> parse roundtrip preserves triple count."""
        original_graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
        )
        original_count = len(original_graph)

        # Parse the serialized file
        parsed_graph = Graph()
        parsed_graph.parse(output_path)
        parsed_count = len(parsed_graph)

        assert parsed_count == original_count, (
            f"Triple count should be preserved: original={original_count}, parsed={parsed_count}"
        )

    def test_pipeline_respects_patients_limit(
        self,
        pipeline_db_path: Path,
        output_path: Path,
    ):
        """patients_limit=1 produces exactly 1 Patient."""
        graph = build_graph(
            db_path=pipeline_db_path,
            ontology_dir=ONTOLOGY_DIR,
            output_path=output_path,
            icd_prefixes=["I63"],
            patients_limit=1,
        )

        # Count patients
        query = """
        SELECT (COUNT(?patient) AS ?count)
        WHERE {
            ?patient rdf:type mimic:Patient .
        }
        """
        results = list(graph.query(query))
        count = int(results[0][0])
        assert count == 1, f"Should have exactly 1 patient, got {count}"

    def test_pipeline_logs_progress(
        self,
        pipeline_db_path: Path,
        output_path: Path,
        caplog,
    ):
        """Captures 'Processing patient X/Y' log messages."""
        with caplog.at_level(logging.INFO):
            build_graph(
                db_path=pipeline_db_path,
                ontology_dir=ONTOLOGY_DIR,
                output_path=output_path,
                icd_prefixes=["I63"],
            )

        # Check for progress log messages
        log_messages = [record.message for record in caplog.records]
        progress_logs = [msg for msg in log_messages if "Processing patient" in msg]
        assert len(progress_logs) > 0, "Should log 'Processing patient X/Y' messages"
