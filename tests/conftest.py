import pytest
import duckdb
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from config.settings import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings with test paths overridden."""
    return Settings(
        mimic_iv_path=tmp_path / "mimic",
        duckdb_path=tmp_path / "test.duckdb",
        clinical_tkg_repo=tmp_path / "tkg",
    )


@pytest.fixture
def synthetic_duckdb(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """DuckDB with tiny synthetic MIMIC tables."""
    db_path = tmp_path / "test.duckdb"
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
            discharge_location VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO admissions VALUES
        (101, 1, '2150-01-15 08:00:00', '2150-01-20 14:00:00', 'EMERGENCY', 'HOME'),
        (102, 1, '2150-02-10 10:00:00', '2150-02-15 12:00:00', 'EMERGENCY', 'HOME'),
        (103, 2, '2151-03-01 06:00:00', '2151-03-10 16:00:00', 'ELECTIVE', 'SNF'),
        (104, 3, '2152-05-20 14:00:00', '2152-05-25 10:00:00', 'EMERGENCY', 'HOME'),
        (105, 4, '2150-07-01 09:00:00', '2150-07-05 11:00:00', 'URGENT', 'HOME'),
        (106, 5, '2151-04-10 12:00:00', '2151-04-20 08:00:00', 'EMERGENCY', 'HOSPICE')
    """)

    # Create icustays table
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
    # 3 stroke (I63.x), 1 epilepsy (G40.x), 1 cardiac (I25.x)
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

    yield conn
    conn.close()


@pytest.fixture
def base_ontology_graph() -> Graph:
    """Minimal rdflib Graph with base ontology loaded."""
    g = Graph()

    # Define namespaces
    CLINICAL = Namespace("http://example.org/clinical#")
    g.bind("clinical", CLINICAL)
    g.bind("owl", OWL)

    # Add base ontology classes
    g.add((CLINICAL.Patient, RDF.type, OWL.Class))
    g.add((CLINICAL.Admission, RDF.type, OWL.Class))
    g.add((CLINICAL.Diagnosis, RDF.type, OWL.Class))
    g.add((CLINICAL.ICUStay, RDF.type, OWL.Class))
    g.add((CLINICAL.TemporalEvent, RDF.type, OWL.Class))

    # Add base properties
    g.add((CLINICAL.hasAdmission, RDF.type, OWL.ObjectProperty))
    g.add((CLINICAL.hasDiagnosis, RDF.type, OWL.ObjectProperty))
    g.add((CLINICAL.hasICUStay, RDF.type, OWL.ObjectProperty))
    g.add((CLINICAL.occursAt, RDF.type, OWL.DatatypeProperty))

    return g
