import pytest
import duckdb
from datetime import datetime
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from config.settings import Settings
from src.graph_construction.ontology import initialize_graph


# Path to real MIMIC-IV data
REAL_MIMIC_IV_PATH = Path("/Users/zacharyrothstein/Code/NeuroResearch/physionet.org/files/mimiciv/3.1")


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
def synthetic_duckdb_with_events(synthetic_duckdb: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Extend synthetic_duckdb with clinical event tables."""
    conn = synthetic_duckdb

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
        (2, 1, 101, 1001, 50971, '2150-01-16 06:00:00', 140.0, 'mEq/L', 136.0, 145.0),
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

    return conn


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


@pytest.fixture
def synthetic_mimic_dir(tmp_path: Path) -> Path:
    """Create synthetic MIMIC-IV directory structure with CSV files for testing."""
    hosp_dir = tmp_path / "hosp"
    icu_dir = tmp_path / "icu"
    hosp_dir.mkdir()
    icu_dir.mkdir()

    # Synthetic CSV data for hosp/ tables
    hosp_files = {
        "patients.csv": """\
subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod
1,M,65,2150,2017 - 2019,
2,F,72,2151,2017 - 2019,
3,M,58,2152,2017 - 2019,
4,F,45,2150,2017 - 2019,
5,M,80,2151,2017 - 2019,2151-06-15
""",
        "admissions.csv": """\
subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,discharge_location,hospital_expire_flag
1,101,2150-01-15 08:00:00,2150-01-20 14:00:00,,EMERGENCY,HOME,0
1,102,2150-02-10 10:00:00,2150-02-15 12:00:00,,EMERGENCY,HOME,0
2,103,2151-03-01 06:00:00,2151-03-10 16:00:00,,ELECTIVE,SNF,0
3,104,2152-05-20 14:00:00,2152-05-25 10:00:00,,EMERGENCY,HOME,0
4,105,2150-07-01 09:00:00,2150-07-05 11:00:00,,URGENT,HOME,0
""",
        "labevents.csv": """\
labevent_id,subject_id,hadm_id,itemid,value,valuenum,charttime,valueuom,ref_range_lower,ref_range_upper
1,1,101,50912,1.2,1.2,2150-01-15 10:00:00,mg/dL,0.7,1.3
2,1,101,50971,140,140.0,2150-01-15 10:00:00,mEq/L,136,145
3,2,103,50912,0.9,0.9,2151-03-02 08:00:00,mg/dL,0.7,1.3
""",
        "d_labitems.csv": """\
itemid,label,fluid,category
50912,Creatinine,Blood,Chemistry
50971,Sodium,Blood,Chemistry
51265,Platelet Count,Blood,Hematology
""",
        "microbiologyevents.csv": """\
microevent_id,subject_id,hadm_id,charttime,spec_type_desc,org_name,ab_name,interpretation
1,1,101,2150-01-16 12:00:00,BLOOD CULTURE,STAPHYLOCOCCUS AUREUS,VANCOMYCIN,S
2,2,103,2151-03-03 14:00:00,URINE,ESCHERICHIA COLI,CIPROFLOXACIN,R
3,3,104,2152-05-21 08:00:00,BLOOD CULTURE,,AMPICILLIN,
""",
        "prescriptions.csv": """\
subject_id,hadm_id,starttime,stoptime,drug,dose_val_rx,dose_unit_rx,route
1,101,2150-01-15 10:00:00,2150-01-20 14:00:00,Vancomycin,1000,mg,IV
2,103,2151-03-01 08:00:00,2151-03-10 16:00:00,Metformin,500,mg,PO
3,104,2152-05-20 16:00:00,2152-05-25 10:00:00,Aspirin,81,mg,PO
""",
        "diagnoses_icd.csv": """\
subject_id,hadm_id,seq_num,icd_code,icd_version
1,101,1,I639,10
1,102,1,I634,10
2,103,1,I630,10
3,104,1,G409,10
4,105,1,I251,10
""",
        "d_icd_diagnoses.csv": """\
icd_code,icd_version,long_title
I639,10,Cerebral infarction unspecified
I634,10,Cerebral infarction due to embolism of cerebral arteries
I630,10,Cerebral infarction due to thrombosis of precerebral arteries
G409,10,Epilepsy unspecified
I251,10,Atherosclerotic heart disease of native coronary artery
""",
        "procedures_icd.csv": """\
subject_id,hadm_id,seq_num,icd_code,icd_version
1,101,1,0016070,10
2,103,1,02H63JZ,10
3,104,1,00JU0ZZ,10
""",
        "d_icd_procedures.csv": """\
icd_code,icd_version,long_title
0016070,10,Bypass Cerebral Ventricle to Nasopharynx with Autologous Tissue
02H63JZ,10,Insertion of Pacemaker Lead into Right Atrium Percutaneous Approach
00JU0ZZ,10,Inspection of Spinal Canal Open Approach
""",
    }

    # Synthetic CSV data for icu/ tables
    icu_files = {
        "icustays.csv": """\
subject_id,hadm_id,stay_id,intime,outtime,los
1,101,1001,2150-01-15 10:00:00,2150-01-18 08:00:00,2.9
2,103,1002,2151-03-02 08:00:00,2151-03-08 12:00:00,6.2
5,106,1003,2151-04-11 00:00:00,2151-04-18 06:00:00,7.25
""",
        "chartevents.csv": """\
subject_id,hadm_id,stay_id,itemid,value,valuenum,charttime
1,101,1001,220045,78,78.0,2150-01-15 12:00:00
1,101,1001,220179,120,120.0,2150-01-15 12:00:00
2,103,1002,220045,82,82.0,2151-03-02 10:00:00
""",
        "d_items.csv": """\
itemid,label,category,unitname,param_type
220045,Heart Rate,Routine Vital Signs,bpm,Numeric
220179,Non Invasive Blood Pressure systolic,Routine Vital Signs,mmHg,Numeric
220180,Non Invasive Blood Pressure diastolic,Routine Vital Signs,mmHg,Numeric
""",
    }

    # Write hosp files
    for filename, content in hosp_files.items():
        (hosp_dir / filename).write_text(content)

    # Write icu files
    for filename, content in icu_files.items():
        (icu_dir / filename).write_text(content)

    return tmp_path


@pytest.fixture
def real_mimic_dir() -> Path:
    """Path to real MIMIC-IV data for integration tests."""
    if not REAL_MIMIC_IV_PATH.exists():
        pytest.skip(f"MIMIC-IV data not found at {REAL_MIMIC_IV_PATH}")
    return REAL_MIMIC_IV_PATH


# Path to ontology files
ONTOLOGY_DIR = Path(__file__).parent.parent / "ontology" / "definition"


@pytest.fixture
def graph_with_ontology() -> Graph:
    """Graph with both base and extended ontologies loaded."""
    return initialize_graph(ONTOLOGY_DIR)


@pytest.fixture
def sample_patient_data() -> dict:
    """Sample patient data for testing."""
    return {"subject_id": 100, "gender": "M", "anchor_age": 65}


@pytest.fixture
def sample_admission_data() -> dict:
    """Sample admission data for testing."""
    return {
        "hadm_id": 200,
        "subject_id": 100,
        "admittime": datetime(2150, 1, 1, 8, 0, 0),
        "dischtime": datetime(2150, 1, 10, 14, 0, 0),
        "admission_type": "EMERGENCY",
        "discharge_location": "HOME",
        "readmitted_30d": True,
        "readmitted_60d": True,
    }


@pytest.fixture
def patient_with_multiple_admissions() -> tuple[dict, list[dict]]:
    """Patient with 2 admissions for followedBy test."""
    patient = {"subject_id": 300, "gender": "F", "anchor_age": 55}
    admissions = [
        {
            "hadm_id": 301,
            "subject_id": 300,
            "admittime": datetime(2150, 1, 1, 8, 0, 0),
            "dischtime": datetime(2150, 1, 5, 14, 0, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "readmitted_30d": True,
            "readmitted_60d": True,
        },
        {
            "hadm_id": 302,
            "subject_id": 300,
            "admittime": datetime(2150, 1, 20, 10, 0, 0),
            "dischtime": datetime(2150, 1, 25, 12, 0, 0),
            "admission_type": "URGENT",
            "discharge_location": "SNF",
            "readmitted_30d": False,
            "readmitted_60d": False,
        },
    ]
    return patient, admissions


# ==================== Layer 2C: Clinical Event Fixtures ====================


@pytest.fixture
def sample_icu_stay_data() -> dict:
    """Sample ICU stay data for testing."""
    return {
        "stay_id": 300,
        "hadm_id": 200,
        "subject_id": 100,
        "intime": datetime(2150, 1, 1, 8, 0, 0),
        "outtime": datetime(2150, 1, 4, 14, 0, 0),
        "los": 3.25,
    }


@pytest.fixture
def sample_biomarker_data() -> dict:
    """Sample biomarker (lab) data for testing."""
    return {
        "labevent_id": 1001,
        "stay_id": 300,
        "itemid": 50912,
        "charttime": datetime(2150, 1, 2, 6, 0, 0),  # Day 2
        "label": "Creatinine",
        "fluid": "Blood",
        "category": "Chemistry",
        "valuenum": 1.2,
        "valueuom": "mg/dL",
        "ref_range_lower": 0.7,
        "ref_range_upper": 1.3,
    }


@pytest.fixture
def sample_clinical_sign_data() -> dict:
    """Sample clinical sign (vital) data for testing."""
    return {
        "stay_id": 300,
        "itemid": 220045,
        "charttime": datetime(2150, 1, 1, 12, 0, 0),  # Day 1
        "label": "Heart Rate",
        "category": "Routine Vital Signs",
        "valuenum": 78.0,
    }


@pytest.fixture
def sample_antibiotic_data() -> dict:
    """Sample antibiotic prescription data for testing."""
    return {
        "hadm_id": 200,
        "stay_id": 300,
        "drug": "Vancomycin",
        "starttime": datetime(2150, 1, 1, 10, 0, 0),
        "stoptime": datetime(2150, 1, 3, 10, 0, 0),
        "dose_val_rx": 1000.0,
        "dose_unit_rx": "mg",
        "route": "IV",
    }


@pytest.fixture
def sample_diagnosis_data() -> dict:
    """Sample diagnosis data for testing."""
    return {
        "hadm_id": 200,
        "seq_num": 1,
        "icd_code": "I63.0",
        "icd_version": 10,
        "long_title": "Cerebral infarction due to thrombosis of precerebral arteries",
    }


@pytest.fixture
def sample_comorbidity_data() -> dict:
    """Sample comorbidity data for testing."""
    return {
        "subject_id": 100,
        "name": "diabetes",
        "value": True,
    }


@pytest.fixture
def full_patient_with_events() -> dict:
    """Complete patient data with all event types for SPARQL integration test.

    Patient: 1 admission, 1 ICU stay (3 days), 2 biomarkers, 1 vital, 1 antibiotic, 1 diagnosis.
    """
    return {
        "patient": {"subject_id": 500, "gender": "M", "anchor_age": 70},
        "admission": {
            "hadm_id": 501,
            "subject_id": 500,
            "admittime": datetime(2150, 3, 1, 6, 0, 0),
            "dischtime": datetime(2150, 3, 10, 12, 0, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "readmitted_30d": False,
            "readmitted_60d": False,
        },
        "icu_stay": {
            "stay_id": 502,
            "hadm_id": 501,
            "subject_id": 500,
            "intime": datetime(2150, 3, 1, 8, 0, 0),
            "outtime": datetime(2150, 3, 4, 8, 0, 0),  # Exactly 3 days
            "los": 3.0,
        },
        "biomarkers": [
            {
                "labevent_id": 5001,
                "stay_id": 502,
                "itemid": 50912,
                "charttime": datetime(2150, 3, 1, 10, 0, 0),  # Day 1
                "label": "Creatinine",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 1.1,
                "valueuom": "mg/dL",
                "ref_range_lower": 0.7,
                "ref_range_upper": 1.3,
            },
            {
                "labevent_id": 5002,
                "stay_id": 502,
                "itemid": 50971,
                "charttime": datetime(2150, 3, 2, 8, 0, 0),  # Day 2
                "label": "Sodium",
                "fluid": "Blood",
                "category": "Chemistry",
                "valuenum": 140.0,
                "valueuom": "mEq/L",
                "ref_range_lower": 136.0,
                "ref_range_upper": 145.0,
            },
        ],
        "vitals": [
            {
                "stay_id": 502,
                "itemid": 220045,
                "charttime": datetime(2150, 3, 1, 12, 0, 0),  # Day 1
                "label": "Heart Rate",
                "category": "Routine Vital Signs",
                "valuenum": 82.0,
            },
        ],
        "antibiotics": [
            {
                "hadm_id": 501,
                "stay_id": 502,
                "drug": "Vancomycin",
                "starttime": datetime(2150, 3, 1, 10, 0, 0),
                "stoptime": datetime(2150, 3, 3, 10, 0, 0),
                "dose_val_rx": 1000.0,
                "dose_unit_rx": "mg",
                "route": "IV",
            },
        ],
        "diagnoses": [
            {
                "hadm_id": 501,
                "seq_num": 1,
                "icd_code": "I63.0",
                "icd_version": 10,
                "long_title": "Cerebral infarction due to thrombosis of precerebral arteries",
            },
        ],
    }
