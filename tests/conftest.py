import pytest
import duckdb
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from config.settings import Settings


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
