# Readmission Predictor MIMIC

Temporal knowledge graph for hospital readmission prediction using MIMIC-IV data.

## Setup

```bash
make setup
cp .env.example .env
# Edit .env with your paths
```

## Testing

```bash
make test          # Run all tests
make test-unit     # Run unit tests only (fast, uses synthetic data)

# Run integration tests with real MIMIC-IV data
pytest -m integration -v
```

## Pipeline

```bash
make all  # Run full pipeline: ingest -> graph -> features -> train
```

## Layer 1A: MIMIC-IV Data Ingestion

The first step in the pipeline loads MIMIC-IV CSV files into a DuckDB database for efficient querying.

### Loaded Tables

| Table | Source | Description |
|-------|--------|-------------|
| patients | hosp/ | Patient demographics (subject_id, gender, anchor_age) |
| admissions | hosp/ | Hospital admissions (hadm_id, admittime, dischtime) |
| icustays | icu/ | ICU stays (stay_id, intime, outtime, los) |
| labevents | hosp/ | Lab test results |
| d_labitems | hosp/ | Lab item definitions |
| chartevents | icu/ | ICU charted observations |
| d_items | icu/ | Chart item definitions |
| microbiologyevents | hosp/ | Microbiology culture results |
| prescriptions | hosp/ | Medication prescriptions |
| diagnoses_icd | hosp/ | ICD diagnosis codes |
| d_icd_diagnoses | hosp/ | ICD diagnosis descriptions |
| procedures_icd | hosp/ | ICD procedure codes |
| d_icd_procedures | hosp/ | ICD procedure descriptions |

### Usage

```python
from pathlib import Path
from src.ingestion.mimic_loader import load_mimic_to_duckdb

# Load MIMIC-IV data
conn = load_mimic_to_duckdb(
    source_dir=Path("/path/to/mimiciv/3.1"),
    db_path=Path("data/processed/mimiciv.duckdb"),
)

# Query the data
result = conn.execute("""
    SELECT p.subject_id, a.hadm_id, i.stay_id
    FROM patients p
    INNER JOIN admissions a ON a.subject_id = p.subject_id
    INNER JOIN icustays i ON i.hadm_id = a.hadm_id
    LIMIT 5
""").fetchdf()
```

### Verification

Run the verification script to load and validate the data:

```bash
# Load and verify (takes a few minutes for large tables)
python scripts/verify_mimic_load.py

# Verify an existing database without reloading
python scripts/verify_mimic_load.py --skip-load

# Custom paths
python scripts/verify_mimic_load.py \
    --source /path/to/mimiciv/3.1 \
    --db data/processed/mimiciv.duckdb
```

The verification script checks:
- All 13 required tables exist
- Each table has non-zero row counts
- Key tables can be joined correctly (patients → admissions → icustays)
