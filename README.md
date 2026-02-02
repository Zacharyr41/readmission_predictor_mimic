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

The first step in the pipeline loads all MIMIC-IV CSV files into a DuckDB database for efficient querying. The loader automatically discovers and loads all tables from the `hosp/` and `icu/` subdirectories.

### Loaded Tables (31 total, ~886M rows)

**Hospital tables (hosp/):**
| Table | Rows | Description |
|-------|------|-------------|
| patients | 364K | Patient demographics |
| admissions | 546K | Hospital admissions |
| diagnoses_icd | 6.4M | ICD diagnosis codes |
| d_icd_diagnoses | 112K | ICD diagnosis descriptions |
| procedures_icd | 860K | ICD procedure codes |
| d_icd_procedures | 86K | ICD procedure descriptions |
| labevents | 158M | Lab test results |
| d_labitems | 1.6K | Lab item definitions |
| microbiologyevents | 4M | Microbiology cultures |
| prescriptions | 20M | Medication prescriptions |
| pharmacy | 18M | Pharmacy orders |
| emar | 43M | Electronic medication admin |
| emar_detail | 87M | EMAR details |
| poe | 52M | Provider order entry |
| poe_detail | 8.5M | POE details |
| hcpcsevents | 186K | HCPCS events |
| d_hcpcs | 89K | HCPCS definitions |
| drgcodes | 762K | Diagnosis-related groups |
| services | 593K | Hospital services |
| transfers | 2.4M | Patient transfers |
| omr | 7.8M | Outpatient medical records |
| provider | 42K | Provider info |

**ICU tables (icu/):**
| Table | Rows | Description |
|-------|------|-------------|
| icustays | 94K | ICU stays |
| chartevents | 433M | ICU charted observations |
| d_items | 4K | Chart item definitions |
| inputevents | 11M | IV/fluid inputs |
| outputevents | 5.4M | Output measurements |
| ingredientevents | 14M | IV ingredient events |
| procedureevents | 809K | ICU procedures |
| datetimeevents | 10M | Datetime events |
| caregiver | 18K | Caregiver info |

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
- All tables from hosp/ and icu/ are loaded (31 tables)
- Each table has non-zero row counts
- Key tables can be joined correctly (patients → admissions → icustays)
