# Troubleshooting Guide

This guide helps diagnose and resolve common issues when running the readmission prediction pipeline.

## Installation Issues

### DuckDB Import Error

**Symptom:**
```
ImportError: cannot import name 'duckdb' from 'duckdb'
```

**Solution:**
```bash
# Reinstall DuckDB
pip uninstall duckdb
pip install duckdb>=1.1
```

### rdflib SPARQL Error

**Symptom:**
```
ImportError: No module named 'rdflib.plugins.sparql'
```

**Solution:**
```bash
# Install rdflib with all extras
pip install rdflib[all]>=7.0
```

### XGBoost Installation Failure (macOS)

**Symptom:**
```
error: command 'gcc' failed with exit status 1
```

**Solution:**
```bash
# Install via conda or use prebuilt wheels
conda install -c conda-forge xgboost

# Or use pip with binary wheels
pip install xgboost --only-binary :all:
```

### Neo4j Connection Error

**Symptom:**
```
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```

**Solution:**
1. Ensure Neo4j is running: `neo4j status`
2. Check connection settings in `.env`:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```
3. Neo4j is optional; the pipeline works without it.

## Data Loading Issues

### MIMIC-IV Path Not Found

**Symptom:**
```
FileNotFoundError: MIMIC-IV directory not found: /path/to/mimiciv
```

**Solution:**
1. Verify the path in `.env`:
   ```
   MIMIC_IV_PATH=/actual/path/to/mimiciv/3.1
   ```
2. Ensure the directory contains `hosp/` and `icu/` subdirectories
3. Check file permissions

### Table Not Found After Loading

**Symptom:**
```
duckdb.CatalogException: Table with name 'patients' does not exist
```

**Solution:**
1. Re-run ingestion:
   ```bash
   python -m src.main --run-ingestion
   ```
2. Check that CSV files exist in MIMIC-IV directory:
   ```bash
   ls /path/to/mimiciv/3.1/hosp/*.csv.gz
   ls /path/to/mimiciv/3.1/icu/*.csv.gz
   ```

### Memory Error During Ingestion

**Symptom:**
```
MemoryError: Unable to allocate memory for CSV loading
```

**Solution:**
1. DuckDB should handle large files via streaming, but for extremely large tables:
   ```bash
   # Increase swap space (Linux)
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```
2. Process in batches using `--patients-limit`:
   ```bash
   python -m src.main --patients-limit 1000
   ```

### Invalid CSV Format

**Symptom:**
```
duckdb.InvalidInputException: CSV parsing error
```

**Solution:**
1. DuckDB uses `ignore_errors=true` which skips problematic rows
2. Check MIMIC-IV version (should be 3.1)
3. Ensure files are not corrupted:
   ```bash
   gunzip -t /path/to/mimiciv/3.1/hosp/patients.csv.gz
   ```

## Graph Construction Issues

### Large Graph Memory Usage

**Symptom:**
```
MemoryError: RDF graph exceeds available memory
```

**Solution:**
1. Limit the number of patients:
   ```bash
   python -m src.main --patients-limit 100
   ```
2. Limit events per ICU stay:
   ```bash
   python -m src.main --biomarkers-limit 1000 --vitals-limit 1000
   ```
3. Skip Allen relations (major memory saver):
   ```bash
   python -m src.main --skip-allen
   ```

### Slow SPARQL Queries

**Symptom:**
Graph construction taking hours, especially during feature extraction.

**Solution:**
1. Skip Allen relations for faster graph build:
   ```bash
   python -m src.main --skip-allen
   ```
2. Reduce cohort size:
   ```bash
   python -m src.main --patients-limit 500
   ```
3. Graph queries are O(n) to O(n²); smaller graphs are exponentially faster.

### Missing Temporal Relations

**Symptom:**
`num_before_relations`, `num_during_relations`, `total_temporal_edges` all zero.

**Solution:**
1. Ensure `--skip-allen` flag was not used
2. Re-run with Allen relations:
   ```bash
   python -m src.main --run-ingestion  # If needed to rebuild
   ```
3. Check that events have valid timestamps:
   ```python
   from rdflib import Graph
   g = Graph()
   g.parse("data/processed/knowledge_graph.rdf")

   query = """
   SELECT (COUNT(?event) AS ?count)
   WHERE {
       ?event time:inXSDDateTimeStamp ?time .
   }
   """
   print(list(g.query(query)))
   ```

### Ontology Loading Failure

**Symptom:**
```
FileNotFoundError: Ontology file not found
```

**Solution:**
1. Ensure ontology files exist:
   ```bash
   ls ontology/definition/
   # Should show: base_ontology.rdf, extended_ontology.rdf
   ```
2. Check file permissions:
   ```bash
   chmod 644 ontology/definition/*.rdf
   ```

## Feature Extraction Issues

### Empty Feature Matrix

**Symptom:**
Feature matrix has 0 rows.

**Solution:**
1. Verify graph has data:
   ```python
   from rdflib import Graph
   g = Graph()
   g.parse("data/processed/knowledge_graph.rdf")
   print(f"Graph has {len(g)} triples")
   ```
2. Check cohort selection:
   ```bash
   python -m src.main --icd-codes I63 --patients-limit 10 --verbose
   ```
3. Verify ICD codes match data:
   ```sql
   -- In DuckDB
   SELECT DISTINCT LEFT(icd_code, 3), COUNT(*)
   FROM diagnoses_icd
   WHERE icd_code LIKE 'I6%'
   GROUP BY 1;
   ```

### NaN Values in Features

**Symptom:**
Model training fails due to NaN values.

**Solution:**
1. The pipeline fills missing values automatically, but check:
   ```python
   import pandas as pd
   df = pd.read_parquet("data/features/feature_matrix.parquet")
   print(df.isna().sum())
   ```
2. For specific columns with NaN:
   ```python
   # Manual fill
   df = df.fillna(0)
   df.to_parquet("data/features/feature_matrix.parquet")
   ```

### SPARQL Query Timeout

**Symptom:**
```
TimeoutError: SPARQL query timed out
```

**Solution:**
1. rdflib doesn't have built-in timeouts; the query may be inefficient
2. Reduce graph size:
   ```bash
   python -m src.main --patients-limit 100 --skip-allen
   ```
3. Check for SPARQL cartesian products in custom queries

## Model Training Issues

### Data Leakage Warning

**Symptom:**
```
UserWarning: Target column contains future information
```

**Cause:**
Diagnosis features (`icd_chapter_*`) are assigned at discharge and may contain information unavailable at prediction time.

**Solution:**
1. For strict temporal validity, exclude diagnosis features:
   ```python
   exclude_cols = ["hadm_id", "subject_id", "readmitted_30d", "readmitted_60d"]
   exclude_cols += [c for c in df.columns if c.startswith("icd_chapter_")]
   feature_cols = [c for c in df.columns if c not in exclude_cols]
   ```
2. Or accept the limitation and note it in results.

### Poor AUROC (< 0.55)

**Symptom:**
Model performs near random (AUROC ~0.5).

**Possible Causes:**
1. **Small sample size**: Need 100+ admissions for meaningful training
2. **Class imbalance**: Few positive cases
3. **Missing features**: Key features have all missing values

**Solutions:**
1. Increase cohort size:
   ```bash
   python -m src.main --patients-limit 500
   ```
2. Check class distribution:
   ```python
   df = pd.read_parquet("data/features/feature_matrix.parquet")
   print(df["readmitted_30d"].value_counts())
   ```
3. Verify features have variance:
   ```python
   print(df.std().sort_values())
   ```

### Insufficient Samples for Split

**Symptom:**
```
ValueError: The least populated class has only 1 member
```

**Solution:**
1. Need at least 2 samples per class for stratified splitting
2. Increase cohort:
   ```bash
   python -m src.main --patients-limit 200
   ```
3. Check positive/negative ratio:
   ```python
   df = pd.read_parquet("data/features/feature_matrix.parquet")
   print(f"Positive: {df['readmitted_30d'].sum()}")
   print(f"Negative: {len(df) - df['readmitted_30d'].sum()}")
   ```

### XGBoost Memory Error

**Symptom:**
```
MemoryError during XGBoost training
```

**Solution:**
1. Reduce features or samples:
   ```python
   # In custom training script
   X_train = X_train.iloc[:5000]  # Limit samples
   ```
2. Use sparse representation:
   ```python
   from scipy.sparse import csr_matrix
   X_sparse = csr_matrix(X_train.values)
   ```
3. Reduce XGBoost parameters:
   ```python
   model = XGBClassifier(
       n_estimators=50,  # Reduce from 100
       max_depth=4,      # Reduce from 6
   )
   ```

## Debugging Tips

### Enable Verbose Logging

```bash
python -m src.main --verbose
```

Or in Python:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Intermediate Artifacts

**DuckDB tables:**
```python
import duckdb
conn = duckdb.connect("data/processed/mimiciv.duckdb", read_only=True)
print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("SELECT COUNT(*) FROM patients").fetchone())
```

**RDF graph:**
```python
from rdflib import Graph
g = Graph()
g.parse("data/processed/knowledge_graph.rdf")

# Count by type
query = """
SELECT ?type (COUNT(?s) AS ?count)
WHERE { ?s rdf:type ?type }
GROUP BY ?type
ORDER BY DESC(?count)
"""
for row in g.query(query):
    print(f"{row[0]}: {row[1]}")
```

**Feature matrix:**
```python
import pandas as pd
df = pd.read_parquet("data/features/feature_matrix.parquet")
print(df.info())
print(df.describe())
```

### Run with Small Test Cohort

```bash
# Quick test with 10 patients
python -m src.main \
    --patients-limit 10 \
    --biomarkers-limit 100 \
    --vitals-limit 100 \
    --diagnoses-limit 50 \
    --skip-allen \
    --verbose
```

### Test Individual Stages

```bash
# Test ingestion only
python -c "
from src.ingestion.mimic_loader import load_mimic_to_duckdb
from pathlib import Path
conn = load_mimic_to_duckdb(
    Path('/path/to/mimiciv/3.1'),
    Path('data/processed/test.duckdb')
)
print(f'Tables: {conn.execute(\"SHOW TABLES\").fetchall()}')
"
```

## Performance Tuning

### Expected Processing Times

| Cohort Size | Ingestion | Graph Build | Allen Relations | Features | Training |
|-------------|-----------|-------------|-----------------|----------|----------|
| 10 patients | N/A | < 1 min | < 1 min | < 30 sec | < 10 sec |
| 100 patients | N/A | 1-2 min | 5-10 min | 1 min | < 30 sec |
| 500 patients | N/A | 5-10 min | 30-60 min | 2-3 min | < 1 min |
| 1000 patients | N/A | 10-20 min | 1-2 hours | 5 min | < 2 min |

*Note: Ingestion is one-time; graph build times include event limits.*

### Optimization Flags

```bash
# Maximum speed (skip expensive operations)
python -m src.main \
    --skip-ingestion \        # Use existing DuckDB
    --skip-allen \            # Skip O(n²) temporal relations
    --biomarkers-limit 500 \  # Limit events
    --vitals-limit 500
```

### Batch Processing

For very large cohorts, process in batches:

```bash
# Process patients 0-999
python -m src.main --patients-limit 1000

# Then manually continue with remaining patients
# (requires custom scripting to offset patient list)
```

## Getting Help

### Information to Include in Bug Reports

1. **Environment:**
   - Python version: `python --version`
   - Package versions: `pip freeze | grep -E "duckdb|rdflib|pandas|xgboost"`
   - OS: `uname -a`

2. **Error message:** Full traceback

3. **Command run:** Exact command with all flags

4. **Data info:**
   - MIMIC-IV version (3.1?)
   - Approximate cohort size
   - Any custom configuration

5. **Logs:** Run with `--verbose` and include output

### Filing Issues

Report bugs at: https://github.com/your-org/readmission_predictor_mimic/issues

Include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
