# Temporal Knowledge Graph-Based Hospital Readmission Prediction

A machine learning pipeline that transforms MIMIC-IV electronic health record data into an OWL-Time-compliant RDF knowledge graph, extracts temporal and clinical features, and trains models to predict 30-day hospital readmission for neurology patients.

## Overview

This system implements a 5-layer pipeline for hospital readmission prediction:

1. **Ingestion**: Load MIMIC-IV CSV files into DuckDB for efficient querying
2. **Graph Construction**: Build an RDF knowledge graph with OWL-Time temporal modeling
3. **Graph Analysis**: Generate structure analysis reports and metrics
4. **Feature Extraction**: Extract tabular and graph-based features via SPARQL
5. **Prediction**: Train and evaluate Logistic Regression and XGBoost models

The pipeline focuses on stroke patients (ICD-10 codes I60, I61, I63) but can be configured for other cohorts.

## Key Features

- **Temporal Knowledge Graph**: Clinical events modeled as OWL-Time intervals with Allen temporal relations (before, during, overlaps, meets, starts, finishes)
- **Cohort Filtering**: Configurable ICD-based cohort selection with age and ICU stay duration criteria
- **Rich Feature Engineering**: Demographics, lab aggregates, vital statistics, medication counts, diagnosis features, and graph structure metrics
- **Class Imbalance Handling**: Automatic class weighting for both Logistic Regression and XGBoost
- **Patient-Level Splitting**: Prevents data leakage by keeping all admissions for a patient in the same split
- **Comprehensive Reporting**: Markdown evaluation reports with AUROC, AUPRC, confusion matrices, and feature importance

## Architecture

```
MIMIC-IV CSVs → Layer 1 (DuckDB) → Layer 2 (RDF Graph) → Layer 3 (Analysis)
                                         ↓
                              Layer 4 (Features) → Layer 5 (Models)
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture diagrams.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-org/readmission_predictor_mimic.git
cd readmission_predictor_mimic
make setup

# 2. Configure environment
cp .env.example .env
# Edit .env with your MIMIC-IV path

# 3. Run the full pipeline (small cohort for testing)
python -m src.main --patients-limit 50 --skip-allen --verbose

# 4. View results
cat outputs/reports/evaluation_xgb.md
```

## Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **MIMIC-IV Access**: Credentialed access to [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/)
- **System Requirements**:
  - 16GB RAM minimum (32GB recommended for full cohort)
  - 50GB disk space for DuckDB database
- **Optional**: Neo4j 5.x for graph visualization

## Installation

### Using uv (recommended)

```bash
make setup
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,graph]"
```

### Dependencies

Core dependencies:
- `duckdb>=1.1` - In-process SQL database
- `rdflib>=7.0` - RDF graph manipulation
- `pandas>=2.0` - Data processing
- `scikit-learn>=1.4` - Machine learning
- `xgboost>=2.0` - Gradient boosting
- `pydantic>=2.0` - Configuration validation

Optional:
- `networkx>=3.0` - Graph algorithms
- `neo4j>=5.0` - Neo4j integration

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MIMIC_IV_PATH` | Path to MIMIC-IV directory containing `hosp/` and `icu/` | Required |
| `DUCKDB_PATH` | Path to DuckDB database file | `data/processed/mimiciv.duckdb` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |

### Pipeline Settings

Settings can be configured via environment variables or CLI arguments:

| Setting | Environment Variable | CLI Flag | Default | Description |
|---------|---------------------|----------|---------|-------------|
| Cohort ICD codes | `COHORT_ICD_CODES` | `--icd-codes` | `["I63", "I61", "I60"]` | ICD-10 prefixes for cohort selection |
| Readmission window | `READMISSION_WINDOW_DAYS` | - | `30` | Days for readmission label |
| Patient limit | `PATIENTS_LIMIT` | `--patients-limit` | `0` (unlimited) | Max patients to process |
| Biomarkers limit | `BIOMARKERS_LIMIT` | `--biomarkers-limit` | `0` (unlimited) | Max biomarker events per ICU stay |
| Vitals limit | `VITALS_LIMIT` | `--vitals-limit` | `0` (unlimited) | Max vital events per ICU stay |
| Diagnoses limit | `DIAGNOSES_LIMIT` | `--diagnoses-limit` | `0` (unlimited) | Max diagnoses per admission |
| Skip Allen relations | `SKIP_ALLEN_RELATIONS` | `--skip-allen` | `false` | Skip expensive temporal relation computation |

## Usage

### Full Pipeline

```bash
# Run complete pipeline with default settings
python -m src.main

# Run with limits for faster iteration
python -m src.main --patients-limit 100 --skip-allen --verbose

# Custom cohort (epilepsy patients)
python -m src.main --icd-codes G40 G41 --patients-limit 50

# Force data re-ingestion
python -m src.main --run-ingestion
```

### Individual Stages

```bash
# Stage 1: Ingest MIMIC-IV data
python -m src.ingestion

# Stage 2: Build RDF knowledge graph
python -m src.graph_construction

# Stage 3: Generate graph analysis report
python -m src.graph_analysis

# Stage 4: Extract features
python -m src.feature_extraction

# Stage 5: Train and evaluate models
python -m src.prediction
```

### Using Make

```bash
make ingest      # Stage 1
make graph       # Stage 2
make analyze     # Stage 3
make features    # Stage 4
make train       # Stage 5
make all         # All stages
make pipeline    # Full pipeline via main.py
```

### Python API

```python
from pathlib import Path
from config.settings import Settings
from src.main import run_pipeline

# Configure settings
settings = Settings()
settings = settings.model_copy(update={
    "patients_limit": 100,
    "skip_allen_relations": True,
})

# Run pipeline
result = run_pipeline(settings, skip_ingestion=True)

print(f"Cohort size: {result['cohort_size']}")
print(f"Graph triples: {result['graph_triples']}")
print(f"XGBoost AUROC: {result['metrics']['xgb']['auroc']:.4f}")
```

## Project Structure

```
readmission_predictor_mimic/
├── config/
│   └── settings.py              # Pydantic configuration
├── src/
│   ├── main.py                  # Pipeline orchestrator
│   ├── ingestion/               # Layer 1: Data ingestion
│   │   ├── mimic_loader.py      # DuckDB loader
│   │   └── derived_tables.py    # Age, cohort, readmission labels
│   ├── graph_construction/      # Layer 2: RDF graph
│   │   ├── pipeline.py          # Graph build orchestrator
│   │   ├── ontology.py          # Namespace definitions
│   │   ├── patient_writer.py    # Patient/admission RDF
│   │   ├── event_writers.py     # Clinical event RDF
│   │   └── temporal/
│   │       └── allen_relations.py  # Allen interval algebra
│   ├── graph_analysis/          # Layer 3: Analysis
│   │   ├── analysis.py          # Graph metrics
│   │   └── rdf_to_networkx.py   # RDF to NetworkX conversion
│   ├── feature_extraction/      # Layer 4: Features
│   │   ├── feature_builder.py   # Feature matrix builder
│   │   ├── tabular_features.py  # Clinical features
│   │   └── graph_features.py    # Temporal/structural features
│   └── prediction/              # Layer 5: Models
│       ├── split.py             # Patient-level splitting
│       ├── model.py             # Model training
│       └── evaluate.py          # Evaluation and reporting
├── ontology/
│   └── definition/
│       ├── base_ontology.rdf    # MIMIC-IV base ontology
│       └── extended_ontology.rdf # Readmission extensions
├── data/
│   ├── processed/               # DuckDB and RDF outputs
│   └── features/                # Feature matrices
├── outputs/
│   ├── models/                  # Trained models
│   └── reports/                 # Evaluation reports
├── tests/                       # Test suite
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── ontology.md
│   ├── features.md
│   ├── troubleshooting.md
│   ├── api.md
│   └── examples/
└── Makefile
```

## Output Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| DuckDB Database | `data/processed/mimiciv.duckdb` | Loaded MIMIC-IV tables + derived tables |
| RDF Graph | `data/processed/knowledge_graph.rdf` | OWL-Time compliant knowledge graph |
| Feature Matrix | `data/features/feature_matrix.parquet` | ML-ready feature matrix |
| LR Model | `outputs/models/logistic_regression.pkl` | Trained Logistic Regression |
| XGBoost Model | `outputs/models/xgboost.json` | Trained XGBoost (JSON format) |
| Graph Report | `outputs/reports/graph_analysis.md` | Graph structure analysis |
| LR Evaluation | `outputs/reports/evaluation_lr.md` | LR metrics and feature importance |
| XGBoost Evaluation | `outputs/reports/evaluation_xgb.md` | XGBoost metrics and feature importance |

## Interpreting Results

### Model Metrics

- **AUROC**: Area under ROC curve (0.5 = random, 1.0 = perfect). Values 0.65-0.75 are typical for readmission prediction.
- **AUPRC**: Area under precision-recall curve. More informative for imbalanced datasets.
- **Threshold**: Optimal classification threshold determined by Youden's J statistic.

### Feature Importance

The evaluation reports list the top-20 most important features:

- For **Logistic Regression**: Absolute coefficient values
- For **XGBoost**: Gain-based importance (total improvement in loss function)

Common predictive features include ICU length of stay, age, and lab value variability.

### Confusion Matrix

```
                    Predicted Negative  Predicted Positive
Actual Negative     True Negative       False Positive
Actual Positive     False Negative      True Positive
```

## Extending the System

### Adding New Cohorts

Modify cohort selection ICD codes:

```bash
# Epilepsy cohort
python -m src.main --icd-codes G40 G41

# Heart failure cohort
python -m src.main --icd-codes I50
```

Or update `config/settings.py`:

```python
cohort_icd_codes: list[str] = Field(default=["G40", "G41"])
```

### Adding New Event Types

1. Add query function in `src/graph_construction/pipeline.py`
2. Add writer function in `src/graph_construction/event_writers.py`
3. Update ontology in `ontology/definition/extended_ontology.rdf`

### Adding New Features

1. Add extraction function in `src/feature_extraction/tabular_features.py` or `graph_features.py`
2. Import and call in `src/feature_extraction/feature_builder.py`
3. Add to the merge chain in `build_feature_matrix()`

### Adding New Models

1. Add model class and training logic in `src/prediction/model.py`
2. Update `train_model()` function with new model type
3. Call from `src/main.py` in `_train_and_evaluate()`

## Testing

```bash
# Run all unit tests
make test

# Run fast unit tests only
make test-unit

# Run integration tests (requires MIMIC-IV or synthetic data)
pytest -m integration -v

# Run with coverage
pytest --cov=src tests/
```

## Known Limitations

1. **Diagnosis Data Leakage**: Discharge diagnoses are coded at discharge and may contain information about the full hospital course. Consider excluding `icd_chapter_*` features for strict temporal validity.

2. **Allen Relations Performance**: Computing Allen temporal relations is O(n²) for events within each ICU stay. Use `--skip-allen` for faster iteration.

3. **Memory Requirements**: Full cohort processing requires 16GB+ RAM. Use `--patients-limit` to reduce memory usage.

4. **Single Hospital**: MIMIC-IV is from a single hospital system (BIDMC). Model generalization to other hospitals is not guaranteed.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [MIMIC-IV](https://physionet.org/content/mimiciv/) - Johnson et al., PhysioNet
- [OWL-Time](https://www.w3.org/TR/owl-time/) - W3C Time Ontology
- [Allen's Interval Algebra](https://en.wikipedia.org/wiki/Allen%27s_interval_algebra) - James F. Allen

## References

If you use this work, please cite:

```bibtex
@software{readmission_predictor_mimic,
  title = {Temporal Knowledge Graph-Based Hospital Readmission Prediction},
  year = {2024},
  url = {https://github.com/your-org/readmission_predictor_mimic}
}
```
