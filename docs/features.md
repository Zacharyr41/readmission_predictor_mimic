# Feature Reference

This document describes all features extracted from the temporal knowledge graph for machine learning model training.

## Feature Overview

Features are organized into two main categories:

1. **Tabular Features** - Extracted from clinical data via SPARQL queries
2. **Graph Features** - Extracted from graph structure and temporal relations

Total features: ~50-100 depending on data coverage (biomarkers and vitals create dynamic columns).

## Feature Categories

### Category A: Tabular Features

These features are extracted by `src/feature_extraction/tabular_features.py`.

#### A1. Demographics

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `age` | integer | Age at hospital admission | `mimic:hasAge` property |
| `gender_M` | binary | Male gender indicator (1 if male, 0 otherwise) | `mimic:hasGender` = "M" |
| `gender_F` | binary | Female gender indicator (1 if female, 0 otherwise) | `mimic:hasGender` = "F" |

#### A2. Stay Characteristics

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `icu_los_hours` | float | ICU length of stay in hours | `time:numericDuration` * 24 |
| `num_icu_days` | integer | Number of ICU day intervals | Count of `mimic:hasICUDay` |
| `admission_type_ELECTIVE` | binary | Elective admission | `mimic:hasAdmissionType` |
| `admission_type_EMERGENCY` | binary | Emergency admission | `mimic:hasAdmissionType` |
| `admission_type_URGENT` | binary | Urgent admission | `mimic:hasAdmissionType` |
| `admission_type_*` | binary | Other admission types (one-hot) | `mimic:hasAdmissionType` |

#### A3. Lab Summary (Biomarkers)

For each biomarker type (e.g., "Glucose", "Creatinine", "Hemoglobin"), the following aggregate features are computed:

| Feature Pattern | Type | Description |
|-----------------|------|-------------|
| `{biomarker}_mean` | float | Mean value during ICU stay |
| `{biomarker}_min` | float | Minimum value |
| `{biomarker}_max` | float | Maximum value |
| `{biomarker}_std` | float | Standard deviation (0 if single measurement) |
| `{biomarker}_count` | integer | Number of measurements |
| `{biomarker}_first` | float | First recorded value (chronologically) |
| `{biomarker}_last` | float | Last recorded value |
| `{biomarker}_abnormal_rate` | float | Fraction of values outside reference range |

**Common biomarkers include:**
- Blood chemistry: Glucose, BUN, Creatinine, Sodium, Potassium, Chloride, Bicarbonate
- Complete blood count: Hemoglobin, Hematocrit, WBC, Platelets
- Liver function: ALT, AST, Bilirubin, Albumin
- Coagulation: PT, INR, PTT
- Inflammatory: Lactate, CRP, Procalcitonin

**Abnormal rate calculation:**
```python
is_abnormal = 0
if ref_lower is not None and value < ref_lower:
    is_abnormal = 1
elif ref_upper is not None and value > ref_upper:
    is_abnormal = 1
abnormal_rate = mean(is_abnormal for all measurements)
```

#### A4. Vital Summary (Clinical Signs)

For each vital sign type, the following aggregate features are computed:

| Feature Pattern | Type | Description |
|-----------------|------|-------------|
| `{vital}_mean` | float | Mean value during ICU stay |
| `{vital}_min` | float | Minimum value |
| `{vital}_max` | float | Maximum value |
| `{vital}_std` | float | Standard deviation |
| `{vital}_cv` | float | Coefficient of variation (std/mean) |
| `{vital}_count` | integer | Number of measurements |

**Common vitals include:**
- Heart Rate
- Respiratory Rate
- SpO2 (Oxygen Saturation)
- Temperature
- Blood Pressure (Systolic, Diastolic, Mean)
- GCS (Glasgow Coma Scale)

**Coefficient of variation (CV):**
```python
cv = std / mean if mean != 0 else 0.0
```

CV captures variability relative to magnitude, useful for detecting instability.

#### A5. Medication Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `num_distinct_meds` | integer | Number of unique medications | Count distinct `mimic:hasDrugName` |
| `total_antibiotic_days` | float | Total days on antibiotics | Sum of prescription durations |
| `has_antibiotic` | binary | Any antibiotic prescribed (1 or 0) | Presence of AntibioticAdmissionEvent |

**Antibiotic duration calculation:**
```python
duration_days = (end_datetime - start_datetime).total_seconds() / (24 * 3600)
total_antibiotic_days = sum(duration_days for all prescriptions)
```

#### A6. Diagnosis Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `num_diagnoses` | integer | Total number of diagnoses | Count of DiagnosisEvent |
| `icd_chapter_A` | binary | Chapter A (Infectious diseases) | Primary ICD code starts with 'A' |
| `icd_chapter_B` | binary | Chapter B (Infectious diseases) | Primary ICD code starts with 'B' |
| `icd_chapter_C` | binary | Chapter C (Neoplasms) | Primary ICD code starts with 'C' |
| `icd_chapter_D` | binary | Chapter D (Blood diseases) | Primary ICD code starts with 'D' |
| `icd_chapter_E` | binary | Chapter E (Endocrine) | Primary ICD code starts with 'E' |
| `icd_chapter_F` | binary | Chapter F (Mental) | Primary ICD code starts with 'F' |
| `icd_chapter_G` | binary | Chapter G (Nervous system) | Primary ICD code starts with 'G' |
| `icd_chapter_H` | binary | Chapter H (Eye/Ear) | Primary ICD code starts with 'H' |
| `icd_chapter_I` | binary | Chapter I (Circulatory) | Primary ICD code starts with 'I' |
| `icd_chapter_J` | binary | Chapter J (Respiratory) | Primary ICD code starts with 'J' |
| `icd_chapter_K` | binary | Chapter K (Digestive) | Primary ICD code starts with 'K' |
| `icd_chapter_*` | binary | Other ICD-10 chapters | Primary ICD code first letter |

**Data leakage warning:** Diagnosis codes are assigned at discharge and may reflect the full hospital course. Consider excluding `icd_chapter_*` features for strict temporal validity in prospective prediction scenarios.

### Category B: Graph Features

These features are extracted by `src/feature_extraction/graph_features.py`.

#### B1. Temporal Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `events_per_icu_day` | float | Average events per ICU day | total_events / num_icu_days |
| `num_before_relations` | integer | Count of "before" relations | `time:before` triples |
| `num_during_relations` | integer | Count of "during" relations | `time:inside` triples |
| `total_temporal_edges` | integer | Total Allen relation edges | Sum of all Allen relations |

**Temporal relations counted:**
- `time:before` - Event A ends before event B starts
- `time:inside` - Event A entirely within event B
- `time:intervalOverlaps` - Event A overlaps with event B
- `time:intervalMeets` - Event A ends exactly when B starts
- `time:intervalStarts` - Events start together, A ends first
- `time:intervalFinishes` - Events end together, A starts later

#### B2. Graph Structure Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `patient_subgraph_nodes` | integer | Nodes in patient's subgraph | BFS from admission node |
| `patient_subgraph_edges` | integer | Edges in patient's subgraph | Edge count in subgraph |
| `patient_subgraph_density` | float | Graph density | edges / (nodes * (nodes-1)) |
| `mean_node_degree` | float | Average node degree | Mean of degree sequence |
| `max_node_degree` | integer | Maximum node degree | Max of degree sequence |

**Subgraph extraction:**
The patient subgraph includes all nodes within 3 hops of the admission node via BFS traversal, capturing:
- Admission → ICU stays → Events → Temporal relations

**Graph density formula (directed graph):**
```python
density = num_edges / (num_nodes * (num_nodes - 1))
```

## Labels

| Label | Type | Description | Source |
|-------|------|-------------|--------|
| `readmitted_30d` | binary | Readmitted within 30 days (0 or 1) | `mimic:readmittedWithin30Days` |
| `readmitted_60d` | binary | Readmitted within 60 days (0 or 1) | `mimic:readmittedWithin60Days` |

**Readmission definition:**
- Computed from next hospital admission date
- Excludes patients who died in hospital (`hospital_expire_flag = 1`)
- Days counted from discharge to next admission

## Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `hadm_id` | integer | Hospital admission identifier (unique per row) |
| `subject_id` | integer | Patient identifier (for patient-level splitting) |

## Missing Value Handling

The `_fill_missing_values()` function in `feature_builder.py` handles missing values:

### Strategy

| Column Pattern | Fill Strategy | Rationale |
|----------------|---------------|-----------|
| `*_count` | 0 | No measurements = 0 count |
| `num_*` | 0 | No events = 0 count |
| `has_*` | 0 | Not present = 0 |
| `total_*` | 0 | No events = 0 total |
| `gender_*` | 0 | Binary encoding |
| `admission_type_*` | 0 | One-hot encoding |
| `icd_chapter_*` | 0 | One-hot encoding |
| Other numeric | median | Continuous features |

### Implementation

```python
for col in df.columns:
    if col in label_cols or col in ["hadm_id", "subject_id"]:
        continue

    if df[col].isna().any():
        is_count = any(pattern in col.lower() for pattern in [
            "_count", "num_", "has_", "total_"
        ])
        is_binary = any(pattern in col.lower() for pattern in [
            "gender_", "admission_type_", "icd_chapter_"
        ])

        if is_count or is_binary:
            df[col] = df[col].fillna(0)
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
```

## Feature Engineering Notes

### Biomarker Aggregation

1. **Temporal ordering**: Values are ordered by `charttime` before computing first/last
2. **Reference ranges**: Used from MIMIC-IV `labevents` table (`ref_range_lower`, `ref_range_upper`)
3. **Missing references**: If reference range is NULL, value is not counted as abnormal

### Vital Sign Aggregation

1. **All measurements included**: No time windowing applied
2. **CV interpretation**: High CV indicates physiological instability

### Temporal Features

1. **Allen relations**: Only computed if `--skip-allen` flag is not set
2. **Computational cost**: O(n²) for n events per ICU stay
3. **Zero values**: If no Allen relations computed, temporal features are 0

### Graph Structure Features

1. **Conversion**: RDF graph converted to NetworkX directed graph
2. **Literals excluded**: Only URI nodes included in structure features
3. **Hop limit**: 3 hops from admission node

## Example Feature Importance

Typical top features for 30-day readmission prediction:

| Rank | Feature | Importance (XGBoost) |
|------|---------|---------------------|
| 1 | `icu_los_hours` | 0.0823 |
| 2 | `age` | 0.0671 |
| 3 | `Creatinine_max` | 0.0543 |
| 4 | `num_icu_days` | 0.0498 |
| 5 | `Hemoglobin_min` | 0.0456 |
| 6 | `events_per_icu_day` | 0.0412 |
| 7 | `BUN_mean` | 0.0389 |
| 8 | `total_antibiotic_days` | 0.0367 |
| 9 | `WBC_max` | 0.0345 |
| 10 | `num_diagnoses` | 0.0321 |

*Note: Actual values vary by cohort and data availability.*

## Feature Extraction Pipeline

```python
from rdflib import Graph
from src.feature_extraction.feature_builder import build_feature_matrix

# Load the knowledge graph
graph = Graph()
graph.parse("data/processed/knowledge_graph.rdf")

# Extract all features
feature_df = build_feature_matrix(
    graph,
    save_path=Path("data/features/feature_matrix.parquet")
)

# Inspect features
print(f"Shape: {feature_df.shape}")
print(f"Columns: {list(feature_df.columns)}")
print(f"Missing values:\n{feature_df.isna().sum()}")
```

## Adding New Features

### Adding a New Tabular Feature

1. Create extraction function in `tabular_features.py`:

```python
def extract_my_feature(graph: Graph) -> pd.DataFrame:
    """Extract my custom feature.

    Args:
        graph: RDF graph containing clinical data.

    Returns:
        DataFrame with columns: hadm_id, my_feature
    """
    query = """
    SELECT ?hadmId ?value
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId .
        # ... your SPARQL query
    }
    """
    results = list(graph.query(query))
    # Process results...
    return pd.DataFrame(data)
```

2. Import and add to `feature_builder.py`:

```python
from src.feature_extraction.tabular_features import extract_my_feature

def build_feature_matrix(graph, save_path=None):
    # ... existing code ...
    my_features = extract_my_feature(graph)

    feature_dfs = [
        # ... existing feature dfs ...
        my_features,
    ]
```

### Adding a New Graph Feature

1. Create extraction function in `graph_features.py`:

```python
def extract_my_graph_feature(graph: Graph) -> pd.DataFrame:
    """Extract graph-based feature.

    Args:
        graph: RDF graph.

    Returns:
        DataFrame with columns: hadm_id, my_graph_feature
    """
    # Convert to NetworkX if needed
    nx_graph = rdf_to_networkx(graph, include_literals=False)
    # Compute graph metrics...
    return pd.DataFrame(data)
```

2. Add to `feature_builder.py` merge chain.
