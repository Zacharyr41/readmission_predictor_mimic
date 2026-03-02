# WLST Prediction Pipeline — Implementation Plan

## Context

This project builds a **new** prediction pipeline alongside the existing stroke/readmission pipeline. The research question: among severe TBI patients (GCS ≤ 8), can a GNN trained on a temporal knowledge graph of the **first 48 hours** of clinical data predict which patients received Withdrawal of Life-Sustaining Treatment (WLST)? Stage 2 then asks whether adding non-clinical confounders (language, hospital service, comorbidities) changes predictive performance — and what that implies about equity in WLST decisions.

**Prediction framing**: Features are restricted to the first 48h after ICU admission. The WLST label reflects code status changes at **any point** during the ICU stay (not just within 48h). This is the standard "predict future events from early data" setup — the model learns whether early clinical trajectory predicts eventual WLST.

The existing codebase provides ~80% of the infrastructure (BigQuery ingestion, RDF graph construction, TD4DD GNN, Vertex AI cloud training). The core changes are: different cohort, different label, 48h time window, new MIMIC tables, new clinical event types, and a two-stage experimental comparison.

---

## Phase 0: Project Scaffolding & Configuration

### 0.1 Create `src/wlst/` module structure

New module that imports shared infrastructure but owns WLST-specific logic:

```
src/wlst/
├── __init__.py
├── cohort.py          # TBI cohort selection + WLST label derivation
├── graph_pipeline.py  # WLST-specific graph construction orchestrator
├── event_writers.py   # New event types (vasopressors, GCS series, procedures)
├── features.py        # 48h feature extraction
├── experiments.py     # WLST experiment registry (Stage 1 + Stage 2)
├── evaluate.py        # WLST-specific evaluation (+ Stage 1 vs 2 comparison)
└── main.py            # CLI entry point
```

### 0.2 Extend `config/settings.py`

Add WLST-specific settings alongside existing stroke config:

```python
# WLST pipeline configuration
wlst_mode: bool = Field(default=False)
wlst_icd_prefixes: list[str] = Field(default=["S06"])  # TBI ICD-10
wlst_gcs_threshold: int = Field(default=8)              # GCS ≤ 8
wlst_observation_window_hours: int = Field(default=48)   # First 48h only
wlst_icu_types: list[str] = Field(default=[              # Neuro/trauma ICUs
    "Neuro Stepdown",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Trauma SICU (TSICU)",
])
wlst_stage: Literal["stage1", "stage2"] = Field(default="stage1")
```

### 0.3 Add new BigQuery/local tables to ingestion

**File**: `src/ingestion/bigquery_loader.py`

Add to Phase 1 (dimension) tables:
- `d_items` (already loaded)
- `omr` (outpatient medical records — for BMI)
- `services` (hospital service assignments)

Add to Phase 2 (cohort-filtered) tables:
- `inputevents` — vasopressors, ICP medications, IV fluids
- `procedureevents` — ventilation, craniectomy, ICP monitor, EVD
- `transfers` — care unit tracking

These tables live in `mimiciv_3_1_icu` (inputevents, procedureevents) and `mimiciv_3_1_hosp` (transfers, services, omr). The existing BigQuery loader already handles both schemas — just add table names to the appropriate phase lists.

Also update `src/ingestion/mimic_loader.py` for local CSV path discovery.

**Verification**: After loading, run `SELECT COUNT(*) FROM inputevents` etc. in DuckDB to confirm row counts.

---

## Phase 1: Cohort Selection & WLST Label Derivation

This is the most critical and nuanced phase. Getting the label right is foundational.

### 1.1 TBI Cohort Selection

**File**: `src/wlst/cohort.py` — new function `select_tbi_cohort()`

SQL logic:

```sql
WITH tbi_diagnoses AS (
    -- ICD-10 S06.x codes for TBI (any seq_num, not just principal)
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE icd_version = 10
      AND icd_code LIKE 'S06%'
),
ranked_icu_stays AS (
    SELECT
        i.subject_id, i.hadm_id, i.stay_id,
        i.intime, i.outtime, i.first_careunit,
        DATE_DIFF('hour', i.intime, i.outtime) AS icu_hours,
        ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS stay_rank
    FROM icustays i
),
eligible_stays AS (
    SELECT * FROM ranked_icu_stays
    WHERE stay_rank = 1
      AND icu_hours > 24
      AND first_careunit IN (
          'Neuro Stepdown',
          'Neuro Surgical Intensive Care Unit (Neuro SICU)',
          'Trauma SICU (TSICU)'
      )
)
SELECT es.*
FROM eligible_stays es
JOIN tbi_diagnoses td ON es.hadm_id = td.hadm_id
JOIN age a ON es.subject_id = a.subject_id AND es.hadm_id = a.hadm_id
WHERE a.age BETWEEN 18 AND 89
```

**Key difference from stroke cohort**: Uses `S06.x` codes, any `seq_num` (not just principal), filters on `first_careunit` for neuro/trauma ICUs.

**Important**: Verify the exact `first_careunit` string values by querying `SELECT DISTINCT first_careunit FROM icustays` before hardcoding.

### 1.2 GCS Filtering (GCS ≤ 8)

**File**: `src/wlst/cohort.py`

After initial cohort selection, filter to severe TBI:

```sql
WITH admission_gcs AS (
    SELECT
        ce.stay_id,
        MIN(CASE WHEN ce.itemid = 220739 THEN ce.valuenum END) AS gcs_eye,
        MIN(CASE WHEN ce.itemid = 223900 THEN ce.valuenum END) AS gcs_verbal,
        MIN(CASE WHEN ce.itemid = 223901 THEN ce.valuenum END) AS gcs_motor,
        COALESCE(MIN(CASE WHEN ce.itemid = 220739 THEN ce.valuenum END), 0)
        + COALESCE(MIN(CASE WHEN ce.itemid = 223900 THEN ce.valuenum END), 0)
        + COALESCE(MIN(CASE WHEN ce.itemid = 223901 THEN ce.valuenum END), 0) AS gcs_total
    FROM chartevents ce
    JOIN cohort c ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN (220739, 223900, 223901)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24' HOUR
    GROUP BY ce.stay_id
)
SELECT c.*
FROM cohort c
JOIN admission_gcs g ON c.stay_id = g.stay_id
WHERE g.gcs_total <= 8
  AND g.gcs_total > 0  -- Exclude missing/invalid
```

**Decision**: Use the *first* recorded GCS within 24h (not lowest), as this represents admission severity. The `MIN` per component handles cases where components are recorded at slightly different times.

### 1.3 WLST Label Derivation

**File**: `src/wlst/cohort.py`

WLST is not a standalone label in MIMIC-IV. We derive it from multiple signals. The label reflects events at **any point during the stay** (not restricted to 48h), since we are predicting eventual WLST from early data.

```sql
CREATE OR REPLACE TABLE wlst_labels AS
WITH code_status_events AS (
    -- itemid 223758 = "Code Status" in chartevents
    -- Values: "Full code", "DNR", "DNI", "DNR / DNI",
    --         "Comfort measures only", "CMO"
    SELECT
        ce.subject_id, ce.hadm_id, ce.stay_id,
        ce.charttime AS code_status_time,
        ce.value AS code_status_value,
        ROW_NUMBER() OVER (
            PARTITION BY ce.stay_id ORDER BY ce.charttime
        ) AS event_rank
    FROM chartevents ce
    WHERE ce.itemid = 223758
),
first_non_full_code AS (
    SELECT *
    FROM code_status_events
    WHERE code_status_value NOT IN ('Full code', 'Full Code')
      AND event_rank = (
          SELECT MIN(e2.event_rank)
          FROM code_status_events e2
          WHERE e2.stay_id = code_status_events.stay_id
            AND e2.code_status_value NOT IN ('Full code', 'Full Code')
      )
),
death_info AS (
    SELECT hadm_id, hospital_expire_flag, deathtime, discharge_location
    FROM admissions
),
cohort_labels AS (
    SELECT
        c.subject_id, c.hadm_id, c.stay_id, c.intime,
        fnfc.code_status_time,
        fnfc.code_status_value,
        d.hospital_expire_flag,
        d.deathtime,
        d.discharge_location,
        DATE_DIFF('hour', c.intime, fnfc.code_status_time)
            AS hours_to_code_change,
        -- WLST label
        CASE
            WHEN fnfc.code_status_value IS NOT NULL THEN 1
            WHEN d.discharge_location = 'HOSPICE' THEN 1
            ELSE 0
        END AS wlst_label,
        -- Outcome categories for descriptive analysis
        CASE
            WHEN fnfc.code_status_value IN ('Comfort measures only', 'CMO')
                 AND d.hospital_expire_flag = 1 THEN 'CMO_death'
            WHEN fnfc.code_status_value IN ('DNR', 'DNI', 'DNR / DNI')
                 AND d.hospital_expire_flag = 1 THEN 'DNR_death'
            WHEN fnfc.code_status_value IS NOT NULL
                 AND d.hospital_expire_flag = 0 THEN 'limited_code_survived'
            WHEN fnfc.code_status_value IS NULL
                 AND d.hospital_expire_flag = 1 THEN 'full_code_death'
            WHEN fnfc.code_status_value IS NULL
                 AND d.hospital_expire_flag = 0 THEN 'full_code_survived'
            WHEN d.discharge_location = 'HOSPICE' THEN 'hospice'
            ELSE 'other'
        END AS outcome_category
    FROM cohort c
    LEFT JOIN first_non_full_code fnfc ON c.stay_id = fnfc.stay_id
    LEFT JOIN death_info d ON c.hadm_id = d.hadm_id
)
SELECT * FROM cohort_labels
```

**Label distribution expectations**: In severe TBI, WLST rates are typically 40-70%.

**Critical verification steps**:
1. `SELECT wlst_label, COUNT(*) FROM wlst_labels GROUP BY wlst_label` — check class balance
2. `SELECT outcome_category, COUNT(*) FROM wlst_labels GROUP BY outcome_category` — understand composition
3. Identify the "interesting" patients: `wlst_label = 0 AND hospital_expire_flag = 0` (survived without WLST — the patients who might have been harmed by premature WLST in a counterfactual)
4. Distribution of `hours_to_code_change` — how many WLST decisions happen within vs. after 48h?

**Important nuance**: Some WLST decisions occur *after* 48h. The model sees no WLST signal in the feature window but the label is still 1. This is by design — it tests whether early clinical trajectory predicts eventual WLST.

### 1.4 Cohort Descriptive Statistics

**File**: `src/wlst/cohort.py`

Generate a summary table before proceeding:
- Total TBI admissions → after GCS filter → after ICU type filter → final cohort size
- WLST rate, mortality rate, discharge disposition distribution
- Demographics breakdown (age, sex, race) by WLST status
- Median time to code status change
- GCS distribution

**Output**: `outputs/reports/wlst_cohort_summary.md`

---

## Phase 2: Data Loading — New MIMIC Tables

### 2.1 Extend BigQuery Loader

**File**: `src/ingestion/bigquery_loader.py`

| Table | Schema | Phase | Filter | Purpose |
|-------|--------|-------|--------|---------|
| `inputevents` | `mimiciv_3_1_icu` | 2 (cohort-filtered) | `subject_id IN (...)` | Vasopressors, ICP meds |
| `procedureevents` | `mimiciv_3_1_icu` | 2 (cohort-filtered) | `subject_id IN (...)` | Ventilation, neurosurgery |
| `omr` | `mimiciv_3_1_hosp` | 1 (full) | None | BMI lookup |
| `transfers` | `mimiciv_3_1_hosp` | 2 (cohort-filtered) | `subject_id IN (...)` | Care unit tracking |
| `services` | `mimiciv_3_1_hosp` | 1 (full) | None | Hospital service |

### 2.2 Verify Table Availability

Before coding, confirm these tables exist in BigQuery:

```sql
SELECT table_name FROM `physionet-data.mimiciv_3_1_icu.INFORMATION_SCHEMA.TABLES`
SELECT table_name FROM `physionet-data.mimiciv_3_1_hosp.INFORMATION_SCHEMA.TABLES`
```

Check `inputevents` columns — MIMIC-IV v3.1 may use `inputevents` (not `inputevents_mv`/`inputevents_cv`).

---

## Phase 3: Graph Construction (Stage 1 — Clinical Trajectory Only)

### 3.1 48-Hour Time Window Enforcement

**The single most important architectural decision**: All clinical event queries get a time filter:

```sql
WHERE ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '48' HOUR
```

This applies to: chartevents (GCS, MAP, vitals), labevents, inputevents, procedureevents.

**Exception**: Admission-level data (diagnoses, demographics, admission GCS) is included regardless since it represents baseline context.

**Do NOT leak the label**: Code status events (itemid 223758) must be **excluded** from the graph features. They are used only for label derivation.

### 3.2 New Node Types for WLST Graph

**File**: `src/wlst/event_writers.py`

| New Node Type | RDF Class | Temporal Type | Source Table | Key Attributes |
|--------------|-----------|---------------|--------------|----------------|
| `GCSEvent` | `MIMIC_NS.GCSEvent` | `time:Instant` | chartevents | eye, verbal, motor, total |
| `VasopressorEvent` | `MIMIC_NS.VasopressorEvent` | `time:ProperInterval` | inputevents | drug, rate, amount |
| `VentilationEvent` | `MIMIC_NS.VentilationEvent` | `time:ProperInterval` | procedureevents | start, end, duration |
| `NeurosurgeryEvent` | `MIMIC_NS.NeurosurgeryEvent` | `time:Instant` or `Interval` | procedureevents | procedure type |
| `ICPMedicationEvent` | `MIMIC_NS.ICPMedicationEvent` | `time:ProperInterval` | inputevents | drug, dose, rate |
| `MAPEvent` | `MIMIC_NS.MAPEvent` | `time:Instant` | chartevents | value, method (arterial/cuff) |

**Reuse existing types**: `BioMarkerEvent` for labs, `DiagnosisEvent` for ICD codes.

### 3.3 GCS Time Series as Graph Structure

The GCS motor subscore evolution over 48h is a key clinical feature:

- Individual `GCSEvent` nodes at each measurement time
- Each linked to the ICU stay and relevant ICU day
- `hasGCSEye`, `hasGCSVerbal`, `hasGCSMotor`, `hasGCSTotal` numeric attributes
- Allen temporal relations between sequential GCS events capture trajectory (improving vs declining)
- Hourly binning happens in the feature extraction layer, not the graph

### 3.4 Vasopressor Events

**Source**: `inputevents`, filtered by drug names:

```sql
SELECT * FROM inputevents
WHERE LOWER(label) IN ('norepinephrine', 'phenylephrine', 'vasopressin',
                        'epinephrine', 'dopamine')
  AND stay_id IN (SELECT stay_id FROM cohort)
  AND starttime BETWEEN c.intime AND c.intime + INTERVAL '48' HOUR
```

Model as interval events (starttime → endtime) with rate/amount attributes.

### 3.5 Mechanical Ventilation

**Source**: `procedureevents` (itemid 225792 for invasive ventilation)

Key features: time from ICU admission to ventilation initiation, total ventilation hours in 48h window.

### 3.6 ICP Medications

**Source**: `inputevents` + `prescriptions`, filtered by drug name:
- Mannitol
- Hypertonic saline (3%, 23.4%)
- Levetiracetam (Keppra)

### 3.7 Neurosurgical Procedures

**Source**: `procedureevents`
- Craniectomy (itemid 225752)
- ICP monitor placement (itemid 228114)
- EVD (External Ventricular Drain) — need to identify itemid via `d_items` lookup:
  `SELECT * FROM d_items WHERE label ILIKE '%crani%' OR label ILIKE '%EVD%' OR label ILIKE '%ventricular drain%'`

### 3.8 MAP (Mean Arterial Pressure) Series

**Source**: `chartevents`
- itemid 220052: Arterial MAP
- itemid 220051: Cuff MAP

Model as `MAPEvent` instant nodes with numeric value and method attribute.

### 3.9 Labs (48h Window)

**Source**: `labevents`, restricted to first 48h:
- Sodium (50983), Lactate (50813), Glucose (50931), INR (51237), Creatinine (50912)

Reuse existing `BioMarkerEvent` writer with SNOMED mapping.

### 3.10 Graph Construction Pipeline Orchestrator

**File**: `src/wlst/graph_pipeline.py`

Adapts `src/graph_construction/pipeline.py`:
1. Use `select_tbi_cohort()` instead of `select_neurology_cohort()`
2. Use `wlst_labels` instead of `readmission_labels`
3. Add 48h window to all event queries
4. **Exclude** code status events (itemid 223758) from graph features
5. Write new event types (GCS, vasopressor, ventilation, neurosurgery, ICP meds, MAP)
6. Compute Allen temporal relations on the expanded event set

**Reuse from `src/graph_construction/`**: Patient writer, ICU stay/day writer, temporal ontology, SNOMED mapper, Oxigraph backend, multiprocessing infrastructure.

### 3.11 Graph Validation & Analysis

After building the Stage 1 graph:
1. Node/edge count report by type
2. Per-patient graph size distribution
3. Temporal coverage: % of patients with GCS series, vasopressor events, etc.
4. Label distribution on admission nodes
5. Export to Neo4j for visual inspection (optional)

**Output**: `outputs/reports/wlst_stage1_graph_analysis.md`

---

## Phase 4: Feature Extraction (48h Features)

### 4.1 Tabular Features for Classical Baselines

**File**: `src/wlst/features.py`

| Feature Group | Features | Source |
|--------------|----------|--------|
| Demographics | age, sex, race_encoded, bmi, insurance_type | patients, admissions, omr |
| Admission GCS | gcs_eye, gcs_verbal, gcs_motor, gcs_total | chartevents (first values) |
| GCS Trajectory | gcs_motor_delta (last-first), gcs_motor_min, gcs_motor_max, gcs_improving (binary) | chartevents hourly bins |
| Injury Severity | head_ais_score (ICD → AIS mapping) | diagnoses_icd |
| Hemodynamics | map_mean, map_min, map_max, map_trend, vasopressor_any, vasopressor_hours, vasopressor_max_dose | chartevents, inputevents |
| Ventilation | vent_initiated (binary), hours_to_vent, vent_duration_48h | procedureevents |
| ICP Management | mannitol_given, hypertonic_saline_given, levetiracetam_given, icp_med_count | inputevents, prescriptions |
| Labs (48h) | sodium_{first,last,min,max}, lactate_{...}, glucose_{...}, inr_{...}, creatinine_{...} | labevents |
| Neurosurgery | craniectomy (binary), icp_monitor (binary), evd (binary), any_neurosurgery | procedureevents |

### 4.2 ICD-to-AIS Mapping for Injury Severity

Implement ICDPIC-R equivalent logic to map ICD-10 S06.x subcodes to AIS head region scores:

| ICD-10 | Description | AIS Score |
|--------|-------------|-----------|
| S06.0 | Concussion | 1-2 |
| S06.1 | Traumatic cerebral edema | 4-5 |
| S06.2 | Diffuse TBI | 4-5 |
| S06.3 | Focal TBI | 3-5 |
| S06.4 | Epidural hemorrhage | 4-5 |
| S06.5 | Subdural hemorrhage | 4-5 |
| S06.6 | Subarachnoid hemorrhage | 3-4 |

Store as `data/mappings/icd10_to_ais_head.json`.

### 4.3 Feature Matrix Output

Save to `data/features/wlst_feature_matrix.parquet` with all features plus `wlst_label`, `outcome_category`, `subject_id`, `hadm_id`, `stay_id`.

---

## Phase 5: GNN Training — Stage 1

### 5.1 RDF → HeteroData Export

**Extend or create alongside** `src/gnn/graph_export.py`:

Key changes from readmission pipeline:
- **New node types**: `gcs_event`, `vasopressor_event`, `ventilation_event`, `neurosurgery_event`, `icp_medication_event`, `map_event`
- **Label**: `data["admission"].y = wlst_label` (binary 0/1) instead of `readmitted_30d`
- **Metadata**: Store `outcome_category` for post-hoc analysis
- **Edge types**: Add edges from new event types to ICU stays/days
- **SapBERT embeddings**: Generate for any new SNOMED-mapped concepts

### 5.2 Experiment Registry

**File**: `src/wlst/experiments.py`

```python
WLST_EXPERIMENT_REGISTRY = {
    # Stage 1 ablations (clinical trajectory only)
    "W1_mlp_baseline": ExperimentConfig(
        use_transformer=False, use_diffusion=False,
        use_temporal_encoding=False,
        description="Floor: projection + classifier"
    ),
    "W2_transformer_only": ExperimentConfig(
        use_transformer=True, use_diffusion=False,
        use_temporal_encoding=False,
        description="Graph structure contribution"
    ),
    "W3_transformer_temporal": ExperimentConfig(
        use_transformer=True, use_diffusion=False,
        use_temporal_encoding=True,
        description="Temporal encoding (48h trajectory)"
    ),
    "W4_full_model": ExperimentConfig(
        use_transformer=True, use_diffusion=True,
        use_temporal_encoding=True,
        description="Full TD4DD on Stage 1 graph"
    ),
}
```

### 5.3 Classical ML Baselines

Reuse `src/prediction/model.py`:

- Logistic Regression with L1 penalty + balanced class weights
- XGBoost with `scale_pos_weight`
- Patient-level train/val/test split (70/15/15), stratified by `wlst_label`
- Metrics: AUROC, AUPRC, sensitivity at 80% specificity, feature importance

These baselines answer: "does the GNN add value over tabular features?"

### 5.4 Training Configuration

```python
TrainingConfig(
    lr=0.001,
    batch_size=64,       # Adjust based on cohort size
    max_epochs=200,
    patience=20,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    device="auto",
)
```

### 5.5 Evaluation Metrics

**File**: `src/wlst/evaluate.py`

Standard metrics:
- AUROC, AUPRC, Sensitivity, Specificity, F1, Confusion matrix
- Calibration curve (Brier score)

WLST-specific metrics:
- **Sensitivity at high specificity** (90%, 95%): clinical utility measure
- **Subgroup analysis**: Performance by `outcome_category`
- **Attention analysis**: Which graph neighborhoods drive WLST predictions?

### 5.6 Stage 1 Deliverables

1. Trained GNN checkpoint (best val AUROC)
2. Classical ML baselines (LR, XGBoost)
3. Evaluation report: `outputs/reports/wlst_stage1_evaluation.md`
4. Attention analysis: Which 48h clinical trajectory features most predict WLST?
5. Key finding: Identify patients who did NOT receive WLST and survived (the "self-fulfilling prophecy" question)

---

## Phase 6: Stage 2 — Non-Clinical Confounders

### 6.1 Additional Features

| Feature | Source | Type | Rationale |
|---------|--------|------|-----------|
| Language barrier | `admissions.language` | Categorical | Communication barriers affect goals-of-care discussions |
| Interpreter used | `chartevents` (interpreter items) | Binary | Direct measure of language accommodation |
| Hospital service | `services.curr_service` | Categorical | Service culture may affect WLST decisions |
| Care unit transfers | `transfers.careunit` | Sequence/count | Transfer patterns may differ |
| Charlson Comorbidity Index | `diagnoses_icd` → ICD-10 mapping | Numeric (0-33) | Baseline health burden |
| Individual Charlson components | `diagnoses_icd` | Binary (17 flags) | Granular comorbidity profile |

### 6.2 Charlson Comorbidity Index

Create `data/mappings/icd10_to_charlson.json` with standard ICD-10 → Charlson component mapping (17 categories, weights 1/2/3/6).

### 6.3 Stage 2 Graph Construction

**File**: `src/wlst/graph_pipeline.py` — add `stage` parameter

**Approach**: Same Stage 1 graph + additional node attributes on Patient/Admission nodes for non-clinical features. This preserves direct comparability between stages.

### 6.4 Stage 2 Experiments

```python
"W5_stage2_mlp_baseline": ExperimentConfig(...)    # Tabular with all features
"W6_stage2_full_model": ExperimentConfig(...)       # TD4DD on extended graph
"W7_stage2_confounders_only": ExperimentConfig(...) # Only non-clinical (ablation)
```

### 6.5 Stage 1 vs Stage 2 Comparison

**File**: `src/wlst/evaluate.py`

Formal comparison:
1. **Same test set**: Both stages evaluated on identical held-out patients
2. **DeLong test / paired bootstrap**: Statistical significance of AUROC difference
3. **Net Reclassification Improvement (NRI)**: How many patients correctly reclassified?
4. **Feature importance shift**: Do attention weights change when confounders are added?
5. **Subgroup equity analysis**: Performance by demographic groups

**Output**: `outputs/reports/wlst_stage1_vs_stage2_comparison.md`

---

## Phase 7: Cloud Training (Vertex AI)

### 7.1 Extend `src/cloud_train.py`

Add WLST pipeline mode triggered by `PIPELINE_MODE=wlst` env var:
- Step 2b: TBI cohort selection + WLST labels
- Step 3b: WLST graph construction (Stage 1 or Stage 2)
- Step 4b: WLST feature extraction
- Step 5b: GNN prep with WLST-specific node types
- Step 6b: WLST experiment training

### 7.2 Update `scripts/cloud_train.sh`

Add flags:
```bash
--pipeline wlst
--wlst-stage stage1|stage2
--wlst-experiment W1_mlp_baseline|W4_full_model|...
```

### 7.3 GCS Output Structure

```
gs://{bucket}/runs/{run_id}/
├── wlst/
│   ├── cohort_summary.md
│   ├── stage1/
│   │   ├── graph_analysis.md
│   │   ├── feature_matrix.parquet
│   │   ├── experiments/
│   │   │   ├── W1_mlp_baseline/{metrics,history,config}.json
│   │   │   ├── W4_full_model/{metrics,history,config,checkpoint,attention}.json|.pt
│   │   │   └── baselines/{logistic_regression.pkl, xgboost.json}
│   │   └── evaluation_report.md
│   ├── stage2/
│   │   ├── graph_analysis.md
│   │   ├── experiments/...
│   │   └── evaluation_report.md
│   └── comparison/
│       └── stage1_vs_stage2.md
```

### 7.4 Monitoring

- Vertex AI dashboard for GPU utilization
- Structured logging for each pipeline step
- Metrics JSON uploaded to GCS after each experiment
- Training history (loss/AUROC curves) for post-hoc analysis

---

## Phase 8: Testing

### 8.1 Unit Tests

```
tests/test_wlst/
├── test_cohort.py           # TBI cohort SQL, GCS filtering, WLST label derivation
├── test_event_writers.py    # New event types (GCS, vasopressor, ventilation, etc.)
├── test_graph_pipeline.py   # 48h window enforcement, graph construction
├── test_features.py         # Feature extraction, Charlson index, AIS mapping
├── test_experiments.py      # WLST experiment registry
└── test_evaluate.py         # Stage comparison metrics
```

### 8.2 Integration Tests

- End-to-end: Synthetic MIMIC data → TBI cohort → WLST graph → HeteroData → 1 epoch
- Verify 48h window: Assert no events beyond 48h in graph
- Verify label correctness: Spot-check against known patients
- Verify code status (223758) is NOT in graph features

### 8.3 Data Validation

- Cohort size check (expect 200-1000 patients in MIMIC-IV)
- Label balance check (expect ~40-70% WLST in severe TBI)
- Feature completeness: % missing per feature
- Graph connectivity: No isolated patient nodes

---

## Implementation Order

| Step | Phase | Description | Depends On | Complexity |
|------|-------|-------------|------------|------------|
| 1 | 0 | Scaffold `src/wlst/` + config | Nothing | Low |
| 2 | 2 | Add new tables to BigQuery loader | Step 1 | Low |
| 3 | 1 | TBI cohort selection + GCS filter | Steps 1-2 | Medium |
| 4 | 1 | WLST label derivation | Step 3 | **High** |
| 5 | 1 | Cohort descriptive statistics | Step 4 | Low |
| 6 | 3 | New event writers (GCS, vasopressor, vent, etc.) | Step 1 | Medium |
| 7 | 3 | WLST graph pipeline (48h window) | Steps 4, 6 | Medium |
| 8 | 3 | Graph validation + analysis | Step 7 | Low |
| 9 | 4 | Tabular feature extraction (48h) | Step 4 | Medium |
| 10 | 4 | ICD-to-AIS + Charlson mappings | Step 1 | Medium |
| 11 | 5 | HeteroData export with WLST labels | Steps 7, 10 | Medium |
| 12 | 5 | Classical ML baselines (LR, XGBoost) | Step 9 | Low |
| 13 | 5 | WLST experiment registry + training | Step 11 | Medium |
| 14 | 5 | Stage 1 evaluation report | Steps 12, 13 | Medium |
| 15 | 6 | Stage 2 features (language, service, Charlson) | Step 10 | Medium |
| 16 | 6 | Stage 2 graph + training | Steps 14, 15 | Medium |
| 17 | 6 | Stage 1 vs Stage 2 comparison | Steps 14, 16 | Medium |
| 18 | 7 | Cloud training integration | Steps 13, 16 | Low |
| 19 | 8 | Tests | Throughout | Ongoing |

---

## Key Files Summary

### Modify (Existing)

| File | Change |
|------|--------|
| `config/settings.py` | Add WLST settings |
| `src/ingestion/bigquery_loader.py` | Add inputevents, procedureevents, omr, transfers, services |
| `src/ingestion/mimic_loader.py` | Add local CSV paths for new tables |
| `src/cloud_train.py` | Add WLST pipeline mode |
| `scripts/cloud_train.sh` | Add `--pipeline wlst` flag |

### Create (New)

| File | Purpose |
|------|---------|
| `src/wlst/__init__.py` | Module init |
| `src/wlst/cohort.py` | TBI cohort + WLST labels |
| `src/wlst/graph_pipeline.py` | 48h-windowed graph construction |
| `src/wlst/event_writers.py` | GCS, vasopressor, vent, neurosurgery, ICP med, MAP writers |
| `src/wlst/features.py` | Tabular feature extraction |
| `src/wlst/experiments.py` | W1-W7 experiment registry |
| `src/wlst/evaluate.py` | WLST metrics + stage comparison |
| `src/wlst/main.py` | CLI entry point |
| `data/mappings/icd10_to_ais_head.json` | ICD-10 → AIS head mapping |
| `data/mappings/icd10_to_charlson.json` | ICD-10 → Charlson mapping |
| `tests/test_wlst/*.py` | Tests |

---

## Verification Plan

| After Phase | Verification |
|-------------|-------------|
| Phase 1 | Run cohort SQL against DuckDB, verify counts (~200-1000), check label balance |
| Phase 3 | Build graph with `--patients-limit 20`, inspect in Neo4j, verify 48h boundary, confirm no code status leakage |
| Phase 4 | Generate feature matrix, check NaN rates, verify distributions |
| Phase 5 | Train locally (small cohort), verify loss decreases, AUROC > 0.5 |
| Phase 6 | Compare Stage 1 vs Stage 2 AUROC, bootstrap significance |
| Phase 7 | Submit Vertex AI job, monitor, download GCS artifacts |

### Smoke Tests

```bash
# Local: full pipeline with 20 patients
python -m src.wlst.main --patients-limit 20 --wlst-stage stage1

# Cloud: full pipeline
./scripts/cloud_train.sh --pipeline wlst --wlst-stage stage1 --patients-limit 50
```
