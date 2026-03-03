# WLST Full-Cohort Run Results

**Date:** 2026-03-03
**Commit:** `49da56b` (seq_num fix + NA handling fix)
**Infrastructure:** Vertex AI, n1-highmem-8 (52 GB) + T4 GPU
**Total Runtime:** Stage 1: 21.7 min, Stage 2: 9.9 min

---

## Cohort Summary

| Metric | Value |
|--------|-------|
| **TBI cohort size** | 344 patients |
| **Unique subjects** | 342 |
| **WLST positive (label=1)** | 31 (9.0%) |
| **WLST negative (label=0)** | 313 (91.0%) |
| **In-hospital mortality** | 73 + 4 + 3 = 80 (23.3%) |
| **Age** | Mean 55.0, Median 57.0, Range 18-89 |
| **Male** | 67.2% |
| **GCS** | Mean 5.2, Median 6.0, Range 2-8 |
| **Ventilated** | 65.1% |
| **Any neurosurgery** | 45.3% |
| **Vasopressor use** | 29.4% |

### Outcome Categories

| Category | N | % |
|----------|---|---|
| Full code, survived | 251 | 73.0% |
| Full code, death | 73 | 21.2% |
| Other | 8 | 2.3% |
| Limited code, survived | 5 | 1.5% |
| CMO, death | 4 | 1.2% |
| DNR, death | 3 | 0.9% |

### Code Status Changes

- Median time to code change: **71.0 hours**
- Within 48h feature window: 6 (30.0%)
- After 48h feature window: 14 (70.0%)

---

## Knowledge Graph

| Metric | Value |
|--------|-------|
| **Total RDF triples** | 568,622 (Stage 1) / 569,652 (Stage 2) |
| **Total nodes** | 64,655 |
| **Total edges** | 198,254 |
| **Weakly connected components** | 7 |
| **Events per ICU day** | 12.0 |
| **Mean degree** | 6.13 |
| **Max degree** | 1,593 |

### Node Type Distribution

| Node Type | Count |
|-----------|-------|
| MAP events | 17,598 |
| GCS events | 9,422 |
| Diagnoses | 8,477 |
| Biomarker events | 6,427 |
| ICU days | 3,065 |
| Vasopressor events | 2,346 |
| ICP medication events | 659 |
| Hospital admissions | 344 |
| ICU stays | 344 |
| Patients | 342 |
| Ventilation events | 225 |
| Neurosurgery events | 168 |

Stage 2 adds non-clinical edges: `hasHospitalService`, `hasLanguage`, `hasTransferCount` (1,030 extra triples).

---

## Model Performance

### Stage 1: Clinical Trajectory Only

**GNN Ablation Study**

| Experiment | AUROC | AUPRC | F1 | Best Epoch |
|------------|-------|-------|----|------------|
| W1: MLP baseline | 0.575 | 0.115 | 0.222 | 81 |
| W2: Transformer only | 0.417 | 0.077 | 0.195 | 23 |
| W3: Transformer + temporal | 0.630 | 0.151 | 0.375 | 9 |
| **W4: Full model (GNN + transformer + temporal)** | **0.728** | **0.487** | **0.571** | **38** |

**Classical Baselines**

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|-------|-------|-------|-------------|-------------|
| Logistic regression | 0.481 | 0.104 | 0.200 | 0.957 |
| XGBoost | 0.387 | 0.072 | 1.000 | 0.170 |

### Stage 2: Non-Clinical Confounders Added

| Model | AUROC | AUPRC | F1 |
|-------|-------|-------|----|
| W6: Full model (stage2) | 0.528 | 0.091 | 0.222 |
| Logistic regression | 0.515 | 0.109 | — |
| XGBoost | 0.426 | 0.078 | — |

### Stage 1 vs Stage 2 Comparison (Best GNN)

| Metric | Stage 1 (W4) | Stage 2 (W6) | Delta |
|--------|-------------|-------------|-------|
| AUROC | **0.728** | 0.528 | -0.200 |
| AUPRC | **0.487** | 0.091 | -0.396 |
| F1 | **0.571** | 0.222 | -0.349 |
| Precision | 1.000 | 0.125 | -0.875 |
| Recall | 0.400 | 1.000 | +0.600 |

---

## Timing Breakdown

| Step | Stage 1 (min) | Stage 2 (min) |
|------|---------------|---------------|
| BigQuery ingestion | 2.6 | 1.8 |
| Graph construction | 2.7 | 2.6 |
| Feature extraction | 0.0 | 0.0 |
| Classical baselines | 0.1 | 0.0 |
| GNN preparation (SapBERT) | 2.6 | 2.5 |
| GNN training | 13.7 | 2.8 |
| **Total** | **21.7** | **9.9** |

---

## Run IDs

| Stage | Vertex AI Job ID | Run ID |
|-------|-----------------|--------|
| Stage 1 | `4850557032685633536` | wlst_E6_full_model_1772490868 |
| Stage 2 | `5399644343503945728` | wlst_E6_full_model_1772492230 |

---

## Interpretation

See `analysis.md` in this folder for detailed interpretation of results.
