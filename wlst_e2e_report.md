# WLST E2E Pipeline Report

## Run Metadata

- **Stage 1 Run ID:** wlst_E6_full_model_1772485860
- **Stage 2 Run ID:** wlst_E6_full_model_1772486402
- **GCP Project:** mimic-485500
- **Patients Limit:** 50
- **Seed:** 42
- **Run All:** True
- **Stage 1 Total Runtime:** 6.2 min
- **Stage 2 Total Runtime:** 5.0 min

## Cohort

# WLST Cohort Summary

## Cohort Size: 10 patients

## WLST Label Distribution
- WLST positive (label=1): 2 (20.0%)
- WLST negative (label=0): 8 (80.0%)


## Outcome Categories
- full_code_survived: 6 (60.0%)
- full_code_death: 2 (20.0%)
- CMO_death: 1 (10.0%)
- limited_code_survived: 1 (10.0%)

## GCS Distribution
- Mean: 5.9
- Median: 6.5
- Range: 3 - 7

## Time to Code Status Change
- Median: 56.0 hours
- Within 48h: 1 (50.0%)
- After 48h: 1 (50.0%)

**Patient IDs (10 patients):** [10037861, 10046724, 10050445, 10058575, 10092572, 10104012, 10135398, 10292285, 10317167, 10476294]

## Stage 1: Clinical Trajectory

### Graph Structure

## Summary

- **Total Nodes**: 1897
- **Total Edges**: 5506
- **Total RDF Triples**: 15418

## Node Counts

| Type | Count |
|------|-------|
| MAPEvent | 493 |
| GCSEvent | 307 |
| DiagnosisEvent | 182 |
| BioMarkerEvent | 133 |
| ICUDay | 79 |
| VasopressorEvent | 70 |
| ICPMedicationEvent | 21 |
| VentilationEvent | 10 |
| Patient | 10 |
| HospitalAdmission | 10 |
| ICUStay | 10 |
| NeurosurgeryEvent | 6 |

## Edge Counts

| Predicate | Count |
|-----------|-------|
| inXSDDateTimeStamp | 1349 |
| associatedWithICUDay | 1040 |
| hasICUStayEvent | 1040 |
| hasICUDayEvent | 1040 |
| associatedWithICUStay | 1040 |
| hasItemId | 642 |
| hasValue | 626 |
| hasMeasurementMethod | 493 |
| hasGCSMotor | 307 |
| hasGCSVerbal | 307 |
| hasGCSTotal | 307 |
| hasGCSEye | 307 |
| hasSnomedCode | 247 |
| hasSnomedTerm | 247 |
| hasSnomedConcept | 247 |
| hasBeginning | 206 |
| hasEnd | 206 |
| hasIcdVersion | 182 |
| hasIcdCode | 182 |
| hasSequenceNumber | 182 |
| hasLongTitle | 182 |
| diagnosisOf | 182 |
| hasDiagnosis | 182 |
| hasRefRangeLower | 133 |
| hasBiomarkerType | 133 |
| hasRefRangeUpper | 133 |
| hasFluid | 133 |
| hasCategory | 133 |
| hasUnit | 133 |
| hasAmountUnit | 91 |
| hasDrugName | 91 |
| hasAmount | 91 |
| hasRateUnit | 82 |
| hasRate | 82 |
| partOf | 79 |
| hasDayNumber | 79 |
| hasICUDay | 79 |
| hasProcedureName | 16 |
| hasStayId | 10 |
| hasWLSTLabel | 10 |
| hasAdmissionType | 10 |
| hasAdmission | 10 |
| hasAdmissionId | 10 |
| unitType | 10 |
| hasGender | 10 |
| numericDuration | 10 |
| containsICUStay | 10 |
| readmittedWithin30Days | 10 |
| admissionOf | 10 |
| hasSubjectId | 10 |
| hasDischargeLocation | 10 |
| readmittedWithin60Days | 10 |
| hasDuration | 10 |
| hasAge | 10 |

## Degree Distribution

- **Mean**: 5.80
- **Median**: 4.00
- **Max**: 340
- **Std Dev**: 19.67

## Connected Components

- **Weakly Connected Components**: 7

## Temporal Density

- **Events per ICU Day**: 13.16

### HeteroData Summary

**Label distribution:** positive=2, negative=8, total=10

### Feature Matrix

| Feature | Count | Mean | Std | Min | Max |
|---------|-------|------|-----|-----|-----|
| count | — | — | — | — | — |
| unique | — | — | — | — | — |
| top | — | — | — | — | — |
| freq | — | — | — | — | — |
| mean | — | — | — | — | — |
| std | — | — | — | — | — |
| min | — | — | — | — | — |
| 25% | — | — | — | — | — |
| 50% | — | — | — | — | — |
| 75% | — | — | — | — | — |
| max | — | — | — | — | — |

### Classical Baselines

| Model | AUROC | AUPRC | Brier | Sensitivity | Specificity |
|-------|-------|-------|-------|-------------|-------------|
| logistic_regression | nan | nan | nan | nan | nan |
| xgboost | nan | nan | nan | nan | nan |

### GNN Experiments

| Experiment | AUROC | AUPRC | Brier |
|------------|-------|-------|-------|
| W1_mlp_baseline | nan | nan | — |
| W2_transformer_only | nan | nan | — |
| W3_transformer_temporal | nan | nan | — |
| W4_full_model | nan | nan | — |

#### Ablation Comparison

| Experiment | AUROC | AUPRC | F1 | Best Epoch | Time (s) |
|------------|-------|-------|----|------------|----------|
| W1_mlp_baseline | nan | nan | nan | 0 | 1.3 |
| W2_transformer_only | nan | nan | nan | 0 | 4.2 |
| W3_transformer_temporal | nan | nan | nan | 0 | 3.1 |
| W4_full_model | nan | nan | nan | 0 | 5.2 |


## Stage 2: Non-Clinical Confounders

### Graph Structure

## Summary

- **Total Nodes**: 1897
- **Total Edges**: 5506
- **Total RDF Triples**: 15448

## Node Counts

| Type | Count |
|------|-------|
| MAPEvent | 493 |
| GCSEvent | 307 |
| DiagnosisEvent | 182 |
| BioMarkerEvent | 133 |
| ICUDay | 79 |
| VasopressorEvent | 70 |
| ICPMedicationEvent | 21 |
| VentilationEvent | 10 |
| Patient | 10 |
| HospitalAdmission | 10 |
| ICUStay | 10 |
| NeurosurgeryEvent | 6 |

## Edge Counts

| Predicate | Count |
|-----------|-------|
| inXSDDateTimeStamp | 1349 |
| associatedWithICUDay | 1040 |
| hasICUStayEvent | 1040 |
| hasICUDayEvent | 1040 |
| associatedWithICUStay | 1040 |
| hasItemId | 642 |
| hasValue | 626 |
| hasMeasurementMethod | 493 |
| hasGCSVerbal | 307 |
| hasGCSTotal | 307 |
| hasGCSEye | 307 |
| hasGCSMotor | 307 |
| hasSnomedConcept | 247 |
| hasSnomedCode | 247 |
| hasSnomedTerm | 247 |
| hasBeginning | 206 |
| hasEnd | 206 |
| hasSequenceNumber | 182 |
| hasLongTitle | 182 |
| diagnosisOf | 182 |
| hasDiagnosis | 182 |
| hasIcdCode | 182 |
| hasIcdVersion | 182 |
| hasUnit | 133 |
| hasBiomarkerType | 133 |
| hasFluid | 133 |
| hasRefRangeLower | 133 |
| hasRefRangeUpper | 133 |
| hasCategory | 133 |
| hasAmount | 91 |
| hasAmountUnit | 91 |
| hasDrugName | 91 |
| hasRateUnit | 82 |
| hasRate | 82 |
| hasICUDay | 79 |
| hasDayNumber | 79 |
| partOf | 79 |
| hasProcedureName | 16 |
| containsICUStay | 10 |
| hasTransferCount | 10 |
| hasAdmissionType | 10 |
| hasAdmissionId | 10 |
| readmittedWithin60Days | 10 |
| hasDischargeLocation | 10 |
| hasAdmission | 10 |
| hasDuration | 10 |
| hasGender | 10 |
| admissionOf | 10 |
| hasSubjectId | 10 |
| hasHospitalService | 10 |
| hasStayId | 10 |
| hasWLSTLabel | 10 |
| numericDuration | 10 |
| hasAge | 10 |
| hasLanguage | 10 |
| readmittedWithin30Days | 10 |
| unitType | 10 |

## Degree Distribution

- **Mean**: 5.80
- **Median**: 4.00
- **Max**: 340
- **Std Dev**: 19.67

## Connected Components

- **Weakly Connected Components**: 7

## Temporal Density

- **Events per ICU Day**: 13.16

### HeteroData Summary

**Label distribution:** positive=2, negative=8, total=10

### Feature Matrix

| Feature | Count | Mean | Std | Min | Max |
|---------|-------|------|-----|-----|-----|
| count | — | — | — | — | — |
| unique | — | — | — | — | — |
| top | — | — | — | — | — |
| freq | — | — | — | — | — |
| mean | — | — | — | — | — |
| std | — | — | — | — | — |
| min | — | — | — | — | — |
| 25% | — | — | — | — | — |
| 50% | — | — | — | — | — |
| 75% | — | — | — | — | — |
| max | — | — | — | — | — |

### Classical Baselines

| Model | AUROC | AUPRC | Brier | Sensitivity | Specificity |
|-------|-------|-------|-------|-------------|-------------|
| logistic_regression | nan | nan | nan | nan | nan |
| xgboost | nan | nan | nan | nan | nan |

### GNN Experiments

| Experiment | AUROC | AUPRC | Brier |
|------------|-------|-------|-------|
| W6_stage2_full_model | nan | nan | — |

#### Ablation Comparison

| Experiment | AUROC | AUPRC | F1 | Best Epoch | Time (s) |
|------------|-------|-------|----|------------|----------|
| W6_stage2_full_model | nan | nan | nan | 0 | 6.0 |


## Stage 1 vs Stage 2 Comparison

| Model | S1 AUROC | S2 AUROC | Delta | S1 AUPRC | S2 AUPRC | Delta |
|-------|----------|----------|-------|----------|----------|-------|
| logistic_regression | nan | nan | +nan | nan | nan | +nan |
| xgboost | nan | nan | +nan | nan | nan | +nan |

## Timing Breakdown

| Step | Stage 1 (min) | Stage 2 (min) |
|------|---------------|---------------|
| baselines | 0.1 | 0.1 |
| feature_extraction | 0.0 | 0.0 |
| gnn_preparation | 3.0 | 2.6 |
| gnn_training | 0.2 | 0.1 |
| graph_construction | 0.2 | 0.1 |
| ingestion | 2.5 | 2.1 |
| **Total** | **6.2** | **5.0** |
