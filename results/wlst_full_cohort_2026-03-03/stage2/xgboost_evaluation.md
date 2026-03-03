# WLST Evaluation Report — xgboost (stage2)

## Performance Metrics

| Metric | Value |
|---|---|
| **AUROC** | 0.4255 |
| **AUPRC** | 0.0776 |
| **Brier Score** | 0.1655 |
| **Sensitivity** | 1.0000 |
| **Specificity** | 0.1277 |
| **Sens @ 90% Spec** | 0.0000 |
| **Sens @ 95% Spec** | 0.0000 |
| **Test Set Size** | 52 (pos=5, neg=47) |

## Confusion Matrix

| | Predicted No WLST | Predicted WLST |
|---|---|---|
| **Actual No WLST** | 6 | 41 |
| **Actual WLST** | 0 | 5 |

## Subgroup Analysis

| Category | N | AUROC | Mean P(WLST) |
|---|---|---|---|
| full_code_death | 16 | N/A | 0.2680 |
| full_code_survived | 33 | 0.6613 | 0.0688 |
