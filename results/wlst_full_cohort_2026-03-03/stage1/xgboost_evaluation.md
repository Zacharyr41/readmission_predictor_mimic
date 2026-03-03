# WLST Evaluation Report — xgboost (stage1)

## Performance Metrics

| Metric | Value |
|---|---|
| **AUROC** | 0.3872 |
| **AUPRC** | 0.0724 |
| **Brier Score** | 0.1662 |
| **Sensitivity** | 1.0000 |
| **Specificity** | 0.1702 |
| **Sens @ 90% Spec** | 0.0000 |
| **Sens @ 95% Spec** | 0.0000 |
| **Test Set Size** | 52 (pos=5, neg=47) |

## Confusion Matrix

| | Predicted No WLST | Predicted WLST |
|---|---|---|
| **Actual No WLST** | 8 | 39 |
| **Actual WLST** | 0 | 5 |

## Subgroup Analysis

| Category | N | AUROC | Mean P(WLST) |
|---|---|---|---|
| full_code_death | 16 | N/A | 0.2490 |
| full_code_survived | 33 | 0.5806 | 0.0777 |
