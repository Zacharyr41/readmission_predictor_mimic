# WLST Evaluation Report — logistic_regression (stage1)

## Performance Metrics

| Metric | Value |
|---|---|
| **AUROC** | 0.4809 |
| **AUPRC** | 0.1036 |
| **Brier Score** | 0.2487 |
| **Sensitivity** | 0.2000 |
| **Specificity** | 0.9574 |
| **Sens @ 90% Spec** | 0.2000 |
| **Sens @ 95% Spec** | 0.2000 |
| **Test Set Size** | 52 (pos=5, neg=47) |

## Confusion Matrix

| | Predicted No WLST | Predicted WLST |
|---|---|---|
| **Actual No WLST** | 45 | 2 |
| **Actual WLST** | 4 | 1 |

## Subgroup Analysis

| Category | N | AUROC | Mean P(WLST) |
|---|---|---|---|
| full_code_death | 16 | N/A | 0.5247 |
| full_code_survived | 33 | 0.5968 | 0.1703 |
