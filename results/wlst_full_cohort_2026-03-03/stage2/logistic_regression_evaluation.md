# WLST Evaluation Report — logistic_regression (stage2)

## Performance Metrics

| Metric | Value |
|---|---|
| **AUROC** | 0.5149 |
| **AUPRC** | 0.1087 |
| **Brier Score** | 0.2276 |
| **Sensitivity** | 0.8000 |
| **Specificity** | 0.4043 |
| **Sens @ 90% Spec** | 0.2000 |
| **Sens @ 95% Spec** | 0.2000 |
| **Test Set Size** | 52 (pos=5, neg=47) |

## Confusion Matrix

| | Predicted No WLST | Predicted WLST |
|---|---|---|
| **Actual No WLST** | 19 | 28 |
| **Actual WLST** | 1 | 4 |

## Subgroup Analysis

| Category | N | AUROC | Mean P(WLST) |
|---|---|---|---|
| full_code_death | 16 | N/A | 0.4993 |
| full_code_survived | 33 | 0.5968 | 0.1544 |
