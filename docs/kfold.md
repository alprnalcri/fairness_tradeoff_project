---
layout: default
title: Cross-Validation Analysis
---

# Cross-Validation Analysis

This section evaluates model stability across different data splits using 5-Fold Stratified Cross Validation.

---

## XGBoost Stability

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.8705 ± 0.0025 |
| SPD | 0.1890 ± 0.0075 |
| DI | 0.3070 ± 0.0249 |

---

## Fairness Stability (Reweighing)

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.8649 ± 0.0042 |
| SPD | 0.1074 ± 0.0128 |
| DI | 0.5391 ± 0.0462 |

---

## Key Findings

- XGBoost is stable across different folds
- Fairness metrics remain consistent
- Reweighing improves fairness without instability

---

## Key Insight

Results confirm that fairness improvements are robust and not dependent on a single train-test split.