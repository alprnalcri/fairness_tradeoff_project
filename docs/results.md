---
layout: default
title: Results
---

# Results

This section presents experimental results in terms of predictive performance, fairness evaluation, and robustness.

---

## Predictive Performance (Baseline Models)

Among baseline models, **XGBoost achieved the best performance** in terms of Accuracy and ROC-AUC.

| Model                | Accuracy | ROC-AUC |
|---------------------|----------|----------|
| Logistic Regression | 0.8589   | 0.9180   |
| Decision Tree       | 0.8155   | 0.7512   |
| Random Forest       | 0.8560   | 0.9083   |
| XGBoost             | 0.8752   | 0.9358   |

---

## Fairness Analysis

Although XGBoost performs best in accuracy, all models exhibit measurable demographic disparity.

This confirms that **high accuracy does not guarantee fairness**.

---

## Fairness Interventions

Different mitigation strategies were evaluated:

- **No-Sex / No-Demographic**: Removing sensitive features alone does not eliminate bias.
- **Reweighing**: Improves fairness while preserving performance.
- **Exponentiated Gradient (ExpGrad)**: Strongest fairness improvement but with higher performance cost.

---

## Cross-Validation Results

A 5-Fold Stratified Cross Validation was applied.

| Model                 | Accuracy        | SPD             | DI              |
|----------------------|-----------------|-----------------|-----------------|
| XGBoost              | 0.8705 ± 0.0025 | 0.1890 ± 0.0075 | 0.3070 ± 0.0249 |
| XGBoost + Reweighing | 0.8649 ± 0.0042 | 0.1074 ± 0.0128 | 0.5391 ± 0.0462 |

---

## Fairness–Accuracy Trade-off

Results clearly show a trade-off:

- Higher accuracy → higher disparity
- Fairer models → slight accuracy drop

This confirms the inherent conflict between fairness and performance.

---

## Key Insight

The best balance is achieved by **XGBoost + Reweighing**, which provides strong accuracy while significantly improving fairness.