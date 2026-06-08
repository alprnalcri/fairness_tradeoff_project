---
title: "Results"
---

# Results

This section presents the experimental results in terms of predictive performance, fairness evaluation, and robustness analysis.

---

## Predictive Performance (Baseline Models)

Among the baseline models, **XGBoost achieved the best predictive performance** in terms of both Accuracy and ROC-AUC.

| Model               | Accuracy | ROC-AUC |
|--------------------|----------|----------|
| Logistic Regression | 0.8589   | 0.9180   |
| Decision Tree       | 0.8155   | 0.7512   |
| Random Forest       | 0.8560   | 0.9083   |
| XGBoost             | 0.8752   | 0.9358   |

These results indicate that ensemble-based methods, particularly XGBoost, are more effective in capturing complex patterns in the dataset.

---

## Fairness Analysis

Although XGBoost achieved the highest predictive performance, all baseline models exhibited measurable levels of group disparity.

This confirms that **high accuracy does not necessarily imply fair decision-making**.

---

## Fairness Interventions

Different fairness-aware strategies were evaluated:

- Removing protected attributes (No-Sex, No-Demographic) did not significantly eliminate bias.
- Preprocessing-based **Reweighing** reduced disparity while preserving predictive performance.
- In-processing **Exponentiated Gradient (ExpGrad)** achieved the strongest fairness improvement.

Overall, fairness interventions demonstrated different trade-offs between accuracy and fairness.

---

## Cross-Validation Results

To ensure robustness, a 5-Fold Stratified Cross Validation was performed.

| Model                | Accuracy        | SPD             | DI              |
|---------------------|-----------------|-----------------|-----------------|
| XGBoost              | 0.8705 ± 0.0025 | 0.1890 ± 0.0075 | 0.3070 ± 0.0249 |
| XGBoost + Reweighing | 0.8649 ± 0.0042 | 0.1074 ± 0.0128 | 0.5391 ± 0.0462 |

Results show that Reweighing consistently improves fairness metrics across different data splits while maintaining stable predictive performance.

---

## Fairness–Accuracy Trade-off

The experimental results clearly demonstrate a **trade-off between predictive performance and fairness**:

- Higher accuracy models (e.g., XGBoost) tend to exhibit higher disparity.
- Fairness-aware models reduce bias but may slightly decrease accuracy.

This confirms that achieving both objectives simultaneously requires careful balancing of model design and mitigation strategies.

---

## Key Insight

The best overall trade-off is achieved by **XGBoost + Reweighing**, which maintains high predictive performance while significantly improving fairness metrics.
---

## Final Model Comparison

![Final Comparison](assets/images/final_comparison.png)

---

## Summary

- Best Accuracy: XGBoost
- Best Fairness: ExpGrad
- Best Trade-off: XGBoost + Reweighing