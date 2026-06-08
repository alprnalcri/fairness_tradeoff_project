---
layout: default
title: Conclusion
---

# Conclusion

This study analyzed the trade-off between predictive performance and fairness using the Adult Census Income dataset.

---

## Key Findings

- High-performing models (e.g., XGBoost) achieve strong accuracy but exhibit demographic bias.
- Removing sensitive attributes alone is insufficient due to proxy variables.
- Fairness-aware methods improve fairness with varying trade-offs:
  - Reweighing → balanced trade-off
  - ExpGrad → strongest fairness, higher cost
- Results are stable under cross-validation.

---

## Main Insight

Fairness and accuracy are inherently competing objectives in machine learning systems.

The best trade-off is achieved by **XGBoost + Reweighing**.

---

## Limitations

- Only one dataset used
- Only one protected attribute (sex)
- Limited fairness methods evaluated

---

## Future Work

- Multi-attribute fairness (race, age, intersectionality)
- Advanced debiasing (adversarial methods)
- Explainability (SHAP-based analysis)

---

## Final Remark

Fair machine learning is a multi-objective optimization problem that requires balancing performance, fairness, and interpretability.