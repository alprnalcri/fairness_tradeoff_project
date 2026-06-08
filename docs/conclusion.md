---
title: "Conclusion"
---

# Conclusion

This study investigated the trade-off between predictive performance and fairness in machine learning models using the Adult Census Income dataset.

The main objective was to analyze whether improving fairness inevitably leads to a reduction in predictive performance and to evaluate mitigation strategies that balance both objectives.

---

## Key Findings

The experimental results demonstrate several important conclusions:

- High-performing models such as **XGBoost** achieve strong predictive accuracy but exhibit measurable levels of demographic bias.
- Removing protected attributes (No-Sex and No-Demographic) alone is not sufficient to eliminate algorithmic bias, indicating the presence of proxy variables.
- Fairness-aware methods significantly improve group fairness:
  - **Reweighing** provides a balanced trade-off between fairness and accuracy.
  - **Exponentiated Gradient (ExpGrad)** achieves the strongest fairness improvements but introduces a larger performance cost.
- Cross-validation results confirm that these findings are stable across different data splits.

---

## Main Contribution

This work demonstrates that:

> Improving fairness in machine learning systems often introduces a measurable trade-off with predictive performance.

Among all evaluated approaches, **XGBoost + Reweighing** provides the most balanced solution between accuracy and fairness.

---

## Limitations

This study has several limitations:

- Only a single dataset (Adult Census Income) was used.
- Fairness analysis was limited to the **sex** attribute.
- Only two fairness mitigation techniques were evaluated in-depth.
- Deep interpretability analysis (e.g., SHAP-based explanations) was not included.

---

## Future Work

Future research can extend this work in several directions:

- Inclusion of additional protected attributes such as race and age.
- Evaluation of intersectional fairness (multiple sensitive attributes simultaneously).
- Integration of advanced fairness techniques such as adversarial debiasing and post-processing methods.
- Application of explainable AI methods (e.g., SHAP) to better understand bias sources.

---

## Final Remark

This study highlights that algorithmic fairness is not only a technical optimization problem but also a **multi-objective design challenge** involving ethical and societal considerations.

Therefore, real-world machine learning systems should balance both fairness and accuracy rather than optimizing for only one objective.