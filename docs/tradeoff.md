---

layout: default
title: Accuracy–Fairness Trade-off
----------------------------------

# Accuracy–Fairness Trade-off

This section analyzes the relationship between predictive performance and fairness.

---

## Trade-off Visualization

![Accuracy vs SPD](assets/images/accuracy_spd_tradeoff_clean.png)

---

## Interpretation

* Models with higher accuracy (e.g., XGBoost) tend to exhibit higher demographic disparity.
* Fairness-aware methods reduce bias but may slightly reduce predictive performance.
* No model achieves both maximum accuracy and maximum fairness simultaneously.

---

## Key Insight

This result confirms that fairness and accuracy are inherently competing objectives in machine learning systems.
