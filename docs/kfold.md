---
title: "Cross-Validation Analysis"
---

# Cross-Validation Results

This section evaluates model stability across different data splits.

---

## XGBoost Stability

![KFold Summary](assets/images/kfold_xgboost_summary.png)

---

## Fairness Stability

![Fairness Summary](assets/images/kfold_fairness_summary.png)

---

## Key Findings

- XGBoost shows stable performance across folds.
- Fairness metrics remain consistent under cross-validation.
- Reweighing improves fairness without destabilizing the model.

---

## Conclusion

The results confirm that both performance and fairness improvements are not dependent on a single train-test split.