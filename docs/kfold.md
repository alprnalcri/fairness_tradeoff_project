---
title: "Cross-Validation Analysis"
---

# Cross-Validation Results

This section evaluates model stability across different data splits.

---

## XGBoost Stability

![KFold Summary](assets/tables/kfold_xgboost_summary.csv)

---

## Fairness Stability

![Fairness Summary](assets/tables/kfold_fairness_summary.csv)

---

## Key Findings

- XGBoost shows stable performance across folds.
- Fairness metrics remain consistent under cross-validation.
- Reweighing improves fairness without destabilizing the model.

---

## Conclusion

The results confirm that both performance and fairness improvements are not dependent on a single train-test split.