---
title: "The Price of Fairness in Algorithmic Decision-Making"
---

# The Price of Fairness in Algorithmic Decision-Making

## Machine Learning Research Project

This study investigates the **trade-off between predictive performance and algorithmic fairness** using the **Adult Census Income dataset**.

The goal is to evaluate whether improving fairness in machine learning systems inevitably leads to reduced predictive performance, and to explore mitigation strategies that balance both objectives.

---

## 📌 Research Problem

Modern machine learning systems often achieve high predictive accuracy but may exhibit **systematic demographic bias**.

This raises a critical question:

> Can we achieve both high accuracy and fair decision-making simultaneously?

---

## 📊 Dataset

- **Dataset:** Adult Census Income (UCI Repository)
- **Task:** Binary classification (Income >50K vs ≤50K)
- **Sensitive Attribute:** Sex (Gender)
- **Size:** ~30,000 samples after preprocessing

---

## 🤖 Machine Learning Models

We evaluate both interpretable and ensemble-based models:

- Logistic Regression (baseline linear model)
- Decision Tree (rule-based model)
- Random Forest (ensemble learning)
- XGBoost (gradient boosting model)

---

## ⚖️ Fairness Interventions

To mitigate algorithmic bias, we test multiple strategies:

### Data-level methods
- No-Sex (removal of gender attribute)
- No-Demographic (removal of sensitive proxy features)

### Algorithm-level methods
- Reweighing (pre-processing rebalancing technique)
- Exponentiated Gradient (in-processing fairness constraint optimization)

---

## 📈 Key Findings

Experimental results reveal the following insights:

- **XGBoost** achieves the highest predictive performance across all models.
- Removing protected attributes alone does not significantly reduce bias due to **proxy variables**.
- Fairness-aware methods improve demographic parity at the cost of performance.
- **Reweighing** provides the best balance between accuracy and fairness.
- **ExpGrad** achieves the strongest fairness improvement but introduces higher performance loss.
- Cross-validation confirms that results are stable across different data splits.

---

## 🧠 Main Insight

> Fairness and accuracy are inherently competing objectives in machine learning systems, requiring explicit trade-off optimization rather than feature removal alone.

---

## 📊 Results Summary

Detailed experimental results include:

- Model performance comparison (Accuracy, ROC-AUC, F1-score)
- Fairness metrics (SPD, EOD, DI)
- Trade-off analysis between accuracy and fairness
- Cross-validation stability analysis

👉 See full results: `results.md`

---

## 🔗 Project Structure

- 📖 Literature Review → theoretical background
- ⚙️ Methodology → data & models
- 📊 Results → experimental evaluation
- ⚖️ Fairness Analysis → bias investigation
- 📉 Trade-off → accuracy vs fairness
- 🧪 Cross-validation → robustness check
- 🎯 Conclusion → final insights

---

## 📂 Repository

Source code and experiments:

👉 https://github.com/alprnalcri/fairness_tradeoff_project

---

## 📄 Report

Full academic report is available in the `/report` directory.

---

## ✨ Contribution

This project demonstrates that:

- fairness cannot be achieved by simply removing sensitive attributes
- proxy variables preserve bias in machine learning systems
- fairness-aware learning methods are necessary for robust mitigation