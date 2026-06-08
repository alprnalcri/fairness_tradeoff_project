# The Price of Fairness in Algorithmic Decision-Making

## Project Overview

This project investigates the trade-off between predictive performance and fairness in machine learning models using the Adult Census Income dataset.

The objective is to determine whether improving fairness inevitably reduces predictive performance and to identify mitigation strategies that provide a balanced solution.

---

## Dataset

* **Dataset:** Adult Census Income Dataset (UCI Repository)
* **Protected Attribute:** Sex
* **Prediction Task:** Income >50K vs ≤50K

---

## Models Evaluated

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

---

## Fairness Interventions

* No-Sex
* No-Demographic
* Reweighing
* Exponentiated Gradient (ExpGrad)

---

## Key Findings

* XGBoost achieved the highest predictive performance.
* Removing protected attributes alone did not eliminate bias.
* ExpGrad achieved the strongest fairness performance.
* XGBoost + Reweighing provided the best balance between accuracy and fairness.
* 5-Fold Cross Validation confirmed the robustness of the findings.

---

## Repository

[GitHub Repository](https://github.com/alprnalcri/fairness_tradeoff_project)

## Report

The full report is available in the repository under the `report` directory.
