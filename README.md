# Fairness–Accuracy Trade-Off Analysis in Machine Learning

## Project Overview

This project investigates the relationship between predictive performance and algorithmic fairness using the Adult Census Income dataset.

The study evaluates whether improving fairness necessarily leads to a reduction in predictive performance and examines the effectiveness of different fairness-aware machine learning approaches.

The project was developed as part of the Software Engineering Graduate Program at Samsun University.

---

## Dataset

Adult Census Income Dataset

Target Variable:

* Income > 50K
* Income ≤ 50K

Protected Attribute:

* Sex (Male / Female)

Dataset size after preprocessing:

* 30,162 samples

---

## Experimental Design

The study consists of five experimental stages:

### 1. Baseline Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

### 2. No-Sex Experiment

The protected attribute (sex) was removed to evaluate whether fairness can be achieved through feature removal.

### 3. No-Demographic Experiment

Demographic attributes were removed:

* sex
* race
* native-country
* marital-status
* relationship

### 4. Reweighing

A preprocessing fairness intervention based on sample weighting.

### 5. Exponentiated Gradient

An in-processing fairness-aware optimization approach.

---

## Evaluation Metrics

### Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### Fairness Metrics

* Statistical Parity Difference (SPD)
* Equal Opportunity Difference (EOD)
* Disparate Impact (DI)

---

## Main Findings

### Best Predictive Performance

XGBoost

* Accuracy: 0.8752
* ROC-AUC: 0.9358

### Best Fairness Performance

Logistic Regression + Exponentiated Gradient

* SPD: 0.0133
* DI: 0.9272

### Best Accuracy-Fairness Balance

XGBoost + Reweighing

* Accuracy: 0.8707
* SPD: 0.0893
* DI: 0.6033

---

## Key Conclusions

* Removing protected attributes alone does not eliminate bias.
* Proxy variables continue to carry demographic information.
* Fairness-aware methods significantly improve fairness metrics.
* A measurable trade-off exists between predictive performance and fairness.

---

## Technologies

* Python 3.10
* Scikit-Learn
* Fairlearn
* XGBoost
* Pandas
* NumPy
* Matplotlib

---

## Author

Alperen A.
Software Engineering Graduate Student
Samsun University
