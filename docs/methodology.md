---
title: "Methodology"
---

# Methodology

This section describes the dataset, preprocessing pipeline, machine learning models, fairness evaluation metrics, and experimental design used in this study.

---

## Dataset

The **Adult Census Income dataset** from the UCI Machine Learning Repository is used in this study.

The task is formulated as a binary classification problem:

- Income ≤ 50K
- Income > 50K

### Protected Attribute

- Sex (Male / Female)

This attribute is used to evaluate group fairness across demographic groups.

---

## Data Preprocessing

The dataset undergoes the following preprocessing steps:

- Removal of missing or unknown values (e.g., "?")
- One-Hot Encoding for categorical variables
- Standardization of numerical features using StandardScaler
- Stratified splitting into training and test sets (80% / 20%)

These steps ensure that the model receives clean and normalized input while preserving class distribution.

---

## Baseline Models

The following machine learning models are used for baseline comparison:

- Logistic Regression (interpretable linear model)
- Decision Tree (rule-based model)
- Random Forest (ensemble bagging method)
- XGBoost (gradient boosting model)

These models represent a spectrum from simple to complex learning approaches.

---

## Fairness Metrics

To evaluate group fairness, the following metrics are used:

### Statistical Parity Difference (SPD)
Measures the difference in positive prediction rates between protected and unprotected groups.

### Equal Opportunity Difference (EOD)
Measures the difference in true positive rates across groups.

### Disparate Impact (DI)
Measures the ratio of positive prediction rates between groups.

These metrics are widely used in algorithmic fairness literature to quantify group-level bias.

---

## Fairness Interventions

To mitigate bias, both preprocessing and in-processing methods are applied.

### No-Sex
The protected attribute (sex) is removed from the dataset to evaluate fairness-through-unawareness.

### No-Demographic
Multiple correlated demographic features are removed:

- sex
- race
- marital-status
- relationship
- native-country

This setup tests whether removing sensitive attributes is sufficient to reduce bias.

### Reweighing
A preprocessing technique that assigns weights to training instances to balance representation across groups.

### Exponentiated Gradient Reduction (ExpGrad)
An in-processing method implemented via Fairlearn that enforces fairness constraints during model optimization.

---

## Experimental Design

The study follows a structured experimental pipeline:

1. Baseline training using all features
2. Feature removal experiments (No-Sex, No-Demographic)
3. Preprocessing-based mitigation (Reweighing)
4. In-processing fairness optimization (ExpGrad)

This design enables a comparative analysis of fairness interventions.

---

## Validation Strategy

A **5-Fold Stratified Cross Validation** approach is used to ensure robustness.

- Dataset is split into 5 folds
- Each fold is used once as test set
- Remaining folds are used for training
- Mean and standard deviation of metrics are reported

This ensures that results are not dependent on a single train-test split.