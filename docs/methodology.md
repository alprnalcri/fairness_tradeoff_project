# Methodology

## Dataset

The Adult Census Income dataset from the UCI Machine Learning Repository was used. The prediction task involves determining whether an individual's annual income exceeds $50,000.

Protected attribute:

* Sex (Female/Male)

## Baseline Models

The following machine learning models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

## Fairness Metrics

To assess group fairness, the following metrics were employed:

* Statistical Parity Difference (SPD)
* Equal Opportunity Difference (EOD)
* Disparate Impact (DI)

## Fairness Interventions

### No-Sex

The protected attribute was removed from the feature set.

### No-Demographic

Multiple demographic variables were removed, including sex, race, marital status, relationship status, and native country.

### Reweighing

Training samples were assigned weights to reduce bias before model training.

### Exponentiated Gradient Reduction

Fairness constraints were incorporated directly into the optimization process using Fairlearn.

## Validation Strategy

A 5-Fold Stratified Cross Validation procedure was performed to evaluate the robustness of the obtained results.
