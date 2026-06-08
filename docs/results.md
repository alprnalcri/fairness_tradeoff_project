# Results

## Predictive Performance

XGBoost achieved the best predictive performance among baseline models.

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.8589   | 0.9180  |
| Decision Tree       | 0.8155   | 0.7512  |
| Random Forest       | 0.8560   | 0.9083  |
| XGBoost             | 0.8752   | 0.9358  |

## Fairness Interventions

* Removing protected attributes alone did not eliminate bias.
* ExpGrad achieved the strongest fairness improvements.
* Reweighing provided the best balance between predictive performance and fairness.

## Cross-Validation Results

| Model                | Accuracy        | SPD             | DI              |
| -------------------- | --------------- | --------------- | --------------- |
| XGBoost              | 0.8705 ± 0.0025 | 0.1890 ± 0.0075 | 0.3070 ± 0.0249 |
| XGBoost + Reweighing | 0.8649 ± 0.0042 | 0.1074 ± 0.0128 | 0.5391 ± 0.0462 |

The cross-validation analysis confirmed that fairness improvements obtained through Reweighing remained consistent across different data partitions.
