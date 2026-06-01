import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate
)

OUTPUT_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/processed/adult_clean.csv")

TARGET = "income"
SENSITIVE = "sex"

X = df.drop(columns=[TARGET])
y = df[TARGET]
sensitive = df[SENSITIVE]

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            categorical_cols
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_cols
        )
    ]
)

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X,
    y,
    sensitive,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Preprocessing...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = (
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

X_test_processed = (
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

X_train_processed = np.asarray(X_train_processed)
X_test_processed = np.asarray(X_test_processed)

base_estimator = LogisticRegression(
    max_iter=5000,
    random_state=42
)

mitigator = ExponentiatedGradient(
    estimator=base_estimator,
    constraints=DemographicParity(),
    eps=0.01
)

print("Training ExpGrad...")

mitigator.fit(
    X_train_processed,
    y_train,
    sensitive_features=s_train
)

y_pred = mitigator.predict(X_test_processed)

try:
    y_prob = mitigator._pmf_predict(X_test_processed)[:, 1]
except Exception:
    y_prob = y_pred

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_test, y_prob)
except Exception:
    auc = np.nan

spd = demographic_parity_difference(
    y_test,
    y_pred,
    sensitive_features=s_test
)

eod = equalized_odds_difference(
    y_test,
    y_pred,
    sensitive_features=s_test
)

metric_frame = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=s_test
)

female_selection = metric_frame.by_group.loc[
    "Female",
    "selection_rate"
]

male_selection = metric_frame.by_group.loc[
    "Male",
    "selection_rate"
]

female_tpr = metric_frame.by_group.loc[
    "Female",
    "tpr"
]

male_tpr = metric_frame.by_group.loc[
    "Male",
    "tpr"
]

female_fpr = metric_frame.by_group.loc[
    "Female",
    "fpr"
]

male_fpr = metric_frame.by_group.loc[
    "Male",
    "fpr"
]

di = (
    female_selection / male_selection
    if male_selection != 0 else 0
)

results = pd.DataFrame([{
    "Experiment": "ExpGrad",
    "Model": "LogisticRegression_ExpGrad",

    "Accuracy": round(acc, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1": round(f1, 4),
    "ROC_AUC": round(float(auc), 4) if not np.isnan(auc) else None,

    "SPD": round(spd, 4),
    "EOD": round(eod, 4),
    "DI": round(di, 4),

    "Female_Selection_Rate": round(female_selection, 4),
    "Male_Selection_Rate": round(male_selection, 4),

    "Female_TPR": round(female_tpr, 4),
    "Male_TPR": round(male_tpr, 4),

    "Female_FPR": round(female_fpr, 4),
    "Male_FPR": round(male_fpr, 4)
}])

save_path = f"{OUTPUT_DIR}/expgrad_results.csv"

results.to_csv(
    save_path,
    index=False
)

print("\n===================")
print(results)
print("===================")

print("\nSaved:", save_path)