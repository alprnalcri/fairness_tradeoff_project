import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


OUTPUT_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/processed/adult_clean.csv")

TARGET = "income"

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
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

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\nRunning 5-Fold CV for XGBoost Baseline...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    fold_result = {
        "Fold": fold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

    results.append(fold_result)

    print(
        f"Fold {fold} | "
        f"ACC={fold_result['Accuracy']:.4f} | "
        f"Precision={fold_result['Precision']:.4f} | "
        f"Recall={fold_result['Recall']:.4f} | "
        f"F1={fold_result['F1']:.4f} | "
        f"AUC={fold_result['ROC_AUC']:.4f}"
    )

results_df = pd.DataFrame(results)

summary_df = pd.DataFrame([{
    "Model": "XGBoost_Baseline_5Fold",
    "Accuracy_Mean": results_df["Accuracy"].mean(),
    "Accuracy_STD": results_df["Accuracy"].std(),
    "Precision_Mean": results_df["Precision"].mean(),
    "Precision_STD": results_df["Precision"].std(),
    "Recall_Mean": results_df["Recall"].mean(),
    "Recall_STD": results_df["Recall"].std(),
    "F1_Mean": results_df["F1"].mean(),
    "F1_STD": results_df["F1"].std(),
    "ROC_AUC_Mean": results_df["ROC_AUC"].mean(),
    "ROC_AUC_STD": results_df["ROC_AUC"].std()
}])

fold_path = f"{OUTPUT_DIR}/kfold_xgboost_folds.csv"
summary_path = f"{OUTPUT_DIR}/kfold_xgboost_summary.csv"

results_df.to_csv(fold_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\n============================")
print("5-FOLD SUMMARY")
print("============================")
print(summary_df.round(4))

print("\nSaved:")
print(fold_path)
print(summary_path)