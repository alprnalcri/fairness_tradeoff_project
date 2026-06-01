import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

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
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numerical_cols
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "LogisticRegression":
        LogisticRegression(
            max_iter=3000,
            random_state=42
        ),

    "DecisionTree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
}

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)

    precision = precision_score(y_test, preds)

    recall = recall_score(y_test, preds)

    f1 = f1_score(y_test, preds)

    auc = roc_auc_score(y_test, probs)

    results.append({
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(auc, 4)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{OUTPUT_DIR}/baseline_results.csv",
    index=False
)

print("\n==========================")
print(results_df)
print("==========================")

print(
    "\nSaved:",
    f"{OUTPUT_DIR}/baseline_results.csv"
)