import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X,
    y,
    sensitive,
    test_size=0.20,
    random_state=42,
    stratify=y
)


def compute_reweighing_weights(y_train, sensitive_train):
    train_df = pd.DataFrame({
        "y": y_train.values,
        "s": sensitive_train.values
    })

    n = len(train_df)

    p_y = train_df["y"].value_counts(normalize=True).to_dict()
    p_s = train_df["s"].value_counts(normalize=True).to_dict()
    p_ys = train_df.groupby(["y", "s"]).size() / n

    weights = []

    for _, row in train_df.iterrows():
        y_val = row["y"]
        s_val = row["s"]

        numerator = p_y[y_val] * p_s[s_val]
        denominator = p_ys.loc[(y_val, s_val)]

        weights.append(numerator / denominator)

    return np.array(weights)


sample_weights = compute_reweighing_weights(y_train, s_train)

print("Sample weights summary:")
print(pd.Series(sample_weights).describe())

models = {
    "LogisticRegression_Reweighing": LogisticRegression(
        max_iter=5000,
        random_state=42
    ),

    "RandomForest_Reweighing": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost_Reweighing": XGBClassifier(
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

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights
    )

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

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

    female_selection = metric_frame.by_group.loc["Female", "selection_rate"]
    male_selection = metric_frame.by_group.loc["Male", "selection_rate"]

    female_tpr = metric_frame.by_group.loc["Female", "tpr"]
    male_tpr = metric_frame.by_group.loc["Male", "tpr"]

    female_fpr = metric_frame.by_group.loc["Female", "fpr"]
    male_fpr = metric_frame.by_group.loc["Male", "fpr"]

    di = female_selection / male_selection if male_selection != 0 else 0

    results.append({
        "Experiment": "Reweighing",
        "Model": model_name,

        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(auc, 4),

        "SPD": round(spd, 4),
        "EOD": round(eod, 4),
        "DI": round(di, 4),

        "Female_Selection_Rate": round(female_selection, 4),
        "Male_Selection_Rate": round(male_selection, 4),

        "Female_TPR": round(female_tpr, 4),
        "Male_TPR": round(male_tpr, 4),

        "Female_FPR": round(female_fpr, 4),
        "Male_FPR": round(male_fpr, 4)
    })

results_df = pd.DataFrame(results)

save_path = f"{OUTPUT_DIR}/reweighing_results.csv"
results_df.to_csv(save_path, index=False)

print("\n======================")
print(results_df)
print("======================")
print("\nSaved:", save_path)