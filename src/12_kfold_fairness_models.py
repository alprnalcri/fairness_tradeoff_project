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

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate
)


OUTPUT_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/processed/adult_clean.csv")

TARGET = "income"
SENSITIVE = "sex"

X = df.drop(columns=[TARGET])
y = df[TARGET]
sensitive = df[SENSITIVE]


def build_preprocessor(X_data):
    categorical_cols = X_data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X_data.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

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

    return preprocessor


def compute_reweighing_weights(y_train, sensitive_train):
    temp_df = pd.DataFrame({
        "y": y_train.values,
        "s": sensitive_train.values
    })

    n = len(temp_df)

    p_y = temp_df["y"].value_counts(normalize=True).to_dict()
    p_s = temp_df["s"].value_counts(normalize=True).to_dict()
    p_ys = temp_df.groupby(["y", "s"]).size() / n

    weights = []

    for _, row in temp_df.iterrows():
        y_val = row["y"]
        s_val = row["s"]

        numerator = p_y[y_val] * p_s[s_val]
        denominator = p_ys.loc[(y_val, s_val)]

        weights.append(numerator / denominator)

    return np.array(weights)


def evaluate_fairness(y_true, y_pred, sensitive_test):
    spd = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_test
    )

    eod = equalized_odds_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_test
    )

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )

    female_selection = mf.by_group.loc["Female", "selection_rate"]
    male_selection = mf.by_group.loc["Male", "selection_rate"]

    di = (
        female_selection / male_selection
        if male_selection != 0 else 0
    )

    return spd, eod, di, female_selection, male_selection


def make_xgb():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_results = []

print("\nRunning 5-Fold Fairness Validation...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    print(f"\n========== Fold {fold} ==========")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    s_train = sensitive.iloc[train_idx]
    s_test = sensitive.iloc[test_idx]

    # -------------------------------
    # 1. XGBoost Baseline
    # -------------------------------

    preprocessor = build_preprocessor(X_train)

    baseline_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", make_xgb())
    ])

    baseline_pipe.fit(X_train, y_train)

    y_pred = baseline_pipe.predict(X_test)
    y_prob = baseline_pipe.predict_proba(X_test)[:, 1]

    spd, eod, di, female_sr, male_sr = evaluate_fairness(
        y_test,
        y_pred,
        s_test
    )

    result = {
        "Experiment": "KFold",
        "Model": "XGBoost_Baseline",
        "Fold": fold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "SPD": spd,
        "EOD": eod,
        "DI": di,
        "Female_Selection_Rate": female_sr,
        "Male_Selection_Rate": male_sr
    }

    all_results.append(result)

    print(
        f"XGB Baseline | "
        f"ACC={result['Accuracy']:.4f} | "
        f"F1={result['F1']:.4f} | "
        f"AUC={result['ROC_AUC']:.4f} | "
        f"SPD={result['SPD']:.4f} | "
        f"EOD={result['EOD']:.4f} | "
        f"DI={result['DI']:.4f}"
    )

    # -------------------------------
    # 2. XGBoost + Reweighing
    # -------------------------------

    preprocessor_rw = build_preprocessor(X_train)

    X_train_processed = preprocessor_rw.fit_transform(X_train)
    X_test_processed = preprocessor_rw.transform(X_test)

    sample_weights = compute_reweighing_weights(
        y_train,
        s_train
    )

    rw_model = make_xgb()

    rw_model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights
    )

    y_pred_rw = rw_model.predict(X_test_processed)
    y_prob_rw = rw_model.predict_proba(X_test_processed)[:, 1]

    spd_rw, eod_rw, di_rw, female_sr_rw, male_sr_rw = evaluate_fairness(
        y_test,
        y_pred_rw,
        s_test
    )

    result_rw = {
        "Experiment": "KFold_Reweighing",
        "Model": "XGBoost_Reweighing",
        "Fold": fold,
        "Accuracy": accuracy_score(y_test, y_pred_rw),
        "Precision": precision_score(y_test, y_pred_rw, zero_division=0),
        "Recall": recall_score(y_test, y_pred_rw, zero_division=0),
        "F1": f1_score(y_test, y_pred_rw, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob_rw),
        "SPD": spd_rw,
        "EOD": eod_rw,
        "DI": di_rw,
        "Female_Selection_Rate": female_sr_rw,
        "Male_Selection_Rate": male_sr_rw
    }

    all_results.append(result_rw)

    print(
        f"XGB Reweighing | "
        f"ACC={result_rw['Accuracy']:.4f} | "
        f"F1={result_rw['F1']:.4f} | "
        f"AUC={result_rw['ROC_AUC']:.4f} | "
        f"SPD={result_rw['SPD']:.4f} | "
        f"EOD={result_rw['EOD']:.4f} | "
        f"DI={result_rw['DI']:.4f}"
    )


results_df = pd.DataFrame(all_results)

fold_path = f"{OUTPUT_DIR}/kfold_fairness_folds.csv"
results_df.to_csv(fold_path, index=False)

summary = (
    results_df
    .groupby("Model")
    .agg({
        "Accuracy": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1": ["mean", "std"],
        "ROC_AUC": ["mean", "std"],
        "SPD": ["mean", "std"],
        "EOD": ["mean", "std"],
        "DI": ["mean", "std"],
        "Female_Selection_Rate": ["mean", "std"],
        "Male_Selection_Rate": ["mean", "std"]
    })
)

summary.columns = [
    "_".join(col).strip()
    for col in summary.columns.values
]

summary = summary.reset_index()

summary_path = f"{OUTPUT_DIR}/kfold_fairness_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n============================")
print("K-FOLD FAIRNESS SUMMARY")
print("============================")
print(summary.round(4))

print("\nSaved:")
print(fold_path)
print(summary_path)