import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# Fairness ölçmek için sex tutulur ama model eğitiminden çıkarılır
y = df[TARGET]
sensitive = df[SENSITIVE]

DROP_COLUMNS = [
    TARGET,
    "sex",
    "sex_binary"
]

X = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

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

models = {
    "LogisticRegression_NoSex": LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42
    ),
    "DecisionTree_NoSex": DecisionTreeClassifier(
        random_state=42
    ),
    "RandomForest_NoSex": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost_NoSex": XGBClassifier(
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

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

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

    group_metrics = metric_frame.by_group

    female_selection = group_metrics.loc["Female", "selection_rate"]
    male_selection = group_metrics.loc["Male", "selection_rate"]

    female_tpr = group_metrics.loc["Female", "tpr"]
    male_tpr = group_metrics.loc["Male", "tpr"]

    female_fpr = group_metrics.loc["Female", "fpr"]
    male_fpr = group_metrics.loc["Male", "fpr"]

    disparate_impact = (
        female_selection / male_selection
        if male_selection != 0 else 0
    )

    results.append({
        "Experiment": "NoSex",
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(auc, 4),
        "SPD": round(spd, 4),
        "EOD": round(eod, 4),
        "DI": round(disparate_impact, 4),
        "Female_Selection_Rate": round(female_selection, 4),
        "Male_Selection_Rate": round(male_selection, 4),
        "Female_TPR": round(female_tpr, 4),
        "Male_TPR": round(male_tpr, 4),
        "Female_FPR": round(female_fpr, 4),
        "Male_FPR": round(male_fpr, 4)
    })

results_df = pd.DataFrame(results)

save_path = f"{OUTPUT_DIR}/no_sex_fairness_results.csv"
results_df.to_csv(save_path, index=False)

print("\n==========================")
print(results_df)
print("==========================")
print("\nSaved:", save_path)