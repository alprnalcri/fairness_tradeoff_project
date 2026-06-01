import pandas as pd
import os

OUTPUT_DIR = "outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dfs = []

files = [
    "baseline_fairness_results.csv",
    "no_sex_fairness_results.csv",
    "no_demographic_results.csv",
    "reweighing_results.csv",
    "expgrad_results.csv"
]

for file in files:
    path = f"{OUTPUT_DIR}/{file}"

    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

cols = [
    "Experiment",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "SPD",
    "EOD",
    "DI"
]

final_df = final_df[cols]

save_path = f"{OUTPUT_DIR}/final_comparison.csv"

final_df.to_csv(
    save_path,
    index=False
)

print("\n======================")
print(final_df)
print("======================")

print("\nSaved:", save_path)