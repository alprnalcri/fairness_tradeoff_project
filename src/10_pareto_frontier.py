import os
import pandas as pd
import matplotlib.pyplot as plt

TABLE_PATH = "outputs/tables/final_comparison.csv"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(TABLE_PATH)

# Daha okunur olması için sadece ana modelleri seçiyoruz
selected_models = [
    "XGBoost",
    "XGBoost_NoSex",
    "XGBoost_NoDemo",
    "XGBoost_Reweighing",
    "LogisticRegression_ExpGrad",
    "LogisticRegression",
    "LogisticRegression_Reweighing"
]

df_plot = df[df["Model"].isin(selected_models)].copy()
df_plot["Abs_SPD"] = df_plot["SPD"].abs()

labels = {
    "XGBoost": "XGBoost\nBest Accuracy",
    "XGBoost_NoSex": "XGBoost\nNo-Sex",
    "XGBoost_NoDemo": "XGBoost\nNo-Demo",
    "XGBoost_Reweighing": "XGBoost +\nReweighing\nBest Balance",
    "LogisticRegression_ExpGrad": "LR + ExpGrad\nBest Fairness",
    "LogisticRegression": "LR Baseline",
    "LogisticRegression_Reweighing": "LR +\nReweighing"
}

plt.figure(figsize=(11, 7))

for _, row in df_plot.iterrows():
    model = row["Model"]

    if model == "XGBoost_Reweighing":
        marker = "*"
        size = 260
    elif model == "LogisticRegression_ExpGrad":
        marker = "D"
        size = 150
    elif model == "XGBoost":
        marker = "o"
        size = 130
    else:
        marker = "o"
        size = 90

    plt.scatter(
        row["Accuracy"],
        row["Abs_SPD"],
        s=size,
        marker=marker
    )

    plt.annotate(
        labels.get(model, model),
        (row["Accuracy"], row["Abs_SPD"]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9
    )

plt.xlabel("Accuracy")
plt.ylabel("|Statistical Parity Difference|")
plt.title("Accuracy–Fairness Trade-off Analysis")

plt.axhline(
    y=0.10,
    linestyle="--",
    linewidth=1
)

plt.text(
    df_plot["Accuracy"].min(),
    0.103,
    "Lower SPD = Fairer",
    fontsize=9
)

plt.grid(True, alpha=0.4)

plt.tight_layout()

save_path = f"{FIG_DIR}/accuracy_spd_tradeoff_clean.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)