import pandas as pd
import matplotlib.pyplot as plt
import os

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(
    "outputs/tables/final_comparison.csv"
)

plt.figure(figsize=(10,7))

x = 1 - df["SPD"].abs()
y = df["Accuracy"]

plt.scatter(
    x,
    y,
    s=120
)

for _, row in df.iterrows():

    plt.annotate(
        row["Model"],
        (
            1 - abs(row["SPD"]),
            row["Accuracy"]
        ),
        fontsize=8
    )

plt.xlabel("Fairness Score (1 - |SPD|)")
plt.ylabel("Accuracy")

plt.title(
    "Accuracy-Fairness Trade-Off (Pareto Analysis)"
)

plt.grid(True)

save_path = (
    f"{FIG_DIR}/pareto_frontier.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", save_path)