import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("data/processed/adult_clean.csv")

# -------------------------
# 1. Cinsiyet Dağılımı
# -------------------------

plt.figure(figsize=(6,4))

df["sex"].value_counts().plot(
    kind="bar"
)

plt.title("Gender Distribution")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/gender_distribution.png",
    dpi=300
)

plt.close()

# -------------------------
# 2. Gelir-Cinsiyet
# -------------------------

sex_income = pd.crosstab(
    df["sex"],
    df["income"],
    normalize="index"
)

sex_income.plot(
    kind="bar",
    stacked=True,
    figsize=(7,5)
)

plt.title("Income Distribution by Gender")
plt.ylabel("Proportion")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/income_by_gender.png",
    dpi=300
)

plt.close()

# -------------------------
# 3. Gelir-Race
# -------------------------

race_income = pd.crosstab(
    df["race"],
    df["income"],
    normalize="index"
)

race_income.plot(
    kind="bar",
    stacked=True,
    figsize=(10,5)
)

plt.title("Income Distribution by Race")
plt.ylabel("Proportion")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/income_by_race.png",
    dpi=300
)

plt.close()

# -------------------------
# 4. Yaş-Gelir
# -------------------------

df["age_group"] = pd.cut(
    df["age"],
    bins=[0,25,35,45,55,65,100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "65+"
    ]
)

age_income = pd.crosstab(
    df["age_group"],
    df["income"],
    normalize="index"
)

age_income.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Income Distribution by Age Group")
plt.ylabel("Proportion")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/income_by_age.png",
    dpi=300
)

plt.close()

print("Grafikler oluşturuldu.")
