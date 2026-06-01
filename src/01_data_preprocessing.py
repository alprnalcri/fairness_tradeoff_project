import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TABLE_DIR = "outputs/tables"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

files = glob.glob(os.path.join(RAW_DIR, "*"))

if len(files) == 0:
    raise FileNotFoundError("data/raw klasöründe veri dosyası bulunamadı.")

DATA_PATH = files[0]
print(f"Kullanılan veri dosyası: {DATA_PATH}")

if DATA_PATH.endswith(".csv"):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.read_csv(
        DATA_PATH,
        names=columns,
        na_values=[" ?", "?"],
        skipinitialspace=True
    )

if list(df.columns) != columns and "income" not in df.columns:
    df.columns = columns

print("İlk veri boyutu:", df.shape)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

df = df.replace("?", pd.NA)
df = df.dropna()

df["income"] = df["income"].replace({
    "<=50K.": "<=50K",
    ">50K.": ">50K"
})

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

df["sex_binary"] = df["sex"].map({
    "Female": 0,
    "Male": 1
})

df = df.dropna(subset=["income", "sex_binary"])

df["income"] = df["income"].astype(int)
df["sex_binary"] = df["sex_binary"].astype(int)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
)

print("Temiz veri boyutu:", df.shape)

print("\nHedef dağılımı:")
print(df["income"].value_counts())

print("\nCinsiyet dağılımı:")
print(df["sex"].value_counts())

sex_income_count = pd.crosstab(df["sex"], df["income"])
sex_income_rate = pd.crosstab(df["sex"], df["income"], normalize="index")

race_income_count = pd.crosstab(df["race"], df["income"])
race_income_rate = pd.crosstab(df["race"], df["income"], normalize="index")

age_income_count = pd.crosstab(df["age_group"], df["income"])
age_income_rate = pd.crosstab(df["age_group"], df["income"], normalize="index")

sex_race_income_rate = pd.crosstab(
    [df["sex"], df["race"]],
    df["income"],
    normalize="index"
)

sex_income_count.to_csv(f"{TABLE_DIR}/sex_income_count.csv")
sex_income_rate.to_csv(f"{TABLE_DIR}/sex_income_rate.csv")

race_income_count.to_csv(f"{TABLE_DIR}/race_income_count.csv")
race_income_rate.to_csv(f"{TABLE_DIR}/race_income_rate.csv")

age_income_count.to_csv(f"{TABLE_DIR}/age_income_count.csv")
age_income_rate.to_csv(f"{TABLE_DIR}/age_income_rate.csv")

sex_race_income_rate.to_csv(f"{TABLE_DIR}/sex_race_income_rate.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["income"]
)

df.to_csv(f"{PROCESSED_DIR}/adult_clean.csv", index=False)
train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

print("\nKaydedilen dosyalar:")
print(f"{PROCESSED_DIR}/adult_clean.csv")
print(f"{PROCESSED_DIR}/train.csv")
print(f"{PROCESSED_DIR}/test.csv")

print(f"{TABLE_DIR}/sex_income_count.csv")
print(f"{TABLE_DIR}/sex_income_rate.csv")
print(f"{TABLE_DIR}/race_income_count.csv")
print(f"{TABLE_DIR}/race_income_rate.csv")
print(f"{TABLE_DIR}/age_income_count.csv")
print(f"{TABLE_DIR}/age_income_rate.csv")
print(f"{TABLE_DIR}/sex_race_income_rate.csv")

print("\nİşlem tamam.")