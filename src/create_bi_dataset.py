from pathlib import Path

import joblib
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_modelisation.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_churn_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "power_bi"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "customer_bi_dataset.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "customer_bi_dataset.csv"
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("POWER BI DATASET GENERATION")
print("=" * 70)

df = pd.read_parquet(INPUT_PATH)

print(f"Customers loaded : {len(df):,}")
print(f"Input columns    : {len(df.columns)}")


# ==========================================================
# LOAD FINAL XGBOOST MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)

features = list(model.feature_names_in_)

print(f"Model features   : {len(features)}")


# ==========================================================
# VALIDATION
# ==========================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing model features: "
        + ", ".join(missing_features)
    )


# ==========================================================
# FINAL MODEL SCORING
# ==========================================================

X = df[features].copy()

probabilities = model.predict_proba(X)[:, 1]

bi_df = df.copy()

bi_df = bi_df.rename(
    columns={
        "CHURN": "ACTUAL_CHURN"
    }
)

bi_df["CHURN_PROBABILITY"] = probabilities

bi_df["CHURN_SCORE"] = (
    bi_df["CHURN_PROBABILITY"] * 100
).round(2)

bi_df["PREDICTED_CHURN"] = (
    bi_df["CHURN_PROBABILITY"] >= 0.50
).astype(int)

bi_df["PREDICTION_LABEL"] = bi_df[
    "PREDICTED_CHURN"
].map(
    {
        0: "NON_CHURN",
        1: "CHURN",
    }
)


# ==========================================================
# RISK LEVEL
# Same business rule used by Flask PredictionService
# ==========================================================

def get_risk_level(probability):

    if probability >= 0.50:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


bi_df["RISK_LEVEL"] = (
    bi_df["CHURN_PROBABILITY"]
    .apply(get_risk_level)
)


# ==========================================================
# BUSINESS-FRIENDLY FIELDS FOR POWER BI
# ==========================================================

bi_df["ACTUAL_CHURN_LABEL"] = (
    bi_df["ACTUAL_CHURN"]
    .map({
        0: "NON_CHURN",
        1: "CHURN",
    })
)

bi_df["REVIEW_STATUS"] = (
    bi_df["REVUE_EN_RETARD"]
    .map({
        True: "OVERDUE",
        False: "UP_TO_DATE",
    })
)

bi_df["SALARY_STATUS"] = (
    bi_df["HAS_SALAIRE"]
    .map({
        True: "AVAILABLE",
        False: "NOT_AVAILABLE",
    })
)

bi_df["KYC_RISK_STATUS"] = (
    bi_df["KYC_RISQUE_ELEVE"]
    .map({
        True: "HIGH_RISK",
        False: "STANDARD",
    })
)

bi_df["FILE_STATUS"] = (
    bi_df["DOSSIER_COMPLET"]
    .map({
        True: "COMPLETE",
        False: "INCOMPLETE",
    })
)


# ==========================================================
# VALIDATION KPIs
# ==========================================================

total_customers = len(bi_df)

actual_churn = int(
    bi_df["ACTUAL_CHURN"].sum()
)

actual_churn_rate = (
    actual_churn
    / total_customers
    * 100
)

predicted_high_risk = int(
    bi_df["PREDICTED_CHURN"].sum()
)

average_score = (
    bi_df["CHURN_SCORE"].mean()
)


print()
print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

print(
    f"Total customers        : "
    f"{total_customers:,}"
)

print(
    f"Actual churn customers : "
    f"{actual_churn:,}"
)

print(
    f"Actual churn rate      : "
    f"{actual_churn_rate:.2f}%"
)

print(
    f"Predicted churn        : "
    f"{predicted_high_risk:,}"
)

print(
    f"Average churn score    : "
    f"{average_score:.2f}%"
)

print()
print("Risk distribution:")

print(
    bi_df["RISK_LEVEL"]
    .value_counts()
    .to_string()
)


# ==========================================================
# CONTROL CUSTOMER
# ==========================================================

control_customer = bi_df[
    bi_df["CUSTOMER_NO"].astype(str)
    == "113425593"
]

if not control_customer.empty:

    row = control_customer.iloc[0]

    print()
    print("=" * 70)
    print("CONTROL CUSTOMER 113425593")
    print("=" * 70)

    print(
        f"Probability : "
        f"{row['CHURN_PROBABILITY']:.4f}"
    )

    print(
        f"Score       : "
        f"{row['CHURN_SCORE']:.2f}%"
    )

    print(
        f"Prediction  : "
        f"{row['PREDICTION_LABEL']}"
    )

    print(
        f"Risk Level  : "
        f"{row['RISK_LEVEL']}"
    )


# ==========================================================
# EXPORT
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

bi_df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

bi_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print()
print("=" * 70)
print("EXPORT COMPLETED")
print("=" * 70)

print(
    f"Parquet : {OUTPUT_PARQUET}"
)

print(
    f"CSV     : {OUTPUT_CSV}"
)

print(
    f"Shape   : {bi_df.shape}"
)