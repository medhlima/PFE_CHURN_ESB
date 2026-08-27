from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_churn_model.joblib"
)

EXPECTED_FEATURES = [
    "AGE",
    "TRANCHE_AGE",
    "ANCIENNETE_CLIENT_ANNEES",
    "SALAIRE",
    "HAS_SALAIRE",
    "EST_TUNISIEN",
    "EST_RESIDENT",
    "SITUATION_FAMILIALE",
    "TYPE_CLIENT",
    "SEGMENT",
    "DOSSIER_COMPLET",
    "SCORE_KYC",
    "KYC_RISQUE_ELEVE",
    "JOURS_AVANT_PROCHAINE_REVUE",
    "REVUE_EN_RETARD",
    "A_HISTORIQUE_REVUE",
    "NB_COMPTES",
]


class PredictionService:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def validate_input(self, customer_data: dict):

        missing_features = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in customer_data
        ]

        if missing_features:
            raise ValueError(
                "Missing features: "
                + ", ".join(missing_features)
            )

    def predict(self, customer_data: dict):

        self.validate_input(customer_data)

        input_df = pd.DataFrame(
            [customer_data],
            columns=EXPECTED_FEATURES
        )

        probability = float(
            self.model.predict_proba(
                input_df
            )[0, 1]
        )

        prediction = int(
            probability >= 0.50
        )
        if probability >= 0.50:
            risk_level = "HIGH"
        elif probability >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "prediction": prediction,
            "prediction_label": (
                "CHURN"
                if prediction == 1
                else "NON_CHURN"
            ),
            "churn_probability": round(
                probability,
                4
            ),
            "churn_probability_percent": round(
                probability * 100,
                2
            ),
            "risk_level": risk_level,
        }
