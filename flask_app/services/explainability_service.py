from pathlib import Path

import joblib
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_churn_model.joblib"
)

ORIGINAL_FEATURES = [
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


class ExplainabilityService:

    def __init__(self):

        self.pipeline = joblib.load(MODEL_PATH)

        self.preprocessing = (
            self.pipeline.named_steps["preprocessing"]
        )

        self.classifier = (
            self.pipeline.named_steps["classifier"]
        )

        self.transformed_feature_names = list(
            self.preprocessing.get_feature_names_out()
        )

        self.explainer = shap.TreeExplainer(
            self.classifier
        )

    def _original_feature_name(self, transformed_name):

        clean_name = (
            transformed_name
            .replace("num__", "")
            .replace("cat__", "")
        )

        # Numeric feature
        if clean_name in ORIGINAL_FEATURES:
            return clean_name

        # One-hot encoded categorical feature
        matches = [
            feature
            for feature in ORIGINAL_FEATURES
            if clean_name.startswith(feature + "_")
        ]

        if matches:
            return max(matches, key=len)

        return clean_name

    def explain(self, customer_data, top_n=5):

        input_df = pd.DataFrame(
            [customer_data],
            columns=ORIGINAL_FEATURES
        )

        transformed = self.preprocessing.transform(
            input_df
        )

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        explanation = self.explainer(
            transformed
        )

        shap_values = explanation.values[0]

        details = pd.DataFrame({
            "transformed_feature":
                self.transformed_feature_names,

            "shap_value":
                shap_values
        })

        details["feature"] = details[
            "transformed_feature"
        ].apply(
            self._original_feature_name
        )

        # Aggregate one-hot encoded columns back
        # to original business variables
        aggregated = (
            details
            .groupby(
                "feature",
                as_index=False
            )["shap_value"]
            .sum()
        )

        aggregated["absolute_impact"] = (
            aggregated["shap_value"].abs()
        )

        aggregated = (
            aggregated
            .sort_values(
                "absolute_impact",
                ascending=False
            )
            .head(top_n)
        )

        results = []

        for _, row in aggregated.iterrows():

            shap_value = float(
                row["shap_value"]
            )

            results.append({
                "feature":
                    row["feature"],

                "impact":
                    round(
                        abs(shap_value),
                        4
                    ),

                "direction": (
                    "INCREASES_RISK"
                    if shap_value > 0
                    else "DECREASES_RISK"
                ),

                "label": (
                    "Increases churn risk"
                    if shap_value > 0
                    else "Reduces churn risk"
                )
            })

        return results