from flask import Flask, jsonify, request, render_template

from flask_app.services.prediction_service import (
    PredictionService,
)
from flask_app.services.explainability_service import (
    ExplainabilityService,
)

from flask_app.services.model_service import ModelService

explainability_service = ExplainabilityService()

app = Flask(__name__)

prediction_service = PredictionService()
model_service = ModelService()

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Banking Customer Churn API",
        "model": "XGBoost",
    })


@app.post("/predict")
def predict():
    try:
        customer_data = request.get_json()

        if not customer_data:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        result = prediction_service.predict(
            customer_data
        )
        explanation = explainability_service.explain(
            customer_data,
            top_n=5
        )

        result["explanation"] = explanation

        return jsonify(result), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500
@app.get("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.get("/customer360")
def customer360_page():
    return render_template("customer360.html")


@app.get("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.get("/governance")
def governance_page():
    return render_template("governance.html")

from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUSTOMER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_modelisation.parquet"
)

customer_df = pd.read_parquet(CUSTOMER_FILE)

@app.get("/customer/<customer_id>")
def get_customer(customer_id):

    customer = customer_df[
        customer_df["CUSTOMER_NO"].astype(str) == str(customer_id)
    ]

    if customer.empty:
        return jsonify({
            "error": "Customer not found."
        }), 404

    row = customer.iloc[0]

    fields = [
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

    customer_data = {}

    for field in fields:

        value = row[field]

        if pd.isna(value):
            customer_data[field] = None

        elif hasattr(value, "item"):
            customer_data[field] = value.item()

        else:
            customer_data[field] = value

    customer_data["CUSTOMER_NO"] = str(row["CUSTOMER_NO"])
    return jsonify(customer_data), 200

@app.get("/api/analytics")
def analytics_data():

    df = customer_df.copy()

    total_customers = int(len(df))
    churn_customers = int(df["CHURN"].sum())

    overall_churn_rate = (
        churn_customers
        / total_customers
        * 100
    )

    def build_group_stats(column):

        stats = (
            df.groupby(
                column,
                dropna=False
            )
            .agg(
                total_customers=(
                    "CUSTOMER_NO",
                    "count"
                ),
                churn_customers=(
                    "CHURN",
                    "sum"
                )
            )
            .reset_index()
        )

        stats["churn_rate"] = (
            stats["churn_customers"]
            / stats["total_customers"]
            * 100
        )

        stats[column] = (
            stats[column]
            .astype(str)
            .replace("nan", "Unknown")
        )

        stats = stats.sort_values(
            "churn_rate",
            ascending=False
        )

        return stats.round({
            "churn_rate": 2
        }).to_dict(
            orient="records"
        )

    return jsonify({
        "portfolio": {
            "total_customers":
                total_customers,

            "churn_customers":
                churn_customers,

            "non_churn_customers":
                total_customers
                - churn_customers,

            "churn_rate":
                round(
                    overall_churn_rate,
                    2
                ),
        },

        "by_segment":
            build_group_stats(
                "SEGMENT"
            ),

        "by_customer_type":
            build_group_stats(
                "TYPE_CLIENT"
            ),

        "by_age_group":
            build_group_stats(
                "TRANCHE_AGE"
            ),

        "by_kyc":
            build_group_stats(
                "SCORE_KYC"
            ),
    })

@app.get("/api/governance")
def governance_data():

    try:

        data = (
            model_service
            .get_governance_summary()
        )

        return jsonify(
            data
        ), 200

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500
        
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )