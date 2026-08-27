from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlops.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

COMPARISON_FILE = (
    TABLE_DIR / "final_models_comparison.xlsx"
)

BEST_BASELINE_FILE = (
    MODEL_DIR / "best_baseline_model.joblib"
)

XGBOOST_FILE = (
    MODEL_DIR / "xgboost_churn_model.joblib"
)

PREPROCESSOR_FILE = (
    MODEL_DIR / "preprocessing_pipeline.joblib"
)

# ==========================================================
# MLFLOW CONFIGURATION
# ==========================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)

# ==========================================================
# LOAD OFFICIAL RESULTS
# ==========================================================

comparison = pd.read_excel(
    COMPARISON_FILE,
    index_col=0
)

print("=" * 70)
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 70)

print("\nOfficial model results:")
print(comparison)

# ==========================================================
# MODEL ARTIFACT MAPPING
# ==========================================================

model_files = {
    "Random Forest": BEST_BASELINE_FILE,
    "XGBoost": XGBOOST_FILE,
}

# ==========================================================
# LOG RUNS
# ==========================================================

for model_name, row in comparison.iterrows():

    run_name = (
        model_name
        .replace(" ", "_")
    )

    print(
        f"\nLogging {model_name}..."
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # ------------------------------
        # TAGS
        # ------------------------------

        mlflow.set_tags({
            "project": "PFE_CHURN_ESB",
            "problem_type": "Binary Classification",
            "target": "CHURN",
            "model_name": model_name,
            "data_version": "Pipeline_V2",
            "business_domain": "Banking Customer Churn",
        })

        # ------------------------------
        # METRICS
        # ------------------------------

        metrics = {
            "accuracy": float(row["Accuracy"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1"]),
            "roc_auc": float(row["ROC_AUC"]),
            "pr_auc": float(row["PR_AUC"]),
        }

        mlflow.log_metrics(
            metrics
        )

        # ------------------------------
        # COMMON PARAMETERS
        # ------------------------------

        mlflow.log_params({
            "test_size": 0.20,
            "random_state": 42,
            "churn_rate": 0.0541,
        })

        # ------------------------------
        # LOG SAVED MODEL WHEN AVAILABLE
        # ------------------------------

        if model_name in model_files:

            model_path = model_files[
                model_name
            ]

            if model_path.exists():

                model = joblib.load(
                    model_path
                )

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                )

                mlflow.log_param(
                    "model_artifact",
                    model_path.name
                )

                print(
                    f"  Model artifact logged: "
                    f"{model_path.name}"
                )

        else:

            mlflow.set_tag(
                "model_artifact_status",
                "Metrics only - model not persisted"
            )

        # ------------------------------
        # OFFICIAL COMPARISON TABLE
        # ------------------------------

        mlflow.log_artifact(
            str(COMPARISON_FILE),
            artifact_path="evaluation"
        )

        # ------------------------------
        # PREPROCESSING ARTIFACT
        # ------------------------------

        if PREPROCESSOR_FILE.exists():

            mlflow.log_artifact(
                str(PREPROCESSOR_FILE),
                artifact_path="preprocessing"
            )

        # ------------------------------
        # FIGURES
        # ------------------------------

        common_figures = [
            "final_model_performance_comparison.png",
            "final_precision_recall_tradeoff.png",
        ]

        for figure_name in common_figures:

            figure_path = (
                FIGURE_DIR / figure_name
            )

            if figure_path.exists():

                mlflow.log_artifact(
                    str(figure_path),
                    artifact_path="figures"
                )

        print(
            f"  Metrics logged for {model_name}"
        )

print("\n" + "=" * 70)
print("MLFLOW TRACKING COMPLETED")
print("=" * 70)