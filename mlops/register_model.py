import mlflow
from mlflow import MlflowClient

from mlops.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)

# ==========================================================
# CONFIGURATION
# ==========================================================

REGISTERED_MODEL_NAME = "Banking_Churn_Model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)

# ==========================================================
# FIND EXPERIMENT
# ==========================================================

experiment = mlflow.get_experiment_by_name(
    MLFLOW_EXPERIMENT
)

if experiment is None:
    raise RuntimeError(
        f"Experiment not found: {MLFLOW_EXPERIMENT}"
    )

# ==========================================================
# FIND LATEST SUCCESSFUL XGBOOST RUN
# ==========================================================

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        'attributes.run_name = "XGBoost" '
        'AND attributes.status = "FINISHED"'
    ),
    order_by=["start_time DESC"],
)

if runs.empty:
    raise RuntimeError(
        "No successful XGBoost run found."
    )

run_id = runs.iloc[0]["run_id"]

print("=" * 70)
print("MLFLOW MODEL REGISTRY")
print("=" * 70)
print(f"Experiment : {MLFLOW_EXPERIMENT}")
print(f"Run        : XGBoost")
print(f"Run ID     : {run_id}")

# ==========================================================
# REGISTER MODEL
# ==========================================================

model_uri = f"runs:/{run_id}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

print(f"Model name : {REGISTERED_MODEL_NAME}")
print(f"Version    : {model_version.version}")
print("Status     : registered successfully")