from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================================
# MLFLOW CONFIGURATION
# ==========================================================

MLFLOW_EXPERIMENT = "Banking_Customer_Churn"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)