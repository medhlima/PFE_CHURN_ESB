from datetime import datetime

import mlflow
from mlflow import MlflowClient

from mlops.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)


REGISTERED_MODEL_NAME = "Banking_Churn_Model"


class ModelService:

    def __init__(self):

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        self.client = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI
        )

    @staticmethod
    def _format_timestamp(timestamp_ms):

        if not timestamp_ms:
            return None

        return datetime.fromtimestamp(
            timestamp_ms / 1000
        ).strftime(
            "%d/%m/%Y %H:%M"
        )

    def get_governance_summary(self):

        # ==================================================
        # EXPERIMENT
        # ==================================================

        experiment = (
            self.client.get_experiment_by_name(
                MLFLOW_EXPERIMENT
            )
        )

        if experiment is None:
            raise RuntimeError(
                f"Experiment not found: "
                f"{MLFLOW_EXPERIMENT}"
            )

        # ==================================================
        # LATEST SUCCESSFUL XGBOOST RUN
        # ==================================================

        runs = self.client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ],
            filter_string=(
                'attributes.run_name = "XGBoost" '
                'AND attributes.status = "FINISHED"'
            ),
            order_by=[
                "attributes.start_time DESC"
            ],
            max_results=1,
        )

        if not runs:
            raise RuntimeError(
                "No successful XGBoost run found."
            )

        run = runs[0]

        # ==================================================
        # REGISTERED MODEL VERSION
        # ==================================================

        versions = (
            self.client.search_model_versions(
                filter_string=(
                    f"name = "
                    f"'{REGISTERED_MODEL_NAME}'"
                )
            )
        )

        latest_version = None

        if versions:

            latest_version = max(
                versions,
                key=lambda item:
                    int(item.version)
            )

        # ==================================================
        # METRICS
        # ==================================================

        metrics = run.data.metrics

        return {

            "model": {
                "registered_name":
                    REGISTERED_MODEL_NAME,

                "algorithm":
                    "XGBoost",

                "version": (
                    latest_version.version
                    if latest_version
                    else None
                ),

                "registry_status": (
                    "REGISTERED"
                    if latest_version
                    else "NOT_REGISTERED"
                ),

                "source": (
                    latest_version.source
                    if latest_version
                    else None
                ),
            },

            "experiment": {
                "name":
                    experiment.name,

                "experiment_id":
                    experiment.experiment_id,

                "lifecycle_stage":
                    experiment.lifecycle_stage,
            },

            "run": {
                "name":
                    run.data.tags.get(
                        "mlflow.runName",
                        "XGBoost"
                    ),

                "run_id":
                    run.info.run_id,

                "status":
                    run.info.status,

                "created_at":
                    self._format_timestamp(
                        run.info.start_time
                    ),
            },

            "metrics": {
                "accuracy":
                    round(
                        metrics.get(
                            "accuracy",
                            0
                        ),
                        4
                    ),

                "precision":
                    round(
                        metrics.get(
                            "precision",
                            0
                        ),
                        4
                    ),

                "recall":
                    round(
                        metrics.get(
                            "recall",
                            0
                        ),
                        4
                    ),

                "f1":
                    round(
                        metrics.get(
                            "f1",
                            0
                        ),
                        4
                    ),

                "roc_auc":
                    round(
                        metrics.get(
                            "roc_auc",
                            0
                        ),
                        4
                    ),

                "pr_auc":
                    round(
                        metrics.get(
                            "pr_auc",
                            0
                        ),
                        4
                    ),
            },

            "deployment": {
                "decision_threshold":
                    0.50,

                "api_status":
                    "ONLINE",

                "prediction_service":
                    "Flask",

                "explainability":
                    "SHAP",
            }
        }