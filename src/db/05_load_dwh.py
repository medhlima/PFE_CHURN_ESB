from __future__ import annotations

import io
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import get_engine


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGING_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "staging.parquet"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cible_client.parquet"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "customer_feature_dataset.csv"
)

RISK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_risk_scores.parquet"
)

BI_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "power_bi"
    / "customer_bi_dataset.parquet"
)

MODELS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_models_comparison.xlsx"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "threshold_analysis.xlsx"
)


# ============================================================
# UTILITIES
# ============================================================

def normalize_customer_id(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_code(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result.columns = (
        result.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    return result


def copy_dataframe(
    engine,
    df: pd.DataFrame,
    schema: str,
    table: str,
    replace: bool = True,
    chunk_size: int = 50000,
):
    """
    Fast PostgreSQL load using COPY.
    """

    df = normalize_columns(df)

    if replace:
        df.head(0).to_sql(
            table,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
        )

    columns_sql = ", ".join(
        f'"{column}"'
        for column in df.columns
    )

    raw_connection = engine.raw_connection()

    try:
        cursor = raw_connection.cursor()

        total = len(df)

        for start in range(0, total, chunk_size):

            end = min(
                start + chunk_size,
                total
            )

            chunk = df.iloc[start:end]

            buffer = io.StringIO()

            chunk.to_csv(
                buffer,
                index=False,
                header=False,
                na_rep="__NULL__",
            )

            buffer.seek(0)

            sql = f"""
                COPY "{schema}"."{table}"
                ({columns_sql})
                FROM STDIN
                WITH (
                    FORMAT CSV,
                    NULL '__NULL__'
                )
            """

            cursor.copy_expert(
                sql,
                buffer
            )

        raw_connection.commit()

    except Exception:
        raw_connection.rollback()
        raise

    finally:
        raw_connection.close()

    print(
        f"[OK] {schema}.{table}: "
        f"{len(df):,} lignes"
    )


# ============================================================
# AUDIT
# ============================================================

def start_run(engine) -> int:

    with engine.begin() as connection:

        run_id = connection.execute(
            text("""
                INSERT INTO audit.etl_run_log (
                    pipeline_name,
                    execution_status
                )
                VALUES (
                    'ATB_CHURN_DWH_PIPELINE',
                    'RUNNING'
                )
                RETURNING run_id;
            """)
        ).scalar_one()

    return run_id


def finish_run(
    engine,
    run_id: int,
    rows_extracted: int,
    rows_transformed: int,
    rows_loaded: int,
):

    with engine.begin() as connection:

        connection.execute(
            text("""
                UPDATE audit.etl_run_log

                SET
                    end_time = CURRENT_TIMESTAMP,
                    rows_extracted =
                        :rows_extracted,
                    rows_transformed =
                        :rows_transformed,
                    rows_loaded =
                        :rows_loaded,
                    execution_status =
                        'SUCCESS'

                WHERE run_id = :run_id;
            """),
            {
                "run_id": run_id,
                "rows_extracted":
                    rows_extracted,
                "rows_transformed":
                    rows_transformed,
                "rows_loaded":
                    rows_loaded,
            }
        )


def fail_run(
    engine,
    run_id: int,
    exc: Exception,
):

    with engine.begin() as connection:

        connection.execute(
            text("""
                UPDATE audit.etl_run_log

                SET
                    end_time =
                        CURRENT_TIMESTAMP,
                    execution_status =
                        'FAILED',
                    error_message =
                        :error_message

                WHERE run_id =
                    :run_id;
            """),
            {
                "run_id": run_id,
                "error_message":
                    str(exc)[:5000],
            }
        )


# ============================================================
# LOAD SOURCES
# ============================================================

def read_sources():

    print("\n[1] Reading existing project outputs")

    staging = pd.read_parquet(
        STAGING_PATH
    )

    target = pd.read_parquet(
        TARGET_PATH
    )

    risk = pd.read_parquet(
        RISK_PATH
    )

    bi = pd.read_parquet(
        BI_PATH
    )

    models = pd.read_excel(
        MODELS_PATH
    )

    thresholds = pd.read_excel(
        THRESHOLD_PATH
    )

    # We only need a few columns from the
    # 91-column feature dataset.
    feature_cols = [
        "CUSTOMER_NO",
        "TOTAL_BALANCE",
        "AVG_BALANCE",
        "NB_ACCOUNTS",
        "ACTIVE_ACCOUNTS",
        "ACTIVE_ACCOUNT_RATIO",
        "HIGH_VALUE_CUSTOMER",
        "HAS_ACTIVE_ACCOUNT",
        "MAIN_BRANCH",
    ]

    features = pd.read_csv(
        FEATURES_PATH,
        usecols=feature_cols,
        low_memory=False,
    )

    return (
        staging,
        target,
        features,
        risk,
        bi,
        models,
        thresholds,
    )


# ============================================================
# CORE LAYER
# ============================================================

def load_core(
    engine,
    staging,
    target,
    bi,
):

    print("\n[2] Loading STAGING / CORE")

    copy_dataframe(
        engine,
        staging,
        "staging",
        "stg_customer_raw",
    )

    copy_dataframe(
        engine,
        target,
        "core",
        "customer_target",
    )

    copy_dataframe(
        engine,
        bi,
        "core",
        "customer_features",
    )


# ============================================================
# DIM DATE
# ============================================================

def load_dim_date(
    engine,
    staging,
):

    print("\n[3] Loading DIM_DATE")

    date_columns = [
        "CUST_OPENING_DATE",
        "DATE_OF_BIRTH",
        "LAST_REVIEW_DATE",
        "NEXT__REVIEW_DATE",
        "ACCT_OPENING_DATE",
        "ACCT_CLOSE_DATE",
    ]

    values = []

    for column in date_columns:

        if column in staging.columns:

            series = pd.to_datetime(
                staging[column],
                errors="coerce"
            )

            values.append(series.min())
            values.append(series.max())

    valid_dates = [
        value
        for value in values
        if pd.notna(value)
        and value.year >= 1900
    ]

    if valid_dates:
        start_date = min(valid_dates)
    else:
        start_date = pd.Timestamp(
            "2000-01-01"
        )

    # We never need future dimension dates
    # beyond a reasonable BI horizon.
    end_date = (
        pd.Timestamp.today()
        + pd.DateOffset(years=2)
    )

    dates = pd.date_range(
        start=start_date.normalize(),
        end=end_date.normalize(),
        freq="D",
    )

    df = pd.DataFrame({
        "full_date": dates
    })

    df["date_key"] = (
        df["full_date"]
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    df["day_number"] = (
        df["full_date"].dt.day
    )

    df["day_name"] = (
        df["full_date"].dt.day_name()
    )

    df["month_number"] = (
        df["full_date"].dt.month
    )

    df["month_name"] = (
        df["full_date"].dt.month_name()
    )

    df["quarter_number"] = (
        df["full_date"].dt.quarter
    )

    df["year_number"] = (
        df["full_date"].dt.year
    )

    df["year_month"] = (
        df["full_date"]
        .dt.strftime("%Y-%m")
    )

    copy_dataframe(
        engine,
        df,
        "dwh",
        "tmp_dim_date",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO dwh.dim_date (
                    date_key,
                    full_date,
                    day_number,
                    day_name,
                    month_number,
                    month_name,
                    quarter_number,
                    year_number,
                    year_month
                )

                SELECT
                    date_key,
                    full_date,
                    day_number,
                    day_name,
                    month_number,
                    month_name,
                    quarter_number,
                    year_number,
                    year_month

                FROM dwh.tmp_dim_date

                ON CONFLICT (date_key)
                DO NOTHING;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS dwh.tmp_dim_date;
            """)
        )

    print(
        f"[OK] DIM_DATE: "
        f"{len(df):,} dates"
    )


# ============================================================
# DIM BRANCH
# ============================================================

def mode_or_first(series):

    series = (
        series
        .dropna()
        .astype("string")
    )

    if series.empty:
        return None

    mode = series.mode()

    if not mode.empty:
        return mode.iloc[0]

    return series.iloc[0]


def build_branch_dimension(
    staging,
):

    df = staging[
        [
            "BRANCH",
            "LIB_ZONE"
        ]
    ].copy()

    df["BRANCH"] = normalize_code(
        df["BRANCH"]
    )

    df = df[
        df["BRANCH"].notna()
    ]

    branch = (
        df
        .groupby(
            "BRANCH",
            as_index=False
        )
        .agg({
            "LIB_ZONE": mode_or_first
        })
    )

    branch = branch.rename(
        columns={
            "BRANCH":
                "branch_code",
            "LIB_ZONE":
                "zone",
        }
    )

    return branch


def load_dim_branch(
    engine,
    staging,
):

    print("\n[4] Loading DIM_BRANCH")

    df = build_branch_dimension(
        staging
    )

    copy_dataframe(
        engine,
        df,
        "dwh",
        "tmp_dim_branch",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO dwh.dim_branch (
                    branch_code,
                    zone
                )

                SELECT
                    branch_code,
                    zone

                FROM dwh.tmp_dim_branch

                ON CONFLICT (branch_code)
                DO UPDATE SET
                    zone =
                        EXCLUDED.zone,
                    updated_at =
                        CURRENT_TIMESTAMP;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS dwh.tmp_dim_branch;
            """)
        )

    print(
        f"[OK] DIM_BRANCH: "
        f"{len(df):,} branches"
    )


# ============================================================
# DIM CUSTOMER
# ============================================================

def build_dim_customer(
    bi,
):

    df = bi.copy()

    df["CUSTOMER_NO"] = (
        normalize_customer_id(
            df["CUSTOMER_NO"]
        )
    )

    result = pd.DataFrame()

    result["customer_id"] = (
        df["CUSTOMER_NO"]
    )

    result["age"] = pd.to_numeric(
        df["AGE"],
        errors="coerce"
    )

    result["age_group"] = (
        df["TRANCHE_AGE"]
        .astype("string")
    )

    result["tenure"] = pd.to_numeric(
        df["ANCIENNETE_CLIENT_ANNEES"],
        errors="coerce"
    )

    # The existing project already has
    # categorized salary status.
    result["salary"] = pd.to_numeric(
        df["SALAIRE"],
        errors="coerce"
    )

    result["salary_band"] = (
        df["SALARY_STATUS"]
        .astype("string")
    )

    result["customer_segment"] = (
        df["SEGMENT"]
        .astype("string")
    )

    result["kyc_status"] = (
        df["KYC_RISK_STATUS"]
        .astype("string")
    )

    result["kyc_score"] = (
        df["SCORE_KYC"]
        .astype("string")
    )

    result["residence_status"] = np.where(
        df["EST_RESIDENT"]
        .fillna(False)
        .astype(bool),
        "RESIDENT",
        "NON_RESIDENT",
    )

    result["marital_status"] = (
        df["SITUATION_FAMILIALE"]
        .astype("string")
    )

    result["customer_type"] = (
        df["TYPE_CLIENT"]
        .astype("string")
    )

    result["file_status"] = (
        df["FILE_STATUS"]
        .astype("string")
    )

    result["review_status"] = (
        df["REVIEW_STATUS"]
        .astype("string")
    )

    result["is_tunisian"] = (
        df["EST_TUNISIEN"]
        .astype("boolean")
    )

    result["is_resident"] = (
        df["EST_RESIDENT"]
        .astype("boolean")
    )

    result["completed_file"] = (
        df["DOSSIER_COMPLET"]
        .astype("boolean")
    )

    return (
        result
        .drop_duplicates(
            subset=["customer_id"]
        )
        .reset_index(drop=True)
    )


def load_dim_customer(
    engine,
    bi,
):

    print("\n[5] Loading DIM_CUSTOMER")

    df = build_dim_customer(
        bi
    )

    copy_dataframe(
        engine,
        df,
        "dwh",
        "tmp_dim_customer",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO dwh.dim_customer (
                    customer_id,
                    age,
                    age_group,
                    tenure,
                    salary,
                    salary_band,
                    customer_segment,
                    kyc_status,
                    kyc_score,
                    residence_status,
                    marital_status,
                    customer_type,
                    file_status,
                    review_status,
                    is_tunisian,
                    is_resident,
                    completed_file
                )

                SELECT
                    customer_id,
                    age,
                    age_group,
                    tenure,
                    salary,
                    salary_band,
                    customer_segment,
                    kyc_status,
                    kyc_score,
                    residence_status,
                    marital_status,
                    customer_type,
                    file_status,
                    review_status,
                    is_tunisian,
                    is_resident,
                    completed_file

                FROM dwh.tmp_dim_customer

                ON CONFLICT (customer_id)
                DO UPDATE SET

                    age =
                        EXCLUDED.age,

                    age_group =
                        EXCLUDED.age_group,

                    tenure =
                        EXCLUDED.tenure,

                    salary =
                        EXCLUDED.salary,

                    salary_band =
                        EXCLUDED.salary_band,

                    customer_segment =
                        EXCLUDED.customer_segment,

                    kyc_status =
                        EXCLUDED.kyc_status,

                    kyc_score =
                        EXCLUDED.kyc_score,

                    residence_status =
                        EXCLUDED.residence_status,

                    marital_status =
                        EXCLUDED.marital_status,

                    customer_type =
                        EXCLUDED.customer_type,

                    file_status =
                        EXCLUDED.file_status,

                    review_status =
                        EXCLUDED.review_status,

                    is_tunisian =
                        EXCLUDED.is_tunisian,

                    is_resident =
                        EXCLUDED.is_resident,

                    completed_file =
                        EXCLUDED.completed_file,

                    updated_at =
                        CURRENT_TIMESTAMP;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS dwh.tmp_dim_customer;
            """)
        )

    print(
        f"[OK] DIM_CUSTOMER: "
        f"{len(df):,} clients"
    )


# ============================================================
# THRESHOLD
# ============================================================

def infer_operational_threshold(
    bi,
    thresholds,
) -> float:

    probabilities = pd.to_numeric(
        bi["CHURN_PROBABILITY"],
        errors="coerce"
    )

    actual_predictions = pd.to_numeric(
        bi["PREDICTED_CHURN"],
        errors="coerce"
    )

    candidates = pd.to_numeric(
        thresholds["THRESHOLD"],
        errors="coerce"
    ).dropna()

    if candidates.empty:
        return 0.5

    best_threshold = None
    best_agreement = -1

    for threshold in candidates:

        predicted = (
            probabilities >= threshold
        ).astype(int)

        agreement = (
            predicted
            == actual_predictions
        ).mean()

        if agreement > best_agreement:

            best_agreement = agreement
            best_threshold = float(
                threshold
            )

    print(
        "[INFO] Operational threshold "
        f"inferred from existing outputs: "
        f"{best_threshold:.4f}"
    )

    print(
        "[INFO] Agreement with existing "
        f"PREDICTED_CHURN: "
        f"{best_agreement:.4%}"
    )

    return best_threshold


# ============================================================
# DIM RISK
# ============================================================

def load_dim_risk(
    engine,
    risk,
):

    print("\n[6] Loading DIM_RISK")

    df = risk[
        [
            "CHURN_PROBABILITY",
            "RISK_LEVEL"
        ]
    ].copy()

    df["RISK_LEVEL"] = (
        df["RISK_LEVEL"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    df["CHURN_PROBABILITY"] = (
        pd.to_numeric(
            df["CHURN_PROBABILITY"],
            errors="coerce"
        )
    )

    summary = (
        df
        .groupby(
            "RISK_LEVEL",
            observed=True
        )["CHURN_PROBABILITY"]
        .agg(
            min_probability="min",
            max_probability="max",
            median_probability="median",
        )
        .reset_index()
        .sort_values(
            "median_probability"
        )
        .reset_index(drop=True)
    )

    rows = []

    total_levels = len(summary)

    for index, row in summary.iterrows():

        level = row["RISK_LEVEL"]

        if "HIGH" in level:
            action = (
                "Priority retention intervention"
            )

        elif "MEDIUM" in level:
            action = (
                "Targeted retention monitoring"
            )

        else:
            action = (
                "Standard portfolio monitoring"
            )

        rows.append({
            "risk_level": level,

            # 1 = highest priority
            "risk_priority":
                total_levels - index,

            "min_probability":
                float(
                    row[
                        "min_probability"
                    ]
                ),

            "max_probability":
                float(
                    row[
                        "max_probability"
                    ]
                ),

            "recommended_action":
                action,
        })

    risk_dim = pd.DataFrame(rows)

    copy_dataframe(
        engine,
        risk_dim,
        "dwh",
        "tmp_dim_risk",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO dwh.dim_risk (
                    risk_level,
                    risk_priority,
                    min_probability,
                    max_probability,
                    recommended_action
                )

                SELECT
                    risk_level,
                    risk_priority,
                    min_probability,
                    max_probability,
                    recommended_action

                FROM dwh.tmp_dim_risk

                ON CONFLICT (risk_level)
                DO UPDATE SET

                    risk_priority =
                        EXCLUDED.risk_priority,

                    min_probability =
                        EXCLUDED.min_probability,

                    max_probability =
                        EXCLUDED.max_probability,

                    recommended_action =
                        EXCLUDED.recommended_action;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS dwh.tmp_dim_risk;
            """)
        )

    print(
        f"[OK] DIM_RISK: "
        f"{len(risk_dim)} niveaux"
    )


# ============================================================
# DIM MODEL
# ============================================================

def load_dim_model(
    engine,
    models,
    threshold,
):

    print("\n[7] Loading DIM_MODEL")

    df = models.copy()

    model_name_col = df.columns[0]

    rows = []

    for _, row in df.iterrows():

        model_name = str(
            row[model_name_col]
        ).strip()

        normalized = (
            model_name
            .lower()
            .replace(" ", "")
        )

        is_xgboost = (
            "xgboost" in normalized
            or "xgb" in normalized
        )

        rows.append({

            "model_name":
                model_name,

            "model_version":
                "current",

            "mlflow_run_id":
                None,

            "mlflow_model_uri":
                None,

            "decision_threshold":
                threshold
                if is_xgboost
                else None,

            "roc_auc":
                row.get(
                    "ROC_AUC",
                    None
                ),

            "pr_auc":
                row.get(
                    "PR_AUC",
                    None
                ),

            "precision_score":
                row.get(
                    "Precision",
                    None
                ),

            "recall_score":
                row.get(
                    "Recall",
                    None
                ),

            "f1_score":
                row.get(
                    "F1",
                    None
                ),

            "model_status":
                "ACTIVE"
                if is_xgboost
                else "CANDIDATE",
        })

    model_dim = pd.DataFrame(
        rows
    )

    copy_dataframe(
        engine,
        model_dim,
        "dwh",
        "tmp_dim_model",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO dwh.dim_model (
                    model_name,
                    model_version,
                    mlflow_run_id,
                    mlflow_model_uri,
                    decision_threshold,
                    roc_auc,
                    pr_auc,
                    precision_score,
                    recall_score,
                    f1_score,
                    model_status
                )

                SELECT
                    model_name,
                    model_version,
                    mlflow_run_id,
                    mlflow_model_uri,
                    decision_threshold,
                    roc_auc,
                    pr_auc,
                    precision_score,
                    recall_score,
                    f1_score,
                    model_status

                FROM dwh.tmp_dim_model

                ON CONFLICT (
                    model_name,
                    model_version
                )
                DO UPDATE SET

                    decision_threshold =
                        EXCLUDED.decision_threshold,

                    roc_auc =
                        EXCLUDED.roc_auc,

                    pr_auc =
                        EXCLUDED.pr_auc,

                    precision_score =
                        EXCLUDED.precision_score,

                    recall_score =
                        EXCLUDED.recall_score,

                    f1_score =
                        EXCLUDED.f1_score,

                    model_status =
                        EXCLUDED.model_status;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS dwh.tmp_dim_model;
            """)
        )

    print(
        f"[OK] DIM_MODEL: "
        f"{len(model_dim)} modèles"
    )


# ============================================================
# CUSTOMER AGGREGATES
# ============================================================

def build_customer_aggregates(
    features,
):

    df = features.copy()

    df["CUSTOMER_NO"] = (
        normalize_customer_id(
            df["CUSTOMER_NO"]
        )
    )

    df["MAIN_BRANCH"] = (
        normalize_code(
            df["MAIN_BRANCH"]
        )
    )

    # These variables are already generated at
    # customer level by the existing feature
    # engineering pipeline and can be repeated
    # over account rows.
    #
    # Therefore we DO NOT SUM TOTAL_BALANCE.
    result = (
        df
        .sort_values(
            "CUSTOMER_NO"
        )
        .drop_duplicates(
            subset=["CUSTOMER_NO"],
            keep="first"
        )
    )

    return result


# ============================================================
# FACT CUSTOMER SNAPSHOT
# ============================================================

def load_fact_snapshot(
    engine,
    bi,
    customer_aggregates,
):

    print(
        "\n[8] Loading "
        "FACT_CUSTOMER_SNAPSHOT"
    )

    base = bi.copy()

    base["CUSTOMER_NO"] = (
        normalize_customer_id(
            base["CUSTOMER_NO"]
        )
    )

    agg = customer_aggregates.copy()

    merged = base.merge(
        agg,
        on="CUSTOMER_NO",
        how="left",
        suffixes=("", "_FEATURE"),
    )

    customer_dim = pd.read_sql(
        """
        SELECT
            customer_key,
            customer_id
        FROM dwh.dim_customer
        """,
        engine
    )

    branch_dim = pd.read_sql(
        """
        SELECT
            branch_key,
            branch_code
        FROM dwh.dim_branch
        """,
        engine
    )

    customer_dim["customer_id"] = (
        normalize_customer_id(
            customer_dim["customer_id"]
        )
    )

    branch_dim["branch_code"] = (
        normalize_code(
            branch_dim["branch_code"]
        )
    )

    merged = merged.merge(
        customer_dim,
        left_on="CUSTOMER_NO",
        right_on="customer_id",
        how="left",
    )

    merged = merged.merge(
        branch_dim,
        left_on="MAIN_BRANCH",
        right_on="branch_code",
        how="left",
    )

    today = pd.Timestamp.today().normalize()

    date_key = int(
        today.strftime("%Y%m%d")
    )

    fact = pd.DataFrame({

        "customer_key":
            merged["customer_key"],

        "date_key":
            date_key,

        "branch_key":
            merged["branch_key"],

        "account_count":
            pd.to_numeric(
                merged["NB_COMPTES"],
                errors="coerce"
            ),

        "total_balance":
            pd.to_numeric(
                merged["TOTAL_BALANCE"],
                errors="coerce"
            ),

        "avg_balance":
            pd.to_numeric(
                merged["AVG_BALANCE"],
                errors="coerce"
            ),

        "active_accounts":
            pd.to_numeric(
                merged["ACTIVE_ACCOUNTS"],
                errors="coerce"
            ),

        "active_account_ratio":
            pd.to_numeric(
                merged[
                    "ACTIVE_ACCOUNT_RATIO"
                ],
                errors="coerce"
            ),

        "actual_churn":
            pd.to_numeric(
                merged["ACTUAL_CHURN"],
                errors="coerce"
            ),

        "is_active":
            merged[
                "HAS_ACTIVE_ACCOUNT"
            ]
            .fillna(0)
            .astype(bool),

        "high_value_customer":
            merged[
                "HIGH_VALUE_CUSTOMER"
            ]
            .fillna(0)
            .astype(bool),
    })

    fact = fact[
        fact["customer_key"].notna()
    ]

    copy_dataframe(
        engine,
        fact,
        "dwh",
        "tmp_fact_customer_snapshot",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO
                dwh.fact_customer_snapshot (
                    customer_key,
                    date_key,
                    branch_key,
                    account_count,
                    total_balance,
                    avg_balance,
                    active_accounts,
                    active_account_ratio,
                    actual_churn,
                    is_active,
                    high_value_customer
                )

                SELECT
                    customer_key,
                    date_key,
                    branch_key,
                    account_count,
                    total_balance,
                    avg_balance,
                    active_accounts,
                    active_account_ratio,
                    actual_churn,
                    is_active,
                    high_value_customer

                FROM
                    dwh.tmp_fact_customer_snapshot

                ON CONFLICT (
                    customer_key,
                    date_key
                )
                DO UPDATE SET

                    branch_key =
                        EXCLUDED.branch_key,

                    account_count =
                        EXCLUDED.account_count,

                    total_balance =
                        EXCLUDED.total_balance,

                    avg_balance =
                        EXCLUDED.avg_balance,

                    active_accounts =
                        EXCLUDED.active_accounts,

                    active_account_ratio =
                        EXCLUDED.active_account_ratio,

                    actual_churn =
                        EXCLUDED.actual_churn,

                    is_active =
                        EXCLUDED.is_active,

                    high_value_customer =
                        EXCLUDED.high_value_customer;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS
                dwh.tmp_fact_customer_snapshot;
            """)
        )

    print(
        f"[OK] FACT_CUSTOMER_SNAPSHOT: "
        f"{len(fact):,} lignes"
    )


# ============================================================
# FACT CHURN SCORING
# ============================================================

def get_active_model_key(
    engine,
):

    with engine.connect() as connection:

        result = connection.execute(
            text("""
                SELECT model_key
                FROM dwh.dim_model
                WHERE model_status = 'ACTIVE'
                ORDER BY model_key DESC
                LIMIT 1;
            """)
        ).scalar()

    if result is None:
        raise ValueError(
            "No ACTIVE model found "
            "in dwh.dim_model"
        )

    return int(result)


def load_fact_scoring(
    engine,
    bi,
    customer_aggregates,
    threshold,
):

    print(
        "\n[9] Loading "
        "FACT_CHURN_SCORING"
    )

    df = bi.copy()

    df["CUSTOMER_NO"] = (
        normalize_customer_id(
            df["CUSTOMER_NO"]
        )
    )

    agg = customer_aggregates[
        [
            "CUSTOMER_NO",
            "MAIN_BRANCH"
        ]
    ].copy()

    df = df.merge(
        agg,
        on="CUSTOMER_NO",
        how="left",
    )

    customer_dim = pd.read_sql(
        """
        SELECT
            customer_key,
            customer_id
        FROM dwh.dim_customer
        """,
        engine
    )

    branch_dim = pd.read_sql(
        """
        SELECT
            branch_key,
            branch_code
        FROM dwh.dim_branch
        """,
        engine
    )

    risk_dim = pd.read_sql(
        """
        SELECT
            risk_key,
            risk_level
        FROM dwh.dim_risk
        """,
        engine
    )

    customer_dim["customer_id"] = (
        normalize_customer_id(
            customer_dim["customer_id"]
        )
    )

    branch_dim["branch_code"] = (
        normalize_code(
            branch_dim["branch_code"]
        )
    )

    risk_dim["risk_level"] = (
        risk_dim["risk_level"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    df["RISK_LEVEL_NORMALIZED"] = (
        df["RISK_LEVEL"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    df = df.merge(
        customer_dim,
        left_on="CUSTOMER_NO",
        right_on="customer_id",
        how="left",
    )

    df = df.merge(
        branch_dim,
        left_on="MAIN_BRANCH",
        right_on="branch_code",
        how="left",
    )

    df = df.merge(
        risk_dim,
        left_on="RISK_LEVEL_NORMALIZED",
        right_on="risk_level",
        how="left",
    )

    model_key = (
        get_active_model_key(
            engine
        )
    )

    today = pd.Timestamp.today().normalize()

    date_key = int(
        today.strftime("%Y%m%d")
    )

    fact = pd.DataFrame({

        "customer_key":
            df["customer_key"],

        "scoring_date_key":
            date_key,

        "model_key":
            model_key,

        "risk_key":
            df["risk_key"],

        "branch_key":
            df["branch_key"],

        "churn_probability":
            pd.to_numeric(
                df["CHURN_PROBABILITY"],
                errors="coerce"
            ),

        "churn_score":
            pd.to_numeric(
                df["CHURN_SCORE"],
                errors="coerce"
            ),

        "predicted_churn":
            pd.to_numeric(
                df["PREDICTED_CHURN"],
                errors="coerce"
            ),

        "decision_threshold":
            threshold,

        "actual_churn":
            pd.to_numeric(
                df["ACTUAL_CHURN"],
                errors="coerce"
            ),

        "top_risk_factor":
            None,

        "top_protective_factor":
            None,
    })

    fact = fact[
        fact["customer_key"].notna()
        & fact[
            "churn_probability"
        ].notna()
    ]

    copy_dataframe(
        engine,
        fact,
        "dwh",
        "tmp_fact_churn_scoring",
    )

    with engine.begin() as connection:

        connection.execute(
            text("""
                INSERT INTO
                dwh.fact_churn_scoring (
                    customer_key,
                    scoring_date_key,
                    model_key,
                    risk_key,
                    branch_key,
                    churn_probability,
                    churn_score,
                    predicted_churn,
                    decision_threshold,
                    actual_churn,
                    top_risk_factor,
                    top_protective_factor
                )

                SELECT
                    customer_key,
                    scoring_date_key,
                    model_key,
                    risk_key,
                    branch_key,
                    churn_probability,
                    churn_score,
                    predicted_churn,
                    decision_threshold,
                    actual_churn,
                    top_risk_factor,
                    top_protective_factor

                FROM
                    dwh.tmp_fact_churn_scoring

                ON CONFLICT (
                    customer_key,
                    scoring_date_key,
                    model_key
                )

                DO UPDATE SET

                    risk_key =
                        EXCLUDED.risk_key,

                    branch_key =
                        EXCLUDED.branch_key,

                    churn_probability =
                        EXCLUDED.churn_probability,

                    churn_score =
                        EXCLUDED.churn_score,

                    predicted_churn =
                        EXCLUDED.predicted_churn,

                    decision_threshold =
                        EXCLUDED.decision_threshold,

                    actual_churn =
                        EXCLUDED.actual_churn;
            """)
        )

        connection.execute(
            text("""
                DROP TABLE
                IF EXISTS
                dwh.tmp_fact_churn_scoring;
            """)
        )

    print(
        f"[OK] FACT_CHURN_SCORING: "
        f"{len(fact):,} lignes"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_dwh(
    engine,
):

    print("\n" + "=" * 70)
    print("DWH VALIDATION")
    print("=" * 70)

    tables = [
        "dim_customer",
        "dim_date",
        "dim_branch",
        "dim_risk",
        "dim_model",
        "fact_customer_snapshot",
        "fact_churn_scoring",
    ]

    with engine.connect() as connection:

        for table in tables:

            count = connection.execute(
                text(
                    f"""
                    SELECT COUNT(*)
                    FROM dwh.{table};
                    """
                )
            ).scalar()

            print(
                f"{table:<30}"
                f"{count:>12,}"
            )

    print("=" * 70)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    engine = get_engine()

    run_id = start_run(
        engine
    )

    try:

        (
            staging,
            target,
            features,
            risk,
            bi,
            models,
            thresholds,
        ) = read_sources()

        rows_extracted = (
            len(staging)
        )

        # ----------------------------------------------------
        # STAGING / CORE
        # ----------------------------------------------------

        load_core(
            engine,
            staging,
            target,
            bi,
        )

        # ----------------------------------------------------
        # DIMS
        # ----------------------------------------------------

        load_dim_date(
            engine,
            staging
        )

        load_dim_branch(
            engine,
            staging
        )

        load_dim_customer(
            engine,
            bi
        )

        operational_threshold = (
            infer_operational_threshold(
                bi,
                thresholds,
            )
        )

        load_dim_risk(
            engine,
            risk
        )

        load_dim_model(
            engine,
            models,
            operational_threshold,
        )

        # ----------------------------------------------------
        # CUSTOMER AGGREGATES
        # ----------------------------------------------------

        customer_aggregates = (
            build_customer_aggregates(
                features
            )
        )

        # ----------------------------------------------------
        # FACTS
        # ----------------------------------------------------

        load_fact_snapshot(
            engine,
            bi,
            customer_aggregates,
        )

        load_fact_scoring(
            engine,
            bi,
            customer_aggregates,
            operational_threshold,
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validate_dwh(
            engine
        )

        finish_run(
            engine,
            run_id,
            rows_extracted=
                rows_extracted,
            rows_transformed=
                len(bi),
            rows_loaded=
                len(bi),
        )

        print("\n")
        print("=" * 70)
        print(
            "ATB CHURN DWH PIPELINE "
            "COMPLETED SUCCESSFULLY"
        )
        print("=" * 70)

    except Exception as exc:

        fail_run(
            engine,
            run_id,
            exc,
        )

        raise


if __name__ == "__main__":
    main()