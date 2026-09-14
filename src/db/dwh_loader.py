from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe column names.
    """
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


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str]
) -> str | None:
    """
    Returns first existing column from list of aliases.
    """
    for column in candidates:
        if column in df.columns:
            return column

    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# AUDIT
# ============================================================

def start_etl_run(
    engine: Engine,
    pipeline_name: str
) -> int:

    sql = text("""
        INSERT INTO audit.etl_run_log (
            pipeline_name,
            start_time,
            execution_status
        )
        VALUES (
            :pipeline_name,
            CURRENT_TIMESTAMP,
            'RUNNING'
        )
        RETURNING run_id;
    """)

    with engine.begin() as connection:
        run_id = connection.execute(
            sql,
            {"pipeline_name": pipeline_name}
        ).scalar_one()

    return run_id


def finish_etl_run(
    engine: Engine,
    run_id: int,
    rows_extracted: int,
    rows_transformed: int,
    rows_loaded: int,
    rows_rejected: int = 0
):

    sql = text("""
        UPDATE audit.etl_run_log
        SET
            end_time = CURRENT_TIMESTAMP,
            rows_extracted = :rows_extracted,
            rows_transformed = :rows_transformed,
            rows_loaded = :rows_loaded,
            rows_rejected = :rows_rejected,
            execution_status = 'SUCCESS'
        WHERE run_id = :run_id;
    """)

    with engine.begin() as connection:
        connection.execute(
            sql,
            {
                "run_id": run_id,
                "rows_extracted": rows_extracted,
                "rows_transformed": rows_transformed,
                "rows_loaded": rows_loaded,
                "rows_rejected": rows_rejected,
            }
        )


def fail_etl_run(
    engine: Engine,
    run_id: int,
    error_message: str
):

    sql = text("""
        UPDATE audit.etl_run_log
        SET
            end_time = CURRENT_TIMESTAMP,
            execution_status = 'FAILED',
            error_message = :error_message
        WHERE run_id = :run_id;
    """)

    with engine.begin() as connection:
        connection.execute(
            sql,
            {
                "run_id": run_id,
                "error_message": error_message[:5000]
            }
        )


# ============================================================
# STAGING
# ============================================================

def load_staging(
    engine: Engine,
    df: pd.DataFrame,
    table_name: str = "stg_customer_raw"
):

    clean_df = normalize_columns(df)

    clean_df.to_sql(
        table_name,
        engine,
        schema="staging",
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi"
    )

    print(
        f"[OK] staging.{table_name}: "
        f"{len(clean_df):,} lignes"
    )


# ============================================================
# CORE FEATURES
# ============================================================

def load_core_features(
    engine: Engine,
    df: pd.DataFrame
):

    features_df = normalize_columns(df)

    features_df.to_sql(
        "customer_features",
        engine,
        schema="core",
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi"
    )

    print(
        f"[OK] core.customer_features: "
        f"{len(features_df):,} lignes"
    )


# ============================================================
# DIM DATE
# ============================================================

def load_dim_date(
    engine: Engine,
    start_date: str = "2020-01-01",
    end_date: str | None = None
):

    if end_date is None:
        end_date = date.today().isoformat()

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    df = pd.DataFrame({
        "full_date": dates
    })

    df["date_key"] = (
        df["full_date"]
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    df["day_number"] = df["full_date"].dt.day
    df["day_name"] = df["full_date"].dt.day_name()

    df["month_number"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.month_name()

    df["quarter_number"] = df["full_date"].dt.quarter
    df["year_number"] = df["full_date"].dt.year

    df["year_month"] = (
        df["full_date"]
        .dt.strftime("%Y-%m")
    )

    df = df[
        [
            "date_key",
            "full_date",
            "day_number",
            "day_name",
            "month_number",
            "month_name",
            "quarter_number",
            "year_number",
            "year_month"
        ]
    ]

    df.to_sql(
        "tmp_dim_date",
        engine,
        schema="dwh",
        if_exists="replace",
        index=False
    )

    sql = text("""
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

    with engine.begin() as connection:
        connection.execute(sql)
        connection.execute(
            text("DROP TABLE IF EXISTS dwh.tmp_dim_date;")
        )

    print(
        f"[OK] dwh.dim_date chargée "
        f"({start_date} -> {end_date})"
    )


# ============================================================
# DIM RISK
# ============================================================

def load_dim_risk(
    engine: Engine,
    low_max: float,
    medium_max: float
):
    """
    IMPORTANT:
    Use real thresholds from the existing Risk Assessment logic.
    Do not invent different thresholds here.
    """

    risk_rows = [
        {
            "risk_level": "LOW",
            "risk_priority": 3,
            "min_probability": 0.0,
            "max_probability": low_max,
            "recommended_action":
                "Monitoring standard"
        },
        {
            "risk_level": "MEDIUM",
            "risk_priority": 2,
            "min_probability": low_max,
            "max_probability": medium_max,
            "recommended_action":
                "Retention monitoring and targeted contact"
        },
        {
            "risk_level": "HIGH",
            "risk_priority": 1,
            "min_probability": medium_max,
            "max_probability": 1.0,
            "recommended_action":
                "Priority retention intervention"
        }
    ]

    sql = text("""
        INSERT INTO dwh.dim_risk (
            risk_level,
            risk_priority,
            min_probability,
            max_probability,
            recommended_action
        )
        VALUES (
            :risk_level,
            :risk_priority,
            :min_probability,
            :max_probability,
            :recommended_action
        )

        ON CONFLICT (risk_level)
        DO UPDATE SET
            risk_priority = EXCLUDED.risk_priority,
            min_probability = EXCLUDED.min_probability,
            max_probability = EXCLUDED.max_probability,
            recommended_action =
                EXCLUDED.recommended_action;
    """)

    with engine.begin() as connection:
        for row in risk_rows:
            connection.execute(sql, row)

    print("[OK] dwh.dim_risk chargée")


# ============================================================
# DIM CUSTOMER
# ============================================================

def build_customer_dimension(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = normalize_columns(df)

    customer_id_col = first_existing_column(
        df,
        [
            "customer_id",
            "customerid",
            "customer_number",
            "customer_no",
            "id_customer"
        ]
    )

    if customer_id_col is None:
        raise ValueError(
            "Impossible de trouver la colonne customer_id."
        )

    result = pd.DataFrame()

    result["customer_id"] = (
        df[customer_id_col]
        .astype(str)
        .str.strip()
    )

    # AGE
    age_col = first_existing_column(
        df,
        ["age", "customer_age"]
    )

    if age_col:
        result["age"] = safe_numeric(df[age_col])

        result["age_group"] = pd.cut(
            result["age"],
            bins=[
                0,
                25,
                35,
                45,
                55,
                65,
                np.inf
            ],
            labels=[
                "<=25",
                "26-35",
                "36-45",
                "46-55",
                "56-65",
                "65+"
            ]
        ).astype("string")

    # TENURE
    tenure_col = first_existing_column(
        df,
        [
            "tenure",
            "customer_tenure",
            "relationship_duration"
        ]
    )

    if tenure_col:
        result["tenure"] = safe_numeric(
            df[tenure_col]
        )

        result["tenure_group"] = pd.cut(
            result["tenure"],
            bins=[
                -np.inf,
                1,
                3,
                5,
                10,
                np.inf
            ],
            labels=[
                "<=1",
                "2-3",
                "4-5",
                "6-10",
                "10+"
            ]
        ).astype("string")

    # SALARY
    salary_col = first_existing_column(
        df,
        [
            "salary",
            "estimatedsalary",
            "estimated_salary",
            "income"
        ]
    )

    if salary_col:
        result["salary"] = safe_numeric(
            df[salary_col]
        )

        result["salary_band"] = pd.qcut(
            result["salary"],
            q=4,
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH",
                "VERY_HIGH"
            ],
            duplicates="drop"
        ).astype("string")

    segment_col = first_existing_column(
        df,
        [
            "customer_segment",
            "segment",
            "customer_category"
        ]
    )

    if segment_col:
        result["customer_segment"] = (
            df[segment_col]
            .astype("string")
        )

    kyc_col = first_existing_column(
        df,
        [
            "kyc_status",
            "kyc",
            "kyc_level"
        ]
    )

    if kyc_col:
        result["kyc_status"] = (
            df[kyc_col]
            .astype("string")
        )

    residence_col = first_existing_column(
        df,
        [
            "residence_status",
            "residency",
            "residence"
        ]
    )

    if residence_col:
        result["residence_status"] = (
            df[residence_col]
            .astype("string")
        )

    # Ensure every target column exists
    expected_columns = [
        "customer_id",
        "age",
        "age_group",
        "tenure",
        "tenure_group",
        "salary",
        "salary_band",
        "customer_segment",
        "kyc_status",
        "residence_status"
    ]

    for column in expected_columns:
        if column not in result.columns:
            result[column] = None

    result = result[expected_columns]

    result = (
        result
        .drop_duplicates(
            subset=["customer_id"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return result


def load_dim_customer(
    engine: Engine,
    source_df: pd.DataFrame
):

    df = build_customer_dimension(source_df)

    df.to_sql(
        "tmp_dim_customer",
        engine,
        schema="dwh",
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi"
    )

    sql = text("""
        INSERT INTO dwh.dim_customer (
            customer_id,
            age,
            age_group,
            tenure,
            tenure_group,
            salary,
            salary_band,
            customer_segment,
            kyc_status,
            residence_status
        )
        SELECT
            customer_id,
            age,
            age_group,
            tenure,
            tenure_group,
            salary,
            salary_band,
            customer_segment,
            kyc_status,
            residence_status
        FROM dwh.tmp_dim_customer

        ON CONFLICT (customer_id)
        DO UPDATE SET
            age = EXCLUDED.age,
            age_group = EXCLUDED.age_group,
            tenure = EXCLUDED.tenure,
            tenure_group = EXCLUDED.tenure_group,
            salary = EXCLUDED.salary,
            salary_band = EXCLUDED.salary_band,
            customer_segment =
                EXCLUDED.customer_segment,
            kyc_status =
                EXCLUDED.kyc_status,
            residence_status =
                EXCLUDED.residence_status,
            updated_at = CURRENT_TIMESTAMP;
    """)

    with engine.begin() as connection:
        connection.execute(sql)

        connection.execute(
            text(
                "DROP TABLE IF EXISTS "
                "dwh.tmp_dim_customer;"
            )
        )

    print(
        f"[OK] dwh.dim_customer: "
        f"{len(df):,} clients préparés"
    )


# ============================================================
# VALIDATION
# ============================================================

def print_dwh_counts(engine: Engine):

    tables = [
        "dim_customer",
        "dim_date",
        "dim_risk",
        "dim_model",
        "fact_customer_snapshot",
        "fact_churn_scoring"
    ]

    print("\n==============================")
    print("DWH ROW COUNTS")
    print("==============================")

    with engine.connect() as connection:

        for table in tables:

            count = connection.execute(
                text(
                    f"SELECT COUNT(*) "
                    f"FROM dwh.{table};"
                )
            ).scalar()

            print(
                f"{table:<30} "
                f"{count:,}"
            )