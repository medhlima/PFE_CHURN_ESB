import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


def get_engine() -> Engine:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "atb_churn")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise ValueError(
            f"DB_PASSWORD est absent du fichier .env : {ENV_FILE}"
        )

    url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        future=True
    )


def test_connection():
    engine = get_engine()

    with engine.connect() as connection:
        db_name = connection.execute(
            text("SELECT current_database();")
        ).scalar()

        version = connection.execute(
            text("SELECT version();")
        ).scalar()

    print("[OK] PostgreSQL connecté")
    print(f"[OK] Database : {db_name}")
    print(f"[OK] Version  : {version}")