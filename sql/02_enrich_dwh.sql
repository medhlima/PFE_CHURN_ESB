-- ============================================================
-- ATB CHURN - DWH ENRICHMENT
-- ============================================================

ALTER TABLE dwh.dim_customer
    ALTER COLUMN tenure TYPE NUMERIC(10,2)
    USING tenure::NUMERIC;

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS marital_status VARCHAR(100);

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS customer_type VARCHAR(100);

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS kyc_score VARCHAR(100);

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS file_status VARCHAR(100);

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS review_status VARCHAR(100);

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS is_tunisian BOOLEAN;

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS is_resident BOOLEAN;

ALTER TABLE dwh.dim_customer
    ADD COLUMN IF NOT EXISTS completed_file BOOLEAN;


-- ============================================================
-- DIM BRANCH
-- ============================================================

CREATE TABLE IF NOT EXISTS dwh.dim_branch (
    branch_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    branch_code VARCHAR(100) NOT NULL UNIQUE,
    zone VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- FACT CUSTOMER SNAPSHOT
-- ============================================================

ALTER TABLE dwh.fact_customer_snapshot
    ADD COLUMN IF NOT EXISTS branch_key BIGINT;

ALTER TABLE dwh.fact_customer_snapshot
    ADD COLUMN IF NOT EXISTS avg_balance NUMERIC(18,2);

ALTER TABLE dwh.fact_customer_snapshot
    ADD COLUMN IF NOT EXISTS active_accounts INTEGER;

ALTER TABLE dwh.fact_customer_snapshot
    ADD COLUMN IF NOT EXISTS active_account_ratio NUMERIC(10,6);

ALTER TABLE dwh.fact_customer_snapshot
    ADD COLUMN IF NOT EXISTS high_value_customer BOOLEAN;


-- ============================================================
-- FACT CHURN SCORING
-- ============================================================

ALTER TABLE dwh.fact_churn_scoring
    ADD COLUMN IF NOT EXISTS branch_key BIGINT;

ALTER TABLE dwh.fact_churn_scoring
    ADD COLUMN IF NOT EXISTS churn_score NUMERIC(10,6);


-- ============================================================
-- FOREIGN KEYS
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_snapshot_branch'
    ) THEN
        ALTER TABLE dwh.fact_customer_snapshot
        ADD CONSTRAINT fk_snapshot_branch
        FOREIGN KEY (branch_key)
        REFERENCES dwh.dim_branch(branch_key);
    END IF;
END $$;


DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_scoring_branch'
    ) THEN
        ALTER TABLE dwh.fact_churn_scoring
        ADD CONSTRAINT fk_scoring_branch
        FOREIGN KEY (branch_key)
        REFERENCES dwh.dim_branch(branch_key);
    END IF;
END $$;


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_snapshot_branch
ON dwh.fact_customer_snapshot(branch_key);

CREATE INDEX IF NOT EXISTS idx_scoring_branch
ON dwh.fact_churn_scoring(branch_key);
