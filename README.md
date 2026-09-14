# Banking Customer Churn Intelligence Platform

End-to-end Business Analytics platform for customer churn prediction, explainability, governance and decision support in a banking context.

The project combines:

- Data Engineering
- Machine Learning
- Explainable AI
- MLOps
- PostgreSQL Data Warehousing
- Power BI
- Flask

---

## Project Objective

The platform aims to:

- consolidate and validate banking data;
- define a reliable point-in-time churn target;
- predict customers at risk of churn;
- explain model predictions;
- track and govern machine learning experiments;
- store analytical outputs in a PostgreSQL Data Warehouse;
- support customer-level and portfolio-level decision making.

---

## Analytical Architecture

```text
Banking Data
    ↓
Data Quality & Staging
    ↓
Point-in-Time Churn Target
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
SHAP Explainability
    ↓
MLflow Governance
    ↓
Customer Scoring
    ↓
PostgreSQL Data Warehouse
    ↓
Power BI + Flask Decision Support
```

---

## Key Results

- Final analytical population: **195,120 customers**
- Churned customers: **10,548**
- Actual churn rate: **5.41%**
- Final selected model: **XGBoost**
- Operational threshold: **0.50**

### Final XGBoost Performance

| Metric | Value |
|---|---:|
| Recall | 0.5223 |
| Precision | 0.1167 |
| F1-score | 0.1907 |
| ROC-AUC | 0.7122 |
| PR-AUC | 0.2093 |

---

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

XGBoost was selected based on its recall, ROC-AUC, PR-AUC and compatibility with SHAP explainability.

---

## Explainability & Governance

### SHAP
Used for:

- global feature importance;
- individual churn prediction explanations.

### MLflow
Used to track:

- experiments;
- parameters;
- metrics;
- model artifacts;
- model versions.

Main experiment:

```text
Banking_Customer_Churn
```

---

## PostgreSQL Data Warehouse

Database:

```text
atb_churn
```

Schemas:

```text
staging
core
dwh
audit
```

### Dimensions

- `dim_customer`
- `dim_date`
- `dim_branch`
- `dim_risk`
- `dim_model`

### Fact Tables

- `fact_customer_snapshot`
- `fact_churn_scoring`

Main DWH volumes:

| Object | Rows |
|---|---:|
| staging.stg_customer_raw | 490,244 |
| core.customer_target | 195,120 |
| core.customer_features | 195,120 |
| dwh.fact_customer_snapshot | 195,120 |
| dwh.fact_churn_scoring | 195,120 |

Run the ETL pipeline with:

```bash
python src/05_load_dwh.py
```

---

## Power BI

The final Power BI report contains three dashboards:

### 1. Customer Portfolio Overview
**Descriptive Analytics**

Main focus:
- portfolio structure;
- customer activity;
- account volumes;
- branch distribution.

### 2. Churn Analysis & Drivers
**Diagnostic Analytics**

Main focus:
- actual churn;
- churn concentration;
- decomposition analysis;
- key influencers;
- branch-level churn.

### 3. Retention & Risk Prioritization
**Predictive / Decision Analytics**

Main focus:
- model-flagged customers;
- risk distribution;
- high-value customers at risk;
- retention prioritization.

Final Power BI file:

```text
power-bi/ATB_Customer_Churn_Analytics.pbix
```

---

## Flask Application

The Flask application provides customer-level decision support.

Main modules:

- Dashboard
- Customer Risk Assessment
- Customer 360
- Analytics
- Power BI Analytics
- Model Governance

Run the application from the project root:

```bash
python -m flask_app.app
```

Then open:

```text
http://127.0.0.1:5001
```

Power BI is embedded directly inside the Flask application.

---

## Flask vs Power BI

| Flask | Power BI |
|---|---|
| Individual customer analysis | Portfolio-level analysis |
| Customer Risk Assessment | Portfolio monitoring |
| Customer 360 | Churn diagnostics |
| Local SHAP explanation | Segmentation |
| Model Governance | Retention prioritization |

---

## Main Technologies

### Data Engineering
- Python
- Pandas
- NumPy
- PyArrow
- PostgreSQL
- SQLAlchemy
- psycopg2

### Machine Learning
- Scikit-learn
- XGBoost

### Explainability & MLOps
- SHAP
- MLflow

### Decision Support
- Flask
- Power BI
- HTML / CSS / JavaScript

### Development
- Jupyter Notebook
- Git
- GitHub
- Visual Studio Code

---

## Project Structure

```text
PFE_CHURN_ESB/
│
├── flask_app/
├── mlops/
├── models/
├── notebooks/
├── outputs/
├── power-bi/
│   └── ATB_Customer_Churn_Analytics.pbix
├── sql/
├── src/
│   ├── 05_load_dwh.py
│   └── db/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Security

Real banking data and database credentials are not included in the repository.

The `.env` file is excluded from Git.

---

## Academic Context

Final Year Project  
Master's Degree in Business Analytics  
Academic Year: **2025–2026**

Project domain:

**Banking Customer Churn Prediction, Explainable AI and Decision Intelligence**