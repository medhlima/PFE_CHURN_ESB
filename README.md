# Banking Customer Churn Intelligence Platform

## PFE 2026

End-to-end customer churn prediction and decision-support platform developed for a banking environment.

The project combines data engineering, machine learning, explainable AI, MLOps and business analytics to identify customers with a high probability of churn and support retention-oriented decision making.

---

## Main Objectives

- Prepare and validate banking customer data.
- Analyze historical customer churn behavior.
- Engineer relevant customer, banking and KYC features.
- Train and compare several machine learning models.
- Select and deploy the final XGBoost churn model.
- Generate customer-level churn probabilities.
- Explain predictions using SHAP.
- Track experiments and model versions using MLflow.
- Provide customer and portfolio analytics through a Flask web application.

---

## Platform Modules

### Executive Dashboard

Provides a global overview of the customer portfolio and the deployed model:

- Total customers
- Observed churn rate
- ROC-AUC
- PR-AUC
- Recall
- Precision
- F1 Score
- Deployed model

### Customer Risk Assessment

Provides customer-level churn prediction and explainability:

- Customer search
- Churn probability
- CHURN / NON_CHURN prediction
- Risk level
- Decision threshold
- SHAP explanation
- Main factors increasing or reducing churn risk

### Customer 360

Provides a unified customer profile including:

- Customer identity
- Customer segment and type
- Age and age group
- Customer tenure
- Number of accounts
- Salary information
- KYC information
- Compliance indicators
- Customer review status and history

### Portfolio Analytics

Provides portfolio-level historical churn analysis by:

- Customer segment
- Customer type
- Age group
- KYC score

It complements individual churn predictions with a business-level view of churn concentration across the customer portfolio.

### Model Governance

Provides visibility into the machine learning lifecycle:

- MLflow experiment
- MLflow run
- Registered model
- Model version
- Registry status
- Model performance metrics
- Prediction service configuration
- SHAP explainability configuration

---

## Machine Learning Model

The final selected and validated model is **XGBoost**.

### Model Performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7122 |
| PR-AUC | 0.2093 |
| Recall | 0.5223 |
| Precision | 0.1167 |
| F1 Score | 0.1907 |
| Accuracy | 0.7604 |

Decision threshold: **0.50**

---

## Explainable AI

The platform integrates **SHAP (SHapley Additive exPlanations)** to provide interpretable customer-level churn predictions.

For each prediction, the application identifies the main factors that:

- increase the predicted churn risk;
- reduce the predicted churn risk.

This provides additional transparency and supports the interpretation of model decisions.

---

## MLOps and Model Governance

MLflow is used throughout the machine learning lifecycle for:

- Experiment tracking
- Model performance tracking
- Artifact management
- Model registration
- Model versioning
- Model governance

### MLflow Configuration

**Experiment:** `Banking_Customer_Churn`

**Registered Model:** `Banking_Churn_Model`

**Final Algorithm:** `XGBoost`

---

## Technologies

### Data Science and Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

### MLOps

- MLflow
- Joblib

### Web Application

- Flask
- HTML
- CSS
- JavaScript

### Development and Analysis

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## Project Structure

The main project directories are organized as follows:

```text
PFE_CHURN_ESB/
│
├── flask_app/
│   ├── services/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   ├── app.py
│   └── config.py
│
├── mlops/
│   ├── compare_models.py
│   ├── config.py
│   ├── register_model.py
│   └── track_experiments.py
│
├── models/
├── notebooks/
├── outputs/
├── src/
│
├── mlflow.db
├── requirements.txt
├── .gitignore
└── README.md
```

### Directory Description

- `src/` — data preparation and feature engineering scripts.
- `notebooks/` — data exploration, preprocessing, modeling, evaluation and explainability notebooks.
- `models/` — trained machine learning model artifacts.
- `mlops/` — MLflow experiment tracking and model registry components.
- `flask_app/` — Flask web application, prediction services and user interface.
- `outputs/` — generated analytical results, figures and evaluation outputs.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Flask Application

From the project root directory, run:

```bash
python -m flask_app.app
```

The application runs locally at:

`http://127.0.0.1:5001`

The main application modules include:

- Executive Dashboard
- Customer Risk Assessment
- Customer 360
- Portfolio Analytics
- Model Governance

---

## MLflow

MLflow is used to inspect experiments, runs, metrics, artifacts and registered model versions.

The project contains the MLflow configuration and tracking components under the `mlops/` directory.

The Model Governance module of the Flask application also exposes the main information associated with the validated and registered model.

---

## Data Confidentiality

The original banking customer datasets are **not published in this repository**.

Raw and processed customer datasets are excluded from GitHub in order to protect confidential banking information and avoid publishing sensitive or large data files.

Consequently, some data-dependent functionalities require the corresponding datasets to be available locally.

---

## Current Project Status

The following components have been implemented:

- Data inventory and quality assessment
- Data preprocessing
- Feature engineering
- Churn target definition
- Exploratory data analysis
- Machine learning preparation
- Baseline model comparison
- XGBoost modeling
- Model evaluation
- SHAP explainability
- MLflow experiment tracking
- MLflow Model Registry
- Flask prediction API
- Executive Dashboard
- Customer Risk Assessment
- Customer 360
- Portfolio Analytics
- Model Governance

---

## Academic Context

**Project:** Projet de Fin d'Études (PFE)  
**Year:** 2026  
**Domain:** Banking, Data Science, Machine Learning and Business Intelligence  
**Topic:** Customer Churn Prediction and Decision-Support Platform