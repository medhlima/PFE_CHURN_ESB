"""
config.py — Point unique de configuration du projet.

Adapte a l'arborescence PFE_CHURN_ESB :

    PFE_CHURN_ESB/
    ├── data/
    │   ├── raw/          fichiers livres par l'encadrante
    │   ├── reference/    tables dim_*
    │   ├── interim/      couche staging (cree automatiquement)
    │   └── processed/    datasets finaux
    ├── notebooks/
    ├── outputs/
    │   └── audit/        rapports d'audit (cree automatiquement)
    └── src/              ce dossier

Un seul fichier a modifier si l'arborescence change.
"""

from pathlib import Path

import pandas as pd

# --- Racine du projet -------------------------------------------------
# src/config.py -> .parent = src/ -> .parent.parent = PFE_CHURN_ESB/
ROOT = Path(__file__).resolve().parent.parent

# --- Couches de donnees -----------------------------------------------
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "audit"

for _d in (INTERIM, PROCESSED, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def trouver(nom: str) -> Path:
    """
    Localise un fichier source.

    Les tables dim_* peuvent se trouver dans data/reference/ ou dans
    data/raw/ selon la facon dont le dossier a ete range. On cherche
    dans les deux plutot que d'imposer un emplacement.
    """
    for dossier in (RAW, REFERENCE):
        chemin = dossier / nom
        if chemin.exists():
            return chemin
    raise FileNotFoundError(
        f"{nom} introuvable dans {RAW} ni dans {REFERENCE}")


# --- Fichiers sources -------------------------------------------------
F_DATA = trouver("DATA (1).csv")
F_JM = trouver("JM.xlsx")
F_ACCOUNTS = trouver("ATB_ACCOUNTS (1).xlsx")

# Referentiels : cle = nom logique, valeur = (fichier, colonne de code)
DIMENSIONS = {
    "CATEGORY": ("dim_CATEGORY.ACCOUNT.xlsx", "CATEGORY_id"),
    "CURRENCY": ("dim_CURRENCY.xlsx", "CURRENCY_CODE"),
    "CLOSURE_REASON": ("dim_Closure_reason.xlsx", "RECID"),
    "DAO": ("dim_DAO.xlsx", "ACCOUNT_OFFICER"),
    "INDUSTRY": ("dim_INDUSTRY.xlsx", "INDUSTRY_CODE"),
    "TARGET": ("dim_TARGET.xlsx", "TARGET_CODE"),
    "TRANSACTION": ("dim_TRANSACTION.xlsx", "Id"),
}

# --- Parametres de lecture du CSV -------------------------------------
# Export SQL francophone : separateur ';', encodage UTF-8 avec BOM
# ('utf-8-sig' retire le BOM, sinon la 1re colonne s'appelle
# '\ufeffCUSTOMER_NO' et toute jointure sur elle echoue en silence),
# absences codees en 'NULL'.
CSV_READ_OPTS = dict(
    sep=";",
    encoding="utf-8-sig",
    dtype=str,
    na_values=["NULL", "", "NULL "],
    keep_default_na=True,
    low_memory=False,
)

# --- Colonnes interdites en variables explicatives --------------------
# Elles servent a CONSTRUIRE la cible. Les laisser dans les features
# produirait une fuite de donnees et un modele indefendable.
LEAKAGE_COLS = [
    "ACCOUNT_STATUS",
    "ACCT_CLOSE_DATE",
    "CLOSURE_REASON",
    "PRODUCT_STATUS",
]

# --- Cles de jointure --------------------------------------------------
KEY_CUSTOMER = "CUSTOMER_NO"
KEY_ACCOUNT = "ACCOUNT_NO"

# --- Reperes temporels -------------------------------------------------
DATE_EXTRACTION = pd.Timestamp("2026-02-19")

# --- Typage ------------------------------------------------------------
DATE_COLS = [
    "CUST_OPENING_DATE",
    "DATE_OF_BIRTH",
    "LAST_REVIEW_DATE",
    "NEXT__REVIEW_DATE",
    "ACCT_OPENING_DATE",
    "ACCT_CLOSE_DATE",
]

# STARTDATE et MATURITYDATE melangent deux formats (YYYYMMDD et le
# format T24 a 7 chiffres, 1251227 = 2025-12-27). Traitees a part.
DATE_COLS_AMBIGUES = ["STARTDATE", "MATURITYDATE"]

NUM_COLS = ["ACCT_BALANCE", "SALARY", "AMOUNT", "FIXEDRATE"]

# --- Bornes de plausibilite -------------------------------------------
AGE_MIN, AGE_MAX = 15, 100
DATE_PLANCHER = pd.Timestamp("1900-01-01")

# --- Definition de la cible -------------------------------------------
# DATE_REFERENCE (T) : date a laquelle on se place pour predire. Toute
# variable explicative doit etre calculable au 31/12/2024 ; tout ce qui
# suit sert uniquement a construire la cible.
# HORIZON_MOIS = 12 : la fenetre 2025 est entierement contenue dans
# l'extraction (arretee au 19/02/2026), donc non tronquee.
DATE_REFERENCE = pd.Timestamp("2024-12-31")
HORIZON_MOIS = 12

# SEUIL_PURGE : au-dela de ce nombre de fermetures dans une meme
# journee, il s'agit d'une campagne administrative et non de departs
# individuels. Volume median : 36 fermetures par jour.
SEUIL_PURGE = 3000
