"""
02_staging.py — Phase 2 : construction de la couche staging.

Perimetre STRICT de cette phase :
  1. suppression des doublons exacts
  2. typage explicite (dates, numeriques, categorielles)
  3. controles de coherence metier, avec tracabilite
  4. jointure des libelles issus des referentiels dim_*

Ce qui n'est PAS fait ici, volontairement :
  - aucune imputation. Remplacer une valeur manquante suppose de choisir
    une statistique (moyenne, mediane, mode) ; si cette statistique est
    calculee sur l'ensemble des donnees, l'information du jeu de test
    contamine l'apprentissage. L'imputation appartient donc au pipeline
    de modelisation, apres separation train/test (phase 7).
  - aucune suppression de colonne. L'audit de phase 1 a montre que les
    manquants sont structurels (un compte courant n'a pas de taux fixe,
    une personne morale n'a pas de salaire) : l'absence est porteuse
    d'information et sera encodee en phase 4.
  - aucune agregation. Le passage au grain client est la phase 4.

Entree  : data/raw/DATA (1).csv
Sorties : data/interim/staging.parquet
          reports/audit/coherence_staging.csv
Execution : python src/02_staging.py
"""

import numpy as np
import pandas as pd

import config as cfg

# Journal des controles de coherence : (regle, nombre de lignes touchees)
JOURNAL = []


def tracer(regle: str, masque: pd.Series, action: str) -> None:
    """Enregistre l'effet d'une regle de coherence."""
    n = int(masque.sum())
    JOURNAL.append({"regle": regle, "lignes_concernees": n, "action": action})
    print(f"  {regle:52s} {n:>8,}  -> {action}")


# ----------------------------------------------------------------------
# 1. DOUBLONS
# ----------------------------------------------------------------------
# Un doublon strict (les 34 colonnes identiques) ne peut pas correspondre
# a deux faits distincts : deux produits differents differeraient au
# moins par PRODUCT ou STARTDATE. C'est donc un artefact d'extraction.
# On les supprime avant tout autre traitement, pour que les statistiques
# calculees ensuite ne soient pas biaisees.
def supprimer_doublons(df: pd.DataFrame) -> pd.DataFrame:
    avant = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\n[1] Doublons stricts supprimes : {avant - len(df):,} "
          f"({avant:,} -> {len(df):,} lignes)")
    return df


# ----------------------------------------------------------------------
# 2. NORMALISATION DU TEXTE
# ----------------------------------------------------------------------
# Les exports T24 contiennent frequemment des espaces de bordure. Un
# 'TND ' et un 'TND' sont deux modalites distinctes pour pandas : la
# jointure avec dim_CURRENCY echouerait sur la premiere sans aucun
# message d'erreur. On normalise donc systematiquement.
def normaliser_texte(df: pd.DataFrame) -> pd.DataFrame:
    obj = df.select_dtypes(include=["object", "string"]).columns
    for c in obj:
        df[c] = df[c].str.strip()
        # Une chaine devenue vide apres strip est une absence, pas une
        # modalite.
        df[c] = df[c].replace("", np.nan)
    print(f"\n[2] Normalisation appliquee a {len(obj)} colonnes texte")
    return df


# ----------------------------------------------------------------------
# 3. TYPAGE DES DATES
# ----------------------------------------------------------------------
def _parser_date_standard(s: pd.Series) -> pd.Series:
    """Format unique YYYYMMDD."""
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def _parser_date_t24(s: pd.Series) -> pd.Series:
    """
    Format mixte des colonnes produit (STARTDATE, MATURITYDATE).

    L'audit a etabli deux encodages coexistants :
      - 8 caracteres : YYYYMMDD classique
      - 7 caracteres : 1YYMMDD, convention T24 ou le '1' de tete designe
        le 21e siecle ('1251227' = 2025-12-27)
    Parser avec un seul format detruirait silencieusement 74 % des
    dates de produit. On traite donc les deux cas separement.

    La valeur sentinelle '-19000000' signale une absence d'echeance.
    """
    s = s.where(s != "-19000000")
    longueur = s.str.len()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m8 = longueur == 8
    out[m8] = pd.to_datetime(s[m8], format="%Y%m%d", errors="coerce")

    m7 = longueur == 7
    out[m7] = pd.to_datetime("20" + s[m7].str[1:], format="%Y%m%d",
                             errors="coerce")
    return out


def typer_dates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] Typage des dates")
    for c in cfg.DATE_COLS:
        brut = df[c].notna().sum()
        df[c] = _parser_date_standard(df[c])
        echecs = brut - df[c].notna().sum()
        print(f"  {c:22s} converties {df[c].notna().sum():>8,} "
              f"| echecs {echecs:,}")

    for c in cfg.DATE_COLS_AMBIGUES:
        brut = df[c].notna().sum()
        df[c] = _parser_date_t24(df[c])
        echecs = brut - df[c].notna().sum()
        print(f"  {c:22s} converties {df[c].notna().sum():>8,} "
              f"| echecs {echecs:,} (format mixte)")
    return df


# ----------------------------------------------------------------------
# 4. TYPAGE NUMERIQUE
# ----------------------------------------------------------------------
# errors='coerce' transforme en NaN toute valeur non convertible plutot
# que de lever une exception. On compare les effectifs avant/apres pour
# verifier qu'aucune valeur legitime n'a ete perdue au passage.
def typer_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Typage numerique")
    for c in cfg.NUM_COLS:
        avant = df[c].notna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        perdu = avant - df[c].notna().sum()
        print(f"  {c:16s} valeurs numeriques {df[c].notna().sum():>8,} "
              f"| non convertibles {perdu:,}")
    return df


# ----------------------------------------------------------------------
# 5. VARIABLES BOOLEENNES
# ----------------------------------------------------------------------
# COMPLETED_FILE est encode 'YES' / absent. L'audit l'avait signale comme
# constante : c'est un faux positif, la modalite negative etant codee par
# l'absence. On restitue le booleen.
def typer_booleens(df: pd.DataFrame) -> pd.DataFrame:
    df["COMPLETED_FILE"] = df["COMPLETED_FILE"].eq("YES")
    print(f"\n[5] COMPLETED_FILE -> booleen "
          f"({df['COMPLETED_FILE'].sum():,} dossiers complets)")
    return df


# ----------------------------------------------------------------------
# 6. CONTROLES DE COHERENCE
# ----------------------------------------------------------------------
# Principe : on ne supprime jamais une ligne pour une date incoherente.
# On invalide uniquement la valeur fautive (mise a NaT) et on journalise.
# Supprimer la ligne ferait disparaitre un compte reel du perimetre et
# fausserait le taux de churn.
def controler_coherence(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[6] Controles de coherence")

    # 6.1 Dates anterieures a 1900 : impossibles pour une banque creee
    # au 20e siecle. L'audit avait releve une naissance en 1190.
    for c in cfg.DATE_COLS:
        m = df[c] < cfg.DATE_PLANCHER
        if m.any():
            tracer(f"{c} anterieure a 1900", m, "valeur -> NaT")
            df.loc[m, c] = pd.NaT

    # 6.2 Age hors bornes de plausibilite bancaire.
    age = (cfg.DATE_EXTRACTION - df["DATE_OF_BIRTH"]).dt.days / 365.25
    m = age.notna() & ((age < cfg.AGE_MIN) | (age > cfg.AGE_MAX))
    tracer(f"age hors [{cfg.AGE_MIN}, {cfg.AGE_MAX}] ans", m,
           "DATE_OF_BIRTH -> NaT")
    df.loc[m, "DATE_OF_BIRTH"] = pd.NaT

    # 6.3 Cloture anterieure a l'ouverture. L'analyse montre que ces cas
    # se concentrent sur des comptes techniques (exigibles, escompte,
    # depots a terme) fermes par lots administratifs, sur des numeros de
    # compte recycles. On ne corrige pas : on marque, pour pouvoir
    # isoler ces comptes en phase 3 et justifier leur traitement.
    m = df["ACCT_CLOSE_DATE"] < df["ACCT_OPENING_DATE"]
    df["FLAG_CLOTURE_AVANT_OUVERTURE"] = m
    tracer("cloture anterieure a l'ouverture du compte", m,
           "colonne FLAG_CLOTURE_AVANT_OUVERTURE")

    # 6.4 Compte ouvert avant l'entree en relation du client.
    m = df["ACCT_OPENING_DATE"] < df["CUST_OPENING_DATE"]
    df["FLAG_COMPTE_AVANT_CLIENT"] = m
    tracer("compte ouvert avant l'entree en relation", m,
           "colonne FLAG_COMPTE_AVANT_CLIENT")

    # 6.5 Echeance anterieure au demarrage du produit.
    m = df["MATURITYDATE"] < df["STARTDATE"]
    df["FLAG_ECHEANCE_INCOHERENTE"] = m
    tracer("echeance anterieure au demarrage du produit", m,
           "colonne FLAG_ECHEANCE_INCOHERENTE")

    # 6.6 Date posterieure a l'extraction : impossible.
    for c in ["ACCT_OPENING_DATE", "ACCT_CLOSE_DATE", "CUST_OPENING_DATE"]:
        m = df[c] > cfg.DATE_EXTRACTION
        if m.any():
            tracer(f"{c} posterieure a l'extraction", m, "valeur -> NaT")
            df.loc[m, c] = pd.NaT

    return df


# ----------------------------------------------------------------------
# 7. JOINTURE DES REFERENTIELS
# ----------------------------------------------------------------------
# On ajoute les libelles SANS remplacer les codes : le code reste la cle
# technique (stable, utilisable en modelisation), le libelle sert a
# l'analyse exploratoire et au dashboard. Jointure en 'left' pour ne
# jamais perdre de ligne, meme quand le code est absent du referentiel
# (cas d'ACCOUNT_CATEGORY, couvert a 58,8 % seulement).
def joindre_referentiels(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[7] Jointure des referentiels")

    paires = [
        (
            "ACCOUNT_CATEGORY",
            "CATEGORY",
            "CATEGORY DESCRIPTION",
            "LIB_CATEGORY"
        ),
        (
            "CURRENCY",
            "CURRENCY",
            "CCY_NAME",
            "LIB_CURRENCY"
        ),
        (
            "INDUSTRY",
            "INDUSTRY",
            "INDUSTRY DESCRIPTION",
            "LIB_INDUSTRY"
        ),
        (
            "LOB",
            "TARGET",
            "TARGET DESCRIPTION",
            "LIB_LOB"
        ),
        (
            "CLOSURE_REASON",
            "CLOSURE_REASON",
            "DESCRIPTION",
            "LIB_CLOSURE"
        ),
    ]

    for col_fait, nom_dim, col_lib, alias in paires:
        fichier, col_cle = cfg.DIMENSIONS[nom_dim]

        # Cherche le fichier dans data/raw ou data/reference
        chemin_dim = cfg.trouver(fichier)

        dim = pd.read_excel(
            chemin_dim,
            dtype=str
        )

        dim.columns = dim.columns.str.strip()
        dim[col_cle] = dim[col_cle].astype("string").str.strip()

        if nom_dim == "CLOSURE_REASON":
            dim[col_cle] = (
                dim[col_cle]
                .str.split("*", regex=False)
                .str[-1]
            )

        if col_lib not in dim.columns:
            raise KeyError(
                f"Colonne '{col_lib}' introuvable dans {chemin_dim.name}. "
                f"Colonnes disponibles: {dim.columns.tolist()}"
            )

        dim = (
            dim[[col_cle, col_lib]]
            .drop_duplicates(subset=[col_cle])
        )

        dim.columns = [
            col_fait,
            alias
        ]

        df = df.merge(
            dim,
            on=col_fait,
            how="left"
        )

        lignes_renseignees = df[col_fait].notna()

        if lignes_renseignees.any():
            taux = (
                df.loc[lignes_renseignees, alias]
                .notna()
                .mean()
                * 100
            )
        else:
            taux = 0.0

        print(
            f"  {col_fait:18s} -> {alias:14s} "
            f"libelle trouve pour {taux:5.1f} % "
            f"des lignes renseignees"
        )

    # Référentiel DAO
    fichier_dao, _ = cfg.DIMENSIONS["DAO"]
    chemin_dao = cfg.trouver(fichier_dao)

    dao = pd.read_excel(
        chemin_dao,
        dtype=str
    )

    dao.columns = dao.columns.str.strip()

    required_dao_columns = [
        "ACCOUNT_OFFICER",
        "AREA"
    ]

    missing_dao_columns = [
        col
        for col in required_dao_columns
        if col not in dao.columns
    ]

    if missing_dao_columns:
        raise KeyError(
            f"Colonnes manquantes dans {chemin_dao.name}: "
            f"{missing_dao_columns}"
        )

    dao["ACCOUNT_OFFICER"] = (
        dao["ACCOUNT_OFFICER"]
        .astype("string")
        .str.strip()
    )

    dao = (
        dao[
            [
                "ACCOUNT_OFFICER",
                "AREA"
            ]
        ]
        .drop_duplicates(subset=["ACCOUNT_OFFICER"])
    )

    dao.columns = [
        "BRANCH",
        "LIB_ZONE"
    ]

    df = df.merge(
        dao,
        on="BRANCH",
        how="left"
    )

    print(
        f"  {'BRANCH':18s} -> {'LIB_ZONE':14s} "
        f"libelle trouve pour "
        f"{df['LIB_ZONE'].notna().mean() * 100:5.1f} % des lignes"
    )

    return df
# ----------------------------------------------------------------------
# CHAINE D'EXECUTION
# ----------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(cfg.F_DATA, **cfg.CSV_READ_OPTS)
    print(f"Charge : {len(df):,} lignes x {df.shape[1]} colonnes")

    df = supprimer_doublons(df)
    df = normaliser_texte(df)
    df = typer_dates(df)
    df = typer_numeriques(df)
    df = typer_booleens(df)
    df = controler_coherence(df)
    df = joindre_referentiels(df)

    # Parquet plutot que CSV : conserve les types (une date reste une
    # date au rechargement), fichier 5 a 10 fois plus compact, lecture
    # bien plus rapide. Le CSV reperdrait tout le travail de typage.
    sortie = cfg.INTERIM / "staging.parquet"
    df.to_parquet(sortie, index=False)

    pd.DataFrame(JOURNAL).to_csv(
        cfg.REPORTS / "coherence_staging.csv", index=False)

    print(f"\nStaging ecrit : {sortie}")
    print(f"Dimensions finales : {len(df):,} lignes x {df.shape[1]} colonnes")
    print(f"Journal de coherence : "
          f"{cfg.REPORTS / 'coherence_staging.csv'}")


if __name__ == "__main__":
    main()
