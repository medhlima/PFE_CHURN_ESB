"""
04_features.py — Phase 4 : table client, ancree a la date de reference.

Regle unique qui gouverne tout ce fichier :
    aucune variable ne peut utiliser une information posterieure au
    31/12/2024 (cfg.DATE_REFERENCE).

Concretement, trois filtres sont appliques avant toute agregation :
  1. les comptes ouverts APRES T sont retires : ils n'existaient pas ;
  2. les comptes fermes APRES T sont consideres comme OUVERTS a T :
     leur fermeture appartient a la fenetre de prediction, c'est la
     cible elle-meme ;
  3. les dates de revue KYC posterieures a T sont mises a NaT : elles
     n'etaient pas connues a la date de reference.

Les comptes sans date d'ouverture sont CONSERVES. La comparaison
`NaT <= T` renvoie False, ce qui les exclurait silencieusement : un
client ne doit pas disparaitre a cause d'une date manquante.

Entrees : data/interim/staging.parquet
          data/processed/cible_client.parquet
Sortie  : data/processed/dataset_modelisation.parquet
Execution : python src/04_features.py
"""

import numpy as np
import pandas as pd

import config as cfg

T = cfg.DATE_REFERENCE

# Regroupement des micro-modalites : une modalite portee par quelques
# clients ne permet aucun taux fiable et desequilibre l'encodage.
NATURE_CLIENT_MAP = {"PPH": "PPH", "PM": "PM", "PRO": "PRO"}


def au_grain_compte_a_T(df: pd.DataFrame, clients: set) -> pd.DataFrame:
    """Comptes existant a la date de reference, un compte par ligne."""
    c = (
        df.dropna(subset=[cfg.KEY_ACCOUNT])
        .drop_duplicates(subset=[cfg.KEY_ACCOUNT])
    )
    c = c[c[cfg.KEY_CUSTOMER].isin(clients)]
    avant = len(c)

    # Filtre 1 : le compte existait-il a T ?
    existe = c["ACCT_OPENING_DATE"].isna() | (c["ACCT_OPENING_DATE"] <= T)
    c = c[existe].copy()

    # Filtre 2 : etat du compte a T, et non a la date d'extraction.
    c["OUVERT_A_T"] = c["ACCT_CLOSE_DATE"].isna() | (c["ACCT_CLOSE_DATE"] > T)

    print(f"[1] Comptes : {avant:,} -> {len(c):,} existant au "
          f"{T:%d/%m/%Y} ({avant - len(c):,} ouverts apres T, retires)")
    print(f"    dont ouverts a T : {c['OUVERT_A_T'].sum():,} "
          f"| fermes avant T : {(~c['OUVERT_A_T']).sum():,}")
    return c


def profil_client(df: pd.DataFrame, clients: set) -> pd.DataFrame:
    """
    Attributs proprement clients.

    On ne prend ici que des colonnes constantes au sein d'un client.
    Prendre la premiere ligne pour une colonne de compte reviendrait a
    choisir un compte au hasard.
    """
    cols = ["NATIONALITY", "RESIDENCE", "MARITAL_STATUS", "DATE_OF_BIRTH",
            "CUST_OPENING_DATE", "NATURE_CLIENT", "PARTYCLASS", "SCORE_KYC",
            "SALARY", "COMPLETED_FILE", "LAST_REVIEW_DATE",
            "NEXT__REVIEW_DATE"]
    p = (
        df[df[cfg.KEY_CUSTOMER].isin(clients)]
        .sort_values(cfg.KEY_CUSTOMER)
        .groupby(cfg.KEY_CUSTOMER)[cols]
        .first()
    )
    print(f"[2] Profil client : {len(p):,} lignes")
    return p


def variables_demographiques(p: pd.DataFrame) -> pd.DataFrame:
    """Age, anciennete et indicateurs de profil, calcules a T."""
    out = pd.DataFrame(index=p.index)

    out["AGE"] = ((T - p["DATE_OF_BIRTH"]).dt.days / 365.25).round(1)
    out["TRANCHE_AGE"] = pd.cut(
    out["AGE"],
    bins=[15, 25, 35, 50, 65, 100],
    labels=["15-25", "26-35", "36-50", "51-65", "65+"],
    include_lowest=True
    )

    out["ANCIENNETE_CLIENT_ANNEES"] = (
        (T - p["CUST_OPENING_DATE"]).dt.days / 365.25).round(1)

    # Le salaire n'est pas impute : l'absence est elle-meme informative
    # (100 % des personnes morales n'en ont pas). L'imputation aura lieu
    # dans le pipeline de modelisation, apres separation train/test.
    out["SALAIRE"] = p["SALARY"]
    out["HAS_SALAIRE"] = p["SALARY"].notna()

    out["EST_TUNISIEN"] = p["NATIONALITY"].eq("TN")
    out["EST_RESIDENT"] = p["RESIDENCE"].eq("TN")
    out["SITUATION_FAMILIALE"] = p["MARITAL_STATUS"]
    out["TYPE_CLIENT"] = p["NATURE_CLIENT"].map(NATURE_CLIENT_MAP).fillna("AUTRE")
    out["SEGMENT"] = p["PARTYCLASS"]
    out["DOSSIER_COMPLET"] = p["COMPLETED_FILE"]
    print(f"[3] Demographie : {out.shape[1]} variables")
    return out


def variables_conformite(p: pd.DataFrame) -> pd.DataFrame:
    """
    KYC et revues reglementaires, connues a T uniquement.

    Une revue datee apres T n'avait pas encore eu lieu : la conserver
    reviendrait a utiliser le futur. On la neutralise et on le signale.
    """
    out = pd.DataFrame(index=p.index)

    derniere = p["LAST_REVIEW_DATE"].where(p["LAST_REVIEW_DATE"] <= T)
    prochaine = p["NEXT__REVIEW_DATE"].where(p["LAST_REVIEW_DATE"] <= T)
    neutralisees = (p["LAST_REVIEW_DATE"] > T).sum()

    out["SCORE_KYC"] = p["SCORE_KYC"]
    out["KYC_RISQUE_ELEVE"] = p["SCORE_KYC"].isin(["H1", "H2", "H3"])
    out["JOURS_DEPUIS_REVUE"] = (T - derniere).dt.days
    out["JOURS_AVANT_PROCHAINE_REVUE"] = (prochaine - T).dt.days
    out["REVUE_EN_RETARD"] = out["JOURS_AVANT_PROCHAINE_REVUE"] < 0
    out["A_HISTORIQUE_REVUE"] = derniere.notna()

    print(f"[4] Conformite : {neutralisees:,} revues posterieures a T "
          f"neutralisees")
    return out


def variables_comptes(c: pd.DataFrame) -> pd.DataFrame:
    """
    Portefeuille de comptes a T.

    ACTIVE_ACCOUNTS, CLOSED_ACCOUNTS et ACTIVE_ACCOUNT_RATIO ne sont
    volontairement pas produits : mesures a T ils sont quasi constants
    (ratio egal a 1 pour 99,6 % des clients) et correles a 0,99 avec
    NB_COMPTES. Le statut des comptes etant homogene par client, une
    attrition partielle anterieure n'est observable que sur 0,37 % de la
    population, essentiellement des numeros de compte recycles.
    """
    g = c.groupby(cfg.KEY_CUSTOMER)
    out = pd.DataFrame(index=g.size().index)

    out["NB_COMPTES"] = g[cfg.KEY_ACCOUNT].nunique()
    out["NB_CATEGORIES_COMPTE"] = g["ACCOUNT_CATEGORY"].nunique()
    out["NB_DEVISES"] = g["CURRENCY"].nunique()
    out["NB_AGENCES"] = g["BRANCH"].nunique()
    out["MULTI_DEVISE"] = out["NB_DEVISES"] > 1

    # Soldes : la valeur manquante est conservee. Remplacer par 0
    # confondrait "solde nul" et "solde non renseigne", deux faits
    # differents, et creerait une masse artificielle a zero.
    out["SOLDE_TOTAL"] = g["ACCT_BALANCE"].sum(min_count=1)
    out["SOLDE_MOYEN"] = g["ACCT_BALANCE"].mean()
    out["SOLDE_MIN"] = g["ACCT_BALANCE"].min()
    out["SOLDE_MAX"] = g["ACCT_BALANCE"].max()
    out["HAS_SOLDE"] = g["ACCT_BALANCE"].count() > 0
    out["SOLDE_NEGATIF"] = out["SOLDE_MIN"] < 0

    out["ANCIENNETE_COMPTE_JOURS"] = (T - g["ACCT_OPENING_DATE"].min()).dt.days

    # Modalite dominante par client. Un `groupby.agg(lambda: mode())`
    # serait exact mais impraticable sur 195 000 groupes : on trie par
    # frequence puis on garde la premiere ligne de chaque client.
    def dominante(col: str, nom: str) -> pd.Series:
        v = c[[cfg.KEY_CUSTOMER, col]].dropna()
        freq = v.groupby([cfg.KEY_CUSTOMER, col]).size().reset_index(name="n")
        freq = freq.sort_values([cfg.KEY_CUSTOMER, "n", col],
                                ascending=[True, False, True])
        return (freq.drop_duplicates(cfg.KEY_CUSTOMER)
                    .set_index(cfg.KEY_CUSTOMER)[col].rename(nom))

    out = out.join(dominante("BRANCH", "AGENCE_PRINCIPALE"))
    out = out.join(dominante("LIB_ZONE", "ZONE"))

    print(f"[5] Comptes : {out.shape[1]} variables sur {len(out):,} clients")
    return out


def variables_produits(df: pd.DataFrame, comptes_a_T: pd.DataFrame,
                       clients: set) -> pd.DataFrame:
    """
    Detention de produits a T.

    On repart des lignes produit, restreintes aux comptes existant a T,
    et on ecarte les produits demarres apres T.
    """
    ids = set(comptes_a_T[cfg.KEY_ACCOUNT])
    p = df[df[cfg.KEY_ACCOUNT].isin(ids) & df[cfg.KEY_CUSTOMER].isin(clients)]
    avant = len(p)
    p = p[p["STARTDATE"].isna() | (p["STARTDATE"] <= T)]
    print(f"[6] Produits : {avant:,} -> {len(p):,} lignes "
          f"(produits demarres apres T retires)")

    g = p.groupby(cfg.KEY_CUSTOMER)
    out = pd.DataFrame(index=g.size().index)
    out["NB_PRODUITS"] = g["PRODUCT"].nunique()
    out["NB_LIGNES_PRODUIT"] = g["PRODUCT_LINE"].nunique()

    # PRODUCT_LINE est un code structure : plus fiable qu'une recherche
    # de mots-cles dans des libelles concatenes. On construit les
    # indicateurs par colonne booleenne puis un simple max par client,
    # bien plus rapide qu'un lambda par groupe.
    lignes = {"CREDIT": "LENDING", "DEPOT": "DEPOSITS",
              "COMPTE_COURANT": "ACCOUNTS", "COFFRE": "SAFE.DEPOSIT.BOX"}
    tmp = p[[cfg.KEY_CUSTOMER]].copy()
    for nom, ligne in lignes.items():
        tmp[f"HAS_{nom}"] = p["PRODUCT_LINE"].eq(ligne)
    out = out.join(tmp.groupby(cfg.KEY_CUSTOMER).max())

    return out


def assembler(cible: pd.DataFrame, *blocs: pd.DataFrame) -> pd.DataFrame:
    """Assemble les blocs sur l'index client, sans perdre de client."""
    out = cible.copy()
    for b in blocs:
        out = out.join(b, how="left")
    return out


def garde_fou_fuite(df: pd.DataFrame) -> None:
    """
    Verifie qu'aucune colonne ayant servi a construire la cible ne
    subsiste. Un echec ici doit interrompre la chaine.
    """
    presentes = [c for c in cfg.LEAKAGE_COLS + ["HAS_ACTIVE_ACCOUNT"]
                 if c in df.columns]
    if presentes:
        raise AssertionError(f"Colonnes de fuite presentes : {presentes}")

    # Aucune variable ne doit etre determinee a plus de 99 % par la cible
    # via une regle triviale. Controle de forme, pas de performance.
    for c in df.select_dtypes(include=[np.number]).columns:
        if c == "CHURN":
            continue
        s = df[[c, "CHURN"]].dropna()
        if len(s) < 1000:
            continue
        if s[c].nunique() > 1:
            corr = abs(s[c].corr(s["CHURN"]))
            if corr > 0.30:
                raise AssertionError(
                    f"Correlation suspecte {c} / CHURN : {corr:.3f}")
    print("[8] Garde-fou anti-fuite : aucune colonne interdite, "
          "aucune correlation superieure a 0,30")


def main() -> None:
    df = pd.read_parquet(cfg.INTERIM / "staging.parquet")
    cible = pd.read_parquet(cfg.PROCESSED / "cible_client.parquet")
    cible = cible.set_index(cfg.KEY_CUSTOMER)
    clients = set(cible.index)
    print(f"Population cible : {len(clients):,} clients, "
          f"{cible['CHURN'].sum():,} churners "
          f"({cible['CHURN'].mean() * 100:.2f} %)\n")

    comptes = au_grain_compte_a_T(df, clients)
    profil = profil_client(df, clients)

    dataset = assembler(
        cible[["CHURN", "ANCIENNETE_JOURS", "FLAG_DATE_OUVERTURE_MANQUANTE"]],
        variables_demographiques(profil),
        variables_conformite(profil),
        variables_comptes(comptes),
        variables_produits(df, comptes, clients),
    )

    # Les comptages sont a zero quand le client n'a aucune ligne : c'est
    # un fait, pas une valeur manquante. Les montants, eux, restent NaN.
    for c in ["NB_PRODUITS", "NB_LIGNES_PRODUIT"]:
        dataset[c] = dataset[c].fillna(0).astype("int64")
    for c in dataset.columns:
        if c.startswith("HAS_") or c in ("MULTI_DEVISE", "SOLDE_NEGATIF"):
            dataset[c] = dataset[c].fillna(False).astype(bool)

    print(f"\n[7] Dataset assemble : {len(dataset):,} lignes x "
          f"{dataset.shape[1]} colonnes")

    # ------------------------------------------------------------------
    # RETRAIT DES VARIABLES CONTAMINEES PAR LA COMPLETUDE DES ATTRIBUTS
    # ------------------------------------------------------------------
    # L'extraction est un instantane pris au 19/02/2026, pas un etat
    # historique au 31/12/2024. Les attributs d'un compte (solde, devise,
    # categorie, date d'ouverture, ligne produit) ne sont renseignes que
    # si le compte figure encore dans la table de details : 0,0 % de
    # manquants sur les comptes actifs, contre 44 % sur les comptes
    # fermes en 2025 et 74 a 79 % au-dela.
    #
    # La PRESENCE d'un attribut est donc un indicateur de la fermeture
    # future. Ce n'est pas une colonne interdite mais un motif de
    # valeurs manquantes : aucun filtrage par date ne le corrige, car le
    # motif est date de l'extraction et non de la periode observee.
    #
    # Toute variable derivee de ces attributs est donc retiree. Seules
    # subsistent les variables issues du referentiel client, complet
    # quel que soit l'etat des comptes, et le nombre de comptes, qui
    # repose sur ACCOUNT_NO, toujours renseigne.
    CONTAMINEES = [
        "SOLDE_TOTAL", "SOLDE_MOYEN", "SOLDE_MIN", "SOLDE_MAX",
        "HAS_SOLDE", "SOLDE_NEGATIF", "NB_DEVISES", "MULTI_DEVISE",
        "NB_CATEGORIES_COMPTE", "NB_AGENCES", "AGENCE_PRINCIPALE", "ZONE",
        "ANCIENNETE_COMPTE_JOURS", "ANCIENNETE_JOURS",
        "FLAG_DATE_OUVERTURE_MANQUANTE",
        "NB_PRODUITS", "NB_LIGNES_PRODUIT",
        "HAS_CREDIT", "HAS_DEPOT", "HAS_COMPTE_COURANT", "HAS_COFFRE",
    ]
    retirees = [c for c in CONTAMINEES if c in dataset.columns]
    dataset = dataset.drop(columns=retirees)
    print(f"    {len(retirees)} variables retirees (completude des "
          f"attributs de compte correlee a la fermeture future)")
    print(f"    dataset final : {dataset.shape[1]} colonnes")

    garde_fou_fuite(dataset)

    sortie = cfg.PROCESSED / "dataset_modelisation.parquet"
    dataset.reset_index().to_parquet(sortie, index=False)
    print(f"\nDataset ecrit : {sortie}")
    print(f"Taux de churn : {dataset['CHURN'].mean() * 100:.2f} %")


if __name__ == "__main__":
    main()
