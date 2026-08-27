"""
03_cible.py — Phase 3 : fenetre d'observation et variable cible.

C'est la phase qui determine la validite de tout le projet. Une cible
mal definie produit un modele qui obtient d'excellents scores en
apprenant autre chose que le churn.

Trois decisions y sont prises, chacune justifiee par les donnees :

  1. FENETRE D'OBSERVATION
     On se place a une date de reference T (31/12/2024). Les variables
     explicatives ne devront utiliser que l'information connue avant T.
     Le depart est observe sur les 12 mois suivants (annee 2025).
     Cette fenetre est entierement contenue dans l'extraction, qui
     s'arrete au 19/02/2026 : elle n'est donc pas tronquee.

  2. DEFINITION DU DEPART
     L'audit de phase 1 a etabli que le statut est homogene par client :
     un client a soit tous ses comptes ouverts, soit tous fermes. Le
     depart d'un client correspond donc a la fermeture de son dernier
     compte, datee par max(ACCT_CLOSE_DATE).

  3. EXCLUSION DES CAMPAGNES ADMINISTRATIVES
     La distribution des fermetures montre 7 journees ou plus de 3 000
     comptes sont fermes simultanement, alors que le volume median est
     de 36 par jour. Ces 7 journees representent 54 % de toutes les
     fermetures. Il ne s'agit pas de departs individuels mais d'apurements
     de comptes dormants. Les inclure reviendrait a apprendre a predire
     les campagnes de la banque, pas le comportement des clients.

Entree  : data/interim/staging.parquet
Sorties : data/processed/cible_client.parquet
          reports/audit/construction_cible.csv
Execution : python src/03_cible.py
"""

import pandas as pd

import config as cfg

RAPPORT = []


def noter(cle: str, valeur) -> None:
    RAPPORT.append({"indicateur": cle, "valeur": valeur})
    print(f"  {cle:52s} {valeur}")


# ----------------------------------------------------------------------
# 1. PASSAGE AU GRAIN COMPTE
# ----------------------------------------------------------------------
# Le staging est au grain (client, compte, produit). Pour raisonner sur
# les fermetures il faut d'abord un compte = une ligne, sinon un compte
# portant 5 produits pese 5 fois dans les comptages.
def au_grain_compte(df: pd.DataFrame) -> pd.DataFrame:
    comptes = (
        df.dropna(subset=[cfg.KEY_ACCOUNT])
        .sort_values([cfg.KEY_CUSTOMER, cfg.KEY_ACCOUNT])
        .drop_duplicates(subset=[cfg.KEY_ACCOUNT], keep="first")
    )
    print(f"\n[1] Grain compte : {len(comptes):,} comptes "
          f"({len(df):,} lignes en entree)")
    return comptes


# ----------------------------------------------------------------------
# 2. DETECTION DES CAMPAGNES ADMINISTRATIVES
# ----------------------------------------------------------------------
# Regle pilotee par les donnees, pas choisie a la main : on compte les
# fermetures par journee et on isole celles qui depassent SEUIL_PURGE.
# Le seuil est declare dans config.py pour etre citable dans le memoire.
def detecter_purges(comptes: pd.DataFrame) -> set:
    print("\n[2] Detection des campagnes de fermeture")
    volume = comptes["ACCT_CLOSE_DATE"].value_counts()
    purges = volume[volume > cfg.SEUIL_PURGE]

    noter("volume median de fermetures par jour actif",
          f"{volume.median():.0f}")
    noter("journees de campagne detectees", len(purges))
    noter("comptes fermes lors de ces campagnes",
          f"{purges.sum():,} ({purges.sum() / volume.sum() * 100:.1f} %)")
    for jour, n in purges.sort_index().items():
        print(f"      {jour:%Y-%m-%d} : {n:,} comptes")
    return set(purges.index)


# ----------------------------------------------------------------------
# 3. AGREGATION AU GRAIN CLIENT
# ----------------------------------------------------------------------
def au_grain_client(comptes: pd.DataFrame, jours_purge: set) -> pd.DataFrame:
    print("\n[3] Agregation au grain client")
    cli = comptes.groupby(cfg.KEY_CUSTOMER).agg(
        NB_COMPTES=("ACCOUNT_NO", "nunique"),
        DATE_DERNIERE_CLOTURE=("ACCT_CLOSE_DATE", "max"),
        DATE_PREMIERE_OUVERTURE=("ACCT_OPENING_DATE", "min"),
        NATURE_CLIENT=("NATURE_CLIENT", "first"),
        PARTYCLASS=("PARTYCLASS", "first"),
    )
    # Marque les clients dont au moins un compte a ete ferme lors d'une
    # campagne : leur situation n'est pas interpretable comme un choix.
    touches = (
        comptes[comptes["ACCT_CLOSE_DATE"].isin(jours_purge)]
        [cfg.KEY_CUSTOMER].unique()
    )
    cli["TOUCHE_PAR_CAMPAGNE"] = cli.index.isin(touches)
    print(f"  {len(cli):,} clients porteurs d'au moins un compte")
    return cli


# ----------------------------------------------------------------------
# 4. POPULATION ELIGIBLE
# ----------------------------------------------------------------------
# On ne peut predire le depart que de clients presents a la date de
# reference. Deux exclusions, toutes deux necessaires :
#   - les clients deja partis avant T : leur depart appartient au passe,
#     les inclure reviendrait a predire un evenement deja survenu ;
#   - les clients sans aucun compte : aucune cible calculable.
def population_eligible(cli: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[4] Population eligible au {cfg.DATE_REFERENCE:%d/%m/%Y}")
    present = (
        cli["DATE_DERNIERE_CLOTURE"].isna()
        | (cli["DATE_DERNIERE_CLOTURE"] > cfg.DATE_REFERENCE)
    )
    noter("clients deja partis avant la date de reference",
          f"{(~present).sum():,}")
    pop = cli[present].copy()
    noter("population retenue", f"{len(pop):,}")
    return pop


# ----------------------------------------------------------------------
# 5. CONSTRUCTION DE LA CIBLE
# ----------------------------------------------------------------------
def construire_cible(pop: pd.DataFrame) -> pd.DataFrame:
    fin = cfg.DATE_REFERENCE + pd.DateOffset(months=cfg.HORIZON_MOIS)
    print(f"\n[5] Cible : depart observe entre "
          f"{cfg.DATE_REFERENCE:%d/%m/%Y} et {fin:%d/%m/%Y}")

    pop["CHURN"] = (
        pop["DATE_DERNIERE_CLOTURE"].notna()
        & (pop["DATE_DERNIERE_CLOTURE"] > cfg.DATE_REFERENCE)
        & (pop["DATE_DERNIERE_CLOTURE"] <= fin)
    ).astype(int)

    # Un client conserve mais dont un compte a ete ferme par campagne
    # n'est ni clairement fidele ni clairement partant : son etiquette
    # serait arbitraire. On l'ecarte plutot que de lui imposer un 0.
    ambigus = pop["TOUCHE_PAR_CAMPAGNE"] & (pop["CHURN"] == 0)
    noter("clients au label ambigu ecartes", f"{ambigus.sum():,}")
    pop = pop[~ambigus].copy()

    noter("population finale", f"{len(pop):,}")
    noter("churners", f"{pop['CHURN'].sum():,}")
    noter("taux de churn", f"{pop['CHURN'].mean() * 100:.2f} %")

    # Anciennete a la date de reference : premiere variable explicative
    # legitime, calculee uniquement a partir du passe.
    pop["ANCIENNETE_JOURS"] = (
        cfg.DATE_REFERENCE - pop["DATE_PREMIERE_OUVERTURE"]
    ).dt.days
    return pop


# ----------------------------------------------------------------------
# 5 bis. RESTRICTION AUX CLIENTS DETENTEURS D'UN COMPTE A LA DATE T
# ----------------------------------------------------------------------
# Un client dont tous les comptes ont ete ouverts APRES la date de
# reference n'etait pas client a cette date : on ne peut pas predire son
# depart a partir d'une situation qui n'existait pas encore.
#
# Attention au piege : la comparaison ACCT_OPENING_DATE <= T renvoie
# False pour une date manquante (NaT), ce qui exclurait silencieusement
# des comptes reels. On distingue donc explicitement les deux cas :
#   - date renseignee et posterieure a T  -> compte inexistant a T
#   - date manquante                      -> compte conserve, signale
def restreindre_aux_detenteurs(pop: pd.DataFrame,
                               comptes: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[5 bis] Clients detenteurs d'un compte au "
          f"{cfg.DATE_REFERENCE:%d/%m/%Y}")
    c = comptes[comptes[cfg.KEY_CUSTOMER].isin(pop.index)]

    existant_a_T = (
        c["ACCT_OPENING_DATE"].isna()
        | (c["ACCT_OPENING_DATE"] <= cfg.DATE_REFERENCE)
    )
    detenteurs = set(c.loc[existant_a_T, cfg.KEY_CUSTOMER])
    retires = set(pop.index) - detenteurs

    noter("comptes sans date d'ouverture (conserves)",
          f"{c['ACCT_OPENING_DATE'].isna().sum():,}")
    noter("clients retires (tous comptes ouverts apres T)",
          f"{len(retires):,}")
    noter("dont churners retires",
          f"{pop.loc[list(retires), 'CHURN'].sum():,}")

    pop = pop.loc[list(detenteurs)].copy()

    # Signale les clients dont au moins un compte n'a pas de date
    # d'ouverture : information utile au modele, sans imputation.
    sans_date = set(c.loc[c["ACCT_OPENING_DATE"].isna(), cfg.KEY_CUSTOMER])
    pop["FLAG_DATE_OUVERTURE_MANQUANTE"] = pop.index.isin(sans_date)

    noter("population finale", f"{len(pop):,}")
    noter("churners", f"{pop['CHURN'].sum():,}")
    noter("taux de churn", f"{pop['CHURN'].mean() * 100:.2f} %")
    return pop


# ----------------------------------------------------------------------
# 6. CONTROLES DE VALIDITE
# ----------------------------------------------------------------------
# Un jury demandera comment vous savez que la cible tient. Ces trois
# controles y repondent, et leur sortie va dans le memoire.
def controler(pop: pd.DataFrame) -> None:
    print("\n[6] Controles de validite")

    # 6.1 Aucun churner ne doit etre parti avant la date de reference.
    a = ((pop["CHURN"] == 1)
         & (pop["DATE_DERNIERE_CLOTURE"] <= cfg.DATE_REFERENCE)).sum()
    noter("churners partis avant T (doit etre 0)", a)
    assert a == 0

    # 6.2 Taux dans une plage plausible pour une banque de detail.
    taux = pop["CHURN"].mean() * 100
    noter("taux dans la plage bancaire usuelle [2 %, 20 %]",
          "oui" if 2 <= taux <= 20 else "NON — a justifier")

    # 6.3 Le taux doit varier selon le type de client : une cible qui
    # ne discrimine aucun segment est generalement un artefact.
    parseg = pop.groupby("NATURE_CLIENT")["CHURN"].agg(["size", "mean"])
    parseg = parseg[parseg["size"] > 1000]
    print("\n  Taux de churn par type de client :")
    for nat, r in parseg.iterrows():
        print(f"      {nat:6s} {r['size']:>8,.0f} clients   "
              f"{r['mean'] * 100:5.2f} %")


def main() -> None:
    df = pd.read_parquet(cfg.INTERIM / "staging.parquet")
    comptes = au_grain_compte(df)
    jours_purge = detecter_purges(comptes)
    cli = au_grain_client(comptes, jours_purge)
    pop = population_eligible(cli)
    pop = construire_cible(pop)
    pop = restreindre_aux_detenteurs(pop, comptes)
    controler(pop)

    sortie = cfg.PROCESSED / "cible_client.parquet"
    pop.reset_index().to_parquet(sortie, index=False)
    pd.DataFrame(RAPPORT).to_csv(
        cfg.REPORTS / "construction_cible.csv", index=False)
    print(f"\nCible ecrite : {sortie}")


if __name__ == "__main__":
    main()
