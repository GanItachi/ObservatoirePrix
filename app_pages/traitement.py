# app_pages/traitement.py – Pipeline complet nettoyage / chaînage
# --------------------------------------------------------------

import os
import time
import pandas as pd
import streamlit as st

from core.preprocessing import chainage_base, build_missing_functions
from core.nomenclature import repartition_ncoa, ncoa_fonction, validate_ncoa

# Dossier de sauvegarde des backups
BACKUP_DIR = "data/backups"
os.makedirs(BACKUP_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------------
def backup_df(df: pd.DataFrame, tag: str):
    """Sauvegarde la DataFrame en CSV avec timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"{tag}_{timestamp}.csv")
    df.to_csv(backup_path, index=True)

def impute_missing(df: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """Imputation des valeurs manquantes selon la méthode choisie."""
    methods = {
        "100": lambda df: df.fillna(100),
        "mean": lambda df: df.fillna(df.mean()),
        "median": lambda df: df.fillna(df.median()),
        "value": lambda df: df.fillna(kwargs.get("value", 100)),
        "ffill": lambda df: df.ffill(),
        "interpolate": lambda df: df.interpolate(method=kwargs.get("kind", "linear")),
        "rolling": lambda df: df.apply(lambda s: s.fillna(s.rolling(kwargs.get("window", 3), 1).mean()))
    }
    return methods[method](df)

# -------------------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------------------
def render_treat():
    st.header("🛠️ Traitement des données & Chaînage")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Aucune base chargée.")
        return

    df = st.session_state.df.copy()

    # Sélection du mode de traitement
    mode = st.radio("Choisir le mode de traitement :", ["Automatique", "Manuel"])

    # =====================================================================
    # MODE AUTOMATIQUE
    # =====================================================================
    if mode == "Automatique":
        st.success("Mode automatique sélectionné : NaN → 100, puis chaînage au pivot 2019‑01‑01.")

        if st.button("🚀 Exécuter le traitement automatique"):
            df_auto = chainage_base(df.fillna(100), "2019-01-01")
            backup_df(df_auto, "auto")
            st.session_state.df = df_auto
            st.success("✅ Traitement automatique terminé et enregistré.")
            st.dataframe(df_auto.head())
            st.rerun()
        return

    # =====================================================================
    # MODE MANUEL
    # =====================================================================
    st.subheader("🔧 Paramètres personnalisés")

    # 1. Imputation des valeurs manquantes
    st.markdown("**1. Méthode d'imputation des valeurs manquantes**")
    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Méthode d'imputation",
            ["100", "mean", "median", "value", "ffill", "interpolate", "rolling"]
        )

    with col2:
        params = {}
        if method == "value":
            params["value"] = st.number_input("Valeur d'imputation personnalisée", value=100.0)
        elif method == "rolling":
            params["window"] = st.slider("Fenêtre glissante", min_value=2, max_value=12, value=3)
        elif method == "interpolate":
            params["kind"] = st.selectbox("Type d'interpolation", ["linear", "spline", "quadratic"])

    # 2. Chaînage des indices
    st.markdown("**2. Chaînage de base des indices**")
    pivot_date = st.date_input("Sélectionner la date pivot du chaînage", value=pd.to_datetime("2019-01-01"))

    # 3. Complétion des fonctions manquantes (optionnelle)
    st.markdown("**3. Complétion des fonctions COICOP (facultatif)**")
    add_functions = st.checkbox("Créer les fonctions NCOA manquantes")
    agg_method = st.selectbox("Méthode d'agrégation", ["mean", "sum", "median"], disabled=not add_functions)

    weights = None
    if add_functions:
        weight_file = st.file_uploader(
            "Pondérations (fichier CSV avec colonnes : poste, poids)",
            type="csv",
            help="Assurez-vous que le CSV contienne exactement deux colonnes : 'poste' et 'poids'."
        )
        if weight_file:
            weights_df = pd.read_csv(weight_file)
            weights = weights_df.set_index("poste")["poids"].to_dict()

    # Bouton pour lancer l'ensemble du traitement
    if st.button("✅ Appliquer le traitement personnalisé"):

        # Imputation
        df_new = impute_missing(df, method, **params)

        # Chaînage
        df_new = chainage_base(df_new, pivot_date)

        # Complétion fonctions NCOA
        if add_functions:
            df_new = build_missing_functions(
                df_new,
                repartition_ncoa=repartition_ncoa,
                ncoa_fonction=ncoa_fonction,
                method=agg_method,
                weights=weights
            )

        # Validation finale des colonnes NCOA
        erreurs_ncoa = validate_ncoa(df_new.columns)
        if erreurs_ncoa:
            st.warning("⚠️ Des incohérences NCOA sont présentes après traitement :")
            for err in erreurs_ncoa:
                st.write(f"- {err}")

        # Sauvegarde en backup
        backup_df(df_new, "manual")

        # Mise à jour complète de session_state
        glob_cols = ["Inflation", "InflationGliss", "IHPC"]
        struct_cols = ["annee_deb_couv", "mois_deb_couv", "annee_fin_couv", "mois_fin_couv"]

        fonction_cols = [f for f in ncoa_fonction.values() if f in df_new.columns]
        poste_cols = [
            c for c in df_new.columns
            if c not in fonction_cols + glob_cols + struct_cols
        ]

        # Actualisation complète dans session_state
        # Mise à jour complète de session_state
        st.session_state.update({
            "df": df_new,
            "df_full": df_new,  # <-- AJOUT : met à jour la base complète !
            "fonction_cols": fonction_cols,
            "poste_cols": poste_cols,
            "glob_cols": [g for g in glob_cols if g in df_new.columns],
        })

        # Confirmation visuelle immédiate
        st.success("✅ Traitement terminé. Base mise à jour dans la session.")
        st.dataframe(df_new.head())
        st.rerun()
        

# -------------------------------------------------------------------------
# Point d'entrée
# -------------------------------------------------------------------------
if __name__ == "__main__":
    if "df" not in st.session_state:
        st.error("❌ DataFrame non chargé : veuillez d'abord importer les données.")
    else:
        render_treat()
