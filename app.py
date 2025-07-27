# app.py ─ Observatoire des Prix (Côte d’Ivoire)
# =================================================
# Main Streamlit entry ‑ version refactor 2025‑07‑16

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Core helpers ────────────────────────────────────────────────────────────
from core.nomenclature import ncoa_fonction, repartition_ncoa, validate_ncoa
from design import inject_css

# ----------------------------------------------------------------------------
# ⚙️  Streamlit & design initialisation
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Observatoire des Prix",
    page_icon="🇨🇮",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()  # ← charge le thème clair par défaut

# ── Logging (facultatif en prod) ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.info("🔄 Application démarrée")

# ----------------------------------------------------------------------------
# 📥 Chargement de la base
# ----------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CSV_PATH = DATA_DIR / "Base.csv"

@st.cache_data(ttl=3600, show_spinner="📑 Lecture de la base …")
def load_data() -> pd.DataFrame:
    """Lit `Base_fin.csv`, coercition numérique, index datetime."""
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:  # gestion d'erreur explicite
        st.error(f"🚫 Fichier introuvable : {CSV_PATH}")
        st.stop()

    df["date"] = pd.to_datetime(
        df["annee_deb_couv"].astype(str) + "-" + df["mois_deb_couv"].astype(str).str.zfill(2) + "-01"
    )
    df = df[df["date"] <= "2024-10"].sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)

    # coercition numérique automatique sauf colonnes structurelles
    struct_cols = [
        "annee_deb_couv",
        "mois_deb_couv",
        "annee_fin_couv",
        "mois_fin_couv",
    ]
    num_cols = df.columns.difference(struct_cols)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 🗂️  Préparation session & constantes
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_data()

df: pd.DataFrame = st.session_state.df

# ── Chargement brut ─────────────────────────────────────────────
if "df_full" not in st.session_state:
    st.session_state.df_full = load_data()          # base complète 1963‑2024

df_full = st.session_state.df_full

# ── Sélecteur de plage temporelle ───────────────────────────────
st.sidebar.markdown("### 🗓️ Plage d’analyse")

min_date = df_full.index.min()
max_date = df_full.index.max()

start_default = pd.to_datetime("2017-01-01")        # valeur par défaut
end_default   = max_date

colA, colB = st.sidebar.columns(2)
with colA:
    date_start = st.date_input("De",  value=start_default,
                               min_value=min_date, max_value=max_date,
                               key="date_start")
with colB:
    date_end   = st.date_input("À",   value=end_default,
                               min_value=min_date, max_value=max_date,
                               key="date_end")

# Contrôle logique :
if date_start > date_end:
    st.sidebar.error("⚠️ La date de début dépasse la date de fin.")
    st.stop()

# ── Filtrage et mise en session ─────────────────────────────────
df_filtered = df_full.loc[str(date_start) : str(date_end)].copy()

# Stocke la version filtrée pour toutes les pages
st.session_state.df = df_filtered


def split_columns(_df: pd.DataFrame):
    """Retourne listes (postes, fonctions, globaux, structure)."""
    fonctions = list(ncoa_fonction.values())
    globaux = ["Inflation", "InflationGliss", "IHPC"]
    struct = ["annee_deb_couv", "mois_deb_couv", "annee_fin_couv", "mois_fin_couv"]
    postes = [c for c in _df.columns if c not in fonctions + globaux + struct]
    return postes, fonctions, globaux, struct


poste_cols, fonction_cols, glob_cols, struct_cols = split_columns(df)
st.session_state.poste_cols=poste_cols
st.session_state.fonction_cols=fonction_cols
st.session_state.glob_cols=glob_cols

# Validation NCOA à chaud
errors = validate_ncoa(poste_cols, verbose=False)
if errors:
    with st.sidebar.expander("⚠️ Incohérences NCOA"):
        for e in errors:
            st.write("•", e)

st.sidebar.success(f"📅 Base mise à jour : {df.index.max():%B %Y}")

# ---------------------------------------------------------------------------
# 🎛️  Sidebar : navigation
# ---------------------------------------------------------------------------
st.sidebar.header("Options de navigation")

with st.sidebar:
    st.header("🔎 Visualisation")
    page_visual = st.radio(
        "",  # pas de label visible
        [
            "Accueil",
            "Exploration",
            "Corrélations",
            "État Base",
            "Choix",
            "Prévisions",
            "Classification",
            "alertes",
        ],
        key="page_visual",
    )

    st.header("🛠️ Outils")
    page_tool = st.radio(
        "", ["Aucun", "Traitement", "Scraping ANSTAT"], key="page_tool"
    )

# Décision
page = page_tool if page_tool != "Aucun" else page_visual

# ---------------------------------------------------------------------------
# 📄 Router — import dynamique des pages
# ---------------------------------------------------------------------------
ROUTES: dict[str, str] = {
    "Accueil": "app_pages.acceuil:render_accueil",
    "Traitement": "app_pages.traitement:render_treat",
    "Exploration": "app_pages.exploration:render_exploration",
    "Corrélations": "app_pages.correlation:render_correlation",
    "Scraping ANSTAT": "app_pages.scraping:render_scraping",
    "Prévisions": "app_pages.previsions:render_previsions",
    "Choix": "app_pages.choix:render_choix",
    "Classification": "app_pages.classification:render_classif",
    "État Base": "app_pages.base:render_base",
    "alertes": "app_pages.alertes:render_alertes",
}

if page not in ROUTES:
    st.error(f"🚫 Route inconnue : {page}")
    st.stop()

module_name, func_name = ROUTES[page].split(":")
module = importlib.import_module(module_name)
getattr(module, func_name)()  # exécute la fonction

# ---------------------------------------------------------------------------
# 🌙  Dark‑mode toggle
# ---------------------------------------------------------------------------
dark = st.sidebar.toggle("🌙 Mode nuit")

if dark:
    st.write(
        """
        <style>
        :root{
            --bg:#1F2329;  --text:#F6F6F5;
            --primary:#F28C28; --accent:#30c48d;
            --card-bg:#2A2F36; --card-shadow:rgba(0,0,0,0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 📑  Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("**Source :** dossier `data/` & scraping ANSTAT")
st.sidebar.write("**Stack :** Streamlit · Pandas · Statsmodels · Prophet · Selenium")
st.sidebar.write("**Auteurs :** \n TCHIMTCHOUA Nono Mylène \n GANAME Abdoulaye Idrissa")
