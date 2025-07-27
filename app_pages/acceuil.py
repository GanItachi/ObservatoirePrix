# app_pages/acceuil.py  – Accueil de l’Observatoire
# -------------------------------------------------
# Affiche les métriques clés, variations récentes et aperçus graphiques.

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from design import card                              # 🔸 Cartes stylées
from core.nomenclature import ncoa_fonction, repartition_ncoa



# ──────────────────────────────────────────────────────────────────────────────
def render_accueil() -> None:
    """Page d’accueil : KPIs, variations mensuelles, barres NCOA, etc."""
    
        # ──────────────────────────────────────────────────────────────────────────────
    # Récupération de la base mise en session par app.py
    # ----------------------------------------------------------------------------
    df = st.session_state.df
    poste_cols=st.session_state.poste_cols
    fonction_cols=st.session_state.fonction_cols
    glob_cols = st.session_state.glob_cols


    # ---- En‑tête ----------------------------------------------------------------
    st.title("📊 Observatoire des Prix à la Consommation – Côte d’Ivoire")
    st.caption("Mise à jour : %s" % df.index.max().strftime("%B %Y"))
    
    
    

    # ---- Cartes KPI -------------------------------------------------------------
    dernier   = df.index.max()
    precedent = df.index[df.index < dernier].max()

    ihpc_now  = df.loc[dernier, "IHPC"]
    ihpc_prev = df.loc[precedent, "IHPC"]
    ihpc_delta_abs = ihpc_now - ihpc_prev
    ihpc_delta_pct = 100 * (ihpc_now / ihpc_prev - 1)

    infl_gliss_now = df.loc[dernier, "InflationGliss"]

    periode_txt = f"{df.index.min():%Y-%m} ➜ {dernier:%Y-%m}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Variation mensuelle IHPC → vert si baisse
        card(
            col1, "IHPC",
            ihpc_now,
            delta=ihpc_delta_pct,
            unit="%",
            icon="📊",
            subtext="vs mois précédent",
            invert_colors=False  # la hausse n'est ni bonne ni mauvaise ici
        )

    with col2:
        # Inflation glissante (plus bas = mieux) => invert_colors=True
        card(
            col2, "Inflation glissante",
            infl_gliss_now,
            unit="%",
            icon="🔥",
            subtext="Moyenne 12m",
            invert_colors=True
        )

    with col3:
        card(
            col3, "Période couverte",
            value=periode_txt,
            icon="🗓️"
        )

    with col4:
        card(
            col4, "Postes suivis",
            value=len(poste_cols),
            icon="📦",
            value_fmt="{}"   # pas de format décimal
        )


    st.markdown("---")

    # ---- Variation mensuelle (postes) ------------------------------------------
    st.subheader("🚀 Mouvement mensuel des postes")
    var_pct = (df.loc[dernier, poste_cols] - df.loc[precedent, poste_cols]) / df.loc[precedent, poste_cols] * 100
    top_plus  = var_pct.nlargest(5).round(2)
    top_moins = var_pct.nsmallest(5).round(2)

    c1, c2 = st.columns(2)
    c1.markdown("### 🔺 Top 5 hausses (%)")
    c1.dataframe(top_plus)
    c2.markdown("### 🔻 Top 5 baisses (%)")
    c2.dataframe(top_moins)

    # ---- Variation moyenne par fonction NCOA -----------------------------------
    st.subheader("📊 Variation mensuelle moyenne par grande fonction NCOA")
    # Agrégation poste ➜ fonction
    agg_fct = {}
    for poste, fct in repartition_ncoa.items():
        if poste in var_pct:
            agg_fct.setdefault(fct, []).append(var_pct[poste])

    data_fct = pd.DataFrame({
        "Fonction": [ncoa_fonction[f] for f in agg_fct],
        "Variation (%)": [np.mean(v) for v in agg_fct.values()]
    }).sort_values("Variation (%)", ascending=False)

    # Tableau + bar chart
    st.dataframe(data_fct.style.format({"Variation (%)": "{:.2f}"}))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=data_fct, y="Fonction", x="Variation (%)", ax=ax)
    ax.set_xlabel("Variation moyenne (%)"); ax.set_ylabel("")
    st.pyplot(fig)

    # ---- Mini‑courbes des postes clés ------------------------------------------
    st.subheader("📈 Évolution rapide des postes les plus volatils")
    tp=top_plus.index.tolist()
    tm=top_moins.index.tolist()
    watch_list = top_plus.index.tolist() + top_moins.index.tolist()
    st.markdown(
        "ℹ️ LE TOP 5"
    )
    st.line_chart(df[tp])
    st.markdown(
        "ℹ️ LE BOTTOM 5"
    )
    st.line_chart(df[tm])

    # ---- Indicateurs d’inflation (si présents) ----------------------------------
    if {"Inflation", "InflationGliss"}.issubset(df.columns):
        st.subheader("🪙 Inflation détaillée")
        c1, c2 = st.columns(2)
        c1.metric("Inflation mensuelle", f"{df.loc[dernier, 'Inflation']:.2f} %")
        c2.metric("Inflation glissante", f"{df.loc[dernier, 'InflationGliss']:.2f} %")
        st.line_chart(df[["Inflation", "InflationGliss"]])
    else:
        st.info("Colonnes d’inflation non disponibles dans la base.")

    # ---- Qualité des données ----------------------------------------------------
    st.subheader("🔧 Qualité de la base")
    n_missing = df.isna().sum().sum()
    st.write(f"Valeurs manquantes totales : **{int(n_missing)}**")
    if n_missing:
        with st.expander("► Voir heatmap des NaN"):
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(df.isna(), cbar=False, ax=ax)
            st.pyplot(fig)

    # ---- Téléchargement rapide --------------------------------------------------
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button("💾 Télécharger la base complète (CSV)", csv_bytes, "base_observatoire.csv", "text/csv")
