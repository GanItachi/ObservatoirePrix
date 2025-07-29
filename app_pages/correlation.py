# app_pages/correlation.py – Analyse des corrélations
# ===================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




def render_correlation() -> None:
    st.header("🔗 Corrélations des indices")
    st.markdown(
        "Le **coefficient de corrélation de Pearson** varie de −1 (relation inverse parfaite) "
        "à +1 (relation directe parfaite). Un coefficient proche de 0 traduit une absence "
        "de lien linéaire mesurable."
    )
    
    # --------------------------------------------------------------------------
    df            = st.session_state.df
    poste_cols    = st.session_state.poste_cols
    fonction_cols = st.session_state.fonction_cols
    glob_cols     = st.session_state.glob_cols
    # --------------------------------------------------------------------------

    # 1️⃣ Paramètres interactifs
    # ------------------------------------------------------------------
    cible = st.selectbox(
        "Variable cible", 
        ["IHPC", "Inflation", "InflationGliss"] + fonction_cols + poste_cols
    )
    
    if cible not in df.columns:
        st.error(f"⚠️ La fonction **{cible}** n’est pas disponible dans les données chargées.")
        st.markdown(
            ":red[**Cette fonction n’est pas encore calculée par l’ANStat.**] "
            "Vous pouvez cependant l’estimer de manière approchée en allant dans la page **Traitement**."
        )
        return
    else:
        pass

    groupe = st.radio(
        "Comparer à…",
        ["Postes", "Fonctions", "Globaux"]
    )

    # 2️⃣ Constitution de l'ensemble à corréler
    # ------------------------------------------------------------------
    if   groupe == "Postes":
        cols = poste_cols
    elif groupe == "Fonctions":
        # on retire les fonctions absentes de la base (valeurs toutes NaN)
        cols = [c for c in fonction_cols if c in df.columns and df[c].notna().any()]
    elif groupe == "Globaux":
        cols = glob_cols
    else:
        cols = poste_cols + fonction_cols + glob_cols

    cols = [c for c in cols if c != cible]      # ne pas se corréler à soi‑même
    sub_df = df[[cible] + cols].dropna()

    if sub_df.empty or not cols:
        st.info("Pas assez de données non manquantes pour calculer la corrélation.")
        return

    # 3️⃣ Heatmap instantanée
    # ------------------------------------------------------------------
    st.subheader("🖼️ Carte de corrélation – toute la période")
    corr_ser = sub_df.corr()[cible].drop(cible).sort_values()

    fig, ax = plt.subplots(figsize=(3, len(corr_ser) * 0.25))
    sns.heatmap(
        corr_ser.to_frame(), annot=True, cmap="coolwarm",
        yticklabels=corr_ser.index, cbar=False, ax=ax
    )
    ax.set_xlabel(""); ax.set_ylabel("")
    st.pyplot(fig)

    # 4️⃣ Top 5 corrélations + et −
    # ------------------------------------------------------------------
    st.subheader("🏅 Séries les plus corrélées")
    top_pos = corr_ser.sort_values(ascending=False).head(5)
    top_neg = corr_ser.sort_values().head(5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔺 Top +")
        st.dataframe(top_pos.to_frame("ρ").style.format("{:.2f}"))
    with c2:
        st.markdown("### 🔻 Top −")
        st.dataframe(top_neg.to_frame("ρ").style.format("{:.2f}"))

    st.success(
        f"Corrélation la plus **forte** avec **{cible}** : **{corr_ser.idxmax()}** "
        f"(ρ = {corr_ser.max():+.2f}).  \n"
        f"Corrélation la plus **inverse** : **{corr_ser.idxmin()}** "
        f"(ρ = {corr_ser.min():+.2f})."
    )

    # 5️⃣ Corrélation glissante (corrige le KeyError 'variable')
    # ------------------------------------------------------------------
    with st.expander("📈 Corrélation glissante (fenêtre mobile)"):
        window = st.slider("Fenêtre (mois)", 6, 36, 12, step=3)

        # Corrélation rolling — extraction des lignes où la 2ᵉ dimension == cible
        roll_corr = df[[cible] + cols].rolling(window).corr()
        roll_corr = roll_corr.loc[(slice(None), cible), cols]   # (date, var1)==cible
        roll_corr.index = roll_corr.index.get_level_values(0)   # on garde la date seule

        st.line_chart(roll_corr)
        st.caption(
            "Lorsque la courbe change brutalement de signe ou d’amplitude, "
            "le lien entre la série et la variable cible se modifie."
        )

    # 6️⃣ Export CSV
    # ------------------------------------------------------------------
    st.download_button(
        "💾 Télécharger les coefficients",
        corr_ser.to_csv().encode("utf‑8"),
        f"correlations_{cible}.csv",
        "text/csv"
    )
