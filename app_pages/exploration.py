# app_pages/exploration.py – Exploration avancée
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import calendar
from statsmodels.tsa.stattools import adfuller  
from core.nomenclature import ncoa_fonction, repartition_ncoa

import scipy.stats as stats

# ────────────────────────────────────────────────────────────────────────────
def _quick_stats(series: pd.Series) -> pd.Series:
    """Statistiques descriptives condensées pour une série."""
    return pd.Series({
        "Moyenne": series.mean(),
        "Écart‑type": series.std(),
        "Min": series.min(),
        "Max": series.max(),
        "Coeff. variation": series.std() / series.mean() if series.mean() else np.nan
    })

# ────────────────────────────────────────────────────────────────────────────
def render_exploration() -> None:
    st.header("🔎 Exploration avancée des indices")
    df              = st.session_state.df
    poste_cols      = st.session_state.poste_cols
    fonction_cols   = st.session_state.fonction_cols
    glob_cols       = st.session_state.glob_cols

    # 1️⃣ Sélection des séries à comparer
    fct_label = st.selectbox("Fonction (NCOA)", list(ncoa_fonction.values()))
    num_fct   = [k for k, v in ncoa_fonction.items() if v == fct_label][0]
    postes_fct = sorted([p for p, f in repartition_ncoa.items() if f == num_fct])
    poste = st.selectbox("Poste détaillé", postes_fct)

    # Gestion fonction absente
    if fct_label not in df.columns:
        st.error(
            f"⚠️ La fonction **{fct_label}** n’est pas disponible dans les données chargées."
        )
        st.markdown(
            ":red[**Cette fonction n’est pas encore calculée par l’ANStat.**] "
            "Vous pouvez cependant l’estimer de manière approchée en allant dans la page **Traitement**."
        )
        serie_fct = pd.Series(dtype=float)
        fct_exists = False
    else:
        serie_fct = df[fct_label].dropna()
        fct_exists = True

    serie_poste = df[poste].dropna()

    # ==== Bloc : Aperçu rapide ===================================
    with st.expander("ℹ️ Statistiques rapides"):
        st.write(_quick_stats(serie_poste).to_frame("Valeur"))

    # 2️⃣ Évolution temporelle
    st.subheader("📈 *Évolution temporelle*")
    st.markdown(
        "ℹ️ **Comment lire ?** L’axe horizontal montre le temps (mois) ; l’axe vertical la "
        "valeur de l’indice. Une courbe qui monte = des prix qui augmentent. "
        "Comparez la pente et les ruptures : plus la pente est raide, plus la hausse est rapide."
    )
    if fct_exists and not serie_fct.empty:
        st.line_chart(pd.concat([serie_fct, serie_poste], axis=1))
    else:
        st.line_chart(serie_poste.to_frame(name=poste))

    # Interprétation automatique : pente linéaire
    try:
        if fct_exists and not serie_fct.empty:
            slope_fct = np.polyfit(range(len(serie_fct)), serie_fct.values, 1)[0]
        else:
            slope_fct = None
    except Exception:
        slope_fct = None

    slope_poste = np.polyfit(range(len(serie_poste)), serie_poste.values, 1)[0]

    if slope_fct is not None:
        msg = (
            f"• **{fct_label}** : tendance {'haussière' if slope_fct > 0 else 'baissière'}\n"
            f"(≈ {slope_fct:+.2f}/mois).  \n"
            f"• **{poste}** : tendance {'haussière' if slope_poste > 0 else 'baissière'}\n"
            f"(≈ {slope_poste:+.2f}/mois)."
        )
        st.info(msg)
    else:
        st.warning(
            f"⚠️ La tendance pour la fonction **{fct_label}** n’est pas affichée (fonction indisponible)."
        )
        msg = (
            f"• **{poste}** : tendance {'haussière' if slope_poste > 0 else 'baissière'}\n"
            f"(≈ {slope_poste:+.2f}/mois)."
        )
        st.info(msg)

    # 3️⃣ Volatilité glissante
    st.subheader("🌪️ Volatilité glissante (12 mois)")
    st.markdown(
        "ℹ️ **Comment lire ?** On mesure la dispersion des variations mensuelles "
        "sur les 12 derniers mois : plus la courbe est haute, plus les prix "
        "sont instables. Les pics signalent des phases de forte incertitude."
    )
    if fct_exists and not serie_fct.empty:
        roll_std = pd.concat({
            fct_label: serie_fct.rolling(12).std(),
            poste: serie_poste.rolling(12).std()
        }, axis=1)
    else:
        roll_std = pd.concat({
            poste: serie_poste.rolling(12).std()
        }, axis=1)
    st.line_chart(roll_std)
    # Interprétation
    vol_peak_date  = roll_std[poste].idxmax()
    vol_peak_value = roll_std[poste].max()
    st.info(
        f"Volatilité maximale du poste **{poste}** en **{vol_peak_date:%b %Y}** \n"
        f"({vol_peak_value:.1f} points)."
    )

    # 6️ Boxplot mensuel (saisonnalité)
    st.markdown("# *ANALYSE SAISONNALITE*")
    st.markdown("ℹ️ **Comment lire ?** Le boxplot montre la distribution des valeurs par mois. "
                "Chaque boîte représente les valeurs du poste pour un mois donné, "
                "avec la médiane, les quartiles et les valeurs extrêmes. "
                "On peut ainsi visualiser les variations mensuelles et détecter des effets saisonniers.")
    detrend_opt = st.checkbox("Retirer la tendance avant l’analyse", value=True)

    if detrend_opt:
        trend     = serie_poste.rolling(12, min_periods=1).mean()
        working   = (serie_poste - trend).dropna()
    else:
        working   = serie_poste.dropna()

    df_box = working.to_frame("val").reset_index()
    df_box["mois_num"] = df_box["date"].dt.month
    df_box["mois"] = df_box["mois_num"].apply(lambda m: calendar.month_name[m])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df_box, x="mois", y="val", ax=ax, color='cyan', order=calendar.month_name[1:])
    ax.set_xlabel("Mois"); ax.set_ylabel("Valeur (sans trend)" if detrend_opt else "Valeur brute")
    st.pyplot(fig)

    groups = [g["val"].values for _, g in df_box.groupby("mois")]
    stat, pval = stats.kruskal(*groups)

    if pval < 0.05:
        st.success(f"Saisonnalité **significative** (p = {pval:.3g})")
    else:
        st.info(f"Aucune saisonnalité détectée (p = {pval:.3g})")

    if pval < 0.05:
        eff_mois = df_box.groupby("mois")["val"].mean().reindex(calendar.month_name[1:])
        base_mean = eff_mois.mean()
        delta = eff_mois - base_mean
        pct   = 100 * delta / base_mean

        interp_df = pd.DataFrame({
            "Effet moyen": eff_mois.round(2)
        })

        st.markdown("#### Effet moyen par mois")
        st.markdown("ℹ️ **Comment lire ?** Chaque mois a un effet moyen sur le poste. "
                    "Un effet positif signifie que ce mois est en moyenne plus élevé que la moyenne annuelle, "
                    "négatif = plus bas.")
        st.dataframe(interp_df.T, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax2.bar(interp_df.index, interp_df["Effet moyen"], color=np.where(interp_df["Effet moyen"]>0, "#d62728", "#097bee"))
        ax2.axhline(0, color="grey", lw=1)
        ax2.set_xlabel("Mois"); ax2.set_ylabel("Effet moyen (%)")
        ax2.set_title("Effet saisonnier moyen par mois")
        st.pyplot(fig2)

        st.caption(
            "🔍 **Lecture** : Le pourcentage indique l'écart par rapport à la moyenne annuelle. C'est à dire l'augmentation moyenne des prix en fonction du mois."
        )

    try:
        with st.expander("👁️ Voir la série détrendue vs originale"):
            st.markdown("**Série détrendue** : la tendance de fond a été retirée pour mieux visualiser les variations mensuelles.")
            st.markdown("**Série originale** : la série brute, avec sa tendance de fond.")
            st.markdown("**Comment lire ?** La série détrendue permet de mieux visualiser les variations mensuelles sans l’influence de la tendance de fond.")
            st.line_chart(pd.concat({"Originale": serie_poste, "Détrendue": trend}, axis=1))
    except:
        pass

    # 7️⃣ Téléchargement CSV
    csv_bytes = pd.concat([serie_fct, serie_poste], axis=1).to_csv().encode("utf‑8")
    st.download_button(
        "💾 Télécharger ces séries",
        csv_bytes,
        f"{fct_label}_{poste}.csv",
        "text/csv"
    )

    # 📌  SECTION – Analyse de stationnarité
    st.markdown("# 🔬 Analyse de la stationnarité")
    with st.expander("⚙️ Paramètres d’analyse", expanded=True):
        colA, colB, colC = st.columns(3)
        ordre_diff = colA.number_input("Ordre de différenciation (d)", 0, 3, 0)
        log_trf    = colB.checkbox("Appliquer un log(x)", value=False,
                                help="Utile pour stabiliser la variance.")
        test_selec = colC.radio("Test statistique",
                                ["Dickey‑Fuller augmentée (ADF)"],
                                index=0)

    serie_work = serie_poste.copy()

    if log_trf:
        serie_work = np.log(serie_work.replace(0, np.nan)).dropna()

    for _ in range(int(ordre_diff)):
        serie_work = serie_work.diff().dropna()

    if test_selec.startswith("Dickey"):
        stat, pval, lags, nobs, crit, _ = adfuller(serie_work)
        interpr = "✅ Stationnaire" if pval < 0.05 else "⚠️ Non‑stationnaire"
        st.info(f"**ADF = {stat:.3f}, p‑value = {pval:.4f} → {interpr}**")
        st.caption("H₀ : la série possède une racine unitaire (non‑stationnaire).")

    crit_str = ", ".join([f"{k}: {v:.3f}" for k, v in crit.items()])
    st.write("*Valeurs critiques* :", crit_str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(serie_poste, label="Série originale", alpha=0.4)
    ax.plot(serie_work,  label=f"Série transformée (log={log_trf}, d={ordre_diff})")
    ax.set_title("Visualisation : origine vs. transformée")
    ax.legend(); ax.grid(alpha=.2)
    st.pyplot(fig)

    with st.expander("💡 Interpréter les résultats"):
        st.markdown(
            """
            - **ADF** : p‑value < 0,05 ⇒ on rejette H₀ ⇒ la série n’a **pas** de racine unitaire ⇒ stationnaire.  
            - **KPSS** : p‑value > 0,05 ⇒ on **ne** rejette pas H₀ ⇒ stationnaire.  
            - **Différenciation (d)** : chaque ordre enlève une tendance.  
            - **Log** : réduit l’amplitude des fortes variations, utile si variance croissante.
            """
        )

    st.markdown("# 📊 Analyse globale des indices")
    # 8️⃣ Heatmap « Par fonction » (vue globale)
    st.subheader("🖼️ Heatmap des indices par fonction (vue globale)")
    st.markdown(
        "ℹ️ **Comment lire ?** Chaque ligne représente une fonction NCOA, "
        "chaque colonne un mois. Les couleurs indiquent l’indice de prix "
        "(plus c’est foncé, plus les prix sont élevés)."
    )
    available_fcts = [
        col for col in fonction_cols
        if col in df.columns and df[col].notna().any()
    ]
    missing_fcts = [f for f in fonction_cols if f not in available_fcts]

    if not available_fcts:
        st.info("Aucune fonction NCOA n’est encore agrégée. "
                "Exécutez d’abord la page **Traitement** pour les créer.")
    else:
        fig, ax = plt.subplots(figsize=(18, len(available_fcts) * 0.35))
        sns.heatmap(df[available_fcts].T, cmap="YlGnBu",
                    cbar_kws={'label': 'Indice de prix'}, ax=ax)
        ax.set_xlabel("Date"); ax.set_ylabel("Fonction")
        st.pyplot(fig)
        if missing_fcts:
            st.caption(f"Fonctions non affichées : {', '.join(missing_fcts)}")

    #4️⃣ Heatmap YoY (variation annuelle des postes de la fonction)
    st.subheader("🔥 Heatmap variation annuelle (%) des postes de la fonction")
    st.markdown(
        "ℹ️ **Comment lire ?** Chaque case représente la variation sur 12 mois "
        "pour un poste donné : rouge = hausse ; bleu = baisse. Les couleurs "
        "vives localisent les périodes de choc prix."
    )
    df_yoy = df[postes_fct].pct_change(12) * 100
    fig, ax = plt.subplots(figsize=(12, len(postes_fct) * 0.35))
    sns.heatmap(df_yoy.T, cmap="coolwarm", center=0, cbar_kws={"label": "% YoY"}, ax=ax)
    st.pyplot(fig)
    last_yoy = df_yoy.iloc[-1].dropna()
    if not last_yoy.empty:
        top_inc  = last_yoy.idxmax(); top_inc_val  = last_yoy.max()
        top_dec  = last_yoy.idxmin(); top_dec_val  = last_yoy.min()
        st.success(
            f"Dernier mois : hausse la plus forte **{top_inc}** (+{top_inc_val:.1f} %), \n"
            f"baisse la plus forte **{top_dec}** ({top_dec_val:.1f} %)."
        )

# ────────────────────────────────────────────────────────────────────────────
