# app_pages/exploration.py – Exploration avancée
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import calendar
from core.nomenclature import ncoa_fonction, repartition_ncoa

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
# ────────────────────────────────────────────────────────────────────────────


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
    # ------------------------------------------------------------------
    fct_label = st.selectbox("Fonction (NCOA)", list(ncoa_fonction.values()))
    num_fct   = [k for k, v in ncoa_fonction.items() if v == fct_label][0]
    postes_fct = sorted([p for p, f in repartition_ncoa.items() if f == num_fct])
    poste = st.selectbox("Poste détaillé", postes_fct)

    serie_fct   = df[fct_label].dropna()
    serie_poste = df[poste].dropna()

    # 2️⃣ Évolution temporelle
    # ------------------------------------------------------------------
    st.subheader("📈 Évolution temporelle")
    st.markdown(
        "ℹ️ **Comment lire ?** L’axe horizontal montre le temps (mois) ; l’axe vertical la "
        "valeur de l’indice. Une courbe qui monte = des prix qui augmentent. "
        "Comparez la pente et les ruptures : plus la pente est raide, plus la hausse est rapide."
    )
    st.line_chart(pd.concat([serie_fct, serie_poste], axis=1))

    # Interprétation automatique : pente linéaire
    slope_fct   = np.polyfit(range(len(serie_fct)),   serie_fct.values,   1)[0]
    slope_poste = np.polyfit(range(len(serie_poste)), serie_poste.values, 1)[0]
    msg  = (
        f"• **{fct_label}** : tendance {'haussière' if slope_fct>0 else 'baissière'} \n"
        f"(≈ {slope_fct:+.2f}/mois).  \n"
        f"• **{poste}** : tendance {'haussière' if slope_poste>0 else 'baissière'} \n"
        f"(≈ {slope_poste:+.2f}/mois)."
    )
    st.info(msg)
    
    # ...existing code...
    # 2️⃣ Anomalies détectées
    # ------------------------------------------------------------------
    anomalies = {}
    zscores = np.abs((df[poste] - df[poste].mean()) / df[poste].std())
    anomalies[poste] = df[poste][zscores > 3]
    for poste, vals in anomalies.items():
        if not vals.empty:
            st.warning(f"Anomalies détectées pour {poste}: {vals.index.strftime('%Y-%m-%d').tolist()}")
    # ...existing code...
    
    
    # 3️⃣ Volatilité glissante
    # ------------------------------------------------------------------
    st.subheader("🌪️ Volatilité glissante (12 mois)")
    st.markdown(
        "ℹ️ **Comment lire ?** On mesure la dispersion des variations mensuelles "
        "sur les 12 derniers mois : plus la courbe est haute, plus les prix "
        "sont instables. Les pics signalent des phases de forte incertitude."
    )
    roll_std = pd.concat({
        fct_label: serie_fct.rolling(12).std(),
        poste:     serie_poste.rolling(12).std()
    }, axis=1)
    st.line_chart(roll_std)
    # Interprétation
    vol_peak_date  = roll_std[poste].idxmax()
    vol_peak_value = roll_std[poste].max()
    st.info(
        f"Volatilité maximale du poste **{poste}** en **{vol_peak_date:%b %Y}** \n"
        f"({vol_peak_value:.1f} points)."
    )

    # 4️⃣ Heatmap YoY (variation annuelle des postes de la fonction)
    # ------------------------------------------------------------------
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

    # 6️⃣ Boxplot mensuel (saisonnalité)
    # ------------------------------------------------------------------

    # ────────────────────────────────────────────────────────────────
    # 1) Option : retirer (ou non) la tendance
    # ────────────────────────────────────────────────────────────────
    st.markdown("#### *ANALYSE SAISONNALITE*")
    detrend_opt = st.checkbox("Retirer la tendance avant l’analyse", value=True)

    if detrend_opt:
        trend     = serie_poste.rolling(12, min_periods=1).mean()
        working   = (serie_poste - trend).dropna()
    else:
        working   = serie_poste.dropna()

    # ────────────────────────────────────────────────────────────────
    # 2) Préparer DataFrame mensuel
    # ────────────────────────────────────────────────────────────────
    df_box = working.to_frame("val").reset_index()
    df_box["mois_num"] = df_box["date"].dt.month
    df_box["mois"] = df_box["mois_num"].apply(lambda m: calendar.month_name[m])


    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df_box, x="mois", y="val", ax=ax, palette="Blues",order=calendar.month_name[1:])
    ax.set_xlabel("Mois"); ax.set_ylabel("Valeur (détrendue)" if detrend_opt else "Valeur brute")
    st.pyplot(fig)

    # ────────────────────────────────────────────────────────────────
    # 3) Test de Kruskal
    # ────────────────────────────────────────────────────────────────
    groups = [g["val"].values for _, g in df_box.groupby("mois")]
    stat, pval = stats.kruskal(*groups)

    if pval < 0.05:
        st.success(f"Saisonnalité **significative** (p = {pval:.3g})")
    else:
        st.info(f"Aucune saisonnalité détectée (p = {pval:.3g})")

    # ────────────────────────────────────────────────────────────────
    # 4) ► Interprétation – Effet moyen de chaque mois
    # ────────────────────────────────────────────────────────────────
    if pval < 0.05:
        eff_mois = df_box.groupby("mois")["val"].mean().reindex(calendar.month_name[1:])
        base_mean = eff_mois.mean()
        delta = eff_mois - base_mean                    # écart à la moyenne
        pct   = 100 * delta / base_mean                 # en %

        interp_df = pd.DataFrame({
            "Mois": eff_mois.index,
            "Effet moyen": eff_mois.round(2),
            "Δ vs moyenne": delta.round(2),
            "Δ (%)": pct.round(1),
        })

        st.markdown("#### Effet moyen par mois")
        st.dataframe(interp_df, use_container_width=True)

        # Mini‑graphe barre des deltas
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar(interp_df["Mois"], interp_df["Δ (%)"], color=np.where(interp_df["Δ (%)"]>0, "#d62728", "#097bee"))
        ax2.axhline(0, color="grey", lw=1)
        ax2.set_xlabel("Mois"); ax2.set_ylabel("Écart moyen (%)")
        ax2.set_title("Effet saisonnier moyen par mois")
        st.pyplot(fig2)

        st.caption(
            "🔍 **Lecture** : un Δ(%) positif signifie que, toutes choses égales "
            "par ailleurs, ce mois est en moyenne plus élevé que la moyenne annuelle ; "
            "négatif = mois plus bas."
        )

    # ------------------------------------------------------------------
    # 4) Affichage facultatif : comparer avant / après
    # ------------------------------------------------------------------
    try :
        with st.expander("👁️ Voir la série détrendue vs originale"):
            st.line_chart(pd.concat({"Originale": serie_poste, "Détrendue": trend}, axis=1))
    except:
        pass

    # 7️⃣ Téléchargement CSV
    # ------------------------------------------------------------------
    csv_bytes = pd.concat([serie_fct, serie_poste], axis=1).to_csv().encode("utf‑8")
    st.download_button(
        "💾 Télécharger ces séries",
        csv_bytes,
        f"{fct_label}_{poste}.csv",
        "text/csv"
    )

    # 8️⃣ Heatmap « Par fonction » (robuste)
    # ------------------------------------------------------------------
    st.subheader("🖼️ Heatmap des indices par fonction (vue globale)")
    # Filtre des fonctions effectivement présentes (au moins une valeur non‑nulle)
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

# ────────────────────────────────────────────────────────────────────────────
