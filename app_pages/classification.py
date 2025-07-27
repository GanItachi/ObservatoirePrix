import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.seasonal import STL

# Palette couleurs (peut être raffinée)
CLUSTER_COLORS = ["#22a879", "#e98d27", "#e5544d", "#4472c4", "#9b59b6"]


def render_classif():
    st.title("🌈 Profils types des postes de consommation")
    st.markdown(
        "Obtenez une vue d'ensemble sur la dynamique des groupes de consommation : "
        "hausse structurelle, saisonnalité, volatilité… Sélectionnez un groupe pour explorer un poste en détail."
    )
    mapping = pd.read_csv("mapping_clusters.csv", index_col=0)
    desc_clusters = pd.read_csv("desc_clusters.csv")
    df = st.session_state.df

    for i, row in desc_clusters.iterrows():
        # Extraction robuste de la liste des postes
        try:
            postes = ast.literal_eval(row['postes'])
            if not isinstance(postes, list):
                postes = [str(postes)]
        except Exception:
            postes = [p.strip() for p in str(row['postes']).split(',') if p.strip()]

        present_cols = [col for col in postes if col in df.columns]
        missing_cols = [col for col in postes if col not in df.columns]

        cluster_color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        st.markdown(
            f"<h3 style='color:{cluster_color};font-weight:bold'>🟢 {row['nom']} "
            f"<span style='font-size:0.8em'>({len(postes)} postes)</span></h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='font-size:1.08em'><b>Description :</b> "
            f"Tendance : <b>{row['trend']:.2f}</b>, "
            f"Saisonnalité : <b>{row['season']:.2f}</b>, "
            f"Volatilité : <b>{row['volatility']:.2f}%</b>. "
            f"Exemples : {', '.join(postes[:6])}{'…' if len(postes) > 6 else ''}</div>",
            unsafe_allow_html=True
        )

        with st.container():
            # Recommandation synthétique
            if "saisonnier" in row['nom'].lower():
                st.info("🔎 **Recommandation :** Surveillez ces postes surtout lors des pics saisonniers.")
            elif "volatil" in row['nom'].lower():
                st.info("🚨 **Recommandation :** Renforcez la veille sur ces postes à variations brusques.")
            else:
                st.success("ℹ️ **Recommandation :** Suivi standard ; ces postes sont stables ou peu sensibles.")

            if present_cols:
                sub = df[present_cols]
                means = sub.mean()
                volas = sub.pct_change().std() * 100
                saison_metric = sub.groupby(sub.index.month).mean().std()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Le plus cher (moy)", means.idxmax(), f"{means.max():.2f}")
                with col2:
                    st.metric("Le plus volatil (%)", volas.idxmax(), f"{volas.max():.2f}")
                with col3:
                    st.metric("Saisonnalité max.", saison_metric.idxmax(), f"{saison_metric.max():.2f}")

                # Sélecteur interactif
                st.subheader("🔍 Explorer un poste du cluster")
                poste_sel = st.selectbox("Choisissez un poste à explorer :", present_cols, key=f"poste_{i}")
                serie = df[poste_sel]
                csv_bytes = serie.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les données du poste",
                    data=csv_bytes,
                    file_name=f"{poste_sel}_series.csv",
                    mime="text/csv"
                )

                # 1️⃣ Série sélectionnée
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(serie.index, serie.values, color=cluster_color, linewidth=2)
                ax.set_title(f"Évolution de {poste_sel}")
                ax.set_xlabel('Date'); ax.set_ylabel('Indice')
                plt.xticks(rotation=45)
                st.pyplot(fig)

                buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight')
                st.download_button(
                    label="📥 Télécharger le graphique PNG",
                    data=buf.getvalue(),
                    file_name=f"{poste_sel}_evolution.png",
                    mime="image/png"
                )

                # Calcul de l'indice moyen du cluster
                cluster_mean = sub.mean(axis=1)
                st.subheader("📈 Tendance moyenne du cluster")
                # Décomposer pour extraire la tendance
                stl = STL(cluster_mean, period=12, robust=True).fit()
                trend = stl.trend
                fig_trend, ax_trend = plt.subplots(figsize=(6, 3))
                ax_trend.plot(trend.index, trend.values, color=cluster_color, linewidth=2)
                ax_trend.set_title('Composante tendance (moyenne cluster)')
                ax_trend.set_xlabel('Date'); ax_trend.set_ylabel('Tendance')
                plt.xticks(rotation=45)
                st.pyplot(fig_trend)
                buf_tr = BytesIO(); fig_trend.savefig(buf_tr, format='png', bbox_inches='tight')
                st.download_button('📥 Télécharger tendance PNG', buf_tr.getvalue(),
                                   f"{row['nom']}_tendance.png", 'image/png')

                # 2️⃣ Signature saisonnière détrendée
                st.markdown("**Signature saisonnière moyenne du cluster (détrend)**")
                seasonal = stl.seasonal
                saison_moy = seasonal.groupby(seasonal.index.month).mean()
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.bar(saison_moy.index, saison_moy.values, color=cluster_color)
                ax2.set_title('Signature saisonnière détrendée')
                ax2.set_xlabel('Mois'); ax2.set_ylabel('Saisonnalité')
                plt.xticks(rotation=45)
                st.pyplot(fig2)
                buf2 = BytesIO(); fig2.savefig(buf2, format='png', bbox_inches='tight')
                st.download_button('📥 Télécharger saisonnière PNG', buf2.getvalue(),
                                   f"{row['nom']}_saisonniere_detrend.png", 'image/png')

                # 3️⃣ Variations mensuelles moyennes
                st.markdown("**Variations mensuelles moyennes (%) du cluster**")
                vols_mean = cluster_mean.pct_change() * 100
                fig3, ax3 = plt.subplots(figsize=(6, 3))
                ax3.plot(vols_mean.index, vols_mean.values, color=cluster_color)
                ax3.set_title('Variations mensuelles moyennes (%)')
                ax3.set_xlabel('Date'); ax3.set_ylabel('% Variation')
                plt.xticks(rotation=45)
                st.pyplot(fig3)
                buf3 = BytesIO(); fig3.savefig(buf3, format='png', bbox_inches='tight')
                st.download_button('📥 Télécharger variations PNG', buf3.getvalue(),
                                   f"{row['nom']}_variations.png", 'image/png')

            else:
                st.warning("Aucune colonne valide dans ce cluster pour la base de données actuelle.")
                
            if missing_cols:
                st.markdown(
                    f"<span style='color:#e67e22;font-size:0.95em'>Colonnes absentes : {'; '.join(missing_cols)}</span>",
                    unsafe_allow_html=True
                )

        st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("ℹ️ Glossaire"):
        st.markdown("""
        - **Tendance :** Composante de fond de la série, extraite par décomposition STL.
        - **Saisonnalité :** Composante cyclique après retrait de la tendance.
        - **Volatilité :** Amplitude des variations mensuelles.
        - **CV (coefficient de variation) :** Rapport de la dispersion à la moyenne.
        """)

    st.caption(
        "Typologie calculée automatiquement ; mise à jour régulière. Pour détails méthodologiques, voir la section Méthodologie."
    )
    
        # Lien vers la documentation détaillée
    with st.expander("📖 En savoir plus"):
        st.markdown(
            "Ce cluster regroupe les postes dont le profil est détaillé dans le mémoire (section 4.X). "
            "Pour une description complète et les recommandations associées, consultez le document de référence."
        )
        st.markdown("[Télécharger le mémoire (PDF)](docs/memoire.pdf)")


if __name__ == "__main__":
    if "df" not in st.session_state:
        st.error("Charge la base via app.py avant d’ouvrir cette page.")
    else:
        render_classif()
