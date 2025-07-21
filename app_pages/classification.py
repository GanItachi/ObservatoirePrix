# app_pages/classification.py – Analyse avancée interactive des groupes de postes
# --------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA

def render_classif():
    # --- Données de session ---
    df = st.session_state.df
    poste_cols = st.session_state.poste_cols
    

    st.title("🔗 Classification avancée des postes de consommation")

    # --- Sélection temporelle ---
    st.markdown("**📅 Période analysée**")
    min_date, max_date = df.index.min(), df.index.max()
    periode = st.date_input("Sélectionne la période", [min_date, max_date], min_value=min_date, max_value=max_date)
    df_ = df.loc[periode[0]:periode[1]]

    # --- Type de données ---
    var_type = st.selectbox(
        "Type de données pour classification",
        ["Indices bruts", "Variations mensuelles (%)", "Indicateurs dérivés"]
    )

    if var_type == "Indices bruts":
        X = df_[poste_cols].T
    elif var_type == "Variations mensuelles (%)":
        X = df_[poste_cols].pct_change().dropna().T
    else:  # Indicateurs dérivés
        feats = pd.DataFrame({
            poste: {
                "Tendance": np.polyfit(range(len(df_[poste])), df_[poste], 1)[0],
                "Volatilité": df_[poste].pct_change().std(),
                "Variation annuelle": df_[poste].pct_change(12).mean() * 100
            } for poste in poste_cols
        }).T
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(feats), index=feats.index, columns=feats.columns)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # --- Automatisation choix optimal k ---
    st.subheader("🔍 Choix optimal du nombre de groupes")
    k_range = range(2, 11)
    inerties, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_std)
        inerties.append(km.inertia_)
        sil_scores.append(silhouette_score(X_std, km.labels_))

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(x=list(k_range), y=inerties, mode='lines+markers', name='Inertie (méthode du coude)'))
    fig_k.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode='lines+markers', name='Score Silhouette', yaxis="y2"))

    fig_k.update_layout(
        yaxis=dict(title="Inertie"),
        yaxis2=dict(title="Score Silhouette", overlaying='y', side='right'),
        xaxis=dict(title="Nombre de groupes (k)"),
        legend=dict(x=0.1, y=1.1),
        height=400
    )
    st.plotly_chart(fig_k)

    k = st.slider("Choisir k basé sur l'analyse ci-dessus", 2, 10, 4)

    # --- Algorithme ---
    algo = st.selectbox("Algorithme de clustering", ["K-means", "CAH (Hiérarchique)"])

    if algo == "K-means":
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_std)
        labels = km.labels_ + 1
    else:
        Z = linkage(X_std, method='average', metric='euclidean')
        labels = fcluster(Z, t=k, criterion='maxclust')

    # --- Qualité des groupes ---
    st.subheader("📌 Évaluation qualitative des groupes")
    sil = silhouette_score(X_std, labels)
    intra_var = np.mean([X_std[labels == c].var() for c in np.unique(labels)])
    inter_var = X_std.var() - intra_var

    st.metric("Score Silhouette", f"{sil:.2f}")
    st.write(f"Variance intra-groupe moyenne : {intra_var:.4f}")
    st.write(f"Variance inter-groupe : {inter_var:.4f}")

    qual_txt = ("✅ Bonne qualité : groupes distincts" if sil > 0.5 else
                "⚠️ Qualité moyenne : groupes partiellement séparés" if sil > 0.3 else
                "❌ Qualité faible : interpréter avec précaution")
    st.info(qual_txt)

    # --- Mapping poste → groupe ---
    clust_df = pd.DataFrame({"Poste": poste_cols, "Groupe": labels}).sort_values("Groupe")
    st.dataframe(clust_df, use_container_width=True)
    st.download_button("💾 Exporter le mapping", clust_df.to_csv(index=False).encode('utf-8'), "groupes.csv")

    # --- Analyse fine des groupes ---
    st.subheader("📊 Analyse détaillée des groupes")
    for c in np.unique(labels):
        membres = clust_df[clust_df.Groupe == c].Poste.tolist()
        st.markdown(f"### 📌 Groupe {c} ({len(membres)} postes)")
        st.write(", ".join(membres))

        group_data = df_[membres].mean(axis=1)
        volatility = df_[membres].pct_change().std().mean() * 100
        correlation_ihpc = group_data.corr(df_["IHPC"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group_data.index, y=group_data, name=f"Groupe {c} (moyenne)"))
        fig.add_trace(go.Scatter(x=df_.index, y=df_["IHPC"], name="IHPC", line=dict(dash='dot')))
        fig.update_layout(height=300, title=f"Groupe {c} vs IHPC")
        st.plotly_chart(fig)

        st.write(f"Volatilité moyenne : **{volatility:.2f}%**")
        st.write(f"Corrélation avec IHPC : **{correlation_ihpc:.2f}**")

        interpret = ("Groupe stable aligné avec l'IHPC" if correlation_ihpc > 0.7 else
                     "Groupe volatile à surveiller" if volatility > 5 else
                     "Groupe atypique (faible corrélation avec l'IHPC)")
        st.info(interpret)

    # --- PCA interactif ---
    st.subheader("🌐 Projection PCA interactive des groupes")
    pca = PCA(n_components=2).fit_transform(X_std)
    pca_df = pd.DataFrame(pca, columns=["PC1", "PC2"], index=poste_cols)
    pca_df["Groupe"] = labels

    if "IHPC" in df.columns:
        ihpc_vecteur = scaler.transform(df_[["IHPC"]].T)
        ihpc_pca = PCA(n_components=2).fit(X_std).transform(ihpc_vecteur)[0]
        pca_df.loc["IHPC"] = [ihpc_pca[0], ihpc_pca[1], "IHPC"]

    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Groupe", text=pca_df.index,
                         title="PCA : Groupes et position de l'IHPC")
    fig_pca.update_traces(textposition='top center')
    st.plotly_chart(fig_pca)

    st.success("🚀 Analyse interactive complète !")

