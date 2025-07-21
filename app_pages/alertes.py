# app_pages/alertes.py – Système d’alertes avancées (version optimisée)
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def render_alertes():
    df = st.session_state.df
    poste_cols = st.session_state.poste_cols

    st.title("🚨 Système d’Alertes intelligentes – Optimisé")

    # -- Aide utilisateur --
    st.info("""
    Ce module détecte :
    - Les ruptures inhabituelles de tendance ou de niveau sur toute la série
    - Les hausses rapides ou volatilités anormales selon **vos paramètres**
    - **Toutes les alertes sont expliquées, visualisées et exportables**
    """)

    with st.expander("🔧 Paramètres avancés d’alerte"):
        with st.form("alert_form"):
            c1, c2 = st.columns(2)
            seuil_pct    = c1.number_input("🔺 Hausse cumulée (%)", 0.0, 100.0, 5.0)
            fenetre_mois = c2.number_input("📈 Fenêtre (mois)", 1, 12, 3)
            seuil_vol    = c1.number_input("🌪 Volatilité (%)", 0.0, 100.0, 8.0)
            n_bkps_max   = c2.slider("💥 Nb max. ruptures", 1, 5, 2)
            pen_detect   = c1.number_input("⚙️ Sensibilité rupture (pénalité)", 2.0, 30.0, 6.0)
            detection_mode = c2.radio(
                "🔎 Détection", 
                ["Sur toute la série (rétrospectif)", "Sur la période récente seulement"], 
                index=1
            )
            fusion_rupts = st.checkbox("Fusionner ruptures très proches (1 mois)", value=True)
            filtrage_type = st.multiselect(
                "Types d’alertes à afficher", 
                ["Rupture structurelle", "Hausse rapide", "Volatilité"],
                default=["Rupture structurelle", "Hausse rapide", "Volatilité"]
            )
            submit = st.form_submit_button("Enregistrer mes réglages")
        if submit:
            st.session_state["alert_rules"] = dict(
                pct=seuil_pct, window=fenetre_mois, vol=seuil_vol,
                n_bkps=n_bkps_max, pen=pen_detect,
                mode=detection_mode, fusion=fusion_rupts,
                types=filtrage_type
            )
        st.markdown("""
        - **Pénalité rupture** : plus élevé = moins de ruptures détectées, à ajuster selon la volatilité des postes.
        - **Détection toute la série** : repère aussi les alertes historiques, pas juste sur la période actuelle.
        - **Fusion ruptures** : évite de signaler deux ruptures très proches.
        - **Filtrage types** : affichez seulement les alertes qui vous intéressent.
        """)

    rules = st.session_state.get("alert_rules", {
        "pct": 5.0, "window": 3, "vol": 8.0, "n_bkps": 2, "pen": 6.0,
        "mode": "Sur la période récente seulement", "fusion": True,
        "types": ["Rupture structurelle", "Hausse rapide", "Volatilité"]
    })

    # --- 1. Détection des ruptures (changepoints) sur toute la série ---
    rupture_rows = []
    for poste in poste_cols:
        s = df[poste].dropna()
        if len(s) < 12 or s.nunique() < 3: continue
        try:
            algo = rpt.Pelt(model="rbf", min_size=6).fit(s.values)
            bkps = algo.predict(pen=rules["pen"])
            ruptures = []
            for b in bkps[:-1]:  # Exclut la fin de série
                if b-2 < 0: continue
                date_rupt = s.index[b-1]
                delta = s.iloc[b-1] - s.iloc[b-2]
                ruptures.append((date_rupt, delta))
            # Option fusion ruptures trop proches (<1 mois)
            if rules["fusion"] and len(ruptures) > 1:
                filtered = []
                prev_date = None
                for date, delta in ruptures:
                    if prev_date is None or (date - prev_date).days > 32:
                        filtered.append((date, delta))
                        prev_date = date
                ruptures = filtered
            # Limite au nombre max
            for i, (date, delta) in enumerate(ruptures[:rules["n_bkps"]]):
                rupture_rows.append({
                    "Poste": poste, "Date": date, "Amplitude": f"{delta:.1f}", 
                    "Type": "Rupture structurelle", 
                    "Commentaire": f"Changement brusque de {delta:.1f} à la date {date:%Y-%m}"
                })
        except Exception:
            continue

    # --- 2. Alertes sur hausses rapides / volatilité ---
    alerte_rows = []
    for poste in poste_cols:
        s = df[poste].dropna()
        # Mode rétrospectif : recherche sur toute la série
        if rules["mode"] == "Sur toute la série (rétrospectif)" and len(s) > rules["window"]:
            for t in range(rules["window"], len(s)):
                val_start = s.iloc[t-rules["window"]]
                val_end = s.iloc[t]
                pct = 100 * (val_end - val_start) / val_start if val_start else np.nan
                window_dates = (s.index[t-rules["window"]], s.index[t])
                if pct >= rules["pct"] and "Hausse rapide" in rules["types"]:
                    alerte_rows.append({
                        "Poste": poste, "Date": window_dates[1],
                        "Amplitude": f"{pct:.1f}%", "Type": "Hausse rapide",
                        "Commentaire": f"Hausse de {pct:.1f}% entre {window_dates[0]:%Y-%m} et {window_dates[1]:%Y-%m}"
                    })
                # Volatilité sur la fenêtre glissante
                vols = s.pct_change().iloc[t-rules["window"]:t].std() * 100
                if vols >= rules["vol"] and "Volatilité" in rules["types"]:
                    alerte_rows.append({
                        "Poste": poste, "Date": window_dates[1],
                        "Amplitude": f"{vols:.1f}%", "Type": "Volatilité",
                        "Commentaire": f"Volatilité de {vols:.1f}% entre {window_dates[0]:%Y-%m} et {window_dates[1]:%Y-%m}"
                    })
        # Mode actuel : uniquement dernière période
        elif len(s) > rules["window"]:
            val_start, val_end = s.iloc[-rules["window"]], s.iloc[-1]
            pct = 100 * (val_end - val_start) / val_start if val_start else np.nan
            window_dates = (s.index[-rules["window"]], s.index[-1])
            if pct >= rules["pct"] and "Hausse rapide" in rules["types"]:
                alerte_rows.append({
                    "Poste": poste, "Date": window_dates[1],
                    "Amplitude": f"{pct:.1f}%", "Type": "Hausse rapide",
                    "Commentaire": f"Hausse de {pct:.1f}% entre {window_dates[0]:%Y-%m} et {window_dates[1]:%Y-%m}"
                })
            vols = s.pct_change().iloc[-rules["window"]:].std() * 100
            if vols >= rules["vol"] and "Volatilité" in rules["types"]:
                alerte_rows.append({
                    "Poste": poste, "Date": window_dates[1],
                    "Amplitude": f"{vols:.1f}%", "Type": "Volatilité",
                    "Commentaire": f"Volatilité de {vols:.1f}% entre {window_dates[0]:%Y-%m} et {window_dates[1]:%Y-%m}"
                })

    # --- 3. Filtrage des types d'alertes à afficher ---
    all_alertes = rupture_rows + alerte_rows
    if rules["types"]:
        all_alertes = [a for a in all_alertes if a["Type"] in rules["types"]]

    alertes_df = pd.DataFrame(all_alertes)
    n_alertes = len(alertes_df)
    st.session_state["n_alertes"] = n_alertes

    # --- 4. Affichage synthétique, export et tri par importance ---
    if n_alertes:
        st.sidebar.error(f"🚨 {n_alertes} alerte(s) détectée(s)")
        st.error(f"⚠️ **{n_alertes} alerte(s) active(s) détectée(s) dans la base.**")
        st.dataframe(alertes_df.sort_values("Date", ascending=False), use_container_width=True)
        st.download_button("💾 Exporter les alertes", alertes_df.to_csv(index=False).encode('utf-8'), "alertes.csv")
    else:
        st.success("✅ Aucun signal d’alerte sur la période/postes sélectionnés.")

    # --- 5. Visualisation et explication détaillée ---
    if n_alertes:
        poste_sel = st.selectbox("Voir le détail d’un poste en alerte", alertes_df["Poste"].unique())
        s = df[poste_sel]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s, label=poste_sel)
        rupt_dates = alertes_df[(alertes_df.Poste==poste_sel) & (alertes_df.Type=="Rupture structurelle")]["Date"]
        for r in rupt_dates:
            ax.axvline(r, color="red", ls="--", lw=1.2, label="Rupture" if "Rupture" not in ax.get_legend_handles_labels()[1] else "")
        ax.set_title(f"Historique et ruptures – {poste_sel}")
        ax.grid(True); ax.legend()
        st.pyplot(fig)
        st.write("Distribution des variations :", s.pct_change().dropna().describe())

        st.info("💡 **Commentaires sur les alertes sélectionnées**")
        for _, row in alertes_df[alertes_df.Poste==poste_sel].iterrows():
            st.markdown(f"- **{row['Type']} :** {row['Commentaire']}")

    st.caption("Ce module est conçu pour offrir une veille réactive et personnalisée, adaptée aux besoins des décideurs publics et analystes économiques. Chaque alerte est contextualisée et visualisable pour action rapide.")

# --- Point d’entrée debug (optionnel) ---
if __name__ == "__main__":
    if "df" not in st.session_state:
        st.error("DataFrame absent → exécutez d’abord app.py")
    else:
        render_alertes()
