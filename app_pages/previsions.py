# app_pages/previsions.py – Projection finale & simulation de prix
# ===============================================================
from __future__ import annotations
import pandas as pd, numpy as np, matplotlib.pyplot as plt, streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta

# ────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────
def _months_between(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    """Nombre de mois entiers entre deux timestamps (d2 > d1)."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

# ----------------------------------------------------------------
def _forecast_with_saved(model_obj, model_name: str,
                         full_series: pd.Series,
                         horizon: int) -> pd.DataFrame:
    """
    Renvoie DataFrame index futur : ['Prévision','IC_lower','IC_upper']  
    (ajoute la dernière obs en première ligne pour la continuité visuelle)
    """
    last_date = full_series.index[-1]
    future_idx = pd.date_range(last_date + pd.DateOffset(months=1),
                               periods=horizon, freq="MS")

    # --- prédiction suivant le type de modèle déjà sauvegardé ------------
    if model_name == "Prophet":
        from prophet import Prophet
        dfp = full_series.rename("y").reset_index().rename(columns={full_series.index.name or "index": "ds"})
        model_obj.fit(dfp)  # ré‑entraînement rapide
        fc = model_obj.predict(pd.DataFrame({"ds": future_idx})).set_index("ds")
        core = pd.DataFrame({"Prévision": fc["yhat"],
                             "IC_lower": fc["yhat_lower"],
                             "IC_upper": fc["yhat_upper"]})

    elif model_name == "SARIMA":
        fc = model_obj.get_forecast(len(future_idx))
        core = pd.DataFrame({"Prévision": fc.predicted_mean,
                             "IC_lower": fc.conf_int(alpha=0.05).iloc[:, 0],
                             "IC_upper": fc.conf_int(alpha=0.05).iloc[:, 1]},
                            index=future_idx)

    elif model_name in ("HW_Add", "HW_Mul"):
        preds = model_obj.predict(start=future_idx[0], end=future_idx[-1])
        rmse  = model_obj.resid.std()
        core  = pd.DataFrame({"Prévision": preds,
                              "IC_lower": preds - 1.96 * rmse,
                              "IC_upper": preds + 1.96 * rmse})

    elif model_name == "S+ARIMA":
        decomp  = model_obj["decomp"]; arima = model_obj["arima"]; season = model_obj["season"]
        pred_tr = arima.predict(start=future_idx[0], end=future_idx[-1])
        seas    = np.tile(decomp.seasonal[-season:].values,
                          int(np.ceil(len(pred_tr)/season)))[:len(pred_tr)]
        preds   = pred_tr + pd.Series(seas, index=future_idx)
        rmse    = np.sqrt(preds.var())  # approximation
        core    = pd.DataFrame({"Prévision": preds,
                                "IC_lower": preds - 1.96 * rmse,
                                "IC_upper": preds + 1.96 * rmse})
    else:
        raise ValueError("Type de modèle non géré.")

    # ----- on ajoute la dernière valeur réelle pour continuité ----------
    last_val = full_series.iloc[-1]
    first_row = pd.DataFrame({"Prévision": [last_val],
                              "IC_lower": [last_val],
                              "IC_upper": [last_val]},
                             index=[last_date])
    return pd.concat([first_row, core])


# ────────────────────────────────────────────────────────────────
def render_previsions() -> None:
    st.header("🔮 Projection finale (modèle enregistré)")
    DF: pd.DataFrame = st.session_state.df
    if "saved_model_obj" not in st.session_state:
        st.warning("Aucun modèle enregistré – sélectionne d’abord un modèle dans **Choix modèle**.")
        return

    model_obj   = st.session_state["saved_model_obj"]
    model_name  = st.session_state["saved_model_name"]
    series_name = st.session_state["saved_model_series"]
    full_series = DF[series_name]

    # --- sélection de la date de fin -------------------------------------
    min_date = (full_series.index[-1] + pd.DateOffset(months=1)).to_pydatetime()
    default  = (min_date + relativedelta(years=2)).date()
    fin = st.date_input("📅 Date de fin de projection", value=default,
                        min_value=min_date.date(), key="end_date")
    horizon = _months_between(full_series.index[-1], pd.Timestamp(fin))

    if horizon <= 0:
        st.error("La date de fin doit être postérieure au dernier point de la série.")
        return

    # --- calcul forecast --------------------------------------------------
    fc_df = _forecast_with_saved(model_obj, model_name, full_series, horizon)

    # ---------- GRAPHE ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(full_series, label="Historique", color="black")
    ax.plot(fc_df["Prévision"], "--", label=f"Prévision ({model_name})")
    ax.fill_between(fc_df.index, fc_df["IC_lower"], fc_df["IC_upper"],
                    alpha=0.20, label="IC 95 %")
    ax.set_title(f"{series_name} – projection jusqu’à {fin.strftime('%b %Y')} ({model_name})")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

    # ------------------------------------------------------------------
    # ◼️ Simulation de prix indexée sur l’IHPC
    # ------------------------------------------------------------------
    if series_name.upper() == "IHPC":
        st.subheader("💶 Simulation du prix futur d’un produit")

        prix_actuel = st.number_input(
            "Prix actuel du produit (FCFA, €…)", min_value=0.0, value=0.0, step=0.5
        )

        if prix_actuel > 0:
            # ---- Projection ---------------------------------------------------
            ihpc_now  = full_series.iloc[-1]                       # IHPC courant
            prix_pred = prix_actuel * fc_df["Prévision"] / ihpc_now
            prix_df   = prix_pred.to_frame("Prix prédit")

            st.markdown(
                "**Formule appliquée** :  \n"
                r"$P_{t}\;=\;P_{0}\times \dfrac{\text{IHPC}_{t}}{\text{IHPC}_{0}}$"
            )
            dernier_prix = prix_df["Prix prédit"].iloc[-1]          # valeur
            derniere_date = prix_df.index[-1]      
            # ---- Visuel : lollipop divergeant autour du prix actuel ----------

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.vlines(
                x=prix_df.index, ymin=prix_actuel, ymax=prix_df["Prix prédit"],
                color="grey", lw=1.2, alpha=0.7
            )
            ax.scatter(prix_df.index, prix_df["Prix prédit"], s=35)
            ax.axhline(prix_actuel, color="black", ls="--",
                    label=f"Prix actuel : {prix_actuel:.2f}")
            ax.axhline(dernier_prix, color="red", ls="--",
                    label=f"Prix au {derniere_date} : {dernier_prix:.2f}")
            ax.set_ylabel("Prix projeté")
            ax.set_xlabel("Date")
            ax.set_title("Projection du prix à partir de l’IHPC")
            ax.grid(True, alpha=0.2)
            ax.legend()
            fig.autofmt_xdate()
            st.pyplot(fig)

            # ---- Export CSV ----------------------------------------------------
            st.download_button(
                "💾 Télécharger les prix projetés",
                prix_df.to_csv().encode("utf‑8"),
                file_name="prix_projection.csv",
                mime="text/csv",
            )
        else:
            st.info("➡️ Entrez un **prix actuel** pour obtenir la projection.")
            
            
    # ------------------------------------------------------------------
    # ◼️  Simulation de prix indexée sur l’indice choisi
    # ------------------------------------------------------------------
    #  ▸ On exclut les séries qui sont des variations % (Inflation) car
    #    la règle de trois n’a plus de sens en niveau de prix.
    if series_name.upper() not in {"Inflation", "InflationGliss", "IHPC"}:

        st.subheader(f"💶 Simulation du prix futur de ''{series_name}''")

        prix_actuel = st.number_input(
            f"Prix actuel d'une unité de '{series_name}' en FCFA",
            min_value=0.0, value=0.0, step=0.5
        )

        if prix_actuel > 0:
            # ---- Projection ---------------------------------------------------
            #   • obs  = série observée (train+test)  (créée plus haut)
            #   • pred = série prévisionnelle continue (train[-1] recousu + futur)
            #   • On utilise la dernière valeur réelle comme ancrage IHPC₀
            indice_0   = full_series.iloc[-1]
            prix_pred  = prix_actuel * fc_df["Prévision"]/ indice_0
            prix_df    = prix_pred.to_frame("Prix prédit")

            st.markdown(
                "**Formule appliquée** :  \n"
                r"$P_{t}\;=\;P_{0}\times \dfrac{I_{t}}{I_{0}}$  "
                "où $I$ est l’indice « " + series_name.upper() + " »."
            )

            # ---- Dernière projection (metric) ---------------------------------
            dernier_prix  = prix_df.iloc[-1, 0]
            derniere_date = prix_df.index[-1].strftime("%b %Y")
            st.metric(f"Prix projeté {derniere_date}", f"{dernier_prix:,.2f}")

            # ---- Visuel : lollipop -------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.vlines(
                x=prix_df.index, ymin=prix_actuel, ymax=prix_df["Prix prédit"],
                color="grey", lw=1.2, alpha=0.7
            )
            for date, prix in prix_df["Prix prédit"].items():
                ax.annotate(f"{prix:.0f}", xy=(date, prix),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=8, alpha=0.8)
            ax.scatter(prix_df.index, prix_df["Prix prédit"], s=35)
            ax.axhline(prix_actuel, color="black", ls="--",
                    label=f"Prix actuel : {prix_actuel:,.2f}")
            ax.axhline(dernier_prix, color="red", ls="--",
                    label=f"Prix fin horizon : {dernier_prix:,.2f}")
            ax.set_ylabel("Prix projeté")
            ax.set_xlabel("Date")
            ax.set_title(f"Projection de prix via l’indice « {series_name} »")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
            st.pyplot(fig)

            # ---- Export CSV ----------------------------------------------------
            st.download_button(
                "💾 Télécharger les prix projetés",
                prix_df.to_csv().encode("utf‑8"),
                file_name=f"prix_projection_{series_name}.csv",
                mime="text/csv",
            )
        else:
            st.info("➡️ Saisis d’abord un **prix actuel** pour lancer la projection.")
