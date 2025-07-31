# ================================================================
""" …doc‑string identique + ajouts décrits ci‑dessus… """

from __future__ import annotations
import logging, time                               # ⏱️
from typing import Dict, List, Tuple

import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet                        # ≥ 1.1.5

from core.models import find_best_sarima, find_best_arima
from design import card                            # 🥇

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. compute_forecasts – avec temps d’exécution                               #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=20, ttl=3_600)
def compute_forecasts(series: pd.Series, season: int,
                      test_len: int, future_len: int) -> Dict[str, object]:
    series = (series.dropna().sort_index().asfreq("MS").interpolate("linear"))
    if len(series) < 3 * season:
        raise ValueError("Série trop courte pour la périodicité choisie.")

    split = len(series) - test_len
    train, test = series.iloc[:split], series.iloc[split:]
    future_idx  = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                                periods=future_len, freq="MS")

    preds, cis, scores = {}, {}, []

    # ====================================================================== #
    # 1️⃣  PROPHET
    # ====================================================================== #
    t0 = time.perf_counter()                        # ⏱️ start
    df_prophet = (train.rename("y").reset_index()
                        .rename(columns={train.index.name or "index": "ds"}))
    m_prophet = Prophet(); m_prophet.random_seed = 42
    m_prophet.fit(df_prophet)
    fc_df = m_prophet.predict(
        pd.DataFrame({"ds": test.index.append(future_idx)})).set_index("ds")
    exec_time = time.perf_counter() - t0            # ⏱️ end

    preds_raw = fc_df["yhat"]
    ci_raw    = fc_df[["yhat_lower", "yhat_upper"]].rename(
                   columns={"yhat_lower": "lower", "yhat_upper": "upper"})
    preds["Prophet"] = pd.concat([train.iloc[-1:], preds_raw])
    cis["Prophet"]   = pd.concat([pd.DataFrame({"lower": [train.iloc[-1]],
                                               "upper": [train.iloc[-1]]},
                                              index=train.index[-1:]), ci_raw])

    rmse = mean_squared_error(test, preds_raw.loc[test.index], squared=False)
    mape = (np.abs((test - preds_raw.loc[test.index]) / test).mean()) * 100
    mae  = mean_absolute_error(test, preds_raw.loc[test.index])
    scores.append(("Prophet", rmse, mape, mae, exec_time,
                   "add. trend", "auto", len(m_prophet.params)))

    # ====================================================================== #
    # 2️⃣  SARIMA
    # ====================================================================== #
    t0 = time.perf_counter()                        # ⏱️
    p = d = q = P = D = Q = range(0, 2)
    order, s_order, sarima = find_best_sarima(train, p, d, q, P, D, Q, season)
    fc_obj = sarima.get_forecast(steps=len(test) + future_len)
    exec_time = time.perf_counter() - t0            # ⏱️

    preds_raw = fc_obj.predicted_mean
    ci_raw    = fc_obj.conf_int(alpha=0.05).rename(columns={0: "lower", 1: "upper"})
    preds["SARIMA"] = pd.concat([train.iloc[-1:], preds_raw])
    cis["SARIMA"]   = pd.concat([pd.DataFrame({"lower": [train.iloc[-1]],
                                               "upper": [train.iloc[-1]]},
                                              index=train.index[-1:]), ci_raw])

    rmse = mean_squared_error(test, preds_raw.loc[test.index], squared=False)
    mape = (np.abs((test - preds_raw.loc[test.index]) / test).mean()) * 100
    mae  = mean_absolute_error(test, preds_raw.loc[test.index])
    scores.append(("SARIMA", rmse, mape, mae, exec_time,
                   f"{order}", f"{s_order}", sarima.params.shape[0]))

    # ====================================================================== #
    # 3️⃣  Holt‑Winters (add & mul)
    # ====================================================================== #
    if len(train) >= 2 * season:
        for name, (trend, seas) in {"HW_Add": ("add", "add"),
                                    "HW_Mul": ("add", "mul")}.items():
            try:
                serie_hw = train.dropna()
                if seas == "mul":
                    if (serie_hw <= 0).any():
                        raise ValueError("Holt-Winters multiplicatif requiert des valeurs strictement positives.")

                # Ajustement du range pour la prédiction
                all_dates = serie_hw.index.append(test.index).append(future_idx)
                all_dates = all_dates.drop_duplicates().sort_values()
                pred_start = test.index[0]
                pred_end = future_idx[-1]
                # Fit le modèle sur les données cleans
                model = ExponentialSmoothing(
                    serie_hw, seasonal_periods=season,
                    trend=trend, seasonal=seas,
                    initialization_method="estimated"
                ).fit()

                preds_raw = model.predict(start=pred_start, end=pred_end)

                # Gestion des NaN dans la prédiction
                if preds_raw.isna().any():
                    raise ValueError("NaN dans les prévisions Holt-Winters. Vérifiez la cohérence des index.")

                exec_time = time.perf_counter() - t0
                rmse = mean_squared_error(test, preds_raw.loc[test.index], squared=False)
                ci_raw = pd.DataFrame({
                    "lower": preds_raw - 1.96 * rmse,
                    "upper": preds_raw + 1.96 * rmse
                })
                preds[name] = pd.concat([train.iloc[-1:], preds_raw])
                cis[name] = pd.concat(
                    [pd.DataFrame({"lower": [train.iloc[-1]], "upper": [train.iloc[-1]]},
                                index=train.index[-1:]), ci_raw])

                mape = (np.abs((test - preds_raw.loc[test.index]) / test).mean()) * 100
                mae  = mean_absolute_error(test, preds_raw.loc[test.index])
                scores.append((name, rmse, mape, mae, exec_time,
                            trend, seas, len(model.params)))
            except Exception as exc:
                print(f"Erreur Holt-Winters ({name}): {exc}")


    else:
        if len(train) < 2 * season:
            st.warning(f"Holt-Winters non calculé : {len(train)} valeurs, minimum requis : {2 * season}")

    """
    # ====================================================================== #
    # 4️⃣  S + ARIMA
    # ====================================================================== #
    if len(train) >= 2 * season:
        t0 = time.perf_counter()                    # ⏱️
        decomp = seasonal_decompose(train, model="additive", period=season)
        trend  = decomp.trend.dropna()
        order_a, arima = find_best_arima(trend, range(0, 3), range(0, 3), range(0, 3))
        pred_trend = arima.predict(start=test.index[0], end=future_idx[-1])
        last_seas  = decomp.seasonal[-season:].values
        tiled_seas = np.tile(last_seas, int(np.ceil(len(pred_trend)/season)))[:len(pred_trend)]
        preds_raw  = pred_trend + pd.Series(tiled_seas, index=pred_trend.index)
        exec_time  = time.perf_counter() - t0       # ⏱️

        rmse = mean_squared_error(test, preds_raw.loc[test.index], squared=False)
        mape = (np.abs((test - preds_raw.loc[test.index]) / test).mean()) * 100
        mae  = mean_absolute_error(test, preds_raw.loc[test.index])
        ci_raw = pd.DataFrame({
            "lower": preds_raw - 1.96 * rmse,
            "upper": preds_raw + 1.96 * rmse})
        preds["S+ARIMA"] = pd.concat([train.iloc[-1:], preds_raw])
        cis["S+ARIMA"]   = pd.concat(
            [pd.DataFrame({"lower": [train.iloc[-1]], "upper": [train.iloc[-1]]},
                           index=train.index[-1:]), ci_raw])

        scores.append(("S+ARIMA", rmse, mape, mae, exec_time,
                       f"ARIMA{order_a}", "add. seas.", arima.params.shape[0]))
    
    """
    # ----------------------------------------------------------------------
    score_df = (pd.DataFrame(scores, columns=[
        "Modèle", "RMSE", "MAPE (%)", "MAE",
        "Durée (s)", "Tendance", "Saison", "Params"
    ]).set_index("Modèle").sort_values("RMSE"))

    return {"train": train, "test": test, "future_idx": future_idx,
            "pred": preds, "ci": cis, "scores": score_df}


# --------------------------------------------------------------------------- #
# 2. Interface Streamlit                                                     #
# --------------------------------------------------------------------------- #
def _fit_full_model(series: pd.Series, season: int, model_name: str):
    """Ré‑entraîne le modèle choisi sur toutes les données (train+test)."""
    if model_name == "Prophet":
        dfp = series.rename("y").reset_index().rename(columns={series.index.name or "index": "ds"})
        m = Prophet(); m.fit(dfp); return m
    if model_name == "SARIMA":
        p=d=q=P=D=Q=range(0,2)
        _, _, m = find_best_sarima(series, p,d,q,P,D,Q, season); return m
    if model_name in ("HW_Add", "HW_Mul"):
        trend="add"; seas="add" if model_name=="HW_Add" else "mul"
        return ExponentialSmoothing(series, seasonal_periods=season,
                                    trend=trend, seasonal=seas,
                                    initialization_method="estimated").fit()
    """
    if model_name == "S+ARIMA":
        decomp = seasonal_decompose(series, model="additive", period=season)
        trend  = decomp.trend.dropna()
        order_a, arima = find_best_arima(trend, range(0,3), range(0,3), range(0,3))
        return {"decomp": decomp, "arima": arima, "season": season}
    return None
    """


def render_choix() -> None:
    st.header("📉 Prévisions : comparez & choisissez")
    DF: pd.DataFrame = st.session_state.df             # base globale

    series_name = st.selectbox("🔎 Série cible", DF.columns)
    base = DF[series_name]
    
    # ---------------------------------------------------------------------------
    # ⛔ VERIFICATIONS PREALABLES (stationnarité, saisonnalité, données manquantes)
    # ---------------------------------------------------------------------------



    def check_conditions(series, season=12):
        conditions = {}

        # --- 1. Stationnarité (ARIMA/SARIMA → Différenciation gérée automatiquement) ---
        adf_pvalue = adfuller(series.dropna())[1]
        conditions["Stationnarité"] = ("✅ Stationnaire (ADF p ≤ 0.05)"
                                    if adf_pvalue <= 0.05
                                    else f"⚠️ Non stationnaire (p = {adf_pvalue:.3f}), "
                                            "ARIMA/SARIMA et Prophet gèrent automatiquement ce cas.")

        # --- 2. Saisonnalité (pour orienter les choix modèles) ---
        try:
            decomposition = seasonal_decompose(series.dropna(), period=season)
            seasonal_strength = 1 - decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var()
            has_seasonality = seasonal_strength > 0.1
            conditions["Saisonnalité"] = ("✅ Saisonnière détectée"
                                        if has_seasonality
                                        else "⚠️ Faible saisonnalité ou absence détectée, Holt-Winters pourrait être impacté.")
        except:
            conditions["Saisonnalité"] = "⚠️ Décomposition impossible (série trop courte)"

        # --- 3. Longueur minimale (critère bloquant obligatoire) ---
        min_length = 3 * season
        conditions["Longueur série"] = ("✅ Longueur suffisante"
                                        if len(series.dropna()) >= min_length
                                        else f"❌ Série courte (minimum {min_length} points requis).")

        # --- 4. Valeurs manquantes (gestion par interpolation) ---
        missing_count = series.isna().sum()
        conditions["Valeurs manquantes"] = ("✅ Aucune valeur manquante"
                                            if missing_count == 0
                                            else f"⚠️ {missing_count} valeurs manquantes présentes (interpolation automatique appliquée).")

        return conditions

    # Exécution des vérifications
    st.subheader("🔍 Vérifications préalables avant modélisation")
    conditions = check_conditions(base)

    for cond, msg in conditions.items():
        st.write(f"**{cond}** : {msg}")

    # Condition critique uniquement sur longueur série
    if "❌" in conditions["Longueur série"]:
        st.error("⚠️ Série trop courte. Allongez votre série ou réduisez la période saisonnière avant de modéliser.")
        st.stop()
    elif "⚠️" in conditions["Valeurs manquantes"]:
        base.interpolate("linear", inplace=True)  # correction automatique
        st.warning("➡️ Valeurs manquantes corrigées automatiquement par interpolation linéaire.")

    # Informations complémentaires non bloquantes (stationnarité, saisonnalité)
    if "⚠️" in conditions["Stationnarité"]:
        st.info("ℹ️ Information : La non-stationnarité est automatiquement gérée par ARIMA/SARIMA (différenciation) et Prophet (décomposition).")
    if "⚠️" in conditions["Saisonnalité"]:
        st.info("ℹ️ Information : En l’absence de saisonnalité claire, le modèle Holt-Winters peut être moins efficace.")


    # ------------------------ Fin des vérifications -----------------------------


    c1, c2, c3 = st.columns(3)
    with c1:
        test_len = st.slider("Mois en test", 6, 24, 12)
    with c2:
        future_len = st.slider("Horizon futur", 6, 36, 12)
    with c3:
        season = st.slider("Période saisonnière", 3, 24, 12)

    # ---------------- Calcul éventuel --------------------------------------
    if st.button("🚀 Lancer / Actualiser", key="compute_btn"):
        with st.spinner("Calcul des modèles…"):
            try:
                res = compute_forecasts(base, season, test_len, future_len)
            except Exception as exc:
                st.exception(exc)
                return
        st.session_state["fc_res"] = res
        st.session_state["chosen_model"] = res["scores"].index[0]

    if "fc_res" not in st.session_state:
        st.info("Clique sur **Lancer / Actualiser** pour démarrer la modélisation.")
        return

    res = st.session_state["fc_res"]

    # -------- Card meilleur modèle -----------------------------------------
    best_model_name = res["scores"].index[0]
    best_rmse       = res["scores"].iloc[0]["RMSE"]
    card(st, "🥇 Meilleur modèle",
         f"{best_model_name}<br/><small>RMSE {best_rmse:.2f}</small>")

    # -------- Tableau comparatif -------------------------------------------
    st.subheader("🏁 Performance des modèles (tri RMSE)")
    st.dataframe(
        res["scores"].style.format({
            "RMSE": "{:.2f}", "MAPE (%)": "{:.1f}",
            "MAE": "{:.2f}", "Durée (s)": "{:.2f}", "Params": "{:.0f}"
        })
    )

    # -------- Choix du modèle & graphe -------------------------------------
    choice = st.radio("Modèle retenu", list(res["pred"].keys()),
                      index=list(res["pred"].keys()).index(
                          st.session_state["chosen_model"])
                      )

    train, test = res["train"], res["test"]
    pred, ci    = res["pred"][choice], res["ci"][choice]

    fig, ax = plt.subplots(figsize=(12, 6))
    obs = pd.concat([train, test])                       # continuité
    ax.plot(obs, label="Observé (train+test)", color="black")
    ax.plot(pred, "--", label=f"Prévision – {choice}")
    common = ci.index.intersection(pred.index)
    ax.fill_between(common, ci.loc[common, "lower"], ci.loc[common, "upper"],
                    alpha=0.20)
    ax.axvspan(test.index[0], test.index[-1], color="grey", alpha=0.10,
               label="Période test")
    ax.set_title(f"{series_name} – modèle {choice}")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

    # -------- Enregistrement du modèle choisi ------------------------------
    if st.button("🚀 Enregistrer le modèle", key="save_btn"):
        try :
            if (st.session_state.get("saved_model_name") != choice
                    or st.session_state.get("saved_model_series") != series_name):
                st.session_state["saved_model_obj"]   = _fit_full_model(obs, season, choice)
                st.session_state["saved_model_name"]  = choice
                st.session_state["saved_model_series"]= series_name
        except :
            st.session_state["saved_model_obj"]   = _fit_full_model(obs, season, choice)
            st.session_state["saved_model_name"]  = choice
            st.session_state["saved_model_series"]= series_name
    # -------- Téléchargements ----------------------------------------------
    full_pred = pd.concat(
        {"Réel": obs, "Prévision": pred,
         "IC_lower": ci["lower"], "IC_upper": ci["upper"]}, axis=1)

    colA, colB = st.columns(2)
    with colA:
        st.download_button("💾 Comparatif modèles",
                           res["scores"].to_csv().encode("utf-8"),
                           f"{series_name}_comparatif.csv", "text/csv")
    with colB:
        st.download_button("💾 Prévision finale",
                           full_pred.to_csv().encode("utf-8"),
                           f"{series_name}_prevision_{choice}.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Point d’entrée standalone (debug)                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if "df" not in st.session_state:
        st.error("DataFrame absent → exécutez d’abord app.py")
    else:
        render_choix()
