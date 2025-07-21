import streamlit as st
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import kruskal
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm
import numpy as np
import os
import time
import shutil
import re
import unicodedata
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import tempfile
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from core.preprocessing import chainage_base, corriger_mois
from core.utils import clean_filename
from core.models import evaluate_arima, find_best_arima, find_best_sarima, evaluate_sarima, plot_stl_decomposition
from core.nomenclature import ncoa_fonction,repartition_ncoa, validate_ncoa




def render_base():
    st.title("📊 Observatoire des Prix à la Consommation – Côte d’Ivoire")
    st.caption("v2025‑07‑16")   
    st.markdown("""---Ici vous est presenté l'etat de la base que vous observez---
    """)
    df = st.session_state.df
    poste_cols=st.session_state.poste_cols
    fonction_cols=st.session_state.fonction_cols
    glob_cols = st.session_state.glob_cols

    # ➤ SECTION 1 : APERÇU DE LA BASE
    st.subheader("📁 Aperçu de la base de données")

    col1, col2, col3 = st.columns(3)
    col1.metric("🧮 Lignes", f"{df.shape[0]}")
    col2.metric("📊 Variables", f"{df.shape[1]}")
    col3.metric("🗓️ Période", f"{df.index.min().strftime('%Y-%m')} ➡ {df.index.max().strftime('%Y-%m')}")


    with st.expander("📌 Voir les colonnes disponibles"):
        st.write(df.columns.tolist())

    # ➤ SECTION 2 : DONNÉES MANQUANTES
    st.subheader("🚨 Données manquantes")
    missing = df.isna().sum().sort_values(ascending=False)
    if missing[missing > 0].empty:
        st.success("✅ Aucune valeur manquante")
    else:
        st.dataframe(missing[missing > 0])
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.heatmap(df.isna(), cbar=False, ax=ax)
        st.pyplot(fig)

    # ➤ SECTION 6 : ACCÈS BASE COMPLÈTE
    with st.expander("📄 Consulter la base complète (attention à la taille)"):
        st.dataframe(df)

 