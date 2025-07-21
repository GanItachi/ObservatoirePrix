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




# Fonction pour la décomposition STL
def plot_stl_decomposition(df_long, poste, period=12):
    sub = df_long[df_long['Poste'] == poste].dropna()
    if len(sub) < period * 2:
        st.warning(f"Pas assez de données pour {poste}")
        return
    sub_sorted = sub.sort_values('date')
    stl = STL(sub_sorted['Indice'], period=period, robust=True)
    res = stl.fit()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))
    ax1.plot(sub_sorted['date'], res.trend, label='Tendance')
    ax1.legend()
    ax2.plot(sub_sorted['date'], res.seasonal, label='Saisonnalité')
    ax2.legend()
    ax3.plot(sub_sorted['date'], res.resid, label='Résidu')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig)

# Fonctions utilitaires pour SARIMA et ARIMA
def evaluate_sarima(data, order, seasonal_order):
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order, 
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        return model_fit, model_fit.aic
    except:
        return None, np.inf

def find_best_sarima(data, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    orders = list(product(p_range, d_range, q_range))
    seasonal_orders = list(product(P_range, D_range, Q_range, [s]))
    for order in orders:
        for seasonal_order in seasonal_orders:
            model_fit, aic = evaluate_sarima(data, order, seasonal_order)
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_seasonal_order = seasonal_order
                best_model = model_fit
    return best_order, best_seasonal_order, best_model

def evaluate_arima(data, order):
    try:
        model = SARIMAX(data, order=order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        return model_fit, model_fit.aic
    except:
        return None, np.inf

def find_best_arima(data, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_model = None
    orders = list(product(p_range, d_range, q_range))
    for order in orders:
        model_fit, aic = evaluate_arima(data, order)
        if aic < best_aic:
            best_aic = aic
            best_order = order
            best_model = model_fit
    return best_order, best_model