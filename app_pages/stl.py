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

df = st.session_state.df

def render_stl():
    st.header("📊 Décomposition STL")
    st.write("Décomposition saisonnière, tendance, résidu.")

    