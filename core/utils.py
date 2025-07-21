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


def clean_filename(text):
        motif_a_enlever = "INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR POSTE DE CONSOMMATION (NATIONAL) : "
        text = text.replace(motif_a_enlever, "")
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        text = re.sub(r'[\\/*?:"<>|\-,]', " ", text)
        text = re.sub(r"\s+", "_", text)
        text = text.strip("_")
        return text