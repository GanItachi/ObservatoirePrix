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
