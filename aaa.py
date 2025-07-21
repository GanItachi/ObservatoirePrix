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



# Configuration de Streamlit
st.set_page_config(page_title="Observatoire des Prix - EDA", layout="wide")
st.title("📊 Observatoire des Prix  de la Côte d'Ivoire")
print("--------------------------------------------------------------------------------")

repartition_ncoa = {
    # 1. Produits alimentaires et boissons non alcoolisées
    'AGRUMES': 1, 'AUTRES_FRUITS_FRAIS': 1, 'AUTRES_MATIERES_GRASSES': 1,
    'AUTRES_PRODUITS_A_BASE_DE_TUBERCULES_ET_DE_PLANTAIN': 1, 'AUTRES_CONSERVES_DE_POISSONS': 1,
    'AUTRES_PRODUITS_FRAIS_DE_MER_OU_DE_FLEUVE': 1, 'BEURRE,_MARGARINE': 1, 'BOEUF': 1,
    'CAFE,_THE,_CACAO_ET_AUTRES_VEGETAUX_POUR_TISANES': 1, 'CEREALES_NON_TRANSFORMEES': 1,
    'CHARCUTERIE_ET_CONSERVES,_AUTRES_VIANDES_ET_PREPARATIONS_A_BASE_DE_VIANDE': 1,
    'CONFITURE,_MIEL,_CHOCOLAT_ET_CONFISERIE': 1, 'FARINES,_SEMOULES_ET_GRUAUX': 1,
    'FRUITS_SECS_ET_NOIX': 1, 'HUILES': 1, 'LAIT': 1, 'LAITS_INFANTILES_ET_FARINES_LACTEES_POUR_BEBE': 1,
    'LEGUMES_FRAIS_EN_FEUILLES': 1, 'LEGUMES_FRAIS_EN_FRUITS_OU_RACINE': 1,
    'LEGUMES_SECS_ET_OLEAGINEUX': 1, 'MOUTON_-_CHEVRE': 1, 'OEUFS': 1, 'PAINS': 1,
    'PATES_ALIMENTAIRES': 1, 'PATISSERIES,_GATEAUX,_BISCUITS,_VIENOISERIES': 1,
    'POISSONS_ET_AUTRES_PRODUITS_SECHES_OU_FUMES': 1, 'POISSONS_FRAIS': 1, 'PORC': 1,
    'PRODUITS_LAITIERS': 1, 'SEL,_EPICES,_SAUCES_ET_PRODUITS_ALIMENTAIRES_N.D.A': 1,
    'SUCRE': 1, 'TUBERCULES_ET_PLANTAIN': 1, 'VOLAILLE': 1,

    # 2. Boissons alcoolisées, tabac et stupéfiants
    'ALCOOL_DE_BOUCHE': 2, 'BIERE': 2, 'TABAC_ET_STUPEFIANTS': 2, 'VIN_ET_BOISSONS_FERMENTEES': 2,

    # 3. Articles d'habillement et chaussures
    'AUTRES_ARTICLES_VESTIMENTAIRES_ET_ACCESSOIRES_D\'HABILLEMENT': 3,
    'CONFECTION_ET_REPARATIONS_VETEMENTS_ENFANTS': 3,
    'CONFECTION_ET_REPARATIONS_VETEMENTS_FEMMES': 3,
    'CONFECTION_ET_REPARATIONS_VETEMENTS_HOMMES': 3, 'NETTOYAGE_ET_BLANCHISSAGE_DES_VETEMENTS': 3,
    'REPARATION_ET_LOCATION_D\'ARTICLES_CHAUSSANTS': 3, 'SOUS-VETEMENTS_FEMMES': 3,
    'SOUS-VETEMENTS_HOMMES': 3, 'TENUES_SCOLAIRES': 3, 'TISSUS_D\'HABILLEMENT': 3,
    'VETEMENTS_DE_DESSUS_FEMMES': 3, 'VETEMENTS_DE_DESSUS_HOMMES': 3,
    'VETEMENTS_ENFANTS_(3_A_13_ANS)_ET_NOURRISSONS_(0_A_2_ANS)': 3,

    # 4. Logement, eau, gaz, électricité et autres combustibles
    'LOYERS_EFFECTIFS_DES_LOCATAIRES_ET_SOUS-LOCATAIRES': 4,
    'COMBUSTIBLES_LIQUIDES': 4, 'COMBUSTIBLES_SOLIDES_ET_AUTRES': 4,
    "SERVICES_POUR_L'HABITATION_SAUF_SERVICES_DOMESTIQUES": 4,

    # 5. Meubles, articles de ménage et entretien courant du foyer
    'PRODUITS_POUR_L\'ENTRETIEN_ET_REPARATION_COURANTE': 5,
    'OUTILLAGE,_MATERIEL_ET_ACCESSOIRES_DIVERS': 5,

    # 6. Santé
    'APPAREILS_ET_MATERIEL_THERAPEUTIQUES': 6, 'MEDICAMENTS_MODERNES': 6,
    'MEDICAMENTS_TRADITIONNELS': 6, 'PRODUITS_MEDICAUX_DIVERS': 6,
    'SERVICES_DES_AUXILIAIRES_MEDICAUX': 6, 'SERVICES_HOSPITALIERS': 6,
    'SERVICES_DE_LABORATOIRES_ET_DE_RADIOLOGIE': 6,
    'SERVICES_MEDICAUX_ET_DENTAIRES': 6, 'SANTE': 6,

    # 7. Transports
    'AUTOMOBILE': 7, 'CARBURANTS_ET_LUBRIFIANTS': 7,
    'ENTRETIEN_ET_REPARATIONS_DE_VEHICULES_PARTICULIERS': 7,
    'ASSURANCE_TRANSPORT': 7, 'CYCLE,_MOTOCYCLE_ET_VEHICULE_A_TRACTION_ANIMALE': 7,
    'PIECES_DETACHEES_ET_ACCESSOIRES': 7,
    'TRANSPORT_AERIEN_DE_PASSAGERS': 7, 'TRANSPORT_ROUTIER_DE_PASSAGERS': 7,
    'AUTRES_SERVICES_RELATIFS_AUX_VEHICULES_PERSONNELS': 7,

    # 8. Communication
    'COMMUNICATION_TELEPHONIQUE': 8, 'FRAIS_DE_CONNEXION_INTERNET_ET_ASSIMILES': 8,
    'COMMUNICATION': 8, 'MATERIEL_DE_TELEPHONIE_ET_DE_TELECOPIE': 8,

    # 9. Loisirs et culture
    'APPAREILS_DE_RECEPTION,_ENREGISTREMENT_ET_REPRODUCTION': 9,
    'EQUIPEMENT_PHOTOGRAPHIQUE,_CINEMATOGRAPHIQUE,_OPTIQUE,_AUTRE_BIEN_DURABLE_A_FONCTION_RECREATIVE_ET_CULTURELLE': 9,
    'SERVICES_CULTURELS,_RECREATIFS_ET_SPORTIFS': 9,
    'JEUX_ET_JOUETS,_PASSE-TEMPS_ET_ARTICLES_DE_SPORT,_MATERIEL_C': 9,
    'LIVRES_SCOLAIRES_ET_AUTRES_LIVRES': 9,
    'JOURNAUX_ET_PUBLICATIONS_PERIODIQUES': 9,
    'JEUX_DE_HASARD': 9,

    # 10. Enseignement
    'ENSEIGNEMENT_NON_DEFINI_PAR_NIVEAU': 10, 'ENSEIGNEMENT_PRE-ELEMENTAIRE_ET_PRIMAIRE': 10,
    'ENSEIGNEMENT_SECONDAIRE': 10, 'ENSEIGNEMENT_SUPERIEUR': 10,
    'ENSEIGNEMENT_POST-SECONDAIRE_NON_SUPERIEUR': 10,

    # 11. Restaurants et hôtels
    'RESTAURANTS,_CAFES_ET_ETABLISSEMENTS_SIMILAIRES': 11,
    "HOTELS_ET_AUTRES_SERVICES_D'HEBERGEMENT": 11, 'CANTINES': 11,

    # 12. Biens et services divers
    'APPAREILS_ET_ARTICLES_POUR_SOINS_CORPORELS': 12,
    'PRODUITS_POUR_SOINS_CORPORELS': 12,
    'SALONS_DE_COIFFURE_ET_INSTITUTS_DE_SOINS_ET_DE_BEAUTE': 12,
    'SERVICES_DOMESTIQUES': 12,
    'SERVICES_FINANCIERS': 12,
    "ARTICLES_DE_BIJOUTERIE_ET_D'HORLOGERIE": 12,
    'PAPETERIE_ET_IMPRIMES_DIVERS': 12,
    'PROTECTION_SOCIALE_ET_AUTRES_ASSURANCES': 12,
    'FORFAIT_ET_CIRCUITS_TOURISTIQUES_COMPOSITES': 12,
    'AUTRES_EFFETS_PERSONNELS': 12,
    'PRODUITS_POUR_JARDINS,_PLANTES_ET_FLEURS,_ANIMAUX_DE_COMPAGNIE_ET_ARTICLES_CONNEXES_ET_AUTRES_SERVICES_POUR_ANIMAUX_DE_COMPAGNIE': 12,
    'SERVICES_D\'ENTRETIEN_ET_DE_REPARATIONS_COURANTES': 12,
    'AUTRES_SERVICES_N.C.A.': 12,
    "MATERIEL_DE_TRAITEMENT_DE_L'INFORMATION_ET_SUPPORTS_D'ENREGISTREMENT_DE_L'IMAGE_ET_DU_SON": 12
}

ncoa_fonction = {
1:"PRODUITS_ALIMENTAIRES_ET_BOISSONS_NON_ALCOOLISEES",
2:"BOISSONS_ALCOOLISEES_TABAC_ET_STUPEFIANTS",
3:"ARTICLES_D_HABILLEMENT_ET_CHAUSSURES",
4:"LOGEMENT_EAU_GAZ_ELECTRICITE_ET_AUTRES_COMBUSTIBLES",
5:"MEUBLES_ARTICLES_DE_MENAGE_ET_ENTRETIEN_COURANT_DU_FOYER",
6:"SANTE",
7:"TRANSPORTS",
8:"COMMUNICATION",
9:"LOISIRS_ET_CULTURE",
10:"ENSEIGNEMENT",
11:"RESTAURANTS_ET_HOTELS",
12:"BIENS_ET_SERVICES_DIVERS"
}

# Fonction pour corriger les mois
def corriger_mois(df, colonne_mois, colonne_annee):
    df = df.copy()
    mois_col = df[colonne_mois].tolist()
    annee_col = df[colonne_annee].tolist()
    mois_corriges = []
    for i in range(len(mois_col)):
        m = mois_col[i]
        if m != 0:
            mois_corriges.append(m)
        else:
            prev = mois_corriges[-1] if i > 0 else None
            next_ = mois_col[i+1] if i + 1 < len(mois_col) else None
            if prev is not None and next_ is not None and next_ - prev == 2:
                mois_corriges.append(prev + 1)
            else:
                mois_corriges.append(0)
    mois_corriges = [12 if m == 0 else m for m in mois_corriges]
    df[colonne_mois] = mois_corriges
    return df

def chainage_base(df, date_pivot_str='2019-01-01'):
    """
    Chaine les séries du DataFrame `df` autour de la date_pivot.
    df doit avoir un index de type datetime.

    Paramètres :
    - df : DataFrame avec des colonnes de séries temporelles
    - date_pivot_str : chaîne de caractères format 'YYYY-MM-DD'

    Retour :
    - df_chaine : DataFrame chaîné
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("L'index du DataFrame doit être de type datetime.")

    date_pivot = pd.to_datetime(date_pivot_str)
    date_avant = date_pivot - pd.DateOffset(months=1)

    df_chaine = df.copy()

    for col in df.columns:
        try:
            val_new = df.loc[date_pivot, col]
            val_old = df.loc[date_avant, col]
            if pd.notna(val_old) and val_old != 0:
                coeff = val_new / val_old
                df_chaine.loc[df.index < date_pivot, col] = df.loc[df.index < date_pivot, col] * coeff
            else:
                print(f"⚠️ Coefficient non calculé pour '{col}' (val_old = {val_old})")
        except KeyError:
            print(f"⛔️ Dates manquantes pour '{col}' — chaînage impossible")
            continue

    return df_chaine


# Charger les données
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "Base_fin.csv")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['annee_deb_couv'].astype(str) + '-' + df['mois_deb_couv'].astype(str).str.zfill(2) + '-01')
    df = df[df['date'] <= '2024-10'].sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    return df

df = load_data()

st.markdown("""
    Bienvenue dans l’interface de visualisation des indices des prix à la consommation !
    """)
st.success("📅 Dernière mise à jour : " + df.index.max().strftime("%B %Y"))
if st.button("Chainer la base"):
    df = chainage_base(df)


# Définir les colonnes par catégorie
fonc = [
    'LOISIRS_ET_CULTURE', 'PRODUITS_ALIMENTAIRES_ET_BOISSONS_NON_ALCOOLISEES',
    'BOISSONS_ALCOOLISEES_TABAC_ET_STUPEFIANTS', 'COMMUNICATION', 'ENSEIGNEMENT',
    'RESTAURANTS_ET_HOTELS', 'BIENS_ET_SERVICES_DIVERS', 'SANTE', 'TRANSPORTS'
]
glob = ['Inflation', 'InflationGliss', 'IHPC']
poste_cols = [col for col in df.columns if col not in fonc + glob + ['annee_deb_couv', 'mois_deb_couv', 'annee_fin_couv', 'mois_fin_couv']]
fonction_cols = fonc
glob_cols = glob

# Sidebar pour la navigation
st.sidebar.header("Options de Visualisation")
analysis_type = st.sidebar.selectbox(
    "Choisir une analyse",
    ["Aperçu des Données", "Évolution des Indices", "Heatmap des Indices", "Corrélations", 
     "Décomposition STL", "Saisonnalité", "Scraping et Création de Base", "Prévisions des Séries Temporelles"]
)



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

# 1. Aperçu des Données
if analysis_type == "Aperçu des Données":
    st.header("Aperçu des Données")
    st.write(f"**Dimensions** : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    st.write(f"**Période couverte** : {df.index.min().strftime('%Y-%m')} à {df.index.max().strftime('%Y-%m')}")
    st.write("**Colonnes disponibles** :")
    st.write(df.columns.tolist())
    st.write("**Valeurs manquantes** :")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing[missing > 0])
    st.write("**Heatmap des valeurs manquantes** :")
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    st.pyplot(fig)

    st.write("BASE UTILISEE :")
    st.dataframe(df)

# 2. Évolution des Indices
elif analysis_type == "Évolution des Indices":
    
    
    # Choix de la fonction NCOA
    fonction_choisie_label = st.selectbox(
        "Choisir une fonction de consommation (NCOA)", 
        options=list(ncoa_fonction.values())
    )

    # Récupérer le numéro de la fonction choisie
    fonction_choisie_num = [k for k, v in ncoa_fonction.items() if v == fonction_choisie_label][0]

    # Filtrer les postes correspondants
    postes_correspondants = [poste for poste, fct in repartition_ncoa.items() if fct == fonction_choisie_num]
    postes_correspondants.sort()

    # Choix du poste dans la fonction
    poste_choisi = st.selectbox(
        "Choisir un poste de consommation dans cette fonction", 
        options=postes_correspondants
    )

    # Résultat affiché
    st.success(f"Vous avez choisi : **{poste_choisi}** dans la fonction **{fonction_choisie_label}**")
    
    st.title(f"📈 Representation {fonction_choisie_label} ")
    st.line_chart(df[fonction_choisie_label].dropna())
    st.markdown("### 📥 Télécharger les données")
    csv = df[[fonction_choisie_label]].dropna().reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger en CSV la fonction", csv, f"{fonction_choisie_label}.csv", "text/csv")
    
    st.title(f"📈 Representation {poste_choisi} ")
    st.line_chart(df[poste_choisi].dropna())
    st.markdown("### 📥 Télécharger les données")
    csv = df[[poste_choisi]].dropna().reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(f"Télécharger en CSV du poste {poste_choisi}", csv, f"{poste_choisi}.csv", "text/csv")
    
    st.header("Comparaisons")
    index_type = st.selectbox("Choisir le type d'indice", ["Globaux", "Par Fonction", "Par Poste"])
    if index_type == "Globaux":
        selected_cols = st.multiselect("Sélectionner les indices globaux", glob_cols, default=glob_cols)
        fig, ax = plt.subplots(figsize=(20, 8))
        for col in selected_cols:
            ax.plot(df.index, df[col], label=col)
        ax.set_title("Évolution des indices globaux")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        st.pyplot(fig)
    elif index_type == "Par Fonction":
        selected_cols = st.multiselect("Sélectionner les fonctions", fonction_cols, default=fonction_cols[:3])
        fig, ax = plt.subplots(figsize=(20, 8))
        for col in selected_cols:
            ax.plot(df.index, df[col], label=col)
        ax.set_title("Évolution des indices par fonction")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        st.pyplot(fig)
    elif index_type == "Par Poste":
        selected_cols = st.multiselect("Sélectionner les postes", poste_cols, default=poste_cols[:5])
        fig, ax = plt.subplots(figsize=(20, 8))
        for col in selected_cols:
            ax.plot(df.index, df[col], label=col)
        ax.set_title("Évolution des indices par poste")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        st.pyplot(fig)

# 3. Heatmap des Indices
elif analysis_type == "Heatmap des Indices":
    st.header("Heatmap des Indices de Prix")
    heatmap_type = st.selectbox("Choisir le type de heatmap", ["Par Poste", "Par Fonction"])
    if heatmap_type == "Par Poste":
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(df[poste_cols].T, cmap="YlGnBu", cbar_kws={'label': 'Indice de prix'}, ax=ax)
        ax.set_title("Heatmap des indices de prix par poste")
        ax.set_xlabel("Date")
        ax.set_ylabel("Poste")
        plt.tight_layout()
        st.pyplot(fig)
    elif heatmap_type == "Par Fonction":
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(df[fonction_cols].T, cmap="YlGnBu", cbar_kws={'label': 'Indice de prix'}, ax=ax)
        ax.set_title("Heatmap des indices de prix par fonction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Poste")
        plt.tight_layout()
        st.pyplot(fig)

# 4. Corrélations
elif analysis_type == "Corrélations":
    st.header("Analyse des Corrélations")
    corr_type = st.selectbox("Choisir le type de corrélation", ["Entre Postes", "Avec INFLATION", "Avec IHPC"])
    if corr_type == "Entre Postes":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[poste_cols].corr(), cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Corrélation entre indices de prix (postes)")
        st.pyplot(fig)
    elif corr_type == "Avec INFLATION":
        correlation_with_inflation = df[poste_cols + ['INFLATION']].corr()['INFLATION']
        correlation_sorted = correlation_with_inflation.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(3, len(correlation_sorted) * 0.25))
        sns.heatmap(correlation_sorted.to_frame(), annot=True, cmap="coolwarm", cbar_kws={'label': 'Corrélation'}, yticklabels=correlation_sorted.index, ax=ax)
        ax.set_title("Corrélation à INFLATION")
        ax.set_xlabel("INFLATION")
        ax.set_ylabel("Poste")
        plt.tight_layout()
        st.pyplot(fig)
        st.write("🔝 Top 5 positifs corrélés à INFLATION :")
        st.write(correlation_sorted.tail(5))
        st.write("🔻 Top 5 négatifs corrélés à INFLATION :")
        st.write(correlation_sorted.head(5))
    elif corr_type == "Avec IHPC":
        correlation_with_ihpc = df[poste_cols + ['IHPC']].corr()['IHPC']
        correlation_sorted = correlation_with_ihpc.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(3, len(correlation_sorted) * 0.25))
        sns.heatmap(correlation_sorted.to_frame(), annot=True, cmap="coolwarm", cbar_kws={'label': 'Corrélation'}, yticklabels=correlation_sorted.index, ax=ax)
        ax.set_title("Corrélation à IHPC")
        ax.set_xlabel("IHPC")
        ax.set_ylabel("Poste")
        plt.tight_layout()
        st.pyplot(fig)
        st.write("🔝 Top 5 positifs corrélés à IHPC :")
        st.write(correlation_sorted.tail(5))
        st.write("🔻 Top 5 négatifs corrélés à IHPC :")
        st.write(correlation_sorted.head(5))

# 5. Décomposition STL
elif analysis_type == "Décomposition STL":
    st.header("Décomposition STL des Séries Temporelles")
    poste = st.selectbox("Sélectionner un poste", poste_cols)
    df_l = df.melt(ignore_index=False, var_name='Poste', value_name='Indice').reset_index()
    plot_stl_decomposition(df_l, poste)

# 6. Saisonnalité
elif analysis_type == "Saisonnalité":
    st.header("Analyse de la Saisonnalité")
    df_l = df.melt(ignore_index=False, var_name='Poste', value_name='Indice').reset_index()
    resultats = []
    df_l['mois'] = df_l['date'].dt.month
    for poste in df_l['Poste'].unique():
        sub = df_l[df_l['Poste'] == poste].dropna()
        if len(sub) < 24:
            continue
        groupes = [g['Indice'].values for _, g in sub.groupby('mois')]
        if len(groupes) == 12:
            stat, pval = kruskal(*groupes)
        else:
            pval = np.nan
        resultats.append({
            'Poste': poste,
            'p-value_Kruskal': pval,
            'Saisonnalité': pval < 0.05 if not np.isnan(pval) else False
        })
    saison_df = pd.DataFrame(resultats).sort_values(by='Saisonnalité', ascending=False)
    st.write("Résultats de la détection de saisonnalité (Kruskal-Wallis) :")
    st.dataframe(saison_df)
    saisonnalite_scores = {}
    for poste in df.columns:
        try:
            stl = STL(df[poste].dropna(), period=12)
            res = stl.fit()
            var_resid = res.resid.var()
            var_total = df[poste].var()
            saisonnalite_scores[poste] = 1 - (var_resid / var_total)
        except:
            saisonnalite_scores[poste] = None
    s = pd.Series(saisonnalite_scores).dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(20, len(s) * 0.3))
    s.plot(kind='barh', ax=ax)
    ax.set_title('Force de la saisonnalité par poste')
    ax.set_xlabel('Score (1 = forte saisonnalité)')
    plt.tight_layout()
    st.pyplot(fig)

# 7. Scraping et Création de Base
elif analysis_type == "Scraping et Création de Base":
    st.header("Scraping et Création de la Base de Données")
    # Fonction de nettoyage des noms de fichiers
    def clean_filename(text):
        motif_a_enlever = "INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR POSTE DE CONSOMMATION (NATIONAL) : "
        text = text.replace(motif_a_enlever, "")
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        text = re.sub(r'[\\/*?:"<>|\-,]', " ", text)
        text = re.sub(r"\s+", "_", text)
        text = text.strip("_")
        return text
    
    if st.button("Lancer le Scraping"):
        st.write("Démarrage du processus de scraping...")
        download_dir = os.path.join(os.getcwd(), "telechargements_temp")
        os.makedirs(download_dir, exist_ok=True)
        dossier_final = os.path.join(os.getcwd(), "excels_anstat_new")
        os.makedirs(dossier_final, exist_ok=True)
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # ✅ Créer un user-data-dir temporaire pour éviter les conflits
        unique_profile = tempfile.mkdtemp(prefix="selenium-profile-")
        chrome_options.add_argument(f"--user-data-dir={unique_profile}")
        
        
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        driver = webdriver.Chrome(options=chrome_options)
        url = "https://www.anstat.ci/publication-details/aef5bf59fb787e047589bc63ba61dc0ccec8a173aa7bd8f14c297f90343ab2db8de90b413ba6a8080fefcbf525818c6f2a5169bcc441885f5530ab75bdd1e59faEGl_I_t9luulwuSsV_1hIVtwzoniR4qxyqa0gbhhLY"
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "specification-tab"))).click()
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "toggle-btn"))).click()
        time.sleep(2)
        divs = driver.find_elements(By.CSS_SELECTOR, "div.flex-fill.w-80.indicateur-item")
        liens_et_textes = []
        for div in divs:
            try:
                a_tag = div.find_element(By.TAG_NAME, "a")
                href = a_tag.get_attribute("href")
                texte = a_tag.text.strip()
                if href:
                    liens_et_textes.append((texte, href))
            except:
                pass
        transformed_data = []
        for texte, lien in liens_et_textes:
            parts = texte.split('. ', 1)
            if len(parts) > 1:
                col1 = parts[0]
                remaining = parts[1].split(': ', 1)
                if len(remaining) > 1:
                    col2 = remaining[0].strip()
                    col3 = remaining[1].strip()
                    col4 = lien
                else:
                    col2 = parts[1].strip()
                    col3 = ""
                    col4 = lien
            else:
                parts_colon = texte.split(': ', 1)
                if len(parts_colon) > 1:
                    col1 = ""
                    col2 = parts_colon[0].strip()
                    col3 = parts_colon[1].strip()
                    col4 = lien
                else:
                    col1 = ""
                    col2 = texte.strip()
                    col3 = ""
                    col4 = lien
            if col2 == "INDICE HARMONISÉ DES PRIX À LA CONSOMMATION (NATIONAL)":
                col3_transformed = "IHPC"
            elif col2 == "INFLATION EN GLISSEMENT EN MOYENNE ANNUELLE (NATIONALE)":
                col3_transformed = "InflationGliss"
            elif col2 == "INFLATION EN MOYENNE MENSUELLE (NATIONALE)":
                col3_transformed = "Inflation"
            else:
                col3_transformed = col3
            transformed_data.append([col1, col2, col3_transformed, col4])
        csv_filename = "liens_et_textes.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in transformed_data:
                writer.writerow(row)
        st.write(f"✅ Fichier {csv_filename} créé avec succès.")
        df_list = pd.read_csv(csv_filename, header=None, engine='python')
        df_list[4] = df_list[1].apply(
            lambda x: 'Poste' if x == 'INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR POSTE DE CONSOMMATION (NATIONAL)'
            else 'Fonction' if x == 'INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR FONCTION DE CONSOMMATION (NATIONAL)'
            else 'Autre'
        )
        st.write("Liens extraits :")
        st.dataframe(df_list)
        bon = [
            'INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR POSTE DE CONSOMMATION (NATIONAL)',
            'INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR FONCTION DE CONSOMMATION (NATIONAL)',
            'INDICE HARMONISÉ DES PRIX À LA CONSOMMATION (NATIONAL)',
            'INFLATION EN GLISSEMENT EN MOYENNE ANNUELLE (NATIONALE)',
            'INFLATION EN MOYENNE MENSUELLE (NATIONALE)'
        ]
        for texte, lien, i in zip(df_list[2], df_list[3], df_list[1]):
            if i in bon:
                st.write(f"Téléchargement de : {texte}")
                driver.get(lien)
                try:
                    bouton_dl = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "btnDownloadData")))
                    bouton_dl.click()
                    time.sleep(5)
                    fichiers = os.listdir(download_dir)
                    fichiers_csv = [f for f in fichiers if f.endswith('.csv')]
                    if fichiers_csv:
                        dernier = max([os.path.join(download_dir, f) for f in fichiers_csv], key=os.path.getctime)
                        nom_fichier_propre = clean_filename(texte) + ".csv"
                        chemin_final = os.path.join(dossier_final, nom_fichier_propre)
                        shutil.move(dernier, chemin_final)
                        st.write(f"✅ Fichier rangé : {chemin_final}")
                    else:
                        st.write(f"⚠️ Aucun fichier .csv détecté pour : {texte}")
                except Exception as e:
                    st.write(f"❌ Erreur sur {texte} → {e}")
            else:
                st.write(f"Rejet de : {texte} car {i}")
        driver.quit()
               # Fusion des fichiers
    if st.button("Fusionner"):
        dossier_final = os.path.join(os.getcwd(), "excels_anstat_new")
        os.makedirs(dossier_final, exist_ok=True)
        st.write("Fusion des fichiers téléchargés...")
        fichiers = [f for f in os.listdir(dossier_final) if f.endswith(".csv")]
        data_all = []
        for fichier in fichiers:
            chemin = os.path.join(dossier_final, fichier)
            try:
                df_temp = pd.read_csv(chemin)
                colonnes_requises = {"annee_deb_couv", "mois_deb_couv", "annee_fin_couv", "mois_fin_couv", "valeur"}
                if not colonnes_requises.issubset(df_temp.columns):
                    st.write(f"⚠️ Fichier ignoré (colonnes manquantes) : {fichier}")
                    continue
                df_temp = df_temp[list(colonnes_requises)].copy()
                df_temp["annee_deb_couv"] = df_temp["annee_deb_couv"].astype(int)
                df_temp["mois_deb_couv"] = df_temp["mois_deb_couv"].astype(int)
                df_temp["annee_fin_couv"] = df_temp["annee_fin_couv"].astype(int)
                df_temp["mois_fin_couv"] = df_temp["mois_fin_couv"].astype(int)
                df_temp["valeur"] = pd.to_numeric(df_temp["valeur"], errors="coerce")
                nom_indicateur = os.path.splitext(fichier)[0]
                df_temp["nom_indicateur"] = nom_indicateur
                data_all.append(df_temp)
            except Exception as e:
                st.write(f"❌ Erreur fichier {fichier} → {e}")
        
        if data_all:
            df_long = pd.concat(data_all, ignore_index=True)
            new_df = corriger_mois(df_long, 'mois_deb_couv', 'annee_deb_couv')
            new_df = corriger_mois(new_df, 'mois_fin_couv', 'annee_fin_couv')
            df_wide = new_df.pivot_table(
                index=["annee_deb_couv", "mois_deb_couv", "annee_fin_couv", "mois_fin_couv"],
                columns="nom_indicateur",
                values="valeur"
            ).reset_index()
            os.makedirs("data", exist_ok=True)
            df_wide.to_csv("data/Base.csv", index=False, encoding="utf-8-sig")
            Base_fin = df_wide[df_wide['annee_deb_couv'] > 2016]
            Base_fin.to_csv("data/Base_fin.csv", index=False, encoding="utf-8-sig")
            df_long.to_csv("data/Base_long.csv", index=False, encoding="utf-8-sig")
            st.write("✅ Fusion terminée avec succès : 'data/Base.csv', 'data/Base_fin.csv', 'data/Base_long.csv'")
        else:
            st.write("⚠️ Aucun fichier exploitable.")

# 8. Prévisions des Séries Temporelles
elif analysis_type == "Prévisions des Séries Temporelles":
    st.header("Prévisions des Séries Temporelles")
    
    # Sélection de la série temporelle
    series_options = df.columns.tolist()
    selected_series = st.selectbox("Sélectionner une série temporelle", series_options, 
                                   index=series_options.index('CAFE,_THE,_CACAO_ET_AUTRES_VEGETAUX_POUR_TISANES') if 'CAFE,_THE,_CACAO_ET_AUTRES_VEGETAUX_POUR_TISANES' in series_options else 0)
    
    # Paramètres personnalisables
    forecast_start_offset = st.slider("Nombre de mois pour la période de test", min_value=1, max_value=24, value=12)
    future_periods = st.slider("Nombre de mois à prédire dans le futur", min_value=1, max_value=36, value=12)
    s = st.slider("Période saisonnière (en mois)", min_value=1, max_value=24, value=12)
    
    if st.button("Lancer l'analyse des prévisions"):
        # Préparation des données
        base = df[selected_series].dropna()
        name = selected_series
        
        # Diviser les données en ensembles d'entraînement et de test
        if len(base) <= forecast_start_offset:
            st.error("La série temporelle est trop courte pour la période de test spécifiée.")
        else:
            split_point = len(base) - forecast_start_offset
            train_data = base[:split_point]
            test_data = base[split_point:]
            future_forecast_index = pd.date_range(start=base.index[-1] + pd.DateOffset(months=1), periods=future_periods, freq='MS')
            
            # Modèle Prophet
            st.subheader("Modèle Prophet")
            df_prophet_train = train_data.reset_index()
            df_prophet_train.columns = ['ds', 'y']
            model_prophet = Prophet()
            model_prophet.fit(df_prophet_train)
            future_prophet_dates = pd.concat([pd.DataFrame({'ds': test_data.index}), 
                                            pd.DataFrame({'ds': future_forecast_index})]).reset_index(drop=True)
            future_prophet = future_prophet_dates
            forecast_prophet = model_prophet.predict(future_prophet)
            prophet_predictions = forecast_prophet.set_index('ds')['yhat'].loc[test_data.index[0]:future_forecast_index[-1]]
            prophet_predictions_lower = forecast_prophet.set_index('ds')['yhat_lower'].loc[test_data.index[0]:future_forecast_index[-1]]
            prophet_predictions_upper = forecast_prophet.set_index('ds')['yhat_upper'].loc[test_data.index[0]:future_forecast_index[-1]]
            
            # Modèle SARIMA
            st.subheader("Modèle SARIMA")
            p = d = q = P = D = Q = range(0, 2)
            best_order_sarima, best_seasonal_order_sarima, best_sarima_model = find_best_sarima(
                train_data, p, d, q, P, D, Q, s
            )
            start_date_sarima = test_data.index[0]
            end_date_sarima = future_forecast_index[-1]
            sarima_predictions = best_sarima_model.predict(start=start_date_sarima, end=end_date_sarima)
            
            #Modèle Holt-Winters additif
            try:
                model_hw_add = ExponentialSmoothing(
                    train_data,
                    seasonal_periods=s,
                    trend='add',
                    seasonal='add',
                    use_boxcox=True,
                    initialization_method="estimated"
                ).fit()
                hw_add_predictions = model_hw_add.predict(start=test_data.index[0], end=future_forecast_index[-1])
            except Exception as e:
                st.error(f"Erreur Holt-Winters Additif : {e}")
                hw_add_predictions = pd.Series(np.nan, index=pd.date_range(start=test_data.index[0], end=future_forecast_index[-1], freq='MS'))

            # Modèle Holt-Winters Multiplicatif
            try:
                model_hw_mul = ExponentialSmoothing(
                    train_data,
                    seasonal_periods=s,
                    trend='add',
                    seasonal='mul',
                    use_boxcox=True,
                    initialization_method="estimated"
                ).fit()
                hw_mul_predictions = model_hw_mul.predict(start=test_data.index[0], end=future_forecast_index[-1])
            except Exception as e:
                st.error(f"Erreur Holt-Winters Multiplicatif : {e}")
                hw_mul_predictions = pd.Series(np.nan, index=pd.date_range(start=test_data.index[0], end=future_forecast_index[-1], freq='MS'))
            
            # Choisir le meilleur modèle Holt-Winters basé sur le RMSE sur la période de test (si les ajustements ont réussi)
            rmse_hw_add = np.inf
            rmse_hw_mul = np.inf
            best_hw_predictions = pd.Series(np.nan)
            best_hw_name = "Aucun (Ajustement échoué)"
            
            if not hw_add_predictions.isnull().all():
                hw_add_test_predictions_eval = hw_add_predictions.loc[test_data.index]
                if not hw_add_test_predictions_eval.isnull().all():
                    rmse_hw_add = np.sqrt(mean_squared_error(test_data.dropna(), hw_add_test_predictions_eval.dropna())) # Gérer les NaN si présents

            if not hw_mul_predictions.isnull().all():
                hw_mul_test_predictions_eval = hw_mul_predictions.loc[test_data.index]
                if not hw_mul_test_predictions_eval.isnull().all():
                    rmse_hw_mul = np.sqrt(mean_squared_error(test_data.dropna(), hw_mul_test_predictions_eval.dropna()))

            print(f"\n--- Évaluation préliminaire de Holt-Winters sur la période de test ---")
            print(f"RMSE Holt-Winters Additif : {rmse_hw_add:.2f}" if rmse_hw_add != np.inf else "RMSE Holt-Winters Additif : N/A (Échec de l'ajustement)")
            print(f"RMSE Holt-Winters Multiplicatif : {rmse_hw_mul:.2f}" if rmse_hw_mul != np.inf else "RMSE Holt-Winters Multiplicatif : N/A (Échec de l'ajustement)")


            if rmse_hw_add < rmse_hw_mul:
                best_hw_predictions = hw_add_predictions
                best_hw_name = "Holt-Winters Additif"
                best_hw_rmse = rmse_hw_add
            elif rmse_hw_mul < rmse_hw_add:
                best_hw_predictions = hw_mul_predictions
                best_hw_name = "Holt-Winters Multiplicatif"
                best_hw_rmse = rmse_hw_mul
            else:
                # Si les deux RMSE sont infinis ou égaux, aucun modèle n'a réussi ou ils sont équivalents (et échoués)
                best_hw_rmse = np.inf
            
            
            
            # Modèle S+ARIMA
            st.subheader("Modèle S+ARIMA")
            decomposition_train = seasonal_decompose(train_data, model='additive', period=s)
            trend_train = decomposition_train.trend.dropna()
            p_arima = d_arima = q_arima = range(0, 3)
            best_arima_order, best_arima_model = find_best_arima(trend_train, p_arima, d_arima, q_arima)
            start_date_trend = test_data.index[0]
            end_date_trend = future_forecast_index[-1]
            future_forecast_trend = best_arima_model.predict(start=start_date_trend, end=end_date_trend)
            seasonal_component_train = decomposition_train.seasonal.dropna()
            last_seasonal_train = seasonal_component_train[-s:].values
            future_seasonal_index = future_forecast_trend.index
            num_tiles = len(future_seasonal_index) // s + (len(future_seasonal_index) % s > 0)
            future_seasonal = np.tile(last_seasonal_train, num_tiles)[:len(future_seasonal_index)]
            future_seasonal_series = pd.Series(future_seasonal, index=future_seasonal_index)
            final_forecast_sarima_plus = future_forecast_trend + future_seasonal_series
            
            # Visualisation de la décomposition saisonnière
            st.subheader("Décomposition Saisonnière (Entraînement)")
            fig = decomposition_train.plot()
            fig.set_size_inches((12, 8))
            plt.suptitle(f'Décomposition de la série "{name}" (Entraînement)', y=1.02)
            st.pyplot(fig)
            
            
            
            # Comparaison des modèles
            st.subheader("Comparaison des Prévisions")

            # Index commun pour toutes les prédictions
            common_index = pd.date_range(start=train_data.index[-1], end=future_forecast_index[-1], freq='MS')

            # Préparer les séries prédictives avec continuité temporelle
            sarima_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), sarima_predictions])
            prophet_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), prophet_predictions])
            sarima_plus_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), final_forecast_sarima_plus])

            # Holt-Winters : générer série avec continuité aussi
            if not best_hw_predictions.isnull().all():
                best_hw_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), best_hw_predictions])
            else:
                best_hw_plot_data = pd.Series(index=common_index, dtype='float64')

            # Assemblage du DataFrame
            comparison_df = pd.DataFrame({
                'Réel (Entraînement)': train_data,
                'Réel (Test)': test_data,
                'Prédiction SARIMA': sarima_plot_data,
                'Prédiction Prophet': prophet_plot_data,
                'Prévision S+ARIMA': sarima_plus_plot_data,
                f'Prédiction {best_hw_name}': best_hw_plot_data if not best_hw_plot_data.isnull().all() else np.nan
                })

                
                
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(base.index, base.values, label='Réel (Entraînement)', color='blue')
            ax.plot(test_data.index, test_data.values, label='Réel (Test)', color='cyan')
            
            
            sarima_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), sarima_predictions])
            sarima_forecast = best_sarima_model.get_forecast(steps=len(test_data) + future_periods)
            sarima_ci = sarima_forecast.conf_int(alpha=0.05)  # intervalle à 95%
            sarima_predictions = sarima_forecast.predicted_mean
            ax.plot(sarima_plot_data.index, sarima_plot_data.values, label='Prédiction SARIMA', color='green', linestyle='--')
            
            prophet_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), prophet_predictions])
            ax.plot(prophet_plot_data.index, prophet_plot_data.values, label='Prédiction Prophet', color='orange', linestyle='-.')
            prophet_confidence_plot_data_lower = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), prophet_predictions_lower])
            prophet_confidence_plot_data_upper = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), prophet_predictions_upper])
            
            sarima_plus_plot_data = pd.concat([pd.Series([train_data[-1]], index=[train_data.index[-1]]), final_forecast_sarima_plus])
            ax.plot(sarima_plus_plot_data.index, sarima_plus_plot_data.values, label='Prévision S+ARIMA', color='purple', linestyle=':')
            

            ax.plot(best_hw_plot_data.index, best_hw_plot_data.values, label='Prédiction HOLT WINTER', color='red', linestyle='-.')
            residuals = train_data - model_hw_add.fittedvalues
            resid_std = residuals.std()
            z = 1.96  # pour 95 %

            hw_add_lower = hw_add_predictions - z * resid_std
            hw_add_upper = hw_add_predictions + z * resid_std

            ax.axvline(x=train_data.index[-1], color='red', linestyle='-', label='Fin Entraînement / Début Test')
            ax.axvline(x=test_data.index[-1], color='black', linestyle='-', label='Fin Test / Début Prédictions Futures')
            
            
            ax.set_title(f'Comparaison des Prévisions pour {name}')
            ax.set_xlabel('Date')
            ax.set_ylabel(name)
            ax.legend()
            plt.grid(True)
            st.pyplot(fig)
            
            # Évaluation des performances
            prophet_test_predictions_eval = prophet_predictions.loc[test_data.index]
            sarima_test_predictions_eval = sarima_predictions.loc[test_data.index]
            sarima_plus_test_predictions_eval = final_forecast_sarima_plus.loc[test_data.index]
            rmse_sarima = np.sqrt(mean_squared_error(test_data, sarima_test_predictions_eval))
            rmse_prophet = np.sqrt(mean_squared_error(test_data, prophet_test_predictions_eval))
            rmse_sarima_plus = np.sqrt(mean_squared_error(test_data, sarima_plus_test_predictions_eval))
            st.write(f"**Évaluation sur la période de test ({test_data.index[0].strftime('%Y-%m')} à {test_data.index[-1].strftime('%Y-%m')})**")
            st.write(f"RMSE SARIMA: {rmse_sarima:.2f}")
            st.write(f"RMSE Prophet: {rmse_prophet:.2f}")
            st.write(f"RMSE S+ARIMA: {rmse_sarima_plus:.2f}")
            st.write(f"RMSE Holt-Winters Additif : {rmse_hw_add:.2f}" if rmse_hw_add != np.inf else "RMSE Holt-Winters Additif : N/A")
            st.write(f"RMSE Holt-Winters Multiplicatif : {rmse_hw_mul:.2f}" if rmse_hw_mul != np.inf else "RMSE Holt-Winters Multiplicatif : N/A")
            
            # Visualisation séparée pour Prophet
            st.subheader("Prévisions Prophet")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(base.index, base.values, label='Réel (Entraînement)', color='blue')
            ax.plot(test_data.index, test_data.values, label='Réel (Test)', color='cyan')
            ax.plot(prophet_plot_data.index, prophet_plot_data.values, label='Prédiction Prophet', color='orange', linestyle='--')
            ax.fill_between(prophet_confidence_plot_data_lower.index, prophet_confidence_plot_data_lower.values, 
                           prophet_confidence_plot_data_upper.values, color='orange', alpha=0.2)
            ax.axvline(x=train_data.index[-1], color='red', linestyle='-', label='Fin Entraînement / Début Test')
            ax.axvline(x=test_data.index[-1], color='black', linestyle='-', label='Fin Test / Début Prédictions Futures')
            ax.set_title(f'Prévisions Prophet pour {name}')
            ax.set_xlabel('Date')
            ax.set_ylabel(name)
            ax.legend()
            plt.grid(True)
            st.pyplot(fig)
            
            # Visualisation séparée pour SARIMA
            st.subheader("Prévisions SARIMA")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(base.index, base.values, label='Réel (Entraînement)', color='blue')
            ax.plot(test_data.index, test_data.values, label='Réel (Test)', color='cyan')
            ax.plot(sarima_plot_data.index, sarima_plot_data.values, label='Prédiction SARIMA', color='green', linestyle='--')
            ax.fill_between(
                    sarima_ci.index,
                    sarima_ci.iloc[:, 0],  # borne inférieure
                    sarima_ci.iloc[:, 1],  # borne supérieure
                    color='green', alpha=0.2, label='Intervalle SARIMA'
                )
            ax.axvline(x=train_data.index[-1], color='red', linestyle='-', label='Fin Entraînement / Début Test')
            ax.axvline(x=test_data.index[-1], color='black', linestyle='-', label='Fin Test / Début Prédictions Futures')
            ax.set_title(f'Prévisions SARIMA pour {name}')
            ax.set_xlabel('Date')
            ax.set_ylabel(name)
            ax.legend()
            plt.grid(True)
            st.pyplot(fig)
            
            # Visualisation séparée pour S+ARIMA
            st.subheader("Prévisions S+ARIMA")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(base.index, base.values, label='Réel (Entraînement)', color='blue')
            ax.plot(test_data.index, test_data.values, label='Réel (Test)', color='cyan')
            ax.plot(sarima_plus_plot_data.index, sarima_plus_plot_data.values, 
                    label='Prévision S+ARIMA', color='purple', linestyle='--')
            ax.fill_between(
                    lower_bound_sarima_plus.index,
                    lower_bound_sarima_plus.values,
                    upper_bound_sarima_plus.values,
                    color='purple', alpha=0.2, label='Intervalle S+ARIMA'
                )

            ax.axvline(x=train_data.index[-1], color='red', linestyle='-', label='Fin Entraînement / Début Test')
            ax.axvline(x=test_data.index[-1], color='black', linestyle='-', label='Fin Test / Début Prédictions Futures')
            ax.set_title(f'Prévisions S+ARIMA pour {name}')
            ax.set_xlabel('Date')
            ax.set_ylabel(name)
            ax.legend()
            plt.grid(True)
            st.pyplot(fig)
            
            # Visualisation spécifique Holt-Winters
            if not best_hw_predictions.isnull().all():
                st.subheader(f"Prévisions {best_hw_name} pour {name}")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(base.index, base.values, label='Réel', color='blue')
                ax.plot(pd.concat([train_data[-1:], best_hw_predictions]).index,
                        pd.concat([train_data[-1:], best_hw_predictions]).values,
                        label=f'Prédiction {best_hw_name}', color='brown', linestyle='--')
                ax.fill_between(
                    hw_add_lower.index,
                    hw_add_lower.values,
                    hw_add_upper.values,
                    color='brown', alpha=0.2, label="Intervalle Holt-Winters"
                )
                ax.axvline(x=train_data.index[-1], color='red', linestyle='-', label='Fin Entraînement')
                ax.axvline(x=test_data.index[-1], color='black', linestyle='-', label='Fin Test')
                ax.set_title(f'Prévisions {best_hw_name} pour {name}')
                ax.set_xlabel('Date')
                ax.set_ylabel(name)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("**Source** : Base_fin.xlsx, Scraping depuis anstat.ci")
st.sidebar.write("**Créé avec** : Streamlit, Pandas, Seaborn, Statsmodels, Prophet, Selenium")