import streamlit as st
import streamlit as st
import pandas as pd
import os
import time
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import tempfile
from core.preprocessing import corriger_mois
from core.utils import clean_filename


df = st.session_state.df

def render_scraping():
    st.header("🌐 Scraping ANSTAT")
    st.write("Téléchargement et traitement automatisé des fichiers.")

        
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