import re
import unicodedata



def clean_filename(text):
        motif_a_enlever = "INDICE HARMONISÉ DES PRIX À LA CONSOMMATION PAR POSTE DE CONSOMMATION (NATIONAL) : "
        text = text.replace(motif_a_enlever, "")
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        text = re.sub(r'[\\/*?:"<>|\-,]', " ", text)
        text = text.replace("'", "_")
        text = text.replace("’", "_")
        text = re.sub(r"\s+", "_", text)
        text = text.strip("_")
        return text