import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import kruskal
from prophet import Prophet
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
import numpy as np



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


def build_missing_functions(
        df: pd.DataFrame,
        repartition_ncoa: dict,
        ncoa_fonction: dict,
        method: str = "mean",
        weights: dict | None = None,
) -> pd.DataFrame:
    """
    Complète le DataFrame avec les fonctions NCOA manquantes.

    Paramètres
    ----------
    df : DataFrame existant (index : dates ; colonnes : postes + fonctions déjà présentes)
    repartition_ncoa : {poste: num_fonction}
    ncoa_fonction    : {num_fonction: libellé_fonction}
    method           : 'mean', 'sum', ou 'median' (si weights est None)
    weights          : {poste: poids}  -> moyenne pondérée si fourni

    Retour
    ------
    df_out : DataFrame enrichi (copie de df + nouvelles colonnes fonction)
    """
    df_out = df.copy()

    # 1. fonctions manquantes
    missing = {
        num: name for num, name in ncoa_fonction.items()
        if name not in df_out.columns
    }

    for num_fct, name_fct in missing.items():
        # postes liés à cette fonction et présents dans la base
        postes = [p for p, f in repartition_ncoa.items()
                  if f == num_fct and p in df_out.columns]

        if not postes:           # aucun poste disponible → on saute
            continue

        # Agrégation
        if weights:
            w = pd.Series(weights).reindex(postes).fillna(0)
            if w.sum() == 0:
                raise ValueError(f"Aucun poids valide pour {name_fct}")
            df_out[name_fct] = (df_out[postes] * w).sum(axis=1) / w.sum()
        else:
            if method == "sum":
                df_out[name_fct] = df_out[postes].sum(axis=1, skipna=True)
            elif method == "median":
                df_out[name_fct] = df_out[postes].median(axis=1, skipna=True)
            else:  # mean par défaut
                df_out[name_fct] = df_out[postes].mean(axis=1, skipna=True)

    return df_out
