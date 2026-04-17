"""
weather_cleaning.py — Nettoyage du jeu de données weatherAUS.

Ce module regroupe toute la logique de traitement des valeurs manquantes :
- imputation par groupe Location × Mois (mean / median / mode selon la variable) ;
- imputation temporelle de Rainfall à partir des jours voisins ;
- suppression des colonnes trop incomplètes (Evaporation, Sunshine, Clouds) ;
- correction de RainToday quand elle contredit Rainfall.

La fonction principale est `clean_weather_data`, qui prend le DataFrame brut
et renvoie une version nettoyée prête pour la modélisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_mode(series: pd.Series):
    """Renvoie le mode d'une Series, ou NaN si la Series est entièrement vide.

    pandas .mode() peut renvoyer un résultat vide si toutes les valeurs
    sont NaN ; cette fonction gère ce cas proprement.
    """
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def fill_with_fallback(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """Impute les NaN restants d'une colonne par paliers successifs.

    Après une première imputation par Location × Mois (faite en amont),
    il peut rester des NaN quand un groupe entier est vide. Cette fonction
    applique trois niveaux de repli :
      1. par Location seule
      2. par Mois seul
      3. valeur globale du dataset

    Paramètres
    ----------
    df : DataFrame contenant la colonne à imputer.
    col : nom de la colonne cible.
    strategy : "mean", "median" ou "mode".

    Return
    ------
    Le DataFrame avec la colonne imputée (modifié en place).
    """
    if strategy == "mean":
        df[col] = df[col].fillna(df.groupby("Location")[col].transform("mean"))
        df[col] = df[col].fillna(df.groupby("Month")[col].transform("mean"))
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df.groupby("Location")[col].transform("median"))
        df[col] = df[col].fillna(df.groupby("Month")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        df[col] = df[col].fillna(df.groupby("Location")[col].transform(safe_mode))
        df[col] = df[col].fillna(df.groupby("Month")[col].transform(safe_mode))
        df[col] = df[col].fillna(safe_mode(df[col]))
    return df


def impute_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes de Rainfall à partir des jours voisins.

    Logique appliquée dans l'ordre :
      1. NaN en bord de série ou dans un trou de plusieurs jours → 0
         (on considère qu'un trou prolongé correspond à une absence de pluie).
      2. NaN isolé entre deux jours sans pluie → 0.
      3. NaN isolé avec au moins un voisin non nul → moyenne(veille, lendemain).
      4. Tout NaN restant → 0 (filet de sécurité).

    Paramètres
    ----------
    df : DataFrame trié par [Location, Date] avec une colonne "Rainfall".

    Return
    ------
    Le DataFrame avec Rainfall imputé, sans les colonnes temporaires.
    """
    df = df.sort_values(["Location", "Date"]).copy()

    # Créer les colonnes de contexte temporel (veille et lendemain par station)
    df["Rainfall_prev"] = df.groupby("Location")["Rainfall"].shift(1)
    df["Rainfall_next"] = df.groupby("Location")["Rainfall"].shift(-1)

    # Cas 1 : trou en bord de série ou dans un gap de plusieurs jours consécutifs
    long_or_edge_gap = (
        df["Rainfall"].isna()
        & (df["Rainfall_prev"].isna() | df["Rainfall_next"].isna())
    )
    df.loc[long_or_edge_gap, "Rainfall"] = 0

    # Cas 2 : NaN isolé entre deux jours à 0 → forcément 0
    isolated_zero_gap = (
        df["Rainfall"].isna()
        & (df["Rainfall_prev"] == 0)
        & (df["Rainfall_next"] == 0)
    )
    df.loc[isolated_zero_gap, "Rainfall"] = 0

    # Cas 3 : NaN isolé avec au moins un voisin non nul → moyenne des voisins
    remaining_gap = df["Rainfall"].isna()
    df.loc[remaining_gap, "Rainfall"] = (
        df.loc[remaining_gap, ["Rainfall_prev", "Rainfall_next"]]
        .mean(axis=1)
    )

    # Filet de sécurité : tout NaN restant passe à 0
    df["Rainfall"] = df["Rainfall"].fillna(0)
    return df.drop(columns=["Rainfall_prev", "Rainfall_next"])


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de nettoyage du dataset weatherAUS.

    Étapes :
      1. Suppression des lignes sans date, tri chronologique.
      2. Imputation des variables numériques par Location × Mois
         (mean pour les températures/pressions, median pour le vent/humidité).
      3. Imputation des directions de vent par mode.
      4. Imputation de RainToday par mode.
      5. Imputation intelligente de Rainfall (cf. impute_rainfall).
      6. Suppression des colonnes trop incomplètes (>38 % de NaN).
      7. Correction de RainToday quand Rainfall > 1 mm mais RainToday == "No".

    Paramètres
    ----------
    df : DataFrame brut issu de weatherAUS.csv (avec colonne Date en datetime).

    Return
    ------
    DataFrame nettoyé, prêt à être sauvegardé ou utilisé pour la modélisation.
    """
    df = df.copy()
    df = df.dropna(subset=["Date"]).sort_values(["Location", "Date"]).reset_index(drop=True)
    df["Month"] = df["Date"].dt.month

    # Nettoyage des valeurs textuelles parasites dans les colonnes booléennes
    for col in ["RainToday", "RainTomorrow"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("object")
                .replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})
            )

    # --- Imputation des variables numériques ---

    # Températures : mean (distribution normale)
    for col in ["MinTemp", "MaxTemp"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform("mean"))
        df = fill_with_fallback(df, col, "mean")

    # Vent et humidité : median (distributions asymétriques)
    for col in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform("median"))
        df = fill_with_fallback(df, col, "median")

    # Pression et températures horaires : mean (distribution normale)
    for col in ["Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform("mean"))
        df = fill_with_fallback(df, col, "mean")

    # --- Imputation des variables catégorielles ---

    # Directions du vent : mode par Location × Mois
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform(safe_mode))
        df = fill_with_fallback(df, col, "mode")

    # RainToday : mode (78 % "No" en moyenne)
    if "RainToday" in df.columns:
        df["RainToday"] = df["RainToday"].fillna(
            df.groupby(["Location", "Month"])["RainToday"].transform(safe_mode)
        )
        df = fill_with_fallback(df, "RainToday", "mode")

    # Rainfall : imputation temporelle spécifique
    df = impute_rainfall(df)

    # Suppression des colonnes trop incomplètes (>38 % de NaN dans le brut)
    cols_to_drop = ["Cloud9am", "Cloud3pm", "Evaporation", "Sunshine"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Correction de cohérence : si Rainfall > 1 mm, alors RainToday = "Yes"
    if "RainToday" in df.columns:
        df.loc[(df["RainToday"] == "No") & (df["Rainfall"] > 1), "RainToday"] = "Yes"

    # Nettoyage final : suppression de la colonne temporaire Month
    df = df.drop(columns=["Month"])
    return df.reset_index(drop=True)
