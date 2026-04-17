"""
weather_data.py — Chargement des données et feature engineering calendaire.

Ce module fournit les fonctions de chargement du dataset weatherAUS
(version brute ou nettoyée) ainsi que deux stratégies d'ajout de
features temporelles :
- `add_calendar_features` : variables calendaires simples (Year, Month, etc.),
  adaptées aux modèles de boosting comme CatBoost ;
- `add_cyclical_date_features` : encodage sinusoïdal de la date,
  adapté aux réseaux de neurones (évite les discontinuités artificielles,
  par exemple entre décembre et janvier).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("weatherAUS.csv")
CLEAN_PATH = Path("df_clean.csv")
TARGET = "RainTomorrow"


def load_raw_weather(
    path: str | Path = RAW_PATH,
    encode_target: bool = False,
    drop_risk_mm: bool = False,
) -> pd.DataFrame:
    """Charge le dataset brut weatherAUS.csv.

    Paramètres
    ----------
    path : chemin vers le fichier CSV.
    encode_target : si True, encode RainTomorrow en 0/1 et supprime les
        lignes où la cible est manquante.
    drop_risk_mm : si True, supprime la colonne RISK_MM (qui est un proxy
        direct de la cible et causerait une fuite de données).

    Return
    ------
    DataFrame avec la colonne Date convertie en datetime.
    """
    path = Path(path)
    assert path.exists(), f"Le fichier {path.name} est introuvable."

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    if encode_target and TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
        df = df.dropna(subset=[TARGET]).copy()
        df[TARGET] = df[TARGET].astype(int)

    if drop_risk_mm and "RISK_MM" in df.columns:
        df = df.drop(columns=["RISK_MM"])

    return df


def load_clean_weather(
    path: str | Path = CLEAN_PATH,
    add_binary_targets: bool = True,
) -> pd.DataFrame:
    """Charge la version nettoyée du dataset (df_clean.csv).

    Paramètres
    ----------
    path : chemin vers le fichier CSV nettoyé.
    add_binary_targets : si True, ajoute les colonnes RainToday_bin et
        RainTomorrow_bin (0/1), utiles pour les modèles baseline qui
        travaillent par station.

    Return
    ------
    DataFrame nettoyé avec Date en datetime.
    """
    path = Path(path)
    assert path.exists(), f"Le fichier {path.name} est introuvable."

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if add_binary_targets:
        if "RainToday" in df.columns:
            df["RainToday_bin"] = df["RainToday"].map({"Yes": 1, "No": 0})
        if TARGET in df.columns:
            df[f"{TARGET}_bin"] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features calendaires simples : Year, Month, DayOfYear, Weekday.

    Ces features sont adaptées aux modèles de boosting (CatBoost, XGBoost)
    qui peuvent capturer des seuils non linéaires sur des valeurs entières.
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Weekday"] = df["Date"].dt.weekday
    return df


def add_cyclical_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute un encodage cyclique (sin/cos) de la date.

    Cet encodage est adapté aux réseaux de neurones : il garantit
    que le 31 décembre et le 1er janvier sont proches dans l'espace
    des features (contrairement à DayOfYear = 365 vs 1 qui crée
    une discontinuité artificielle).

    Trois cycles sont encodés :
    - Mois (période = 12)
    - Jour de l'année (période = 365.25)
    - Jour de la semaine (période = 7)
    """
    df = df.copy()
    dayofyear = df["Date"].dt.dayofyear
    month = df["Date"].dt.month
    weekday = df["Date"].dt.weekday

    df["Year"] = df["Date"].dt.year
    df["Month_sin"] = np.sin(2 * np.pi * month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month / 12)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    df["Weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return df