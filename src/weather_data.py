from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("weatherAUS.csv")
CLEAN_PATH = Path("df_clean.csv")
TARGET = "RainTomorrow"


def load_raw_weather(path: str | Path = RAW_PATH, encode_target: bool = False, drop_risk_mm: bool = False) -> pd.DataFrame:
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


def load_clean_weather(path: str | Path = CLEAN_PATH, add_binary_targets: bool = True) -> pd.DataFrame:
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
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Weekday"] = df["Date"].dt.weekday
    return df


def add_cyclical_date_features(df: pd.DataFrame) -> pd.DataFrame:
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
