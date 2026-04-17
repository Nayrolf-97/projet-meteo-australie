from __future__ import annotations

import numpy as np
import pandas as pd


def safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def fill_with_fallback(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
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
    df = df.sort_values(["Location", "Date"]).copy()

    df["Rainfall_prev"] = df.groupby("Location")["Rainfall"].shift(1)
    df["Rainfall_next"] = df.groupby("Location")["Rainfall"].shift(-1)

    long_or_edge_gap = (
        df["Rainfall"].isna()
        & (df["Rainfall_prev"].isna() | df["Rainfall_next"].isna())
    )
    df.loc[long_or_edge_gap, "Rainfall"] = 0

    isolated_zero_gap = (
        df["Rainfall"].isna()
        & (df["Rainfall_prev"] == 0)
        & (df["Rainfall_next"] == 0)
    )
    df.loc[isolated_zero_gap, "Rainfall"] = 0

    remaining_gap = df["Rainfall"].isna()
    df.loc[remaining_gap, "Rainfall"] = (
        df.loc[remaining_gap, ["Rainfall_prev", "Rainfall_next"]]
        .mean(axis=1)
    )

    df["Rainfall"] = df["Rainfall"].fillna(0)
    return df.drop(columns=["Rainfall_prev", "Rainfall_next"])


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Date"]).sort_values(["Location", "Date"]).reset_index(drop=True)
    df["Month"] = df["Date"].dt.month

    for col in ["RainToday", "RainTomorrow"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("object")
                .replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})
            )

    for col in ["MinTemp", "MaxTemp"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform("mean"))
        df = fill_with_fallback(df, col, "mean")

    for col in ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform("median"))
        df = fill_with_fallback(df, col, "median")

    for col in ["Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform("mean"))
        df = fill_with_fallback(df, col, "mean")

    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].fillna(df.groupby(["Location", "Month"])[col].transform(safe_mode))
        df = fill_with_fallback(df, col, "mode")

    if "RainToday" in df.columns:
        df["RainToday"] = df["RainToday"].fillna(
            df.groupby(["Location", "Month"])["RainToday"].transform(safe_mode)
        )
        df = fill_with_fallback(df, "RainToday", "mode")

    df = impute_rainfall(df)

    cols_to_drop = ["Cloud9am", "Cloud3pm", "Evaporation", "Sunshine"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    if "RainToday" in df.columns:
        df.loc[(df["RainToday"] == "No") & (df["Rainfall"] > 1), "RainToday"] = "Yes"

    df = df.drop(columns=["Month"])
    return df.reset_index(drop=True)
