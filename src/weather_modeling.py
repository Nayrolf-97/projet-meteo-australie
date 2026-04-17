from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def time_split(df_loc: pd.DataFrame, train_frac: float = 0.8):
    df_loc = df_loc.sort_values("Date")
    cut = int(len(df_loc) * train_frac)
    return df_loc.iloc[:cut].copy(), df_loc.iloc[cut:].copy()


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    df = df.sort_values("Date").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def temporal_train_val_test_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def score_probabilities(y_true, y_proba) -> pd.Series:
    return pd.Series({
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Brier": brier_score_loss(y_true, y_proba),
    })


def evaluate_naive_by_location(df: pd.DataFrame, min_rows: int = 200):
    rows = []
    y_true_all = []
    y_pred_all = []

    for location, df_loc in df.groupby("Location"):
        df_loc = df_loc.dropna(subset=["RainToday_bin", "RainTomorrow_bin"]).copy()

        if len(df_loc) < min_rows or df_loc["RainTomorrow_bin"].nunique() < 2:
            continue

        _, test = time_split(df_loc)

        y_true = test["RainTomorrow_bin"]
        y_pred = test["RainToday_bin"]

        rows.append({
            "Location": location,
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1_pluie": f1_score(y_true, y_pred, zero_division=0),
            "Nb_test": len(test),
        })

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    results = pd.DataFrame(rows).sort_values("F1_pluie", ascending=False).reset_index(drop=True)
    report = pd.DataFrame(classification_report(y_true_all, y_pred_all, output_dict=True)).T
    cm = confusion_matrix(y_true_all, y_pred_all)

    return results, report, cm


def evaluate_logreg_by_location(df: pd.DataFrame, feature_cols, min_rows: int = 200):
    rows = []
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for location, df_loc in df.groupby("Location"):
        df_loc = df_loc.dropna(subset=["RainTomorrow_bin"]).copy()

        if len(df_loc) < min_rows or df_loc["RainTomorrow_bin"].nunique() < 2:
            continue

        train, test = time_split(df_loc)

        X_train = train[feature_cols]
        y_train = train["RainTomorrow_bin"]
        X_test = test[feature_cols]
        y_test = test["RainTomorrow_bin"]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        rows.append({
            "Location": location,
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
            "F1_pluie": f1_score(y_test, y_pred, zero_division=0),
            "Nb_test": len(test),
        })

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_proba_all.extend(y_proba.tolist())

    results = pd.DataFrame(rows).sort_values("F1_pluie", ascending=False).reset_index(drop=True)
    report = pd.DataFrame(classification_report(y_true_all, y_pred_all, output_dict=True)).T
    cm = confusion_matrix(y_true_all, y_pred_all)
    auc = roc_auc_score(y_true_all, y_proba_all)

    return results, report, cm, auc
