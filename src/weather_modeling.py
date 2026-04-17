"""
weather_modeling.py — Fonctions d'évaluation et utilitaires de modélisation.

Ce module regroupe :
- les fonctions de split temporel (simple et train/val/test) ;
- le scoring probabiliste (ROC-AUC, PR-AUC, Brier) utilisé pour les
  modèles globaux (CatBoost, MLP) ;
- l'évaluation par station de la baseline naïve et de la régression logistique,
  avec agrégation des résultats en rapport de classification global.

Le split temporel (et non aléatoire) est important pour éviter d'évaluer
un modèle météo sur des jours proches de ceux qu'il a vus en entraînement,
ce qui gonflerait artificiellement les métriques.
"""

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


# ---------------------------------------------------------------------------
# Splits temporels
# ---------------------------------------------------------------------------

def time_split(df_loc: pd.DataFrame, train_frac: float = 0.8):
    """Split temporel simple pour une station individuelle.

    Trie par date puis coupe à train_frac. Utilisé dans les boucles
    d'évaluation par Location (baseline, logreg).

    Return
    ------
    (train, test) : deux DataFrames.
    """
    df_loc = df_loc.sort_values("Date")
    cut = int(len(df_loc) * train_frac)
    return df_loc.iloc[:cut].copy(), df_loc.iloc[cut:].copy()


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split temporel global (toutes stations confondues).

    Utilisé pour les modèles globaux comme CatBoost où l'on entraîne
    un seul modèle sur toutes les stations.

    Return
    ------
    (train, test) : deux DataFrames.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def temporal_train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1
):
    """Split temporel en trois parties : train, validation, test.

    Utilisé pour le modèle de deep learning (MLP) qui a besoin d'un
    jeu de validation séparé pour l'early stopping.

    Return
    ------
    (train, val, test) : trois DataFrames.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


# ---------------------------------------------------------------------------
# Scoring probabiliste
# ---------------------------------------------------------------------------

def score_probabilities(y_true, y_proba) -> pd.Series:
    """Calcule les trois métriques probabilistes utilisées pour les modèles globaux.

    - ROC-AUC : capacité globale de classement (insensible au seuil).
    - PR-AUC  : qualité de détection de la classe rare (pluie, ~22 %).
    - Brier   : calibration des probabilités (plus bas = mieux calibré).

    Paramètres
    ----------
    y_true : valeurs réelles (0 ou 1).
    y_proba : probabilités prédites pour la classe positive.

    Return
    ------
    pd.Series avec les trois métriques.
    """
    return pd.Series({
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Brier": brier_score_loss(y_true, y_proba),
    })


# ---------------------------------------------------------------------------
# Évaluation par station
# ---------------------------------------------------------------------------

def evaluate_naive_by_location(df: pd.DataFrame, min_rows: int = 200):
    """Évalue la baseline déterministe : RainToday → RainTomorrow.

    Pour chaque station, la prédiction est simplement la valeur de
    RainToday. Ce modèle capte l'inertie météo (il pleut peu en
    Australie, donc "demain = aujourd'hui" fonctionne souvent).

    Paramètres
    ----------
    df : DataFrame nettoyé avec colonnes RainToday_bin et RainTomorrow_bin.
    min_rows : nombre minimum de lignes par station pour être évaluée.

    Return
    ------
    (results, report, cm) :
        - results : DataFrame avec F1 et accuracy par station.
        - report : rapport de classification global (toutes stations).
        - cm : matrice de confusion globale.
    """
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


def evaluate_logreg_by_location(df: pd.DataFrame, feature_cols: list[str], min_rows: int = 200):
    """Évalue une régression logistique entraînée station par station.

    Pour chaque station : StandardScaler + LogisticRegression(balanced),
    avec split temporel 80/20. Les prédictions sont agrégées pour
    produire un rapport global.

    Paramètres
    ----------
    df : DataFrame nettoyé avec colonnes RainTomorrow_bin et les features.
    feature_cols : liste des colonnes numériques à utiliser comme features.
    min_rows : nombre minimum de lignes par station.

    Return
    ------
    (results, report, cm, auc) :
        - results : DataFrame avec F1, accuracy et ROC-AUC par station.
        - report : rapport de classification global.
        - cm : matrice de confusion globale.
        - auc : ROC-AUC global (toutes stations agrégées).
    """
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
