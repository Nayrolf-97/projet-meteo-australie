# Prévision météo en Australie — Projet Machine Learning

Projet de classification binaire visant à prédire la variable `RainTomorrow` (pleuvra-t-il demain ?) à partir du jeu de données météorologiques australien du Bureau of Meteorology, couvrant la période du 1er novembre 2007 au 25 juin 2017 sur 49 stations.

## Problématique

L'Australie est un continent vaste caractérisé par une grande diversité de climats (arides, tropicaux, tempérés). La prévision météorologique y est un enjeu critique pour la gestion des risques naturels et l'optimisation agricole. Ce projet utilise le machine learning pour essayer de prédire la pluie à J+1 à partir de relevés quotidiens.

## Structure du projet

```
.
├── data/
│   └── weatherAUS.csv                    # dataset brut
├── src/
│   ├── weather_cleaning.py               # nettoyage & imputation
│   ├── weather_geo.py                    # géocodage & cartes folium
│   ├── weather_data.py                   # chargement & feature engineering
│   └── weather_modeling.py               # évaluation & modélisation
├── notebooks/
│   ├── 01_exploration_and_cleaning.ipynb # EDA + nettoyage
│   ├── 02_baseline_et_reg_log.ipynb      # baseline + régression logistique
│   └── 03_modeles_catboost_et_deep.ipynb # CatBoost + MLP
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <url-du-repo>
cd projet-meteo-australie
pip install -r requirements.txt
```

## Utilisation

Les notebooks sont prévus pour être exécutés dans l'ordre :

1. **`01_exploration_and_cleaning.ipynb`** — exploration, analyse des valeurs manquantes par station (cartes folium), nettoyage par groupes Location × Mois, export de `df_clean.csv`
2. **`02_baseline_et_reg_log.ipynb`** — baseline déterministe + régression logistique entraînée par station avec split temporel
3. **`03_modeles_catboost_et_deep.ipynb`** — CatBoost global (Location en feature) et MLP Keras avec encodage cyclique des dates


## Auteurs

- Florian LESOIL
- Clément DAUCHEZ
