"""
weather_geo.py — Géocodage des stations météo et cartographie des valeurs manquantes.

Ce module fournit :
- les coordonnées GPS des 49 stations du dataset weatherAUS
  (vérifiées manuellement après géocodage initial via geopy/ArcGIS) ;
- des fonctions pour construire des cartes folium montrant la
  distribution spatiale des valeurs manquantes par station.

Ces cartes sont utiles pour comprendre que certains manques ne sont pas
aléatoires mais concentrés sur certaines stations (Evaporation, Sunshine,
Clouds sont quasi absentes dans plusieurs stations de l'intérieur).
"""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

# Coordonnées (latitude, longitude) des 49 stations météo.
# Sources : géocodage ArcGIS via geopy, puis correction manuelle
# pour les stations mal localisées (PearceRAAF notamment).
LOCATION_COORDS = {
    "Adelaide": (-34.9281805, 138.5999312),
    "Albany": (-35.0247822, 117.883608),
    "Albury": (-36.0737734, 146.9135265),
    "AliceSprings": (-23.6983884, 133.8812885),
    "BadgerysCreek": (-33.8831452, 150.742466),
    "Ballarat": (-37.5623013, 143.8605645),
    "Bendigo": (-36.7590183, 144.2826718),
    "Brisbane": (-27.4689623, 153.0235009),
    "Cairns": (-16.9206657, 145.7721854),
    "Canberra": (-35.2975906, 149.1012676),
    "Cobar": (-31.9666628, 145.3045054),
    "CoffsHarbour": (-30.2985996, 153.1094116),
    "Dartmoor": (-37.895212, 141.2679429),
    "Darwin": (-12.46044, 130.8410469),
    "GoldCoast": (-28.0805, 153.4309187),
    "Hobart": (-42.8825088, 147.3281233),
    "Katherine": (-14.4646157, 132.2635993),
    "Launceston": (-41.4340813, 147.1373496),
    "Melbourne": (-37.8142454, 144.9631732),
    "MelbourneAirport": (-37.6667554, 144.8288501),
    "Mildura": (-34.195274, 142.1503146),
    "Moree": (-29.4617202, 149.8407153),
    "MountGambier": (-37.8301386, 140.7842627),
    "MountGinini": (-35.5297437, 148.7725396),
    "Newcastle": (-32.9192953, 151.7795348),
    "Nhil": (-35.4325403, 141.2833862),
    "NorahHead": (-33.2816667, 151.5677778),
    "NorfolkIsland": (-29.0328038, 167.9483137),
    "Nuriootpa": (-34.4693354, 138.9939006),
    "PearceRAAF": (-31.6678, 116.015),
    "Penrith": (-33.7511954, 150.6941711),
    "Perth": (-31.9558967, 115.8605784),
    "PerthAirport": (-31.9415213, 115.9655769),
    "Portland": (-38.3456231, 141.6042304),
    "Richmond": (-37.80745, 144.9907213),
    "Sale": (-38.1094463, 147.0656717),
    "SalmonGums": (-32.9815167, 121.6440785),
    "Sydney": (-33.8698439, 151.2082848),
    "SydneyAirport": (-33.9498935, 151.1819682),
    "Townsville": (-19.2569391, 146.8239537),
    "Tuggeranong": (-35.4209771, 149.0921341),
    "Uluru": (-25.3455545, 131.0369615),
    "WaggaWagga": (-35.115, 147.3677778),
    "Walpole": (-34.9776796, 116.7310063),
    "Watsonia": (-37.7109468, 145.0837808),
    "Williamtown": (-32.815, 151.8427778),
    "Witchcliffe": (-34.0263348, 115.1004768),
    "Wollongong": (-34.4243941, 150.89385),
    "Woomera": (-31.1999142, 136.8253532),
}


def location_coordinates_frame() -> pd.DataFrame:
    """Renvoie un DataFrame avec les colonnes Location, Latitude, Longitude."""
    return (
        pd.DataFrame.from_dict(LOCATION_COORDS, orient="index", columns=["Latitude", "Longitude"])
        .rename_axis("Location")
        .reset_index()
    )


def add_location_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes Latitude et Longitude au DataFrame via un merge sur Location."""
    coords_df = location_coordinates_frame()
    return df.merge(coords_df, on="Location", how="left")


def build_location_missing_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Construit un résumé du nombre de NaN par station pour les colonnes données.

    Paramètres
    ----------
    df : DataFrame brut (avec colonne Location).
    columns : liste des colonnes pour lesquelles compter les NaN.

    Return
    ------
    DataFrame avec une ligne par station, contenant les coordonnées GPS
    et le nombre de NaN pour chaque colonne demandée.
    """
    df_geo = add_location_coordinates(df)
    grouped = (
        df_geo.groupby("Location")
        .apply(
            lambda g: pd.Series(
                {
                    "Latitude": g["Latitude"].iloc[0],
                    "Longitude": g["Longitude"].iloc[0],
                    **{f"{col}_NaN_count": g[col].isna().sum() for col in columns},
                }
            )
        )
        .reset_index()
    )
    return grouped


def make_missing_value_map(
    grouped: pd.DataFrame,
    column: str,
    radius_scale: float = 50,
    min_radius: float = 2,
) -> folium.Map:
    """Crée une carte folium avec des cercles proportionnels au nombre de NaN.

    Chaque station est représentée par un cercle rouge dont le rayon
    est proportionnel au nombre de valeurs manquantes pour la colonne donnée.

    Paramètres
    ----------
    grouped : résumé par station (sortie de build_location_missing_summary).
    column : nom de la variable à cartographier.
    radius_scale : diviseur pour convertir le nombre de NaN en rayon de cercle.
    min_radius : rayon minimum pour les stations sans ou avec très peu de NaN.

    Return
    ------
    Objet folium.Map prêt à être affiché ou sauvegardé en HTML.
    """
    nan_col = f"{column}_NaN_count"
    m = folium.Map(location=[-25.2744, 133.7751], zoom_start=4, tiles="cartodbpositron")

    data = grouped.dropna(subset=["Latitude", "Longitude"]).copy()

    for _, row in data.iterrows():
        n_nan = row[nan_col]
        radius = max(min_radius, n_nan / radius_scale)
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            color="crimson",
            fill=True,
            fill_opacity=0.6,
            popup=f"{row['Location']}<br>{column} : {int(n_nan)} NaN",
        ).add_to(m)

    return m


def build_missing_value_maps(
    grouped: pd.DataFrame,
    columns: list[str],
    output_dir: str | Path | None = None,
) -> dict[str, folium.Map]:
    """Génère une carte de NaN par colonne et les sauvegarde en HTML si demandé.

    Paramètres
    ----------
    grouped : résumé par station (sortie de build_location_missing_summary).
    columns : liste des colonnes à cartographier.
    output_dir : si fourni, chaque carte est sauvegardée dans ce dossier
        sous le nom australia_nan_{colonne}.html.

    Return
    ------
    Dictionnaire {nom_colonne: objet folium.Map}.
    """
    maps = {}
    output_path = Path(output_dir) if output_dir is not None else None

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for column in columns:
        m = make_missing_value_map(grouped, column)
        maps[column] = m
        if output_path is not None:
            m.save(output_path / f"australia_nan_{column}.html")

    return maps
