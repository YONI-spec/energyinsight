"""
Module de feature engineering avancé pour la prédiction énergétique des bâtiments
Réutilisable pour SiteEnergyUse ET TotalGHGEmissions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def haversine_vectorized(lat, lon, center_lat, center_lon):
    """
    Calcule la distance haversine entre des points et un centre.
    Optimisé avec numpy pour traiter tout le dataset d'un coup.
    """
    R = 6371  # Rayon de la Terre en km
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)
    
    dlat = lat_rad - center_lat_rad
    dlon = lon_rad - center_lon_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(center_lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def create_geometric_features(df):
    """
    Crée des features géométriques et de densité.
    """
    df = df.copy()
    
    # Âge du bâtiment
    df['building_age'] = 2016 - df['YearBuilt']
    df['building_age'] = df['building_age'].clip(lower=0)
    
    # Compacité thermique (rapport hauteur/surface)
    df['ThermalCompactness'] = df['NumberofFloors'] / np.sqrt(df['PropertyGFATotal'] + 1)
    
    # Densité par étage
    df['FloorDensity'] = df['PropertyGFATotal'] / (df['NumberofFloors'] + 1)
    
    # Ratio de parking
    df['parking_ratio'] = df['PropertyGFAParking'] / (df['PropertyGFATotal'] + 1)
    
    # Surface par étage
    df['surface_per_floor'] = df['PropertyGFATotal'] / (df['NumberofFloors'] + 1)
    
    # Surface par bâtiment
    df['surface_per_building'] = df['PropertyGFATotal'] / (df['NumberofBuildings'] + 1)
    
    return df


def create_interaction_features(df):
    """
    Crée des interactions pertinentes entre variables.
    """
    df = df.copy()
    
    # Âge × surface (vieux grands bâtiments = moins efficaces)
    df['age_x_surface'] = df['building_age'] * np.log1p(df['PropertyGFATotal'])
    
    # Étages × compacité (tours denses)
    df['floors_x_compactness'] = df['NumberofFloors'] * df['ThermalCompactness']
    
    # Parking × surface (centres commerciaux vs bureaux)
    df['parking_x_surface'] = df['parking_ratio'] * np.log1p(df['PropertyGFATotal'])
    
    return df


def create_geographic_features(df, n_clusters=5):
    """
    Crée des features géographiques et de clustering.
    """
    df = df.copy()
    
    # Distance au centre-ville
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    df['distance_to_center_km'] = haversine_vectorized(
        df['Latitude'].values,
        df['Longitude'].values,
        center_lat,
        center_lon
    )
    
    # Clustering géographique
    cluster_features = ['Latitude', 'Longitude', 'YearBuilt', 'PropertyGFATotal']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features].fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(scaled_data)
    
    return df


def create_energy_star_features(df):
    """
    Impute et enrichit ENERGYSTARScore.
    """
    df = df.copy()
    
    # Flag binaire
    df['has_energy_star'] = df['ENERGYSTARScore'].notna().astype(int)
    
    # Imputation par groupe
    df['ENERGYSTARScore_imputed'] = df.groupby(
        ['BuildingType', 'PrimaryPropertyType']
    )['ENERGYSTARScore'].transform(lambda x: x.fillna(x.median()))
    
    # Fallback sur BuildingType seul
    df['ENERGYSTARScore_imputed'] = df['ENERGYSTARScore_imputed'].fillna(
        df.groupby('BuildingType')['ENERGYSTARScore'].transform('median')
    )
    
    # Dernière option : médiane globale
    df['ENERGYSTARScore_imputed'] = df['ENERGYSTARScore_imputed'].fillna(
        df['ENERGYSTARScore'].median()
    )
    
    # Interaction score × surface
    df['energystar_x_surface'] = df['ENERGYSTARScore_imputed'] * np.log1p(df['PropertyGFATotal'])
    
    return df


def apply_log_transform(df, columns_to_log):
    """
    Applique une transformation log1p aux colonnes spécifiées.
    """
    df = df.copy()
    
    for col in columns_to_log:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    return df


def engineer_features(df, target='SiteEnergyUse(kBtu)', mode='energy'):
    """
    Pipeline complet de feature engineering.
    
    Args:
        df: DataFrame brut
        target: Variable cible
        mode: 'energy' ou 'ghg' (détermine quelles features sont légitimes)
    
    Returns:
        DataFrame avec toutes les features engineerées
    """
    df = df.copy()
    
    # 1. Features géométriques
    df = create_geometric_features(df)
    
    # 2. Features géographiques
    df = create_geographic_features(df, n_clusters=5)
    
    # 3. ENERGYSTARScore
    if 'ENERGYSTARScore' in df.columns:
        df = create_energy_star_features(df)
    
    # 4. Interactions
    df = create_interaction_features(df)
    
    # 5. Transformation log des variables asymétriques
    base_log_cols = [
        'PropertyGFATotal', 'PropertyGFAParking',
        'NumberofBuildings', 'NumberofFloors',
        'ThermalCompactness', 'FloorDensity',
        'parking_ratio', 'distance_to_center_km',
        'surface_per_floor', 'surface_per_building'
    ]
    
    # Pour le mode energy, on peut utiliser TotalGHGEmissions
    if mode == 'energy' and 'TotalGHGEmissions' in df.columns:
        base_log_cols.append('TotalGHGEmissions')
    
    df = apply_log_transform(df, base_log_cols)
    
    return df


def get_feature_list(mode='energy', include_energystar=True):
    """
    Retourne la liste des features à utiliser selon le mode.
    
    Args:
        mode: 'energy' ou 'ghg'
        include_energystar: si ENERGYSTARScore est disponible
    
    Returns:
        tuple (num_features, ohe_features, te_features)
    """
    num_features = [
        'log_PropertyGFATotal',
        'log_NumberofBuildings',
        'log_NumberofFloors',
        'log_PropertyGFAParking',
        'log_ThermalCompactness',
        'log_FloorDensity',
        'log_parking_ratio',
        'log_distance_to_center_km',
        'log_surface_per_floor',
        'log_surface_per_building',
        'YearBuilt',
        'building_age',
        'Latitude',
        'Longitude',
        'age_x_surface',
        'floors_x_compactness',
        'parking_x_surface',
    ]
    
    if include_energystar:
        num_features.extend([
            'ENERGYSTARScore_imputed',
            'has_energy_star',
            'energystar_x_surface'
        ])
    
    if mode == 'energy':
        num_features.append('log_TotalGHGEmissions')
    
    ohe_features = ['BuildingType', 'Cluster_ID']
    te_features = ['Neighborhood', 'PrimaryPropertyType']
    
    return num_features, ohe_features, te_features


def clean_dataset(df, target):
    """
    Nettoyage final du dataset avant entraînement.
    """
    df = df.copy()
    
    # Supprimer lignes sans target
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    
    # Supprimer les valeurs infinies
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df
