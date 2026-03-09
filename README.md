# Prédiction Énergétique et GHG - Application Streamlit

Application professionnelle de prédiction pour les bâtiments non-résidentiels de Seattle.

## Fonctionnalités

- **Prédiction simple** : Estimation pour un bâtiment unique
- **Prédiction batch** : Traitement de fichiers CSV avec plusieurs bâtiments
- **Analyse du modèle** : Visualisation de l'importance des features
- **Double modèle** : Consommation énergétique ET émissions GHG

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner les modèles

Exécutez le script d'entraînement pour générer les deux modèles :

```bash
python train_models.py
```

Cela créera :
- `model_energy.pkl` : Modèle de prédiction de la consommation énergétique
- `model_ghg.pkl` : Modèle de prédiction des émissions GHG

### 2. Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

## Structure des fichiers

```
.
├── app.py                    # Application Streamlit principale
├── feature_engineering.py    # Module de feature engineering
├── train_models.py           # Script d'entraînement des modèles
├── requirements.txt          # Dépendances Python
├── model_energy.pkl          # Modèle consommation (généré)
├── model_ghg.pkl            # Modèle GHG (généré)
└── README.md                # Ce fichier
```

## Feature Engineering

Le module applique automatiquement :

### Features géométriques
- `building_age` : Âge du bâtiment (2016 - année de construction)
- `ThermalCompactness` : Ratio hauteur/surface
- `FloorDensity` : Surface par étage
- `parking_ratio` : Ratio parking/surface totale
- `surface_per_floor` : Surface moyenne par étage
- `surface_per_building` : Surface moyenne par bâtiment

### Features géographiques
- `distance_to_center_km` : Distance au centre-ville (haversine)
- `Cluster_ID` : Cluster géographique (K-means, k=5)

### Interactions
- `age_x_surface` : Âge × log(surface)
- `floors_x_compactness` : Étages × compacité thermique
- `parking_x_surface` : Ratio parking × log(surface)
- `energystar_x_surface` : Score ENERGY STAR × log(surface)

### Transformations log
Application de `log1p()` sur toutes les variables asymétriques pour normaliser les distributions.

## Modèles

### Consommation Énergétique (SiteEnergyUse)
- **Algorithme** : LightGBM
- **Features** : Données physiques + TotalGHGEmissions (légitime)
- **Performance cible** : R² ≈ 0.75-0.85

### Émissions GHG (TotalGHGEmissions)
- **Algorithme** : LightGBM
- **Features** : Données physiques uniquement (pas de données énergie)
- **Performance cible** : R² ≈ 0.50-0.65

## Utilisation en production

### Prédiction simple

```python
import joblib
import pandas as pd
from feature_engineering import engineer_features, get_feature_list

# Charger le modèle
model = joblib.load('model_energy.pkl')

# Préparer les données
input_data = pd.DataFrame({
    'BuildingType': ['NonResidential'],
    'PrimaryPropertyType': ['Office'],
    'Neighborhood': ['DOWNTOWN'],
    'YearBuilt': [2000],
    'PropertyGFATotal': [50000],
    'PropertyGFAParking': [5000],
    'NumberofBuildings': [1],
    'NumberofFloors': [5],
    'Latitude': [47.6062],
    'Longitude': [-122.3321],
    'ENERGYSTARScore': [50],
    'TotalGHGEmissions': [100],  # Requis pour mode energy
    'SiteEnergyUse(kBtu)': [0]
})

# Feature engineering
df_featured = engineer_features(input_data, mode='energy')

# Récupérer les features
num_feats, ohe_feats, te_feats = get_feature_list(mode='energy')
all_feats = num_feats + ohe_feats + te_feats
available = [f for f in all_feats if f in df_featured.columns]

# Prédire
X = df_featured[available]
prediction_log = model.predict(X)[0]
prediction = np.expm1(prediction_log)

print(f"Consommation prédite: {prediction:,.0f} kBtu")
```

### Prédiction batch

```python
# Charger un CSV
df_batch = pd.read_csv('buildings.csv')

# Même pipeline
df_featured = engineer_features(df_batch, mode='energy')
X_batch = df_featured[available_features]
predictions = np.expm1(model.predict(X_batch))

df_batch['Prediction_kBtu'] = predictions
df_batch.to_csv('predictions.csv', index=False)
```

## Paramètres du modèle

### LightGBM Configuration

```python
LGBMRegressor(
    n_estimators=400,        # Nombre d'arbres
    learning_rate=0.04,      # Vitesse d'apprentissage
    max_depth=6,             # Profondeur max
    num_leaves=25,           # Feuilles par arbre (anti-overfitting)
    min_child_samples=25,    # Min observations par feuille
    subsample=0.75,          # Échantillonnage lignes
    colsample_bytree=0.75,   # Échantillonnage colonnes
    reg_alpha=0.4,           # Régularisation L1
    reg_lambda=1.5,          # Régularisation L2
    random_state=42
)
```

## Preprocessing

### OneHotEncoder
- `BuildingType` : Type de bâtiment (5 modalités)
- `Cluster_ID` : Cluster géographique (5 clusters)

### TargetEncoder
- `Neighborhood` : Quartier (~19 modalités)
- `PrimaryPropertyType` : Usage principal (~22 modalités)
- Smoothing=30 pour éviter l'overfitting sur petits groupes

## Limitations

1. **Données 2016** : Les prédictions sont basées sur le dataset 2016, les performances énergétiques peuvent avoir changé depuis.

2. **Pas de météo** : L'application ne prend pas en compte les variations météorologiques annuelles.

3. **Généralisation** : Le modèle est entraîné sur Seattle uniquement, peut ne pas généraliser à d'autres villes.

4. **Features manquantes pour GHG** : Sans données énergétiques détaillées (gaz, électricité), le R² pour GHG est limité à ~0.60.

## Améliorations futures

- Ajouter les données 2015-2023 pour plus de robustesse
- Intégrer des données météo (degrés-jours de chauffage/climatisation)
- Déploiement cloud (Streamlit Cloud, Heroku, AWS)
- API REST pour intégration dans d'autres systèmes
- Interface mobile responsive

## Contact

Version 1.0 - Mars 2026
