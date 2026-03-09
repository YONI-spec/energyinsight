"""
Script d'entraînement pour générer les deux modèles
Exécuter une fois pour créer model_energy.pkl et model_ghg.pkl
"""

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import (
    engineer_features,
    get_feature_list,
    clean_dataset
)


def load_and_prepare_data(filepath, target, mode='energy'):
    """
    Charge et prépare les données.
    """
    print(f"Chargement des données pour {target}...")
    
    df = pd.read_csv(filepath)
    
    print(f"  Shape initiale: {df.shape}")
    
    columns_to_drop = [
        'PropertyName', 'Address', 'TaxParcelIdentificationNumber',
        'ZipCode', 'DataYear', 'City', 'CouncilDistrictCode', 'State',
        'Comments', 'Electricity(kWh)', 'NaturalGas(therms)',
        'ComplianceStatus', 'OSEBuildingID', 'SourceEUI(kBtu/sf)',
        'DefaultData', 'YearsENERGYSTARCertified', 'Outlier',
        'SiteEUI(kBtu/sf)', 'LargestPropertyUseTypeGFA',
        'SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType',
        'SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA',
        'SiteEnergyUseWN(kBtu)', 'SourceEUIWN(kBtu/sf)',
        'GHGEmissionsIntensity', 'SiteEUIWN(kBtu/sf)',
        'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes'
    ]
    
    if mode == 'ghg':
        columns_to_drop.extend([
            'SteamUse(kBtu)', 'NaturalGas(kBtu)', 'Electricity(kBtu)'
        ])
    elif mode == 'energy':
        columns_to_drop.extend([
            'SteamUse(kBtu)', 'NaturalGas(kBtu)', 'Electricity(kBtu)'
        ])
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    df = df[df['PropertyGFATotal'] > 0]
    df.loc[df['NumberofBuildings'] == 0, 'NumberofBuildings'] = 1
    df.loc[df['TotalGHGEmissions'] < 0, 'TotalGHGEmissions'] = 0
    
    non_residential = [
        'NonResidential', 'Nonresidential COS',
        'Nonresidential WA', 'SPS-District K-12', 'Campus'
    ]
    df = df[df['BuildingType'].isin(non_residential)].copy()
    
    print(f"  Après filtrage: {df.shape}")
    
    df = engineer_features(df, target=target, mode=mode)
    
    print(f"  Après feature engineering: {df.shape}")
    
    df = clean_dataset(df, target)
    
    q99 = df[target].quantile(0.99)
    df = df[df[target] <= q99]
    
    print(f"  Dataset final: {df.shape}")
    
    return df


def train_model(df, target, mode='energy'):
    """
    Entraîne un modèle sur le dataset.
    """
    print(f"\nEntraînement du modèle pour {target}...")
    
    y = np.log1p(df[target])
    
    num_features, ohe_features, te_features = get_feature_list(
        mode=mode,
        include_energystar='ENERGYSTARScore' in df.columns
    )
    
    available_num = [f for f in num_features if f in df.columns]
    available_ohe = [f for f in ohe_features if f in df.columns]
    available_te = [f for f in te_features if f in df.columns]
    
    all_features = available_num + available_ohe + available_te
    X = df[all_features]
    
    print(f"  Features: {len(available_num)} num, {len(available_ohe)} OHE, {len(available_te)} TE")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', available_num),
            ('ohe', OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            ), available_ohe),
            ('te', ce.TargetEncoder(
                cols=available_te,
                smoothing=30
            ), available_te)
        ]
    )
    
    model = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=25,
            min_child_samples=25,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.4,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1
        ))
    ])
    
    print("  Entraînement...")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\n  Résultats:")
    print(f"    R² Train: {r2_train:.4f}")
    print(f"    R² Test:  {r2_test:.4f}")
    print(f"    RMSE:     {rmse_test:.4f}")
    print(f"    Écart:    {r2_train - r2_test:.4f}")
    
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='r2', n_jobs=-1
    )
    print(f"    CV R²:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model


def main():
    """
    Pipeline complet d'entraînement.
    """
    filepath = '/content/2016_Building_Energy_Benchmarking.csv'
    
    print("="*60)
    print("ENTRAÎNEMENT MODÈLE CONSOMMATION ÉNERGÉTIQUE")
    print("="*60)
    
    df_energy = load_and_prepare_data(
        filepath,
        target='SiteEnergyUse(kBtu)',
        mode='energy'
    )
    
    model_energy = train_model(
        df_energy,
        target='SiteEnergyUse(kBtu)',
        mode='energy'
    )
    
    joblib.dump(model_energy, 'model_energy.pkl')
    print("\nModèle sauvegardé: model_energy.pkl")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT MODÈLE ÉMISSIONS GHG")
    print("="*60)
    
    df_ghg = load_and_prepare_data(
        filepath,
        target='TotalGHGEmissions',
        mode='ghg'
    )
    
    model_ghg = train_model(
        df_ghg,
        target='TotalGHGEmissions',
        mode='ghg'
    )
    
    joblib.dump(model_ghg, 'model_ghg.pkl')
    print("\nModèle sauvegardé: model_ghg.pkl")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print("\nFichiers générés:")
    print("  - model_energy.pkl")
    print("  - model_ghg.pkl")
    print("\nPour lancer l'application:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
