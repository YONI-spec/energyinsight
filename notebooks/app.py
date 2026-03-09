"""
Application Streamlit pour la prédiction énergétique et GHG des bâtiments
Modèle dual : SiteEnergyUse(kBtu) ET TotalGHGEmissions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from feature_engineering import engineer_features, get_feature_list


st.set_page_config(
    page_title="Prédiction Énergétique Seattle",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models():
    """
    Charge les modèles sauvegardés.
    """
    try:
        model_energy = joblib.load('model_energy.pkl')
        model_ghg = joblib.load('model_ghg.pkl')
        return model_energy, model_ghg, True
    except FileNotFoundError:
        return None, None, False


def predict_single_building(features, model):
    """
    Fait une prédiction pour un bâtiment unique.
    """
    prediction_log = model.predict([features])[0]
    prediction_original = np.expm1(prediction_log)
    return prediction_original, prediction_log


def create_gauge_chart(value, max_value, title, unit):
    """
    Crée une jauge pour visualiser une métrique.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        number={'suffix': f' {unit}'},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, max_value*0.3], 'color': "#d4edda"},
                {'range': [max_value*0.3, max_value*0.7], 'color': "#fff3cd"},
                {'range': [max_value*0.7, max_value], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_comparison_chart(actual, predicted, title):
    """
    Graphique de comparaison valeurs réelles vs prédites.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        marker=dict(size=8, color='steelblue', opacity=0.6),
        name='Prédictions'
    ))
    
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Parfait'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Valeurs réelles',
        yaxis_title='Valeurs prédites',
        height=500,
        hovermode='closest'
    )
    
    return fig


def create_feature_importance_chart(model, feature_names):
    """
    Graphique d'importance des features.
    """
    try:
        regressor = model.named_steps['regressor']
        importances = regressor.feature_importances_
        
        preprocessor = model.named_steps['preprocess']
        ohe_transformer = preprocessor.named_transformers_.get('ohe')
        
        if ohe_transformer:
            ohe_names = list(ohe_transformer.get_feature_names_out())
        else:
            ohe_names = []
        
        all_names = feature_names[0] + ohe_names + feature_names[2]
        
        n = min(len(importances), len(all_names))
        
        df_imp = pd.DataFrame({
            'Feature': all_names[:n],
            'Importance': importances[:n]
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            df_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Features Importantes',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Impossible d'afficher l'importance des features: {e}")
        return None


def main():
    model_energy, model_ghg, models_loaded = load_models()
    
    st.title("🏢 Prédiction Énergétique et Émissions GHG")
    st.markdown("### Application de prédiction pour les bâtiments non-résidentiels de Seattle")
    
    if not models_loaded:
        st.error("Modèles non trouvés. Assurez-vous que model_energy.pkl et model_ghg.pkl sont présents.")
        st.stop()
    
    tabs = st.tabs(["Prédiction Simple", "Prédiction Batch", "Analyse du Modèle", "À Propos"])
    
    with tabs[0]:
        st.header("Prédiction pour un bâtiment unique")
        
        mode = st.selectbox(
            "Choisir la prédiction",
            ["Consommation Énergétique (kBtu)", "Émissions GHG (tonnes CO₂)"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Caractéristiques du bâtiment")
            
            building_type = st.selectbox(
                "Type de bâtiment",
                ["NonResidential", "Nonresidential COS", "Nonresidential WA",
                 "SPS-District K-12", "Campus"]
            )
            
            property_type = st.selectbox(
                "Usage principal",
                ["Office", "Retail Store", "Hotel", "Multifamily Housing",
                 "Warehouse", "K-12 School", "College/University", "Hospital",
                 "Other", "Parking"]
            )
            
            neighborhood = st.selectbox(
                "Quartier",
                ["DOWNTOWN", "CAPITOL HILL", "BALLARD", "QUEEN ANNE",
                 "UNIVERSITY DISTRICT", "FREMONT", "WALLINGFORD", "Other"]
            )
            
            year_built = st.number_input(
                "Année de construction",
                min_value=1900,
                max_value=2016,
                value=2000
            )
            
            gfa_total = st.number_input(
                "Surface totale (sq ft)",
                min_value=1000,
                max_value=2000000,
                value=50000
            )
            
            gfa_parking = st.number_input(
                "Surface parking (sq ft)",
                min_value=0,
                max_value=500000,
                value=5000
            )
        
        with col2:
            st.subheader("Paramètres supplémentaires")
            
            num_buildings = st.number_input(
                "Nombre de bâtiments",
                min_value=1,
                max_value=50,
                value=1
            )
            
            num_floors = st.number_input(
                "Nombre d'étages",
                min_value=1,
                max_value=80,
                value=5
            )
            
            latitude = st.number_input(
                "Latitude",
                min_value=47.0,
                max_value=48.0,
                value=47.6062,
                format="%.4f"
            )
            
            longitude = st.number_input(
                "Longitude",
                min_value=-123.0,
                max_value=-122.0,
                value=-122.3321,
                format="%.4f"
            )
            
            energystar_score = st.slider(
                "Score ENERGY STAR (optionnel)",
                min_value=1,
                max_value=100,
                value=50
            )
            
            has_energystar = st.checkbox("Score ENERGY STAR disponible", value=True)
        
        if st.button("Calculer la prédiction", type="primary"):
            input_data = pd.DataFrame({
                'BuildingType': [building_type],
                'PrimaryPropertyType': [property_type],
                'Neighborhood': [neighborhood],
                'YearBuilt': [year_built],
                'PropertyGFATotal': [gfa_total],
                'PropertyGFAParking': [gfa_parking],
                'NumberofBuildings': [num_buildings],
                'NumberofFloors': [num_floors],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'ENERGYSTARScore': [energystar_score if has_energystar else np.nan],
                'TotalGHGEmissions': [0],
                'SiteEnergyUse(kBtu)': [0]
            })
            
            target_mode = 'energy' if 'Consommation' in mode else 'ghg'
            
            df_featured = engineer_features(input_data, mode=target_mode)
            
            num_feats, ohe_feats, te_feats = get_feature_list(
                mode=target_mode,
                include_energystar=has_energystar
            )
            
            all_feats = num_feats + ohe_feats + te_feats
            available_feats = [f for f in all_feats if f in df_featured.columns]
            
            X_input = df_featured[available_feats]
            
            model = model_energy if target_mode == 'energy' else model_ghg
            
            try:
                prediction_log = model.predict(X_input)[0]
                prediction = np.expm1(prediction_log)
                
                st.success("Prédiction réussie")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    if target_mode == 'energy':
                        st.metric(
                            "Consommation prédite",
                            f"{prediction:,.0f} kBtu",
                            delta=None
                        )
                        
                        kwh = prediction * 0.293071
                        st.caption(f"≈ {kwh:,.0f} kWh")
                    else:
                        st.metric(
                            "Émissions prédites",
                            f"{prediction:,.1f} tonnes CO₂",
                            delta=None
                        )
                
                with col_res2:
                    intensity = prediction / gfa_total
                    st.metric(
                        "Intensité",
                        f"{intensity:.2f}",
                        delta=None
                    )
                    st.caption("par sq ft")
                
                with col_res3:
                    per_floor = prediction / num_floors
                    st.metric(
                        "Par étage",
                        f"{per_floor:,.0f}",
                        delta=None
                    )
                
                if target_mode == 'energy':
                    max_gauge = 5000000
                    unit = "kBtu"
                else:
                    max_gauge = 2000
                    unit = "tonnes CO₂"
                
                st.plotly_chart(
                    create_gauge_chart(prediction, max_gauge, "Niveau de performance", unit),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
    
    with tabs[1]:
        st.header("Prédiction en lot")
        
        st.markdown("""
        Téléchargez un fichier CSV contenant plusieurs bâtiments pour obtenir
        des prédictions en masse.
        
        **Colonnes requises:**
        - BuildingType
        - PrimaryPropertyType
        - Neighborhood
        - YearBuilt
        - PropertyGFATotal
        - PropertyGFAParking
        - NumberofBuildings
        - NumberofFloors
        - Latitude
        - Longitude
        - ENERGYSTARScore (optionnel)
        """)
        
        uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df_batch = pd.read_csv(uploaded_file)
                
                st.write(f"Fichier chargé: {len(df_batch)} bâtiments")
                st.dataframe(df_batch.head())
                
                mode_batch = st.selectbox(
                    "Type de prédiction",
                    ["Consommation Énergétique", "Émissions GHG"],
                    key="batch_mode"
                )
                
                if st.button("Lancer les prédictions", type="primary", key="batch_predict"):
                    target_mode = 'energy' if 'Consommation' in mode_batch else 'ghg'
                    
                    if 'TotalGHGEmissions' not in df_batch.columns:
                        df_batch['TotalGHGEmissions'] = 0
                    if 'SiteEnergyUse(kBtu)' not in df_batch.columns:
                        df_batch['SiteEnergyUse(kBtu)'] = 0
                    
                    df_featured = engineer_features(df_batch, mode=target_mode)
                    
                    num_feats, ohe_feats, te_feats = get_feature_list(
                        mode=target_mode,
                        include_energystar='ENERGYSTARScore' in df_batch.columns
                    )
                    
                    all_feats = num_feats + ohe_feats + te_feats
                    available_feats = [f for f in all_feats if f in df_featured.columns]
                    
                    X_batch = df_featured[available_feats]
                    
                    model = model_energy if target_mode == 'energy' else model_ghg
                    
                    predictions_log = model.predict(X_batch)
                    predictions = np.expm1(predictions_log)
                    
                    df_results = df_batch.copy()
                    
                    if target_mode == 'energy':
                        df_results['Prediction_kBtu'] = predictions
                        df_results['Intensity_kBtu_sqft'] = predictions / df_batch['PropertyGFATotal']
                    else:
                        df_results['Prediction_tCO2'] = predictions
                        df_results['Intensity_tCO2_sqft'] = predictions / df_batch['PropertyGFATotal']
                    
                    st.success(f"{len(df_results)} prédictions effectuées")
                    
                    st.dataframe(df_results)
                    
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Télécharger les résultats",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    col_name = 'Prediction_kBtu' if target_mode == 'energy' else 'Prediction_tCO2'
                    
                    fig_dist = px.histogram(
                        df_results,
                        x=col_name,
                        nbins=50,
                        title="Distribution des prédictions"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    if 'PrimaryPropertyType' in df_results.columns:
                        fig_box = px.box(
                            df_results,
                            x='PrimaryPropertyType',
                            y=col_name,
                            title="Prédictions par type de bâtiment"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier: {e}")
    
    with tabs[2]:
        st.header("Analyse du modèle")
        
        model_choice = st.radio(
            "Choisir le modèle à analyser",
            ["Consommation Énergétique", "Émissions GHG"]
        )
        
        model_to_analyze = model_energy if 'Consommation' in model_choice else model_ghg
        target_mode = 'energy' if 'Consommation' in model_choice else 'ghg'
        
        feature_lists = get_feature_list(mode=target_mode, include_energystar=True)
        
        st.subheader("Importance des features")
        
        fig_imp = create_feature_importance_chart(model_to_analyze, feature_lists)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        
        st.subheader("Informations sur le modèle")
        
        try:
            regressor = model_to_analyze.named_steps['regressor']
            regressor_name = type(regressor).__name__
            
            st.write(f"**Type de modèle:** {regressor_name}")
            
            if hasattr(regressor, 'n_estimators'):
                st.write(f"**Nombre d'arbres:** {regressor.n_estimators}")
            if hasattr(regressor, 'max_depth'):
                st.write(f"**Profondeur max:** {regressor.max_depth}")
            if hasattr(regressor, 'learning_rate'):
                st.write(f"**Learning rate:** {regressor.learning_rate}")
            
        except Exception as e:
            st.error(f"Impossible d'extraire les informations: {e}")
    
    with tabs[3]:
        st.header("À propos")
        
        st.markdown("""
        ### Modèle de Prédiction Énergétique Seattle
        
        Cette application utilise des modèles d'apprentissage automatique pour prédire:
        - La consommation énergétique annuelle (SiteEnergyUse en kBtu)
        - Les émissions de gaz à effet de serre (TotalGHGEmissions en tonnes CO₂)
        
        **Données utilisées:**
        - Dataset: 2016 Building Energy Benchmarking (Seattle)
        - Périmètre: Bâtiments non-résidentiels uniquement
        - Features: Caractéristiques physiques, géographiques et énergétiques
        
        **Méthodologie:**
        - Feature engineering avancé (interactions, clustering géographique)
        - Transformation log pour les variables asymétriques
        - Encodage mixte (OneHot + Target Encoder)
        - Modèles: XGBoost/LightGBM avec régularisation
        
        **Performance typique:**
        - Consommation: R² ≈ 0.75-0.85
        - GHG: R² ≈ 0.50-0.65 (sans données énergétiques détaillées)
        
        **Limitations:**
        - Prédictions basées sur des moyennes historiques
        - Ne prend pas en compte les variations météo annuelles
        - Ne considère pas les rénovations post-2016
        
        **Contact:** Version 1.0 - 2026
        """)


if __name__ == "__main__":
    main()
