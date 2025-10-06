#!/usr/bin/env python3
"""
Demo script showing how to use the trained exoplanet classification model
to make predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List

# --- Configuration ---
MODEL_DIR = 'trained_model'

def load_model_artifacts():
    """Load all trained model artifacts."""
    print("Loading trained model and preprocessors...")
    
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_combined_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_combined.joblib'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_combined.joblib'))
    imputer_medians = joblib.load(os.path.join(MODEL_DIR, 'imputer_medians_combined.joblib'))
    
    with open(os.path.join(MODEL_DIR, 'feature_columns_combined.txt'), 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    return rf_model, scaler, encoder, imputer_medians, feature_cols

def preprocess_data(data: pd.DataFrame, imputer_medians: pd.Series, feature_cols: List[str]) -> np.ndarray:
    """Apply the same preprocessing pipeline used during training."""
    
    # Feature engineering
    epsilon = 1e-6
    data['R_PLANET_R_STAR_RATIO'] = (data['PLANET_RADIUS'] + epsilon) / (data['STELLAR_RADIUS'] + epsilon)
    data['DEPTH_PER_RADIUS'] = (data['TRANSIT_DEPTH'] + epsilon) / (data['PLANET_RADIUS'] + epsilon)
    
    # Imputation
    data_imputed = data.fillna(imputer_medians)
    
    # Log transformation
    log_cols = ['ORB_PERIOD', 'TRANSIT_DEPTH', 'INSOL_FLUX']
    for col in log_cols:
        if col in data_imputed.columns:
            data_imputed[col] = np.log1p(data_imputed[col])
    
    # Ensure column order matches training
    data_final = data_imputed.reindex(columns=feature_cols, fill_value=0.0)
    
    return data_final

def predict_exoplanet_disposition(sample_data: Dict) -> str:
    """
    Predict whether a planetary candidate is a confirmed exoplanet or false positive.
    
    Parameters:
    -----------
    sample_data : dict
        Dictionary containing the following keys:
        - orbital_period_days: Orbital period in days
        - transit_duration_hours: Transit duration in hours  
        - transit_depth_ppm: Transit depth in parts per million
        - planet_radius_re: Planet radius in Earth radii
        - equilibrium_temp_k: Equilibrium temperature in Kelvin
        - insolation_flux_earth: Insolation flux relative to Earth
        - stellar_teff_k: Stellar effective temperature in Kelvin
        - stellar_radius_re: Stellar radius in Earth radii  
        - apparent_mag: Apparent magnitude
        - ra: Right ascension in degrees
        - dec: Declination in degrees
    
    Returns:
    --------
    str: Either 'CONFIRMED' or 'FALSE POSITIVE'
    """
    
    # Load model artifacts
    rf_model, scaler, encoder, imputer_medians, feature_cols = load_model_artifacts()
    
    # Convert input to DataFrame with correct column names
    feature_map = {
        'orbital_period_days': 'ORB_PERIOD',
        'transit_duration_hours': 'transit_duration_hours', 
        'transit_depth_ppm': 'TRANSIT_DEPTH',
        'planet_radius_re': 'PLANET_RADIUS',
        'equilibrium_temp_k': 'equilibrium_temp_k',
        'insolation_flux_earth': 'INSOL_FLUX',
        'stellar_teff_k': 'stellar_teff_k',
        'stellar_radius_re': 'STELLAR_RADIUS',
        'apparent_mag': 'apparent_mag',
        'ra': 'ra',
        'dec': 'dec'
    }
    
    # Create DataFrame from input
    mapped_data = {feature_map[k]: v for k, v in sample_data.items() if k in feature_map}
    df = pd.DataFrame([mapped_data])
    
    # Preprocess the data
    df_processed = preprocess_data(df, imputer_medians, feature_cols)
    
    # Scale the features
    df_scaled = scaler.transform(df_processed)
    
    # Make prediction
    prediction_encoded = rf_model.predict(df_scaled)[0]
    prediction_label = encoder.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probability
    probabilities = rf_model.predict_proba(df_scaled)[0]
    confidence = np.max(probabilities)
    
    print(f"Prediction: {prediction_label}")
    print(f"Confidence: {confidence:.3f}")
    
    return prediction_label

# --- Demo Examples ---
if __name__ == "__main__":
    print("=== Exoplanet Classification Demo ===\n")
    
    # Example 1: Typical confirmed exoplanet (Earth-like)
    print("Example 1: Earth-like candidate")
    earth_like = {
        'orbital_period_days': 365.25,
        'transit_duration_hours': 13.0,
        'transit_depth_ppm': 84.0,  # Earth's transit depth
        'planet_radius_re': 1.0,    # Earth radius
        'equilibrium_temp_k': 288.0, # Earth's temperature
        'insolation_flux_earth': 1.0, # Earth's insolation
        'stellar_teff_k': 5778.0,   # Sun's temperature
        'stellar_radius_re': 1.0,   # Sun's radius
        'apparent_mag': 4.83,       # Sun's apparent magnitude
        'ra': 180.0,
        'dec': 0.0
    }
    
    result1 = predict_exoplanet_disposition(earth_like)
    print()
    
    # Example 2: Hot Jupiter (likely confirmed)
    print("Example 2: Hot Jupiter candidate")
    hot_jupiter = {
        'orbital_period_days': 3.5,
        'transit_duration_hours': 2.8,
        'transit_depth_ppm': 10000.0,  # Large transit depth
        'planet_radius_re': 11.2,      # Jupiter-like radius
        'equilibrium_temp_k': 1500.0,  # Very hot
        'insolation_flux_earth': 2000.0, # High insolation
        'stellar_teff_k': 6200.0,
        'stellar_radius_re': 1.2,
        'apparent_mag': 12.0,
        'ra': 45.0,
        'dec': 30.0
    }
    
    result2 = predict_exoplanet_disposition(hot_jupiter)
    print()
    
    # Example 3: Suspicious candidate (likely false positive)
    print("Example 3: Suspicious candidate with unusual parameters")
    suspicious = {
        'orbital_period_days': 0.5,     # Very short period
        'transit_duration_hours': 0.1,  # Very short transit
        'transit_depth_ppm': 50000.0,   # Extremely deep transit
        'planet_radius_re': 50.0,       # Unrealistically large
        'equilibrium_temp_k': 3000.0,   # Extremely hot
        'insolation_flux_earth': 10000.0, # Extreme insolation
        'stellar_teff_k': 3000.0,       # Cool star
        'stellar_radius_re': 0.1,       # Very small star
        'apparent_mag': 20.0,           # Very faint
        'ra': 270.0,
        'dec': -45.0
    }
    
    result3 = predict_exoplanet_disposition(suspicious)
    print()
    
    print("=== Demo Complete ===")
    print("\nThe model has been successfully trained and tested!")
    print("You can use the predict_exoplanet_disposition() function to classify new candidates.")