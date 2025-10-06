# backend/ml_model/data_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Literal

# Define the common features only (excluding Kepler-unique features)
FEATURE_MAP = {
    'ORB_PERIOD': ('koi_period', 'pl_orbper'),
    'TRANSIT_DURATION': ('koi_duration', 'pl_trandurh'),
    'TRANSIT_DEPTH': ('koi_depth', 'pl_trandep'),
    'PLANET_RADIUS': ('koi_prad', 'pl_rade'),
    'EQUIL_TEMP': ('koi_teq', 'pl_eqt'),
    'INSOL_FLUX': ('koi_insol', 'pl_insol'),
    'STELLAR_TEMP': ('koi_steff', 'st_teff'),
    'STELLAR_LOGG': ('koi_slogg', 'st_logg'),
    'STELLAR_RADIUS': ('koi_srad', 'st_rad'),
    'RA_DECIMAL': ('ra', 'ra'),
    'DEC_DECIMAL': ('dec', 'dec'),
}

# --- A. Data Loading and Unification Function ---
def load_and_standardize_data(file_path: str, source: Literal['kepler', 'tess']):
    """Loads a CSV and renames columns to a unified standard."""
    df = pd.read_csv(file_path, comment='#', low_memory=False)

    # 1. Select the correct column names based on source
    renames = {}
    for unified_name, (kep_col, tess_col) in FEATURE_MAP.items():
        if source == 'kepler' and kep_col:
            renames[kep_col] = unified_name
        elif source == 'tess' and tess_col:
            renames[tess_col] = unified_name
    
    # Target column
    target_col = 'koi_disposition' if source == 'kepler' else 'tfopwg_disp'
    renames[target_col] = 'TARGET'
    
    # 2. Rename and select only the relevant columns
    df.rename(columns=renames, inplace=True)
    df = df.filter(items=list(FEATURE_MAP.keys()) + ['TARGET']).copy()

    # 3. Clean and ensure numeric types
    for col in df.columns.drop('TARGET', errors='ignore'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# --- B. Feature Engineering and Imputation Pipeline (CRITICAL) ---
def clean_and_scale_features(df, source_type: Literal['train', 'test'], metadata=None):
    """
    Cleans, imputes, and scales data, including new feature engineering.
    """    
    # --- STEP 0: Filter and copy features ---
    if source_type == 'train':
        df['TARGET'] = df['TARGET'].str.upper()
        df = df[df['TARGET'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    X = df.drop(columns=['TARGET'], errors='ignore').copy()
    
    # Define features for log-transformation
    LOG_COLS = ['ORB_PERIOD', 'TRANSIT_DEPTH', 'INSOL_FLUX']
    
    # ----------------------------------------------------
    # ⚠️ 1. FEATURE ENGINEERING (MUST HAPPEN BEFORE IMPUTATION/SCALING)
    # ----------------------------------------------------
    
    # Add small constant to avoid division/log by zero in edge cases
    epsilon = 1e-6 
    
    # 1. Planet-to-Star Radius Ratio (Planetary Radius / Stellar Radius)
    # This is highly predictive, as FPs often have high ratios.
    X['R_PLANET_R_STAR_RATIO'] = (X['PLANET_RADIUS'] + epsilon) / (X['STELLAR_RADIUS'] + epsilon)
    
    # 2. Transit Depth per Unit of Planetary Radius (Depth / R_Planet)
    # High Depth / low R_Planet might indicate a systematic error or a large body on a small star.
    X['DEPTH_PER_RADIUS'] = (X['TRANSIT_DEPTH'] + epsilon) / (X['PLANET_RADIUS'] + epsilon)

    # 3. Stellar Density Proxy (Stellar Radius / Stellar Gravity)
    # Used to probe the host star properties, which often differs between true systems and EBs.
    X['STELLAR_DENSITY_PROXY'] = (X['STELLAR_RADIUS'] + epsilon) / (10**X['STELLAR_LOGG'] + epsilon)
    
    # ----------------------------------------------------
    
    # 2. Impute Missing Values and Log Transform
    if source_type == 'train':
        imputer_medians = X.median(numeric_only=True)
        X_imputed = X.fillna(imputer_medians)
        
        for col in LOG_COLS:
            if col in X_imputed.columns: X_imputed[col] = np.log1p(X_imputed[col])

    elif source_type == 'test':
        X_imputed = X.fillna(metadata['imputer_medians'])
        
        for col in LOG_COLS:
            if col in X_imputed.columns: X_imputed[col] = np.log1p(X_imputed[col])
            
        # CRITICAL: ENFORCE FEATURE ORDER
        required_cols = metadata['feature_columns']
        X_imputed = X_imputed.reindex(columns=required_cols, fill_value=0.0)

    # 3. Feature Scaling (StandardScaler)
    if source_type == 'train':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        metadata = {
            'imputer_medians': imputer_medians,
            'scaler': scaler,
            'feature_columns': X_imputed.columns.tolist()
        }

    elif source_type == 'test':
        X_scaled = metadata['scaler'].transform(X_imputed)

    # 4. Label Encoding
    if source_type == 'train':
        y = df['TARGET'].str.upper() 
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        metadata['encoder'] = encoder
        
        return X_scaled, y_encoded, metadata
    
    return X_scaled, metadata