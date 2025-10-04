# backend/ml_model/data_pipeline.py

import pandas as pd
import numpy as np
import joblib
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
    'RA_DECIMAL': ('ra', 'ra'),           # RA (Both files have decimal degree columns)
    'DEC_DECIMAL': ('dec', 'dec'),        # Dec (Both files have decimal degree columns)
    # Note: koi_time0bk / pl_tranmid (Transit Midpoint) is often excluded as it's not a physical property
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
    
    # Select only the columns that have a unified name
    df = df.filter(items=list(FEATURE_MAP.keys()) + ['TARGET']).copy()

    # 3. Clean and ensure numeric types
    for col in df.columns.drop('TARGET', errors='ignore'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# --- B. Feature Engineering and Imputation Pipeline (CRITICAL) ---
def clean_and_scale_features(df, source_type: Literal['train', 'test'], metadata=None):
    """
    Cleans, imputes, and scales data.
    - If source_type='train', it calculates and returns metadata (median, scaler, encoder, columns).
    - If source_type='test', it applies the provided metadata from training.
    """    
    # --- STEP 0: Filter out CANDIDATEs for Binary Classification ---
    if source_type == 'train':
        df['TARGET'] = df['TARGET'].str.upper()
        # Filter: keep only the two definitive outcomes
        df = df[df['TARGET'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    X = df.drop(columns=['TARGET'], errors='ignore').copy()
    
    # Define features for log-transformation (highly skewed data)
    LOG_COLS = ['ORB_PERIOD', 'TRANSIT_DEPTH', 'INSOL_FLUX']

    # 1. Impute Missing Values (NaN)
    if source_type == 'train':
        # Fit Imputation: Calculate median from the training data
        imputer_medians = X.median(numeric_only=True)
        X_imputed = X.fillna(imputer_medians)
        
        # Apply Log-transform
        for col in LOG_COLS:
            X_imputed[col] = np.log1p(X_imputed[col])

    elif source_type == 'test':
        # Apply Imputation: Use medians calculated from the training data
        X_imputed = X.fillna(metadata['imputer_medians'])
        
        # Apply Log-transform
        for col in LOG_COLS:
            X_imputed[col] = np.log1p(X_imputed[col])
            
        # ⚠️ CRITICAL: ENFORCE FEATURE ORDER
        # Use the column list saved during the 'fit' stage to enforce exact order
        required_cols = metadata['feature_columns']
        # Re-index to enforce order (fill_value=0.0 is technically not needed here but kept for safety)
        X_imputed = X_imputed.reindex(columns=required_cols, fill_value=0.0)
    
    # 2. Feature Scaling (StandardScaler)
    if source_type == 'train':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Store metadata
        metadata = {
            'imputer_medians': imputer_medians,
            'scaler': scaler,
            'feature_columns': X_imputed.columns.tolist()
        }

    elif source_type == 'test':
        # Apply fitted scaler from training
        X_scaled = metadata['scaler'].transform(X_imputed)


    # 3. Label Encoding (Only for the target variable on training data)
    if source_type == 'train':
        y = df['TARGET'].str.upper()
        
        # Encoder is fit ONLY on 'CONFIRMED' and 'FALSE POSITIVE'
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        metadata['encoder'] = encoder
        
        return X_scaled, y_encoded, metadata
    
    # Return features for prediction
    return X_scaled, metadata