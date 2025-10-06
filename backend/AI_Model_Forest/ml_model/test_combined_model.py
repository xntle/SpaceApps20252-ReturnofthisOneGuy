# backend/ml_model/test_combined_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
from typing import Literal

# --- Configuration ---
TESS_FILE = os.path.join('data', 'tess_tois_candidates.csv') # Original TESS file for ground truth
MODEL_DIR = 'trained_model'

# --- 1. Load Model and Artifacts ---
print("1. Loading trained combined model and preprocessors...")

try:
    # Load NEW artifacts saved by train_combined_model.py
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_combined_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_combined.joblib'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_combined.joblib'))
    imputer_medians = joblib.load(os.path.join(MODEL_DIR, 'imputer_medians_combined.joblib')) 

    with open(os.path.join(MODEL_DIR, 'feature_columns_combined.txt'), 'r') as f:
        FEATURE_COLS_COMBINED = [line.strip() for line in f.readlines()]

except FileNotFoundError as e:
    print(f"\nERROR: Could not find combined model artifact. Did you run train_combined_model.py? Missing file: {e}")
    exit()
    
target_names = encoder.classes_

# --- 2. Data Loading and Feature Mapping ---
print(f"2. Loading TESS data from {TESS_FILE} and mapping features...")

# This loads the original TESS file and maps it to the combined model's expected features
tess_df = pd.read_csv(TESS_FILE, comment='#', low_memory=False)

# Define column mapping from TESS original file to the model's expected features
TESS_COL_MAP = {
    'pl_orbper': 'ORB_PERIOD', 'pl_trandurh': 'transit_duration_hours', 'pl_trandep': 'TRANSIT_DEPTH',
    'pl_rade': 'PLANET_RADIUS', 'pl_eqt': 'equilibrium_temp_k', 'pl_insol': 'INSOL_FLUX',
    'st_teff': 'stellar_teff_k', 'st_rad': 'STELLAR_RADIUS', 'st_tmag': 'apparent_mag', 
    'ra': 'ra', 'dec': 'dec',
    'tfopwg_disp': 'TARGET' # Ground Truth Label
}

# Select and rename columns
X_tess = tess_df.rename(columns=TESS_COL_MAP).filter(items=FEATURE_COLS_COMBINED + ['TARGET']).copy()

# Ensure all columns are numeric (converts errors/strings to NaN)
for col in X_tess.columns.drop('TARGET', errors='ignore'):
    X_tess[col] = pd.to_numeric(X_tess[col], errors='coerce')


# --- 3. Feature Engineering and Preprocessing (Applying Combined Model's Logic) ---
print("3. Applying feature engineering and scaling...")

# 3a. Re-Engineer Features for TESS Data
epsilon = 1e-6 
X_tess['R_PLANET_R_STAR_RATIO'] = (X_tess['PLANET_RADIUS'] + epsilon) / (X_tess['STELLAR_RADIUS'] + epsilon)
X_tess['DEPTH_PER_RADIUS'] = (X_tess['TRANSIT_DEPTH'] + epsilon) / (X_tess['PLANET_RADIUS'] + epsilon)

# 3b. Imputation (using combined training medians)
X_imputed = X_tess.drop(columns=['TARGET'], errors='ignore').fillna(imputer_medians)

# 3c. Log Transform
LOG_COLS = ['ORB_PERIOD', 'TRANSIT_DEPTH', 'INSOL_FLUX']
for col in LOG_COLS:
    if col in X_imputed.columns: X_imputed[col] = np.log1p(X_imputed[col])

# 3d. Final Scaling and Order Enforcement
X_imputed = X_imputed.reindex(columns=FEATURE_COLS_COMBINED, fill_value=0.0)
X_tess_scaled = scaler.transform(X_imputed)


# --- 4. Prediction and Output ---
print("4. Generating predictions...")
tess_predictions_encoded = rf_model.predict(X_tess_scaled)
tess_predictions_labels = encoder.inverse_transform(tess_predictions_encoded)

X_tess['Predicted_Disposition'] = tess_predictions_labels


# --- 5. Evaluate Against TESS Ground Truth ---

# TESS Disposition Mapping Function (defined in your original script)
def map_tess_to_binary(label: str) -> str:
    if pd.isna(label): return 'UNKNOWN'
    label = label.upper().strip()
    if label in ['PC', 'CP', 'KP', 'APC']: return 'CONFIRMED'
    elif label in ['FP', 'FA']: return 'FALSE POSITIVE'
    return 'UNKNOWN'

X_tess['TESS_TRUE_BINARY'] = X_tess['TARGET'].apply(map_tess_to_binary)
tess_eval_df = X_tess[X_tess['TESS_TRUE_BINARY'] != 'UNKNOWN'].copy()

# Prepare true and predicted labels for the report
y_true = tess_eval_df['TESS_TRUE_BINARY']
y_pred = tess_eval_df['Predicted_Disposition']

print("\n--- FINAL EVALUATION (Combined Model on TESS Data) ---")
if len(y_true) == 0:
    print("WARNING: No valid TESS ground truth labels found for evaluation.")
else:
    print(f"Total TESS Targets Evaluated: {len(y_true)}")
    
    # Generate the full classification report
    print("\n   -> Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    print("   -> Overall Accuracy on TESS Data: {:.4f}".format(accuracy_score(y_true, y_pred)))