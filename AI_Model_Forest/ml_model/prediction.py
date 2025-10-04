# backend/ml_model/prediction.py

import pandas as pd
import joblib
import os
from data_pipeline import load_and_standardize_data, clean_and_scale_features
from typing import Literal
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
# Assuming you run this script from the 'backend/' directory (or adjusting path)
TESS_FILE = os.path.join('data', 'tess_tois_candidates.csv')
MODEL_DIR = 'trained_model'

def map_tess_to_binary(label: str) -> str:
    """Maps TESS TFOPWG dispositions to the model's binary output classes."""
    if pd.isna(label):
        return 'UNKNOWN'
    
    label = label.upper().strip()
    
    # Classify all planet/candidate labels as the model's 'CONFIRMED'
    if label in ['PC', 'CP', 'KP', 'APC']: # PC=Candidate, CP=Confirmed, KP=Known Planet, APC=Ambiguous
        return 'CONFIRMED'
    
    # Classify all definite rejection labels as 'FALSE POSITIVE'
    elif label in ['FP', 'FA']: # FP=False Positive, FA=False Alarm
        return 'FALSE POSITIVE'
    
    return 'UNKNOWN'

# --- 1. Load Model and Preprocessing Artifacts ---
print("1. Loading trained model and preprocessors...")

try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_classifier.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    # IMPORTANT: Load the medians dictionary for robust imputation
    imputer_medians = joblib.load(os.path.join(MODEL_DIR, 'imputer_medians.joblib')) 

    with open(os.path.join(MODEL_DIR, 'feature_columns.txt'), 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

except FileNotFoundError as e:
    print(f"\nERROR: Could not find model artifact. Did you run 02_model_train.py and ensure all artifacts were saved? Missing file: {e}")
    exit()

# Combine loaded artifacts into a single metadata dictionary
loaded_metadata = {
    'scaler': scaler,
    'encoder': encoder,
    'imputer_medians': imputer_medians, 
    'feature_columns': feature_cols 
}

# --- 2. Load and Prepare TESS Data ---
print(f"2. Loading and standardizing TESS data from {TESS_FILE}...")
# Note: The load_and_standardize_data function MUST be updated to use comment='#' for TESS too.
tess_df = load_and_standardize_data(TESS_FILE, source='tess')

# --- 3. Clean and Scale TESS Features ---
print("3. Cleaning and scaling TESS features using Kepler's fitted parameters...")

# The data_pipeline must ensure TESS data is in the same column order as Kepler
X_tess_scaled, _ = clean_and_scale_features(
    tess_df, 
    source_type='test', 
    metadata=loaded_metadata
)


# --- 4. Predict Dispositions ---
print("4. Generating predictions on TESS candidates...")
tess_predictions_encoded = rf_model.predict(X_tess_scaled)
tess_predictions_labels = encoder.inverse_transform(tess_predictions_encoded)

tess_df['Predicted_Disposition'] = tess_predictions_labels

# --- 5. Output Results ---
print("\n--- TESS Prediction Summary (Model applied to unseen data) ---")
print("TESS Targets predicted as Confirmed vs. False Positive:")
print(tess_df['Predicted_Disposition'].value_counts())

# Save the predicted TESS file
tess_df.to_csv(os.path.join('data', 'tess_predictions.csv'), index=False)
print(f"\nFull predictions saved to {os.path.join('data', 'tess_predictions.csv')} (File: tess_predictions.csv)")

# --- 6. Evaluate Against TESS Ground Truth ---
print("\n--- 6. Evaluating Model Performance on TESS Data (Ground Truth) ---")

# 1. Map the TESS actual disposition column ('TARGET' is mapped from 'tfopwg_disp')
tess_df['TESS_TRUE_BINARY'] = tess_df['TARGET'].apply(map_tess_to_binary)

# 2. Filter out objects with Unknown ground truth labels (e.g., TESS CANDIDATEs that haven't been vetted)
tess_eval_df = tess_df[tess_df['TESS_TRUE_BINARY'] != 'UNKNOWN'].copy()

# 3. Prepare true and predicted labels for the report
y_true = tess_eval_df['TESS_TRUE_BINARY']
y_pred = tess_eval_df['Predicted_Disposition']

# Check if there are any samples left to evaluate
if len(y_true) == 0:
    print("WARNING: No valid TESS ground truth labels found (CONFIRMED/FALSE POSITIVE) for evaluation.")
else:
    print(f"Total TESS Targets Evaluated: {len(y_true)}")
    
    # Generate the full classification report
    target_names = ['CONFIRMED', 'FALSE POSITIVE'] 
    
    print("\n   -> Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    print("   -> Overall Accuracy on TESS Data: {:.4f}".format(accuracy_score(y_true, y_pred)))