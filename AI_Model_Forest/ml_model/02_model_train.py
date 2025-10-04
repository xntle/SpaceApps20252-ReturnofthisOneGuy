# backend/ml_model/02_model_train.py

import pandas as pd
import numpy as np
import joblib 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from data_pipeline import load_and_standardize_data, clean_and_scale_features

# --- Configuration ---
KEPLER_FILE = os.path.join('data', 'kepler_koi_cumulative.csv')
MODEL_DIR = 'trained_model'

# Ensure the trained_model directory exists to save artifacts
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
# --- 1. Load and Prepare Kepler Training Data ---
print(f"1. Loading and standardizing data from {KEPLER_FILE}...")
kepler_df = load_and_standardize_data(KEPLER_FILE, source='kepler')

# Drop rows where the TARGET is still missing/unlabeled (should be minimal in KOI cumulative)
kepler_df.dropna(subset=['TARGET'], inplace=True)

# --- 2. Clean, Impute, Scale, and Encode the Training Data ---
print("2. Cleaning, imputing, and scaling features (fitting preprocessors)...")

# We use the full Kepler dataset to fit the preprocessors (median imputer, scaler, encoder)
X_full_scaled, y_encoded, metadata = clean_and_scale_features(
    kepler_df, 
    source_type='train' # This mode calculates and saves the preprocessor parameters
)

# --- 3. Split Data for Training and Internal Validation ---
# Use an 80/20 split to train the model and check its performance immediately
X_train, X_val, y_train, y_val = train_test_split(
    X_full_scaled, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded # Ensures the class balance is maintained in both sets
)

print(f"   -> Training set size: {X_train.shape[0]} samples")
print(f"   -> Validation set size: {X_val.shape[0]} samples")


# --- 4. Initialize and Train the Random Forest Model ---
print("3. Training the Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,      # Number of trees (can be a hyperparameter)
    max_depth=12,          # Max depth to prevent deep overfitting
    random_state=42,
    class_weight='balanced', # Crucial for handling class imbalance (FP > Confirmed)
    n_jobs=-1              # Use all available CPU cores for speed
)

rf_model.fit(X_train, y_train)
print("   -> Training complete.")


# --- 5. Evaluate Performance on Validation Set ---
print("\n4. Evaluating model performance (on 20% validation set):")
y_val_pred = rf_model.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
print(f"   -> Overall Accuracy: {accuracy:.4f}")
print("\n   -> Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=metadata['encoder'].classes_))


# --- 6. Save Model and Preprocessing Artifacts ---
print("\n5. Saving model and preprocessors...")

# Save the trained model
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_classifier.joblib'))

# Save the fitted scaler (crucial for transforming new data from the Next.js frontend)
joblib.dump(metadata['scaler'], os.path.join(MODEL_DIR, 'scaler.joblib'))

# Save the fitted label encoder (crucial for decoding predictions back to 'CONFIRMED', etc.)
joblib.dump(metadata['encoder'], os.path.join(MODEL_DIR, 'label_encoder.joblib'))

# Save the imputer medians dictionary 
joblib.dump(metadata['imputer_medians'], os.path.join(MODEL_DIR, 'imputer_medians.joblib')) 


# Save the feature column order (CRITICAL for the API to match inputs)
feature_cols = metadata['feature_columns']
with open(os.path.join(MODEL_DIR, 'feature_columns.txt'), 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"   -> Artifacts saved to the '{MODEL_DIR}' directory.")
print("\nTraining pipeline successfully completed! You can now move to the prediction script.")