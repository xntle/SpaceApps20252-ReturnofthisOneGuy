# backend/ml_model/02_model_train.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from data_pipeline import load_and_standardize_data, clean_and_scale_features
from imblearn.over_sampling import SMOTE

# --- Configuration ---
KEPLER_FILE = os.path.join('data', 'kepler_koi_cumulative.csv')
MODEL_DIR = 'trained_model'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
# --- 1. Load and Prepare Kepler Training Data ---
print(f"1. Loading and standardizing data from {KEPLER_FILE}...")
kepler_df = load_and_standardize_data(KEPLER_FILE, source='kepler')
kepler_df.dropna(subset=['TARGET'], inplace=True)

# --- 2. Clean, Impute, Scale, and Encode the Training Data ---
print("2. Cleaning, imputing, and scaling features (fitting preprocessors)...")
X_full_scaled, y_encoded, metadata = clean_and_scale_features(
    kepler_df, source_type='train'
)

# --- 3. Split Data for Training and Internal Validation ---
# Use an 80/20 split for training and validation
print("3. Applying SMOTE to balance the training set...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"   -> Unbalanced Training set size: {X_train.shape[0]} samples")

# Initialize SMOTE
sm = SMOTE(random_state=42)

# Apply SMOTE only to the training portion (X_train, y_train)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Use the resampled data for training from now on:
X_train = X_train_resampled
y_train = y_train_resampled

print(f"   -> Balanced Training set size: {X_train.shape[0]} samples")


# --- 4. Hyperparameter Tuning using Grid Search (High Impact Step) ---
print("\n4. Starting Grid Search for optimal Random Forest parameters (Optimizing for F1-score)...")

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [15, None],
    'min_samples_leaf': [1, 5]
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=base_model, 
    param_grid=param_grid, 
    # ⚠️ New Optimization Metric: Use F1_weighted to balance precision/recall across all samples
    scoring='f1_weighted', 
    cv=3,
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Select the best model found by the grid search
rf_model = grid_search.best_estimator_

print(f"\n   -> Grid Search Complete. Best parameters: {grid_search.best_params_}")
print(f"   -> Best Cross-Validation Score: {grid_search.best_score_:.4f}")


# --- 5. Evaluate Performance on Validation Set ---
print("\n5. Evaluating best model performance (on 20% validation set):")
y_val_pred = rf_model.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
print(f"   -> Overall Accuracy: {accuracy:.4f}")
print("\n   -> Classification Report:")
target_names = metadata['encoder'].classes_
print(classification_report(y_val, y_val_pred, target_names=target_names))


# --- 6. Save Model and Preprocessing Artifacts ---
print("\n6. Saving final model and preprocessors...")

joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_classifier.joblib'))
joblib.dump(metadata['scaler'], os.path.join(MODEL_DIR, 'scaler.joblib'))
joblib.dump(metadata['encoder'], os.path.join(MODEL_DIR, 'label_encoder.joblib'))
joblib.dump(metadata['imputer_medians'], os.path.join(MODEL_DIR, 'imputer_medians.joblib')) 

feature_cols = metadata['feature_columns']
with open(os.path.join(MODEL_DIR, 'feature_columns.txt'), 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"   -> Artifacts saved to the '{MODEL_DIR}' directory.")
print("\nTraining pipeline successfully completed! Rerun prediction.py next.")