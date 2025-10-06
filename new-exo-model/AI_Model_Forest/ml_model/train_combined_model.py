# backend/ml_model/train_combined_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Configuration ---
# Use the new combined dataset
COMBINED_FILE = os.path.join('data', 'NEW_combined_kepler_tess_exoplanets.csv')
MODEL_DIR = 'trained_model'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 1. Data Loading and Initial Cleaning ---
print(f"1. Loading and cleaning data from {COMBINED_FILE}...")

# Load the file (assumes standard CSV/comment handling is no longer needed)
df = pd.read_csv(COMBINED_FILE, low_memory=False)

# Filter for the definitive binary classes only (user confirmed this is pre-filtered)
df['disposition'] = df['disposition'].str.upper().str.strip()
df_filtered = df[df['disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
print(f"   -> Samples retained: {len(df_filtered)}")

# Define all predictive features (excluding IDs/labels)
PREDICTIVE_FEATURES = [
    'orbital_period_days', 'transit_duration_hours', 'transit_depth_ppm',
    'planet_radius_re', 'equilibrium_temp_k', 'insolation_flux_earth',
    'stellar_teff_k', 'stellar_radius_re', 'apparent_mag', 'ra', 'dec'
]

X = df_filtered[PREDICTIVE_FEATURES].copy()
y = df_filtered['disposition']

# --- 2. Feature Engineering and Preprocessing ---
print("2. Applying feature engineering and initial imputation...")

# Define map for Feature Engineering (internal variable names)
FE_MAP = {
    'planet_radius_re': 'PLANET_RADIUS',
    'stellar_radius_re': 'STELLAR_RADIUS',
    'insolation_flux_earth': 'INSOL_FLUX',
    'orbital_period_days': 'ORB_PERIOD',
    'transit_depth_ppm': 'TRANSIT_DEPTH',
}

X.rename(columns=FE_MAP, inplace=True)

# Add small constant to avoid division/log by zero
epsilon = 1e-6 

# ⚠️ Feature Engineering (CRITICAL for accuracy)
X['R_PLANET_R_STAR_RATIO'] = (X['PLANET_RADIUS'] + epsilon) / (X['STELLAR_RADIUS'] + epsilon)
X['DEPTH_PER_RADIUS'] = (X['TRANSIT_DEPTH'] + epsilon) / (X['PLANET_RADIUS'] + epsilon)

# 3. Handle Missing Data and Scaling
imputer_medians = X.median(numeric_only=True)
X_imputed = X.fillna(imputer_medians)

# Apply Log-transform to highly skewed features
LOG_COLS = ['ORB_PERIOD', 'TRANSIT_DEPTH', 'INSOL_FLUX']
for col in LOG_COLS:
    if col in X_imputed.columns: X_imputed[col] = np.log1p(X_imputed[col])

# Final feature set (including engineered columns)
FINAL_FEATURE_COLS = X_imputed.columns.tolist()

# Encode Target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
target_names = encoder.classes_ 

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# --- 3. Split Data for Training and Validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 4. Oversampling with SMOTE ---
print("4. Applying SMOTE to balance the training set...")
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(f"   -> Balanced Training set size: {X_train.shape[0]} samples")


# --- 5. Hyperparameter Tuning using Grid Search ---
print("\n5. Starting Grid Search for optimal Random Forest parameters (Optimizing for F1-score)...")

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [15, None],
    'min_samples_leaf': [1, 5]
}

base_model = RandomForestClassifier(random_state=42, class_weight=None, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=base_model, 
    param_grid=param_grid, 
    scoring='f1_weighted', 
    cv=3,
    verbose=0, 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
rf_model = grid_search.best_estimator_

print(f"\n   -> Best Cross-Validation Score: {grid_search.best_score_:.4f}")


# --- 6. Evaluate Performance and Save Artifacts ---
print("\n6. Evaluating final model performance on validation set...")
y_val_pred = rf_model.predict(X_val)

print(f"   -> Overall Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("\n   -> Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=target_names))


# Save the necessary artifacts (with new, distinct names)
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_combined_model.joblib'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_combined.joblib'))
joblib.dump(encoder, os.path.join(MODEL_DIR, 'label_encoder_combined.joblib'))
joblib.dump(imputer_medians, os.path.join(MODEL_DIR, 'imputer_medians_combined.joblib')) 

with open(os.path.join(MODEL_DIR, 'feature_columns_combined.txt'), 'w') as f:
    f.write('\n'.join(FINAL_FEATURE_COLS))

print(f"   -> Artifacts saved to the '{MODEL_DIR}' directory.")
print("\nTraining complete. You now have a single, robust model for deployment.")