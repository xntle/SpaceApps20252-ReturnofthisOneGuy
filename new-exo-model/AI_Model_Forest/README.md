# Exoplanet Classification Model

This project implements a machine learning model to classify planetary candidates as either **CONFIRMED** exoplanets or **FALSE POSITIVE** detections using data from Kepler and TESS missions.

## Overview

The model uses a Random Forest classifier with advanced feature engineering to achieve high accuracy in distinguishing between real exoplanets and false positive detections. It's trained on a combined dataset of Kepler and TESS observations.

## Model Performance

- **Training Accuracy**: 88.74% on validation set
- **F1-Score**: 0.89 (weighted average)
- **Features**: 13 engineered features including planetary and stellar parameters

### Performance Breakdown:
- **Confirmed Exoplanets**: 82% precision, 90% recall
- **False Positives**: 94% precision, 88% recall

## Project Structure

```
AI_Model_Forest/
├── data/                                    # Dataset files
│   ├── NEW_combined_kepler_tess_exoplanets.csv  # Training data (10,609 samples)
│   ├── tess_tois_candidates.csv            # Test data  
│   └── kepler_koi_cumulative.csv           # Additional Kepler data
├── ml_model/                               # Model training and testing
│   ├── train_combined_model.py             # Main training script
│   ├── test_combined_model.py              # Model evaluation script
│   └── OLD_*.py                            # Legacy scripts
├── trained_model/                          # Saved model artifacts
│   ├── rf_combined_model.joblib            # Trained Random Forest model
│   ├── scaler_combined.joblib              # Feature scaler
│   ├── label_encoder_combined.joblib       # Label encoder
│   ├── imputer_medians_combined.joblib     # Missing value imputer
│   └── feature_columns_combined.txt        # Feature column names
├── demo_predictions.py                     # Demo script for predictions
└── requirements.txt                        # Python dependencies
```

## Features Used

The model uses 11 primary astronomical features plus 2 engineered features:

### Primary Features:
- `orbital_period_days` - Planet's orbital period 
- `transit_duration_hours` - Duration of planetary transit
- `transit_depth_ppm` - Depth of transit signal 
- `planet_radius_re` - Planet radius (Earth radii)
- `equilibrium_temp_k` - Planet's equilibrium temperature
- `insolation_flux_earth` - Stellar flux received (Earth units)
- `stellar_teff_k` - Star's effective temperature  
- `stellar_radius_re` - Star radius (Earth radii)
- `apparent_mag` - Star's apparent magnitude
- `ra`, `dec` - Sky coordinates

### Engineered Features:
- `R_PLANET_R_STAR_RATIO` - Planet-to-star radius ratio
- `DEPTH_PER_RADIUS` - Transit depth per planet radius

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn astropy joblib imbalanced-learn
```

### 2. Train the Model
```bash
python ml_model/train_combined_model.py
```

### 3. Test the Model  
```bash
python ml_model/test_combined_model.py
```

### 4. Run Demo Predictions
```bash
python demo_predictions.py
```

## Usage Example

```python
from demo_predictions import predict_exoplanet_disposition

# Define a planetary candidate
candidate = {
    'orbital_period_days': 365.25,
    'transit_duration_hours': 13.0,
    'transit_depth_ppm': 84.0,
    'planet_radius_re': 1.0,
    'equilibrium_temp_k': 288.0,
    'insolation_flux_earth': 1.0,
    'stellar_teff_k': 5778.0,
    'stellar_radius_re': 1.0,
    'apparent_mag': 4.83,
    'ra': 180.0,
    'dec': 0.0
}

# Get prediction
result = predict_exoplanet_disposition(candidate)
print(f"Classification: {result}")
```

## Model Pipeline

1. **Data Loading**: Loads combined Kepler+TESS dataset
2. **Feature Engineering**: Creates derived features and applies log transforms
3. **Data Preprocessing**: Handles missing values and scales features
4. **Class Balancing**: Uses SMOTE to balance confirmed vs false positive samples
5. **Hyperparameter Tuning**: Grid search optimization for Random Forest parameters
6. **Training**: Trains final model with optimal parameters
7. **Evaluation**: Tests on held-out validation set and TESS data

## Key Techniques

- **SMOTE Oversampling**: Balances the dataset to prevent bias toward false positives
- **Feature Engineering**: Creates physically meaningful ratios and relationships
- **Log Transformations**: Handles highly skewed astronomical distributions
- **Grid Search CV**: Optimizes model hyperparameters
- **Cross-Validation**: Ensures robust performance estimates

## Data Sources

- **Kepler**: NASA's Kepler Space Telescope observations
- **TESS**: NASA's Transiting Exoplanet Survey Satellite data
- **Combined Dataset**: 10,609 samples with confirmed dispositions

## Model Artifacts

The trained model saves the following artifacts:
- `rf_combined_model.joblib` - The trained Random Forest classifier
- `scaler_combined.joblib` - StandardScaler for feature normalization
- `label_encoder_combined.joblib` - Encoder for target labels
- `imputer_medians_combined.joblib` - Median values for missing data imputation
- `feature_columns_combined.txt` - Column names and order

## Notes

- The model shows high precision for false positives (94%) which is important for reducing false discoveries
- Earth-like planets may be classified as false positives due to their subtle signals
- Hot Jupiters and other easily detectable planets typically classify correctly as confirmed
- The model generalizes well from Kepler training data to TESS test data

## Future Improvements

- Include additional stellar and planetary parameters
- Experiment with ensemble methods beyond Random Forest
- Incorporate time-series features from light curves
- Add uncertainty quantification to predictions