# 🚀 Exoplanet Detection Fusion API - User Guide

## 📖 Overview

This API provides a unified multi-modal exoplanet detection system that combines:
- **Random Forest** (tabular features) - 88.7% accuracy
- **Residual CNN** (1D time series) - 60.1% PR-AUC  
- **Pixel CNN** (2D image classification) - 63.2% PR-AUC
- **XGBoost Fusion** (meta-model) - **89.5% PR-AUC**

The system intelligently combines predictions from all available models and provides a final classification decision.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- 8GB+ RAM recommended (for model loading)

### 1. Clone the Repository
```bash
git clone https://github.com/RoshanKattil/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional Packages (if needed)
```bash
pip install fastapi uvicorn python-multipart xgboost pyyaml
```

### 5. Start the Fusion API Server
```bash
# From project root directory
python -m uvicorn serve_fusion:app --host 0.0.0.0 --port 8001
```

The API will be available at: **http://localhost:8001**

---

## 🔍 API Endpoints

### Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "random_forest": true,
    "residual_cnn": true, 
    "pixel_cnn": true,
    "stacker": true
  },
  "device": "cpu",
  "threshold": 0.43
}
```

### Exoplanet Prediction
```bash
POST /predict_exoplanet
```

---

## 📝 How to Use the Prediction Endpoint

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `kepid` | integer | No | Kepler ID for reference |
| `features` | JSON string | No* | 13 tabular features (see below) |
| `residual_window_path` | string | No* | Path to residual .npy file |
| `pixel_image_path` | string | No* | Path to pixel difference .npy file |

*At least one model input is recommended for meaningful predictions

### Required Tabular Features

The `features` parameter should be a JSON string containing these 13 features:

```json
{
  "ORB_PERIOD": 12.3,              // Orbital period (days)
  "transit_duration_hours": 2.1,   // Transit duration (hours)
  "TRANSIT_DEPTH": 120,            // Transit depth (ppm)
  "PLANET_RADIUS": 1.5,            // Planet radius (Earth radii)
  "equilibrium_temp_k": 300,       // Equilibrium temperature (K)
  "INSOL_FLUX": 1.2,               // Insolation flux
  "stellar_teff_k": 5800,          // Stellar effective temperature (K)
  "STELLAR_RADIUS": 1.1,           // Stellar radius (Solar radii)
  "apparent_mag": 14.2,            // Apparent magnitude
  "ra": 285.0,                     // Right ascension (degrees)
  "dec": 45.0,                     // Declination (degrees)
  "R_PLANET_R_STAR_RATIO": 0.02,   // Planet-to-star radius ratio
  "DEPTH_PER_RADIUS": 80           // Transit depth per planet radius
}
```

### Data File Requirements

#### Residual Window File (`residual_window_path`)
- **Format**: `.npy` file
- **Shape**: `(2, 128)` - Two channels with 128 features each
- **Content**: Standardized residual time series data

#### Pixel Image File (`pixel_image_path`)  
- **Format**: `.npy` file
- **Shape**: `(1, 24, 24)` or `(24, 24)` - Single channel 24x24 pixel image
- **Content**: Pixel difference image data

---

## 💡 Usage Examples

### Example 1: Complete Multi-Modal Prediction
```bash
curl -X POST "http://localhost:8001/predict_exoplanet" \
  -F 'kepid=10014097' \
  -F 'features={"ORB_PERIOD": 12.3, "transit_duration_hours": 2.1, "TRANSIT_DEPTH": 120, "PLANET_RADIUS": 1.5, "equilibrium_temp_k": 300, "INSOL_FLUX": 1.2, "stellar_teff_k": 5800, "STELLAR_RADIUS": 1.1, "apparent_mag": 14.2, "ra": 285.0, "dec": 45.0, "R_PLANET_R_STAR_RATIO": 0.02, "DEPTH_PER_RADIUS": 80}' \
  -F "residual_window_path=/path/to/residual_10014097.npy" \
  -F "pixel_image_path=/path/to/pixdiff_10014097_clean.npy"
```

### Example 2: Tabular Features Only
```bash
curl -X POST "http://localhost:8001/predict_exoplanet" \
  -F 'kepid=12345' \
  -F 'features={"ORB_PERIOD": 365.25, "transit_duration_hours": 6.5, "TRANSIT_DEPTH": 84, "PLANET_RADIUS": 1.0, "equilibrium_temp_k": 288, "INSOL_FLUX": 1.0, "stellar_teff_k": 5778, "STELLAR_RADIUS": 1.0, "apparent_mag": 10.5, "ra": 180.0, "dec": 0.0, "R_PLANET_R_STAR_RATIO": 0.01, "DEPTH_PER_RADIUS": 84}'
```

### Example 3: CNN Models Only
```bash
curl -X POST "http://localhost:8001/predict_exoplanet" \
  -F 'kepid=67890' \
  -F "residual_window_path=/path/to/residual_67890.npy" \
  -F "pixel_image_path=/path/to/pixdiff_67890_clean.npy"
```

---

## 📊 Response Format

### Successful Prediction Response
```json
{
  "kepid": 10014097,
  "p_rf": 0.85,                    // Random Forest probability
  "p_residual": 0.62,              // Residual CNN probability  
  "p_pixel": 0.41,                 // Pixel CNN probability
  "p_final": 0.74,                 // Fusion probability (XGBoost)
  "decision": "CONFIRMED"          // Final classification
}
```

### Decision Logic
- **CONFIRMED**: `p_final >= 0.43` (optimized threshold)
- **FALSE_POSITIVE**: `p_final < 0.43`

### Model Availability
- Missing models automatically use neutral fill values
- Availability tracked internally with mask bits
- Fusion adapts to available model combinations

---

## 🚨 Error Handling

### Common Issues

#### 1. Model Loading Errors
```json
{"error": "Model failed to load", "detail": "..."}
```
**Solution**: Ensure all model files exist in the correct paths

#### 2. Invalid Feature Format
```json
{"error": "invalid features JSON"}
```
**Solution**: Verify JSON formatting and required feature names

#### 3. File Not Found
```json
{"error": "File not found: /path/to/file.npy"}
```
**Solution**: Check file paths and permissions

#### 4. Port Already in Use
```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8001): address already in use
```
**Solution**: Use a different port or kill existing processes
```bash
pkill -f "uvicorn fusion.serve_fusion"
python -m uvicorn fusion.serve_fusion:app --host 0.0.0.0 --port 8002
```

---

## 📈 Performance Metrics

| Model | Metric | Performance |
|-------|--------|-------------|
| Random Forest | Accuracy | 88.7% |
| Residual CNN | PR-AUC | 60.1% |
| Pixel CNN | PR-AUC | 63.2% |
| **Fusion System** | **PR-AUC** | **89.5%** |

### Fusion Advantages
- **Robust**: Gracefully handles missing model inputs
- **Accurate**: Outperforms individual models significantly  
- **Fast**: Real-time inference (~100ms per prediction)
- **Scalable**: Handles concurrent requests efficiently

---

## 🔧 Development & Debugging

### View API Documentation
Visit **http://localhost:8001/docs** for interactive Swagger UI

### Test API Health
```bash
curl http://localhost:8001/health
```

### Enable Debug Logging
```bash
export PYTHONPATH="/path/to/exoplanet-detection-pipeline"
python -m uvicorn fusion.serve_fusion:app --host 0.0.0.0 --port 8001 --log-level debug
```

### Check Model Paths
Verify these paths exist in your installation:
```
AI_Model_Forest/trained_model/rf_combined_model.joblib
residual_model/models/residual_cnn_best_fold0.pt
pixel_CNN/models/pixel_cnn_best_fold0.pt
fusion/models/stacker_xgb.pkl
fusion/config.yaml
```

---

## 📞 Support

For issues, questions, or contributions:
- **Repository**: https://github.com/RoshanKattil/exoplanet-detection-pipeline
- **Issues**: Create a GitHub issue with detailed error logs
- **Documentation**: Check the repository README and code comments

---

## 🎯 Quick Start Checklist

- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Start API server (`uvicorn fusion.serve_fusion:app --port 8001`)
- [ ] Test health endpoint (`curl localhost:8001/health`)
- [ ] Try prediction with sample data
- [ ] Verify all models loaded successfully

**🚀 Ready to detect exoplanets with cutting-edge AI fusion!**