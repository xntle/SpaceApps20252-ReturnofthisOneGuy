# 🚀 Exoplanet Detection Pipeline - Inference Guide

## 📋 Quick Start

This repository contains both **tabular-only** and **multimodal** exoplanet detection models. Choose your approach based on available data:

- **Tabular-Only**: 93.6% accuracy with just stellar parameters (recommended for most use cases)
- **Multimodal**: Enhanced accuracy when CNN data is available (requires lightcurve and pixel data)

## 🔑 Key Files for Inference

### Core Model Files
- `models/enhanced_multimodal_fusion_model.pth` - Trained multimodal model (1.18M parameters)
- `models/tabular_model.pth` - Standalone tabular model (52K parameters)
- `inference_template.py` - Ready-to-use prediction template
- `demo_inference.py` - Interactive inference demo

### Data Requirements Scripts
- `inference_data_requirements.py` - Comprehensive data requirements guide
- `check_data_overlap.py` - Data leakage analysis (shows clean splits)
- `analyze_performance_gap.py` - Performance comparison analysis

### Training and Evaluation
- `train_multimodal_enhanced.py` - Enhanced multimodal training pipeline
- `analyze_tabular_standalone.py` - Tabular model analysis
- `src/models.py` - Model architectures (TabularNet, ResidualCNN1D, PixelCNN2D)

## 🎯 Making Predictions

### Option 1: Tabular-Only Prediction (Recommended)
**Best for: Most use cases with stellar parameters only**

```python
import torch
import numpy as np
from src.models import TabularNet

# Load tabular model
model = TabularNet(input_size=39)
model.load_state_dict(torch.load('models/tabular_model.pth'))
model.eval()

# Your 39 stellar/transit features (standardized)
features = np.array([...])  # Replace with actual features

# Make prediction
with torch.no_grad():
    prediction = model(torch.FloatTensor(features).unsqueeze(0))
    probability = torch.sigmoid(prediction).item()

print(f"Planet probability: {probability:.3f}")
print(f"Classification: {'CONFIRMED' if probability > 0.5 else 'FALSE POSITIVE'}")
```

### Option 2: Multimodal Prediction (Advanced)
**Best for: When lightcurve and pixel data are available**

```python
from inference_template import predict_exoplanet

# Required: 39 tabular features
tabular_features = np.array([...])  # Stellar parameters

# Optional: CNN data (use None if not available)
residual_windows = np.array([...])  # Shape: (5, 128) - lightcurve residuals
pixel_diffs = np.array([...])       # Shape: (32, 24, 24) - pixel differences

result = predict_exoplanet(
    kepid=10797460,
    tabular_features=tabular_features,
    residual_windows=residual_windows,  # Optional
    pixel_diffs=pixel_diffs            # Optional
)

print(f"Result: {result['classification']} (confidence: {result['confidence']:.3f})")
```

## 📊 Required Data Formats

### Tabular Features (39 features - ALWAYS REQUIRED)
The model expects these specific features in order:

1. **Transit Parameters**: period, epoch, depth, duration, impact (+ uncertainties)
2. **Stellar Parameters**: temperature, radius, mass, surface gravity, metallicity (+ uncertainties)  
3. **Derived Features**: duty_cycle, log_period, log_duration, error ratios, etc.

**Full list (run for details):**
```bash
python inference_data_requirements.py
```

### CNN Data (OPTIONAL - for multimodal)
- **1D CNN**: Residual windows from lightcurve analysis - shape `(5, 128)`
- **2D CNN**: Pixel differences from Target Pixel Files - shape `(32, 24, 24)`

## 🚀 Quick Demo

### Test Inference
```bash
# See complete data requirements
python inference_data_requirements.py

# Interactive inference demo
python demo_inference.py --interactive

# Analyze model performance
python analyze_performance_gap.py
```

### Verify Setup
```bash
# Check data availability
python check_data_overlap.py

# Test model loading
python -c "from inference_template import predict_exoplanet; print('✅ Models loaded successfully')"
```

## 📈 Model Performance

### Current Results
- **Tabular-Only**: 93.6% validation accuracy (52K parameters)
- **Multimodal**: 87.2% validation accuracy (1.18M parameters)

### Why Tabular Outperforms
Our analysis (`analyze_performance_gap.py`) shows:
1. **Low CNN Coverage**: Only 3.9% of targets have CNN data
2. **Overfitting**: 22.6x more parameters with limited multimodal data
3. **Missing Data Handling**: Model degrades with 96% missing CNN inputs

### Recommendations
- **Use tabular-only** for reliable 93.6% accuracy
- **Expand CNN coverage to 10%+** before expecting multimodal benefits
- **Consider ensemble approaches** for production systems

## 🔍 Data Sources

### Stellar Parameters
Download from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/):
- Kepler Objects of Interest (KOI)
- TESS Objects of Interest (TOI)
- Confirmed exoplanets

### Lightcurve Data (Optional)
```python
from src.features import download_lightcurve, create_residual_windows
lc = download_lightcurve("10797460", mission="Kepler")
windows = create_residual_windows(lc, period, t0, duration)
```

### Pixel Data (Optional)
```python
from src.pixel_diff import download_target_pixel_file, compute_pixel_differences
tpf = download_target_pixel_file("10797460", mission="Kepler")
pixel_diffs = compute_pixel_differences(tpf, period, t0, duration)
```

## 🛠️ Setup Instructions

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Models
Trained models should be in `models/` directory:
- `enhanced_multimodal_fusion_model.pth`
- `tabular_model.pth`

### Verify Installation
```bash
python -c "import torch; from src.models import TabularNet; print('✅ Setup complete')"
```

## 📚 Architecture Details

### Tabular Model
- **Input**: 39 standardized features
- **Architecture**: 3-layer MLP with batch normalization and dropout
- **Output**: Binary classification (planet/non-planet)
- **Strengths**: Robust, fast, excellent performance

### Multimodal Model
- **TabularNet**: Processes stellar parameters
- **ResidualCNN1D**: Analyzes lightcurve residuals  
- **PixelCNN2D**: Processes pixel-level differences
- **Fusion Network**: Combines all modalities
- **Strengths**: Comprehensive analysis when data is available

## ⚠️ Important Notes

1. **Data Preprocessing**: All inputs must be standardized using the same preprocessing pipeline
2. **Missing Data**: CNN components accept zeros for missing data
3. **Performance**: Tabular-only currently outperforms multimodal due to data coverage limitations
4. **Validation**: No data leakage detected - performance metrics are trustworthy

## 🤝 Contributing

To improve multimodal performance:
1. Expand CNN data coverage beyond current 3.9%
2. Implement advanced missing data strategies
3. Optimize fusion architecture for sparse multimodal data

---

**Quick Start**: Use `inference_template.py` for immediate predictions with your stellar parameters!