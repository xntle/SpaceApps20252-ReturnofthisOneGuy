# 🚀 Enhanced Multimodal Exoplanet Detection Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-red.svg)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-93.47%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **🔥 Quick Start**: Looking to make predictions? See **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** for immediate usage instructions!

**🏆 UPGRADED: 93.47% Validation Accuracy Enhanced Multi-Modal Pipeline**

A state-of-the-art multimodal machine learning pipeline for exoplanet detection combining tabular features with deep CNN analysis of Kepler light curves and target pixel files.

## 📋 Quick Navigation

### 🔥 **Ready to Use**
- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** - Complete prediction guide (START HERE)
- **[inference_template.py](inference_template.py)** - Drop-in prediction function
- **[FILE_STRUCTURE_GUIDE.md](FILE_STRUCTURE_GUIDE.md)** - Navigate all files easily

### 🧠 **Understanding the Models**  
- **[analyze_performance_gap.py](analyze_performance_gap.py)** - Why tabular outperforms multimodal
- **[check_data_overlap.py](check_data_overlap.py)** - Data validation (no leakage)
- **[MULTI_MODAL_RESULTS.md](MULTI_MODAL_RESULTS.md)** - Performance analysis

### 🛠️ **For Developers**
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Retrain models
- **[TECHNICAL_BREAKDOWN.md](TECHNICAL_BREAKDOWN.md)** - Architecture details
- **[CNN_COVERAGE_STRATEGY.md](CNN_COVERAGE_STRATEGY.md)** - Expand CNN dataed Multimodal Exoplanet Detection Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-red.svg)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-93.47%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**🏆 UPGRADED: 93.47% Validation Accuracy Enhanced Multi-Modal Pipeline**

A state-of-the-art multimodal machine learning pipeline for exoplanet detection combining tabular features with deep CNN analysis of Kepler light curves and target pixel files.

## 🎯 Performance Highlights

- **🏆 93.47% Validation Accuracy** on Kepler Object of Interest (KOI) classification
- **🤖 Multimodal Architecture**: TabularNet + ResidualCNN1D + PixelCNN2D fusion
- **📊 2.5% CNN Coverage**: 243 residual windows + 134 pixel difference arrays
- **⚡ Fast Training**: 54.6 seconds with early stopping at epoch 45
- **🎚️ Optimal Threshold**: 0.9922 (88.56% TPR at 4.55% FPR)

## 🏗️ Architecture Overview

```
Tabular Features (39) → TabularNet → 
                                      → Fusion Layer → Classification
Light Curves → ResidualCNN1D →      ↗
Target Pixels → PixelCNN2D →       ↗
```

### Model Components:
- **TabularNet**: Dense layers processing orbital parameters and stellar properties
- **ResidualCNN1D**: 1D CNN analyzing detrended light curve residuals
- **PixelCNN2D**: 2D CNN processing target pixel file differences
- **Fusion Layer**: Attention-weighted combination of multimodal features
## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.models import EnhancedMultiModalFusionModel
import torch

# Load trained model
model = EnhancedMultiModalFusionModel(
    tabular_dim=39,
    residual_length=128,
    pixel_shape=(32, 24, 24)
)
model.load_state_dict(torch.load('models/enhanced_multimodal_fusion_model.pth'))
model.eval()

# Make predictions (see full example below)
```

## 📊 Data Pipeline

### 1. Raw Data Sources
```bash
data/raw/
├── lighkurve_KOI_dataset_enriched.csv  # Main dataset with stellar parameters
├── all_global_synthetic.csv           # Synthetic global features  
└── all_local_synthetic.csv            # Synthetic local features
```

### 2. Pull and Process Data
```bash
# Download and enrich KOI dataset
python scripts/enrich_koi.py

# Generate CNN data (4.5 hours for 200 targets)
python scripts/rapid_cnn_expansion.py --max-targets 200 --max-time 270

# Standardize CNN data
python scripts/standardize_cnn_data.py
```

### 3. Processed Data Structure
```bash
data/processed/
├── residual_windows_std/     # 243 standardized residual windows (128 points)
└── pixel_diffs_std/         # 134 standardized pixel differences (32×24×24)
```

## 🎓 Training the Model

### Full Training Pipeline
```bash
# 1. Configure Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate/expand CNN data
python scripts/rapid_cnn_expansion.py --max-targets 200 --max-time 270

# 4. Standardize CNN data
python scripts/standardize_cnn_data.py

# 5. Train enhanced multimodal model
python train_multimodal_enhanced.py
```

### Training Configuration
```python
# Key training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20
MAX_EPOCHS = 100

# Data splits
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2  
TEST_RATIO = 0.2
```

### Expected Training Output
```
🌟 ENHANCED MULTI-MODAL EXOPLANET DETECTION PIPELINE
=================================================================
📊 Training samples: 462
📊 Validation samples: 245
Epoch  45: Loss=0.0840, Val AUC=0.9683, Val Acc=0.9347, Time=54.6s
🛑 Early stopping at epoch 45 (patience=20)
✅ Training complete! Best validation AUC: 0.9751
🏆 ENHANCED MULTI-MODAL RESULTS
🎯 Validation Accuracy: 93.47%
💾 Enhanced model saved to models/enhanced_multimodal_fusion_model.pth
```

## 🔮 Making Inferences

### Single Sample Prediction
```python
import torch
import pandas as pd
import numpy as np
from src.models import EnhancedMultiModalFusionModel
from src.data_loader import prepare_single_sample

# Load model
model = EnhancedMultiModalFusionModel(
    tabular_dim=39,
    residual_length=128, 
    pixel_shape=(32, 24, 24)
)
model.load_state_dict(torch.load('models/enhanced_multimodal_fusion_model.pth'))
model.eval()

# Prepare sample data
kepid = "10797460"  # Example KepID
sample_data = prepare_single_sample(kepid)

# Make prediction
with torch.no_grad():
    tabular_features = torch.FloatTensor(sample_data['tabular']).unsqueeze(0)
    residual_windows = torch.FloatTensor(sample_data['residual']).unsqueeze(0)
    pixel_diffs = torch.FloatTensor(sample_data['pixel']).unsqueeze(0)
    
    outputs = model(tabular_features, residual_windows, pixel_diffs)
    probability = torch.sigmoid(outputs).item()
    
    prediction = "CONFIRMED" if probability > 0.9922 else "FALSE POSITIVE"
    confidence = probability * 100
    
    print(f"KepID {kepid}: {prediction} ({confidence:.2f}% confidence)")
```

### Batch Predictions
```python
# Load test dataset
test_loader = create_test_dataloader(batch_size=32)

predictions = []
true_labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        tabular, residual, pixel, labels = batch
        outputs = model(tabular, residual, pixel)
        probs = torch.sigmoid(outputs)
        
        predictions.extend(probs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate metrics
from sklearn.metrics import accuracy_score, roc_auc_score

threshold = 0.9922  # Optimal threshold
pred_labels = (np.array(predictions) > threshold).astype(int)

accuracy = accuracy_score(true_labels, pred_labels)
auc = roc_auc_score(true_labels, predictions)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")
```

## � Data Generation Details

### CNN Data Expansion Strategy
The pipeline generates two types of CNN inputs:

#### 1. Residual Windows (1D CNN)
```python
# Generated from Kepler light curves
def create_residual_windows(lightcurve, period, t0, duration):
    """
    Extract transit residuals from detrended light curves
    
    Returns:
        windows: (n_transits, 128) array of residual windows
    """
```

#### 2. Pixel Differences (2D CNN)  
```python
# Generated from Target Pixel Files (TPF)
def compute_pixel_differences(tpf, period, t0, duration, phase_bins=32):
    """
    Compute phase-folded pixel differences
    
    Returns:
        diffs: (32, height, width) array of pixel differences
    """
```

### Data Coverage Scaling
```bash
# Start with high-priority confirmed planets
df_priority = df[
    (df['koi_disposition'] == 'CONFIRMED') &
    (df['koi_period'] >= 1) & (df['koi_period'] <= 50) &
    (df['koi_duration'] >= 1) & (df['koi_duration'] <= 12) &
    (df['koi_depth'] >= 10)
].head(200)

## 🎛️ Model Configuration

### Hyperparameters
```python
class ModelConfig:
    # Architecture
    TABULAR_DIM = 39
    RESIDUAL_LENGTH = 128
    PIXEL_SHAPE = (32, 24, 24)
    
    # TabularNet
    TABULAR_HIDDEN = [512, 256, 128]
    TABULAR_DROPOUT = 0.3
    
    # ResidualCNN1D  
    RESIDUAL_FILTERS = [64, 128, 256]
    RESIDUAL_KERNEL_SIZE = 7
    RESIDUAL_DROPOUT = 0.5
    
    # PixelCNN2D
    PIXEL_FILTERS = [32, 64, 128] 
    PIXEL_KERNEL_SIZE = 3
    PIXEL_DROPOUT = 0.4
    
    # Fusion
    FUSION_HIDDEN = 256
    FUSION_DROPOUT = 0.3
    
    # Training
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 20
```

## 📋 Dependencies

### Core Requirements
```txt
torch>=2.8.0
torchvision>=0.23.0
numpy>=2.3.3
pandas>=2.3.3
scikit-learn>=1.7.2
lightkurve>=2.5.1
astropy>=7.1.0
scikit-image>=0.25.2
scipy>=1.16.2
tqdm>=4.67.1
matplotlib>=3.10.6
```

### Optional for Development
```txt
jupyter>=1.0.0
ipykernel>=6.0.0
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
```

## � Performance Analysis

### Model Evolution
| Version | CNN Samples | Coverage | Val Accuracy | Val AUC |
|---------|-------------|----------|--------------|---------|
| Original | 70 | 0.7% | ~88.85% | 95.10% |
| Enhanced V1 | 139 | 0.7% | ~88.85% | 96.55% |
| **Enhanced V2** | **240** | **2.5%** | **93.47%** | **97.51%** |

### Coverage Impact
- **3.6x CNN coverage increase** (0.7% → 2.5%)
- **+4.62% validation accuracy** improvement  
- **Balanced dataset**: 113 confirmed + 127 false positives
- **Stable training**: Early stopping at epoch 45

### Optimal Operating Point
- **Threshold**: 0.9922
- **True Positive Rate**: 88.56%
- **False Positive Rate**: 4.55%
- **Precision**: ~95% (estimated)
- **Recall**: 88.56%

## 🗂️ Repository Structure

```
spaceapps2025/
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── train_multimodal_enhanced.py          # Main training script
├── demo_inference.py                     # Interactive inference demo
├── src/                                   # Core source code
│   ├── models.py                         # Model architectures
│   ├── data_loader.py                    # Data loading utilities
│   ├── features.py                       # Light curve processing
│   ├── pixel_diff.py                     # Pixel processing
│   ├── train.py                          # Training utilities
│   └── evaluate.py                       # Evaluation metrics
├── scripts/                              # Data generation scripts
│   ├── rapid_cnn_expansion.py           # CNN data expansion
│   ├── standardize_cnn_data.py          # Data standardization
│   └── enrich_koi.py                    # Dataset enrichment
├── data/                                 # Data directory
│   ├── raw/                             # Raw datasets
│   └── processed/                       # Processed CNN data
├── models/                              # Saved model checkpoints
│   └── enhanced_multimodal_fusion_model.pth
└── notebooks/                           # Jupyter notebooks
    └── demo_pipeline.ipynb              # Interactive demo
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Kepler Mission** for providing the light curve and target pixel file data
- **Lightkurve** for excellent astronomical data processing tools
- **PyTorch** for the deep learning framework
- **NASA Space Apps Challenge 2025** for inspiration

## 📚 Citation

```bibtex
@software{enhanced_multimodal_exoplanet_detection,
  title={Enhanced Multimodal Exoplanet Detection Pipeline},
  author={NASA Space Apps 2025 Team},
  year={2025},
  url={https://github.com/Shyam-723/NasaExoSkyChallenge}
}
```

## 🚀 Future Improvements

- [ ] Scale CNN coverage to 5%+ (500+ samples)
- [ ] Implement ensemble methods
- [ ] Add TESS mission support
- [ ] Optimize fusion architecture weights
- [ ] Add model interpretability features
- [ ] Deploy as web API service

---

**⭐ Star this repository if you find it useful!**

For questions or support, please open an issue or contact the development team.
  - **Training**: 25 epochs, best validation AUC 0.8549
- **Dataset Balance**: 23.8% confirmed exoplanets (realistic distribution)

### Expected Performance (Multi-Modal)
| Model | ROC-AUC | PR-AUC | Recall@1%FPR |
|-------|---------|--------|--------------|
| **TabularNet** (Real Data) | **0.78** | **0.41** | **0.79** |
| ResidualCNN1D | 0.85 | 0.42 | 0.72 |
| PixelCNN2D | 0.83 | 0.38 | 0.68 |
| **Fusion Stacker** | **0.93** | **0.58** | **0.85** |

*TabularNet performance measured on real NASA Kepler data. Other models projected.*

*TabularNet performance measured on real NASA Kepler data. Other models projected.*

## 🔧 Usage Examples

### Interactive Demo
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/demo_pipeline.ipynb
```

### Command Line Training
```bash
# Basic training
python src/train.py

# Advanced options
python src/train.py \
    --data-dir data/raw \
    --output-dir models \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --max-samples 1000 \
    --device cuda
```

### Programmatic Usage
```python
from src.data_loader import load_and_prepare_data
from src.models import create_models
from src.train import main_training_pipeline

# Load data
splits, scaler, features = load_and_prepare_data("data/raw")

# Create models  
tabular_net, residual_net, pixel_net = create_models(
    tabular_input_dim=len(features)
)

# Run full training pipeline
results = main_training_pipeline(
    data_dir="data/raw",
    epochs=50,
    batch_size=32
)
```

## 📈 Data Processing Pipeline

### 1. Light Curve Processing
- Download PDCSAP flux using Lightkurve
- Detrend with Savitzky-Golay filter
- Box Least Squares (BLS) period search
- Phase-fold and create residual windows
- Save as numpy arrays [2, 512]

### 2. Pixel Difference Processing  
- Download Target Pixel Files (TPF)
- Identify in-transit vs out-of-transit frames
- Create median difference images
- Normalize and resize to [1, 16, 16]
- Save as numpy arrays

### 3. Tabular Feature Processing
- Load and merge global + local features
- Handle missing values
- StandardScaler normalization
- Train/val/test split using GroupKFold

## 🔍 Key Features

- **Hybrid Architecture**: Combines tabular, time-series, and image data
- **End-to-end Pipeline**: From raw data to trained models
- **Robust Evaluation**: ROC, PR curves, confusion matrices, recall@FPR
- **GPU Support**: CUDA acceleration for training
- **Modular Design**: Easy to extend and modify components
- **Comprehensive Logging**: Detailed progress and error reporting
- **Jupyter Integration**: Interactive demo notebook

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt  # If requirements.txt exists
```

**CUDA Errors**: If GPU training fails, use CPU
```bash
python src/train.py --device cpu
```

**Memory Issues**: Reduce batch size
```bash
python src/train.py --batch-size 16
```

**Download Failures**: Lightkurve downloads may timeout
- Reduce the number of KOIs processed
- Check internet connection
- Some KOIs may not have available data

### Data Requirements

- **Minimum**: Tabular features only (TabularNet training)
- **Recommended**: Tabular + some light curves for residuals
- **Optimal**: All three data types for full hybrid training

## 📚 References & Data Sources

- **Primary Dataset**: "Automated Light Curve Processing for Exoplanet Detection Using Machine Learning Algorithms" (Macedo, B. H. D., & Zalewski, W., 2024)
  - 5,302 Kepler light curves with confirmed classifications
  - DOI: 10.17632/wctrv34962.3
- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Kepler/K2 Archive](https://archive.stsci.edu/kepler/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for real KOI datasets and dispositions
- **Kepler/K2 and TESS missions** for groundbreaking exoplanet observations  
- **Lightkurve library** for Kepler data access and processing
- **Dataset contributors** (Macedo et al.) for curated machine learning training data
- **PyTorch and Scikit-learn communities** for excellent ML frameworks

## 🤝 Contributing

This pipeline was developed for NASA Space Apps Challenge 2025. Feel free to:
- Report issues and bugs
- Suggest improvements  
- Add new features
- Optimize performance

## 📄 License

MIT License - Feel free to use for research and educational purposes.

---

**Happy Exoplanet Hunting! 🌍🔭✨**