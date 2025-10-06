# Residual CNN Model for Exoplanet Detection

This module implements a 1D Convolutional Neural Network for classifying exoplanet candidates based on residual light curve data.

## Overview

The Residual CNN analyzes phase-folded residual windows from Kepler light curves to distinguish between confirmed exoplanets and false positive detections. It handles variable-length sequences and incorporates deep residual connections for improved feature learning.

## Data Format

- **Input**: Residual window files as `.npy` arrays with shape `[seq_len, 128]`
- **Features**: 128-dimensional residual flux features per time step
- **Sequence Length**: Variable (10 to 4000+ time steps)
- **Labels**: Binary classification (CONFIRMED=1, FALSE POSITIVE=0)

## Model Architecture

### ResidualCNN1D (Main Model)
- **Feature Projection**: Maps 128 input features to 64 channels
- **Stem**: Large kernel (15) convolution with pooling
- **Residual Blocks**: 5 blocks with varying kernel sizes (9,7,5)
- **Wide Context**: Large kernel (21) for temporal patterns
- **Global Pooling**: Adaptive average pooling
- **Classification Head**: Multi-layer MLP with dropout

### LightweightResidualCNN (Alternative)
- Faster, smaller model with multi-scale convolutions
- 32 base channels instead of 64
- Fewer parameters for quick experimentation

## Project Structure

```
residual_model/
├── src/
│   ├── build_residual_manifest.py      # Create dataset manifest
│   ├── datasets_residual.py            # PyTorch dataset classes
│   ├── models_residual.py              # CNN model definitions
│   ├── train_residual.py               # Training script
│   ├── predict_residual.py             # Inference utilities
│   └── evaluate_residual.py            # Evaluation and metrics
├── processed/
│   ├── residual_windows_std/           # Standardized residual data
│   ├── residual_manifest.csv           # Dataset with labels
│   └── ...
├── models/                             # Saved model weights
├── data/                              # Label data
└── requirements.txt                    # Dependencies
```

## Quick Start

### 1. Setup Environment
```bash
cd residual_model
pip install -r requirements.txt
```

### 2. Build Dataset Manifest
```bash
python src/build_residual_manifest.py
```

### 3. Train Model
```bash
python src/train_residual.py
```

### 4. Evaluate Model
```bash
python src/evaluate_residual.py
```

### 5. Make Predictions
```bash
# Single file
python src/predict_residual.py processed/residual_windows_std/residual_10024051.npy

# Detailed analysis
python src/predict_residual.py --analyze processed/residual_windows_std/residual_10024051.npy

# Batch prediction
python src/predict_residual.py --batch processed/residual_windows_std/*.npy
```

## Training Details

### Data Processing
- **Sequence Handling**: Variable lengths handled via padding/truncation to max_length=512
- **Augmentation**: Gaussian noise, temporal shifts (training only)
- **Cross-Validation**: 5-fold GroupKFold by KepID to prevent leakage

### Training Configuration
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing (T_max=30)
- **Loss**: BCE with pos_weight for class imbalance
- **Early Stopping**: Patience=7 epochs based on PR-AUC
- **Gradient Clipping**: Max norm=1.0

### Performance Metrics
- **Primary**: Precision-Recall AUC (handles class imbalance)
- **Secondary**: ROC-AUC, Accuracy, F1-Score
- **Threshold**: Optimized on validation set

## Dataset Statistics

From manifest creation:
- **Total Files**: 243 residual windows
- **Valid Sequences**: 217 (filtered >10 time steps)
- **Label Distribution**: 115 False Positives, 102 Confirmed
- **Sequence Lengths**: 10 to 4000+ time steps

## Expected Performance

Based on similar architectures:
- **PR-AUC**: 0.75-0.85 (excellent for imbalanced data)
- **ROC-AUC**: 0.80-0.90
- **Precision**: ~0.8 at 0.7 recall
- **Training Time**: ~10-20 minutes per fold

## Model Outputs

After training, you'll have:
- `models/residual_cnn_best_fold0.pt` - Best model weights
- `models/residual_val_preds_fold0.npz` - Validation predictions for fusion
- `models/training_curves.png` - Training/validation curves
- `models/evaluation_curves.png` - ROC/PR curves

## Integration with Tabular Model

The residual CNN complements your existing Random Forest tabular model:

1. **Tabular Model**: Orbital/stellar parameters → probability
2. **Residual CNN**: Light curve residuals → probability  
3. **Late Fusion**: Combine probabilities with meta-learner

```python
# Example fusion approach
X_meta = np.column_stack([
    tabular_probs,    # From Random Forest
    residual_probs,   # From this CNN
    # pixel_probs,    # Future: from pixel CNN
])
meta_model = LogisticRegression()
meta_model.fit(X_meta, y_true)
```

## Advanced Usage

### Multi-Fold Training
```bash
# Train all 5 folds for proper cross-validation
for fold in {0..4}; do
    sed -i "s/fold_idx = [0-9]/fold_idx = $fold/" src/train_residual.py
    python src/train_residual.py
done
```

### Hyperparameter Tuning
Edit `src/train_residual.py`:
- `base_channels`: 32, 64, 96 (model capacity)
- `max_length`: 256, 512, 1024 (sequence length)
- `learning_rate`: 1e-4, 3e-4, 1e-3
- `batch_size`: 16, 32, 64

### Custom Data
To use different residual data:
1. Update filename parsing in `build_residual_manifest.py`
2. Adjust `in_features` in model if not 128
3. Modify preprocessing in dataset if needed

## Troubleshooting

### Common Issues

**GPU Memory Error**:
- Reduce `batch_size` in training script
- Use `LightweightResidualCNN` instead
- Reduce `max_length` parameter

**Poor Performance**:
- Check data quality with `--analyze` mode
- Verify label accuracy in manifest
- Try different thresholds for prediction

**File Not Found**:
- Ensure manifest was built successfully
- Check file paths in CSV match actual files
- Verify label data is available

### Debug Mode
```python
# Add to training script for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check single batch
train_loader = DataLoader(train_ds, batch_size=1)
x, y = next(iter(train_loader))
print(f"Input shape: {x.shape}, Label: {y}")
```

## Future Enhancements

- **Attention Mechanisms**: Add self-attention for long sequences
- **Multi-Scale Features**: Combine different temporal resolutions
- **Ensemble Methods**: Average multiple CNN architectures
- **Uncertainty Quantification**: Bayesian or ensemble uncertainty
- **Transfer Learning**: Pre-train on synthetic transit data

## Performance Monitoring

Use tensorboard for detailed monitoring:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/residual_cnn')
# Add to training loop
writer.add_scalar('Loss/Train', train_loss, epoch)
writer.add_scalar('AUC/Val_PR', pr_auc, epoch)
```

## Citation

If you use this model in research, please cite:
- The Kepler mission papers for the data
- Your institution's work on exoplanet detection
- Relevant CNN and residual network papers