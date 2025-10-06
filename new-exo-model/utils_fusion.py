"""
fusion/utils_fusion.py
----------------------
Common utilities for loading models, preprocessing inputs, and aligning predictions
across Random Forest (tabular), Residual CNN (1D), and Pixel CNN (2D).
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# =========================
# 🔹 PATHS CONFIG
# =========================
import os
BASE = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "rf_model": f"{BASE}/AI_Model_Forest/trained_model/rf_combined_model.joblib",
    "rf_scaler": f"{BASE}/AI_Model_Forest/trained_model/scaler_combined.joblib",
    "rf_imputer": f"{BASE}/AI_Model_Forest/trained_model/imputer_medians_combined.joblib",
    "rf_features": f"{BASE}/AI_Model_Forest/trained_model/feature_columns_combined.txt",

    "residual_weights": [f"{BASE}/residual_model/models/residual_cnn_best_fold{i}.pt" for i in range(5)],
    "pixel_weights": [f"{BASE}/pixel_CNN/models/pixel_cnn_best_fold{i}.pt" for i in range(5)],
    "stacker_model": f"{BASE}/models/stacker_xgb.pkl",
    "config": f"{BASE}/config.yaml"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 🔹 LOADING UTILITIES
# =========================
def load_rf_pipeline():
    """Load Random Forest + preprocessors."""
    rf = joblib.load(PATHS["rf_model"])
    imputer = joblib.load(PATHS["rf_imputer"])
    scaler = joblib.load(PATHS["rf_scaler"])
    with open(PATHS["rf_features"]) as f:
        features = [x.strip() for x in f.readlines()]
    return rf, imputer, scaler, features


def load_torch_model(path: str, model_builder) -> torch.nn.Module:
    """Load a PyTorch model from weights and send to proper device."""
    model = model_builder()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


# =========================
# 🔹 PREDICTION HELPERS
# =========================
def predict_rf(rf, imputer, scaler, features, feature_dict: Dict[str, float]) -> float:
    """Return probability from RF given raw feature dict."""
    # Convert any numpy arrays to scalars and handle NaN properly
    feature_values = []
    for f in features:
        val = feature_dict.get(f, np.nan)
        if hasattr(val, 'item'):  # Convert numpy scalars
            val = val.item()
        elif isinstance(val, (list, tuple, np.ndarray)):
            val = float(val[0]) if len(val) > 0 else np.nan
        feature_values.append(float(val) if not np.isnan(val) else np.nan)
    
    x = np.array([feature_values])
    x = imputer.transform(x)
    x = scaler.transform(x)
    prob = rf.predict_proba(x)[0, 1]
    return float(prob)


def predict_residual(model: torch.nn.Module, npy_path: str) -> float:
    """Predict from a residual CNN window (.npy)."""
    arr = np.load(npy_path)  # Shape should be [2, 128]
    if arr.ndim == 3 and arr.shape[0] > 1:
        arr = np.nanmedian(arr, axis=0)
    
    # Ensure correct shape [128, 2] and transpose to [2, 128] for model input
    if arr.shape == (2, 128):
        pass  # Already correct
    elif arr.shape == (128, 2):
        arr = arr.T  # [2, 128]
    else:
        raise ValueError(f"Unexpected residual shape: {arr.shape}, expected (2,128) or (128,2)")
    
    # Model expects [B, features=128, seq_len] format
    # arr is [2, 128], we need [128, seq_len] where seq_len is small
    # Transpose to [128, 2] for proper input format
    arr = arr.T.astype(np.float32)  # [128, 2]
    arr = arr[None, ...]  # [1, 128, 2] - batch dimension
    
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(arr, device=DEVICE))).cpu().numpy()
        if p.ndim > 0:
            p = p[0]  # Extract scalar
    return float(p)


def predict_pixel(model: torch.nn.Module, npy_path: str) -> float:
    """Predict from a pixel CNN image (.npy)."""
    arr = np.load(npy_path)  # Should be [1, 24, 24] or [24, 24]
    
    # Handle different input formats
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            # Already correct format [1, H, W]
            pass
        else:
            # Multiple images, take median
            arr = np.nanmedian(arr, axis=0)
            arr = arr[None, ...]  # Add channel dimension
    elif arr.ndim == 2:
        # Single image [H, W] -> [1, H, W]
        arr = arr[None, ...]
    
    # Standardize the image
    m, s = np.nanmedian(arr), np.nanstd(arr) + 1e-8
    arr = ((arr - m) / s).astype(np.float32)
    
    # Add batch dimension: [1, 1, H, W]
    arr = arr[None, ...]
    
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(arr, device=DEVICE))).cpu().numpy()
        if p.ndim > 0:
            p = p[0]  # Extract scalar
    return float(p)


# =========================
# 🔹 STACKER HELPER
# =========================
def predict_stacker(meta_model, probs: Dict[str, float], masks: Dict[str, int]) -> float:
    """Predict fused probability from available heads."""
    feats = np.array([[probs["rf"], probs["residual"], probs["pixel"],
                       masks["rf"], masks["residual"], masks["pixel"]]])
    return float(meta_model.predict_proba(feats)[0, 1])


# =========================
# 🔹 ALIGNMENT / OOF LOADING
# =========================
def load_oof_npzs(pattern: str, name: str) -> pd.DataFrame:
    """Load OOF validation predictions for stacker training."""
    import glob
    dfs = []
    for p in glob.glob(pattern):
        d = np.load(p)
        dfs.append(pd.DataFrame({
            "id": d["ids"], name: d["y_probs"], "y": d["y_true"]
        }))
    return pd.concat(dfs, ignore_index=True)