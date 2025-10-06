#!/usr/bin/env python3
"""
serve_fusion.py
---------------
FastAPI endpoint for unified exoplanet prediction.

Inputs:
  • kepid (int)  — optional, for reference
  • features (dict) — 13 tabular features
  • residual_window_path (str) — optional .npy file for 1D residual CNN
  • pixel_image_path (str) — optional .npy file for 2D pixel CNN

Output JSON:
{
  "kepid": 10014097,
  "p_rf": 0.91,
  "p_residual": 0.62,
  "p_pixel": 0.41,
  "p_final": 0.84,
  "decision": "CONFIRMED"
}
"""

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np, torch, joblib, yaml, os, sys
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import our utilities
from utils_fusion import (
    PATHS, load_rf_pipeline, load_torch_model,
    predict_rf, predict_residual, predict_pixel,
    predict_stacker, DEVICE
)

# -------------------------------------------------------------------
# Model builders (re-use your project's architectures)
# -------------------------------------------------------------------
# Import model architectures from your project
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + residual)

class LightweightResidualCNN(nn.Module):
    def __init__(self, in_features=128, base_channels=16):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Conv1d(in_features, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 2, 1)
        )
    
    def forward(self, x):
        h = self.feature_proj(x)
        h = self.conv1(h)
        h = self.conv2(h).squeeze(-1)
        return self.head(h).squeeze(-1)

class PixelCNN2D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base*2, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base*4, base*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base*2, 1)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x).squeeze(-1)

app = FastAPI(title="Exoplanet Fusion Predictor", version="1.0.0")

# -------------------------------------------------------------------
# Load all artifacts once at startup
# -------------------------------------------------------------------
rf, imputer, scaler, rf_features = load_rf_pipeline()
stacker = joblib.load(PATHS["stacker_model"])

with open(PATHS["config"], "r") as f:
    cfg = yaml.safe_load(f)

TAU = cfg["fusion"]["threshold"]["tau"]
MEANS = cfg["fusion"]["means"]

# pick one best fold for now (can ensemble later)
residual_net = load_torch_model(PATHS["residual_weights"][0], lambda: LightweightResidualCNN())
pixel_net    = load_torch_model(PATHS["pixel_weights"][0], lambda: PixelCNN2D())

print(f"[INFO] Models loaded on {DEVICE}. Threshold τ={TAU:.3f}")

# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------
@app.post("/predict_exoplanet")
async def predict_exoplanet(
    kepid: Optional[str] = Form(None),
    features: Optional[str] = Form(None),
    residual_window_path: Optional[str] = Form(None),
    pixel_image_path: Optional[str] = Form(None)
):
    """
    Predict using available heads; any missing heads are neutral-filled.
    features: JSON string of 13 numeric feature values.
    """
    import json
    try:
        feature_dict = json.loads(features) if features else {}
    except Exception:
        return JSONResponse({"error": "invalid features JSON"}, status_code=400)

    probs = {"rf": None, "residual": None, "pixel": None}
    masks = {"rf": 0, "residual": 0, "pixel": 0}

    # --- Random Forest
    try:
        probs["rf"] = predict_rf(rf, imputer, scaler, rf_features, feature_dict)
        masks["rf"] = 1
    except Exception as e:
        print("[WARN] RF failed:", e)
        probs["rf"] = MEANS.get("p_rf", 0.5)

    # --- Residual CNN
    if residual_window_path and os.path.exists(residual_window_path):
        try:
            probs["residual"] = predict_residual(residual_net, residual_window_path)
            masks["residual"] = 1
        except Exception as e:
            print("[WARN] Residual CNN failed:", e)
            probs["residual"] = MEANS.get("p_res", 0.5)
    else:
        probs["residual"] = MEANS.get("p_res", 0.5)

    # --- Pixel CNN
    if pixel_image_path and os.path.exists(pixel_image_path):
        try:
            probs["pixel"] = predict_pixel(pixel_net, pixel_image_path)
            masks["pixel"] = 1
        except Exception as e:
            print("[WARN] Pixel CNN failed:", e)
            probs["pixel"] = MEANS.get("p_pix", 0.5)
    else:
        probs["pixel"] = MEANS.get("p_pix", 0.5)

    # --- Stacker fusion
    p_final = predict_stacker(stacker, probs, masks)
    decision = "CONFIRMED" if p_final >= TAU else "FALSE_POSITIVE"

    result = {
        "kepid": kepid,
        **{f"p_{k}": float(v) for k, v in probs.items()},
        "p_final": float(p_final),
        "decision": decision
    }
    return JSONResponse(result)


@app.get("/")
def root():
    return {"message": "Fusion exoplanet API is running 🚀"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "random_forest": rf is not None,
            "residual_cnn": residual_net is not None,
            "pixel_cnn": pixel_net is not None,
            "stacker": stacker is not None
        },
        "device": str(DEVICE),
        "threshold": TAU
    }


# -------------------------------------------------------------------
# Run locally
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fusion.serve_fusion:app", host="0.0.0.0", port=8000, reload=True)