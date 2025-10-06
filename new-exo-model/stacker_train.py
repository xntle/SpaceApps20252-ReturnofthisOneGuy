#!/usr/bin/env python3
"""
Train the late-fusion stacker on out-of-fold (OOF) predictions from:
- Random Forest (tabular) [optional if you don't have OOF]
- Residual CNN (1D)
- Pixel CNN (2D)

Outputs:
- fusion/models/stacker_xgb.pkl
- fusion/config.yaml  (adds/updates: fusion.threshold.tau, feature order)
- A quick metric printout (PR-AUC, ROC-AUC, F1 @ tau)
"""

import os, glob, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, f1_score
)
import joblib
import yaml

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).parent.absolute()
RES_DIR = BASE / "residual_model" / "models"
PIX_DIR = BASE / "pixel_CNN" / "models"
RF_DIR  = BASE / "AI_Model_Forest" / "models"   # change if you do export RF OOF later

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = BASE / "config.yaml"

# ----------------------------
# Utils
# ----------------------------
def load_oof_with_manifest(npz_pattern: str, manifest_path: str, colname: str) -> pd.DataFrame:
    """Load OOF predictions and reconstruct IDs from manifest using GroupKFold logic."""
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    
    # Load manifest to get IDs and labels
    manifest = pd.read_csv(manifest_path)
    ids = manifest["kepid"].values
    labels = manifest["label"].values
    
    # Use same GroupKFold split logic as training
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(manifest, labels, groups=ids))
    
    rows = []
    for fold_idx, npz_path in enumerate(sorted(glob.glob(npz_pattern))):
        if fold_idx >= len(splits):
            continue
            
        d = np.load(npz_path, allow_pickle=True)
        keys = set(d.files)
        
        # Handle different key variants
        y_true = d["y_true"] if "y_true" in keys else d["y"]
        y_prob = d["y_prob"] if "y_prob" in keys else (d["y_probs"] if "y_probs" in keys else d["p"])
        
        # Get validation indices for this fold
        _, val_idx = splits[fold_idx]
        val_ids = ids[val_idx]
        
        # Verify lengths match
        if len(val_ids) != len(y_prob):
            print(f"Warning: fold {fold_idx} length mismatch: manifest={len(val_ids)}, npz={len(y_prob)}")
            continue
            
        df = pd.DataFrame({
            "id": val_ids.astype(np.int64),
            colname: y_prob.astype(np.float32),
            "y": y_true.astype(np.int64)
        })
        rows.append(df)
    
    if not rows:
        return pd.DataFrame(columns=["id", colname, "y"])
    return pd.concat(rows, ignore_index=True)

def load_oof_simple(pattern: str, colname: str) -> pd.DataFrame:
    """Load OOF without IDs (for RF if no manifest available)."""
    rows = []
    for p in glob.glob(pattern):
        d = np.load(p, allow_pickle=True)
        keys = set(d.files)
        y_true = d["y_true"] if "y_true" in keys else d["y"]
        y_prob = d["y_prob"] if "y_prob" in keys else d["y_probs"]
        
        # Generate sequential IDs if no IDs available
        fake_ids = np.arange(len(y_prob))
        df = pd.DataFrame({
            "id": fake_ids.astype(np.int64),
            colname: y_prob.astype(np.float32),
            "y": y_true.astype(np.int64)
        })
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["id", colname, "y"])
    return pd.concat(rows, ignore_index=True)

def pick_threshold_f1(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Scan thresholds to maximize F1; return (best_tau, best_f1)."""
    prec, rec, thr = precision_recall_curve(y, p)
    # precision_recall_curve returns thresholds for all but first point
    # Evaluate F1 at those thresholds
    best_tau, best_f1 = 0.5, -1.0
    # Derive predicted labels by threshold
    for t in np.linspace(0.01, 0.99, 99):
        yhat = (p >= t).astype(int)
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    return float(best_tau), float(best_f1)

# ----------------------------
# Load OOF predictions
# ----------------------------
# Residual CNN folds (with manifest for IDs)
res_manifest = BASE / "pixel_CNN" / "processed" / "residual_manifest.csv"
res_df = load_oof_with_manifest(str(RES_DIR / "residual_val_preds_fold*.npz"), str(res_manifest), "p_res")

# Pixel CNN folds (with manifest for IDs)  
pix_manifest = BASE / "pixel_CNN" / "processed" / "pixel_manifest.csv"
pix_df = load_oof_with_manifest(str(PIX_DIR / "pixel_val_preds_fold*.npz"), str(pix_manifest), "p_pix")

# Random Forest OOF (optional; you may not have these yet)
rf_df = load_oof_simple(str(RF_DIR / "rf_val_preds_fold*.npz"), "p_rf")

# Merge on id & y (outer to allow missing heads)
dfs = [df for df in [rf_df, res_df, pix_df] if not df.empty]
if not dfs:
    raise SystemExit("No OOF predictions found. Train at least one head to produce *_val_preds_fold*.npz")

from functools import reduce
oof = reduce(lambda a,b: pd.merge(a, b, on=["id","y"], how="outer"), dfs).sort_values("id").reset_index(drop=True)

# Neutral fill for missing heads and availability masks
cols = ["p_rf","p_res","p_pix"]
for c in cols:
    if c not in oof.columns:
        oof[c] = np.nan
means = {c: float(oof[c].mean(skipna=True)) if c in oof.columns else 0.5 for c in cols}
for c in cols:
    oof[c] = oof[c].fillna(means[c])

oof["m_rf"]  = (oof["p_rf"].notna()).astype(int)
oof["m_res"] = (oof["p_res"].notna()).astype(int)
oof["m_pix"] = (oof["p_pix"].notna()).astype(int)

X = oof[["p_rf","p_res","p_pix","m_rf","m_res","m_pix"]].values.astype(np.float32)
y = oof["y"].values.astype(int)

# ----------------------------
# Train XGBoost stacker
# ----------------------------
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise SystemExit("xgboost not installed: pip install xgboost") from e

meta = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42
)
meta.fit(X, y)

p_meta = meta.predict_proba(X)[:,1]
pr = average_precision_score(y, p_meta)
roc = roc_auc_score(y, p_meta)
tau, f1 = pick_threshold_f1(y, p_meta)

print("\n=== FUSION STACKER (OOF) ===")
print(f"PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f}")
print(f"Best F1 threshold: {tau:.3f} (F1={f1:.3f})")

# ----------------------------
# Persist artifacts
# ----------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(meta, MODEL_DIR / "stacker_xgb.pkl")
print(f"Saved stacker to {MODEL_DIR / 'stacker_xgb.pkl'}")

# Update or create config.yaml
cfg = {}
if CFG_PATH.exists():
    with open(CFG_PATH, "r") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

cfg.setdefault("fusion", {})
cfg["fusion"]["features"]  = ["p_rf","p_res","p_pix","m_rf","m_res","m_pix"]
cfg["fusion"]["threshold"] = {"tau": float(tau)}
cfg["fusion"]["means"]     = means  # used as neutral fills at inference

with open(CFG_PATH, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Updated config at {CFG_PATH}")