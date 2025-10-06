#!/usr/bin/env python3
"""
Single-file prediction for pixel difference images
"""
import numpy as np, torch, sys, os
from models_pixel import get_model

@torch.no_grad()
def predict(path, weights_path="models/pixel_cnn_best_fold0.pt", 
           model_type="standard", base=16, dropout=0.3):
    """Predict probability for a single pixel diff file"""
    
    # Load and process image
    try:
        x = np.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    
    # Handle different input formats
    if x.ndim == 3 and x.shape[0] > 1:   # (T,H,W) -> (1,H,W)
        x = np.nanmedian(x, axis=0)[None, ...].astype(np.float32)
    elif x.ndim == 2:                    # (H,W) -> (1,H,W)
        x = x[None, ...].astype(np.float32)
    elif x.ndim == 3 and x.shape[0] == 1:  # Already (1,H,W)
        x = x.astype(np.float32)
    else:
        print(f"Unexpected shape {x.shape} for {path}")
        return None
    
    # Standardize
    m, s = np.nanmedian(x), np.nanstd(x) + 1e-8
    x = (x - m) / s
    
    # Add batch dimension: (1,H,W) -> (1,1,H,W)
    x = torch.tensor(x[None, ...], dtype=torch.float32)
    
    # Load model
    try:
        model = get_model(model_type=model_type, base=base, dropout=dropout)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Predict
    logits = model(x)
    prob = torch.sigmoid(logits).item()
    
    return prob

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python predict_pixel.py <path_to_npy_file> [model_weights]")
        print("Example: python predict_pixel.py processed/pixel_diffs_clean/pixdiff_9967771_std_clean.npy")
        return
    
    path = sys.argv[1]
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "models/pixel_cnn_best_fold0.pt"
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    if not os.path.exists(weights_path):
        print(f"Model weights not found: {weights_path}")
        return
    
    # Predict
    prob = predict(path, weights_path)
    
    if prob is not None:
        pred = "CONFIRMED" if prob > 0.5 else "FALSE POSITIVE"
        print(f"{os.path.basename(path)} -> p_pixel={prob:.4f} ({pred})")
    else:
        print(f"Prediction failed for {path}")

if __name__ == "__main__":
    main()