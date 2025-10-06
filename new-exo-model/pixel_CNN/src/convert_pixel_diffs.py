#!/usr/bin/env python3
"""
Convert pixel diff stacks (T,H,W) to standardized single images (1,H,W)
"""
from pathlib import Path
import numpy as np, re

IN_DIR  = Path("processed/pixel_diffs_std")      # where your stacks live
OUT_DIR = Path("processed/pixel_diffs_clean")    # standardized single-image diffs
OUT_DIR.mkdir(parents=True, exist_ok=True)

def collapse_and_standardize(p):
    """Convert (T,H,W) stack to standardized (1,H,W) image"""
    arr = np.load(p)           # expect (T,H,W) or (H,W)
    
    if arr.ndim == 3 and arr.shape[0] > 1:
        # Take median across time axis to get robust single image
        img = np.nanmedian(arr, axis=0)
    elif arr.ndim == 3:        # (1,H,W)
        img = arr[0]
    elif arr.ndim == 2:
        img = arr
    else:
        raise ValueError(f"Unexpected shape {arr.shape} for {p}")
    
    # Standardize: zero mean, unit std
    m, s = np.nanmedian(img), np.nanstd(img) + 1e-8
    x = ((img - m)/s).astype(np.float32)[None, ...]  # (1,H,W)
    return x

def main():
    """Process all .npy files in input directory"""
    processed = 0
    for p in IN_DIR.glob("*.npy"):
        try:
            x = collapse_and_standardize(p)
            out = OUT_DIR / (p.stem + "_clean.npy")
            np.save(out, x)
            print(f"✓ {p.name} -> {x.shape} -> {out.name}")
            processed += 1
        except Exception as e:
            print(f"✗ Failed {p.name}: {e}")
    
    print(f"\nProcessed {processed} files successfully!")

if __name__ == "__main__":
    main()