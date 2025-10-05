#!/usr/bin/env python3
"""
Batch CNN Data Generation Script
================================

Generates both residual windows (1D CNN) and pixel differences (2D CNN) 
for all available KOI targets in one go.

Usage:
    python scripts/generate_cnn_data_batch.py [--max-targets 1000] [--skip-existing]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features import download_lightcurve, preprocess_lightcurve, create_residual_windows
from pixel_diff import download_target_pixel_file, compute_pixel_differences

def process_target_residual(target_info):
    """Process a single target for residual windows"""
    kepid, period, t0, duration, output_dir, skip_existing = target_info
    
    output_path = output_dir / f"residual_{kepid}.npy"
    if skip_existing and output_path.exists():
        return f"SKIP-{kepid}", "already exists"
    
    try:
        # Download and process lightcurve
        lc = download_lightcurve(str(kepid), mission="Kepler")
        if lc is None:
            return f"FAIL-{kepid}", "no lightcurve data"
        
        lc = preprocess_lightcurve(lc)
        windows = create_residual_windows(lc, period, t0, duration)
        
        if windows.size == 0:
            return f"FAIL-{kepid}", "no windows generated"
        
        # Save residual windows
        np.save(output_path, windows.astype("float32"))
        return f"OK-{kepid}", f"saved {windows.shape[0]} windows"
        
    except Exception as e:
        # Handle TimeDelta formatting issues and other Lightkurve errors gracefully
        error_msg = str(e)
        if "TimeDelta.__format__" in error_msg:
            return f"FAIL-{kepid}", "lightkurve TimeDelta format error"
        else:
            return f"ERROR-{kepid}", error_msg[:50]

def process_target_pixel(target_info):
    """Process a single target for pixel differences"""
    kepid, period, t0, duration, output_dir, skip_existing = target_info
    
    output_path = output_dir / f"pixdiff_{kepid}.npy"
    if skip_existing and output_path.exists():
        return f"SKIP-{kepid}", "already exists"
    
    try:
        # Download and process target pixel file
        tpf = download_target_pixel_file(str(kepid), mission="Kepler")
        if tpf is None:
            return f"FAIL-{kepid}", "no TPF data"
        
        diffs = compute_pixel_differences(tpf, period, t0, duration, phase_bins=32)
        
        if diffs.size == 0:
            return f"FAIL-{kepid}", "no pixel differences generated"
        
        # Save pixel differences
        np.save(output_path, diffs.astype("float32"))
        return f"OK-{kepid}", f"saved {diffs.shape} pixel stack"
        
    except Exception as e:
        return f"ERROR-{kepid}", str(e)

def load_koi_data():
    """Load KOI data, preferring enriched version"""
    base_dir = Path(__file__).parent.parent
    
    # Try enriched first, then fallback to basic
    for csv_path in [
        base_dir / "data/raw/lighkurve_KOI_dataset_enriched.csv",
        base_dir / "data/raw/lighkurve_KOI_dataset.csv"
    ]:
        if csv_path.exists():
            print(f"📊 Loading KOI data from: {csv_path}")
            df = pd.read_csv(csv_path)
            break
    else:
        raise FileNotFoundError("No KOI dataset found!")
    
    # Filter to confirmed and false positives only
    initial_count = len(df)
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    print(f"📋 Filtered {initial_count} → {len(df)} targets (CONFIRMED/FALSE POSITIVE only)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Batch generate CNN data for all KOI targets")
    parser.add_argument("--max-targets", type=int, default=1000, 
                       help="Maximum number of targets to process")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip targets that already have data files")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--residual-only", action="store_true",
                       help="Generate only residual windows (1D CNN data)")
    parser.add_argument("--pixel-only", action="store_true", 
                       help="Generate only pixel differences (2D CNN data)")
    
    args = parser.parse_args()
    
    print("🚀 BATCH CNN DATA GENERATION")
    print("=" * 40)
    
    # Load KOI data
    df = load_koi_data()
    
    # Limit targets if specified
    if args.max_targets and len(df) > args.max_targets:
        df = df.head(args.max_targets)
        print(f"🎯 Limited to first {args.max_targets} targets")
    
    # Create output directories
    base_dir = Path(__file__).parent.parent
    residual_dir = base_dir / "data/processed/residual_windows"
    pixel_dir = base_dir / "data/processed/pixel_diffs"
    
    residual_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare target information
    targets = []
    for _, row in df.iterrows():
        try:
            kepid = int(row["kepid"])
            period = float(row["koi_period"])
            t0 = float(row["koi_time0bk"])
            duration = float(row["koi_duration"])
            targets.append((kepid, period, t0, duration))
        except (ValueError, KeyError) as e:
            print(f"⚠️  Skipping invalid row: {e}")
            continue
    
    print(f"📊 Processing {len(targets)} valid targets")
    print(f"👥 Using {args.workers} parallel workers")
    
    # Generate residual windows (1D CNN data)
    if not args.pixel_only:
        print(f"\n📡 GENERATING RESIDUAL WINDOWS (1D CNN)")
        print("-" * 45)
        
        target_info_residual = [
            (kepid, period, t0, duration, residual_dir, args.skip_existing)
            for kepid, period, t0, duration in targets
        ]
        
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_target_residual, target_info_residual),
                total=len(targets),
                desc="Residual windows"
            ))
        
        # Summarize residual results
        ok_count = sum(1 for status, _ in results if status.startswith("OK"))
        skip_count = sum(1 for status, _ in results if status.startswith("SKIP"))
        fail_count = len(results) - ok_count - skip_count
        
        print(f"✅ Residual windows: {ok_count} success, {skip_count} skipped, {fail_count} failed")
    
    # Generate pixel differences (2D CNN data)
    if not args.residual_only:
        print(f"\n🖼️  GENERATING PIXEL DIFFERENCES (2D CNN)")
        print("-" * 48)
        
        target_info_pixel = [
            (kepid, period, t0, duration, pixel_dir, args.skip_existing)
            for kepid, period, t0, duration in targets
        ]
        
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_target_pixel, target_info_pixel),
                total=len(targets),
                desc="Pixel differences"
            ))
        
        # Summarize pixel results
        ok_count = sum(1 for status, _ in results if status.startswith("OK"))
        skip_count = sum(1 for status, _ in results if status.startswith("SKIP"))
        fail_count = len(results) - ok_count - skip_count
        
        print(f"✅ Pixel differences: {ok_count} success, {skip_count} skipped, {fail_count} failed")
    
    print(f"\n🎊 BATCH GENERATION COMPLETE!")
    print("=" * 35)
    print(f"📁 Residual windows saved to: {residual_dir}")
    print(f"📁 Pixel differences saved to: {pixel_dir}")
    print(f"\nNext steps:")
    print(f"1. Run standardization: python scripts/standardize_cnn_data.py")
    print(f"2. Train multi-modal: python train_multimodal.py")

if __name__ == "__main__":
    main()