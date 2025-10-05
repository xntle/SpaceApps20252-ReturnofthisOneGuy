#!/usr/bin/env python3
"""
Rapid CNN Coverage Expansion Script
==================================

Quickly expand CNN coverage to demonstrate fusion performance scaling.
Focuses on targets most likely to succeed for rapid results.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features import download_lightcurve, preprocess_lightcurve, create_residual_windows
from pixel_diff import download_target_pixel_file, compute_pixel_differences

def rapid_cnn_expansion(max_targets=500, max_time_minutes=60):
    """
    Rapidly expand CNN coverage focusing on highest success probability targets.
    
    Args:
        max_targets: Maximum number of new targets to process
        max_time_minutes: Maximum time to spend (stops early if time limit reached)
    """
    
    print("🚀 RAPID CNN COVERAGE EXPANSION")
    print("=" * 40)
    
    start_time = time.time()
    
    # Load KOI data
    base_dir = Path(__file__).parent.parent
    
    for csv_path in [
        base_dir / "data/raw/lighkurve_KOI_dataset_enriched.csv",
        base_dir / "data/raw/lighkurve_KOI_dataset.csv"
    ]:
        if csv_path.exists():
            print(f"📊 Loading KOI data from: {csv_path}")
            df = pd.read_csv(csv_path)
            break
    
    # Filter and prioritize targets
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Prioritize confirmed planets and targets with good parameters
    confirmed_df = df[df['koi_disposition'] == 'CONFIRMED'].copy()
    false_pos_df = df[df['koi_disposition'] == 'FALSE POSITIVE'].copy()
    
    # Sort by period (shorter periods more likely to have good data)
    confirmed_df = confirmed_df.sort_values('koi_period')
    false_pos_df = false_pos_df.sort_values('koi_period')
    
    # Combine with preference for confirmed planets
    targets_df = pd.concat([
        confirmed_df.head(max_targets // 2),
        false_pos_df.head(max_targets // 2)
    ]).reset_index(drop=True)
    
    print(f"🎯 Targeting {len(targets_df)} high-priority KOIs")
    
    # Create output directories
    residual_dir = base_dir / "data/processed/residual_windows"
    pixel_dir = base_dir / "data/processed/pixel_diffs"
    residual_dir.mkdir(parents=True, exist_ok=True)
    pixel_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing files
    existing_residual = set()
    existing_pixel = set()
    
    if residual_dir.exists():
        existing_residual = {f.stem.replace('residual_', '') for f in residual_dir.glob('residual_*.npy')}
    
    if pixel_dir.exists():
        existing_pixel = {f.stem.replace('pixdiff_', '') for f in pixel_dir.glob('pixdiff_*.npy')}
    
    print(f"📁 Existing data: {len(existing_residual)} residual, {len(existing_pixel)} pixel")
    
    # Process targets
    new_residual = 0
    new_pixel = 0
    failed_targets = 0
    
    for idx, row in tqdm(targets_df.iterrows(), total=len(targets_df), desc="Generating CNN data"):
        # Check time limit
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > max_time_minutes:
            print(f"⏰ Time limit reached ({max_time_minutes} minutes)")
            break
        
        try:
            kepid = int(row["kepid"])
            kepid_str = str(kepid)
            period = float(row["koi_period"])
            t0 = float(row["koi_time0bk"])
            duration = float(row["koi_duration"])
            
            # Skip if we already have both types of data for this target
            has_residual = kepid_str in existing_residual
            has_pixel = kepid_str in existing_pixel
            
            if has_residual and has_pixel:
                continue
            
            # Try residual windows first (usually faster)
            if not has_residual:
                try:
                    lc = download_lightcurve(kepid_str, mission="Kepler")
                    if lc is not None:
                        lc = preprocess_lightcurve(lc)
                        windows = create_residual_windows(lc, period, t0, duration)
                        
                        if windows.size > 0:
                            np.save(residual_dir / f"residual_{kepid}.npy", windows.astype("float32"))
                            existing_residual.add(kepid_str)
                            new_residual += 1
                except Exception:
                    pass  # Continue to pixel differences
            
            # Try pixel differences
            if not has_pixel:
                try:
                    tpf = download_target_pixel_file(kepid_str, mission="Kepler")
                    if tpf is not None:
                        diffs = compute_pixel_differences(tpf, period, t0, duration, phase_bins=32)
                        
                        if diffs.size > 0:
                            np.save(pixel_dir / f"pixdiff_{kepid}.npy", diffs.astype("float32"))
                            existing_pixel.add(kepid_str)
                            new_pixel += 1
                except Exception:
                    pass
            
            # Count as failed if we couldn't get either type of data
            if kepid_str not in existing_residual and kepid_str not in existing_pixel:
                failed_targets += 1
                
        except Exception as e:
            failed_targets += 1
            continue
        
        # Progress update every 20 targets
        if (idx + 1) % 20 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Progress: {idx+1}/{len(targets_df)}, New: +{new_residual} residual, +{new_pixel} pixel, "
                  f"Failed: {failed_targets}, Time: {elapsed:.1f}m")
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    total_residual = len(existing_residual)
    total_pixel = len(existing_pixel)
    
    print(f"\n✅ RAPID EXPANSION COMPLETE!")
    print("=" * 35)
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    print(f"📊 Final counts:")
    print(f"   📡 Residual windows: {total_residual} (+{new_residual} new)")
    print(f"   🖼️  Pixel differences: {total_pixel} (+{new_pixel} new)")
    print(f"   ❌ Failed targets: {failed_targets}")
    
    # Calculate new coverage
    total_targets = len(df)
    unique_cnn_targets = len(existing_residual.union(existing_pixel))
    coverage_pct = (unique_cnn_targets / total_targets) * 100
    
    print(f"📈 Total CNN coverage: {unique_cnn_targets}/{total_targets} ({coverage_pct:.1f}%)")
    
    if coverage_pct > 5:
        print(f"\n🎊 EXCELLENT! CNN coverage >5% should enable strong fusion performance!")
        print(f"Next steps:")
        print(f"1. python scripts/standardize_cnn_data.py")
        print(f"2. python train_multimodal_enhanced.py")
    else:
        print(f"\n💡 Recommendation: Continue expanding CNN coverage for better fusion results")
    
    return {
        'new_residual': new_residual,
        'new_pixel': new_pixel,
        'total_residual': total_residual,
        'total_pixel': total_pixel,
        'coverage_pct': coverage_pct,
        'time_minutes': total_time
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rapidly expand CNN coverage")
    parser.add_argument("--max-targets", type=int, default=200, 
                       help="Maximum number of targets to process")
    parser.add_argument("--max-time", type=int, default=270,
                       help="Maximum time in minutes")
    
    args = parser.parse_args()
    
    rapid_cnn_expansion(max_targets=args.max_targets, max_time_minutes=args.max_time)