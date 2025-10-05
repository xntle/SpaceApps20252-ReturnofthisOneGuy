#!/usr/bin/env python3
"""
Simple CNN Coverage Booster - Focus on reliability over speed
Uses conservative approach to maximize success rate
"""

import sys
sys.path.append('../src')

from features import download_lightcurve, preprocess_lightcurve, create_residual_windows
from pixel_diff import download_target_pixel_file, compute_pixel_differences
import pandas as pd
import numpy as np
import os
import time

def process_targets_sequential(df_targets, data_type='residual'):
    """Process targets one by one for maximum reliability"""
    successes = 0
    
    for i, (_, row) in enumerate(df_targets.iterrows()):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(df_targets)} ({successes} successful)")
            
        try:
            kepid = int(row['kepid'])
            
            if data_type == 'residual':
                filename = f'../data/processed/residual_windows/residual_{kepid}.npy'
                if os.path.exists(filename):
                    continue
                    
                period = float(row['koi_period'])
                t0 = float(row['koi_time0bk'])
                duration = float(row['koi_duration'])
                
                lc = download_lightcurve(str(kepid), mission='Kepler')
                if lc is None:
                    continue
                    
                lc_proc = preprocess_lightcurve(lc)
                if lc_proc is None:
                    continue
                    
                windows = create_residual_windows(lc_proc, period, t0, duration)
                if windows.size == 0:
                    continue
                    
                np.save(filename, windows)
                successes += 1
                print(f"✅ {kepid} -> {windows.shape}")
                
            elif data_type == 'pixel':
                filename = f'../data/processed/pixel_diffs/pixdiff_{kepid}.npy'
                if os.path.exists(filename):
                    continue
                    
                period = float(row['koi_period'])
                t0 = float(row['koi_time0bk'])
                duration = float(row['koi_duration'])
                
                tpf = download_target_pixel_file(str(kepid), mission='Kepler')
                if tpf is None:
                    continue
                    
                pixel_diffs = compute_pixel_differences(tpf, period, t0, duration)
                if pixel_diffs.size == 0:
                    continue
                    
                np.save(filename, pixel_diffs)
                successes += 1
                print(f"✅ {kepid} -> {pixel_diffs.shape}")
                
            # Small delay to avoid overwhelming MAST
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ {kepid}: {str(e)[:50]}")
            time.sleep(1)  # Longer delay after errors
            continue
            
    return successes

def main():
    # Load data
    df = pd.read_csv('../data/raw/lighkurve_KOI_dataset_enriched.csv')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Very conservative filtering for maximum success
    df_high_success = df[
        (df['koi_disposition'] == 'CONFIRMED') &      # Confirmed planets only
        (df['koi_period'] >= 1) & (df['koi_period'] <= 20) &  # Sweet spot periods
        (df['koi_duration'] >= 1) & (df['koi_duration'] <= 8) &  # Reasonable durations
        (df['koi_depth'] >= 50) &                     # Deep enough transits
        (pd.notna(df['koi_period'])) & 
        (pd.notna(df['koi_time0bk'])) &
        (pd.notna(df['koi_duration']))
    ].head(100)  # Start with 100 high-confidence targets
    
    print(f"🎯 Processing {len(df_high_success)} ultra-high-confidence targets")
    print(f"   Target success rate: >80%")
    
    # Ensure directories exist
    os.makedirs('../data/processed/residual_windows', exist_ok=True)
    os.makedirs('../data/processed/pixel_diffs', exist_ok=True)
    
    # Process residual windows first
    print(f"\n🔧 Processing residual windows...")
    start_time = time.time()
    residual_success = process_targets_sequential(df_high_success, 'residual')
    residual_time = time.time() - start_time
    
    print(f"\n✅ Residual windows: {residual_success}/{len(df_high_success)} successful ({residual_time:.1f}s)")
    
    # Process pixel differences
    print(f"\n🖼️ Processing pixel differences...")
    start_time = time.time()
    pixel_success = process_targets_sequential(df_high_success, 'pixel')
    pixel_time = time.time() - start_time
    
    print(f"\n✅ Pixel differences: {pixel_success}/{len(df_high_success)} successful ({pixel_time:.1f}s)")
    
    # Calculate coverage
    total_targets = len(df)
    print(f"\n📊 COVERAGE UPDATE:")
    print(f"   Added residual windows: +{residual_success}")
    print(f"   Added pixel differences: +{pixel_success}")
    print(f"   New coverage estimate: ~{(residual_success + pixel_success)/(total_targets*2)*100:.1f}%")

if __name__ == "__main__":
    main()