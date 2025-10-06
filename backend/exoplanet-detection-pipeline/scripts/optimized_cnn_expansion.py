#!/usr/bin/env python3
"""
Advanced CNN Coverage Expansion - Optimized for 5%+ coverage
============================================================
Uses smart filtering and parallel processing to maximize success rate
"""

import sys
sys.path.append('../src')

from features import download_lightcurve, preprocess_lightcurve, create_residual_windows
from pixel_diff import download_target_pixel_file, compute_pixel_differences
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_residual_target(args):
    """Process a single target for residual windows"""
    row, output_dir = args
    try:
        kepid = int(row['kepid'])
        filename = f'{output_dir}/residual_{kepid}.npy'
        
        # Skip if already exists
        if os.path.exists(filename):
            return f"Skip {kepid} (exists)"
            
        period = float(row['koi_period'])
        t0 = float(row['koi_time0bk'])
        duration = float(row['koi_duration'])
        
        # Download and process
        lc = download_lightcurve(str(kepid), mission='Kepler')
        if lc is None:
            return f"Skip {kepid} (no LC)"
            
        lc_proc = preprocess_lightcurve(lc)
        if lc_proc is None:
            return f"Skip {kepid} (proc failed)"
            
        windows = create_residual_windows(lc_proc, period, t0, duration)
        if windows.size == 0:
            return f"Skip {kepid} (no windows)"
            
        # Save
        np.save(filename, windows)
        return f"✅ {kepid} -> {windows.shape}"
        
    except Exception as e:
        return f"❌ {kepid}: {str(e)[:50]}"

def process_pixel_target(args):
    """Process a single target for pixel differences"""
    row, output_dir = args
    try:
        kepid = int(row['kepid'])
        filename = f'{output_dir}/pixdiff_{kepid}.npy'
        
        # Skip if already exists
        if os.path.exists(filename):
            return f"Skip {kepid} (exists)"
            
        period = float(row['koi_period'])
        t0 = float(row['koi_time0bk'])
        duration = float(row['koi_duration'])
        
        # Download and process
        tpf = download_target_pixel_file(str(kepid), mission='Kepler')
        if tpf is None:
            return f"Skip {kepid} (no TPF)"
            
        pixel_diffs = compute_pixel_differences(tpf, period, t0, duration)
        if pixel_diffs.size == 0:
            return f"Skip {kepid} (no diffs)"
            
        # Save
        np.save(filename, pixel_diffs)
        return f"✅ {kepid} -> {pixel_diffs.shape}"
        
    except Exception as e:
        return f"❌ {kepid}: {str(e)[:50]}"

def main():
    # Load and filter targets  
    df = pd.read_csv('../data/raw/lighkurve_KOI_dataset_enriched.csv')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Smart filtering for high-success targets
    df_filtered = df[
        (df['koi_period'] > 0.5) &      # Avoid very short periods
        (df['koi_period'] < 100) &      # Avoid very long periods  
        (df['koi_duration'] > 0.5) &    # Avoid very short transits
        (df['koi_duration'] < 24) &     # Avoid very long transits
        (df['koi_depth'] > 0) &         # Must have transit depth
        (pd.notna(df['koi_period'])) &  # Valid period
        (pd.notna(df['koi_time0bk']))   # Valid epoch
    ].copy()
    
    # Prioritize confirmed planets (higher success rate)
    df_confirmed = df_filtered[df_filtered['koi_disposition'] == 'CONFIRMED'].head(300)
    df_fp = df_filtered[df_filtered['koi_disposition'] == 'FALSE POSITIVE'].head(200)
    df_priority = pd.concat([df_confirmed, df_fp]).sample(frac=1, random_state=42)
    
    print(f"🎯 Processing {len(df_priority)} high-priority targets")
    print(f"   Confirmed: {len(df_confirmed)}")
    print(f"   False Pos: {len(df_fp)}")
    print(f"   Target coverage: {len(df_priority)/len(df)*100:.1f}%")
    
    # Ensure directories exist
    os.makedirs('../data/processed/residual_windows', exist_ok=True)
    os.makedirs('../data/processed/pixel_diffs', exist_ok=True)
    
    # Process residual windows
    print(f"\n🔧 Processing residual windows...")
    start_time = time.time()
    residual_args = [(row, '../data/processed/residual_windows') for _, row in df_priority.iterrows()]
    
    residual_success = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_residual_target, args) for args in residual_args]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if i % 50 == 0:
                print(f"Progress: {i}/{len(futures)}")
            if result.startswith('✅'):
                residual_success += 1
    
    residual_time = time.time() - start_time
    print(f"✅ Residual windows: {residual_success}/{len(df_priority)} successful ({residual_time:.1f}s)")
    
    # Process pixel differences  
    print(f"\n🖼️ Processing pixel differences...")
    start_time = time.time()
    pixel_args = [(row, '../data/processed/pixel_diffs') for _, row in df_priority.iterrows()]
    
    pixel_success = 0
    with ThreadPoolExecutor(max_workers=2) as executor:  # Lower for larger files
        futures = [executor.submit(process_pixel_target, args) for args in pixel_args]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if i % 25 == 0:
                print(f"Progress: {i}/{len(futures)}")
            if result.startswith('✅'):
                pixel_success += 1
    
    pixel_time = time.time() - start_time
    print(f"✅ Pixel differences: {pixel_success}/{len(df_priority)} successful ({pixel_time:.1f}s)")
    
    # Calculate final coverage
    total_targets = len(df)
    residual_coverage = (residual_success / total_targets) * 100
    pixel_coverage = (pixel_success / total_targets) * 100
    
    print(f"\n📊 FINAL COVERAGE:")
    print(f"   Residual windows: {residual_success} files ({residual_coverage:.1f}% coverage)")
    print(f"   Pixel differences: {pixel_success} files ({pixel_coverage:.1f}% coverage)")
    print(f"   🎯 Target: >5% coverage for effective fusion")
    
    if residual_coverage > 5 and pixel_coverage > 5:
        print(f"🏆 SUCCESS: Ready for enhanced fusion training!")
    else:
        print(f"⚠️  Run again or increase target count to reach >5% coverage")

if __name__ == "__main__":
    main()