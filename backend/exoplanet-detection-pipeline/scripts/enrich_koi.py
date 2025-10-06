#!/usr/bin/env python3
"""
Enrich KOI dataset with stellar parameters from NASA Exoplanet Archive.
Run this once to enhance the dataset with depth + stellar properties.
"""
import pandas as pd
import requests
from urllib.parse import quote
from io import StringIO
import os

def enrich_koi_dataset():
    """Enrich the minimal KOI dataset with full stellar parameters."""
    
    in_path = "data/raw/lighkurve_KOI_dataset.csv"
    out_path = "data/raw/lighkurve_KOI_dataset_enriched.csv"
    
    if not os.path.exists(in_path):
        print(f"❌ Input file not found: {in_path}")
        return
    
    print(f"🔄 Loading {in_path}...")
    df = pd.read_csv(in_path)
    print(f"   Original: {len(df)} rows, {df.shape[1]} columns")
    
    # Query NASA Exoplanet Archive for full KOI table with stellar params
    print("🌌 Querying NASA Exoplanet Archive for stellar parameters...")
    
    query = """
    SELECT kepid, kepoi_name,
           koi_depth, koi_depth_err1, koi_depth_err2,
           koi_steff, koi_steff_err1, koi_steff_err2,
           koi_slogg, koi_slogg_err1, koi_slogg_err2,
           koi_smet,  koi_smet_err1,  koi_smet_err2,
           koi_srad,  koi_srad_err1,  koi_srad_err2,
           koi_smass, koi_smass_err1, koi_smass_err2,
           koi_impact, koi_impact_err1, koi_impact_err2
    FROM Q1_Q17_DR25_KOI
    """
    
    try:
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={quote(query)}&format=csv"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        full_koi = pd.read_csv(StringIO(response.text))
        print(f"   Archive: {len(full_koi)} rows, {full_koi.shape[1]} columns")
        
        # Merge on kepid (primary key)
        print("🔗 Merging datasets on kepid...")
        merged = df.merge(full_koi, on='kepid', how='left', suffixes=('', '_archive'))
        
        # Check merge success
        new_cols = sorted(set(merged.columns) - set(df.columns))
        print(f"   Added columns: {new_cols}")
        
        # Report merge statistics
        for col in ['koi_depth', 'koi_steff', 'koi_slogg', 'koi_smet', 'koi_srad', 'koi_smass']:
            if col in merged.columns:
                non_null = merged[col].notna().sum()
                print(f"   {col}: {non_null}/{len(merged)} ({100*non_null/len(merged):.1f}%) non-null")
        
        # Save enriched dataset
        merged.to_csv(out_path, index=False)
        print(f"\\n✅ Enriched dataset saved to: {out_path}")
        print(f"   Final: {len(merged)} rows, {merged.shape[1]} columns")
        print(f"   Added {len(new_cols)} new columns with stellar parameters")
        
        return out_path
        
    except Exception as e:
        print(f"❌ Error querying NASA Exoplanet Archive: {e}")
        print("   Continuing with original dataset...")
        return in_path

if __name__ == "__main__":
    enrich_koi_dataset()