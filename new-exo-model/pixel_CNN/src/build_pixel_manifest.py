#!/usr/bin/env python3
"""
Build pixel manifest linking pixel diff files to labels
"""
import re, glob, os, pandas as pd

PIX_DIR = "processed/pixel_diffs_clean"               # outputs from step 1
RES_MAN = "processed/residual_manifest.csv"           # has kepid + label

def parse_kepid(name):
    """Extract KepID from filename like pixdiff_9967771_std_clean.npy"""
    m = re.search(r"(\d+)", name)  # files named like pixdiff_9967771_std.npy OR kepler_9967771__...
    return int(m.group(1)) if m else None

def main():
    """Build manifest CSV linking pixel files to labels"""
    
    # Get all pixel diff files and extract KepIDs
    paths = glob.glob(f"{PIX_DIR}/*.npy")
    print(f"Found {len(paths)} pixel diff files")
    
    pix = pd.DataFrame([
        {"pix_path": p, "kepid": parse_kepid(os.path.basename(p))} 
        for p in paths
    ])
    pix = pix.dropna(subset=["kepid"]).astype({"kepid": int})
    print(f"Extracted {len(pix)} valid KepIDs")

    # Load labels from residual manifest
    try:
        labels = pd.read_csv(RES_MAN)[["kepid", "label"]].drop_duplicates("kepid")
        print(f"Loaded {len(labels)} labels from {RES_MAN}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return
    
    # Merge pixel files with labels
    df = pix.merge(labels, on="kepid", how="inner")
    print(f"Matched {len(df)} pixel files with labels")
    
    # Show label distribution
    label_counts = df["label"].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Save manifest
    out = "processed/pixel_manifest.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out} with {len(df)} rows")
    
    # Show first few examples
    print(f"\nFirst 5 entries:")
    print(df.head())

if __name__ == "__main__":
    main()