import re, glob, os
import pandas as pd
import numpy as np

RES_DIR = "processed/residual_windows_std"  
KOI_CSV = "data/kepler_koi_cumulative.csv"

def parse_id_from_path(p: str):
    # Expect: processed/residual_windows_std/residual_<kepid>.npy
    # Extract kepid as int
    m = re.search(r"residual_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else None

def check_data_format():
    """Check and report the actual data format"""
    paths = sorted(glob.glob(f"{RES_DIR}/*.npy"))[:5]
    print("=== Data Format Check ===")
    for p in paths:
        arr = np.load(p)
        kepid = parse_id_from_path(p)
        print(f"File: {os.path.basename(p)}")
        print(f"  KepID: {kepid}")
        print(f"  Shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        print(f"  Min/Max: {arr.min():.6f} / {arr.max():.6f}")
        print(f"  Sequence length: {arr.shape[0]}, Features: {arr.shape[1]}")
        print()

def main():
    # First check the data format
    check_data_format()
    
    paths = sorted(glob.glob(f"{RES_DIR}/*.npy"))
    print(f"Found {len(paths)} residual window files")
    
    rows = []
    for p in paths:
        kepid = parse_id_from_path(p)
        if kepid is not None:
            # Load and check if file has reasonable data
            try:
                arr = np.load(p)
                if arr.shape[0] > 10:  # Skip very short sequences
                    rows.append({"kepid": kepid, "path": p})
                else:
                    print(f"Skipping {p} - too short ({arr.shape[0]} samples)")
            except Exception as e:
                print(f"Error loading {p}: {e}")
    
    man = pd.DataFrame(rows)
    print(f"Valid residual files: {len(man)}")

    # Load KOI labels
    print("Loading KOI labels...")
    koi = pd.read_csv(KOI_CSV, comment='#')
    print(f"KOI data columns: {list(koi.columns)}")
    
    # Check for different possible disposition column names
    disp_col = None
    for col in ["koi_disposition", "disposition", "koi_pdisposition"]:
        if col in koi.columns:
            disp_col = col
            break
    
    if disp_col is None:
        print("Available columns:", list(koi.columns))
        raise ValueError("Could not find disposition column in KOI data")
    
    print(f"Using disposition column: {disp_col}")
    print(f"Unique dispositions: {koi[disp_col].value_counts()}")
    
    # Get unique KepIDs with their dispositions
    koi_clean = koi[["kepid", disp_col]].drop_duplicates("kepid")
    
    # Merge with our residual files
    df = man.merge(koi_clean, on="kepid", how="left")
    
    print(f"Matched files: {len(df.dropna(subset=[disp_col]))}")
    print(f"Dispositions in matched data:")
    print(df[disp_col].value_counts())
    
    # Create binary labels
    label_map = {
        "CONFIRMED": 1, 
        "CANDIDATE": 1,  # Treat candidates as positive
        "FALSE POSITIVE": 0
    }
    
    df["label"] = df[disp_col].map(label_map)
    df_labeled = df.dropna(subset=["label"]).reset_index(drop=True)
    
    print(f"\nFinal dataset:")
    print(f"Total samples: {len(df_labeled)}")
    print(f"Label distribution:")
    print(df_labeled["label"].value_counts())
    
    # Save the manifest
    output_path = "processed/residual_manifest.csv"
    df_labeled.to_csv(output_path, index=False)
    print(f"\nWrote {output_path} with {len(df_labeled)} rows")
    
    return df_labeled

if __name__ == "__main__":
    main()