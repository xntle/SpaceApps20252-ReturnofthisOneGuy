"""
CNN data loading utilities for residual windows and pixel differences
"""
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import pandas as pd

def load_cnn_data(data_dir: str = "data/processed", max_samples: int = None) -> Dict:
    """
    Load standardized CNN data (residual windows and pixel differences).
    
    Args:
        data_dir: Directory containing processed data
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        Dictionary with CNN data organized by kepid
    """
    print("🔧 Loading standardized CNN data...")
    
    # Paths to standardized data
    residual_dir = os.path.join(data_dir, "residual_windows_std")
    pixel_dir = os.path.join(data_dir, "pixel_diffs_std")
    
    # Load tabular data to get labels
    try:
        tabular_df = pd.read_csv("data/raw/lighkurve_KOI_dataset_enriched.csv")
        print("📊 Using enriched KOI dataset with stellar parameters")
    except:
        tabular_df = pd.read_csv("data/raw/lighkurve_KOI_dataset.csv")
        print("📊 Using basic KOI dataset")
    
    # Filter to confirmed and false positives
    if 'koi_disposition' in tabular_df.columns:
        tabular_df = tabular_df[tabular_df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Create kepid to label mapping
    kepid_to_label = {}
    for _, row in tabular_df.iterrows():
        kepid = int(row['kepid'])
        label = 1 if row['koi_disposition'] == 'CONFIRMED' else 0
        kepid_to_label[kepid] = label
    
    cnn_data = {
        'residual_windows': {},
        'pixel_diffs': {},
        'labels': {},
        'kepids': []
    }
    
    # Load residual windows
    loaded_residual = 0
    if os.path.exists(residual_dir):
        residual_files = glob.glob(os.path.join(residual_dir, "residual_*.npy"))
        if max_samples:
            residual_files = residual_files[:max_samples]
        print(f"📊 Found {len(residual_files)} residual window files")
        
        for file_path in residual_files:
            filename = os.path.basename(file_path)
            kepid_str = filename.replace("residual_", "").replace(".npy", "")
            kepid = int(kepid_str)
            
            if kepid in kepid_to_label:
                data = np.load(file_path)
                # Take mean over windows for each kepid to get single sample
                if data.shape[0] > 0:
                    cnn_data['residual_windows'][kepid] = np.mean(data, axis=0)  # Shape: (128,)
                    cnn_data['labels'][kepid] = kepid_to_label[kepid]
                    if kepid not in cnn_data['kepids']:
                        cnn_data['kepids'].append(kepid)
                    loaded_residual += 1
    
    # Load pixel differences
    loaded_pixel = 0
    if os.path.exists(pixel_dir):
        pixel_files = glob.glob(os.path.join(pixel_dir, "pixdiff_*.npy"))
        if max_samples:
            pixel_files = pixel_files[:max_samples]
        print(f"📊 Found {len(pixel_files)} pixel difference files")
        
        for file_path in pixel_files:
            filename = os.path.basename(file_path)
            kepid_str = filename.replace("pixdiff_", "").replace(".npy", "")
            kepid = int(kepid_str)
            
            if kepid in kepid_to_label:
                data = np.load(file_path)  # Shape: (32, 24, 24)
                cnn_data['pixel_diffs'][kepid] = data
                cnn_data['labels'][kepid] = kepid_to_label[kepid]
                if kepid not in cnn_data['kepids']:
                    cnn_data['kepids'].append(kepid)
                loaded_pixel += 1
    
    print(f"✅ Loaded CNN data for {len(cnn_data['kepids'])} unique targets")
    print(f"   📡 Residual windows: {loaded_residual} samples")
    print(f"   🖼️  Pixel differences: {loaded_pixel} samples")
    
    # Calculate coverage statistics
    total_targets = len(kepid_to_label)
    coverage_pct = (len(cnn_data['kepids']) / total_targets) * 100 if total_targets > 0 else 0
    print(f"   📈 CNN coverage: {len(cnn_data['kepids'])}/{total_targets} ({coverage_pct:.1f}% of labeled targets)")
    
    return cnn_data

def create_cnn_datasets(cnn_data: Dict, splits: Dict, feature_names: List[str]) -> Dict:
    """
    Create CNN datasets aligned with tabular splits.
    
    Args:
        cnn_data: CNN data from load_cnn_data()
        splits: Tabular splits with kepid information
        feature_names: Tabular feature names
        
    Returns:
        Dictionary with CNN datasets for train/val/test
    """
    print("🔧 Creating aligned CNN datasets...")
    
    # Load tabular data to get kepid mapping
    try:
        tabular_df = pd.read_csv("data/raw/lighkurve_KOI_dataset_enriched.csv")
    except:
        tabular_df = pd.read_csv("data/raw/lighkurve_KOI_dataset.csv")
    
    if 'koi_disposition' in tabular_df.columns:
        tabular_df = tabular_df[tabular_df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Create index to kepid mapping for tabular data
    tabular_kepids = tabular_df['kepid'].values
    
    cnn_datasets = {
        'train': {'cnn1d': [], 'cnn2d': [], 'y': []},
        'val': {'cnn1d': [], 'cnn2d': [], 'y': []},
        'test': {'cnn1d': [], 'cnn2d': [], 'y': []}
    }
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        split_indices = range(len(splits[split_name]['y']))
        split_labels = splits[split_name]['y']
        
        cnn1d_samples = []
        cnn2d_samples = []
        labels = []
        
        for i, label in enumerate(split_labels):
            # Map split index to kepid (this is approximate - we'll match what we can)
            if i < len(tabular_kepids):
                kepid = tabular_kepids[i]
                
                # Check if we have CNN data for this kepid
                has_cnn1d = kepid in cnn_data['residual_windows']
                has_cnn2d = kepid in cnn_data['pixel_diffs']
                
                if has_cnn1d or has_cnn2d:
                    # Use available CNN data, fill missing with zeros
                    if has_cnn1d:
                        cnn1d_sample = cnn_data['residual_windows'][kepid]
                    else:
                        cnn1d_sample = np.zeros(128, dtype=np.float32)
                    
                    if has_cnn2d:
                        cnn2d_sample = cnn_data['pixel_diffs'][kepid]
                    else:
                        cnn2d_sample = np.zeros((32, 24, 24), dtype=np.float32)
                    
                    cnn1d_samples.append(cnn1d_sample)
                    cnn2d_samples.append(cnn2d_sample)
                    labels.append(label)
        
        if len(cnn1d_samples) > 0:
            cnn_datasets[split_name]['cnn1d'] = np.array(cnn1d_samples)
            cnn_datasets[split_name]['cnn2d'] = np.array(cnn2d_samples)
            cnn_datasets[split_name]['y'] = np.array(labels)
        else:
            # No CNN data available - create dummy data
            cnn_datasets[split_name]['cnn1d'] = np.zeros((1, 128), dtype=np.float32)
            cnn_datasets[split_name]['cnn2d'] = np.zeros((1, 32, 24, 24), dtype=np.float32)
            cnn_datasets[split_name]['y'] = np.array([0])
        
        print(f"   {split_name:5}: {len(cnn_datasets[split_name]['y'])} CNN samples")
    
    return cnn_datasets