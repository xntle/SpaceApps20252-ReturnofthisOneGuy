#!/usr/bin/env python3
"""
PyTorch dataset for pixel difference images with cross-validation
"""
import numpy as np, torch, pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

class PixelDiffDataset(Dataset):
    """Dataset for pixel difference images with GroupKFold cross-validation"""
    
    def __init__(self, csv_path, split, folds=5, fold_idx=0, group_col="kepid",
                 augment=True, std=False):
        """
        Args:
            csv_path: Path to pixel manifest CSV
            split: 'train' or 'val'
            folds: Number of CV folds
            fold_idx: Which fold to use for validation
            group_col: Column to group by (prevents data leakage)
            augment: Apply data augmentation (train only)
            std: Apply standardization (if not done in preprocessing)
        """
        df = pd.read_csv(csv_path)
        
        # Group K-Fold to prevent data leakage by KepID
        gkf = GroupKFold(n_splits=folds)
        splits = list(gkf.split(df, df["label"].values, df[group_col].values))
        (tr_idx, va_idx) = splits[fold_idx]
        
        # Select train or validation split
        self.df = df.iloc[tr_idx].reset_index(drop=True) if split=="train" else df.iloc[va_idx].reset_index(drop=True)
        self.aug = (augment and split=="train")
        self.std = std  # if you didn't standardize in step 1, set True
        
        print(f"{split.upper()} fold {fold_idx}: {len(self.df)} samples")

    def _collapse(self, arr):
        """Ensure input is (1,H,W) format"""
        if arr.ndim == 3 and arr.shape[0] > 1:   # (T,H,W)
            arr = np.nanmedian(arr, axis=0)[None, ...]
        elif arr.ndim == 2:                      # (H,W)
            arr = arr[None, ...]
        return arr.astype(np.float32)

    def _standardize(self, x):
        """Apply per-image standardization"""
        m, s = np.nanmedian(x), np.nanstd(x) + 1e-8
        return (x - m)/s

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        """Get single sample"""
        row = self.df.iloc[i]
        
        # Load and ensure correct format
        x = np.load(row["pix_path"])
        x = self._collapse(x)  # ensure (1,H,W)
        
        # Optional standardization
        if self.std: 
            x = self._standardize(x)
        
        # Data augmentation for training
        if self.aug:
            # 1-pixel spatial jitter
            if np.random.rand() < 0.5: 
                x = np.roll(x, np.random.choice([-1,0,1]), axis=-1)
            if np.random.rand() < 0.5: 
                x = np.roll(x, np.random.choice([-1,0,1]), axis=-2)
            
            # Small amount of Gaussian noise
            x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
        
        # Convert to tensors
        y = np.float32(row["label"])
        return torch.tensor(x), torch.tensor(y)

def get_label_weights(dataset):
    """Calculate class weights for balanced sampling"""
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    y_array = np.array(labels)
    
    # Weight positive class to balance dataset
    pos_weight = (y_array == 0).sum() / max(1, (y_array == 1).sum())
    sample_weights = np.where(y_array == 1, pos_weight, 1.0).astype(np.float32)
    
    print(f"Class distribution: {np.bincount(y_array.astype(int))}")
    print(f"Positive class weight: {pos_weight:.3f}")
    
    return sample_weights, pos_weight