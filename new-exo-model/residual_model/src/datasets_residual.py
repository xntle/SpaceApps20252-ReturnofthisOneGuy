import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from typing import Optional
import random

class VariableLengthResidualDataset(Dataset):
    """
    Expects a CSV with columns: ['path','label','kepid'].
    Each .npy should be shape [seq_len, features].
    Handles variable length sequences through padding/truncation.
    """
    def __init__(self, csv_path: str, split: str, folds: int = 5, fold_idx: int = 0,
                 group_col: str = "kepid", seed: int = 42,
                 max_length: int = 512, augment: bool = True, 
                 standardize_if_needed: bool = False):
        df = pd.read_csv(csv_path)
        
        # GroupKFold split by star ID to prevent leakage
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=folds)
        groups = df[group_col].values
        idxs = list(gkf.split(df, df["label"].values, groups))

        train_idx, val_idx = idxs[fold_idx]
        if split == "train":
            self.df = df.iloc[train_idx].reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[val_idx].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.max_length = max_length
        self.augment = augment if split == "train" else False
        self.standardize_if_needed = standardize_if_needed
        
        print(f"{split} dataset: {len(self.df)} samples")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def process_sequence(self, arr):
        """Process variable length sequence to fixed length"""
        seq_len, n_features = arr.shape
        
        if seq_len > self.max_length:
            # If too long, take a random window during training, center during validation
            if self.augment:
                start_idx = random.randint(0, seq_len - self.max_length)
            else:
                start_idx = (seq_len - self.max_length) // 2
            arr = arr[start_idx:start_idx + self.max_length]
        elif seq_len < self.max_length:
            # If too short, pad with zeros or repeat the sequence
            if seq_len < self.max_length // 4:  # Very short - repeat
                repeats = (self.max_length // seq_len) + 1
                arr = np.tile(arr, (repeats, 1))[:self.max_length]
            else:  # Pad with zeros
                padding = np.zeros((self.max_length - seq_len, n_features), dtype=arr.dtype)
                arr = np.concatenate([arr, padding], axis=0)
        
        return arr

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = np.load(row["path"])   # [seq_len, features]
        
        # Standardize if needed (per-feature across time)
        if self.standardize_if_needed:
            for feat_idx in range(arr.shape[1]):
                col = arr[:, feat_idx]
                m = np.nanmedian(col)
                s = np.nanstd(col) + 1e-8
                arr[:, feat_idx] = (col - m) / s

        # Process to fixed length
        arr = self.process_sequence(arr)
        
        # Augmentations for training
        if self.augment:
            # Add small gaussian noise
            arr = arr + np.random.normal(0, 0.003, size=arr.shape).astype(arr.dtype)
            
            # Random time shift (circular)
            if arr.shape[0] > 10:
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    arr = np.roll(arr, shift=shift, axis=0)

        # Convert to tensor: [seq_len, features] -> [features, seq_len] for Conv1D
        x = torch.tensor(arr.T, dtype=torch.float32)   # [features, seq_len]
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    xs, ys = zip(*batch)
    xs = torch.stack(xs)  # [batch, features, seq_len]
    ys = torch.stack(ys)  # [batch]
    return xs, ys