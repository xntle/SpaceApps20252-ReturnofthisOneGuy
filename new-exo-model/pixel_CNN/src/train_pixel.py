#!/usr/bin/env python3
"""
Train Pixel CNN with cross-validation, weighted sampling, and early stopping
"""
import os, numpy as np, torch, torch.nn as nn, argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from datasets_pixel import PixelDiffDataset, get_label_weights
from models_pixel import get_model

def evaluate(model, loader, device):
    """Evaluate model on validation set"""
    model.eval()
    ys, ps = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            ys.append(y.numpy())
            ps.append(probs)
    
    y_true = np.concatenate(ys)
    y_probs = np.concatenate(ps)
    
    # Calculate metrics
    try:
        pr_auc = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else float("nan")
        roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else float("nan")
    except:
        pr_auc = roc_auc = float("nan")
    
    # Accuracy at 0.5 threshold
    y_pred = (y_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    return pr_auc, roc_auc, acc, y_true, y_probs

def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    
    return total_loss / total_samples

def main(fold=0, epochs=40, base=16, dropout=0.3, model_type="standard", 
         batch_size=32, lr=3e-4, weight_decay=5e-4):
    """Train pixel CNN for one fold"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load datasets
    csv_path = "processed/pixel_manifest.csv"
    print(f"Loading data from {csv_path}")
    
    train_ds = PixelDiffDataset(csv_path, "train", folds=5, fold_idx=fold, 
                               augment=True, std=False)
    val_ds = PixelDiffDataset(csv_path, "val", folds=5, fold_idx=fold, 
                             augment=False, std=False)
    
    # Get class weights for balanced sampling
    sample_weights, pos_weight = get_label_weights(train_ds)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), 
                                   replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                             num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Create model
    model = get_model(model_type=model_type, base=base, dropout=dropout).to(device)
    
    # Loss function with class weighting
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training setup
    os.makedirs("models", exist_ok=True)
    best_pr_auc = -1
    patience_counter = 0
    patience = 8
    
    print(f"\nStarting training for fold {fold}")
    print(f"Model: {model_type}, base_channels: {base}, dropout: {dropout}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        pr_auc, roc_auc, acc, y_true, y_probs = evaluate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"PR-AUC: {pr_auc:.4f} | "
              f"ROC-AUC: {roc_auc:.4f} | "
              f"Acc: {acc:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if pr_auc > best_pr_auc + 1e-4:
            best_pr_auc = pr_auc
            patience_counter = 0
            
            # Save model and predictions
            torch.save(model.state_dict(), f"models/pixel_cnn_best_fold{fold}.pt")
            np.savez(f"models/pixel_val_preds_fold{fold}.npz", 
                    y_true=y_true, y_probs=y_probs)
            print(f"✓ New best PR-AUC: {pr_auc:.4f} - Model saved")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch} epochs (patience={patience})")
            break
    
    print(f"\nFold {fold} complete! Best PR-AUC: {best_pr_auc:.4f}")
    return best_pr_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pixel CNN")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--epochs", type=int, default=40, help="Max epochs")
    parser.add_argument("--base", type=int, default=16, help="Base channels")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--model_type", type=str, default="standard", 
                       choices=["standard", "lightweight"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    
    args = parser.parse_args()
    
    main(
        fold=args.fold,
        epochs=args.epochs,
        base=args.base,
        dropout=args.dropout,
        model_type=args.model_type,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )