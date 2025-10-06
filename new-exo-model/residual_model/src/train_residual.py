import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

from datasets_residual import VariableLengthResidualDataset, collate_fn
from models_residual import ResidualCNN1D, LightweightResidualCNN

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch=0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        num_batches += 1
        
        if batch_idx % 10 == 0 and batch_idx > 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(y.cpu().numpy())
        all_probs.append(probs)
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    
    # Calculate metrics
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    
    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate accuracy at threshold 0.5
    y_pred = (y_prob > 0.5).astype(int)
    accuracy = (y_pred == y_true).mean()
    
    return pr_auc, roc_auc, accuracy, avg_loss, y_true, y_prob

def plot_training_curves(train_losses, val_losses, val_aucs, save_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label="Train Loss", color='blue')
    ax1.plot(val_losses, label="Val Loss", color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot AUCs
    ax2.plot([auc[0] for auc in val_aucs], label="PR-AUC", color='green')
    ax2.plot([auc[1] for auc in val_aucs], label="ROC-AUC", color='orange')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("Validation AUCs")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    csv_path = "processed/residual_manifest.csv"
    folds = 5
    fold_idx = 0  # Change this to run different folds
    max_length = 512
    
    # Create datasets
    print("Creating datasets...")
    train_ds = VariableLengthResidualDataset(
        csv_path, split="train", folds=folds, fold_idx=fold_idx,
        max_length=max_length, augment=True, standardize_if_needed=False
    )
    
    val_ds = VariableLengthResidualDataset(
        csv_path, split="val", folds=folds, fold_idx=fold_idx,
        max_length=max_length, augment=False, standardize_if_needed=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, 
        num_workers=2, pin_memory=True, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn
    )
    
    # Calculate class weights for imbalanced data
    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor(neg_count / max(1, pos_count), dtype=torch.float32, device=device)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Positive samples: {pos_count}, Negative: {neg_count}")
    print(f"Positive weight: {pos_weight.item():.3f}")
    
    # Create model
    model = ResidualCNN1D(in_features=128, base_channels=64, max_length=max_length).to(device)
    
    # Alternative lighter model (uncomment to use):
    # model = LightweightResidualCNN(in_features=128, base_channels=32, max_length=max_length).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    best_pr_auc = -1.0
    patience = 7
    bad_epochs = 0
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    print("\\nStarting training...")
    
    for epoch in range(1, 51):  # Up to 50 epochs
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validation
        pr_auc, roc_auc, accuracy, val_loss, y_true, y_prob = evaluate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append((pr_auc, roc_auc))
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f}")
        
        # Early stopping and model saving
        if pr_auc > best_pr_auc + 1e-4:
            best_pr_auc = pr_auc
            bad_epochs = 0
            
            # Save best model
            os.makedirs("models", exist_ok=True)
            model_path = f"models/residual_cnn_best_fold{fold_idx}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save validation predictions for later fusion
            pred_path = f"models/residual_val_preds_fold{fold_idx}.npz"
            np.savez(pred_path, y_true=y_true, y_prob=y_prob)
            
            print(f"✓ New best PR-AUC: {best_pr_auc:.4f} - Model saved")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_aucs, "models/training_curves.png")
    
    # Final evaluation with detailed classification report
    print("\\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f"models/residual_cnn_best_fold{fold_idx}.pt", map_location=device))
    pr_auc, roc_auc, accuracy, val_loss, y_true, y_prob = evaluate(model, val_loader, device)
    
    print(f"Best Validation Results:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Loss: {val_loss:.4f}")
    
    # Classification report
    y_pred = (y_prob > 0.5).astype(int)
    print(f"\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["False Positive", "Confirmed"]))
    
    print(f"\\nModel and predictions saved in 'models/' directory")
    print(f"Training curves saved as 'models/training_curves.png'")

if __name__ == "__main__":
    main()