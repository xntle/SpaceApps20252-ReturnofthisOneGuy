#!/usr/bin/env python3
"""
Evaluate Pixel CNN cross-validation results
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
import os

def load_fold_results(fold):
    """Load validation results for one fold"""
    try:
        data = np.load(f"models/pixel_val_preds_fold{fold}.npz")
        return data['y_true'], data['y_probs']
    except:
        return None, None

def evaluate_fold(fold, y_true, y_probs):
    """Evaluate one fold"""
    if len(y_true) == 0:
        return None
    
    pr_auc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # Accuracy at default threshold
    y_pred = (y_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'fold': fold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'n_samples': len(y_true),
        'n_positive': y_true.sum()
    }

def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold for F1 score"""
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    
    for t in thresholds:
        y_pred = (y_probs > t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def plot_curves(all_y_true, all_y_probs, save_path="models/pixel_evaluation_curves.png"):
    """Plot PR and ROC curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_probs)
    pr_auc = average_precision_score(all_y_true, all_y_probs)
    ax1.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_probs)
    roc_auc = roc_auc_score(all_y_true, all_y_probs)
    ax2.plot(fpr, tpr, 'r-', label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Curves saved to {save_path}")

def main():
    """Evaluate all folds"""
    print("=== Pixel CNN Cross-Validation Evaluation ===\n")
    
    results = []
    all_y_true = []
    all_y_probs = []
    
    # Evaluate each fold
    for fold in range(5):
        y_true, y_probs = load_fold_results(fold)
        
        if y_true is not None:
            result = evaluate_fold(fold, y_true, y_probs)
            if result:
                results.append(result)
                all_y_true.extend(y_true)
                all_y_probs.extend(y_probs)
                
                print(f"=== Fold {fold} Results ===")
                print(f"PR-AUC: {result['pr_auc']:.4f}")
                print(f"ROC-AUC: {result['roc_auc']:.4f}")
                print(f"Accuracy: {result['accuracy']:.4f}")
                print(f"Samples: {result['n_samples']} ({result['n_positive']} positive)")
                
                # Classification report
                y_pred = (y_probs > 0.5).astype(int)
                print("\nClassification Report (threshold=0.5):")
                target_names = ['False Positive', 'Confirmed']
                print(classification_report(y_true, y_pred, target_names=target_names))
        else:
            print(f"No results found for fold {fold}")
    
    if not results:
        print("No fold results found!")
        return
    
    # Overall statistics
    print("=== Cross-Validation Summary ===")
    pr_aucs = [r['pr_auc'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    
    print(f"PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Folds evaluated: {len(results)}")
    
    # Combined results
    if all_y_true:
        all_y_true = np.array(all_y_true)
        all_y_probs = np.array(all_y_probs)
        
        # Find optimal threshold
        opt_threshold, opt_f1 = find_optimal_threshold(all_y_true, all_y_probs)
        print(f"\n=== Threshold Analysis ===")
        print(f"Optimal F1 threshold: {opt_threshold:.3f} (F1={opt_f1:.4f})")
        
        # Plot curves
        plot_curves(all_y_true, all_y_probs)
        
        print(f"\nTotal samples: {len(all_y_true)}")
        print(f"Class distribution: {np.bincount(all_y_true.astype(int))}")

if __name__ == "__main__":
    main()