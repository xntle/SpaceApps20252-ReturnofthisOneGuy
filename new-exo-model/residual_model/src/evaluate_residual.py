import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import os

def evaluate_fold(fold_idx=0):
    """Evaluate a single fold"""
    pred_file = f"models/residual_val_preds_fold{fold_idx}.npz"
    
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        print("Please train the model first.")
        return None
    
    # Load predictions
    data = np.load(pred_file)
    y_true = data['y_true']
    y_prob = data['y_prob']
    
    # Calculate metrics
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Classification at threshold 0.5
    y_pred = (y_prob > 0.5).astype(int)
    
    print(f"=== Fold {fold_idx} Results ===")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, y_pred, target_names=["False Positive", "Confirmed"]))
    
    return {
        'fold': fold_idx,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_prob': y_prob
    }

def plot_roc_pr_curves(results, save_path="models/evaluation_curves.png"):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, result in enumerate(results):
        if result is None:
            continue
            
        y_true = result['y_true']
        y_prob = result['y_prob']
        fold = result['fold']
        color = colors[i % len(colors)]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, color=color, label=f"Fold {fold} (AUC={result['roc_auc']:.3f})")
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(recall, precision, color=color, label=f"Fold {fold} (AUC={result['pr_auc']:.3f})")
    
    # ROC plot formatting
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR plot formatting  
    ax2.axhline(y=sum(results[0]['y_true'])/len(results[0]['y_true']), 
                color='k', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Curves saved to {save_path}")

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find optimal threshold based on different metrics"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(int)
        
        if metric == 'f1':
            from sklearn.metrics import f1_score
            score = f1_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score

def comprehensive_evaluation():
    """Run comprehensive evaluation across all folds"""
    print("=== Residual CNN Model Evaluation ===\\n")
    
    results = []
    pr_aucs = []
    roc_aucs = []
    
    # Evaluate each fold
    for fold in range(5):
        result = evaluate_fold(fold)
        if result is not None:
            results.append(result)
            pr_aucs.append(result['pr_auc'])
            roc_aucs.append(result['roc_auc'])
        print()
    
    if not results:
        print("No evaluation results found. Please train the model first.")
        return
    
    # Summary statistics
    print("=== Cross-Validation Summary ===")
    print(f"PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Folds evaluated: {len(results)}")
    
    # Plot curves
    if results:
        plot_roc_pr_curves(results)
    
    # Threshold analysis on fold 0
    if results:
        print("\\n=== Threshold Analysis (Fold 0) ===")
        y_true = results[0]['y_true']
        y_prob = results[0]['y_prob']
        
        # Find optimal thresholds
        f1_thresh, f1_score = find_optimal_threshold(y_true, y_prob, 'f1')
        ba_thresh, ba_score = find_optimal_threshold(y_true, y_prob, 'balanced_accuracy')
        
        print(f"Optimal F1 threshold: {f1_thresh:.3f} (F1={f1_score:.4f})")
        print(f"Optimal Balanced Accuracy threshold: {ba_thresh:.3f} (BA={ba_score:.4f})")
        
        # Show predictions at different thresholds
        for thresh, name in [(0.3, "Conservative"), (0.5, "Default"), (0.7, "Aggressive")]:
            y_pred = (y_prob > thresh).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            print(f"{name} (t={thresh}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    comprehensive_evaluation()