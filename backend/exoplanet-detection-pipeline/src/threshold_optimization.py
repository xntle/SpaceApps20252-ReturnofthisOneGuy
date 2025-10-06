"""
Threshold optimization for mission-critical FPR targets
"""
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report

def calculate_recall_at_fpr(y_true, y_prob, fpr_threshold=0.05):
    """Calculate recall at a specific false positive rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Find the threshold that gives us the target FPR
    idx = np.where(fpr <= fpr_threshold)[0]
    if len(idx) == 0:
        return 0.0  # Cannot achieve target FPR
    
    # Return the recall (TPR) at the target FPR
    return tpr[idx[-1]]

def threshold_for_fpr(y_true, y_prob, target_fpr=0.05):
    """
    Find the optimal threshold that achieves target false positive rate.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_fpr: Target false positive rate (e.g., 0.05 for 5%)
        
    Returns:
        float: Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Find thresholds that give FPR <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    if len(valid_indices) == 0:
        print(f"Warning: No threshold achieves FPR <= {target_fpr}. Using default 0.5")
        return 0.5
    
    # Pick the threshold with highest TPR among valid FPR values
    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    optimal_threshold = thresholds[best_idx]
    
    achieved_fpr = fpr[best_idx]
    achieved_tpr = tpr[best_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Achieved FPR: {achieved_fpr:.4f} (target: {target_fpr})")
    print(f"Achieved TPR (recall): {achieved_tpr:.4f}")
    
    return optimal_threshold

def validate_threshold_performance(y_true, y_prob, threshold, target_fpr=0.05):
    """
    Validate threshold performance on test set.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities  
        threshold: Threshold to evaluate
        target_fpr: Target FPR for comparison
        
    Returns:
        dict: Performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'recall_at_1pct_fpr': calculate_recall_at_fpr(y_true, y_prob, 0.01),
        'recall_at_5pct_fpr': calculate_recall_at_fpr(y_true, y_prob, 0.05),
        'recall_at_10pct_fpr': calculate_recall_at_fpr(y_true, y_prob, 0.10)
    }
    
    return metrics