import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Tuple, Optional, Any
import os

logger = logging.getLogger(__name__)

def evaluate_single_model(model: torch.nn.Module,
                         data_loader: DataLoader,
                         device: torch.device,
                         threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate a single PyTorch model on given data.
    
    Args:
        model: Trained PyTorch model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Handle different model types
            if hasattr(model, 'forward') and 'tabular_data' in model.forward.__code__.co_varnames:
                # Hybrid model
                outputs = model(data, torch.empty(0, device=device), torch.empty(0, device=device))[0]
            else:
                outputs = model(data)
            
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= threshold).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    probabilities = np.array(all_probabilities).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(targets, predictions, probabilities)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'targets': targets
    }

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Advanced metrics
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = 0.5
    
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['pr_auc'] = np.mean(y_true)
    
    # False positive rate metrics (important for exoplanet detection)
    fpr_thresholds = [0.01, 0.05, 0.1]
    for fpr_thresh in fpr_thresholds:
        recall_at_fpr = calculate_recall_at_fpr(y_true, y_prob, fpr_thresh)
        metrics[f'recall_at_{fpr_thresh:.0%}_fpr'] = recall_at_fpr
    
    # Additional metrics
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['support_positive'] = int(np.sum(y_true))
    metrics['support_negative'] = int(len(y_true) - np.sum(y_true))
    
    return metrics

def calculate_recall_at_fpr(y_true: np.ndarray, 
                           y_prob: np.ndarray, 
                           target_fpr: float) -> float:
    """
    Calculate recall at a specific false positive rate.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_fpr: Target false positive rate
        
    Returns:
        Recall at the target FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Find the threshold that gives the target FPR
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    
    # Take the highest TPR among valid FPR values
    return tpr[idx[-1]]

def evaluate_ensemble(models: Dict[str, torch.nn.Module],
                     stacker: Optional[Any],
                     data_loader: DataLoader,
                     device: torch.device,
                     threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate ensemble of models with optional stacking.
    
    Args:
        models: Dictionary of trained models
        stacker: Trained stacking model (optional)
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Classification threshold
        
    Returns:
        Dictionary with ensemble evaluation results
    """
    # Get predictions from individual models
    individual_results = {}
    individual_probs = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} model")
        result = evaluate_single_model(model, data_loader, device, threshold)
        individual_results[model_name] = result
        individual_probs[model_name] = result['probabilities']
    
    # Simple ensemble (average)
    if len(individual_probs) > 1:
        ensemble_probs = np.mean(list(individual_probs.values()), axis=0)
        ensemble_preds = (ensemble_probs >= threshold).astype(int)
        
        targets = individual_results[list(models.keys())[0]]['targets']
        ensemble_metrics = calculate_metrics(targets, ensemble_preds, ensemble_probs)
        
        ensemble_result = {
            'metrics': ensemble_metrics,
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'targets': targets
        }
    else:
        ensemble_result = None
    
    # Stacked ensemble
    stacked_result = None
    if stacker is not None and len(individual_probs) > 1:
        logger.info("Evaluating stacked ensemble")
        
        # Prepare features for stacker
        stacker_features = np.column_stack(list(individual_probs.values()))
        
        # Get stacker predictions
        if hasattr(stacker, 'predict_proba'):
            stacked_probs = stacker.predict_proba(stacker_features)[:, 1]
        else:
            stacked_probs = stacker.predict(stacker_features)
        
        stacked_preds = (stacked_probs >= threshold).astype(int)
        stacked_metrics = calculate_metrics(targets, stacked_preds, stacked_probs)
        
        stacked_result = {
            'metrics': stacked_metrics,
            'predictions': stacked_preds,
            'probabilities': stacked_probs,
            'targets': targets
        }
    
    return {
        'individual': individual_results,
        'ensemble': ensemble_result,
        'stacked': stacked_result
    }

def plot_roc_curves(results: Dict[str, Dict], 
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot ROC curves for all models.
    
    Args:
        results: Results from evaluate_ensemble
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual models
    for model_name, result in results['individual'].items():
        targets = result['targets']
        probabilities = result['probabilities']
        
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    # Plot ensemble if available
    if results['ensemble'] is not None:
        targets = results['ensemble']['targets']
        probabilities = results['ensemble']['probabilities']
        
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        ax.plot(fpr, tpr, label=f'Simple Ensemble (AUC = {auc:.3f})', 
                linewidth=3, linestyle='--')
    
    # Plot stacked ensemble if available
    if results['stacked'] is not None:
        targets = results['stacked']['targets']
        probabilities = results['stacked']['probabilities']
        
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        ax.plot(fpr, tpr, label=f'Stacked Ensemble (AUC = {auc:.3f})', 
                linewidth=3, linestyle=':')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Exoplanet Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
    
    return fig

def plot_precision_recall_curves(results: Dict[str, Dict],
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot Precision-Recall curves for all models.
    
    Args:
        results: Results from evaluate_ensemble
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual models
    for model_name, result in results['individual'].items():
        targets = result['targets']
        probabilities = result['probabilities']
        
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        avg_precision = average_precision_score(targets, probabilities)
        
        ax.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
    
    # Plot ensemble if available
    if results['ensemble'] is not None:
        targets = results['ensemble']['targets']
        probabilities = results['ensemble']['probabilities']
        
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        avg_precision = average_precision_score(targets, probabilities)
        
        ax.plot(recall, precision, label=f'Simple Ensemble (AP = {avg_precision:.3f})', 
                linewidth=3, linestyle='--')
    
    # Plot stacked ensemble if available
    if results['stacked'] is not None:
        targets = results['stacked']['targets']
        probabilities = results['stacked']['probabilities']
        
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        avg_precision = average_precision_score(targets, probabilities)
        
        ax.plot(recall, precision, label=f'Stacked Ensemble (AP = {avg_precision:.3f})', 
                linewidth=3, linestyle=':')
    
    # Baseline (random classifier)
    baseline = np.mean(targets) if results['individual'] else 0.1
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
               label=f'Baseline (AP = {baseline:.3f})')
    
    # Formatting
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Exoplanet Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PR curves to {save_path}")
    
    return fig

def create_confusion_matrix_plot(results: Dict[str, Dict],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create confusion matrix plots for all models.
    
    Args:
        results: Results from evaluate_ensemble
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_models = len(results['individual'])
    if results['ensemble'] is not None:
        n_models += 1
    if results['stacked'] is not None:
        n_models += 1
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot individual models
    for model_name, result in results['individual'].items():
        cm = confusion_matrix(result['targets'], result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Planet', 'Planet'],
                   yticklabels=['Non-Planet', 'Planet'],
                   ax=axes[plot_idx])
        
        axes[plot_idx].set_title(f'{model_name}\nAccuracy: {result["metrics"]["accuracy"]:.3f}')
        axes[plot_idx].set_ylabel('True Label')
        axes[plot_idx].set_xlabel('Predicted Label')
        
        plot_idx += 1
    
    # Plot ensemble
    if results['ensemble'] is not None:
        cm = confusion_matrix(results['ensemble']['targets'], results['ensemble']['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Non-Planet', 'Planet'],
                   yticklabels=['Non-Planet', 'Planet'],
                   ax=axes[plot_idx])
        
        axes[plot_idx].set_title(f'Simple Ensemble\nAccuracy: {results["ensemble"]["metrics"]["accuracy"]:.3f}')
        axes[plot_idx].set_ylabel('True Label')
        axes[plot_idx].set_xlabel('Predicted Label')
        
        plot_idx += 1
    
    # Plot stacked ensemble
    if results['stacked'] is not None:
        cm = confusion_matrix(results['stacked']['targets'], results['stacked']['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=['Non-Planet', 'Planet'],
                   yticklabels=['Non-Planet', 'Planet'],
                   ax=axes[plot_idx])
        
        axes[plot_idx].set_title(f'Stacked Ensemble\nAccuracy: {results["stacked"]["metrics"]["accuracy"]:.3f}')
        axes[plot_idx].set_ylabel('True Label')
        axes[plot_idx].set_xlabel('Predicted Label')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrices to {save_path}")
    
    return fig

def create_metrics_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table of metrics across all models.
    
    Args:
        results: Results from evaluate_ensemble
        
    Returns:
        DataFrame with metrics comparison
    """
    metrics_data = []
    
    # Individual models
    for model_name, result in results['individual'].items():
        metrics = result['metrics'].copy()
        metrics['model'] = model_name
        metrics['type'] = 'Individual'
        metrics_data.append(metrics)
    
    # Ensemble
    if results['ensemble'] is not None:
        metrics = results['ensemble']['metrics'].copy()
        metrics['model'] = 'Simple Ensemble'
        metrics['type'] = 'Ensemble'
        metrics_data.append(metrics)
    
    # Stacked ensemble
    if results['stacked'] is not None:
        metrics = results['stacked']['metrics'].copy()
        metrics['model'] = 'Stacked Ensemble'
        metrics['type'] = 'Ensemble'
        metrics_data.append(metrics)
    
    df = pd.DataFrame(metrics_data)
    
    # Reorder columns
    first_cols = ['model', 'type', 'roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1_score']
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]
    
    return df

def comprehensive_evaluation_report(models: Dict[str, torch.nn.Module],
                                  stacker: Optional[Any],
                                  data_loaders: Dict[str, DataLoader],
                                  device: torch.device,
                                  output_dir: str = 'evaluation_results/') -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report for all models.
    
    Args:
        models: Dictionary of trained models
        stacker: Trained stacking model (optional)
        data_loaders: Dictionary of data loaders (train, val, test)
        device: Device to run evaluation on
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all evaluation results
    """
    logger.info("Starting comprehensive evaluation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Evaluate on each dataset split
    for split_name, data_loader in data_loaders.items():
        logger.info(f"Evaluating on {split_name} set")
        
        results = evaluate_ensemble(models, stacker, data_loader, device)
        all_results[split_name] = results
        
        # Create metrics comparison table
        metrics_df = create_metrics_comparison_table(results)
        metrics_path = os.path.join(output_dir, f"{split_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved {split_name} metrics to {metrics_path}")
        
        # Print summary
        print(f"\n{split_name.upper()} SET RESULTS:")
        print("="*50)
        print(metrics_df[['model', 'roc_auc', 'pr_auc', 'accuracy', 'recall_at_1%_fpr']].to_string(index=False))
        
        # Generate plots
        roc_path = os.path.join(output_dir, f"{split_name}_roc_curves.png")
        plot_roc_curves(results, roc_path)
        
        pr_path = os.path.join(output_dir, f"{split_name}_pr_curves.png")
        plot_precision_recall_curves(results, pr_path)
        
        cm_path = os.path.join(output_dir, f"{split_name}_confusion_matrices.png")
        create_confusion_matrix_plot(results, cm_path)
    
    # Create summary report
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EXOPLANET DETECTION EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for split_name, results in all_results.items():
            f.write(f"{split_name.upper()} SET RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            metrics_df = create_metrics_comparison_table(results)
            f.write(metrics_df.to_string(index=False))
            f.write("\n\n")
    
    logger.info(f"Comprehensive evaluation completed. Results saved to {output_dir}")
    
    return all_results

if __name__ == "__main__":
    # Test evaluation functions
    logging.basicConfig(level=logging.INFO)
    
    print("Evaluation module loaded successfully!")
    print("Available functions:")
    print("- evaluate_single_model")
    print("- evaluate_ensemble") 
    print("- comprehensive_evaluation_report")
    print("- Various plotting functions")