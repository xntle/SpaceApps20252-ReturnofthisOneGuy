#!/usr/bin/env python3
"""
Detailed accuracy analysis for Pixel CNN validation results
"""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd

def analyze_fold(fold):
    """Analyze one fold in detail"""
    try:
        data = np.load(f"models/pixel_val_preds_fold{fold}.npz")
        y_true = data['y_true']
        y_probs = data['y_probs']
        
        # Different threshold analyses
        thresholds = [0.3, 0.4, 0.459, 0.5, 0.6, 0.7]
        results = []
        
        for thresh in thresholds:
            y_pred = (y_probs > thresh).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            
            # Handle edge cases for precision/recall
            try:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            except:
                precision = recall = f1 = 0.0
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            results.append({
                'fold': fold,
                'threshold': thresh,
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'n_samples': len(y_true),
                'n_positive': y_true.sum(),
                'n_negative': len(y_true) - y_true.sum()
            })
        
        return results
        
    except Exception as e:
        print(f"Error analyzing fold {fold}: {e}")
        return []

def main():
    """Generate detailed accuracy report"""
    print("=" * 60)
    print("DETAILED PIXEL CNN VALIDATION ACCURACY ANALYSIS")
    print("=" * 60)
    
    all_results = []
    
    # Analyze each fold
    for fold in range(5):
        fold_results = analyze_fold(fold)
        all_results.extend(fold_results)
        
        if fold_results:
            print(f"\n📊 FOLD {fold} DETAILED RESULTS:")
            print(f"Validation samples: {fold_results[0]['n_samples']} "
                  f"({fold_results[0]['n_positive']} confirmed, "
                  f"{fold_results[0]['n_negative']} false positives)")
            
            print("\nAccuracy at different thresholds:")
            print("Thresh  |  Accuracy  |  Bal.Acc  |  Precision |  Recall  |  F1-Score")
            print("-" * 70)
            
            for r in fold_results:
                print(f"{r['threshold']:5.3f}  |  {r['accuracy']:7.3f}  |  "
                      f"{r['balanced_accuracy']:6.3f}  |  {r['precision']:8.3f}  |  "
                      f"{r['recall']:6.3f}  |  {r['f1_score']:7.3f}")
    
    # Overall summary
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        
        # Group by threshold
        summary = df.groupby('threshold').agg({
            'accuracy': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std']
        }).round(4)
        
        print("\nMean ± Std across all folds:")
        print("Thresh  |  Accuracy     |  Balanced Acc |  Precision    |  Recall       |  F1-Score")
        print("-" * 85)
        
        for thresh in df['threshold'].unique():
            subset = df[df['threshold'] == thresh]
            acc_mean, acc_std = subset['accuracy'].mean(), subset['accuracy'].std()
            bal_mean, bal_std = subset['balanced_accuracy'].mean(), subset['balanced_accuracy'].std()
            prec_mean, prec_std = subset['precision'].mean(), subset['precision'].std()
            rec_mean, rec_std = subset['recall'].mean(), subset['recall'].std()
            f1_mean, f1_std = subset['f1_score'].mean(), subset['f1_score'].std()
            
            print(f"{thresh:5.3f}  |  {acc_mean:4.3f}±{acc_std:4.3f}  |  "
                  f"{bal_mean:4.3f}±{bal_std:4.3f}  |  {prec_mean:4.3f}±{prec_std:4.3f}  |  "
                  f"{rec_mean:4.3f}±{rec_std:4.3f}  |  {f1_mean:4.3f}±{f1_std:4.3f}")
        
        # Find best performing threshold
        best_acc_thresh = df.loc[df['accuracy'].idxmax()]
        best_bal_thresh = df.loc[df['balanced_accuracy'].idxmax()]
        best_f1_thresh = df.loc[df['f1_score'].idxmax()]
        
        print(f"\n🎯 BEST PERFORMANCE BY METRIC:")
        print(f"Highest Accuracy: {best_acc_thresh['accuracy']:.3f} at threshold {best_acc_thresh['threshold']:.3f} (Fold {best_acc_thresh['fold']})")
        print(f"Highest Balanced Accuracy: {best_bal_thresh['balanced_accuracy']:.3f} at threshold {best_bal_thresh['threshold']:.3f} (Fold {best_bal_thresh['fold']})")
        print(f"Highest F1-Score: {best_f1_thresh['f1_score']:.3f} at threshold {best_f1_thresh['threshold']:.3f} (Fold {best_f1_thresh['fold']})")
        
        # Overall dataset summary
        total_samples = df['n_samples'].sum() // len(df['threshold'].unique())  # Avoid double counting
        total_positive = df['n_positive'].sum() // len(df['threshold'].unique())
        total_negative = df['n_negative'].sum() // len(df['threshold'].unique())
        
        print(f"\n📈 DATASET SUMMARY:")
        print(f"Total validation samples: {total_samples}")
        print(f"Confirmed exoplanets: {total_positive} ({total_positive/total_samples*100:.1f}%)")
        print(f"False positives: {total_negative} ({total_negative/total_samples*100:.1f}%)")
        
        # Recommended threshold
        optimal_thresh = 0.459  # From previous analysis
        opt_results = df[df['threshold'] == optimal_thresh]
        if not opt_results.empty:
            opt_acc = opt_results['accuracy'].mean()
            opt_bal = opt_results['balanced_accuracy'].mean()
            opt_f1 = opt_results['f1_score'].mean()
            
            print(f"\n🎯 RECOMMENDED THRESHOLD: {optimal_thresh}")
            print(f"   Accuracy: {opt_acc:.3f}")
            print(f"   Balanced Accuracy: {opt_bal:.3f}")
            print(f"   F1-Score: {opt_f1:.3f}")

if __name__ == "__main__":
    main()