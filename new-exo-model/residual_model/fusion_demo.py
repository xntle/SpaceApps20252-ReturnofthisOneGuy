"""
Demo script showing how to combine tabular Random Forest predictions 
with Residual CNN predictions for improved exoplanet classification.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add path to access the tabular model
sys.path.append('../AI_Model_Forest')
sys.path.append('src')

def load_tabular_model():
    """Load the trained Random Forest tabular model"""
    try:
        import joblib
        rf_model = joblib.load('../AI_Model_Forest/trained_model/rf_combined_model.joblib')
        scaler = joblib.load('../AI_Model_Forest/trained_model/scaler_combined.joblib')
        encoder = joblib.load('../AI_Model_Forest/trained_model/label_encoder_combined.joblib')
        imputer_medians = joblib.load('../AI_Model_Forest/trained_model/imputer_medians_combined.joblib')
        
        with open('../AI_Model_Forest/trained_model/feature_columns_combined.txt', 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        return rf_model, scaler, encoder, imputer_medians, feature_cols
    except Exception as e:
        print(f"Error loading tabular model: {e}")
        return None, None, None, None, None

def load_residual_model():
    """Load the trained Residual CNN model"""
    try:
        from models_residual import ResidualCNN1D
        import torch
        
        model = ResidualCNN1D(in_features=128, base_channels=64, max_length=512)
        model.load_state_dict(torch.load('models/residual_cnn_best_fold0.pt', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading residual model: {e}")
        return None

def predict_tabular(kepid, tabular_features=None):
    """Get prediction from tabular model (mock implementation)"""
    # In a real implementation, you would:
    # 1. Look up the KepID in your tabular dataset
    # 2. Extract the orbital/stellar features
    # 3. Apply the same preprocessing as in training
    # 4. Get prediction from the Random Forest model
    
    # For demo purposes, return a mock prediction
    # This would normally come from your trained Random Forest
    return np.random.random()  # Placeholder - replace with actual tabular prediction

def predict_residual(residual_path):
    """Get prediction from residual CNN model"""
    try:
        from predict_residual import predict_single
        prob = predict_single(residual_path, "models/residual_cnn_best_fold0.pt")
        return prob if prob is not None else 0.5
    except Exception as e:
        print(f"Error in residual prediction: {e}")
        return 0.5

def simple_fusion(prob_tabular, prob_residual, method='average'):
    """Combine probabilities from both models"""
    if method == 'average':
        return (prob_tabular + prob_residual) / 2
    elif method == 'weighted':
        # Weight residual model more heavily (adjust weights based on validation performance)
        return 0.4 * prob_tabular + 0.6 * prob_residual
    elif method == 'max':
        return max(prob_tabular, prob_residual)
    elif method == 'product':
        # Multiplication (more conservative)
        return prob_tabular * prob_residual
    else:
        return (prob_tabular + prob_residual) / 2

def analyze_candidate(residual_path):
    """Comprehensive analysis of a single exoplanet candidate"""
    
    print("="*60)
    print("EXOPLANET CANDIDATE ANALYSIS")
    print("="*60)
    
    # Extract KepID from filename
    import re
    kepid_match = re.search(r'residual_(\d+)', os.path.basename(residual_path))
    kepid = int(kepid_match.group(1)) if kepid_match else None
    
    print(f"File: {os.path.basename(residual_path)}")
    print(f"KepID: {kepid}")
    print("-" * 40)
    
    # Get tabular prediction
    print("1. Tabular Model (Random Forest):")
    prob_tabular = predict_tabular(kepid)
    print(f"   Probability: {prob_tabular:.4f}")
    print(f"   Prediction: {'CONFIRMED' if prob_tabular > 0.5 else 'FALSE POSITIVE'}")
    
    # Get residual prediction
    print("\\n2. Residual CNN Model:")
    prob_residual = predict_residual(residual_path)
    print(f"   Probability: {prob_residual:.4f}")
    print(f"   Prediction: {'CONFIRMED' if prob_residual > 0.5 else 'FALSE POSITIVE'}")
    
    # Fusion predictions
    print("\\n3. Fusion Results:")
    methods = ['average', 'weighted', 'max', 'product']
    
    for method in methods:
        prob_fused = simple_fusion(prob_tabular, prob_residual, method)
        pred_fused = 'CONFIRMED' if prob_fused > 0.5 else 'FALSE POSITIVE'
        print(f"   {method.capitalize():10s}: {prob_fused:.4f} ({pred_fused})")
    
    # Confidence analysis
    print("\\n4. Confidence Analysis:")
    avg_prob = simple_fusion(prob_tabular, prob_residual, 'average')
    confidence = abs(avg_prob - 0.5) + 0.5
    agreement = abs(prob_tabular - prob_residual)
    
    print(f"   Overall confidence: {confidence:.3f}")
    print(f"   Model agreement: {1 - agreement:.3f}")
    
    if agreement < 0.2:
        print("   ✓ Models strongly agree")
    elif agreement < 0.4:
        print("   ⚠ Models moderately agree")
    else:
        print("   ⚠ Models disagree - requires expert review")
    
    # Recommendation
    print("\\n5. Recommendation:")
    if confidence > 0.8 and agreement < 0.3:
        print("   HIGH CONFIDENCE - Suitable for automated classification")
    elif confidence > 0.6:
        print("   MEDIUM CONFIDENCE - Consider for follow-up observations")
    else:
        print("   LOW CONFIDENCE - Requires expert review")
    
    return {
        'kepid': kepid,
        'prob_tabular': prob_tabular,
        'prob_residual': prob_residual,
        'prob_fused': avg_prob,
        'confidence': confidence,
        'agreement': 1 - agreement
    }

def batch_analysis(residual_files):
    """Analyze multiple candidates and create summary report"""
    
    results = []
    
    print("Processing candidates...")
    for i, residual_path in enumerate(residual_files):
        print(f"\\rProgress: {i+1}/{len(residual_files)}", end="", flush=True)
        
        # Extract KepID
        import re
        kepid_match = re.search(r'residual_(\d+)', os.path.basename(residual_path))
        kepid = int(kepid_match.group(1)) if kepid_match else None
        
        if kepid is None:
            continue
            
        try:
            prob_tabular = predict_tabular(kepid)
            prob_residual = predict_residual(residual_path)
            prob_fused = simple_fusion(prob_tabular, prob_residual, 'weighted')
            
            confidence = abs(prob_fused - 0.5) + 0.5
            agreement = 1 - abs(prob_tabular - prob_residual)
            
            results.append({
                'kepid': kepid,
                'filename': os.path.basename(residual_path),
                'prob_tabular': prob_tabular,
                'prob_residual': prob_residual,
                'prob_fused': prob_fused,
                'prediction': 'CONFIRMED' if prob_fused > 0.5 else 'FALSE POSITIVE',
                'confidence': confidence,
                'agreement': agreement
            })
        except Exception as e:
            print(f"\\nError processing {residual_path}: {e}")
    
    print("\\nDone!")
    
    # Create summary report
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print("\\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total candidates analyzed: {len(df)}")
        print(f"Confirmed predictions: {len(df[df['prediction'] == 'CONFIRMED'])}")
        print(f"False positive predictions: {len(df[df['prediction'] == 'FALSE POSITIVE'])}")
        
        print(f"\\nConfidence statistics:")
        print(f"  Mean confidence: {df['confidence'].mean():.3f}")
        print(f"  High confidence (>0.8): {len(df[df['confidence'] > 0.8])}")
        print(f"  Low confidence (<0.6): {len(df[df['confidence'] < 0.6])}")
        
        print(f"\\nModel agreement statistics:")
        print(f"  Mean agreement: {df['agreement'].mean():.3f}")
        print(f"  Strong agreement (>0.8): {len(df[df['agreement'] > 0.8])}")
        print(f"  Poor agreement (<0.6): {len(df[df['agreement'] < 0.6])}")
        
        # Save results
        df.to_csv('fusion_results.csv', index=False)
        print(f"\\nResults saved to fusion_results.csv")
        
        # Show top candidates
        print(f"\\nTop 5 CONFIRMED candidates by confidence:")
        confirmed = df[df['prediction'] == 'CONFIRMED'].sort_values('confidence', ascending=False)
        print(confirmed[['kepid', 'prob_fused', 'confidence', 'agreement']].head().to_string(index=False))
    
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fusion_demo.py <residual_file.npy>                    # Single analysis")
        print("  python fusion_demo.py --batch <dir_or_files>                # Batch analysis")
        print("\\nExample:")
        print("  python fusion_demo.py processed/residual_windows_std/residual_10024051.npy")
        print("  python fusion_demo.py --batch processed/residual_windows_std/*.npy")
        return
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Please provide residual files for batch analysis")
            return
        
        import glob
        if len(sys.argv) == 3 and '*' in sys.argv[2]:
            # Handle glob pattern
            files = glob.glob(sys.argv[2])
        else:
            # Handle multiple files
            files = sys.argv[2:]
        
        batch_analysis(files)
    
    else:
        # Single file analysis
        residual_file = sys.argv[1]
        if not os.path.exists(residual_file):
            print(f"File not found: {residual_file}")
            return
        
        analyze_candidate(residual_file)

if __name__ == "__main__":
    main()