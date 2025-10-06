#!/usr/bin/env python3
"""
🌟 Real Data Inference Demo

Shows inference with actual stellar parameters from the dataset
"""

import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('src')

from data_loader import load_and_prepare_data, create_train_val_test_splits
from simple_inference_demo import predict_tabular_only, predict_multimodal

def get_real_examples():
    """Get real examples from the dataset"""
    
    print("📂 Loading real exoplanet data...")
    
    # Load the dataset
    all_data = load_and_prepare_data()
    splits = create_train_val_test_splits(all_data)
    
    # Get validation data
    X_val = splits['val']['X']
    y_val = splits['val']['y']
    
    # Get the original data to show KepIDs
    df = pd.read_csv('data/raw/lighkurve_KOI_dataset_enriched.csv')
    df_filtered = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Select some interesting examples
    confirmed_indices = np.where(y_val == 1)[0][:3]  # 3 confirmed planets
    false_positive_indices = np.where(y_val == 0)[0][:3]  # 3 false positives
    
    example_indices = np.concatenate([confirmed_indices, false_positive_indices])
    
    examples = {
        'features': X_val[example_indices],
        'true_labels': y_val[example_indices],
        'descriptions': []
    }
    
    # Add descriptions
    for i, idx in enumerate(example_indices):
        if i < 3:
            examples['descriptions'].append(f"Confirmed Exoplanet #{i+1}")
        else:
            examples['descriptions'].append(f"False Positive #{i-2}")
    
    return examples

def demonstrate_real_predictions():
    """Demonstrate predictions on real data"""
    
    print("🌟 REAL DATA INFERENCE DEMONSTRATION")
    print("=" * 50)
    
    # Get real examples
    examples = get_real_examples()
    
    print(f"📊 Testing on {len(examples['features'])} real targets from validation set")
    print()
    
    # Make predictions on each example
    for i, (features, true_label, description) in enumerate(zip(
        examples['features'], 
        examples['true_labels'], 
        examples['descriptions']
    )):
        print(f"🎯 TARGET {i+1}: {description}")
        print("-" * 40)
        
        true_class = "CONFIRMED PLANET" if true_label == 1 else "FALSE POSITIVE"
        print(f"Ground Truth: {true_class}")
        print()
        
        # Tabular prediction
        tab_result = predict_tabular_only(features)
        print(f"📊 Tabular Model:")
        print(f"   Prediction: {tab_result['classification']}")
        print(f"   Confidence: {tab_result['confidence']:.3f}")
        print(f"   Correct: {'✅' if (tab_result['classification'] == 'CONFIRMED') == (true_label == 1) else '❌'}")
        
        # Multimodal prediction
        mm_result = predict_multimodal(features)
        print(f"🚀 Multimodal Model:")
        print(f"   Prediction: {mm_result['classification']}")
        print(f"   Confidence: {mm_result['confidence']:.3f}")
        print(f"   Correct: {'✅' if (mm_result['classification'] == 'CONFIRMED') == (true_label == 1) else '❌'}")
        
        print()
    
    # Summary
    print("📋 SUMMARY")
    print("=" * 20)
    
    tabular_correct = 0
    multimodal_correct = 0
    
    for features, true_label in zip(examples['features'], examples['true_labels']):
        tab_result = predict_tabular_only(features)
        mm_result = predict_multimodal(features)
        
        if (tab_result['classification'] == 'CONFIRMED') == (true_label == 1):
            tabular_correct += 1
        
        if (mm_result['classification'] == 'CONFIRMED') == (true_label == 1):
            multimodal_correct += 1
    
    total = len(examples['features'])
    print(f"Tabular accuracy: {tabular_correct}/{total} ({tabular_correct/total*100:.1f}%)")
    print(f"Multimodal accuracy: {multimodal_correct}/{total} ({multimodal_correct/total*100:.1f}%)")

def show_feature_importance():
    """Show what the 39 features represent"""
    
    print("\n🔍 UNDERSTANDING THE 39 INPUT FEATURES")
    print("=" * 45)
    
    # Load feature names
    all_data = load_and_prepare_data()
    splits = create_train_val_test_splits(all_data)
    feature_names = splits['feature_names']
    
    print("These are the 39 stellar and transit parameters required:")
    print()
    
    # Group features by category
    categories = {
        'Transit Parameters': ['period', 'epoch', 'depth', 'duration', 'impact'],
        'Stellar Parameters': ['star_logg', 'star_metallicity', 'star_temp', 'star_radius', 'star_mass'],
        'Error/Uncertainty': ['err1', 'err2'],
        'Derived Features': ['duty_cycle', 'log_', '_err_rel', 'err_asym', 'n_quarters']
    }
    
    for category, keywords in categories.items():
        print(f"📊 {category}:")
        matching_features = []
        for feature in feature_names:
            if any(keyword in feature for keyword in keywords):
                matching_features.append(feature)
        
        for feature in matching_features[:5]:  # Show first 5 of each category
            print(f"   • {feature}")
        
        if len(matching_features) > 5:
            print(f"   ... and {len(matching_features) - 5} more")
        print()
    
    print("💡 Key Points:")
    print("   • All features are standardized (mean=0, std=1)")
    print("   • Missing values filled with median imputation")  
    print("   • Derived features capture complex relationships")
    print("   • Error terms quantify measurement uncertainty")

if __name__ == "__main__":
    demonstrate_real_predictions()
    show_feature_importance()
    
    print("\n" + "="*50)
    print("🎯 READY TO USE")
    print("="*50)
    print("""
To make predictions on your own data:

1. Prepare 39 stellar/transit features (standardized)
2. Use predict_tabular_only() for best results (93.6% accuracy)
3. Use predict_multimodal() if you have CNN data available

Example:
--------
from simple_inference_demo import predict_tabular_only

# Your 39 standardized features
features = np.array([...])  # stellar parameters

result = predict_tabular_only(features)
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.3f}")

See INFERENCE_GUIDE.md for complete documentation!
""")