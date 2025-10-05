#!/usr/bin/env python3
"""
🔮 Exoplanet Detection Inference Demo

This script demonstrates how to make predictions using both:
1. Tabular-only model (93.6% accuracy - recommended)
2. Multimodal model (enhanced with CNN data when available)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append('src')
from models import TabularNet
from train_multimodal_enhanced import EnhancedMultiModalFusionModel
from data_loader import load_and_prepare_data, create_train_val_test_splits

def load_tabular_model():
    """Load the standalone tabular model"""
    print("📊 Loading tabular-only model...")
    
    model = TabularNet(input_size=39)
    
    # Try to load the model
    model_path = 'models/tabular_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Tabular model loaded from {model_path}")
        return model
    else:
        print(f"❌ Tabular model not found at {model_path}")
        return None

def load_multimodal_model():
    """Load the enhanced multimodal model"""
    print("🚀 Loading multimodal fusion model...")
    
    model = EnhancedMultiModalFusionModel(n_tabular_features=39)
    
    # Try to load the model
    model_path = 'models/enhanced_multimodal_fusion_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Multimodal model loaded from {model_path}")
        return model
    else:
        print(f"❌ Multimodal model not found at {model_path}")
        return None

def get_sample_data():
    """Get sample data for demonstration"""
    print("📂 Loading sample data...")
    
    try:
        # Load the actual dataset
        all_data = load_and_prepare_data()
        splits = create_train_val_test_splits(all_data)
        
        # Get a few samples from validation set
        X_val = splits['val']['X']
        y_val = splits['val']['y']
        feature_names = splits['feature_names']
        
        # Select 5 samples for demo
        n_samples = min(5, len(X_val))
        sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
        
        sample_X = X_val[sample_indices]
        sample_y = y_val[sample_indices]
        
        print(f"✅ Loaded {n_samples} validation samples for demonstration")
        print(f"   Features: {len(feature_names)}")
        print(f"   True labels: {sample_y}")
        
        return sample_X, sample_y, feature_names
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        
        # Create mock data as fallback
        print("🔧 Creating mock data for demonstration...")
        sample_X = np.random.randn(3, 39)  # 3 samples, 39 features
        sample_y = np.array([1, 0, 1])     # Mock labels
        feature_names = [f"feature_{i}" for i in range(39)]
        
        return sample_X, sample_y, feature_names

def predict_tabular_only(model, X, feature_names=None):
    """Make predictions using tabular-only model"""
    print("\n" + "="*50)
    print("📊 TABULAR-ONLY PREDICTIONS")
    print("="*50)
    
    if model is None:
        print("❌ Tabular model not available")
        return None
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X)
    
    # Make predictions
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).numpy().flatten()
    
    # Display results
    print(f"🎯 Predicting {len(X)} samples...")
    print(f"📈 Model: TabularNet (52,353 parameters)")
    print(f"📊 Input: {X.shape[1]} stellar/transit features")
    print()
    
    results = []
    for i, prob in enumerate(probabilities):
        is_planet = prob > 0.5
        confidence = prob if is_planet else (1 - prob)
        classification = "CONFIRMED" if is_planet else "FALSE POSITIVE"
        
        print(f"Sample {i+1}:")
        print(f"   Probability: {prob:.4f}")
        print(f"   Classification: {classification}")
        print(f"   Confidence: {confidence:.4f}")
        print()
        
        results.append({
            'sample': i+1,
            'probability': prob,
            'classification': classification,
            'confidence': confidence
        })
    
    return results

def predict_multimodal(model, X):
    """Make predictions using multimodal model"""
    print("\n" + "="*50)
    print("🚀 MULTIMODAL PREDICTIONS")
    print("="*50)
    
    if model is None:
        print("❌ Multimodal model not available")
        return None
    
    # Prepare inputs
    tabular_tensor = torch.FloatTensor(X)
    
    # Create dummy CNN data (since we don't have CNN data for most samples)
    batch_size = len(X)
    cnn1d_tensor = torch.zeros(batch_size, 5, 128)     # 5 windows, 128 points each
    cnn2d_tensor = torch.zeros(batch_size, 32, 24, 24) # 32 phases, 24x24 pixels
    
    print(f"🎯 Predicting {len(X)} samples...")
    print(f"📈 Model: EnhancedMultiModalFusionModel (1,182,341 parameters)")
    print(f"📊 Tabular input: {X.shape[1]} features")
    print(f"📈 CNN 1D input: {cnn1d_tensor.shape} (zeros - missing data)")
    print(f"🖼️  CNN 2D input: {cnn2d_tensor.shape} (zeros - missing data)")
    print(f"⚠️  Note: Using zero-padding for missing CNN data")
    print()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(tabular_tensor, cnn1d_tensor, cnn2d_tensor)
        probabilities = predictions.numpy().flatten()
    
    # Display results
    results = []
    for i, prob in enumerate(probabilities):
        is_planet = prob > 0.5
        confidence = prob if is_planet else (1 - prob)
        classification = "CONFIRMED" if is_planet else "FALSE POSITIVE"
        
        print(f"Sample {i+1}:")
        print(f"   Probability: {prob:.4f}")
        print(f"   Classification: {classification}")
        print(f"   Confidence: {confidence:.4f}")
        print()
        
        results.append({
            'sample': i+1,
            'probability': prob,
            'classification': classification,
            'confidence': confidence
        })
    
    return results

def compare_predictions(tabular_results, multimodal_results, true_labels):
    """Compare predictions from both models"""
    print("\n" + "="*50)
    print("🔍 PREDICTION COMPARISON")
    print("="*50)
    
    if tabular_results is None or multimodal_results is None:
        print("❌ Cannot compare - one or both models unavailable")
        return
    
    print(f"{'Sample':<8} {'True':<6} {'Tabular':<10} {'Multimodal':<12} {'Tab Conf':<10} {'MM Conf':<10} {'Agreement':<10}")
    print("-" * 80)
    
    agreements = 0
    for i in range(len(tabular_results)):
        true_label = "PLANET" if true_labels[i] == 1 else "NON-PLANET"
        tab_pred = "PLANET" if tabular_results[i]['classification'] == "CONFIRMED" else "NON-PLANET"
        mm_pred = "PLANET" if multimodal_results[i]['classification'] == "CONFIRMED" else "NON-PLANET"
        
        tab_conf = tabular_results[i]['confidence']
        mm_conf = multimodal_results[i]['confidence']
        
        agreement = "✅ YES" if tab_pred == mm_pred else "❌ NO"
        if tab_pred == mm_pred:
            agreements += 1
        
        print(f"{i+1:<8} {true_label:<6} {tab_pred:<10} {mm_pred:<12} {tab_conf:<10.3f} {mm_conf:<10.3f} {agreement:<10}")
    
    agreement_rate = agreements / len(tabular_results) * 100
    print(f"\n📊 Model Agreement: {agreements}/{len(tabular_results)} ({agreement_rate:.1f}%)")
    
    # Performance insights
    print(f"\n💡 Key Insights:")
    print(f"   • Tabular model: 93.6% validation accuracy (recommended)")
    print(f"   • Multimodal model: 87.2% validation accuracy")
    print(f"   • Low agreement may indicate multimodal overfitting with missing CNN data")
    print(f"   • For production: Use tabular-only model for best performance")

def demonstrate_inference_with_real_cnn_data():
    """Show how inference would work with real CNN data"""
    print("\n" + "="*50)
    print("🎯 INFERENCE WITH REAL CNN DATA")
    print("="*50)
    
    print("For complete multimodal inference, you would need:")
    print()
    print("1️⃣  TABULAR FEATURES (39 features):")
    print("   • Stellar parameters: temperature, radius, mass, metallicity")
    print("   • Transit parameters: period, depth, duration, impact") 
    print("   • Derived features: duty cycle, log ratios, error metrics")
    print()
    print("2️⃣  CNN 1D DATA (Lightcurve residuals):")
    print("   • Download lightcurve: lk.search_lightcurve('KIC 10797460')")
    print("   • Extract residuals: create_residual_windows(lc, period, t0, duration)")
    print("   • Shape: (5, 128) - 5 windows of 128 data points")
    print()
    print("3️⃣  CNN 2D DATA (Pixel differences):")
    print("   • Download TPF: lk.search_targetpixelfile('KIC 10797460')")
    print("   • Extract differences: compute_pixel_differences(tpf, period, t0)")
    print("   • Shape: (32, 24, 24) - 32 phase bins × 24×24 pixels")
    print()
    print("📝 Example with real data:")
    print("""
# Complete multimodal inference
from inference_template import predict_exoplanet

result = predict_exoplanet(
    kepid=10797460,
    tabular_features=stellar_params,      # 39 features (required)
    residual_windows=lightcurve_data,     # (5, 128) optional
    pixel_diffs=pixel_data               # (32, 24, 24) optional
)

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.3f}")
""")

def main():
    """Main demonstration function"""
    print("🔮 EXOPLANET DETECTION INFERENCE DEMONSTRATION")
    print("=" * 60)
    print("This demo shows both tabular-only and multimodal predictions")
    print()
    
    # Load models
    tabular_model = load_tabular_model()
    multimodal_model = load_multimodal_model()
    
    # Get sample data
    sample_X, sample_y, feature_names = get_sample_data()
    
    # Make tabular predictions
    tabular_results = predict_tabular_only(tabular_model, sample_X, feature_names)
    
    # Make multimodal predictions  
    multimodal_results = predict_multimodal(multimodal_model, sample_X)
    
    # Compare predictions
    compare_predictions(tabular_results, multimodal_results, sample_y)
    
    # Show real CNN data workflow
    demonstrate_inference_with_real_cnn_data()
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDATION")
    print("="*60)
    print("✅ Use TABULAR-ONLY model for production:")
    print("   • 93.6% accuracy (better than multimodal 87.2%)")
    print("   • 22.6x fewer parameters (52K vs 1.18M)")
    print("   • No CNN data required")
    print("   • Faster inference")
    print("   • More reliable results")
    print()
    print("🚀 Use MULTIMODAL when:")
    print("   • CNN coverage >10% (currently 3.9%)")
    print("   • Extensive lightcurve/pixel data available")
    print("   • Research/experimental applications")
    print()
    print("📚 See INFERENCE_GUIDE.md for complete usage instructions")

if __name__ == "__main__":
    main()