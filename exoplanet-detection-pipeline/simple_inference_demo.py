#!/usr/bin/env python3
"""
🎯 Simple Inference Examples

Ready-to-use functions for making exoplanet predictions
"""

import torch
import numpy as np
import sys
sys.path.append('src')

from models import TabularNet
from train_multimodal_enhanced import EnhancedMultiModalFusionModel

def predict_tabular_only(stellar_features):
    """
    Make prediction using tabular-only model (recommended)
    
    Args:
        stellar_features: Array of 39 standardized stellar/transit features
        
    Returns:
        dict: Prediction results
    """
    
    # Load model
    model = TabularNet(input_size=39)
    model.load_state_dict(torch.load('models/tabular_model.pth', map_location='cpu'))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        features_tensor = torch.FloatTensor(stellar_features).unsqueeze(0)
        logits = model(features_tensor)
        probability = torch.sigmoid(logits).item()
    
    # Interpret result
    is_planet = probability > 0.5
    confidence = probability if is_planet else (1 - probability)
    
    return {
        'model': 'tabular_only',
        'probability': probability,
        'classification': 'CONFIRMED' if is_planet else 'FALSE POSITIVE',
        'confidence': confidence,
        'accuracy': '93.6%',
        'parameters': '52,353'
    }

def predict_multimodal(stellar_features, residual_windows=None, pixel_diffs=None):
    """
    Make prediction using multimodal model
    
    Args:
        stellar_features: Array of 39 standardized stellar/transit features
        residual_windows: Array of shape (5, 128) with lightcurve residuals (optional)
        pixel_diffs: Array of shape (32, 24, 24) with pixel differences (optional)
        
    Returns:
        dict: Prediction results
    """
    
    # Load model
    model = EnhancedMultiModalFusionModel(n_tabular_features=39)
    model.load_state_dict(torch.load('models/enhanced_multimodal_fusion_model.pth', map_location='cpu'))
    model.eval()
    
    # Prepare inputs
    tabular_tensor = torch.FloatTensor(stellar_features).unsqueeze(0)
    
    # Handle missing CNN data with zeros
    if residual_windows is None:
        cnn1d_tensor = torch.zeros(1, 5, 128)
    else:
        cnn1d_tensor = torch.FloatTensor(residual_windows).unsqueeze(0)
    
    if pixel_diffs is None:
        cnn2d_tensor = torch.zeros(1, 32, 24, 24)
    else:
        cnn2d_tensor = torch.FloatTensor(pixel_diffs).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(tabular_tensor, cnn1d_tensor, cnn2d_tensor)
        probability = prediction.item()
    
    # Interpret result
    is_planet = probability > 0.5
    confidence = probability if is_planet else (1 - probability)
    
    return {
        'model': 'multimodal',
        'probability': probability,
        'classification': 'CONFIRMED' if is_planet else 'FALSE POSITIVE',
        'confidence': confidence,
        'accuracy': '87.2%',
        'parameters': '1,182,341',
        'cnn_data_used': residual_windows is not None or pixel_diffs is not None
    }

def demo_both_models():
    """Demonstrate both models with example data"""
    
    print("🎯 SIMPLE INFERENCE DEMO")
    print("=" * 40)
    
    # Example stellar features (39 features - would come from real data)
    # These should be standardized stellar/transit parameters
    example_features = np.random.randn(39)
    
    print("📊 Making predictions with example stellar parameters...")
    print(f"Features shape: {example_features.shape}")
    print()
    
    # Tabular-only prediction
    print("1️⃣  TABULAR-ONLY PREDICTION:")
    print("-" * 30)
    
    tabular_result = predict_tabular_only(example_features)
    
    print(f"Model: {tabular_result['model']}")
    print(f"Classification: {tabular_result['classification']}")
    print(f"Probability: {tabular_result['probability']:.4f}")
    print(f"Confidence: {tabular_result['confidence']:.4f}")
    print(f"Model accuracy: {tabular_result['accuracy']}")
    print(f"Parameters: {tabular_result['parameters']}")
    print()
    
    # Multimodal prediction (without CNN data)
    print("2️⃣  MULTIMODAL PREDICTION (no CNN data):")
    print("-" * 30)
    
    multimodal_result = predict_multimodal(example_features)
    
    print(f"Model: {multimodal_result['model']}")
    print(f"Classification: {multimodal_result['classification']}")
    print(f"Probability: {multimodal_result['probability']:.4f}")
    print(f"Confidence: {multimodal_result['confidence']:.4f}")
    print(f"Model accuracy: {multimodal_result['accuracy']}")
    print(f"Parameters: {multimodal_result['parameters']}")
    print(f"CNN data used: {multimodal_result['cnn_data_used']}")
    print()
    
    # Show what multimodal would look like with CNN data
    print("3️⃣  MULTIMODAL PREDICTION (with mock CNN data):")
    print("-" * 30)
    
    # Create mock CNN data
    mock_residuals = np.random.randn(5, 128)      # 5 windows, 128 points
    mock_pixels = np.random.randn(32, 24, 24)     # 32 phases, 24x24 pixels
    
    multimodal_cnn_result = predict_multimodal(
        example_features, 
        residual_windows=mock_residuals,
        pixel_diffs=mock_pixels
    )
    
    print(f"Model: {multimodal_cnn_result['model']}")
    print(f"Classification: {multimodal_cnn_result['classification']}")
    print(f"Probability: {multimodal_cnn_result['probability']:.4f}")
    print(f"Confidence: {multimodal_cnn_result['confidence']:.4f}")
    print(f"CNN data used: {multimodal_cnn_result['cnn_data_used']}")
    print()
    
    # Recommendation
    print("💡 RECOMMENDATION:")
    print("-" * 30)
    print("✅ Use tabular-only model for production:")
    print(f"   • Better accuracy: {tabular_result['accuracy']} vs {multimodal_result['accuracy']}")
    print(f"   • Simpler: {tabular_result['parameters']} vs {multimodal_result['parameters']} parameters")
    print("   • No CNN data required")
    print("   • More reliable results")

if __name__ == "__main__":
    demo_both_models()
    
    print("\n" + "="*50)
    print("📚 USAGE IN YOUR CODE:")
    print("="*50)
    print("""
# Tabular-only prediction (recommended)
result = predict_tabular_only(your_39_features)
print(f"Result: {result['classification']} ({result['confidence']:.3f})")

# Multimodal prediction
result = predict_multimodal(
    your_39_features,
    residual_windows=your_cnn1d_data,  # Optional
    pixel_diffs=your_cnn2d_data        # Optional
)
print(f"Result: {result['classification']} ({result['confidence']:.3f})")
""")