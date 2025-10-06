#!/usr/bin/env python3
"""
🔮 Multimodal Exoplanet Detection Inference Template

Use this template to make predictions on new targets.
"""

import torch
import numpy as np
import sys
sys.path.append('src')

from train_multimodal_enhanced import EnhancedMultiModalFusionModel

def predict_exoplanet(kepid, tabular_features, residual_windows=None, pixel_diffs=None):
    """
    Make exoplanet prediction for a target
    
    Args:
        kepid: Kepler ID of target
        tabular_features: Array of shape (39,) with stellar/transit parameters
        residual_windows: Array of shape (5, 128) with lightcurve residuals (optional)
        pixel_diffs: Array of shape (32, 24, 24) with pixel differences (optional)
    
    Returns:
        dict: Prediction results
    """
    
    # Load trained model
    model = EnhancedMultiModalFusionModel(n_tabular_features=39)
    model.load_state_dict(torch.load('models/enhanced_multimodal_fusion_model.pth', 
                                   map_location='cpu'))
    model.eval()
    
    # Prepare data
    tabular_tensor = torch.FloatTensor(tabular_features).unsqueeze(0)  # Add batch dim
    
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
        probability = prediction.sigmoid().item()
    
    # Interpret result
    is_planet = probability > 0.5
    confidence = probability if is_planet else (1 - probability)
    
    return {
        'kepid': kepid,
        'is_planet': is_planet,
        'probability': probability,
        'confidence': confidence,
        'classification': 'CONFIRMED' if is_planet else 'FALSE POSITIVE'
    }

# Example usage:
if __name__ == "__main__":
    # Example with mock data (replace with real features)
    example_kepid = 10797460
    example_features = np.random.randn(39)  # Replace with real stellar parameters
    
    result = predict_exoplanet(example_kepid, example_features)
    print(f"KepID {result['kepid']}: {result['classification']} "
          f"(confidence: {result['confidence']:.3f})")
