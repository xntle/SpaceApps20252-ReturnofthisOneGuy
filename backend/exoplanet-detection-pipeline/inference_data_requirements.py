#!/usr/bin/env python3
"""
🚀 Multimodal Model Inference Data Requirements Guide

This script demonstrates exactly what data is needed to make an inference
with the enhanced multimodal model and how to prepare it.
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append('src')
from data_loader import load_and_prepare_data, create_train_val_test_splits
from cnn_data_loader import load_cnn_data
from features import download_lightcurve, preprocess_lightcurve, create_residual_windows
from pixel_diff import compute_pixel_differences

def demonstrate_inference_data_requirements():
    """Show exactly what data is needed for multimodal inference"""
    
    print("🚀 MULTIMODAL MODEL INFERENCE DATA REQUIREMENTS")
    print("=" * 60)
    print("This guide shows what data you need to make a prediction")
    print("with the enhanced multimodal exoplanet detection model.")
    print()
    
    print("📋 THREE DATA MODALITIES REQUIRED:")
    print("-" * 40)
    
    # 1. TABULAR DATA
    print("1️⃣  TABULAR FEATURES (39 features required)")
    print("   📊 Stellar and transit parameters:")
    print()
    
    # Load actual feature names
    try:
        all_data = load_and_prepare_data()
        splits = create_train_val_test_splits(all_data)
        feature_names = splits['feature_names']
        
        print(f"   Required features ({len(feature_names)}):")
        for i, feature in enumerate(feature_names, 1):
            print(f"   {i:2d}. {feature}")
        print()
        
    except Exception as e:
        print(f"   ❌ Could not load feature names: {e}")
        print("   📝 Typical features include:")
        print("      • koi_period (orbital period)")
        print("      • koi_depth (transit depth)")
        print("      • koi_duration (transit duration)")
        print("      • stellar temperature, radius, mass")
        print("      • And 30+ other derived features")
        print()
    
    # 2. CNN 1D DATA
    print("2️⃣  RESIDUAL WINDOWS (1D CNN data)")
    print("   📈 Lightcurve analysis:")
    print("   • Input shape: (n_windows, 128) - multiple 128-point residual windows")
    print("   • Data type: Normalized flux residuals around transit events")
    print("   • Source: Kepler/TESS lightcurve data processed with detrending")
    print("   • Window extraction: Phase-folded transit windows")
    print("   • Default: 5 windows per target (if available)")
    print()
    
    # 3. CNN 2D DATA
    print("3️⃣  PIXEL DIFFERENCES (2D CNN data)")
    print("   🖼️  Pixel-level analysis:")
    print("   • Input shape: (32, 24, 24) - 32 phase bins × 24×24 pixel images")
    print("   • Data type: Pixel brightness differences during transit")
    print("   • Source: Target Pixel Files (TPF) from Kepler/TESS")
    print("   • Phase binning: Transit event divided into 32 temporal phases")
    print("   • Spatial: 24×24 pixel postage stamps around target star")
    print()
    
    print("🎯 DATA PREPARATION WORKFLOW:")
    print("-" * 40)
    print("For a new target (KepID/TIC), you need:")
    print()
    print("STEP 1: Extract tabular features")
    print("   • Download stellar parameters from NASA Exoplanet Archive")
    print("   • Calculate derived features (period ratios, error metrics, etc.)")
    print("   • Apply same preprocessing (median imputation, standardization)")
    print()
    print("STEP 2: Process lightcurve data (1D CNN)")
    print("   • Download lightcurve: download_lightcurve(kepid)")
    print("   • Preprocess: preprocess_lightcurve(lc)")
    print("   • Extract windows: create_residual_windows(lc, period, t0, duration)")
    print("   • Shape: Pad/truncate to (5, 128) for model input")
    print()
    print("STEP 3: Process pixel data (2D CNN)")
    print("   • Download Target Pixel File (TPF)")
    print("   • Extract pixel differences: compute_pixel_differences(tpf, period, t0)")
    print("   • Shape: (32, 24, 24) phase-binned pixel differences")
    print()
    
    return feature_names if 'feature_names' in locals() else None

def create_example_inference_data():
    """Create example data showing the exact format needed"""
    
    print("💡 EXAMPLE INFERENCE DATA FORMAT:")
    print("-" * 40)
    
    # Example tabular data (would come from real stellar parameters)
    print("📊 Example tabular features (39 features):")
    tabular_example = np.random.randn(1, 39)  # 1 sample, 39 features
    print(f"   Shape: {tabular_example.shape}")
    print(f"   Type: Standardized float values")
    print(f"   Sample values: [{tabular_example[0, :5].round(3)}...]")
    print()
    
    # Example CNN 1D data
    print("📈 Example residual windows (1D CNN):")
    cnn1d_example = np.random.randn(1, 5, 128)  # 1 sample, 5 windows, 128 points each
    print(f"   Shape: {cnn1d_example.shape}")
    print(f"   Type: Normalized flux residuals")
    print(f"   Sample window: [{cnn1d_example[0, 0, :5].round(3)}...]")
    print()
    
    # Example CNN 2D data  
    print("🖼️  Example pixel differences (2D CNN):")
    cnn2d_example = np.random.randn(1, 32, 24, 24)  # 1 sample, 32 phases, 24x24 pixels
    print(f"   Shape: {cnn2d_example.shape}")
    print(f"   Type: Pixel brightness differences")
    print(f"   Sample pixel: [{cnn2d_example[0, 0, 0, :5].round(3)}...]")
    print()
    
    return tabular_example, cnn1d_example, cnn2d_example

def show_model_inference_process():
    """Demonstrate the actual inference process"""
    
    print("🔮 MODEL INFERENCE PROCESS:")
    print("-" * 40)
    
    # Create example data
    tabular_data, cnn1d_data, cnn2d_data = create_example_inference_data()
    
    print("STEP 1: Load trained model")
    print("   model = EnhancedMultiModalFusionModel()")
    print("   model.load_state_dict(torch.load('models/enhanced_multimodal_fusion_model.pth'))")
    print("   model.eval()")
    print()
    
    print("STEP 2: Prepare input tensors")
    print("   tabular_tensor = torch.FloatTensor(tabular_data)")
    print("   cnn1d_tensor = torch.FloatTensor(cnn1d_data)")
    print("   cnn2d_tensor = torch.FloatTensor(cnn2d_data)")
    print()
    
    print("STEP 3: Make prediction")
    print("   with torch.no_grad():")
    print("       prediction = model(tabular_tensor, cnn1d_tensor, cnn2d_tensor)")
    print("       probability = prediction.sigmoid().item()")
    print()
    
    print("STEP 4: Interpret result")
    print("   if probability > 0.5:")
    print("       print(f'CONFIRMED exoplanet (confidence: {probability:.3f})')")
    print("   else:")
    print("       print(f'FALSE POSITIVE (confidence: {1-probability:.3f})')")
    print()

def check_data_availability():
    """Check what data is currently available for inference"""
    
    print("📂 CURRENT DATA AVAILABILITY:")
    print("-" * 40)
    
    # Check tabular data
    try:
        all_data = load_and_prepare_data()
        splits = create_train_val_test_splits(all_data)
        print(f"✅ Tabular data: {len(splits['train']['X']) + len(splits['val']['X']) + len(splits['test']['X'])} samples")
        print(f"   Features: {len(splits['feature_names'])}")
    except Exception as e:
        print(f"❌ Tabular data: Error loading ({e})")
    
    # Check CNN 1D data
    residual_dir = Path('data/processed/residual_windows_std')
    if residual_dir.exists():
        residual_files = list(residual_dir.glob('*.npy'))
        print(f"✅ CNN 1D data: {len(residual_files)} residual window files")
    else:
        residual_dir_alt = Path('data/processed/residual_windows')
        if residual_dir_alt.exists():
            residual_files = list(residual_dir_alt.glob('*.npy'))
            print(f"⚠️  CNN 1D data: {len(residual_files)} residual files (unstandardized)")
        else:
            print("❌ CNN 1D data: No residual window files found")
    
    # Check CNN 2D data
    pixel_dir = Path('data/processed/pixel_diffs_std')
    if pixel_dir.exists():
        pixel_files = list(pixel_dir.glob('*.npy'))
        print(f"✅ CNN 2D data: {len(pixel_files)} pixel difference files")
    else:
        pixel_dir_alt = Path('data/processed/pixel_diffs')
        if pixel_dir_alt.exists():
            pixel_files = list(pixel_dir_alt.glob('*.npy'))
            print(f"⚠️  CNN 2D data: {len(pixel_files)} pixel files (unstandardized)")
        else:
            print("❌ CNN 2D data: No pixel difference files found")
    
    # Coverage analysis
    print()
    print("📊 Data coverage analysis:")
    total_targets = 9777  # From our dataset
    
    if 'residual_files' in locals() and 'pixel_files' in locals():
        # Extract KepIDs from filenames
        residual_kepids = set()
        pixel_kepids = set()
        
        for f in residual_files:
            try:
                kepid = int(f.stem.split('_')[1])
                residual_kepids.add(kepid)
            except:
                pass
                
        for f in pixel_files:
            try:
                kepid = int(f.stem.split('_')[1])
                pixel_kepids.add(kepid)
            except:
                pass
        
        both_kepids = residual_kepids & pixel_kepids
        
        print(f"   Residual windows: {len(residual_kepids)} KepIDs ({len(residual_kepids)/total_targets*100:.1f}%)")
        print(f"   Pixel differences: {len(pixel_kepids)} KepIDs ({len(pixel_kepids)/total_targets*100:.1f}%)")
        print(f"   Both modalities: {len(both_kepids)} KepIDs ({len(both_kepids)/total_targets*100:.1f}%)")
        
        if len(both_kepids) < total_targets * 0.1:
            print("   ⚠️  Low multimodal coverage - may limit model performance")
        else:
            print("   ✅ Good multimodal coverage for training")
    
    print()

def create_inference_template():
    """Create a template script for making inferences"""
    
    template_code = '''#!/usr/bin/env python3
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
'''
    
    with open('inference_template.py', 'w') as f:
        f.write(template_code)
    
    print("📝 INFERENCE TEMPLATE CREATED:")
    print("-" * 40)
    print("✅ Created 'inference_template.py'")
    print("   This template shows how to make predictions on new targets")
    print("   Modify the example to use real stellar parameters and CNN data")
    print()

def main():
    """Main demonstration function"""
    
    print("🚀 MULTIMODAL EXOPLANET DETECTION INFERENCE GUIDE")
    print("=" * 60)
    print()
    
    # Show data requirements
    feature_names = demonstrate_inference_data_requirements()
    
    # Show example data format
    create_example_inference_data()
    
    # Show inference process
    show_model_inference_process()
    
    # Check current data availability
    check_data_availability()
    
    # Create inference template
    create_inference_template()
    
    print("🎯 SUMMARY:")
    print("=" * 40)
    print("To make an inference with the multimodal model, you need:")
    print()
    print("1️⃣  TABULAR: 39 stellar/transit features (always required)")
    print("2️⃣  CNN 1D: Residual windows from lightcurve (optional, improves accuracy)")
    print("3️⃣  CNN 2D: Pixel differences from TPF (optional, improves accuracy)")
    print()
    print("✅ Model can work with just tabular data (93.6% accuracy)")
    print("🚀 Full multimodal gives best performance when CNN coverage is high")
    print()
    print("📂 Current CNN coverage: ~3.9% (377 files / 9777 targets)")
    print("🎯 For better multimodal performance, expand CNN coverage to 10%+")

if __name__ == "__main__":
    main()