#!/usr/bin/env python3
"""
🔮 Enhanced Multimodal Exoplanet Detection - Inference Demo

This script demonstrates how to make predictions with the trained enhanced
multimodal fusion model for exoplanet detection.

Usage:
    python demo_inference.py --kepid 10797460
    python demo_inference.py --batch-predict
    python demo_inference.py --interactive
"""

import torch
import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from models import EnhancedMultiModalFusionModel
from data_loader import load_kepler_features, prepare_sample_features
from cnn_data_loader import load_cnn_sample

def load_trained_model(model_path='models/enhanced_multimodal_fusion_model.pth'):
    """Load the trained enhanced multimodal model"""
    
    # Initialize model architecture
    model = EnhancedMultiModalFusionModel(
        tabular_dim=39,
        residual_length=128,
        pixel_shape=(32, 24, 24)
    )
    
    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using: python train_multimodal_enhanced.py")
        return None
    
    return model

def prepare_single_sample(kepid, df=None):
    """Prepare a single sample for inference"""
    
    if df is None:
        # Load the dataset
        df = load_kepler_features('data/raw')
    
    # Find the target
    target_row = df[df['kepid'] == int(kepid)]
    if target_row.empty:
        print(f"❌ KepID {kepid} not found in dataset")
        return None
    
    # Prepare tabular features
    tabular_features = prepare_sample_features(target_row.iloc[0])
    
    # Load CNN data if available
    residual_windows = load_cnn_sample(kepid, 'residual_windows_std')
    pixel_diffs = load_cnn_sample(kepid, 'pixel_diffs_std')
    
    # Handle missing CNN data
    if residual_windows is None:
        print(f"⚠️ No residual window data for KepID {kepid}, using zeros")
        residual_windows = np.zeros((1, 128))
    
    if pixel_diffs is None:
        print(f"⚠️ No pixel difference data for KepID {kepid}, using zeros")
        pixel_diffs = np.zeros((32, 24, 24))
    
    return {
        'tabular': tabular_features,
        'residual': residual_windows,
        'pixel': pixel_diffs,
        'true_label': target_row.iloc[0]['koi_disposition']
    }

def make_prediction(model, sample_data, threshold=0.9922):
    """Make a prediction for a single sample"""
    
    with torch.no_grad():
        # Convert to tensors
        tabular = torch.FloatTensor(sample_data['tabular']).unsqueeze(0)
        residual = torch.FloatTensor(sample_data['residual']).unsqueeze(0)
        pixel = torch.FloatTensor(sample_data['pixel']).unsqueeze(0)
        
        # Make prediction
        outputs = model(tabular, residual, pixel)
        probability = torch.sigmoid(outputs).item()
        
        # Apply threshold
        prediction = "CONFIRMED" if probability > threshold else "FALSE POSITIVE"
        confidence = probability * 100
        
        return {
            'probability': probability,
            'prediction': prediction,
            'confidence': confidence,
            'threshold': threshold
        }

def single_target_demo(kepid):
    """Demo prediction for a single target"""
    
    print(f"🎯 Making prediction for KepID {kepid}")
    print("=" * 50)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Prepare sample
    sample_data = prepare_single_sample(kepid)
    if sample_data is None:
        return
    
    # Make prediction
    result = make_prediction(model, sample_data)
    
    # Display results
    print(f"🔮 PREDICTION RESULTS:")
    print(f"   KepID: {kepid}")
    print(f"   True Label: {sample_data['true_label']}")
    print(f"   Predicted: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2f}%")
    print(f"   Raw Probability: {result['probability']:.4f}")
    print(f"   Threshold: {result['threshold']}")
    
    # Correctness
    true_confirmed = sample_data['true_label'] == 'CONFIRMED'
    pred_confirmed = result['prediction'] == 'CONFIRMED'
    correct = true_confirmed == pred_confirmed
    
    print(f"   Correct: {'✅ YES' if correct else '❌ NO'}")

def batch_prediction_demo(num_samples=10):
    """Demo batch predictions"""
    
    print(f"🚀 Batch prediction demo ({num_samples} samples)")
    print("=" * 60)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Load dataset
    df = load_kepler_features('data/raw')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    
    # Sample targets
    sample_targets = df.sample(n=min(num_samples, len(df)))
    
    results = []
    correct_predictions = 0
    
    for idx, row in sample_targets.iterrows():
        kepid = str(int(row['kepid']))
        
        # Prepare sample
        sample_data = prepare_single_sample(kepid, df)
        if sample_data is None:
            continue
        
        # Make prediction
        result = make_prediction(model, sample_data)
        
        # Check correctness
        true_confirmed = sample_data['true_label'] == 'CONFIRMED'
        pred_confirmed = result['prediction'] == 'CONFIRMED'
        correct = true_confirmed == pred_confirmed
        
        if correct:
            correct_predictions += 1
        
        results.append({
            'kepid': kepid,
            'true_label': sample_data['true_label'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'correct': correct
        })
        
        # Print result
        status = "✅" if correct else "❌"
        print(f"{status} KepID {kepid}: {result['prediction']} ({result['confidence']:.1f}%)")
    
    # Summary
    accuracy = correct_predictions / len(results) * 100
    print("\n📊 BATCH RESULTS SUMMARY:")
    print(f"   Total samples: {len(results)}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1f}%")

def interactive_demo():
    """Interactive prediction interface"""
    
    print("🎮 Interactive Prediction Mode")
    print("=" * 40)
    print("Enter KepID numbers to get predictions, or 'quit' to exit")
    
    # Load model once
    model = load_trained_model()
    if model is None:
        return
    
    while True:
        kepid = input("\n🔍 Enter KepID: ").strip()
        
        if kepid.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        try:
            kepid = str(int(kepid))  # Validate numeric
            single_target_demo(kepid)
        except ValueError:
            print("❌ Please enter a valid numeric KepID")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multimodal Exoplanet Detection - Inference Demo")
    parser.add_argument('--kepid', type=str, help='KepID for single prediction')
    parser.add_argument('--batch-predict', action='store_true', help='Run batch prediction demo')
    parser.add_argument('--interactive', action='store_true', help='Interactive prediction mode')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples for batch prediction')
    
    args = parser.parse_args()
    
    print("🌟 Enhanced Multimodal Exoplanet Detection - Inference Demo")
    print("=" * 65)
    
    if args.kepid:
        single_target_demo(args.kepid)
    elif args.batch_predict:
        batch_prediction_demo(args.num_samples)
    elif args.interactive:
        interactive_demo()
    else:
        # Default demo
        print("🚀 Running default demo...")
        print("\n1. Single target prediction:")
        single_target_demo("10797460")  # Example confirmed planet
        
        print("\n" + "="*60)
        print("2. Batch prediction demo:")
        batch_prediction_demo(5)
        
        print("\n💡 TIP: Use --help to see all available options")
        print("   Examples:")
        print("   python demo_inference.py --kepid 10797460")
        print("   python demo_inference.py --batch-predict --num-samples 20")
        print("   python demo_inference.py --interactive")

if __name__ == "__main__":
    main()