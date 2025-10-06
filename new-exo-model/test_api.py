#!/usr/bin/env python3
"""
Sample script to test the exoplanet fusion API
"""

import requests
import json
import os

# API Configuration
API_BASE = "http://localhost:8001"
SAMPLE_DATA_DIR = "pixel_CNN/processed"

def test_health():
    """Test API health endpoint"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ API is healthy!")
            print(f"   Models loaded: {health['models_loaded']}")
            print(f"   Device: {health['device']}")
            print(f"   Threshold: {health['threshold']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_prediction_full():
    """Test full multi-modal prediction"""
    print("\n🚀 Testing full multi-modal prediction...")
    
    # Sample features for a typical exoplanet candidate
    features = {
        "ORB_PERIOD": 365.25,              # Earth-like orbit
        "transit_duration_hours": 6.5,     # Reasonable transit duration  
        "TRANSIT_DEPTH": 84,               # Earth-like transit depth
        "PLANET_RADIUS": 1.0,              # Earth-sized planet
        "equilibrium_temp_k": 288,         # Earth-like temperature
        "INSOL_FLUX": 1.0,                 # Earth-like insolation
        "stellar_teff_k": 5778,            # Sun-like star
        "STELLAR_RADIUS": 1.0,             # Sun-like stellar radius
        "apparent_mag": 10.5,              # Moderate brightness
        "ra": 180.0,                       # Right ascension
        "dec": 0.0,                        # Declination
        "R_PLANET_R_STAR_RATIO": 0.01,     # Earth/Sun radius ratio
        "DEPTH_PER_RADIUS": 84             # Consistent depth per radius
    }
    
    # Paths to sample data files
    residual_path = "pixel_CNN/processed/residual_windows_std/residual_10014097.npy"
    pixel_path = "pixel_CNN/processed/pixel_diffs_clean/pixdiff_10014097_clean.npy"
    
    # Check if files exist
    if not os.path.exists(residual_path):
        print(f"⚠️  Residual file not found: {residual_path}")
        residual_path = None
        
    if not os.path.exists(pixel_path):
        print(f"⚠️  Pixel file not found: {pixel_path}")
        pixel_path = None
    
    # Prepare request data
    data = {
        'kepid': '10014097',
        'features': json.dumps(features)
    }
    
    files = {}
    if residual_path:
        data['residual_window_path'] = residual_path
    if pixel_path:
        data['pixel_image_path'] = pixel_path
    
    try:
        response = requests.post(f"{API_BASE}/predict_exoplanet", data=data, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   KepID: {result.get('kepid', 'N/A')}")
            print(f"   Random Forest:  {result.get('p_rf', 'N/A'):.3f}")
            print(f"   Residual CNN:   {result.get('p_residual', 'N/A'):.3f}")
            print(f"   Pixel CNN:      {result.get('p_pixel', 'N/A'):.3f}")
            print(f"   🎯 Final Score:  {result.get('p_final', 'N/A'):.3f}")
            print(f"   🔮 Decision:     {result.get('decision', 'N/A')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_prediction_features_only():
    """Test prediction with only tabular features"""
    print("\n📊 Testing tabular-only prediction...")
    
    features = {
        "ORB_PERIOD": 12.3,
        "transit_duration_hours": 2.1,
        "TRANSIT_DEPTH": 120,
        "PLANET_RADIUS": 1.5,
        "equilibrium_temp_k": 300,
        "INSOL_FLUX": 1.2,
        "stellar_teff_k": 5800,
        "STELLAR_RADIUS": 1.1,
        "apparent_mag": 14.2,
        "ra": 285.0,
        "dec": 45.0,
        "R_PLANET_R_STAR_RATIO": 0.02,
        "DEPTH_PER_RADIUS": 80
    }
    
    data = {
        'kepid': '12345',
        'features': json.dumps(features)
    }
    
    try:
        response = requests.post(f"{API_BASE}/predict_exoplanet", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Features-only prediction successful!")
            print(f"   🎯 Final Score: {result.get('p_final', 'N/A'):.3f}")
            print(f"   🔮 Decision:    {result.get('decision', 'N/A')}")
            return True
        else:
            print(f"❌ Features-only prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Features-only prediction error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Exoplanet Fusion API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ API is not available. Please start the server:")
        print("   python -m uvicorn fusion.serve_fusion:app --host 0.0.0.0 --port 8001")
        return
    
    # Test 2: Full prediction
    test_prediction_full()
    
    # Test 3: Features only
    test_prediction_features_only()
    
    print("\n🎉 Test suite completed!")
    print("\n📚 For more examples, see USEENDPOINT.md")

if __name__ == "__main__":
    main()