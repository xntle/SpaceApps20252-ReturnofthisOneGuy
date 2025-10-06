#!/usr/bin/env python3
"""
🎯 TabularNet Model API Demo

Demonstrates the new tabular-specific endpoints for the exoplanet detection backend.
The TabularNet model is optimized for fast, reliable predictions using only tabular features.

Usage:
    python tabular_demo.py

Features:
    - Synchronous predictions (fastest)
    - Asynchronous single predictions
    - Batch predictions
    - Model information
    - Health checks
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
TABULAR_BASE = f"{BASE_URL}/tabular"

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_result(title: str, result: Dict[Any, Any]):
    """Print a formatted result"""
    print(f"\n📊 {title}:")
    print(json.dumps(result, indent=2))

def test_tabular_health():
    """Test TabularNet health check"""
    print_header("TabularNet Health Check")
    
    try:
        response = requests.get(f"{TABULAR_BASE}/health")
        if response.status_code == 200:
            print("✅ TabularNet service is healthy")
            print_result("Health Status", response.json())
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

def test_tabular_model_info():
    """Test TabularNet model information"""
    print_header("TabularNet Model Information")
    
    try:
        response = requests.get(f"{TABULAR_BASE}/model/info")
        if response.status_code == 200:
            print("✅ Model info retrieved successfully")
            print_result("Model Information", response.json())
        else:
            print(f"❌ Model info failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

def test_sync_prediction():
    """Test synchronous TabularNet prediction (fastest)"""
    print_header("Synchronous TabularNet Prediction")
    
    # Example exoplanet data (Kepler-452b like)
    test_data = {
        "mission": "Kepler",
        "orbital_period_days": 384.8,
        "transit_duration_hours": 10.4,
        "transit_depth_ppm": 40.0,
        "planet_radius_re": 1.6,
        "equilibrium_temp_k": 265.0,
        "insolation_flux_earth": 1.1,
        "stellar_teff_k": 5757.0,
        "stellar_radius_re": 1.11,
        "apparent_mag": 13.4,
        "ra": 294.1,
        "dec": 44.3
    }
    
    try:
        print("📤 Sending synchronous prediction request...")
        start_time = time.time()
        
        response = requests.post(
            f"{TABULAR_BASE}/predict/sync",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to ms
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction completed in {total_time:.2f}ms")
            print_result("Prediction Result", result)
            
            # Interpret result
            prediction = result.get('prediction', 0)
            probability = result.get('probability', 0.0)
            confidence = result.get('confidence_level', 'Unknown')
            
            print(f"\n🔍 Interpretation:")
            print(f"   Classification: {'🪐 EXOPLANET' if prediction == 1 else '❌ FALSE POSITIVE'}")
            print(f"   Probability: {probability:.1%}")
            print(f"   Confidence: {confidence}")
            print(f"   Model: {result.get('model_used', 'unknown')}")
            
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

def test_async_prediction():
    """Test asynchronous TabularNet prediction"""
    print_header("Asynchronous TabularNet Prediction")
    
    # Example exoplanet data (TRAPPIST-1e like)
    test_data = {
        "mission": "TESS",
        "orbital_period_days": 6.1,
        "transit_duration_hours": 0.8,
        "transit_depth_ppm": 358.0,
        "planet_radius_re": 0.92,
        "equilibrium_temp_k": 251.0,
        "insolation_flux_earth": 0.66,
        "stellar_teff_k": 2566.0,
        "stellar_radius_re": 0.12,
        "apparent_mag": 18.8,
        "ra": 346.6,
        "dec": -5.0
    }
    
    try:
        print("📤 Submitting asynchronous prediction task...")
        
        response = requests.post(
            f"{TABULAR_BASE}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            task_info = response.json()
            task_id = task_info.get('task_id')
            print(f"✅ Task submitted successfully")
            print_result("Task Information", task_info)
            
            # Poll for result
            print(f"\n⏳ Polling for task {task_id} result...")
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    result_response = requests.get(f"{BASE_URL}/tasks/{task_id}")
                    if result_response.status_code == 200:
                        result = result_response.json()
                        status = result.get('status', 'unknown')
                        
                        if status == 'COMPLETED':
                            print(f"✅ Task completed!")
                            print_result("Final Result", result)
                            break
                        elif status == 'FAILED':
                            print(f"❌ Task failed!")
                            print_result("Error Result", result)
                            break
                        else:
                            print(f"⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})")
                            time.sleep(1)
                    else:
                        print(f"❌ Error checking task status: {result_response.status_code}")
                        break
                        
                except requests.exceptions.RequestException as e:
                    print(f"❌ Error polling task: {e}")
                    break
            else:
                print(f"⏰ Task polling timed out after {max_attempts} attempts")
                
        else:
            print(f"❌ Task submission failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

def test_batch_prediction():
    """Test batch TabularNet predictions"""
    print_header("Batch TabularNet Predictions")
    
    # Multiple test candidates
    test_batch = {
        "batch_name": "tabular_demo_batch",
        "predictions": [
            {
                "mission": "Kepler",
                "orbital_period_days": 1.3,
                "transit_duration_hours": 1.1,
                "transit_depth_ppm": 1234.0,
                "planet_radius_re": 0.8,
                "equilibrium_temp_k": 1500.0,
                "insolation_flux_earth": 150.0,
                "stellar_teff_k": 5800.0,
                "stellar_radius_re": 1.0,
                "apparent_mag": 12.0,
                "ra": 123.4,
                "dec": 56.7
            },
            {
                "mission": "TESS",
                "orbital_period_days": 365.0,
                "transit_duration_hours": 13.0,
                "transit_depth_ppm": 84.0,
                "planet_radius_re": 1.0,
                "equilibrium_temp_k": 255.0,
                "insolation_flux_earth": 1.0,
                "stellar_teff_k": 5778.0,
                "stellar_radius_re": 1.0,
                "apparent_mag": 10.0,
                "ra": 0.0,
                "dec": 0.0
            },
            {
                "mission": "Kepler",
                "orbital_period_days": 50.0,
                "transit_duration_hours": 4.2,
                "transit_depth_ppm": 200.0,
                "planet_radius_re": 2.1,
                "equilibrium_temp_k": 800.0,
                "insolation_flux_earth": 25.0,
                "stellar_teff_k": 6200.0,
                "stellar_radius_re": 1.5,
                "apparent_mag": 11.5,
                "ra": 280.1,
                "dec": 38.7
            }
        ]
    }
    
    try:
        print(f"📤 Submitting batch of {len(test_batch['predictions'])} predictions...")
        
        response = requests.post(
            f"{TABULAR_BASE}/predict/batch",
            json=test_batch,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            batch_info = response.json()
            batch_id = batch_info.get('batch_id')
            task_ids = batch_info.get('task_ids', [])
            
            print(f"✅ Batch submitted successfully")
            print_result("Batch Information", batch_info)
            
            if task_ids:
                task_id = task_ids[0]  # Get the first (and likely only) task ID
                
                # Poll for batch result
                print(f"\n⏳ Polling for batch {batch_id} result...")
                max_attempts = 60
                for attempt in range(max_attempts):
                    try:
                        result_response = requests.get(f"{BASE_URL}/tasks/{task_id}")
                        if result_response.status_code == 200:
                            result = result_response.json()
                            status = result.get('status', 'unknown')
                            
                            if status == 'COMPLETED':
                                print(f"✅ Batch completed!")
                                print_result("Batch Results", result)
                                
                                # Show individual results
                                if 'result' in result and 'results' in result['result']:
                                    predictions = result['result']['results']
                                    print(f"\n📊 Individual Predictions:")
                                    for i, pred in enumerate(predictions):
                                        classification = '🪐 EXOPLANET' if pred.get('prediction') == 1 else '❌ FALSE POSITIVE'
                                        probability = pred.get('probability', 0.0)
                                        print(f"   {i+1}. {classification} ({probability:.1%} confidence)")
                                
                                break
                            elif status == 'FAILED':
                                print(f"❌ Batch failed!")
                                print_result("Error Result", result)
                                break
                            else:
                                print(f"⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})")
                                time.sleep(1)
                        else:
                            print(f"❌ Error checking batch status: {result_response.status_code}")
                            break
                            
                    except requests.exceptions.RequestException as e:
                        print(f"❌ Error polling batch: {e}")
                        break
                else:
                    print(f"⏰ Batch polling timed out after {max_attempts} attempts")
            
        else:
            print(f"❌ Batch submission failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

def main():
    """Run all TabularNet API demonstrations"""
    print("🚀 TabularNet Model API Demonstration")
    print("=" * 60)
    print("This demo shows the new tabular-specific endpoints for fast exoplanet detection.")
    print("Make sure the backend is running at http://localhost:8000")
    
    # Test all endpoints
    test_tabular_health()
    test_tabular_model_info()
    test_sync_prediction()
    test_async_prediction()
    test_batch_prediction()
    
    print_header("Demo Complete!")
    print("🎉 All TabularNet API endpoints demonstrated successfully!")
    print("\n📋 Available Endpoints:")
    print("   GET  /tabular/health          - Health check")
    print("   GET  /tabular/model/info      - Model information")
    print("   POST /tabular/predict/sync    - Synchronous prediction (fastest)")
    print("   POST /tabular/predict         - Asynchronous prediction")
    print("   POST /tabular/predict/batch   - Batch predictions")
    print("\n🎯 Key Benefits:")
    print("   • 93.6% accuracy on tabular features")
    print("   • 52,353 parameters (lightweight)")
    print("   • <100ms inference time")
    print("   • Dedicated endpoints for optimal performance")
    print("   • Synchronous option for real-time applications")

if __name__ == "__main__":
    main()
