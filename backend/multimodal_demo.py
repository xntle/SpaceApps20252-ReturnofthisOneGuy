#!/usr/bin/env python3
"""
Enhanced Multimodal Fusion Model API Demo
=========================================

Comprehensive demonstration of the Enhanced Multimodal Fusion model endpoints
for exoplanet detection using state-of-the-art multimodal deep learning.

Features:
- Synchronous and asynchronous predictions
- Batch processing capabilities  
- Individual component analysis
- Complete multimodal data handling (tabular + CNN1D + CNN2D)
- Performance benchmarking

Usage:
    python multimodal_demo.py

Requirements:
    - FastAPI backend running on http://localhost:8000
    - Enhanced Multimodal model loaded
    - Redis and Celery workers active
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = "http://localhost:8000"
MULTIMODAL_ENDPOINTS = {
    "sync": f"{BASE_URL}/multimodal/predict/sync",
    "async": f"{BASE_URL}/multimodal/predict",
    "batch": f"{BASE_URL}/multimodal/predict/batch",
    "analyze": f"{BASE_URL}/multimodal/analyze",
    "info": f"{BASE_URL}/multimodal/model/info",
    "health": f"{BASE_URL}/multimodal/health",
    "status": f"{BASE_URL}/status"
}

class MultimodalDemo:
    """Enhanced Multimodal Fusion API demonstration"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
    def generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample exoplanet data for testing"""
        return {
            "mission": "TESS",
            "orbital_period_days": 3.14159,
            "transit_duration_hours": 2.5,
            "transit_depth_ppm": 1200.0,
            "planet_radius_re": 1.1,
            "equilibrium_temp_k": 800.0,
            "insolation_flux_earth": 15.0,
            "stellar_teff_k": 5800.0,
            "stellar_radius_re": 1.05,
            "apparent_mag": 10.5,
            "ra": 180.0,
            "dec": 45.0
        }
    
    def generate_cnn_data(self) -> tuple:
        """Generate sample CNN data for multimodal testing"""
        # CNN1D data: 5 windows × 128 points (light curve segments)
        cnn1d_data = np.random.normal(1.0, 0.001, (5, 128)).tolist()
        
        # CNN2D data: 32 phases × 24×24 pixels (phase-folded images)
        cnn2d_data = np.random.normal(0.5, 0.1, (32, 24, 24)).tolist()
        
        return cnn1d_data, cnn2d_data
    
    def test_health_check(self) -> bool:
        """Test Enhanced Multimodal model health"""
        print("\n" + "="*60)
        print("🏥 ENHANCED MULTIMODAL HEALTH CHECK")
        print("="*60)
        
        try:
            response = self.session.get(MULTIMODAL_ENDPOINTS["health"])
            health_data = response.json()
            
            print(f"Status: {health_data.get('status', 'unknown')}")
            print(f"Model Type: {health_data.get('model_type', 'unknown')}")
            print(f"Model Loaded: {health_data.get('model_loaded', False)}")
            print(f"Device: {health_data.get('device', 'unknown')}")
            print(f"Accuracy: {health_data.get('accuracy', 'unknown')}")
            print(f"Architecture: {health_data.get('architecture', 'unknown')}")
            
            is_healthy = health_data.get('status') == 'healthy'
            print(f"✅ Health Check: {'PASSED' if is_healthy else 'FAILED'}")
            
            return is_healthy
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_model_info(self):
        """Test Enhanced Multimodal model information endpoint"""
        print("\n" + "="*60)
        print("📊 ENHANCED MULTIMODAL MODEL INFORMATION")
        print("="*60)
        
        try:
            response = self.session.get(MULTIMODAL_ENDPOINTS["info"])
            model_info = response.json()
            
            print(f"Model Type: {model_info.get('model_type', 'unknown')}")
            print(f"Device: {model_info.get('device', 'unknown')}")
            print(f"Is Loaded: {model_info.get('is_loaded', False)}")
            
            # Input features
            input_features = model_info.get('input_features', {})
            print(f"Input Features:")
            print(f"  - Tabular: {input_features.get('tabular', 'unknown')}")
            print(f"  - CNN1D: {input_features.get('cnn1d', 'unknown')}")
            print(f"  - CNN2D: {input_features.get('cnn2d', 'unknown')}")
            
            # Architecture
            architecture = model_info.get('architecture', {})
            print(f"Architecture:")
            for component, desc in architecture.items():
                print(f"  - {component}: {desc}")
            
            # Parameters
            parameters = model_info.get('parameters', {})
            print(f"Parameters:")
            print(f"  - Total: {parameters.get('total', 'unknown'):,}")
            print(f"  - Trainable: {parameters.get('trainable', 'unknown'):,}")
            
            # Performance
            performance = model_info.get('performance', {})
            print(f"Performance:")
            for metric, value in performance.items():
                print(f"  - {metric}: {value}")
            
            print(f"Feature Count: {model_info.get('feature_count', 'unknown')}")
            print(f"Model Path: {model_info.get('model_path', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Model info request failed: {e}")
    
    def test_sync_prediction(self):
        """Test Enhanced Multimodal synchronous prediction"""
        print("\n" + "="*60)
        print("⚡ ENHANCED MULTIMODAL SYNCHRONOUS PREDICTION")
        print("="*60)
        
        try:
            # Generate test data
            sample_data = self.generate_sample_data()
            cnn1d_data, cnn2d_data = self.generate_cnn_data()
            
            # Prepare request payload
            payload = {
                "request": sample_data,
                "cnn1d_data": [cnn1d_data],
                "cnn2d_data": [cnn2d_data]
            }
            
            print("Sending Enhanced Multimodal prediction request...")
            print(f"Mission: {sample_data['mission']}")
            print(f"Orbital Period: {sample_data['orbital_period_days']:.3f} days")
            print(f"Transit Depth: {sample_data['transit_depth_ppm']} ppm")
            print(f"Planet Radius: {sample_data['planet_radius_re']:.1f} R⊕")
            print(f"CNN1D Shape: {np.array(cnn1d_data).shape}")
            print(f"CNN2D Shape: {np.array(cnn2d_data).shape}")
            
            start_time = time.time()
            response = self.session.post(MULTIMODAL_ENDPOINTS["sync"], json=payload)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n🎯 ENHANCED MULTIMODAL PREDICTION RESULTS:")
                print(f"  Prediction: {'🪐 EXOPLANET' if result['prediction'] == 1 else '❌ FALSE POSITIVE'}")
                print(f"  Probability: {result['probability']:.4f}")
                print(f"  Confidence: {result['confidence_level']}")
                print(f"  Model Processing Time: {result['processing_time_ms']:.1f} ms")
                print(f"  Total Request Time: {request_time:.1f} ms")
                print(f"  Model Used: {result.get('model_used', 'unknown')}")
                
                return result
            else:
                print(f"❌ Sync prediction failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Sync prediction error: {e}")
            return None
    
    def test_async_prediction(self):
        """Test Enhanced Multimodal asynchronous prediction"""
        print("\n" + "="*60)
        print("🔄 ENHANCED MULTIMODAL ASYNCHRONOUS PREDICTION")
        print("="*60)
        
        try:
            # Generate test data
            sample_data = self.generate_sample_data()
            cnn1d_data, cnn2d_data = self.generate_cnn_data()
            
            # Prepare request payload
            payload = {
                "request": sample_data,
                "cnn1d_data": [cnn1d_data],
                "cnn2d_data": [cnn2d_data]
            }
            
            print("Submitting Enhanced Multimodal async prediction task...")
            
            # Submit async task
            response = self.session.post(MULTIMODAL_ENDPOINTS["async"], json=payload)
            
            if response.status_code == 200:
                task_response = response.json()
                task_id = task_response["task_id"]
                
                print(f"✅ Task submitted: {task_id}")
                print(f"Status: {task_response['status']}")
                print(f"Message: {task_response['message']}")
                print(f"Model Type: {task_response.get('model_type', 'unknown')}")
                
                # Poll for results
                return self._poll_task_status(task_id, "Enhanced Multimodal async")
            else:
                print(f"❌ Async prediction submission failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Async prediction error: {e}")
            return None
    
    def test_batch_prediction(self, batch_size: int = 5):
        """Test Enhanced Multimodal batch prediction"""
        print("\n" + "="*60)
        print(f"📦 ENHANCED MULTIMODAL BATCH PREDICTION ({batch_size} items)")
        print("="*60)
        
        try:
            # Generate batch test data
            batch_requests = []
            cnn1d_batch = []
            cnn2d_batch = []
            
            for i in range(batch_size):
                # Vary parameters for each test case
                sample_data = self.generate_sample_data()
                sample_data["orbital_period_days"] = 1.0 + i * 2.0  # Vary orbital periods
                sample_data["transit_depth_ppm"] = 800 + i * 400  # Vary transit depths
                
                cnn1d_data, cnn2d_data = self.generate_cnn_data()
                
                batch_requests.append(sample_data)
                cnn1d_batch.append(cnn1d_data)
                cnn2d_batch.append(cnn2d_data)
            
            # Prepare batch payload
            payload = {
                "requests": batch_requests,
                "cnn1d_batch": cnn1d_batch,
                "cnn2d_batch": cnn2d_batch
            }
            
            print(f"Submitting Enhanced Multimodal batch of {batch_size} predictions...")
            
            # Submit batch task
            response = self.session.post(MULTIMODAL_ENDPOINTS["batch"], json=payload)
            
            if response.status_code == 200:
                task_response = response.json()
                task_id = task_response["task_id"]
                batch_id = task_response.get("batch_id", "unknown")
                
                print(f"✅ Batch task submitted: {task_id}")
                print(f"Batch ID: {batch_id}")
                print(f"Total Items: {task_response.get('total_items', 'unknown')}")
                print(f"Model Type: {task_response.get('model_type', 'unknown')}")
                
                # Poll for batch results
                return self._poll_task_status(task_id, f"Enhanced Multimodal batch ({batch_size})")
            else:
                print(f"❌ Batch prediction submission failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return None
    
    def test_component_analysis(self):
        """Test Enhanced Multimodal individual component analysis"""
        print("\n" + "="*60)
        print("🔍 ENHANCED MULTIMODAL COMPONENT ANALYSIS")
        print("="*60)
        
        try:
            # Generate test data
            sample_data = self.generate_sample_data()
            cnn1d_data, cnn2d_data = self.generate_cnn_data()
            
            # Prepare request payload
            payload = {
                "request": sample_data,
                "cnn1d_data": [cnn1d_data],
                "cnn2d_data": [cnn2d_data]
            }
            
            print("Analyzing Enhanced Multimodal components...")
            
            response = self.session.post(MULTIMODAL_ENDPOINTS["analyze"], json=payload)
            
            if response.status_code == 200:
                analysis = response.json()
                
                print(f"\n🔬 COMPONENT ANALYSIS RESULTS:")
                print(f"Model Used: {analysis.get('model_used', 'unknown')}")
                print(f"Final Fusion Prediction: {analysis.get('fusion_prediction', 'unknown'):.4f}")
                
                # Feature dimensions
                dimensions = analysis.get('feature_dimensions', {})
                print(f"\nFeature Dimensions:")
                for component, dim in dimensions.items():
                    print(f"  - {component}: {dim}")
                
                # Component features (first few values)
                print(f"\nTabular Features (first 10): {analysis.get('tabular_features', [])[:10]}")
                print(f"CNN1D Features (first 10): {analysis.get('cnn1d_features', [])[:10]}")
                print(f"CNN2D Features (first 10): {analysis.get('cnn2d_features', [])[:10]}")
                
                return analysis
            else:
                print(f"❌ Component analysis failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Component analysis error: {e}")
            return None
    
    def _poll_task_status(self, task_id: str, task_description: str, max_wait: int = 60):
        """Poll task status until completion"""
        print(f"\n⏳ Polling {task_description} task status...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{MULTIMODAL_ENDPOINTS['status']}/{task_id}")
                
                if response.status_code == 200:
                    status_data = response.json()
                    task_status = status_data.get("telemetry", {}).get("status", "unknown")
                    progress = status_data.get("telemetry", {}).get("progress", 0)
                    
                    print(f"Task Status: {task_status}, Progress: {progress}%")
                    
                    if task_status in ["completed", "success"]:
                        result = status_data.get("result")
                        if result:
                            print(f"\n🎯 {task_description.upper()} RESULTS:")
                            
                            if isinstance(result, list):  # Batch results
                                print(f"Total Predictions: {len(result)}")
                                exoplanet_count = sum(1 for r in result if r.get("prediction") == 1)
                                print(f"Detected Exoplanets: {exoplanet_count}")
                                print(f"False Positives: {len(result) - exoplanet_count}")
                                
                                avg_probability = sum(r.get("probability", 0) for r in result) / len(result)
                                avg_processing_time = sum(r.get("processing_time_ms", 0) for r in result) / len(result)
                                
                                print(f"Average Probability: {avg_probability:.4f}")
                                print(f"Average Processing Time: {avg_processing_time:.1f} ms")
                                
                                # Show first few predictions
                                print(f"\nFirst 3 predictions:")
                                for i, r in enumerate(result[:3]):
                                    prediction_text = "🪐 EXOPLANET" if r.get("prediction") == 1 else "❌ FALSE POSITIVE"
                                    print(f"  {i+1}: {prediction_text} (p={r.get('probability', 0):.4f})")
                            else:  # Single result
                                prediction_text = "🪐 EXOPLANET" if result.get("prediction") == 1 else "❌ FALSE POSITIVE"
                                print(f"  Prediction: {prediction_text}")
                                print(f"  Probability: {result.get('probability', 0):.4f}")
                                print(f"  Confidence: {result.get('confidence_level', 'unknown')}")
                                print(f"  Processing Time: {result.get('processing_time_ms', 0):.1f} ms")
                                print(f"  Model Used: {result.get('model_used', 'unknown')}")
                        
                        return status_data
                    
                    elif task_status == "failed":
                        error = status_data.get("telemetry", {}).get("error_message", "Unknown error")
                        print(f"❌ Task failed: {error}")
                        return None
                    
                    # Wait before next poll
                    time.sleep(2)
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"❌ Polling error: {e}")
                return None
        
        print(f"⏰ Task polling timeout after {max_wait} seconds")
        return None
    
    def run_performance_benchmark(self):
        """Run performance benchmark comparing sync vs async"""
        print("\n" + "="*60)
        print("🏃 ENHANCED MULTIMODAL PERFORMANCE BENCHMARK")
        print("="*60)
        
        try:
            # Test synchronous prediction timing
            print("Testing synchronous prediction performance...")
            sync_times = []
            for i in range(3):
                sample_data = self.generate_sample_data()
                cnn1d_data, cnn2d_data = self.generate_cnn_data()
                
                payload = {
                    "request": sample_data,
                    "cnn1d_data": [cnn1d_data],
                    "cnn2d_data": [cnn2d_data]
                }
                
                start_time = time.time()
                response = self.session.post(MULTIMODAL_ENDPOINTS["sync"], json=payload)
                total_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    model_time = result.get("processing_time_ms", 0)
                    sync_times.append((total_time, model_time))
                    print(f"  Run {i+1}: Total={total_time:.1f}ms, Model={model_time:.1f}ms")
                
            avg_total = sum(t[0] for t in sync_times) / len(sync_times)
            avg_model = sum(t[1] for t in sync_times) / len(sync_times)
            
            print(f"\n📊 BENCHMARK RESULTS:")
            print(f"Enhanced Multimodal Sync Average:")
            print(f"  - Total Request Time: {avg_total:.1f} ms")
            print(f"  - Model Processing Time: {avg_model:.1f} ms")
            print(f"  - Network Overhead: {avg_total - avg_model:.1f} ms")
            
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
    
    def run_full_demo(self):
        """Run complete Enhanced Multimodal model demonstration"""
        print("🚀 ENHANCED MULTIMODAL FUSION MODEL API DEMO")
        print("="*80)
        print("Testing state-of-the-art multimodal exoplanet detection")
        print("Combining tabular features with CNN1D and CNN2D neural networks")
        print("="*80)
        
        # Run all tests
        print("\n🔧 Starting Enhanced Multimodal demo...")
        
        # Health check
        if not self.test_health_check():
            print("❌ Enhanced Multimodal health check failed - stopping demo")
            return
        
        # Model information
        self.test_model_info()
        
        # Sync prediction
        self.test_sync_prediction()
        
        # Component analysis
        self.test_component_analysis()
        
        # Async prediction
        self.test_async_prediction()
        
        # Batch prediction
        self.test_batch_prediction(batch_size=3)
        
        # Performance benchmark
        self.run_performance_benchmark()
        
        print("\n" + "="*80)
        print("✅ ENHANCED MULTIMODAL FUSION MODEL DEMO COMPLETED")
        print("="*80)
        print("The Enhanced Multimodal model combines:")
        print("  🧠 Tabular neural network for stellar/planetary parameters")
        print("  📈 CNN1D for light curve temporal analysis")
        print("  🖼️  CNN2D for phase-folded image analysis")
        print("  🔗 Deep fusion network for integrated predictions")
        print("")
        print("Performance: 87.2% accuracy with ~200ms inference time")
        print("Use cases: Complex multimodal analysis with attention mechanisms")
        print("="*80)

if __name__ == "__main__":
    # Run Enhanced Multimodal API demonstration
    demo = MultimodalDemo()
    demo.run_full_demo()
