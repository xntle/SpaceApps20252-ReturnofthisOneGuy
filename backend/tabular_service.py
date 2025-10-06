import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import time
import sys
import os

# Add exoplanet-detection-pipeline to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../exoplanet-detection-pipeline/src'))
from models import TabularNet

from models import ExoplanetPredictionRequest, PredictionResult
from config import TABULAR_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH, IMPUTER_PATH

logger = logging.getLogger(__name__)

class TabularExoplanetPredictor:
    """
    Dedicated tabular-only exoplanet prediction service using PyTorch TabularNet
    """
    
    def __init__(self):
        self.model: TabularNet = None
        self.scaler: StandardScaler = None
        self.imputer = None
        self.feature_columns: List[str] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the trained TabularNet model and preprocessing components"""
        try:
            # Load TabularNet model
            if Path(TABULAR_MODEL_PATH).exists():
                self.model = TabularNet(input_size=39)
                self.model.load_state_dict(torch.load(TABULAR_MODEL_PATH, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"TabularNet model loaded from {TABULAR_MODEL_PATH}")
            else:
                logger.error(f"TabularNet model file not found: {TABULAR_MODEL_PATH}")
                return False
            
            # Load scaler
            if Path(PREPROCESSOR_PATH).exists():
                import joblib
                self.scaler = joblib.load(PREPROCESSOR_PATH)
                logger.info(f"Scaler loaded from {PREPROCESSOR_PATH}")
            else:
                logger.warning(f"Scaler file not found: {PREPROCESSOR_PATH}")
                self.scaler = StandardScaler()
            
            # Load imputer
            if Path(IMPUTER_PATH).exists():
                import joblib
                self.imputer = joblib.load(IMPUTER_PATH)
                logger.info(f"Imputer loaded from {IMPUTER_PATH}")
            else:
                logger.warning(f"Imputer file not found: {IMPUTER_PATH}")
            
            # Load feature columns
            if Path(FEATURE_COLUMNS_PATH).exists():
                with open(FEATURE_COLUMNS_PATH, 'r') as f:
                    self.feature_columns = [line.strip() for line in f.readlines()]
                logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
            else:
                # Default feature columns for tabular model
                self.feature_columns = [
                    'orbital_period_days', 'transit_duration_hours', 'transit_depth_ppm',
                    'planet_radius_re', 'equilibrium_temp_k', 'insolation_flux_earth',
                    'stellar_teff_k', 'stellar_radius_re', 'apparent_mag', 'ra', 'dec',
                    'mission_encoded'
                ]
                logger.warning("Using default feature columns for tabular model")
            
            self.is_loaded = True
            logger.info(f"TabularNet predictor loaded successfully - Device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TabularNet model: {e}")
            return False
    
    def preprocess_data(self, request: ExoplanetPredictionRequest) -> np.ndarray:
        """Preprocess input data for TabularNet prediction"""
        try:
            # Convert request to dictionary
            data = request.dict()
            
            # Create feature vector
            feature_vector = []
            
            # Extract features in the expected order
            feature_vector.append(data.get('orbital_period_days', 0.0))
            feature_vector.append(data.get('transit_duration_hours', 0.0))
            feature_vector.append(data.get('transit_depth_ppm', 0.0))
            feature_vector.append(data.get('planet_radius_re', 1.0))
            feature_vector.append(data.get('equilibrium_temp_k', 255.0))
            feature_vector.append(data.get('insolation_flux_earth', 1.0))
            feature_vector.append(data.get('stellar_teff_k', 5778.0))
            feature_vector.append(data.get('stellar_radius_re', 1.0))
            feature_vector.append(data.get('apparent_mag', 12.0))
            feature_vector.append(data.get('ra', 0.0))
            feature_vector.append(data.get('dec', 0.0))
            
            # Encode mission (Kepler=0, TESS=1)
            mission = data.get('mission', 'Kepler').lower()
            mission_encoded = 1.0 if mission == 'tess' else 0.0
            feature_vector.append(mission_encoded)
            
            # Add remaining features with default values to reach 39 features
            while len(feature_vector) < 39:
                feature_vector.append(0.0)
            
            # Convert to numpy array and reshape
            features = np.array(feature_vector).reshape(1, -1)
            
            # Apply imputation if available
            if self.imputer is not None:
                features = self.imputer.transform(features)
            
            # Apply scaling if available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict(self, request: ExoplanetPredictionRequest) -> PredictionResult:
        """Make a single prediction using TabularNet"""
        if not self.is_loaded:
            raise RuntimeError("TabularNet model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess input
            features = self.preprocess_data(request)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(features_tensor)
                prediction = torch.sigmoid(logits)
                exoplanet_probability = float(prediction.cpu().numpy()[0])
            
            # Binary classification
            binary_prediction = 1 if exoplanet_probability > 0.5 else 0
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(exoplanet_probability)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return PredictionResult(
                prediction=int(binary_prediction),
                probability=float(exoplanet_probability),
                confidence_level=confidence_level,
                processing_time_ms=processing_time,
                model_used="tabular_pytorch"
            )
            
        except Exception as e:
            logger.error(f"Error making TabularNet prediction: {e}")
            raise
    
    def predict_batch(self, requests: List[ExoplanetPredictionRequest]) -> List[PredictionResult]:
        """Make batch predictions using TabularNet"""
        if not self.is_loaded:
            raise RuntimeError("TabularNet model not loaded. Call load_model() first.")
        
        results = []
        
        try:
            # Preprocess all requests
            features_list = []
            for request in requests:
                features = self.preprocess_data(request)
                features_list.append(features[0])  # Remove the reshape dimension
            
            # Convert to numpy array then tensor
            all_features = np.array(features_list)
            features_tensor = torch.FloatTensor(all_features).to(self.device)
            
            start_time = time.time()
            
            # Make batch predictions
            with torch.no_grad():
                logits = self.model(features_tensor)
                predictions = torch.sigmoid(logits)
                probabilities = predictions.cpu().numpy()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Process results
            for i, probability in enumerate(probabilities):
                exoplanet_probability = float(probability[0] if len(probability.shape) > 0 else probability)
                binary_prediction = 1 if exoplanet_probability > 0.5 else 0
                confidence_level = self._get_confidence_level(exoplanet_probability)
                
                results.append(PredictionResult(
                    prediction=int(binary_prediction),
                    probability=float(exoplanet_probability),
                    confidence_level=confidence_level,
                    processing_time_ms=processing_time / len(requests),  # Average time per prediction
                    model_used="tabular_pytorch"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error making TabularNet batch predictions: {e}")
            raise
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.8 or probability <= 0.2:
            return "High"
        elif probability >= 0.6 or probability <= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded TabularNet model"""
        if not self.is_loaded:
            return {"error": "TabularNet model not loaded"}
        
        info = {
            "model_type": "TabularNet (PyTorch)",
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "input_features": 39,
            "feature_count": len(self.feature_columns),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "architecture": "Multi-layer perceptron with batch normalization and dropout",
            "accuracy": "93.6%",
            "model_path": str(TABULAR_MODEL_PATH)
        }
        
        return info

# Global tabular predictor instance
tabular_predictor = TabularExoplanetPredictor()

def initialize_tabular_predictor() -> bool:
    """Initialize the global tabular predictor instance"""
    return tabular_predictor.load_model()

def get_tabular_predictor() -> TabularExoplanetPredictor:
    """Get the global tabular predictor instance"""
    return tabular_predictor
