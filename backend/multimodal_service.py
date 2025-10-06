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
sys.path.append(os.path.join(os.path.dirname(__file__), '../exoplanet-detection-pipeline'))
from train_multimodal_enhanced import EnhancedMultiModalFusionModel

from models import ExoplanetPredictionRequest, PredictionResult
from config import PYTORCH_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH, IMPUTER_PATH

logger = logging.getLogger(__name__)

class MultimodalExoplanetPredictor:
    """
    Dedicated Enhanced Multimodal Fusion model service using PyTorch
    Combines tabular, CNN1D, and CNN2D inputs for state-of-the-art predictions
    """
    
    def __init__(self):
        self.model: EnhancedMultiModalFusionModel = None
        self.scaler: StandardScaler = None
        self.imputer = None
        self.feature_columns: List[str] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the Enhanced Multimodal Fusion model and preprocessing components"""
        try:
            # Load Enhanced Multimodal model
            if Path(PYTORCH_MODEL_PATH).exists():
                self.model = EnhancedMultiModalFusionModel(n_tabular_features=39)
                self.model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Enhanced Multimodal model loaded from {PYTORCH_MODEL_PATH}")
            else:
                logger.error(f"Enhanced Multimodal model file not found: {PYTORCH_MODEL_PATH}")
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
                # Default feature columns for multimodal model
                self.feature_columns = [
                    'orbital_period_days', 'transit_duration_hours', 'transit_depth_ppm',
                    'planet_radius_re', 'equilibrium_temp_k', 'insolation_flux_earth',
                    'stellar_teff_k', 'stellar_radius_re', 'apparent_mag', 'ra', 'dec',
                    'mission_encoded'
                ]
                logger.warning("Using default feature columns for multimodal model")
            
            self.is_loaded = True
            logger.info(f"Enhanced Multimodal predictor loaded successfully - Device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Enhanced Multimodal model: {e}")
            return False
    
    def preprocess_data(self, request: ExoplanetPredictionRequest) -> np.ndarray:
        """Preprocess input data for Enhanced Multimodal prediction"""
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
    
    def predict(self, request: ExoplanetPredictionRequest, 
                cnn1d_data: Optional[np.ndarray] = None,
                cnn2d_data: Optional[np.ndarray] = None) -> PredictionResult:
        """Make a single prediction using Enhanced Multimodal Fusion"""
        if not self.is_loaded:
            raise RuntimeError("Enhanced Multimodal model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess tabular input
            tabular_features = self.preprocess_data(request)
            
            # Convert to tensors
            tabular_tensor = torch.FloatTensor(tabular_features).to(self.device)
            
            # Handle CNN data (use zero-filled if not provided)
            if cnn1d_data is not None:
                cnn1d_tensor = torch.FloatTensor(cnn1d_data).unsqueeze(0).to(self.device)
            else:
                # Zero-filled CNN1D data (5 windows, 128 points each)
                cnn1d_tensor = torch.zeros(1, 5, 128, device=self.device)
            
            if cnn2d_data is not None:
                cnn2d_tensor = torch.FloatTensor(cnn2d_data).unsqueeze(0).to(self.device)
            else:
                # Zero-filled CNN2D data (32 phases, 24x24 pixels)
                cnn2d_tensor = torch.zeros(1, 32, 24, 24, device=self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(tabular_tensor, cnn1d_tensor, cnn2d_tensor)
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
                model_used="enhanced_multimodal_fusion"
            )
            
        except Exception as e:
            logger.error(f"Error making Enhanced Multimodal prediction: {e}")
            raise
    
    def predict_batch(self, requests: List[ExoplanetPredictionRequest],
                      cnn1d_data_batch: Optional[List[np.ndarray]] = None,
                      cnn2d_data_batch: Optional[List[np.ndarray]] = None) -> List[PredictionResult]:
        """Make batch predictions using Enhanced Multimodal Fusion"""
        if not self.is_loaded:
            raise RuntimeError("Enhanced Multimodal model not loaded. Call load_model() first.")
        
        results = []
        batch_size = len(requests)
        
        try:
            # Preprocess all tabular requests
            features_list = []
            for request in requests:
                features = self.preprocess_data(request)
                features_list.append(features[0])  # Remove the reshape dimension
            
            # Convert to numpy array then tensor
            all_features = np.array(features_list)
            tabular_tensor = torch.FloatTensor(all_features).to(self.device)
            
            # Handle CNN data batches
            if cnn1d_data_batch is not None and len(cnn1d_data_batch) == batch_size:
                cnn1d_batch = np.array(cnn1d_data_batch)
                cnn1d_tensor = torch.FloatTensor(cnn1d_batch).to(self.device)
            else:
                # Zero-filled CNN1D data for batch
                cnn1d_tensor = torch.zeros(batch_size, 5, 128, device=self.device)
            
            if cnn2d_data_batch is not None and len(cnn2d_data_batch) == batch_size:
                cnn2d_batch = np.array(cnn2d_data_batch)
                cnn2d_tensor = torch.FloatTensor(cnn2d_batch).to(self.device)
            else:
                # Zero-filled CNN2D data for batch
                cnn2d_tensor = torch.zeros(batch_size, 32, 24, 24, device=self.device)
            
            start_time = time.time()
            
            # Make batch predictions
            with torch.no_grad():
                predictions = self.model(tabular_tensor, cnn1d_tensor, cnn2d_tensor)
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
                    processing_time_ms=processing_time / batch_size,  # Average time per prediction
                    model_used="enhanced_multimodal_fusion"
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error making Enhanced Multimodal batch predictions: {e}")
            raise
    
    def get_individual_predictions(self, request: ExoplanetPredictionRequest,
                                   cnn1d_data: Optional[np.ndarray] = None,
                                   cnn2d_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get predictions from individual model components for analysis"""
        if not self.is_loaded:
            raise RuntimeError("Enhanced Multimodal model not loaded. Call load_model() first.")
        
        try:
            # Preprocess tabular input
            tabular_features = self.preprocess_data(request)
            tabular_tensor = torch.FloatTensor(tabular_features).to(self.device)
            
            # Handle CNN data
            if cnn1d_data is not None:
                cnn1d_tensor = torch.FloatTensor(cnn1d_data).unsqueeze(0).to(self.device)
            else:
                cnn1d_tensor = torch.zeros(1, 5, 128, device=self.device)
            
            if cnn2d_data is not None:
                cnn2d_tensor = torch.FloatTensor(cnn2d_data).unsqueeze(0).to(self.device)
            else:
                cnn2d_tensor = torch.zeros(1, 32, 24, 24, device=self.device)
            
            # Get individual component outputs
            individual_outputs = self.model.get_individual_predictions(
                tabular_tensor, cnn1d_tensor, cnn2d_tensor
            )
            
            # Get final fusion prediction
            with torch.no_grad():
                fusion_prediction = self.model(tabular_tensor, cnn1d_tensor, cnn2d_tensor)
                fusion_probability = float(fusion_prediction.cpu().numpy()[0])
            
            return {
                "tabular_features": individual_outputs["tabular_features"].tolist(),
                "cnn1d_features": individual_outputs["cnn1d_features"].tolist(),
                "cnn2d_features": individual_outputs["cnn2d_features"].tolist(),
                "fusion_prediction": fusion_probability,
                "model_used": "enhanced_multimodal_fusion",
                "feature_dimensions": {
                    "tabular": individual_outputs["tabular_features"].shape,
                    "cnn1d": individual_outputs["cnn1d_features"].shape,
                    "cnn2d": individual_outputs["cnn2d_features"].shape
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting individual predictions: {e}")
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
        """Get information about the Enhanced Multimodal model"""
        if not self.is_loaded:
            return {"error": "Enhanced Multimodal model not loaded"}
        
        info = {
            "model_type": "Enhanced Multimodal Fusion (PyTorch)",
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "input_features": {
                "tabular": 39,
                "cnn1d": "5 windows x 128 points",
                "cnn2d": "32 phases x 24x24 pixels"
            },
            "architecture": {
                "tabular_net": "Multi-layer perceptron",
                "cnn1d": "Residual CNN with attention",
                "cnn2d": "Pixel CNN with conv blocks",
                "fusion": "Deep fusion network (129→128→64→32→16→1)"
            },
            "parameters": {
                "total": sum(p.numel() for p in self.model.parameters()),
                "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            "performance": {
                "accuracy": "87.2%",
                "test_set": "Kepler + TESS combined",
                "strengths": "Complex multimodal analysis, attention mechanisms"
            },
            "feature_count": len(self.feature_columns),
            "model_path": str(PYTORCH_MODEL_PATH)
        }
        
        return info

# Global multimodal predictor instance
multimodal_predictor = MultimodalExoplanetPredictor()

def initialize_multimodal_predictor() -> bool:
    """Initialize the global multimodal predictor instance"""
    return multimodal_predictor.load_model()

def get_multimodal_predictor() -> MultimodalExoplanetPredictor:
    """Get the global multimodal predictor instance"""
    return multimodal_predictor
