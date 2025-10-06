import os
from typing import List

# API Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# WebSocket Configuration
WEBSOCKET_PATH = "/ws"
CORS_ORIGINS = [
    "http://localhost:3000",  # Next.js development server
    "http://localhost:3001",
    "https://your-frontend-domain.com",  # Add your production domain
]

# Worker Configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", 300))  # 5 minutes
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", 30))  # 30 seconds

# Model Configuration - Using AI_Model_Forest trained models
MODEL_PATH = os.getenv("MODEL_PATH", "./AI_Model_Forest/trained_model/rf_combined_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "./AI_Model_Forest/trained_model/scaler_combined.joblib")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "./AI_Model_Forest/trained_model/feature_columns_combined.txt")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "./AI_Model_Forest/trained_model/label_encoder_combined.joblib")
IMPUTER_PATH = os.getenv("IMPUTER_PATH", "./AI_Model_Forest/trained_model/imputer_medians_combined.joblib")

# PyTorch Model Configuration
PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH", "./exoplanet-detection-pipeline/models/enhanced_multimodal_fusion_model.pth")
TABULAR_MODEL_PATH = os.getenv("TABULAR_MODEL_PATH", "./exoplanet-detection-pipeline/models/tabular_model.pth")

# Task Configuration
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 1000))
DEFAULT_TASK_TIMEOUT = int(os.getenv("DEFAULT_TASK_TIMEOUT", 120))  # 2 minutes
TASK_RETENTION_HOURS = int(os.getenv("TASK_RETENTION_HOURS", 24))  # Keep results for 24 hours

# Telemetry Configuration
TELEMETRY_UPDATE_INTERVAL = float(os.getenv("TELEMETRY_UPDATE_INTERVAL", 1.0))  # 1 second
ENABLE_DETAILED_METRICS = os.getenv("ENABLE_DETAILED_METRICS", "True").lower() == "true"

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "*").split(",")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Database Configuration (if needed for persistent storage)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./exoplanet_tasks.db")

# Monitoring Configuration
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "True").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", 9090))

# Feature Engineering Configuration
FEATURE_DEFAULTS = {
    "planet_radius_re": 1.0,
    "equilibrium_temp_k": 500.0,
    "insolation_flux_earth": 1.0,
    "stellar_teff_k": 5500.0,
    "stellar_radius_re": 1.0,
    "apparent_mag": 12.0,
    "ra": 0.0,
    "dec": 0.0
}
