from celery import Celery
import logging
from typing import List, Dict, Any
import time
import uuid
from datetime import datetime

from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from models import ExoplanetPredictionRequest, PredictionResult, TaskTelemetry, TaskStatus
from ml_service import get_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'exoplanet_worker',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,
)

@celery_app.task(bind=True, name='predict_single_exoplanet')
def predict_single_exoplanet(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for single exoplanet prediction
    """
    task_id = self.request.id
    logger.info(f"Starting prediction task {task_id}")
    
    try:
        # Update task status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 10,
                'message': 'Initializing prediction...'
            }
        )
        
        # Get predictor instance
        predictor = get_predictor()
        if not predictor.is_loaded:
            if not predictor.load_model():
                raise Exception("Failed to load ML model")
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 30,
                'message': 'Model loaded, preparing data...'
            }
        )
        
        # Create prediction request
        request = ExoplanetPredictionRequest(**prediction_data)
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 60,
                'message': 'Making prediction...'
            }
        )
        
        # Make prediction
        result = predictor.predict_single(request)
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 90,
                'message': 'Finalizing results...'
            }
        )
        
        logger.info(f"Prediction completed for task {task_id}")
        
        return {
            'status': TaskStatus.COMPLETED.value,
            'result': result.dict(),
            'task_id': task_id,
            'completed_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in prediction task {task_id}: {str(e)}")
        
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                'status': TaskStatus.FAILED.value,
                'error': str(e),
                'task_id': task_id
            }
        )
        
        raise e

@celery_app.task(bind=True, name='predict_batch_exoplanets')
def predict_batch_exoplanets(self, batch_data: List[Dict[str, Any]], batch_id: str = None) -> Dict[str, Any]:
    """
    Celery task for batch exoplanet predictions
    """
    task_id = self.request.id
    batch_id = batch_id or str(uuid.uuid4())
    total_predictions = len(batch_data)
    
    logger.info(f"Starting batch prediction task {task_id} with {total_predictions} predictions")
    
    try:
        # Update task status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 5,
                'message': f'Initializing batch prediction for {total_predictions} items...',
                'batch_id': batch_id,
                'total_items': total_predictions
            }
        )
        
        # Get predictor instance
        predictor = get_predictor()
        if not predictor.is_loaded:
            if not predictor.load_model():
                raise Exception("Failed to load ML model")
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 15,
                'message': 'Model loaded, preparing batch data...',
                'batch_id': batch_id
            }
        )
        
        # Create prediction requests
        requests = [ExoplanetPredictionRequest(**data) for data in batch_data]
        
        # Process in smaller chunks for better progress updates
        chunk_size = min(100, max(1, total_predictions // 10))
        results = []
        
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            chunk_results = predictor.predict_batch(chunk)
            results.extend(chunk_results)
            
            # Update progress
            progress = 15 + (70 * (i + len(chunk)) / total_predictions)
            self.update_state(
                state=TaskStatus.PROCESSING.value,
                meta={
                    'status': TaskStatus.PROCESSING.value,
                    'progress': int(progress),
                    'message': f'Processed {i + len(chunk)}/{total_predictions} predictions...',
                    'batch_id': batch_id,
                    'completed_items': i + len(chunk)
                }
            )
        
        # Final update
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 95,
                'message': 'Finalizing batch results...',
                'batch_id': batch_id
            }
        )
        
        logger.info(f"Batch prediction completed for task {task_id}")
        
        return {
            'status': TaskStatus.COMPLETED.value,
            'results': [result.dict() for result in results],
            'batch_id': batch_id,
            'task_id': task_id,
            'total_processed': len(results),
            'completed_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction task {task_id}: {str(e)}")
        
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                'status': TaskStatus.FAILED.value,
                'error': str(e),
                'task_id': task_id,
                'batch_id': batch_id
            }
        )
        
        raise e

@celery_app.task(name='health_check')
def health_check() -> Dict[str, Any]:
    """
    Health check task for workers
    """
    try:
        predictor = get_predictor()
        model_status = predictor.is_loaded
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_status,
            'worker_id': celery_app.current_worker_task.request.hostname if hasattr(celery_app, 'current_worker_task') else 'unknown'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

# Task routing (for multiple worker types if needed)
celery_app.conf.task_routes = {
    'predict_single_exoplanet': {'queue': 'prediction'},
    'predict_batch_exoplanets': {'queue': 'batch_prediction'},
    'health_check': {'queue': 'monitoring'}
}

# Tabular-specific tasks
@celery_app.task(bind=True, name='predict_single_tabular')
def predict_single_tabular(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for single exoplanet prediction using TabularNet only
    """
    task_id = self.request.id
    logger.info(f"Starting tabular prediction task {task_id}")
    
    try:
        # Import tabular service
        from tabular_service import get_tabular_predictor
        
        # Update task status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 0,
                'message': 'Processing tabular prediction...',
                'model_type': 'tabular_pytorch'
            }
        )
        
        # Get tabular predictor
        tabular_predictor = get_tabular_predictor()
        
        # Validate input data
        try:
            request = ExoplanetPredictionRequest(**prediction_data)
        except Exception as e:
            logger.error(f"Invalid prediction data: {e}")
            raise ValueError(f"Invalid input data: {e}")
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 50,
                'message': 'Running TabularNet inference...',
                'model_type': 'tabular_pytorch'
            }
        )
        
        # Make prediction
        start_time = time.time()
        result = tabular_predictor.predict(request)
        processing_time = time.time() - start_time
        
        # Update final status
        self.update_state(
            state=TaskStatus.COMPLETED.value,
            meta={
                'status': TaskStatus.COMPLETED.value,
                'progress': 100,
                'message': 'Tabular prediction completed successfully',
                'processing_time': processing_time,
                'model_type': 'tabular_pytorch'
            }
        )
        
        logger.info(f"Tabular prediction task {task_id} completed in {processing_time:.2f}s")
        
        return {
            'task_id': task_id,
            'result': result.dict(),
            'processing_time': processing_time,
            'status': TaskStatus.COMPLETED.value,
            'model_type': 'tabular_pytorch'
        }
        
    except Exception as e:
        error_msg = f"TabularNet prediction failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}")
        
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                'status': TaskStatus.FAILED.value,
                'error': error_msg,
                'progress': 0,
                'model_type': 'tabular_pytorch'
            }
        )
        
        raise Exception(error_msg)

@celery_app.task(bind=True, name='predict_batch_tabular')
def predict_batch_tabular(self, predictions_data: List[Dict[str, Any]], batch_id: str) -> Dict[str, Any]:
    """
    Celery task for batch exoplanet predictions using TabularNet only
    """
    task_id = self.request.id
    total_predictions = len(predictions_data)
    logger.info(f"Starting tabular batch prediction task {task_id} with {total_predictions} predictions")
    
    try:
        # Import tabular service
        from tabular_service import get_tabular_predictor
        
        # Initial status update
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 0,
                'message': f'Processing batch of {total_predictions} tabular predictions...',
                'batch_id': batch_id,
                'total_predictions': total_predictions,
                'completed_predictions': 0,
                'model_type': 'tabular_pytorch'
            }
        )
        
        # Get tabular predictor
        tabular_predictor = get_tabular_predictor()
        
        # Validate input data
        requests = []
        for i, prediction_data in enumerate(predictions_data):
            try:
                request = ExoplanetPredictionRequest(**prediction_data)
                requests.append(request)
            except Exception as e:
                logger.error(f"Invalid prediction data at index {i}: {e}")
                raise ValueError(f"Invalid input data at index {i}: {e}")
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 25,
                'message': 'Running TabularNet batch inference...',
                'batch_id': batch_id,
                'total_predictions': total_predictions,
                'completed_predictions': 0,
                'model_type': 'tabular_pytorch'
            }
        )
        
        # Make batch predictions
        start_time = time.time()
        results = tabular_predictor.predict_batch(requests)
        processing_time = time.time() - start_time
        
        # Update final status
        self.update_state(
            state=TaskStatus.COMPLETED.value,
            meta={
                'status': TaskStatus.COMPLETED.value,
                'progress': 100,
                'message': f'Batch tabular prediction completed successfully',
                'batch_id': batch_id,
                'total_predictions': total_predictions,
                'completed_predictions': total_predictions,
                'processing_time': processing_time,
                'model_type': 'tabular_pytorch'
            }
        )
        
        logger.info(f"Tabular batch prediction task {task_id} completed {total_predictions} predictions in {processing_time:.2f}s")
        
        return {
            'task_id': task_id,
            'batch_id': batch_id,
            'results': [result.dict() for result in results],
            'total_predictions': total_predictions,
            'processing_time': processing_time,
            'status': TaskStatus.COMPLETED.value,
            'model_type': 'tabular_pytorch'
        }
        
    except Exception as e:
        error_msg = f"TabularNet batch prediction failed: {str(e)}"
        logger.error(f"Batch task {task_id} failed: {error_msg}")
        
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                'status': TaskStatus.FAILED.value,
                'error': error_msg,
                'batch_id': batch_id,
                'progress': 0,
                'model_type': 'tabular_pytorch'
            }
        )
        
        raise Exception(error_msg)

@celery_app.task(bind=True, name='predict_single_multimodal')
def predict_single_multimodal(self, prediction_data: Dict[str, Any], 
                               cnn1d_data: List[float] = None,
                               cnn2d_data: List[List[List[float]]] = None) -> Dict[str, Any]:
    """
    Celery task for single Enhanced Multimodal prediction
    """
    task_id = self.request.id
    logger.info(f"Starting multimodal prediction task {task_id}")
    
    try:
        # Update task status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 25,
                'message': 'Processing multimodal prediction...',
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        # Get multimodal predictor
        from multimodal_service import get_multimodal_predictor
        predictor = get_multimodal_predictor()
        
        if not predictor.is_loaded:
            if not predictor.load_model():
                raise Exception("Failed to load Enhanced Multimodal model")
        
        # Update status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 50,
                'message': 'Running Enhanced Multimodal inference...',
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        # Create prediction request
        request = ExoplanetPredictionRequest(**prediction_data)
        
        # Convert CNN data if provided
        import numpy as np
        cnn1d_array = np.array(cnn1d_data) if cnn1d_data else None
        cnn2d_array = np.array(cnn2d_data) if cnn2d_data else None
        
        # Make prediction
        result = predictor.predict(request, cnn1d_array, cnn2d_array)
        
        # Update final status
        self.update_state(
            state=TaskStatus.COMPLETED.value,
            meta={
                'status': TaskStatus.COMPLETED.value,
                'progress': 100,
                'message': 'Enhanced Multimodal prediction completed',
                'prediction': result.prediction,
                'probability': result.probability,
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        logger.info(f"Multimodal prediction task {task_id} completed: {result.prediction} (p={result.probability:.3f})")
        
        return {
            'task_id': task_id,
            'result': result.dict(),
            'status': TaskStatus.COMPLETED.value,
            'model_type': 'enhanced_multimodal_fusion'
        }
        
    except Exception as e:
        error_msg = f"Enhanced Multimodal prediction failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}")
        
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                'status': TaskStatus.FAILED.value,
                'error': error_msg,
                'progress': 0,
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        raise Exception(error_msg)

@celery_app.task(bind=True, name='predict_batch_multimodal')
def predict_batch_multimodal(self, batch_data: List[Dict[str, Any]], 
                             batch_cnn1d_data: List[List[float]] = None,
                             batch_cnn2d_data: List[List[List[List[float]]]] = None,
                             batch_id: str = None) -> Dict[str, Any]:
    """
    Celery task for batch Enhanced Multimodal predictions
    """
    task_id = self.request.id
    batch_id = batch_id or str(uuid.uuid4())
    total_predictions = len(batch_data)
    
    logger.info(f"Starting multimodal batch prediction task {task_id} with {total_predictions} predictions")
    
    try:
        # Update task status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 10,
                'message': f'Initializing multimodal batch prediction for {total_predictions} items...',
                'batch_id': batch_id,
                'total_items': total_predictions,
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        # Get multimodal predictor
        from multimodal_service import get_multimodal_predictor
        predictor = get_multimodal_predictor()
        
        if not predictor.is_loaded:
            if not predictor.load_model():
                raise Exception("Failed to load Enhanced Multimodal model")
        
        # Create prediction requests
        requests = [ExoplanetPredictionRequest(**data) for data in batch_data]
        
        # Convert CNN data if provided
        import numpy as np
        cnn1d_arrays = None
        cnn2d_arrays = None
        
        if batch_cnn1d_data and len(batch_cnn1d_data) == total_predictions:
            cnn1d_arrays = [np.array(data) for data in batch_cnn1d_data]
        
        if batch_cnn2d_data and len(batch_cnn2d_data) == total_predictions:
            cnn2d_arrays = [np.array(data) for data in batch_cnn2d_data]
        
        # Update status
        self.update_state(
            state=TaskStatus.PROCESSING.value,
            meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': 30,
                'message': f'Processing {total_predictions} multimodal predictions...',
                'batch_id': batch_id,
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        start_time = time.time()
        
        # Make batch predictions
        results = predictor.predict_batch(requests, cnn1d_arrays, cnn2d_arrays)
        
        processing_time = time.time() - start_time
        
        # Update final status
        self.update_state(
            state=TaskStatus.COMPLETED.value,
            meta={
                'status': TaskStatus.COMPLETED.value,
                'progress': 100,
                'message': f'Multimodal batch prediction completed successfully',
                'batch_id': batch_id,
                'total_predictions': total_predictions,
                'completed_predictions': total_predictions,
                'processing_time': processing_time,
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        logger.info(f"Multimodal batch prediction task {task_id} completed {total_predictions} predictions in {processing_time:.2f}s")
        
        return {
            'task_id': task_id,
            'batch_id': batch_id,
            'results': [result.dict() for result in results],
            'total_predictions': total_predictions,
            'processing_time': processing_time,
            'status': TaskStatus.COMPLETED.value,
            'model_type': 'enhanced_multimodal_fusion'
        }
        
    except Exception as e:
        error_msg = f"Enhanced Multimodal batch prediction failed: {str(e)}"
        logger.error(f"Batch task {task_id} failed: {error_msg}")
        
        self.update_state(
            state=TaskStatus.FAILED.value,
            meta={
                'status': TaskStatus.FAILED.value,
                'error': error_msg,
                'batch_id': batch_id,
                'progress': 0,
                'model_type': 'enhanced_multimodal_fusion'
            }
        )
        
        raise Exception(error_msg)

if __name__ == '__main__':
    # Start worker with: celery -A worker worker --loglevel=info
    celery_app.start()
