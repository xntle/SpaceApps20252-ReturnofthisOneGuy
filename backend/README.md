# Exoplanet Detection Backend

A distributed microservices backend for exoplanet detection using Random Forest ML models with real-time telemetry and scalable worker architecture.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│   FastAPI       │────│   Celery        │
│   (Next.js)     │    │   Backend       │    │   Workers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │   ML Models     │
                       │  (Queue/Cache)  │    │ (Random Forest) │
                       └─────────────────┘    └─────────────────┘
```

## Features

- **Distributed Processing**: Celery workers for scalable ML inference
- **Real-time Telemetry**: WebSocket connections for live progress updates  
- **Task Orchestration**: Queue management with Redis for multiple workers
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Health Monitoring**: Built-in health checks and worker status monitoring
- **Containerized**: Docker support for easy deployment

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to project
cd SpaceApps2025-ReturnofthisOneGuy

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

Services will be available at:
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs  
- **WebSocket**: ws://localhost:8000/ws/{client_id}
- **Worker Monitoring**: http://localhost:5555 (Flower)

### Option 2: Local Development

```bash
# Navigate to backend directory
cd backend

# Run setup script
./start.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Redis (separate terminal)
redis-server

# Start FastAPI backend (separate terminal)
source venv/bin/activate
python main.py

# Start Celery worker (separate terminal)
source venv/bin/activate
celery -A worker.celery_app worker --loglevel=info
```

## API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check with system metrics
- `GET /docs` - Interactive API documentation

### Prediction Endpoints
- `POST /predict` - Submit single prediction task
- `POST /predict/batch` - Submit batch prediction tasks
- `GET /tasks/{task_id}` - Get task status and results

### Monitoring Endpoints  
- `GET /workers/status` - Worker status and capacity
- `GET /model/info` - ML model information
- `WebSocket /ws/{client_id}` - Real-time telemetry

## WebSocket Protocol

Connect to `/ws/{client_id}` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client123');

// Subscribe to task updates
ws.send(JSON.stringify({
  type: 'subscribe_task',
  task_id: 'task-uuid-here'
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Task update:', data);
};
```

## Request Examples

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "period": 1.5,
    "radius": 1.2,
    "period_err": 0.01,
    "radius_err": 0.05,
    "source": "KEPLER"
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "period": 1.5,
        "radius": 1.2,
        "period_err": 0.01,
        "radius_err": 0.05,
        "source": "KEPLER"
      },
      {
        "period": 2.1,
        "radius": 0.8,
        "period_err": 0.02,
        "radius_err": 0.03,
        "source": "TESS"
      }
    ]
  }'
```

## Configuration

Environment variables (create `.env` file):

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://localhost:3001"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Task Configuration
MAX_BATCH_SIZE=100
TASK_TIMEOUT=300
TELEMETRY_UPDATE_INTERVAL=2
```

## Scaling Workers

### Docker Scaling
```bash
# Scale to 4 workers
docker-compose up -d --scale celery_worker=4

# Check worker status
docker-compose exec backend celery -A worker.celery_app inspect active
```

### Manual Worker Scaling
```bash
# Start additional workers on different machines
celery -A worker.celery_app worker --hostname=worker2@%h
celery -A worker.celery_app worker --hostname=worker3@%h
```

## Development

### Project Structure
```
backend/
├── main.py              # FastAPI application
├── models.py            # Pydantic data models
├── config.py            # Configuration settings
├── ml_service.py        # ML model service
├── worker.py            # Celery worker tasks
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
├── start.sh            # Setup script
└── venv/               # Virtual environment
```

### Adding New Endpoints
1. Define data models in `models.py`
2. Add business logic to appropriate service file
3. Create endpoint in `main.py`
4. Add worker tasks in `worker.py` if needed

### Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Test WebSocket connection
python -c "
import asyncio
import websockets

async def test():
    uri = 'ws://localhost:8000/ws/test'
    async with websockets.connect(uri) as ws:
        await ws.send('{\"type\": \"ping\"}')
        response = await ws.recv()
        print(f'Response: {response}')

asyncio.run(test())
"
```

## Production Deployment

### Docker Production
```bash
# Build production image
docker build -t exoplanet-backend .

# Run with production settings
docker run -d \
  --name exoplanet-backend \
  -p 8000:8000 \
  -e DEBUG=false \
  -e LOG_LEVEL=WARNING \
  exoplanet-backend
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: exoplanet-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: exoplanet-backend
  template:
    metadata:
      labels:
        app: exoplanet-backend
    spec:
      containers:
      - name: backend
        image: exoplanet-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Start Redis if not running
   redis-server
   ```

2. **Worker Not Connecting**
   ```bash
   # Check Celery broker connection
   celery -A worker.celery_app inspect ping
   
   # Check worker registration
   celery -A worker.celery_app inspect registered
   ```

3. **Model Loading Issues**
   ```bash
   # Check model files exist
   ls -la models/
   
   # Test model loading
   python -c "from ml_service import get_predictor; print(get_predictor().is_loaded)"
   ```

4. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install --force-reinstall -r requirements.txt
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

### Monitoring

- View worker stats: http://localhost:5555 (Flower)
- Check API health: http://localhost:8000/health
- Monitor logs: `docker-compose logs -f backend`
- Worker inspection: `celery -A worker.celery_app inspect active`

## License

MIT License - see LICENSE file for details.
