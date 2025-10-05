#!/bin/bash

# Exoplanet Detection Backend Startup Script

set -e

echo "🚀 Starting Exoplanet Detection Backend Setup"

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    exit 1
fi

# Option 1: Docker Setup (Recommended)
echo ""
echo "📦 OPTION 1: Docker Setup (Recommended)"
echo "Run the following commands from the project root:"
echo ""
echo "  # Start Redis and Backend services"
echo "  docker-compose up -d redis backend"
echo ""
echo "  # Start Celery workers"
echo "  docker-compose up -d celery_worker"
echo ""
echo "  # Optional: Start monitoring with Flower"
echo "  docker-compose up -d flower"
echo ""
echo "  # Check status"
echo "  docker-compose ps"
echo ""

# Option 2: Virtual Environment Setup
echo "🐍 OPTION 2: Virtual Environment Setup"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing system dependencies (if needed)..."
# On macOS, you might need to install these:
# brew install redis

echo "Installing Python dependencies..."
# Install dependencies one by one to avoid conflicts
pip install --upgrade pip
pip install fastapi>=0.104.0
pip install uvicorn[standard]>=0.24.0
pip install websockets>=12.0
pip install redis>=5.0.0
pip install celery>=5.3.0
pip install pydantic>=2.5.0
pip install python-dotenv>=1.0.0
pip install python-multipart>=0.0.6

# Install ML dependencies with pre-built wheels
pip install numpy
pip install pandas
pip install scikit-learn
pip install joblib

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚦 To start the services:"
echo ""
echo "1. Start Redis (in a separate terminal):"
echo "   redis-server"
echo ""
echo "2. Start the FastAPI backend (in a separate terminal):"
echo "   cd backend && source venv/bin/activate && python main.py"
echo ""
echo "3. Start Celery workers (in a separate terminal):"
echo "   cd backend && source venv/bin/activate && celery -A worker.celery_app worker --loglevel=info"
echo ""
echo "4. Optional: Start Flower monitoring (in a separate terminal):"
echo "   cd backend && source venv/bin/activate && celery -A worker.celery_app flower"
echo ""
echo "🌐 Services will be available at:"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - WebSocket: ws://localhost:8000/ws/{client_id}"
echo "   - Flower (if started): http://localhost:5555"
echo ""
