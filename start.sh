#!/bin/bash
# Complete startup script for Extractive Summarization System
# This script:
# 1. Checks for trained models
# 2. Starts the Flask API
# 3. Starts the React frontend

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Extractive Summarization - Full Stack"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

if ! command_exists node; then
    echo -e "${RED}✗ Node.js not found. Please install Node.js 14+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found${NC}"

if ! command_exists npm; then
    echo -e "${RED}✗ npm not found. Please install npm${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm found${NC}"

echo ""

# Check for models
echo "Checking for trained models..."
MODELS=(
    "experiments/logistic_regression_15k_model.pkl"
    "experiments/linear_svm_15k_model.pkl"
    "experiments/random_forest_15k_model.pkl"
    "experiments/mlp_15k_model.pkl"
)

MISSING_MODELS=()
for model in "${MODELS[@]}"; do
    if [ -f "$model" ]; then
        echo -e "${GREEN}✓ Found: $model${NC}"
    else
        echo -e "${YELLOW}⚠ Missing: $model${NC}"
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠ Some models are missing. You can train them using:${NC}"
    echo "python scripts/train_extractive_model.py --model-type [model_name] --train-limit 15000 --validation-limit 2000 --output-model-path experiments/[model_name]_15k_model.pkl --max-tfidf-features 8000"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Setup API
echo "Setting up Flask API..."
cd api

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

if [ -f "requirements.txt" ]; then
    echo "Installing API dependencies..."
    pip install -q -r requirements.txt
fi

cd "$PROJECT_ROOT"

# Setup Frontend
echo "Setting up React Frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies (this may take a few minutes)..."
    npm install --quiet
fi

cd "$PROJECT_ROOT"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "=========================================="
echo "Starting services..."
echo "=========================================="
echo ""

# Start Flask API in the background
echo "Starting Flask API (port 5000)..."
cd api
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
python app.py &
API_PID=$!
echo -e "${GREEN}✓ API started (PID: $API_PID)${NC}"

# Wait a bit for API to start
sleep 2

# Start React in another terminal/process
echo "Starting React Frontend (port 3000)..."
cd "$PROJECT_ROOT/frontend"
npm start &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ All services started!${NC}"
echo "=========================================="
echo ""
echo "Frontend:  http://localhost:3000"
echo "API:       http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services..."
echo ""

# Wait for processes
wait $API_PID $FRONTEND_PID

# Cleanup
trap "kill $API_PID $FRONTEND_PID 2>/dev/null" EXIT
