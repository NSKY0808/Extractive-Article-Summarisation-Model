@echo off
REM Complete startup script for Extractive Summarization System (Windows)
REM This script starts the Flask API and React frontend

setlocal enabledelayedexpansion
color 0B

echo ==========================================
echo Extractive Summarization - Full Stack
echo ==========================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found

REM Check for Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 14+
    pause
    exit /b 1
)
echo [OK] Node.js found

REM Check for npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Please install npm
    pause
    exit /b 1
)
echo [OK] npm found

echo.
echo Checking for trained models...

setlocal enabledelayedexpansion
set models_found=1

if not exist "project-root\experiments\logistic_regression_15k_model.pkl" (
    echo [WARNING] Missing: project-root\experiments\logistic_regression_15k_model.pkl
    set models_found=0
) else (
    echo [OK] Found: project-root\experiments\logistic_regression_15k_model.pkl
)

if not exist "project-root\experiments\linear_svm_15k_model.pkl" (
    echo [WARNING] Missing: project-root\experiments\linear_svm_15k_model.pkl
    set models_found=0
) else (
    echo [OK] Found: project-root\experiments\linear_svm_15k_model.pkl
)

if not exist "project-root\experiments\random_forest_15k_model.pkl" (
    echo [WARNING] Missing: project-root\experiments\random_forest_15k_model.pkl
    set models_found=0
) else (
    echo [OK] Found: project-root\experiments\random_forest_15k_model.pkl
)

if not exist "project-root\experiments\mlp_15k_model.pkl" (
    echo [WARNING] Missing: project-root\experiments\mlp_15k_model.pkl
    set models_found=0
) else (
    echo [OK] Found: project-root\experiments\mlp_15k_model.pkl
)

if !models_found! equ 0 (
    echo.
    echo [WARNING] Some models are missing. Train them using:
    echo cd project-root ^&^& python scripts/train_extractive_model.py --model-type [name] --train-limit 15000 --validation-limit 2000 --output-model-path experiments/[name]_15k_model.pkl --max-tfidf-features 8000
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" exit /b 1
)

echo.
echo Setting up Flask API...
cd project-root\api

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing API dependencies...
pip install -q -r requirements.txt

cd ..\..

echo.
echo Setting up React Frontend...
cd project-root\frontend

if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install --quiet
)

cd ..\..

echo.
echo [OK] Setup complete!
echo.
echo ==========================================
echo Starting services...
echo ==========================================
echo.

REM Start Flask API
echo Starting Flask API on port 5000...
start "Extractive Summarization - API" cmd /k "cd project-root\api && call venv\Scripts\activate.bat && python app.py"

REM Wait a bit for API to start
timeout /t 3 /nobreak

REM Start React
echo Starting React Frontend on port 3000...
start "Extractive Summarization - Frontend" cmd /k "cd project-root\frontend && npm start"

echo.
echo ==========================================
echo [OK] All services started!
echo ==========================================
echo.
echo Frontend:  http://localhost:3000
echo API:       http://localhost:5000
echo.
echo Close the terminal windows to stop the services.
echo.
pause
