@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHON_EXE=%CD%\api\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found at:
    echo         %PYTHON_EXE%
    echo.
    echo Start the demo setup first or create the API venv before training.
    exit /b 1
)

echo ============================================
echo Training all 4 models on the 15k dataset
echo ============================================
echo Using Python: %PYTHON_EXE%
echo.

call :run_model logistic_regression
if errorlevel 1 exit /b 1

call :run_model linear_svm
if errorlevel 1 exit /b 1

call :run_model random_forest
if errorlevel 1 exit /b 1

call :run_model mlp
if errorlevel 1 exit /b 1

echo.
echo ============================================
echo All 15k training runs completed successfully.
echo Models and metrics are in project-root\experiments
echo ============================================
exit /b 0

:run_model
set "MODEL_NAME=%~1"
echo.
echo --------------------------------------------
echo Training %MODEL_NAME%
echo --------------------------------------------

"%PYTHON_EXE%" scripts\train_extractive_model.py ^
  --model-type %MODEL_NAME% ^
  --train-limit 15000 ^
  --validation-limit 2000 ^
  --output-model-path experiments\%MODEL_NAME%_15k_model.pkl ^
  --metrics-output-path experiments\%MODEL_NAME%_15k_metrics.json ^
  --max-tfidf-features 8000

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for %MODEL_NAME%
    exit /b 1
)

echo [OK] Finished %MODEL_NAME%
exit /b 0
