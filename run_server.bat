@echo off
REM ChurnIQ · Startup Script for Windows
REM Double-click this file to launch the dashboard

echo.
echo ============================================
echo   ChurnIQ  Customer Intelligence Platform
echo ============================================
echo.

REM Activate virtual environment if present
if not defined VIRTUAL_ENV (
    if exist .venv\Scripts\activate.bat (
        echo Activating virtual environment...
        call .venv\Scripts\activate.bat
    )
)

REM Install dependencies if needed
if not exist .deps_installed (
    echo Installing dependencies...
    pip install -r requirements.txt --quiet
    echo. > .deps_installed
)

REM Train models if missing
if not exist models\ridge_model.pkl (
    echo.
    echo First run detected — training models (this takes ~30s)...
    echo.
    python bootstrap_models.py
    if errorlevel 1 (
        echo ERROR: Model training failed. Check that Python and requirements are installed.
        pause
        exit /b 1
    )
)

echo.
echo Starting server...
echo Dashboard: http://localhost:5000
echo Press Ctrl+C to stop.
echo.

python app.py

pause
