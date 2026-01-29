@echo off
echo ========================================
echo AI Image Quality Enhancer - Backend Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Creating weights directory...
if not exist "weights" mkdir weights

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo IMPORTANT: Model weights will be downloaded automatically on first run.
echo.
echo To start the backend server, run:
echo   start_backend.bat
echo.
pause
