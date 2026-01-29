@echo off
echo ========================================
echo Starting AI Image Enhancer Backend
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to set up the backend.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting FastAPI server on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py
