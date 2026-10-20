@echo off
REM ============================================================
REM   Student Feedback Analysis - 1-Click Runner for Windows
REM ============================================================

echo ============================================================
echo   Student Feedback Analysis - Starting Web Dashboard
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10-3.13 and check "Add python.exe to PATH".
    pause
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Installing dependencies from requirements.txt...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [INFO] Launching Streamlit Application...
echo [INFO] Open your browser at http://localhost:8501 if it doesn't open automatically.
echo.
streamlit run app.py

pause
