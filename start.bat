@echo off
echo ========================================
echo SPT2025B - Starting Application
echo ========================================

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Verify Streamlit is installed
echo.
echo Checking Streamlit installation...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit not found in virtual environment
    echo Please run install.bat to install dependencies.
    pause
    exit /b 1
)

REM Start the application
echo.
echo ========================================
echo Starting SPT2025B Application...
echo ========================================
echo.
echo The application will open in your default browser.
echo Server running at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py --server.port 8501

REM If streamlit exits
echo.
echo Application stopped.
pause