@echo off
setlocal enabledelayedexpansion

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

REM Check .python-version file
if exist .python-version (
    set /p EXPECTED_VERSION=<.python-version
    echo Expected Python version: !EXPECTED_VERSION!
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

REM Verify Python version in venv
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set CURRENT_VERSION=%%i
echo Current Python version: !CURRENT_VERSION!

if defined EXPECTED_VERSION (
    if "!CURRENT_VERSION!" NEQ "!EXPECTED_VERSION!" (
        echo WARNING: Python version mismatch!
        echo Expected: !EXPECTED_VERSION!
        echo Current:  !CURRENT_VERSION!
        echo.
        echo You may need to recreate the virtual environment.
        echo Run install.bat to update.
        echo.
    )
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