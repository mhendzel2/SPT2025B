@echo off
setlocal enabledelayedexpansion

echo ========================================
echo SPT2025B Installation Script (Python 3.12)
echo ========================================

REM Check if py launcher is available
echo.
echo Checking for Python Launcher...
py -0 >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher (py) not found
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Python 3.12 is available
echo.
echo Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 not found
    echo.
    echo Available Python versions:
    py -0
    echo.
    echo Please install Python 3.12 from:
    echo https://www.python.org/downloads/release/python-31210/
    pause
    exit /b 1
)

REM Get Python 3.12 version
for /f "tokens=2" %%i in ('py -3.12 --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Found Python %PYTHON_VERSION%

REM Check pip
echo.
echo Checking pip...
py -3.12 -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found for Python 3.12
    echo Please reinstall Python 3.12 with pip included
    pause
    exit /b 1
)
echo ✓ pip is available

REM Create virtual environment
echo.
echo Creating virtual environment with Python 3.12...
if exist venv (
    echo Virtual environment 'venv' already exists.
    set /p RECREATE="Recreate it? (y/N): "
    if /i "!RECREATE!" NEQ "y" (
        echo Skipping virtual environment creation.
        goto :install_deps
    )
    echo Removing existing virtual environment...
    rmdir /s /q venv
    if errorlevel 1 (
        echo ERROR: Failed to remove existing virtual environment
        echo Please close any programs using the virtual environment
        pause
        exit /b 1
    )
)

py -3.12 -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created successfully!

:install_deps

REM Add venv to .gitignore if not already present
echo.
echo Updating .gitignore...
if not exist .gitignore (
    echo venv/ > .gitignore
    echo __pycache__/ >> .gitignore
    echo *.pyc >> .gitignore
    echo .python-version >> .gitignore
    echo Created .gitignore
) else (
    findstr /C:"venv/" .gitignore >nul 2>&1
    if errorlevel 1 (
        echo venv/ >> .gitignore
        echo Added 'venv/' to .gitignore
    ) else (
        echo 'venv/' already in .gitignore
    )
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

REM Verify we're using Python 3.12
echo.
echo Verifying virtual environment Python version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set VENV_VERSION=%%i
echo Virtual environment Python version: %VENV_VERSION%

REM Upgrade pip
echo.
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

REM Install requirements
echo.
echo Installing requirements from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        echo.
        echo This may be due to missing C++ build tools.
        echo Try installing Visual C++ Build Tools:
        echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
        pause
        exit /b 1
    )
) else (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import streamlit; print('✓ Streamlit:', streamlit.__version__)" 2>nul
python -c "import pandas; print('✓ Pandas:', pandas.__version__)" 2>nul
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>nul
python -c "import plotly; print('✓ Plotly:', plotly.__version__)" 2>nul
python -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" 2>nul

if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly
) else (
    echo.
    echo ✓ All core packages installed successfully
)

REM Create .python-version file
echo %VENV_VERSION% > .python-version
echo ✓ Created .python-version file with version %VENV_VERSION%

REM Create Streamlit configuration directory
echo.
echo Creating Streamlit configuration...
if not exist .streamlit mkdir .streamlit

REM Create config.toml if it doesn't exist
if not exist .streamlit\config.toml (
    (
    echo [server]
    echo headless = true
    echo address = "0.0.0.0"
    echo port = 8501
    echo maxUploadSize = 200
    echo.
    echo [theme]
    echo base = "light"
    echo.
    echo [browser]
    echo gatherUsageStats = false
    ) > .streamlit\config.toml
    echo ✓ Created .streamlit\config.toml
) else (
    echo .streamlit\config.toml already exists
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Python version: %VENV_VERSION%
echo Virtual environment: venv\
echo.
echo To start the application, run:
echo     start.bat
echo.
echo Or manually:
echo     venv\Scripts\activate
echo     streamlit run app.py --server.port 5000
echo.
pause
