@echo off
setlocal enabledelayedexpansion

echo ========================================
echo SPT2025B Installation Script
echo ========================================

REM Check if Python is installed
echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check Python version compatibility
echo.
echo Checking Python version compatibility...
if !PYTHON_MAJOR! LSS 3 (
    echo ERROR: Python 3.11 or 3.12 is required
    echo Current version: %PYTHON_VERSION%
    echo Please install Python 3.11 or 3.12
    pause
    exit /b 1
)

if !PYTHON_MAJOR! EQU 3 (
    if !PYTHON_MINOR! LSS 11 (
        echo ERROR: Python 3.11 or 3.12 is required
        echo Current version: %PYTHON_VERSION%
        echo Please upgrade to Python 3.11 or 3.12
        pause
        exit /b 1
    )
    if !PYTHON_MINOR! GTR 12 (
        echo WARNING: Python 3.13+ detected!
        echo Current version: %PYTHON_VERSION%
        echo.
        echo This application requires Python 3.11 or 3.12 due to dependency compatibility.
        echo Python 3.13+ has breaking changes with NumPy 2.x and scientific packages.
        echo.
        echo Please install Python 3.12 from:
        echo https://www.python.org/downloads/release/python-31210/
        echo.
        echo Then you can specify the version with: py -3.12 install.bat
        pause
        exit /b 1
    )
)

echo ✓ Python version compatible: %PYTHON_VERSION%

REM Check pip
echo.
echo Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment 'venv' with Python %PYTHON_VERSION%...
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
        pause
        exit /b 1
    )
)

python -m venv venv
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
findstr /C:"venv/" .gitignore >nul 2>&1
if errorlevel 1 (
    echo venv/ >> .gitignore
    echo Added 'venv/' to .gitignore
) else (
    echo 'venv/' already in .gitignore
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

REM Upgrade pip
echo.
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
echo.
echo Installing requirements from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        echo.
        echo Try installing Visual C++ Build Tools:
        echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
        pause
        exit /b 1
    )
) else (
    echo WARNING: requirements.txt not found
    echo Installing core dependencies manually...
    pip install streamlit pandas "numpy>=1.24.0,<2.0.0" matplotlib plotly scipy scikit-learn scikit-image opencv-python pillow seaborn statsmodels openpyxl xmltodict lxml h5py
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import streamlit, pandas, numpy, plotly, matplotlib; print('✓ Core packages verified')" 2>nul
if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly
) else (
    echo ✓ All core packages installed successfully
)

REM Create .python-version file
echo %PYTHON_VERSION% > .python-version
echo ✓ Created .python-version file

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
    echo Created .streamlit\config.toml
) else (
    echo .streamlit\config.toml already exists
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Python version: %PYTHON_VERSION%
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
