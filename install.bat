@echo off
echo ========================================
echo SPT2025B Installation Script
echo ========================================

REM Check if Python is installed
echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo Python found!

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
echo Creating virtual environment 'venv'...
if exist venv (
    echo Virtual environment 'venv' already exists. Skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)

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
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
) else (
    echo WARNING: requirements.txt not found
    echo Installing core dependencies manually...
    pip install streamlit pandas numpy matplotlib plotly scipy scikit-learn scikit-image opencv-python pillow seaborn statsmodels openpyxl xmltodict lxml h5py
)

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
echo To run the application:
echo   1. Run: start.bat
echo   OR
echo   2. Activate environment: call venv\Scripts\activate.bat
echo   3. Run: streamlit run app.py
echo.
echo The application will be available at: http://localhost:8501
echo.
pause
