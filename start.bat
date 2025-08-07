@echo off
echo ========================================
echo SPT Analysis Platform - Windows Setup
echo ========================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found. Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if exist spt_env (
    echo Virtual environment already exists
) else (
    python -m venv spt_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo Activating virtual environment...
call spt_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install plotly>=5.15.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0
pip install scikit-image>=0.21.0
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install seaborn>=0.12.0
pip install statsmodels>=0.14.0
pip install openpyxl>=3.1.0
pip install xmltodict>=0.13.0
pip install lxml>=4.9.0

echo.
echo Creating Streamlit configuration...
if not exist .streamlit mkdir .streamlit
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

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the application:
echo 1. Activate environment: spt_env\Scripts\activate
echo 2. Run application: streamlit run app_enhanced_comprehensive.py
echo 3. Open browser to: http://localhost:8501
echo.
echo Press any key to start the application now...
pause

echo.
echo Starting SPT Analysis Platform...
streamlit run app_enhanced_comprehensive.py --server.port 8501

pause