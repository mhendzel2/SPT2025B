@echo off
REM Start script for SPT2025B with Python 3.12 virtual environment

echo Starting SPT2025B with Python 3.12...
echo.

REM Run Streamlit with the Python 3.12 virtual environment
.\venv312\Scripts\python.exe -m streamlit run app.py --server.port 8501

pause
