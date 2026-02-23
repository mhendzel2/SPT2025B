# Python 3.12 Virtual Environment Setup

## Overview
This document describes the new Python 3.12 virtual environment created to resolve compatibility issues with the previous Python 3.10.0rc2 environment.

## Issue Resolved
The original environment was using Python 3.10.0rc2 (a release candidate), which caused:
- `RuntimeError: no running event loop`
- `TypeError: WebSocketHandler.__init__() missing 2 required positional arguments`
- Incompatibility with Streamlit 1.46+ and Tornado 6.5+

## New Environment Details
- **Location**: `venv312/`
- **Python Version**: 3.12.10 (stable release)
- **Streamlit**: Latest compatible version (1.50.0+)
- **Tornado**: 6.5.2 (excludes problematic 6.5.0)

## Usage

### Option 1: Using the Batch Script (Easiest)
Simply run:
```batch
start_venv312.bat
```

### Option 2: Manual Activation
If PowerShell execution policy allows:
```powershell
.\venv312\Scripts\Activate.ps1
streamlit run app.py --server.port 8501
```

### Option 3: Direct Python Execution (No Activation Needed)
```powershell
.\venv312\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

## Installing Additional Packages
To install packages in this environment:
```powershell
.\venv312\Scripts\python.exe -m pip install <package_name>
```

Or install from requirements.txt:
```powershell
.\venv312\Scripts\python.exe -m pip install -r requirements.txt
```

## Verifying the Environment
Check Python version:
```powershell
.\venv312\Scripts\python.exe --version
```
Expected output: `Python 3.12.10`

Check installed packages:
```powershell
.\venv312\Scripts\python.exe -m pip list
```

## Notes
- The old environment (if any) remains untouched
- Python 3.12 provides better async/await support needed by modern Streamlit
- Tornado 6.5.0 is excluded via Streamlit's dependencies (uses 6.5.2)
- All project functionality should work identically with improved stability

## Troubleshooting

### If you get "execution policy" errors:
Use the direct execution method (Option 3) or the batch script (Option 1).

### If packages are missing:
```powershell
.\venv312\Scripts\python.exe -m pip install -r requirements.txt
```

### If Streamlit won't start:
1. Ensure port 8501 is available
2. Try a different port: `--server.port 5000`
3. Check for any lingering Python processes in Task Manager
