# SPT2025B Installation Guide

## Python Version Requirements

**IMPORTANT:** This application requires **Python 3.11 or 3.12**

### Why Not Python 3.13+?
Python 3.13 introduced breaking changes that cause compatibility issues with several scientific packages used in this application, including:
- NumPy 2.x incompatibilities
- Some Cython-based packages (scikit-image, etc.)
- TensorFlow and other ML libraries

## Installation Steps

### 1. Check Your Python Version

```powershell
python --version
```

You should see: `Python 3.11.x` or `Python 3.12.x`

If you have Python 3.13 or higher, you need to install Python 3.12:

#### Installing Python 3.12 on Windows:
1. Download from: https://www.python.org/downloads/release/python-31210/
2. Run the installer
3. **IMPORTANT:** Check "Add Python 3.12 to PATH"
4. Complete the installation

### 2. Create a Virtual Environment

```powershell
# Navigate to the project directory
cd C:\path\to\SPT2025B

# Create virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

**Note:** If you get a PowerShell execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

**If you encounter errors during installation:**

#### Common Issues and Solutions:

1. **Microsoft Visual C++ Build Tools Error (for Windows)**
   - Download and install: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "Desktop development with C++"
   - Restart PowerShell and try again

2. **NumPy Version Conflict**
   ```powershell
   pip uninstall numpy
   pip install "numpy>=1.24.0,<2.0.0"
   ```

3. **Specific Package Failing**
   Try installing the problematic package separately:
   ```powershell
   pip install --no-cache-dir package_name
   ```

4. **Dependency Resolver Taking Too Long**
   ```powershell
   pip install --use-deprecated=legacy-resolver -r requirements.txt
   ```

### 5. Verify Installation

```powershell
python -c "import streamlit, pandas, numpy, plotly, matplotlib; print('All core packages imported successfully!')"
```

If this runs without errors, your installation is successful!

## Running the Application

### Option 1: Using start.bat (Recommended for Windows)

```powershell
.\start.bat
```

This script will:
1. Activate the virtual environment
2. Check dependencies
3. Start the Streamlit server on port 8501

### Option 2: Manual Start

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the application
streamlit run app.py --server.port 5000
```

The application will open in your default web browser at http://localhost:5000

## Optional Components

### Machine Learning Features (TensorFlow)

If you need ML features, install TensorFlow:

```powershell
pip install "tensorflow>=2.13.0,<2.16.0"
```

**Note:** TensorFlow 2.16+ requires Python 3.12 or earlier.

### Advanced Segmentation (Cellpose/SAM)

These require PyTorch and may have large download sizes:

```powershell
# For Cellpose
pip install cellpose

# For Segment Anything (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Troubleshooting

### Issue: "ModuleNotFoundError" after installation

**Solution:** Make sure the virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

### Issue: "DLL load failed" errors on Windows

**Solutions:**
1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Update Windows
3. Reinstall the problematic package

### Issue: Application crashes with "Figure object has no attribute 'get'"

This was a bug in the visualization module. Make sure you have the latest version of the code with the fixes applied.

### Issue: "No available analyses" in microrheology tab

This has been fixed in the latest code. Ensure you're using the updated version.

## Updating Dependencies

To update all packages to their latest compatible versions:

```powershell
pip install --upgrade -r requirements.txt
```

## Uninstallation

To completely remove the application:

```powershell
# Deactivate virtual environment
deactivate

# Delete the virtual environment folder
Remove-Item -Recurse -Force venv

# Delete Python cache
Remove-Item -Recurse -Force __pycache__
```

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully - it often indicates the problem
2. Ensure you're using Python 3.11 or 3.12
3. Verify all dependencies installed successfully
4. Try reinstalling in a fresh virtual environment
5. Check the project's GitHub issues page

## System Requirements

- **OS:** Windows 10/11, macOS, or Linux
- **Python:** 3.11 or 3.12 (NOT 3.13+)
- **RAM:** Minimum 8GB, 16GB+ recommended for large datasets
- **Storage:** 2GB for dependencies, additional space for data files
- **Processor:** Multi-core recommended for analysis tasks
