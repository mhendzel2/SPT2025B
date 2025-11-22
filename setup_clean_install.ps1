# SPT2025B Clean Installation Script for PowerShell
# This script will completely reinstall the application with Python 3.12

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SPT2025B Clean Installation (Python 3.12)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Working directory: $ScriptDir" -ForegroundColor Yellow
Write-Host ""

# Step 1: Check for Python 3.12
Write-Host "Step 1: Checking for Python 3.12..." -ForegroundColor Green
try {
    $pyVersions = py -0 2>&1
    Write-Host $pyVersions
    
    $hasPy312 = py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python 3.12 found: $hasPy312" -ForegroundColor Green
    } else {
        Write-Host "✗ ERROR: Python 3.12 not found" -ForegroundColor Red
        Write-Host "Please install Python 3.12 from: https://www.python.org/downloads/release/python-31210/" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host "✗ ERROR: Python launcher not found" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Step 2: Clean up old environment
Write-Host "Step 2: Cleaning up old environment..." -ForegroundColor Green

if (Test-Path "venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv" -ErrorAction SilentlyContinue
    if (Test-Path "venv") {
        Write-Host "✗ Could not remove venv. Please close any programs using it and try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✓ Old venv removed" -ForegroundColor Green
}

if (Test-Path ".python-version") {
    Remove-Item ".python-version" -Force
}

Write-Host "✓ Cleanup complete" -ForegroundColor Green
Write-Host ""

# Step 3: Create new virtual environment with Python 3.12
Write-Host "Step 3: Creating virtual environment with Python 3.12..." -ForegroundColor Green
py -3.12 -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ ERROR: Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✓ Virtual environment created" -ForegroundColor Green
Write-Host ""

# Step 4: Activate and verify
Write-Host "Step 4: Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

$venvPython = & python --version
Write-Host "✓ Virtual environment Python: $venvPython" -ForegroundColor Green
Write-Host ""

# Step 5: Upgrade pip
Write-Host "Step 5: Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ pip upgraded" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: pip upgrade had issues, continuing..." -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Install requirements
Write-Host "Step 6: Installing requirements..." -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ All requirements installed" -ForegroundColor Green
    } else {
        Write-Host "✗ ERROR: Failed to install some requirements" -ForegroundColor Red
        Write-Host "You may need Visual C++ Build Tools:" -ForegroundColor Yellow
        Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
        Read-Host "Press Enter to continue anyway"
    }
} else {
    Write-Host "✗ ERROR: requirements.txt not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Step 7: Verify installation
Write-Host "Step 7: Verifying installation..." -ForegroundColor Green
$packages = @("streamlit", "pandas", "numpy", "plotly", "matplotlib", "scipy", "scikit-learn")
foreach ($pkg in $packages) {
    try {
        $version = python -c "import $pkg; print($pkg.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $pkg : $version" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $pkg : FAILED" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ✗ $pkg : FAILED" -ForegroundColor Red
    }
}

Write-Host ""

# Step 8: Save Python version
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$pythonVersion | Out-File -FilePath ".python-version" -Encoding ASCII -NoNewline
Write-Host "✓ Saved Python version: $pythonVersion" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Yellow
Write-Host "  1. Make sure venv is activated: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Run: streamlit run app.py --server.port 5000" -ForegroundColor White
Write-Host ""
Write-Host "Or use the start.bat script" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
