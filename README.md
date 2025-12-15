# SPT2025B - Single Particle Tracking Analysis Platform

<p align="center">
  <strong>A comprehensive Streamlit-based platform for biophysical single particle tracking analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.50+-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/NumPy-2.x%20Compatible-green" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Last%20Updated-December%202025-purple" alt="Updated">
</p>

---

## 📋 Overview

SPT2025B is a production-ready single particle tracking analysis platform designed for biophysical research. It provides comprehensive tools for particle detection, tracking, diffusion analysis, microrheology, and motion classification with **25+ analysis modules**.

### Key Capabilities
- 🔬 **Particle Detection & Tracking** - Multi-algorithm tracking with TrackPy integration
- 📊 **Diffusion Analysis** - MSD, anomalous diffusion, multiple diffusion models
- 🧬 **Microrheology** - Storage/loss modulus, creep compliance, viscoelastic characterization
- 🤖 **Machine Learning** - HMM-based state classification, ML trajectory analysis
- 📈 **Publication-Ready Reports** - Interactive HTML, PDF, and JSON exports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12 (Python 3.13+ not yet supported)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/mhendzel2/SPT2025B.git
cd SPT2025B

# Create virtual environment
python -m venv spt_env

# Activate environment
# Windows:
.\spt_env\Scripts\Activate.ps1
# Linux/Mac:
source spt_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port 5000
```

### Windows Quick Start
```powershell
# Simply run the batch script
.\start.bat
```

---

## 📦 Core Features

### Analysis Modules (25+)

| Category | Analyses |
|----------|----------|
| **Core** | Basic Statistics, Diffusion Analysis, Motion Classification, Velocity Correlation |
| **Biophysical** | Microrheology (G'/G''), Creep Compliance, Relaxation Modulus, Polymer Physics |
| **Advanced** | Confinement Analysis, Dwell Time, Clustering, Active Transport Detection |
| **ML-Enhanced** | HMM State Classification, iHMM Analysis, Bayesian Changepoint Detection |
| **Specialized** | Chromatin Dynamics, Nuclear Architecture, Percolation Analysis |

### Report Generation
- **Interactive UI Mode** - Point-and-click analysis selection
- **Batch Processing** - Automate multi-file analysis
- **Quick Presets** - Basic Package, Core Physics, Machine Learning, Complete Analysis
- **Export Formats** - HTML (interactive), PDF, JSON

### Supported File Formats
| Format | Extension | Handler |
|--------|-----------|---------|
| Generic CSV/Excel | `.csv`, `.xlsx` | `data_loader.py` |
| MetaMorph MVD2 | `.mvd2` | `mvd2_handler.py` |
| Volocity XML | `.xml` | `volocity_handler.py` |
| Imaris | `.ims` | `special_file_handlers.py` |
| TrackMate XML | `.xml` | `special_file_handlers.py` |
| MS2 Spots | `.csv` | `special_file_handlers.py` |

---

## 🏗️ Architecture

### Project Structure
```
SPT2025B/
├── app.py                      # Main Streamlit application
├── analysis.py                 # Core analysis functions (25+ methods)
├── analysis_manager.py         # High-level analysis coordinator
├── data_access_utils.py        # Unified data access layer
├── state_manager.py            # Session state management
├── enhanced_report_generator.py # Multi-analysis report builder
├── visualization.py            # Plotting and visualization
├── msd_calculation.py          # Vectorized MSD computation
├── rheology.py                 # Microrheology calculations
├── biophysical_models.py       # Polymer physics models
├── constants.py                # Configuration constants
├── config.toml                 # User-editable settings
├── requirements.txt            # Python dependencies
├── tests/                      # Pytest test modules
├── sample data/                # Example datasets
└── spt_projects/               # Saved project files
```

### Data Access Pattern
**Always use the data access utilities** - never access `st.session_state` directly:

```python
from data_access_utils import get_track_data, get_analysis_results, get_units

tracks_df, has_data = get_track_data()
if not has_data:
    st.error("No data loaded")
    return

# tracks_df guaranteed to have: track_id, frame, x, y columns
```

### Analysis Function Contract
All analysis functions follow a standardized signature:

```python
def analyze_X(tracks_df, pixel_size=1.0, frame_interval=1.0, **kwargs):
    """
    Args:
        tracks_df: DataFrame with track_id, frame, x, y columns
        pixel_size: Conversion factor (μm/pixel), default 0.1
        frame_interval: Time between frames (s), default 0.1
    
    Returns:
        dict: {
            'success': bool,
            'data': {...},      # Raw results
            'summary': {...},   # Summary statistics  
            'figures': {...}    # Plotly figure objects
        }
    """
```

---

## 🔧 Configuration

### Default Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `pixel_size` | 0.1 μm | Spatial calibration |
| `frame_interval` | 0.1 s | Temporal calibration |
| `max_lag` | 20 frames | MSD calculation lag |
| `cache_memory` | 10 GB | Analysis cache limit |
| `cache_items` | 1000 | Max cached results |

### Configuration Files
- **`config.toml`** - User-editable Streamlit and analysis settings
- **`constants.py`** - Application constants and session state keys
- **`.spt_settings.json`** - Persisted user preferences

---

## 🧪 Testing

```bash
# Run pytest suite
python -m pytest tests/test_app_logic.py -v

# Run standalone test scripts
python test_functionality.py
python test_report_generation.py
python test_comprehensive.py

# Verify all functions
python verify_all_functions.py
```

---

## 📊 Sample Data

Included sample datasets for testing:
- `Cell1_spots.csv` - Primary test dataset (recommended)
- `Cell2_spots.csv` - Additional cell data
- `Cropped_cell3_spots.csv` - Cropped field of view
- `MS2_spots_F1C1.csv` - MS2 labeling experiment

---

## 🔬 Scientific Background

### Diffusion Analysis
- **MSD Calculation**: $\langle r^2(\tau) \rangle = 4D\tau^\alpha$
- **Anomalous Exponent**: α < 1 (subdiffusion), α = 1 (normal), α > 1 (superdiffusion)

### Microrheology (GSER)
- **Storage Modulus**: $G'(\omega)$ - elastic component
- **Loss Modulus**: $G''(\omega)$ - viscous component
- **Complex Viscosity**: $\eta^*(\omega) = \frac{G^*(\omega)}{i\omega}$

### Motion Classification
- **Confined**: Restricted diffusion within boundaries
- **Normal**: Brownian motion (α ≈ 1)
- **Directed**: Active transport (α > 1.5)
- **Anomalous**: Subdiffusive or superdiffusive behavior

---

## 📚 Dependencies

### Core Requirements (December 2025)
```
streamlit>=1.50.0          # Web application framework
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0,<3.0.0       # Numerical computing (NumPy 2.x compatible)
scipy>=1.14.0              # Scientific computing
scikit-learn>=1.5.0        # Machine learning
plotly>=6.0.0              # Interactive visualization
trackpy>=0.7.0             # Particle tracking
```

### Advanced Analysis
```
emcee>=3.1.0               # MCMC parameter estimation
arviz>=0.22.0              # Bayesian diagnostics
hmmlearn>=0.3.3            # Hidden Markov Models
lmfit>=1.3.0               # Curve fitting
numba>=0.60.0              # JIT compilation
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the coding conventions in `.github/copilot-instructions.md`
4. Run tests before committing
5. Submit a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Documentation**: See `ANALYSIS_CATALOG.md` for complete analysis reference
- **Issues**: Open an issue on GitHub
- **Sample Data**: Use included CSV files for testing

---

## 🏷️ Version History

| Version | Date | Highlights |
|---------|------|------------|
| **2025.12** | Dec 2025 | Performance optimization, 10GB cache, vectorized MSD |
| **2025.10** | Oct 2025 | Microrheology enhancements, bug fixes |
| **2025.06** | Jun 2025 | Initial enhanced release with 16+ analyses |

---

<p align="center">
  <em>Developed for biophysical research • Single Particle Tracking made accessible</em>
</p>
