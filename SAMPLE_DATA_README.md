# Sample Data Documentation

## Overview

The `sample data/` directory contains 15 curated single-particle tracking datasets organized into 3 categories for testing and demonstration purposes.

## Directory Structure

```
sample data/
├── C2C12_40nm_SC35/          # C2C12 cell line with 40nm SC35 particles (8 files)
│   ├── Aligned_Cropped_spots_cell14.csv
│   ├── Aligned_Cropped_spots_cell16.csv
│   ├── Aligned_Cropped_spots_cell2.csv
│   ├── Aligned_Cropped_spots_cell7.csv
│   ├── Cropped_cell3_spots.csv
│   ├── Cropped_spots_cell1.csv
│   ├── Cropped_spots_cell13.csv
│   └── Cropped_spots_cell5.csv
│
├── U2OS_40_SC35/             # U2OS cell line with 40nm SC35 particles (2 files)
│   ├── Cropped_spots_cell2.csv
│   └── Cropped_spots_cell3.csv
│
└── U2OS_MS2/                 # U2OS cells with MS2 mRNA tracking (5 files)
    ├── Cell1_spots.csv
    ├── Cell2_spots.csv
    ├── Frame8_spots.csv
    ├── MS2_spots_F1C1.csv
    └── MS2_spots_F2C2.csv
```

## Dataset Categories

### 1. C2C12_40nm_SC35 (8 datasets, 2.5 MB total)

**Description**: Mouse C2C12 myoblast cells labeled with 40nm fluorescent particles tracking SC35 nuclear speckle proteins.

**Characteristics**:
- **Cell Type**: C2C12 mouse myoblast
- **Particle Size**: 40nm fluorescent
- **Target**: SC35 nuclear speckle proteins
- **Columns**: 38 (includes TRACK_ID, FRAME, POSITION_X/Y/Z, multi-channel intensities)
- **Features**: Track data, temporal dynamics, multi-channel fluorescence

**Recommended Use Cases**:
- Diffusion coefficient estimation
- Anomalous diffusion analysis (α values)
- State segmentation (HMM/iHMM)
- Microrheology analysis
- Multi-channel correlation studies

**File Details**:
| File | Size | Tracks | Frames | Particles |
|------|------|--------|--------|-----------|
| Aligned_Cropped_spots_cell14.csv | 304 KB | 28 | 90 | 1,194 |
| Aligned_Cropped_spots_cell16.csv | 180 KB | 25 | 90 | 707 |
| Aligned_Cropped_spots_cell2.csv | 122 KB | 28 | 54 | 479 |
| Aligned_Cropped_spots_cell7.csv | 189 KB | 22 | 90 | 742 |
| Cropped_cell3_spots.csv | 469 KB | 32 | 90 | 1,842 |
| Cropped_spots_cell1.csv | 259 KB | - | - | 1,014 |
| Cropped_spots_cell13.csv | 46 KB | 12 | 30 | 181 |
| Cropped_spots_cell5.csv | 883 KB | 55 | 90 | 3,464 |

### 2. U2OS_40_SC35 (2 datasets, 635 KB total)

**Description**: Human U2OS osteosarcoma cells labeled with 40nm fluorescent particles tracking SC35 nuclear speckle proteins.

**Characteristics**:
- **Cell Type**: U2OS human osteosarcoma
- **Particle Size**: 40nm fluorescent
- **Target**: SC35 nuclear speckle proteins
- **Columns**: 38 (full tracking and intensity data)
- **Features**: Track data, temporal dynamics, multi-channel fluorescence

**Recommended Use Cases**:
- Cross-species comparison (vs C2C12)
- Nuclear dynamics analysis
- Confined diffusion studies
- Two-point microrheology

**File Details**:
| File | Size | Tracks | Frames | Particles |
|------|------|--------|--------|-----------|
| Cropped_spots_cell2.csv | 458 KB | 29 | 90 | 1,797 |
| Cropped_spots_cell3.csv | 177 KB | 20 | 80 | 693 |

### 3. U2OS_MS2 (5 datasets, 2.8 MB total)

**Description**: U2OS cells with MS2-tagged mRNA molecules for single-mRNA tracking studies.

**Characteristics**:
- **Cell Type**: U2OS human osteosarcoma
- **Target**: MS2-tagged mRNA molecules
- **Columns**: 30-38 (varies by file)
- **Features**: Track data, temporal dynamics, multi-channel fluorescence
- **Application**: Single-molecule mRNA tracking

**Recommended Use Cases**:
- Single-molecule diffusion analysis
- mRNA transport dynamics
- Directional motion detection
- Velocity autocorrelation analysis
- Jump distance distribution analysis

**File Details**:
| File | Size | Tracks | Frames | Particles |
|------|------|--------|--------|-----------|
| Cell1_spots.csv | 564 KB | 32 | 90 | 2,211 |
| Cell2_spots.csv | 1,122 KB | 60 | 90 | 4,405 |
| Frame8_spots.csv | 130 KB | 10 | 90 | 509 |
| MS2_spots_F1C1.csv | 533 KB | 31 | 90 | 2,091 |
| MS2_spots_F2C2.csv | 517 KB | 29 | 90 | 2,028 |

## Data Format

All CSV files follow a consistent format with these key columns:

### Required Tracking Columns
- `TRACK_ID`: Unique identifier for each trajectory
- `FRAME`: Time frame index
- `POSITION_X`: X-coordinate (μm)
- `POSITION_Y`: Y-coordinate (μm)
- `POSITION_Z`: Z-coordinate (μm, if 3D)

### Quality Metrics
- `QUALITY`: Track quality score
- `SNR_CH1/2/3`: Signal-to-noise ratio per channel
- `MEAN_INTENSITY_CH1/2/3`: Mean intensity per channel
- `TOTAL_INTENSITY_CH1/2/3`: Integrated intensity per channel

### Additional Columns
- `LABEL`: Spot detection label
- `ID`: Global particle ID
- Various intensity statistics per channel

## Using Sample Data in SPT2025B

### Method 1: Through UI (Recommended)

1. **Navigate to Data Loading Tab**
   - Open SPT2025B application
   - Go to "Data Loading" tab
   - Click on "Sample Data" sub-tab

2. **Select Dataset**
   - Choose from dropdown (organized by category)
   - View dataset information (size, features, columns)
   - Click "Load Selected Dataset"

3. **Verify Loading**
   - Check data preview table
   - Review metrics (particles, tracks, frames)
   - Proceed to analysis tabs

### Method 2: Programmatic Access

```python
from sample_data_manager import SampleDataManager

# Initialize manager
manager = SampleDataManager(sample_data_dir="sample data")

# Get available datasets
datasets = manager.get_available_datasets()

# Load specific dataset
df = manager.load_dataset("U2OS_MS2/Cell1_spots.csv")

# Use in analysis
from analysis import calculate_msd
msd_result = calculate_msd(df, pixel_size=0.1, frame_interval=0.1)
```

### Method 3: Direct File Access

```python
import pandas as pd

# Load directly
df = pd.read_csv("sample data/U2OS_MS2/Cell1_spots.csv")

# Format for analysis (if needed)
from data_loader import format_track_data
formatted_df = format_track_data(df)
```

## Recommended Analysis Workflows

### Workflow 1: Basic Diffusion Analysis
**Dataset**: `U2OS_MS2/Cell1_spots.csv`

1. Load dataset through UI
2. Calculate MSD → estimate D and α
3. Generate van Hove distribution
4. Perform anomaly detection
5. Export report

**Expected Results**:
- D ≈ 0.05-0.2 μm²/s (mRNA typically subdiffusive)
- α ≈ 0.6-0.9 (anomalous/confined diffusion)

### Workflow 2: State Segmentation
**Dataset**: `C2C12_40nm_SC35/Cropped_cell3_spots.csv`

1. Load dataset (1,842 particles, 32 tracks)
2. Run HMM/iHMM analysis (2-3 states expected)
3. Analyze dwell times
4. Compare diffusion coefficients per state
5. Generate state transition plot

**Expected Results**:
- 2-3 distinct mobility states
- Slow state: D ~ 0.01-0.05 μm²/s (bound/confined)
- Fast state: D ~ 0.1-0.5 μm²/s (free diffusion)

### Workflow 3: Microrheology
**Dataset**: `U2OS_40_SC35/Cropped_spots_cell2.csv`

1. Load dataset (1,797 particles, 29 tracks)
2. Calculate GSER-based rheology
3. Compute G'(ω) and G"(ω)
4. Estimate viscosity
5. Perform equilibrium validation

**Expected Results**:
- G' and G" cross over at ~1-10 Hz
- Viscosity ~ 0.1-1 Pa·s (nuclear environment)

### Workflow 4: 2025 Methods Testing
**Dataset**: Any C2C12 or U2OS dataset

1. Load dataset
2. Run CVE/MLE bias-corrected estimation
3. Use acquisition advisor for frame rate validation
4. Perform equilibrium validity checks
5. Test microsecond sampling detection
6. Generate comprehensive report

## Git Synchronization

The sample data is **included in the repository** with special `.gitignore` rules:

```gitignore
# Ignore all CSV files
*.csv

# But allow sample data
!sample data/
!sample data/**/*.csv
```

This ensures:
- ✅ Sample data is versioned and shared
- ✅ User-generated CSVs are excluded
- ✅ Repository stays manageable (total sample data ~5.2 MB)

## Adding New Sample Data

To add new sample datasets:

1. **Create subdirectory** in `sample data/` (e.g., `sample data/MyExperiment/`)
2. **Copy CSV files** into subdirectory
3. **Auto-discovery**: Files automatically appear in UI (no code changes needed)
4. **Commit to git**:
   ```bash
   git add "sample data/MyExperiment/"
   git commit -m "Add MyExperiment sample data"
   ```

The `sample_data_manager.py` will automatically:
- Discover new files
- Detect structure (tracks, temporal, multi-channel)
- Categorize by subdirectory
- Display in UI with metadata

## Data Provenance

All sample data originates from:
- **Source**: Single-particle tracking experiments
- **Tracking Software**: Likely TrackMate (based on column names)
- **Export Format**: CSV with 30-38 columns
- **Quality Control**: Pre-filtered for tracking quality

**Citation**: If publishing results using this sample data, acknowledge:
> Sample tracking data provided with SPT2025B software (github.com/mhendzel2/SPT2025B)

## Data Quality

All datasets have been verified for:
- ✅ Valid TRACK_ID and FRAME columns
- ✅ Reasonable position ranges (no NaN/Inf)
- ✅ Multi-channel intensity data present
- ✅ Track lengths > 5 frames (suitable for MSD analysis)
- ✅ CSV format compatible with pandas

## Performance Considerations

| Dataset Size | Load Time | Memory Usage | Recommended For |
|--------------|-----------|--------------|-----------------|
| < 200 KB | < 0.5s | < 5 MB | Quick tests |
| 200-500 KB | 0.5-1s | 5-15 MB | Standard workflows |
| 500-1000 KB | 1-2s | 15-30 MB | Full analysis |
| > 1000 KB | 2-5s | 30-50 MB | Batch processing |

**Largest Dataset**: `C2C12_40nm_SC35/Cropped_spots_cell5.csv` (883 KB, 3,464 particles)

## Troubleshooting

### Problem: "No sample datasets found"
**Solution**: 
1. Check that `sample data/` directory exists in project root
2. Verify CSV files are present: `ls "sample data" -Recurse -Filter *.csv`
3. Restart Streamlit application

### Problem: "Error loading dataset"
**Solution**:
1. Check CSV file is not corrupted
2. Verify pandas can read it: `pd.read_csv("sample data/path/to/file.csv")`
3. Check for proper column names (TRACK_ID, FRAME, etc.)

### Problem: Dataset doesn't appear in UI
**Solution**:
1. Clear Streamlit cache: Click "Clear cache" in settings
2. Reload page (Ctrl+F5)
3. Check `sample_data_manager.py` logs for errors

## Updates and Maintenance

**Last Updated**: October 6, 2025  
**Total Datasets**: 15  
**Total Size**: ~5.2 MB  
**Format Version**: 1.0  

For issues or suggestions, open an issue on GitHub: [SPT2025B Issues](https://github.com/mhendzel2/SPT2025B/issues)
