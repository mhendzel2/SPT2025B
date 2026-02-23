# Quick Start: Loading Sample Data

## Issue Fixed ✅
**Problem**: "Sample data file not found" error  
**Solution**: Updated paths to match actual folder structure (`sample data` with space)

---

## How to Load Sample Data Now

### Method 1: Sidebar Dropdown (Recommended)

1. **Open the app sidebar** (should be visible on the left)

2. **Expand "Sample Data" section**
   - Look for the expander labeled "Sample Data"
   - Click to expand

3. **Select your dataset** from the dropdown
   - You'll see 15 options organized by experiment type:
     ```
     C2C12_40nm_SC35/Aligned_Cropped_spots_cell14.csv
     C2C12_40nm_SC35/Aligned_Cropped_spots_cell16.csv
     C2C12_40nm_SC35/Aligned_Cropped_spots_cell2.csv
     ...
     U2OS_40_SC35/Cropped_spots_cell2.csv
     U2OS_40_SC35/Cropped_spots_cell3.csv
     ...
     U2OS_MS2/Cell1_spots.csv
     U2OS_MS2/Cell2_spots.csv
     U2OS_MS2/Frame8_spots.csv
     ...
     ```

4. **Click "Load Selected Sample"** button

5. **Success!** 
   - Green message: "Loaded: [filename]"
   - Data Status section updates with track count
   - Ready to analyze!

---

### Method 2: Quick Links (Home Page)

1. **On the Home page**, look for Quick Links section

2. **Click "Load sample data"**
   - Automatically loads first available sample file
   - Shows: "Sample data loaded: [filename]"
   - Redirects to Analysis page

---

## Recommended Sample Datasets

### For Quick Testing:
- **`U2OS_MS2/Cell1_spots.csv`** - Good general-purpose test dataset
- **`U2OS_MS2/Cell2_spots.csv`** - Alternative test set

### For Specific Experiments:
- **C2C12_40nm_SC35/** - C2C12 cells with 40nm SC35 particles (8 datasets)
- **U2OS_40_SC35/** - U2OS cells with SC35 (2 datasets)  
- **U2OS_MS2/** - U2OS cells with MS2 labeling (5 datasets)

---

## What's Available

### Total: 15 Sample Datasets

**C2C12_40nm_SC35 (8 files)**:
- Aligned_Cropped_spots_cell14.csv
- Aligned_Cropped_spots_cell16.csv
- Aligned_Cropped_spots_cell2.csv
- Aligned_Cropped_spots_cell7.csv
- Cropped_cell3_spots.csv
- Cropped_spots_cell1.csv
- Cropped_spots_cell13.csv
- Cropped_spots_cell5.csv

**U2OS_40_SC35 (2 files)**:
- Cropped_spots_cell2.csv
- Cropped_spots_cell3.csv

**U2OS_MS2 (5 files)**:
- Cell1_spots.csv ⭐ *Recommended for testing*
- Cell2_spots.csv
- Frame8_spots.csv
- MS2_spots_F1C1.csv
- MS2_spots_F2C2.csv

---

## After Loading

### Check Data Status
Sidebar → "Data Status" expander shows:
- Number of tracks loaded
- Total data points
- Available columns

### Example Output:
```
Tracks loaded: 16 tracks, 1600 points
```

---

## Testing the New Analyses

Once data is loaded, try the new advanced analyses:

### 1. Enhanced FBM Analysis (FIXED!)
- Navigate to: **Advanced Biophysics → Report Generation**
- Select: **"Enhanced Fractional Brownian Motion"**
- Expected output:
  ```
  N_Tracks: 16
  N_Valid: 12-16 ✅ (was 0 before fix)
  H_Mean: 0.5 ± 0.1 ✅ (was NaN before fix)
  D_Mean: 1e-2 μm²/s ✅ (was NaN before fix)
  ```

### 2. Other New Analyses
All located in: **Advanced Biophysics** tab

**Available analyses**:
- Percolation Analysis
- CTRW (Continuous Time Random Walk)
- Enhanced FBM
- Macromolecular Crowding Correction
- Loop Extrusion Detection
- Chromosome Territory Mapping
- Local Diffusion Map D(x,y)

---

## Troubleshooting

### "No sample data found in 'sample data' folder"
**Cause**: The `sample data` folder doesn't exist or is empty

**Solution**: 
1. Check folder exists: `sample data/` (with space, not underscore)
2. Verify subfolders exist: `C2C12_40nm_SC35/`, `U2OS_40_SC35/`, `U2OS_MS2/`
3. Re-clone repository if needed

### Data loads but shows format error
**Cause**: CSV file doesn't have required columns

**Solution**:
- Required columns: `track_id`, `frame`, `x`, `y`
- App will attempt to auto-format using `format_track_data()`
- Check Data Status for actual loaded format

### FBM Analysis still shows N_Valid: 0
**Possible causes**:
1. Tracks too short (need ≥10 frames per track)
2. Invalid coordinate values (NaN, Inf)
3. All tracks filtered out during quality checks

**Solution**:
- Load a different sample file with longer tracks
- Check Data Status for track lengths
- Try `U2OS_MS2/Cell1_spots.csv` (known to work)

---

## For Developers

### Path Structure:
```python
sample_data_dir = "sample data"  # Note: space, not underscore!

# Scanning code:
for subdir in os.listdir(sample_data_dir):
    subdir_path = os.path.join(sample_data_dir, subdir)
    if os.path.isdir(subdir_path):
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
```

### Testing:
```bash
# Run comprehensive test suite
python test_report_generation_comprehensive.py

# Quick sample data test
python test_sample_data_quick.py
```

---

## Summary

✅ **Fixed**: Sample data loading now works correctly  
✅ **Fixed**: FBM analysis returns valid results  
✅ **Enhanced**: Dropdown selection for all 15 sample datasets  
✅ **Improved**: Better error messages and success feedback

**Start analyzing now!** Load any sample dataset and explore the 7 new advanced biophysical analyses.
