# Quick Testing Guide for App.py Refactoring

## How to Test Each Fixed/New Feature

### Prerequisites
1. Start the application:
   ```powershell
   streamlit run app.py --server.port 5000
   ```
2. Load sample tracking data (e.g., `Cell1_spots.csv`)

---

## 1. Test Changepoint Detection (FIXED)

**Navigate**: Advanced Analysis → Changepoint Detection

**Steps**:
1. Adjust parameters:
   - Window Size: 10 (default)
   - Significance Level: 0.05 (default)
   - Minimum Segment Length: 5 (default)
2. Click "Run Changepoint Detection"
3. **Expected Output**:
   - ✅ Success message
   - Motion segments table with track_id, segment_id, start_frame, duration
   - Changepoints table (if detected)
   - Regime classification chart

**What Was Fixed**:
- Removed nested button causing UI freeze
- Changed from non-existent function to `ChangePointDetector.detect_motion_regime_changes()`
- Added missing parameter sliders

---

## 2. Test Correlative Analysis (FIXED)

**Navigate**: Advanced Analysis → Correlative Analysis

**Prerequisites**: Data must have intensity channels (columns with 'intensity' or 'ch' in name)

**Steps**:
1. (Optional) Add custom channel labels in text inputs
2. Adjust parameters:
   - Lag Range: 5 frames (default)
   - Minimum Track Length: 10 (default)
3. Click "Run Correlative Analysis"
4. **Expected Output**:
   - ✅ Success message
   - Track coupling table (top 20 rows)
   - Summary metrics: Tracks Analyzed, Mean Correlation, Intensity Channels
   - Ensemble Correlations (per channel)
   - Temporal Cross-Correlation plot (lag vs correlation)

**What Was Fixed**:
- Changed from non-existent `analyze_motion_intensity_correlation()` to proper `CorrelativeAnalyzer.analyze_intensity_motion_coupling()`
- Added parameter controls
- Implemented custom channel labeling

---

## 3. Test Advanced Tracking (NEW)

**Navigate**: Advanced Analysis → Advanced Tracking

### Method 1: Particle Filter Tracking

**Steps**:
1. Select "Particle Filter" method
2. Adjust parameters:
   - Number of Particles: 200 (default)
   - Motion Noise: 1.0 (default)
   - Measurement Noise: 0.5 (default)
   - Max Linking Distance: 10 pixels (default)
3. Click "Run Particle Filter Tracking"
4. **Expected Output**:
   - ✅ Success message
   - Tracking statistics: Total Tracks, Total Detections, Mean Track Length
   - Refined tracks table (top 20 rows)

### Method 2: Bayesian Detection Refinement

**Steps**:
1. Select "Bayesian Detection Refinement" method
2. Adjust parameters:
   - Prior Weight: 0.3 (default)
   - Localization Precision: 20 nm (default)
   - Refinement Iterations: 3 (default)
3. Click "Run Bayesian Refinement"
4. **Expected Output**:
   - ✅ Success message
   - Refined detections table (top 20 rows)
   - Adjustment statistics: Mean/Max Adjustment, Detections Refined

---

## 4. Test Intensity Analysis (NEW)

**Navigate**: Advanced Analysis → Intensity Analysis

**Prerequisites**: Data must have intensity channels

### Analysis 1: Intensity-Movement Correlation

**Steps**:
1. Select "Intensity-Movement Correlation"
2. Select intensity channels (default: first 2 channels)
3. Click "Run Intensity-Movement Correlation"
4. **Expected Output**:
   - ✅ Success message
   - Correlation coefficients table
   - Correlation heatmap (plotly interactive)
   - Time series plot

### Analysis 2: Intensity Profiles

**Steps**:
1. Select "Intensity Profiles"
2. Check/uncheck "Normalize Intensities" (default: checked)
3. Click "Analyze Intensity Profiles"
4. **Expected Output**:
   - ✅ Success message
   - Profile statistics table
   - Intensity profiles line chart

### Analysis 3: Intensity Behavior Classification

**Steps**:
1. Select "Intensity Behavior Classification"
2. Adjust "Number of Behavior Classes": 3 (default, range 2-5)
3. Click "Classify Intensity Behavior"
4. **Expected Output**:
   - ✅ Success message
   - Classified tracks table (top 20 rows)
   - Behavior distribution bar chart
   - Cluster centers table

---

## 5. Verify Polymer Physics (ALREADY WORKING)

**Navigate**: Advanced Analysis → Biophysical Models → Polymer Physics

**Steps**:
1. Select model type: "Auto-fit α" (default)
2. Adjust parameters (e.g., Persistence Length: 50 nm)
3. Click "Run Polymer Physics Analysis"
4. **Expected Output**:
   - ✅ Success message
   - Model parameters: Alpha, K values
   - Model fit plot (MSD vs lag time)
   - (Optional) Fractal dimension results
   - (Optional) Crowding correction metrics

---

## 6. Verify Active Transport (ALREADY WORKING)

**Navigate**: Advanced Analysis → Biophysical Models → Active Transport

**Steps**:
1. Adjust parameters:
   - Speed Threshold: 0.5 μm/s (default)
   - Straightness Threshold: 0.7 (default)
   - Minimum Track Length: 10 (default)
   - Velocity Window: 5 frames (default)
2. Click "Analyze Active Transport"
3. **Expected Output**:
   - ✅ Success message
   - Summary: Total Tracks, Active Tracks, Active %
   - Statistics: Mean Speed, Max Speed, Mean Straightness, Mean Path Length
   - Track classification scatter plot (speed vs straightness)

---

## Error Handling Tests

### Test 1: No Data Loaded
1. Start app without loading data
2. Navigate to any analysis tab
3. **Expected**: Warning message "No track data loaded. Please upload track data first."

### Test 2: Missing Module
1. Rename `advanced_tracking.py` temporarily
2. Navigate to Advanced Tracking tab
3. **Expected**: Warning "Advanced tracking module is not available."
4. Restore file

### Test 3: Invalid Parameters
1. Load data with very short tracks (<5 frames)
2. Run Changepoint Detection
3. **Expected**: Either skip short tracks or show informative error with traceback

---

## Session State Verification

After running any analysis, verify results are stored:

```python
# In Streamlit console or debug mode
import streamlit as st

# Check stored results
print(st.session_state.analysis_results.keys())
# Expected keys: 'changepoints', 'correlative', 'particle_filter', 
#                'intensity_movement', 'polymer_physics', 'active_transport'
```

---

## Performance Benchmarks

| Analysis | Expected Time (100 tracks, 50 frames) | Notes |
|----------|---------------------------------------|-------|
| Changepoint Detection | 5-10 seconds | Depends on window size |
| Correlative Analysis | 3-5 seconds | Fast for typical datasets |
| Particle Filter | 10-20 seconds | Scales with n_particles |
| Bayesian Refinement | 2-5 seconds | Fast, mostly numpy operations |
| Intensity Correlation | 3-7 seconds | Depends on lag_range |
| Intensity Classification | 5-10 seconds | K-means clustering |
| Polymer Physics | 8-15 seconds | MSD calculation + fitting |
| Active Transport | 5-10 seconds | Motion parameter calculation |

---

## Common Issues & Solutions

### Issue 1: "No intensity channels detected"
**Solution**: Load data with intensity columns (e.g., from Volocity XML or MetaMorph MVD2 files)

### Issue 2: "MSD calculation failed"
**Solution**: Ensure tracks have at least 10 frames. Filter short tracks before analysis.

### Issue 3: Module import errors
**Solution**: Verify all backend modules exist:
```powershell
ls *.py | Select-String -Pattern "changepoint_detection|correlative_analysis|advanced_tracking|intensity_analysis"
```

### Issue 4: Slow performance
**Solution**: 
- Limit number of tracks analyzed (use first 50-100)
- Reduce parameter ranges (e.g., lower n_particles, lag_range)
- Check for infinite loops in error messages

---

## Success Criteria Checklist

- [ ] Changepoint Detection runs without nested button issue
- [ ] Changepoint Detection displays motion segments table
- [ ] Correlative Analysis runs without function error
- [ ] Correlative Analysis displays correlation plot
- [ ] Advanced Tracking tab appears in navigation
- [ ] Particle Filter produces refined tracks
- [ ] Bayesian Refinement shows adjustment statistics
- [ ] Intensity Analysis tab appears in navigation
- [ ] Intensity-Movement Correlation displays heatmap
- [ ] Intensity Behavior Classification shows cluster distribution
- [ ] Polymer Physics runs and displays model fit
- [ ] Active Transport shows classification scatter plot
- [ ] All analyses store results in session state
- [ ] Error messages include tracebacks for debugging
- [ ] No Python syntax errors in console

---

## Automated Test Command (Optional)

If `test_app_logic.py` or `test_comprehensive.py` exist:

```powershell
python test_comprehensive.py
```

Or with pytest:

```powershell
python -m pytest tests/test_app_logic.py -v
```

---

## Final Verification

After testing all features, verify the complete workflow:

1. **Load Data** → Data Loading page
2. **Run Diffusion Analysis** → Analysis page → Diffusion tab
3. **Run Changepoint Detection** → Advanced Analysis → Changepoint Detection
4. **Run Intensity Analysis** → Advanced Analysis → Intensity Analysis
5. **Generate Report** → Report Generation page
   - Select all analyses
   - Click "Generate Enhanced Report"
   - Verify all results appear in report

**Expected**: Complete report with results from all analyses, no missing sections.

---

## Need Help?

**Check logs**:
```powershell
cat debug.log
```

**Streamlit console**: Look for error messages in terminal where `streamlit run` was executed

**Documentation**:
- Main instructions: `.github/copilot-instructions.md`
- Complete refactoring report: `APP_REFACTORING_COMPLETE.md`
