# Comparative Group Analysis Implementation

## Overview
Added comprehensive functionality with **automatic subpopulation detection as the FIRST step** before pooling data. This workflow identifies heterogeneous cell populations within each condition, then intelligently pools data by subpopulation for accurate group comparisons.

## Updated Workflow

### 🔬 Step 1: Detect Subpopulations (NEW - REQUIRED FIRST STEP)
**Before pooling any data**, the system now:

1. **Analyzes single-cell heterogeneity** within each condition
2. **Detects distinct subpopulations** (e.g., cell cycle stages, metabolic states)
3. **Characterizes each subpopulation** with statistical metrics
4. **Labels cells** with subpopulation assignments

**Why This Matters:**
- Prevents inappropriate pooling of heterogeneous populations
- Reveals hidden biological structure masked by averaging
- Ensures accurate statistical comparisons between matched subpopulations
- Identifies treatment-induced population shifts

**Location:** Project Management → Each Condition → "Step 1: Detect Subpopulations"

**Button:** 🔬 Detect Subpopulations

**What Happens:**
- Automatically assigns cell IDs if missing (spatial clustering)
- Runs clustering analysis (K-means, GMM, or Hierarchical)
- Tests 2-4 cluster solutions, selects optimal
- Reports number of subpopulations detected
- Shows cell distribution across subpopulations

**Output:**
```
✓ Detected 2 subpopulations in 'Control'
📊 Analyzed 45 cells using gmm
Subpopulation Distribution:
- Subpop 0: 27 cells (60.0%)
- Subpop 1: 18 cells (40.0%)
```

### 📊 Step 2: Pool by Subpopulation
**After** subpopulation detection, choose pooling strategy:

**Option A: Pool by Subpopulation (RECOMMENDED)**
- Keeps subpopulation labels attached to each track
- Enables subgroup-level comparisons
- Preserves biological heterogeneity information
- Allows analysis of:
  - Overall group differences
  - Subpopulation-specific effects
  - Treatment-induced population shifts

**Option B: Pool All Together**
- Ignores subpopulation structure
- Traditional approach (not recommended if heterogeneity detected)
- Use only if condition appears homogeneous

**Location:** Project Management → Each Condition → "Step 2: Pool by Subpopulation"

**Button:** 📊 Pool & Load

**Output:**
```
✓ Loaded 1,234 tracks with subpopulation labels
Track distribution by subpopulation:
- Subpopulation 0: 742 tracks
- Subpopulation 1: 492 tracks
```

### ⚡ Quick Pool Option
**For experienced users only:** Skip subpopulation analysis if you're certain your population is homogeneous or want traditional bulk analysis.

**Button:** ⚡ Quick Pool (Skip Analysis)

## New Features

### 1. Individual Condition Reports
**Method:** `generate_condition_reports()` in `EnhancedSPTReportGenerator`

Generates complete analysis reports for each condition in a project:
- Pools track data from all files in a condition
- Runs selected analyses (diffusion, motion classification, etc.)
- Creates visualizations for each condition
- Caches results for performance

**Usage:**
```python
generator = EnhancedSPTReportGenerator(df, pixel_size=0.1, frame_interval=0.1)

condition_datasets = {
    'Control': control_df,
    'Treatment': treatment_df
}

results = generator.generate_condition_reports(
    condition_datasets,
    selected_analyses=['basic_statistics', 'diffusion_analysis'],
    pixel_size=0.1,
    frame_interval=0.1
)
```

### 2. Statistical Group Comparisons
**Method:** `_compare_conditions()` in `EnhancedSPTReportGenerator`

Performs pairwise statistical comparisons between all conditions:

**Metrics Compared:**
- Track lengths (frames)
- Displacements (μm)
- Velocities (μm/s)

**Statistical Tests:**
- Student's t-test (parametric)
- Mann-Whitney U test (non-parametric)
- Automatic significance detection (p < 0.05)

**Output:**
- Summary metrics table
- Pairwise comparison p-values
- Significance indicators
- Comparison boxplots

### 3. Comparison Visualizations
**Method:** `_plot_condition_comparisons()`

Creates 2×2 subplot figure with:
- **Track Lengths:** Boxplots comparing track durations
- **Displacements:** Boxplots of total displacement distributions
- **Velocities:** Boxplots of instantaneous velocity distributions
- **Sample Sizes:** Bar chart showing number of tracks and points per condition

### 4. Enhanced UI in Project Management

**Location:** Project Management tab → Batch Analysis & Comparison section

**New Buttons:**
- **📊 Generate Individual Reports** - Full analysis of each condition with comparisons
- **📈 Generate Comparative Report** - Legacy simplified comparison view

**Features:**
- ✓ Select specific analyses to run
- ✓ Grouped by category (Basic, Core Physics, Spatial, ML, etc.)
- ✓ Default selections (basic_statistics, diffusion_analysis)
- ✓ Progress tracking during pooling and analysis
- ✓ Error handling with detailed warnings
- ✓ Interactive expandable results per condition
- ✓ Statistical comparison tables
- ✓ Significance indicators (* = p < 0.05)
- ✓ Download options (JSON full report, CSV metrics)

## Testing Results

**Test Suite:** `test_comparative_reports.py`

✓ **Test 1: Condition Report Generation**
- Created 3 synthetic conditions (Control, Treatment A, Treatment B)
- Generated individual reports with 3 analyses each
- Performed statistical comparisons showing significant differences
- All pairwise comparisons computed correctly

✓ **Test 2: Pairwise Comparisons**
- Created 4 conditions (A, B, C, D)
- Expected 6 pairwise comparisons: C(4,2) = 6
- Generated exactly 6 comparisons: A_vs_B, A_vs_C, A_vs_D, B_vs_C, B_vs_D, C_vs_D

**Overall: 2/2 tests passed**

## Example Output

### Summary Metrics
```
Condition       | Mean Track | Mean Displacement | Mean Velocity
                | Length     | (μm)              | (μm/s)
----------------|------------|-------------------|---------------
Control         | 50.00      | 3.49              | 0.78
Treatment A     | 50.00      | 6.92              | 1.47
Treatment B     | 50.00      | 10.40             | 2.18
```

### Statistical Tests
```
Control vs Treatment A:
  - displacement: p=0.0000 (*) ✅ Significant
  - velocity: p=0.0000 (*) ✅ Significant

Control vs Treatment B:
  - displacement: p=0.0000 (*) ✅ Significant
  - velocity: p=0.0000 (*) ✅ Significant

Treatment A vs Treatment B:
  - displacement: p=0.0000 (*) ✅ Significant
  - velocity: p=0.0000 (*) ✅ Significant
```

## Complete Workflow Example

### Scenario: Comparing Control vs. Treatment

**1. Create Project with Two Conditions**
```
Project: "Drug Response Study"
├── Condition 1: "Control" (5 files)
└── Condition 2: "Treatment" (5 files)
```

**2. For Each Condition - Detect Subpopulations**

**Control Condition:**
```
Click: 🔬 Detect Subpopulations
Result: ✓ Detected 2 subpopulations
- Subpop 0 (G1 phase): 60% of cells, D=0.05 μm²/s
- Subpop 1 (S/G2 phase): 40% of cells, D=0.12 μm²/s
```

**Treatment Condition:**
```
Click: 🔬 Detect Subpopulations
Result: ✓ Detected 3 subpopulations
- Subpop 0 (G1 phase): 40% of cells, D=0.05 μm²/s
- Subpop 1 (S/G2 phase): 35% of cells, D=0.12 μm²/s
- Subpop 2 (Arrested): 25% of cells, D=0.02 μm²/s ← NEW!
```

**Key Insight:** Treatment induces a new arrested subpopulation (25% of cells)!

**3. Pool by Subpopulation**
```
For each condition:
  Select: "Pool by subpopulation"
  Click: 📊 Pool & Load
  Result: Tracks tagged with subpopulation labels
```

**4. Comparative Analysis**
Now you can:
- **Compare matched subpopulations**: Control Subpop-0 vs. Treatment Subpop-0
- **Analyze population shifts**: Treatment has 25% arrested cells vs. 0% in control
- **Subgroup-specific effects**: Does treatment affect G1 and S/G2 differently?

**5. Generate Reports**
```
Select both conditions
Choose analyses: 
  ☑ Basic Statistics
  ☑ Diffusion Analysis
  ☑ Motion Classification
Click: 📊 Generate Individual Reports
```

**Result:** Complete statistical comparison with subpopulation-aware analysis

## Workflow

1. **Create Project** with multiple conditions
2. **Upload Files** to each condition
3. **Select Conditions** to compare using checkboxes
4. **Choose Analyses** from expandable menu
5. **Click "Generate Individual Reports"**
6. **View Results:**
   - Summary table with status per condition
   - Expandable sections per condition with figures
   - Statistical comparison section with metrics table
   - Pairwise test results with p-values
   - Comparison boxplots
7. **Download:**
   - Full JSON report with all results
   - CSV metrics summary for further analysis

## Benefits

✓ **Automated:** No manual export/import between conditions
✓ **Comprehensive:** Full analysis suite on each condition
✓ **Statistical:** Rigorous pairwise comparisons with multiple tests
✓ **Visual:** Interactive boxplots for immediate insights
✓ **Reproducible:** JSON export preserves all parameters and results
✓ **Efficient:** Results caching prevents redundant computation
✓ **Robust:** Error handling at file, condition, and analysis levels

## Files Modified

1. **enhanced_report_generator.py**
   - Added `generate_condition_reports()` method
   - Added `_compare_conditions()` method
   - Added `_plot_condition_comparisons()` method

2. **app.py**
   - Enhanced Project Management batch analysis UI
   - Added analysis selection interface
   - Added comprehensive result display
   - Added download buttons for JSON and CSV

3. **test_comparative_reports.py** (new)
   - Comprehensive test suite
   - Synthetic data generation
   - Validation of all features
