# Subpopulation-Aware Batch Analysis Implementation

## Overview
Extended the Batch Analysis & Comparison functionality to automatically detect and handle subpopulations within conditions, treating each subpopulation as a separate dataset for comparative analysis.

## Changes Made

### 1. Fixed Data Flow for Subpopulation Detection (project_management.py)

**Modified `Condition.pool_tracks()` method** to preserve cell identifiers:
- Added `cell_id` column assignment using `f"cell_{idx}"` format for each file
- Added `source_file` column to track which file each track originated from
- Implemented in both parallel processing and fallback implementations
- Preserves existing cell_id if already present (for subpopulation workflows)

```python
# Add cell identifiers to each dataframe before pooling
for idx, (df, file_info) in enumerate(zip(dataframes, self.files)):
    if df is not None and not df.empty:
        if 'cell_id' not in df.columns:
            df['cell_id'] = f"cell_{idx}"
        if 'source_file' not in df.columns:
            file_name = file_info.get('file_name') or file_info.get('name', f'file_{idx}')
            df['source_file'] = file_name
```

### 2. Fixed Subpopulation Pooling Workflow (app.py)

**Modified Project Management "Pool by subpopulation" section** (line ~2554):
- Changed from re-pooling files (which lost cell_id) to using pre-existing `result['pooled_tracks']`
- The pooled tracks from subpopulation detection already contain cell_id and subpopulation labels
- Eliminates the "cell_id column not found" error

```python
# Use the pooled tracks from subpopulation detection (already has cell_id and subpopulation labels)
pooled = result.get('pooled_tracks')

if pooled is not None and not pooled.empty and cell_df is not None:
    if 'cell_id' in pooled.columns and 'subpopulation' in pooled.columns:
        pooled['group'] = cond.name
        st.session_state.tracks_data = pooled
```

### 3. Enhanced Batch Analysis with Subpopulation Breakdown (app.py)

**Modified "Generate Individual Reports" button** (line ~2698):
- Checks if subpopulation results exist for each condition
- If subpopulations detected, splits pooled data by subpopulation ID
- Creates separate datasets named `"{condition_name} - Subpop {id}"`
- Falls back to normal pooling if no subpopulations detected

```python
for cond in conditions_to_analyze:
    has_subpop = ('subpopulation_results' in st.session_state and 
                 cond.id in st.session_state.subpopulation_results)
    
    if has_subpop:
        result = st.session_state.subpopulation_results[cond.id]
        
        if result.get('subpopulations_detected'):
            pooled_df = result.get('pooled_tracks')
            
            if pooled_df is not None and 'subpopulation' in pooled_df.columns:
                n_subpops = result['n_subpopulations']
                
                for subpop_id in range(n_subpops):
                    subpop_df = pooled_df[pooled_df['subpopulation'] == subpop_id].copy()
                    if not subpop_df.empty:
                        dataset_name = f"{cond.name} - Subpop {subpop_id}"
                        condition_datasets[dataset_name] = subpop_df
                continue
    
    # No subpopulations - pool normally
    pooled_df = cond.pool_tracks()
    condition_datasets[cond.name] = pooled_df
```

**Modified "Generate Comparative Report" button** (line ~2960):
- Identical subpopulation detection and splitting logic
- Ensures legacy comparative reports also respect subpopulations

### 4. Enhanced User Feedback

Added informative messages showing subpopulation breakdown:
```python
if subpop_info:
    st.success(f"✅ Pooled data from {len(conditions_to_analyze)} conditions ({len(condition_datasets)} datasets including subpopulations)")
    with st.expander("ℹ️ Subpopulation Breakdown", expanded=False):
        for cond_name, info in subpop_info.items():
            st.write(f"**{cond_name}:** {info['n_subpopulations']} subpopulations detected")
```

## Workflow Integration

### Complete Subpopulation Analysis Pipeline:

1. **Project Management → Detect Subpopulations**
   - Loads files and assigns `cell_id = f"cell_{idx}"`
   - Calculates per-cell features from track variation
   - Performs clustering (K-means, GMM, Hierarchical, DBSCAN)
   - Validates with silhouette score (threshold: 0.25)
   - Creates pooled tracks with cell_id and subpopulation labels
   - Stores in `st.session_state.subpopulation_results[cond.id]`

2. **Project Management → Pool & Load** (optional)
   - Uses pre-existing pooled tracks with subpopulation labels
   - Loads into main session state for single-condition analysis

3. **Batch Analysis & Comparison → Generate Reports**
   - Automatically detects if conditions have subpopulations
   - Splits each condition into separate subpopulation datasets
   - Treats each subpopulation as independent entity in comparisons
   - Names: `"Condition A - Subpop 0"`, `"Condition A - Subpop 1"`, etc.

## Example Usage

### Scenario: Comparing Cell Cycle Effects

**Setup:**
- Condition "Control": 20 cells, 2 subpopulations detected (G1 phase, S/G2 phase)
- Condition "Drug Treated": 18 cells, 3 subpopulations detected

**Batch Analysis Output:**
- Dataset 1: "Control - Subpop 0" (G1-like cells)
- Dataset 2: "Control - Subpop 1" (S/G2-like cells)
- Dataset 3: "Drug Treated - Subpop 0"
- Dataset 4: "Drug Treated - Subpop 1"
- Dataset 5: "Drug Treated - Subpop 2"

**Comparative Analysis:**
- Generates individual reports for each subpopulation
- Performs pairwise statistical comparisons (t-tests, Mann-Whitney)
- Creates comparison boxplots showing all 5 groups
- Identifies which drug-treated subpopulations differ from control subpopulations

## Benefits

1. **Automatic Detection**: No manual configuration needed - subpopulations detected automatically
2. **Transparent Workflow**: Clear indication when subpopulations are used vs. pooled data
3. **Backward Compatible**: Conditions without subpopulation analysis use normal pooling
4. **Consistent Naming**: Clear dataset labels (`"Condition - Subpop N"`)
5. **Full Analysis Pipeline**: Each subpopulation gets complete analysis (MSD, VACF, diffusion coefficients, etc.)

## Technical Details

### Cell ID Assignment Consistency
The system ensures consistent cell_id assignment:
- Subpopulation detection: `cell_id = f"cell_{file_idx}"`
- Pool tracks method: `cell_id = f"cell_{idx}"` (same format)
- Uses enumerate order on `cond.files` (consistent between calls)

### Data Storage
- **During subpopulation detection**: Pooled tracks stored at `result['pooled_tracks']`
- **During batch analysis**: Retrieved from session state, split by subpopulation column
- **Cell-level data**: Stored at `result['cell_level_data']` with per-cell features

### Validation
- Subpopulation detection requires minimum 5 cells per condition
- Minimum 5 tracks per cell for meaningful statistics
- Silhouette score > 0.25 required to consider subpopulations "detected"
- Tests 2-5 clusters, selects optimal based on silhouette score

## Future Enhancements

1. **Subpopulation Merging**: Allow user to merge similar subpopulations post-detection
2. **Cross-Condition Clustering**: Detect subpopulations across all conditions simultaneously
3. **Hierarchical Subpopulations**: Support nested subpopulation structure
4. **Feature Importance**: Show which features best distinguish subpopulations
5. **Longitudinal Analysis**: Track subpopulation stability across time-course experiments

---

**Date**: November 21, 2025
**Status**: ✅ Complete and Tested
**Related Docs**: 
- SUBPOPULATION_ANALYSIS_GUIDE.md
- COMPARATIVE_GROUP_ANALYSIS_IMPLEMENTATION.md
