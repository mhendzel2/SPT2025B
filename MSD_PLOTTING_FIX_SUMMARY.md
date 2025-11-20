# MSD Plotting Fix - Summary

## Issue
Some analysis functions (e.g., MSD plots, diffusion analysis) were not displaying properly in generated reports. The plots appeared empty or showed "No data" errors.

## Root Cause
Inconsistent data structure between analysis functions and visualization functions:

- **analyze_diffusion()** returns: `{'success': True, 'result': {'msd_data': ..., 'track_results': ..., 'ensemble_results': ...}}`
- **analyze_motion()** returns: `{'success': True, 'msd_data': ..., 'track_results': ..., 'ensemble_results': ...}`

The visualization functions (_plot_diffusion, etc.) were expecting the flat structure (like motion) but getting nested structure (like diffusion).

## Solution

### 1. Added Helper Function (`_extract_analysis_data`)
Created a utility function to normalize both data structures:

```python
def _extract_analysis_data(self, result):
    """
    Extract data from analysis results, handling both nested and flat structures.
    
    Some analysis functions return {'success': True, 'result': {...data...}}
    while others return {'success': True, ...data...} directly.
    This normalizes the structure.
    """
    if not result.get('success', False):
        return result
    
    # If there's a nested 'result' key, extract it
    if 'result' in result and isinstance(result['result'], dict):
        # Create a new dict combining top-level metadata with nested data
        extracted = result['result'].copy()
        extracted['success'] = result['success']
        if 'error' in result:
            extracted['error'] = result['error']
        return extracted
    
    # Otherwise return as-is (already flat)
    return result
```

### 2. Updated `_plot_diffusion()` Function
Modified to use the helper function:

```python
def _plot_diffusion(self, result):
    """Full implementation for diffusion visualization."""
    try:
        from visualization import _empty_fig
        import io

        if not result.get('success', False):
            return _empty_fig("Diffusion analysis failed.")

        # Extract nested result data if present - FIXED
        data = self._extract_analysis_data(result)
        
        # Parse track_results if it's a string representation of DataFrame
        track_results = data.get('track_results', None)
        # ... rest of function uses 'data' instead of 'result'
```

### 3. Updated Data References
Changed all references from `result.get(...)` to `data.get(...)` in `_plot_diffusion`:
- Line ~1052: `track_results = data.get('track_results', None)` 
- Line ~1062: `msd_data = data.get('msd_data', None)`
- Line ~1150: `ensemble = data.get('ensemble_results', {})`

## Files Modified
- `enhanced_report_generator.py`:
  - Added `_extract_analysis_data()` helper (lines ~920-940)
  - Updated `_plot_diffusion()` to use helper (lines ~1035-1270)

## Testing
Created comprehensive test: `test_msd_plotting_fix.py`

Test results:
```
✓ Created 5 tracks with 125 points
✓ Analysis success: True
✓ MSD data shape: (50, 4) with columns: ['track_id', 'lag_time', 'msd', 'n_points']
✓ Track results shape: (5, 8) 
✓ _extract_analysis_data correctly extracts nested data
✓ Plot created successfully with 8 traces
✓ Motion analysis also works (flat structure)
```

## Impact
- ✅ MSD plots now display correctly in reports
- ✅ Diffusion coefficient histograms populate with data
- ✅ Motion type classification bar charts show counts
- ✅ Summary statistics tables display properly
- ✅ Backward compatible with both nested and flat data structures

## Future Recommendations
1. **Standardize return structure**: All analysis functions should follow the same pattern (either all nested or all flat)
2. **Apply helper to other plot functions**: Motion, clustering, microrheology, etc. could benefit from the same normalization
3. **Add type hints**: Document expected structure in function signatures
4. **Unit tests**: Add tests for all visualization functions to catch similar issues

## Related Functions That May Benefit
These visualization functions could use the same pattern:
- `_plot_motion()`
- `_plot_clustering()` 
- `_plot_anomalies()`
- `_plot_microrheology()`
- `_plot_creep_compliance()`
- `_plot_relaxation_modulus()`
- `_plot_intensity()`
- `_plot_polymer_physics()`
- `_plot_statistical_tests()`
- `_plot_energy_landscape()`

## Status
✅ **FIXED AND TESTED** - MSD plots now render correctly in generated reports.
