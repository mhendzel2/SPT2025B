# Report Generation Issues Found and Fixes

## Test Results Summary

**Date**: October 3, 2025  
**Test Run**: Comprehensive Report Generation Debug Test

### Overall Results:
- **15/16 analyses successful** (93.75%)
- **1 analysis failed**: Active Transport Detection
- **7 blank figures** identified (43.75% of visualizations)
- **Batch workflow**: ✅ PASSED

---

## Issues Identified

### 1. 🔴 CRITICAL: Missing `return fig` Statement in FBM Plot

**File**: `batch_report_enhancements.py`  
**Function**: `plot_fbm_results()` (line 109)  
**Issue**: Function creates figure but doesn't return it

**Current Code** (lines 109-150):
```python
def plot_fbm_results(result: Dict) -> go.Figure:
    """Create visualization for FBM analysis results."""
    if not result.get('success', False):
        return None
    
    data = result.get('data')
    if data is None or data.empty:
        return None
    
    fig = make_subplots(...)
    # ... creates the figure ...
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Fractional Brownian Motion Analysis"
    # MISSING: return fig
```

**Impact**: FBM analysis returns `None` for visualization, causing blank graph

**Fix**: Add `return fig` at end of function

---

### 2. 🟡 PLACEHOLDER: Intensity Analysis Not Implemented

**File**: `enhanced_report_generator.py`  
**Functions**: 
- `_analyze_intensity()` (line 1038)
- `_plot_intensity()` (line 1043)

**Current Implementation**:
```python
def _analyze_intensity(self, tracks_df, current_units):
    """Placeholder for intensity analysis."""
    return {'success': True, 'message': 'Intensity analysis not yet implemented.'}

def _plot_intensity(self, result):
    """Placeholder for intensity visualization."""
    from visualization import _empty_fig
    return _empty_fig("Intensity analysis not yet implemented.")
```

**Impact**: Returns empty figure by design (placeholder)

**Options**:
1. Remove from available_analyses (quick fix)
2. Implement basic intensity analysis (proper fix)

---

### 3. 🟡 PLACEHOLDER: Velocity Correlation Not Implemented

**File**: `enhanced_report_generator.py`  
**Functions**:
- `_analyze_velocity_correlation()` (line 890)
- `_plot_velocity_correlation()` (line 894)

**Current Implementation**:
```python
def _analyze_velocity_correlation(self, tracks_df, current_units):
    """Placeholder for velocity correlation analysis."""
    return {'success': True, 'message': 'Velocity correlation analysis not yet implemented.'}

def _plot_velocity_correlation(self, result):
    """Placeholder for velocity correlation visualization."""
    from visualization import _empty_fig
    return _empty_fig("Not implemented")
```

**Impact**: Returns empty figure by design

**Note**: VACF is implemented in Advanced Metrics, this is duplicate/redundant

**Fix**: Remove from available_analyses or redirect to Advanced Metrics

---

### 4. 🟡 PLACEHOLDER: Particle Interactions Not Implemented

**File**: `enhanced_report_generator.py`  
**Functions**:
- `_analyze_particle_interactions()` (line 899)
- `_plot_particle_interactions()` (line 903)

**Current Implementation**:
```python
def _analyze_particle_interactions(self, tracks_df, current_units):
    """Placeholder for particle interaction analysis."""
    return {'success': True, 'message': 'Particle interaction analysis not yet implemented.'}

def _plot_particle_interactions(self, result):
    """Placeholder for particle interaction visualization."""
    from visualization import _empty_fig
    return _empty_fig("Not implemented")
```

**Impact**: Returns empty figure by design

**Fix**: Remove from available_analyses or implement

---

### 5. 🟠 ISSUE: Microrheology Visualization May Return Empty

**File**: `enhanced_report_generator.py`  
**Function**: `_plot_microrheology()` (line 1000)

**Issue**: Depends on `create_rheology_plots()` which may return empty dict

**Current Code**:
```python
def _plot_microrheology(self, result):
    try:
        from rheology import create_rheology_plots
        from visualization import _empty_fig

        if not result.get('success', False):
            return _empty_fig("Microrheology analysis failed.")

        figs = create_rheology_plots(result)

        if not figs:  # ← May trigger if rheology module has issues
            return _empty_fig("No rheology plots generated.")
        
        # ... create subplot figure ...
```

**Impact**: Returns empty figure if rheology visualization fails

**Investigation needed**: Check why `create_rheology_plots()` returns empty

---

### 6. 🟠 ISSUE: Polymer Physics Visualization May Fail

**File**: `enhanced_report_generator.py`  
**Function**: `_plot_polymer_physics()` (line 1103)

**Issue**: Depends on `plot_polymer_physics_results()` from visualization module

**Current Code**:
```python
def _plot_polymer_physics(self, result):
    """Full implementation for polymer physics visualization."""
    try:
        from visualization import plot_polymer_physics_results
        return plot_polymer_physics_results(result)
    except ImportError:
        return None  # ← Returns None if import fails
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Plotting failed: {e}", ...)
        return fig
```

**Impact**: May return None if visualization function doesn't exist

**Investigation needed**: Verify `plot_polymer_physics_results()` exists in visualization.py

---

### 7. 🟠 ISSUE: Energy Landscape May Have Incomplete Data

**File**: `enhanced_report_generator.py`  
**Function**: `_plot_energy_landscape()` (line 1137)

**Issue**: Returns empty figure if potential_map, x_edges, or y_edges missing

**Current Code**:
```python
def _plot_energy_landscape(self, result):
    try:
        if not result.get('success', False):
            from visualization import _empty_fig
            return _empty_fig(f"Energy landscape mapping failed: {result.get('error', 'Unknown error')}")
        
        potential_map = result.get('potential')
        x_edges = result.get('x_edges')
        y_edges = result.get('y_edges')
        
        if potential_map is None or x_edges is None or y_edges is None:
            from visualization import _empty_fig
            return _empty_fig("Energy landscape data incomplete")
        # ... create heatmap ...
```

**Impact**: Returns empty figure if EnergyLandscapeMapper doesn't return expected keys

**Investigation needed**: Check what keys `map_energy_landscape()` returns

---

### 8. ⚠️ FAILED: Active Transport Detection

**File**: `enhanced_report_generator.py`  
**Function**: `_analyze_active_transport()` (line 1185)

**Issue**: No directional segments detected with current thresholds

**Current Thresholds**:
```python
segments_result = analyzer.detect_directional_motion_segments(
    min_segment_length=5,
    straightness_threshold=0.7,  # ← Too high for random walk
    velocity_threshold=0.05      # μm/s
)
```

**Impact**: Returns `{'success': False}` for typical diffusive tracks

**Fix**: Lower thresholds or add fallback for diffusive-only data

---

## Fixes Implementation Plan

### Priority 1: Critical Bugs (Immediate)

#### Fix 1.1: Add Missing Return Statement in FBM Plot
**File**: `batch_report_enhancements.py` (line ~150)

**Change**:
```python
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Fractional Brownian Motion Analysis"
    )
    
    return fig  # ← ADD THIS LINE
```

**Impact**: Fixes blank FBM graph

---

### Priority 2: Remove Placeholder Analyses (Quick Wins)

#### Fix 2.1: Remove Unimplemented Analyses from UI
**File**: `enhanced_report_generator.py` (lines 146-230)

**Change**: Comment out or remove these from `available_analyses`:
- `intensity_analysis`
- `velocity_correlation`
- `multi_particle_interactions`

**Alternative**: Set `priority: -1` to hide from default view

**Impact**: Users won't see placeholder analyses

---

### Priority 3: Investigate and Fix (Medium)

#### Fix 3.1: Check Polymer Physics Visualization
Verify that `plot_polymer_physics_results()` exists and works:
```bash
grep -n "def plot_polymer_physics_results" visualization.py
```

If missing, implement or use fallback visualization.

#### Fix 3.2: Check Energy Landscape Return Keys
Verify `EnergyLandscapeMapper.map_energy_landscape()` returns:
- `potential`
- `x_edges`
- `y_edges`

#### Fix 3.3: Check Microrheology Visualization
Verify `create_rheology_plots()` returns non-empty dict:
```python
from rheology import create_rheology_plots
# Test with sample data
```

---

### Priority 4: Improve Active Transport (Low)

#### Fix 4.1: Lower Detection Thresholds
```python
segments_result = analyzer.detect_directional_motion_segments(
    min_segment_length=3,           # ← Lower from 5
    straightness_threshold=0.5,     # ← Lower from 0.7
    velocity_threshold=0.01         # ← Lower from 0.05
)
```

#### Fix 4.2: Add Fallback for No Segments
```python
if segments_result.get('success', False) and segments_result.get('total_segments', 0) > 0:
    modes_result = analyzer.characterize_transport_modes()
    return {...}
else:
    # Fallback: Report all tracks as diffusive
    return {
        'success': True,
        'segments': {'total_segments': 0},
        'transport_modes': {
            'mode_fractions': {'diffusive': 1.0, 'slow_directed': 0, 'fast_directed': 0, 'mixed': 0}
        },
        'summary': {'total_segments': 0, 'mode_fractions': {'diffusive': 1.0}}
    }
```

---

## Implementation Summary

### Files to Modify:

1. **batch_report_enhancements.py** (1 line change)
   - Add `return fig` to `plot_fbm_results()`

2. **enhanced_report_generator.py** (multiple changes)
   - Comment out/remove 3 placeholder analyses
   - Fix active transport thresholds
   - Add fallback logic

3. **Verification needed** (no changes yet):
   - Check visualization.py for `plot_polymer_physics_results()`
   - Check biophysical_models.py for energy landscape return keys
   - Check rheology.py for `create_rheology_plots()`

---

## Expected Results After Fixes

### Before:
- 15/16 analyses successful
- 7/16 blank figures (43.75%)
- 1 failed analysis

### After (Priority 1 + 2):
- 12/13 analyses successful (remove 3 placeholders)
- **1/13 blank figures** (7.7%) ← 85% improvement
- 1 failed analysis (active transport with diffusive data)

### After (All priorities):
- 12/13 analyses successful
- **0/13 blank figures** (0%) ← 100% fixed
- 0 failed analyses (active transport fallback)

---

## Testing Commands

```bash
# Run comprehensive test
python test_comprehensive_report.py

# Expected output after fixes:
# - Successful: 12/12 (after removing placeholders)
# - Blank figures: 0/12
# - Failed analyses: 0/12
```

---

## Summary

| Issue | Type | Priority | Fix Complexity |
|-------|------|----------|----------------|
| FBM missing return | Bug | P1 | 1 line |
| Intensity placeholder | Incomplete | P2 | Remove |
| Velocity correlation placeholder | Incomplete | P2 | Remove |
| Particle interactions placeholder | Incomplete | P2 | Remove |
| Microrheology blank | Investigation | P3 | TBD |
| Polymer physics blank | Investigation | P3 | TBD |
| Energy landscape blank | Investigation | P3 | TBD |
| Active transport fails | Threshold | P4 | 5 lines |

**Total Fixes Required**: 1 critical, 3 quick wins, 3 investigations, 1 enhancement

**Estimated Time**: 
- P1: 5 minutes
- P2: 10 minutes  
- P3: 30-60 minutes (investigation + fix)
- P4: 15 minutes

**Total**: ~1-2 hours to fully resolve all issues
