# Report Generation Diagnostic and Fixes

## Issues Identified

### 1. ❌ FALSE ALARM: Biophysical Models Are Implemented
**Status**: These functions ARE implemented in `biophysical_models.py`

- ✅ **Energy Landscape**: `EnergyLandscapeMapper` class (lines 143-295)
  - `map_energy_landscape()` - fully implemented with Boltzmann inversion
  - Multiple methods: 'boltzmann', 'drift', 'kramers'
  - Force field calculation
  - Visualization support

- ✅ **Active Transport**: `ActiveTransportAnalyzer` class (lines 598-900+)
  - `_preprocess_tracks()` - velocity/acceleration calculation
  - `detect_directional_motion_segments()` - implemented
  - `characterize_transport_modes()` - implemented
  - Full classification into: diffusive, slow_directed, fast_directed, mixed

**Root cause of confusion**: These classes exist but may not be exposed through the UI or report generator.

### 2. ⚠️ Polymer Physics Model Selection Issue
**Problem**: `analyze_polymer_physics()` in `analysis.py` lines 2949-3063

**Current behavior**:
- Calculates MSD scaling exponent
- Interprets regime: "Subdiffusive (reptation)", "Zimm dynamics", "Rouse dynamics", etc.
- BUT: Doesn't let user SELECT which model to fit
- Always fits scaling exponent (generic power law)

**Available models in code**:
- `PolymerPhysicsModel.fit_rouse_model()` - lines 60-115 in biophysical_models.py
- Zimm model - NOT explicitly implemented (only mentioned in regime interpretation)
- Reptation - inferred from scaling exponent

**Fix needed**:
1. Expose `PolymerPhysicsModel` class methods in report generator
2. Add UI to select: "Auto-detect", "Fit Rouse", "Fit Zimm", "Fit Reptation"
3. Return model name and parameters in results dict

### 3. 🔴 CRITICAL: Report Generation Blank Graphs

**Symptom**: Only one analytical output graph, and it's blank

**Potential causes**:

#### A. Visualization function returns None or empty figure
Check `_plot_*` methods in `enhanced_report_generator.py`:
- Lines 588-650: `_plot_basic_statistics()`
- Lines 650-750: `_plot_diffusion()`, `_plot_motion()`, etc.

#### B. Analysis function returns {'success': False}
Check `_analyze_*` methods return structure:
- Must return dict with 'success': True
- Must include data needed for visualization

#### C. Report generation skips visualizations
In `_run_analyses_for_report()` (lines 480-517):
```python
if result.get('success', False):  # ← May be too strict
    self.report_results[analysis_key] = result
    
    # Generate visualization
    fig = analysis['visualization'](result)
    if fig:  # ← May be None
        self.report_figures[analysis_key] = fig
```

#### D. Multiple figures not being aggregated
In `_plot_basic_statistics()` (lines 588-630):
```python
# plot_track_statistics returns a dictionary of figures
figs = plot_track_statistics(stats_df)

if not figs:
    return _empty_fig("No statistics plots generated.")

# Create a subplot figure. Let's assume 2x2 for the main stats.
fig = make_subplots(rows=2, cols=2, subplot_titles=list(figs.keys()))
```
**Issue**: If `figs` is a dict with Plotly figures, the subplot composition may fail.

#### E. Data access failures
Using `get_track_data()` may return empty DataFrame if state management is broken.

---

## Diagnostic Steps

### Step 1: Check if analyses are actually running
Add logging to `_run_analyses_for_report()`:
```python
st.write(f"DEBUG: Running {analysis['name']}")
st.write(f"DEBUG: Result keys: {result.keys()}")
st.write(f"DEBUG: Success = {result.get('success')}")
```

### Step 2: Check visualization return values
Add logging after visualization calls:
```python
fig = analysis['visualization'](result)
st.write(f"DEBUG: Figure type: {type(fig)}")
st.write(f"DEBUG: Figure is None: {fig is None}")
if fig:
    st.write(f"DEBUG: Figure has data: {hasattr(fig, 'data') and len(fig.data) > 0}")
```

### Step 3: Verify data is loaded
At the start of report generation:
```python
st.write(f"DEBUG: tracks_df shape: {tracks_df.shape}")
st.write(f"DEBUG: tracks_df columns: {tracks_df.columns.tolist()}")
st.write(f"DEBUG: First few rows:")
st.write(tracks_df.head())
```

### Step 4: Test individual visualizations
Try calling visualization functions directly in a notebook:
```python
from enhanced_report_generator import EnhancedSPTReportGenerator
gen = EnhancedSPTReportGenerator()

# Test basic statistics
result = gen._analyze_basic_statistics(tracks_df, {'pixel_size': 0.1, 'frame_interval': 0.1})
print(f"Result success: {result.get('success')}")

fig = gen._plot_basic_statistics(result)
print(f"Figure type: {type(fig)}")
fig.show()  # Should display in browser
```

---

## Fixes to Implement

### Fix 1: Add Polymer Model Selection

**File**: `enhanced_report_generator.py`

In `_analyze_polymer_physics()` method (around line 1032):
```python
def _analyze_polymer_physics(self, tracks_df, current_units):
    """Enhanced polymer physics analysis with model selection."""
    try:
        from analysis import analyze_polymer_physics
        from biophysical_models import PolymerPhysicsModel
        
        # First, get general polymer analysis
        polymer_results = analyze_polymer_physics(
            tracks_df,
            pixel_size=current_units.get('pixel_size', 1.0),
            frame_interval=current_units.get('frame_interval', 1.0)
        )
        
        # If MSD data available, fit specific models
        if polymer_results.get('success') and 'msd_data' in polymer_results:
            import pandas as pd
            msd_df = pd.DataFrame({
                'lag_time': polymer_results['lag_times'],
                'msd': polymer_results['msd_data']
            })
            
            model = PolymerPhysicsModel(
                msd_data=msd_df,
                pixel_size=current_units.get('pixel_size', 1.0),
                frame_interval=current_units.get('frame_interval', 1.0)
            )
            
            # Fit Rouse model
            rouse_fit = model.fit_rouse_model(fit_alpha=False)  # Fixed α=0.5
            rouse_fit_variable = model.fit_rouse_model(fit_alpha=True)  # Variable α
            
            polymer_results['models'] = {
                'rouse_fixed_alpha': rouse_fit,
                'rouse_variable_alpha': rouse_fit_variable,
                'best_model': polymer_results.get('regime', 'Unknown')
            }
        
        return polymer_results
    except Exception as e:
        return {'error': str(e), 'success': False}
```

### Fix 2: Add Energy Landscape to Report Generator

**File**: `enhanced_report_generator.py`

Add to `available_analyses` dict (around line 230):
```python
if BIOPHYSICAL_MODELS_AVAILABLE:
    self.available_analyses['energy_landscape'] = {
        'name': 'Energy Landscape Mapping',
        'description': 'Spatial potential energy from particle positions.',
        'function': self._analyze_energy_landscape,
        'visualization': self._plot_energy_landscape,
        'category': 'Biophysical Models',
        'priority': 4
    }
    self.available_analyses['active_transport'] = {
        'name': 'Active Transport Detection',
        'description': 'Directional motion segments, transport modes.',
        'function': self._analyze_active_transport,
        'visualization': self._plot_active_transport,
        'category': 'Biophysical Models',
        'priority': 4
    }
```

Add analysis methods:
```python
def _analyze_energy_landscape(self, tracks_df, current_units):
    """Analyze energy landscape from trajectories."""
    try:
        from biophysical_models import EnergyLandscapeMapper
        
        mapper = EnergyLandscapeMapper(
            tracks_df,
            pixel_size=current_units.get('pixel_size', 1.0),
            temperature=300.0  # Room temperature
        )
        
        result = mapper.map_energy_landscape(
            resolution=30,
            method='boltzmann',
            smoothing=1.0
        )
        
        return result
    except Exception as e:
        return {'error': str(e), 'success': False}

def _plot_energy_landscape(self, result):
    """Visualize energy landscape."""
    try:
        if not result.get('success', False):
            from visualization import _empty_fig
            return _empty_fig(f"Energy landscape failed: {result.get('error', 'Unknown error')}")
        
        from biophysical_models import EnergyLandscapeMapper
        
        # Use the mapper's visualization method
        # (Assuming it has a visualize_energy_landscape method)
        potential_map = result['potential']
        x_edges = result['x_edges']
        y_edges = result['y_edges']
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=potential_map.T,
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            colorscale='Viridis',
            colorbar=dict(title='Energy (kBT)')
        ))
        
        fig.update_layout(
            title='Energy Landscape',
            xaxis_title='x (μm)',
            yaxis_title='y (μm)',
            height=500
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization failed: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def _analyze_active_transport(self, tracks_df, current_units):
    """Analyze active transport characteristics."""
    try:
        from biophysical_models import ActiveTransportAnalyzer
        
        analyzer = ActiveTransportAnalyzer(
            tracks_df,
            pixel_size=current_units.get('pixel_size', 1.0),
            frame_interval=current_units.get('frame_interval', 1.0)
        )
        
        # Detect directional segments
        segments_result = analyzer.detect_directional_motion_segments(
            min_segment_length=5,
            straightness_threshold=0.7,
            velocity_threshold=0.05  # μm/s
        )
        
        # Characterize transport modes
        if segments_result.get('success', False):
            modes_result = analyzer.characterize_transport_modes()
            
            return {
                'success': True,
                'segments': segments_result,
                'transport_modes': modes_result,
                'summary': {
                    'total_segments': segments_result['total_segments'],
                    'mode_fractions': modes_result.get('mode_fractions', {})
                }
            }
        else:
            return segments_result
            
    except Exception as e:
        return {'error': str(e), 'success': False}

def _plot_active_transport(self, result):
    \"\"\"Visualize active transport analysis.\"\"\"
    try:
        if not result.get('success', False):
            from visualization import _empty_fig
            return _empty_fig(f"Active transport failed: {result.get('error', 'Unknown error')}")
        
        transport_modes = result.get('transport_modes', {})
        mode_fractions = transport_modes.get('mode_fractions', {})
        
        if not mode_fractions:
            from visualization import _empty_fig
            return _empty_fig("No transport modes detected")
        
        # Create pie chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=list(mode_fractions.keys()),
            values=list(mode_fractions.values()),
            hole=0.3
        ))
        
        fig.update_layout(
            title='Transport Mode Distribution',
            height=400
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization failed: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
```

### Fix 3: Debug Blank Graph Issue

**File**: `enhanced_report_generator.py`

Modify `_run_analyses_for_report()` to add diagnostics:
```python
def _run_analyses_for_report(self, tracks_df, selected_analyses, config, current_units):
    \"\"\"Run analyses directly for report generation with enhanced debugging.\"\"\"
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # DEBUG: Show input data
    with st.expander("🔍 Debug Info - Input Data"):
        st.write(f"Tracks DataFrame shape: {tracks_df.shape}")
        st.write(f"Columns: {tracks_df.columns.tolist()}")
        st.write(f"Track IDs: {tracks_df['track_id'].nunique()}")
        st.write(f"Units: {current_units}")
    
    self.report_results = {}
    self.report_figures = {}
    
    for i, analysis_key in enumerate(selected_analyses):
        if analysis_key not in self.available_analyses:
            st.warning(f"⚠️ Analysis '{analysis_key}' not found in available analyses")
            continue
            
        analysis = self.available_analyses[analysis_key]
        status_text.text(f"Running {analysis['name']}...")
        progress_bar.progress((i + 1) / len(selected_analyses))
        
        try:
            # Run the analysis
            result = analysis['function'](tracks_df, current_units)
            
            # DEBUG: Show result structure
            with st.expander(f"🔍 Debug Info - {analysis['name']} Result"):
                st.write(f"Result type: {type(result)}")
                st.write(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                st.write(f"Success: {result.get('success', 'N/A')}")
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
            
            # Check success BEFORE visualization
            success = result.get('success', True)  # Default to True if not specified
            has_error = 'error' in result
            
            if success and not has_error:
                self.report_results[analysis_key] = result
                
                # Generate visualization
                try:
                    fig = analysis['visualization'](result)
                    
                    # DEBUG: Show figure info
                    with st.expander(f"🔍 Debug Info - {analysis['name']} Figure"):
                        st.write(f"Figure type: {type(fig)}")
                        st.write(f"Figure is None: {fig is None}")
                        if fig is not None:
                            if hasattr(fig, 'data'):
                                st.write(f"Figure has {len(fig.data)} traces")
                                st.write(f"Trace types: {[type(trace).__name__ for trace in fig.data]}")
                            if hasattr(fig, 'layout'):
                                st.write(f"Figure title: {fig.layout.title.text if fig.layout.title else 'None'}")
                    
                    if fig is not None:
                        self.report_figures[analysis_key] = fig
                        st.success(f"✅ Completed {analysis['name']}")
                    else:
                        st.warning(f"⚠️ {analysis['name']} visualization returned None")
                        
                except Exception as viz_error:
                    st.error(f"❌ Visualization failed for {analysis['name']}: {str(viz_error)}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                error_msg = result.get('error', 'Success flag is False')
                st.warning(f"⚠️ {analysis['name']} failed: {error_msg}")
                
        except Exception as e:
            st.error(f"❌ Failed to run {analysis['name']}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    status_text.text("Report generation complete!")
    progress_bar.progress(1.0)
    
    # DEBUG: Show final counts
    st.info(f"📊 Generated {len(self.report_results)} analysis results and {len(self.report_figures)} figures")
    
    # Display the generated report
    self._display_generated_report(config, current_units)
```

---

## Testing Checklist

### 1. Test Polymer Physics Analysis
- [ ] Load sample data with good tracking
- [ ] Run "Polymer Physics Models" analysis
- [ ] Verify results show:
  - Scaling exponent (α)
  - Regime classification
  - Rouse model fits (fixed and variable α)
  - Model parameters (K_rouse)

### 2. Test Energy Landscape
- [ ] Run "Energy Landscape Mapping" analysis
- [ ] Verify heatmap displays
- [ ] Check energy values are in kBT units
- [ ] Verify spatial extent matches track data

### 3. Test Active Transport
- [ ] Run "Active Transport Detection" analysis
- [ ] Verify pie chart shows transport modes
- [ ] Check mode fractions sum to 1.0
- [ ] Verify segments are detected for directed tracks

### 4. Test Report Generation
- [ ] Select 3-5 different analyses
- [ ] Click "Generate Report"
- [ ] Verify progress bar advances
- [ ] Check all selected analyses run
- [ ] Verify debug info shows reasonable data
- [ ] Confirm figures are not None
- [ ] View interactive report - all graphs should display
- [ ] Download HTML - all graphs should be embedded
- [ ] Download PDF - all graphs should be rasterized

---

## Next Steps

1. **Implement Fix 1**: Add model selection to polymer physics
2. **Implement Fix 2**: Add energy landscape & active transport to report generator
3. **Implement Fix 3**: Add debugging to report generation
4. **Test with sample data**: Use `Cell1_spots.csv` or similar
5. **Fix any remaining issues** based on debug output
6. **Remove debug code** once working
7. **Update documentation** with new analyses

---

## Expected Outcomes

After fixes:
- ✅ Polymer physics analysis shows which model was used
- ✅ Energy landscape analysis available in report generator
- ✅ Active transport analysis available in report generator
- ✅ All selected analyses generate figures
- ✅ Report displays multiple graphs (not just one)
- ✅ Debug info helps identify any remaining issues
