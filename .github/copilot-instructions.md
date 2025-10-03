# SPT2025B AI Coding Agent Instructions

## Project Overview
SPT2025B is a Streamlit-based Single Particle Tracking analysis platform for biophysical research. It handles particle detection, tracking, diffusion analysis, microrheology, and motion classification with 25+ analysis modules.

## Critical Architecture Patterns

### Data Access - ALWAYS Use Utilities
**Never access `st.session_state` directly for track data.** Use `data_access_utils.py`:
```python
from data_access_utils import get_track_data, get_analysis_results, get_units

tracks_df, has_data = get_track_data()  # Returns (DataFrame, bool)
if not has_data:
    st.error("No data")
    return
```
The system has 3-level fallback: StateManager → primary keys (`tracks_df`, `tracks_data`) → legacy keys (`raw_tracks`, `track_data`).

### Analysis Function Contract
All analysis functions in `analysis.py` follow this signature:
```python
def analyze_X(tracks_df, pixel_size=1.0, frame_interval=1.0, **kwargs):
    return {
        'success': True/False,
        'data': {...},           # Raw results
        'summary': {...},        # Summary statistics
        'figures': {...}         # Plotly figure objects
    }
```
**Required DataFrame columns**: `track_id`, `frame`, `x`, `y` (z optional)

### State Management Hierarchy
```
StateManager (state_manager.py)
    ↓ manages
st.session_state
    ↓ accessed via
data_access_utils.py (use this)
```
Use `get_state_manager()` singleton for centralized state operations.

## Development Workflows

### Running the Application
```powershell
# Standard startup
streamlit run app.py --server.port 5000

# Via batch script (Windows)
.\start.bat  # Sets up venv, installs deps, runs app on port 8501
```

### Testing Strategy
```powershell
# Run pytest suite
python -m pytest tests/test_app_logic.py

# Run standalone test scripts (no pytest needed)
python test_functionality.py
python test_report_generation.py
python test_comprehensive.py
```
Test files pattern: Root has standalone `test_*.py` scripts; `tests/` has pytest modules.

### Adding New Analysis Module
1. **Implement** in `analysis.py` with standardized return dict
2. **Register** in `EnhancedSPTReportGenerator.available_analyses` (line ~200)
3. **Add visualization** returning plotly figure (not displaying directly)
4. **Test** with sample data (`Cell1_spots.csv` commonly used)
5. **Update** `constants.py` if new thresholds/defaults needed

## File Format Handlers
The system auto-detects formats via specialized handlers:
- `data_loader.py`: CSV/Excel (generic tracking data)
- `mvd2_handler.py`: MetaMorph MVD2 files
- `volocity_handler.py`: Volocity XML exports
- `special_file_handlers.py`: Imaris, MS2 spots formats

After loading, all data passes through `format_track_data()` to normalize to `track_id`, `frame`, `x`, `y` schema.

## Project Structure Conventions

### Configuration Files
- `constants.py`: All magic numbers, session state keys, thresholds (e.g., `DEFAULT_PIXEL_SIZE = 0.1`)
- `config.toml`: Streamlit + user-editable parameters
- `requirements.txt`: Dependency list with version pins

### Units & Defaults
- Default pixel size: **0.1 μm**
- Default frame interval: **0.1 s**
- Unit conversions centralized in `unit_converter.py`

### Optional Dependencies Pattern
Many features degrade gracefully if imports fail:
```python
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    DBSCAN = None

# Later, check before use
if SKLEARN_AVAILABLE:
    clustering = DBSCAN().fit(data)
```

## Report Generation
`EnhancedSPTReportGenerator` (`enhanced_report_generator.py`) orchestrates multi-analysis reports:
- **UI Mode**: `show_enhanced_report_generator()` for Streamlit
- **Batch Mode**: `generate_batch_report()` for scripts/automation
- **Exports**: JSON, HTML (interactive), PDF
- **Selection**: Categorized analyses with priority levels, "Select All" presets

## Common Pitfalls to Avoid

1. **Don't** access `st.session_state.tracks_df` directly → use `get_track_data()`
2. **Don't** display figures in analysis functions → return them for caller to display
3. **Don't** assume optional dependencies exist → check `*_AVAILABLE` flags
4. **Don't** hardcode units → use `get_units()` or `get_current_units()`
5. **Don't** use `st.cache` → use `st.cache_data` for Streamlit >= 1.28

## Key Directories
- Root: Main modules, standalone test scripts, sample CSVs
- `tests/`: Pytest-based unit tests
- `spt_projects/`: JSON project files + associated CSVs
- `cellpose/`, `segment_anything/`: Optional ML segmentation models
- `external_simulators/`: Molecular dynamics integration adapters

## Cross-Platform Notes
- Windows batch: `start.bat` (PowerShell-based setup)
- Unix shell: `run_tests.sh` (bash-based testing)
- Path handling: Always use `os.path.join()` or `pathlib`
- Temp files: Use `tempfile` module, not hardcoded paths

This is a production system with robust error handling. Maintain fallback patterns and structured error responses when extending functionality.