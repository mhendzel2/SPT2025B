# SPT2025B AI Coding Agent Instructions

## Project Overview
SPT2025B is a comprehensive Single Particle Tracking analysis platform built with Streamlit. It provides advanced particle detection, tracking, and analysis capabilities for biophysical research including microrheology, diffusion analysis, and motion classification.

## Core Architecture

### State Management Pattern
The application uses a centralized state management system:
- **StateManager** (`state_manager.py`) - Manages session state, data persistence, and cross-component communication
- **AnalysisManager** (`analysis_manager.py`) - Handles analysis execution and result caching
- **Data Access Utils** (`data_access_utils.py`) - Provides consistent data access with fallback mechanisms

**Critical**: Always use `get_track_data()` and `get_analysis_results()` from `data_access_utils` rather than direct session state access. The system has multiple fallback mechanisms for data retrieval.

### Multi-Page Architecture
Navigation is controlled by `st.session_state.active_page` with pages including:
- Data Loading, Analysis, Visualization, Project Management, Report Generation
- Each page handles its own state initialization and cleanup
- Use `navigate_to(page_name)` helper function for page transitions

## Data Loading & Format Handling

### Supported Formats
The system handles diverse file formats through specialized handlers:
- **CSV/Excel**: Standard tracking data via `data_loader.py`
- **Specialized formats**: MVD2 (`mvd2_handler.py`), Volocity (`volocity_handler.py`), Imaris (`special_file_handlers.py`)
- **Image formats**: TIFF stacks, PNG, JPG via `load_image_file()`

### Data Pipeline
1. Load → `format_track_data()` → Validate → `calculate_track_statistics()`
2. All track data must have columns: `track_id`, `frame`, `x`, `y`
3. Coordinates can be in pixels or microns (tracked via `coordinates_in_microns` flag)

## Analysis System

### Core Analysis Functions
Located in `analysis.py` with standardized signatures:
```python
def analysis_function(tracks_df, max_lag=20, pixel_size=1.0, frame_interval=1.0, **kwargs):
    return {
        'success': bool,
        'data': results_dict,
        'summary': summary_dict,
        'figures': plotly_figures_dict
    }
```

### Analysis Categories
- **Diffusion**: MSD, anomalous diffusion, confined motion
- **Motion**: Velocity, directional persistence, motion classification  
- **Advanced**: Microrheology, clustering, anomaly detection, changepoint detection
- **Statistical**: Comparative analysis, population analysis

### Enhanced Report Generator
The `EnhancedSPTReportGenerator` (`enhanced_report_generator.py`) provides:
- **Analysis Selection**: Categorized with priority levels, "Select All" functionality
- **Batch Processing**: `generate_batch_report()` for non-Streamlit environments
- **Multiple Exports**: JSON, HTML Interactive, PDF reports
- **Validation**: Built-in checks for data availability and analysis prerequisites

## Project Management

### File-Based Projects
Projects are stored as JSON with associated CSV data files:
- **ProjectManager** handles CRUD operations
- **Conditions** represent experimental groups with multiple files
- **Batch Processing** generates reports across all conditions
- Projects support comparative analysis and statistical testing

## Testing & Validation

### Test Structure
- **Standalone Scripts**: `test_*.py` files for comprehensive functionality testing
- **Pytest Integration**: `tests/` directory for unit tests
- **Sample Data**: Multiple CSV files in root directory for testing different scenarios

### Common Test Patterns
```python
# Test imports first
from enhanced_report_generator import EnhancedSPTReportGenerator

# Create sample data
tracks_df = create_sample_track_data()

# Test analysis pipeline
generator = EnhancedSPTReportGenerator()
result = generator._analyze_basic_statistics(tracks_df, {'pixel_size': 0.1, 'frame_interval': 0.1})
assert result.get('success'), f"Analysis failed: {result.get('error')}"
```

## Advanced Features

### Segmentation Pipeline
- **Traditional Methods**: Otsu, watershed, adaptive thresholding in `segmentation.py`
- **ML Methods**: CellSAM and Cellpose in `advanced_segmentation.py` (with graceful fallbacks)
- **Enhanced Detection**: CNN-based particle detection with size filtering

### External Integrations
- **MD Simulation**: Integration with molecular dynamics data (`md_integration.py`)
- **Tracking Libraries**: btrack, DeepTrack support (`track.py`)
- **Scientific Libraries**: Extensive use of scikit-learn, scipy, plotly

## Development Workflows

### Adding New Analysis
1. Implement in `analysis.py` following standard return format
2. Add visualization function returning plotly figure
3. Register in `EnhancedSPTReportGenerator.available_analyses`
4. Add test in appropriate test file
5. Update constants and configuration as needed

### Error Handling
- Use `enhanced_error_handling.py` for statistical validation
- Always provide fallback mechanisms for optional dependencies
- Return structured error responses: `{'success': False, 'error': 'description'}`

### Performance Considerations
- Large datasets are handled via chunking and progress bars
- Use `st.cache_data` for expensive computations
- Optimize with vectorized operations (numpy/pandas)

## Configuration & Constants

### Key Files
- `constants.py`: All magic numbers, default parameters, session state keys
- `config.toml`: User-editable configuration parameters
- `requirements.txt`: Comprehensive dependency list with version constraints

### Units & Conversions
- Consistent unit handling via `unit_converter.py`
- Default: 0.1 μm pixels, 0.1 s frame intervals
- Use `get_current_units()` for analysis parameter passing

## Common Patterns

### Session State Access
```python
# Preferred - robust access
tracks_df, has_data = get_track_data()
if not has_data:
    st.error("No track data available")
    return

# Analysis results
analysis_results = get_analysis_results()
```

### Analysis Execution
```python
# Standard analysis pattern
try:
    result = analysis_function(tracks_df, **params)
    if result.get('success'):
        st.session_state.analysis_results[analysis_key] = result
        display_results(result)
    else:
        st.error(f"Analysis failed: {result.get('error')}")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
```

### Visualization
- Use plotly for interactive plots, matplotlib for static
- All visualization functions should return figures, not display directly
- Store figures in `st.session_state` for report generation

## Special Considerations

### Dependency Management
Many advanced features are optional - always check availability:
```python
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
```

### Cross-Platform Support
- File paths use `os.path.join()` and `pathlib`
- Shell commands in test scripts handle both Unix and Windows
- Default configurations work across different environments

This is a mature, production-ready application with extensive error handling, fallback mechanisms, and comprehensive testing. Focus on maintaining these patterns when extending functionality.