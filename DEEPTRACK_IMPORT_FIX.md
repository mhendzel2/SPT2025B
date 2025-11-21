# DeepTrack Import Error Fix

## Issue
Application crashed on startup with error:
```
ImportError: deeptrack attempted to use a functionality that requires module 
deeptrack.generators, but it couldn't be loaded.
```

The error occurred during Streamlit's inspection phase when loading `app.py`, even before the main UI code executed.

## Root Cause
Import chain: `app.py` → `advanced_tracking.py` → `track.py` → `deeptrack`

The `track.py` module imported `deeptrack` at the module level, but the deeptrack installation was incomplete (missing the `generators` submodule). This caused the import to fail during the lazy loading process when Streamlit tried to inspect the module structure.

## Solution
Wrapped the deeptrack import in `track.py` with try-except handling to gracefully handle incomplete installations:

```python
# Optional deeptrack import - handle incomplete installations gracefully
try:
    import deeptrack as dt
    DEEPTRACK_AVAILABLE = True
except (ImportError, AttributeError) as e:
    DEEPTRACK_AVAILABLE = False
    dt = None
```

Updated the `preprocess_with_deeptrack()` function to check availability before use:

```python
def preprocess_with_deeptrack(image: np.ndarray):
    if not DEEPTRACK_AVAILABLE:
        raise ImportError(
            "deeptrack is not available or not properly installed. "
            "Please install deeptrack2 with all dependencies: pip install deeptrack2"
        )
    # ... rest of function
```

## Benefits
1. **Application starts successfully** even with incomplete deeptrack installation
2. **Graceful degradation** - deeptrack features only fail when actually used
3. **Clear error messages** - users get helpful installation instructions if they try to use deeptrack features
4. **Follows project patterns** - matches the optional dependency handling used elsewhere (e.g., sklearn, cellpose)

## Testing
Verified the fix:
```bash
# Import chain now works
python -c "from advanced_tracking import ParticleFilter, AdvancedTracking, bayesian_detection_refinement; print('Import successful')"
# Output: Import successful

# App module loads without errors
python -c "import importlib.util; spec = importlib.util.spec_from_file_location('app', 'app.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)"
# Output: INFO: SPT Analysis Application Started (no ImportError)
```

## Notes
- DeepTrack2 is an optional dependency for advanced preprocessing
- The `run_btrack()` function (main tracking functionality) does NOT require deeptrack - only the `preprocess_with_deeptrack()` helper function uses it
- If users need deeptrack functionality, they should install: `pip install deeptrack2`
- Current warnings about TensorFlow compatibility are informational only and don't affect functionality

## Files Modified
- `track.py`: Added try-except import wrapper and availability flag check
