# App.py Refactoring Plan

## Current State
- **Total Lines**: 9,882 lines
- **Structure**: Single monolithic file with 12 page sections
- **Issues**: 
  - Hard to maintain and navigate
  - High risk of merge conflicts
  - Difficult to test individual pages
  - Poor separation of concerns

## Proposed Modular Structure

### New Directory: `pages/`
Split app.py into focused page modules:

```
pages/
├── __init__.py                 # Page registry and loader
├── home_page.py               # Home/landing page
├── data_loading_page.py       # Data upload and import
├── image_processing_page.py   # Segmentation and masking
├── tracking_page.py           # Particle detection and linking
├── analysis_page.py           # Basic analysis tools
├── advanced_analysis_page.py  # Advanced biophysical analysis
├── visualization_page.py      # Plotting and visualization
├── project_management_page.py # Save/load projects
├── report_generation_page.py  # Report builder
├── anomaly_detection_page.py  # AI anomaly detection
├── md_integration_page.py     # Molecular dynamics integration
└── simulation_page.py         # Brownian motion simulation
```

### New Directory: `ui_components/`
Reusable UI widgets and helpers:

```
ui_components/
├── __init__.py
├── navigation.py         # Sidebar navigation, page routing
├── mask_selector.py      # Mask selection UI (used in multiple pages)
├── channel_selector.py   # Channel selection for multichannel images
├── track_display.py      # Track visualization widgets
├── metrics_display.py    # Metrics cards and summaries
└── file_upload.py        # File upload widgets with validation
```

### Refactored app.py Structure

```python
# app.py (new structure - ~200 lines)
"""
SPT2025B Main Application Entry Point
Streamlit multi-page application for Single Particle Tracking analysis
"""

import streamlit as st
from pathlib import Path

# Core imports
from state_manager import get_state_manager
from analysis_manager import AnalysisManager
from ui_components.navigation import setup_sidebar, navigate_to
from pages import load_page

# Configuration
st.set_page_config(
    page_title="SPT Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize state
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Home'

# Initialize managers
state_manager = get_state_manager()
analysis_manager = AnalysisManager()

# Sidebar navigation
current_page = setup_sidebar()

# Update active page
if current_page != st.session_state.active_page:
    st.session_state.active_page = current_page

# Load and render the active page
load_page(st.session_state.active_page)
```

## Page Module Template

Each page module follows this pattern:

```python
# pages/example_page.py
"""
Example Page Module
Description of what this page does
"""

import streamlit as st
import numpy as np
from typing import Optional

def render():
    """
    Main render function for the page.
    Called by the page loader when this page is active.
    """
    st.title("Page Title")
    
    # Check prerequisites
    if not _check_prerequisites():
        _show_missing_data_warning()
        return
    
    # Page content organized in tabs/sections
    _render_main_content()

def _check_prerequisites() -> bool:
    """Check if required data is loaded."""
    return 'required_data' in st.session_state

def _show_missing_data_warning():
    """Display warning when prerequisites aren't met."""
    st.warning("Please load data first.")
    if st.button("Go to Data Loading"):
        from ui_components.navigation import navigate_to
        navigate_to("Data Loading")

def _render_main_content():
    """Render the main page content."""
    # Implementation here
    pass

# Helper functions specific to this page
def _helper_function():
    pass
```

## Migration Strategy

### Phase 1: Setup Infrastructure (Day 1)
1. ✅ Create `refactor/modular-app-structure` branch
2. Create `pages/` and `ui_components/` directories
3. Create `pages/__init__.py` with page registry
4. Create `ui_components/navigation.py` with sidebar logic

### Phase 2: Extract Simple Pages (Day 1-2)
Extract pages with minimal dependencies first:
1. Home page (~200 lines)
2. Simulation page (~100 lines)
3. Report Generation page (~300 lines)
4. Project Management page (~300 lines)

### Phase 3: Extract Complex Pages (Day 2-3)
Extract pages with moderate complexity:
1. Data Loading page (~500 lines)
2. Visualization page (~500 lines)
3. AI Anomaly Detection page (~150 lines)

### Phase 4: Extract Core Pages (Day 3-4)
Extract the most complex pages:
1. Tracking page (~1500 lines)
2. Image Processing page (~1500 lines)
3. Analysis page (~2000 lines)
4. Advanced Analysis page (~2000 lines)

### Phase 5: Extract UI Components (Day 4-5)
Extract reusable components:
1. Mask selection UI
2. Channel selection UI
3. File upload widgets
4. Metrics display cards

### Phase 6: Testing & Refinement (Day 5-6)
1. Test each page individually
2. Test navigation between pages
3. Test state persistence
4. Update documentation
5. Create migration guide

## Benefits of Refactoring

### Maintainability
- ✅ Each page is self-contained (~100-500 lines)
- ✅ Easy to find and modify specific features
- ✅ Clear separation of concerns

### Collaboration
- ✅ Multiple developers can work on different pages
- ✅ Reduced merge conflicts
- ✅ Easier code reviews

### Testing
- ✅ Unit test individual pages
- ✅ Mock dependencies easily
- ✅ Isolated bug fixes

### Performance
- ✅ Lazy loading of page modules
- ✅ Faster Streamlit reruns (smaller execution context)
- ✅ Better caching opportunities

## File Size Comparison

| File | Before | After |
|------|--------|-------|
| app.py | 9,882 lines | ~200 lines |
| home_page.py | - | ~150 lines |
| data_loading_page.py | - | ~400 lines |
| image_processing_page.py | - | ~600 lines |
| tracking_page.py | - | ~800 lines |
| analysis_page.py | - | ~1200 lines |
| advanced_analysis_page.py | - | ~1500 lines |
| visualization_page.py | - | ~500 lines |
| project_management_page.py | - | ~300 lines |
| report_generation_page.py | - | ~300 lines |
| anomaly_detection_page.py | - | ~150 lines |
| md_integration_page.py | - | ~200 lines |
| simulation_page.py | - | ~100 lines |
| **Total** | **9,882 lines** | **~6,500 lines** (35% reduction through deduplication) |

## Compatibility Notes

### Backward Compatibility
- ✅ All existing functionality preserved
- ✅ Session state structure unchanged
- ✅ Same user experience
- ✅ All imports/dependencies maintained

### Migration Path
- ✅ Branch-based development (no impact on main)
- ✅ Can revert to monolithic if issues arise
- ✅ Gradual rollout possible

## Next Steps

1. **Immediate**: Create basic infrastructure
   - pages/__init__.py
   - ui_components/navigation.py
   
2. **Short-term**: Extract 2-3 simple pages as proof of concept
   - home_page.py
   - simulation_page.py
   
3. **Medium-term**: Complete full migration
   - All 12 pages extracted
   - UI components modularized
   
4. **Long-term**: Further optimization
   - Page-specific helper modules
   - Shared utility libraries
   - Comprehensive testing suite

---

**Branch**: `refactor/modular-app-structure`  
**Status**: Planning Complete ✅  
**Ready to Start**: Phase 1 - Setup Infrastructure
