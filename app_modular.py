"""
SPT Analysis - Modular Streamlit Application (TEST VERSION)

This is a test version using the new modular page architecture.
Original app.py is preserved as backup.

Modular architecture features:
- Pages organized in pages/ directory
- Reusable UI components in ui_components/ directory
- Page registry system with decorator pattern
- Centralized navigation
"""

import streamlit as st
import numpy as np

# Import core utilities
from utils import initialize_session_state, calculate_track_statistics
from state_manager import get_state_manager
from analysis_manager import AnalysisManager
from unit_converter import UnitConverter
from logging_config import get_logger

# Import modular navigation system
from ui_components.navigation import setup_sidebar, load_page_content
from page_modules import get_available_pages

# Initialize logger
logger = get_logger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SPT Analysis (Modular)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize session state variables
initialize_session_state()

# Initialize unit converter
if 'unit_converter' not in st.session_state:
    st.session_state.unit_converter = UnitConverter()
    st.session_state.unit_converter.set_pixel_size(0.1)  # 0.1 μm/pixel
    st.session_state.unit_converter.set_frame_interval(0.03)  # 0.03 s/frame

# Initialize unit values for consistent access across all analyses
if 'current_pixel_size' not in st.session_state:
    st.session_state.current_pixel_size = st.session_state.unit_converter.pixel_size
if 'current_frame_interval' not in st.session_state:
    st.session_state.current_frame_interval = st.session_state.unit_converter.frame_interval

# Initialize the centralized state and analysis managers
state_manager = get_state_manager()
analysis_manager = AnalysisManager()

# Expose managers in session state for components that expect them
if 'app_state' not in st.session_state:
    st.session_state.app_state = state_manager
if 'analysis_manager' not in st.session_state:
    st.session_state.analysis_manager = analysis_manager

# Initialize mask tracking
if 'available_masks' not in st.session_state:
    st.session_state.available_masks = {}
if 'mask_metadata' not in st.session_state:
    st.session_state.mask_metadata = {}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_units():
    """Get the currently set unit values for use in all analyses."""
    return {
        'pixel_size': st.session_state.current_pixel_size,
        'frame_interval': st.session_state.current_frame_interval
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Setup navigation sidebar and get active page
    active_page = setup_sidebar()
    
    # Display current page info at top (for debugging)
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.info(f"🔍 Active Page: **{active_page}**")
        st.info(f"📦 Session State Keys: {len(st.session_state)} keys")
    
    # Load and render the active page
    try:
        load_page_content(active_page)
    except Exception as e:
        st.error(f"Error loading page '{active_page}': {e}")
        
        # Show detailed error for debugging
        import traceback
        with st.expander("🐛 Error Details (Debug)", expanded=True):
            st.code(traceback.format_exc())
        
        # Offer recovery options
        st.markdown("### Recovery Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Go to Home Page", use_container_width=True):
                st.session_state.active_page = "Home"
                st.rerun()
        with col2:
            if st.button("🔄 Reload Current Page", use_container_width=True):
                st.rerun()

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
