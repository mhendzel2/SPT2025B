"""
SPT Analysis - Streamlit Application

Main entry point for the Single Particle Tracking analysis application.
This application provides comprehensive tools for:
- Particle detection and tracking
- Track analysis (diffusion, motion, clustering, etc.)
- Comparative analysis between datasets
- Project management for multiple tracking files
- Visualization of results
"""

import streamlit as st
import os
import importlib
import sys

# Import centralized state management
from state_manager import get_state_manager
from analysis_manager import AnalysisManager
from unit_converter import UnitConverter
from settings_panel import get_settings_panel, get_global_units
from ui_utils import apply_custom_css

# Configure page first
st.set_page_config(
    page_title="SPT Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state
from utils import initialize_session_state
initialize_session_state()

# Initialize managers
if 'app_state' not in st.session_state:
    st.session_state.app_state = get_state_manager()
if 'analysis_manager' not in st.session_state:
    st.session_state.analysis_manager = AnalysisManager()
if 'unit_converter' not in st.session_state:
    st.session_state.unit_converter = UnitConverter()
    # Set defaults
    st.session_state.unit_converter.set_pixel_size(0.1)
    st.session_state.unit_converter.set_frame_interval(0.03)

# Initialize global parameters if not set
if "pixel_size" not in st.session_state:
    st.session_state.pixel_size = 0.1
if "frame_interval" not in st.session_state:
    st.session_state.frame_interval = 0.03

# Sidebar setup
st.sidebar.title("SPT Analysis")

# Navigation options
nav_options = [
    "Home",
    "Project Management",
    "Data Loading",
    "Image Processing",
    "Tracking",
    "Analysis",
    "Visualization",
    "Advanced Analysis",
    "AI Anomaly Detection",
    "Report Generation",
    "MD Integration",
    "Simulation"
]

# Set active page
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Home"

# Sync sidebar with session state
try:
    default_index = nav_options.index(st.session_state.active_page)
except ValueError:
    default_index = 0

selected_page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=default_index
)

# Update session state if changed via sidebar
if selected_page != st.session_state.active_page:
    st.session_state.active_page = selected_page
    st.rerun()

# Settings Panel in sidebar
try:
    settings_panel = get_settings_panel()
    settings_panel.show_compact_sidebar()
    
    # Sync settings
    global_units = get_global_units()
    st.session_state.pixel_size = global_units['pixel_size']
    st.session_state.frame_interval = global_units['frame_interval']
    st.session_state.unit_converter.set_pixel_size(st.session_state.pixel_size)
    st.session_state.unit_converter.set_frame_interval(st.session_state.frame_interval)
except Exception as e:
    st.sidebar.warning(f"Settings panel unavailable: {e}")

# Data Status in sidebar
with st.sidebar.expander("Data Status", expanded=False):
    if st.session_state.tracks_data is not None:
        try:
            import pandas as pd
            if isinstance(st.session_state.tracks_data, pd.DataFrame) and not st.session_state.tracks_data.empty:
                if 'track_id' in st.session_state.tracks_data.columns:
                    n_tracks = st.session_state.tracks_data['track_id'].nunique()
                    st.success(f"Tracks: {n_tracks}")
                else:
                    st.info(f"Data: {len(st.session_state.tracks_data)} rows")
            else:
                st.warning("Empty dataset")
        except Exception:
            st.warning("Invalid data format")
    else:
        st.info("No tracks loaded")
        
    if st.session_state.image_data is not None:
        try:
            if isinstance(st.session_state.image_data, list):
                st.success(f"Images: {len(st.session_state.image_data)} frames")
            elif hasattr(st.session_state.image_data, 'shape'):
                if len(st.session_state.image_data.shape) > 2:
                    st.success(f"Images: {st.session_state.image_data.shape[0]} frames")
                else:
                    st.success("Single image loaded")
        except Exception:
            st.success("Images loaded")
    else:
        st.info("No images loaded")

# Sample Data Loader in sidebar
with st.sidebar.expander("Sample Data"):
    sample_data_dir = "sample data"
    if os.path.exists(sample_data_dir):
        # Scan for CSV files
        csv_files = []
        for root, dirs, files in os.walk(sample_data_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            selected_sample = st.selectbox("Select Dataset", csv_files, format_func=lambda x: os.path.basename(x))
            if st.button("Load Sample"):
                try:
                    import pandas as pd
                    from utils import format_track_data, calculate_track_statistics
                    
                    df = pd.read_csv(selected_sample)
                    st.session_state.tracks_data = format_track_data(df)
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.success(f"Loaded {os.path.basename(selected_sample)}")
                    st.session_state.active_page = "Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No CSV files found in sample data.")
    else:
        st.info("Sample data directory not found.")

# Routing Logic
def render_page():
    page = st.session_state.active_page
    
    try:
        if page == "Home":
            from pages.home import show_home_page
            show_home_page()
            
        elif page == "Project Management":
            from pages.project_management import show_project_management_page
            show_project_management_page()
            
        elif page == "Data Loading":
            from pages.data_loading import show_data_loading_page
            show_data_loading_page()
            
        elif page == "Image Processing":
            from pages.image_processing import show_image_processing_page
            show_image_processing_page()
            
        elif page == "Tracking":
            from pages.tracking import show_tracking_page
            show_tracking_page()
            
        elif page == "Analysis":
            from pages.analysis import show_analysis_page
            show_analysis_page()
            
        elif page == "Visualization":
            from pages.visualization import show_visualization_page
            show_visualization_page()
            
        elif page == "Advanced Analysis":
            from pages.advanced_analysis import show_advanced_analysis_page
            show_advanced_analysis_page()
            
        elif page == "AI Anomaly Detection":
            from pages.anomaly_detection_page import show_anomaly_detection_page
            show_anomaly_detection_page()
            
        elif page == "Report Generation":
            from pages.report_generation_page import show_report_generation_page
            show_report_generation_page()
            
        elif page == "MD Integration":
            from pages.md_integration import show_md_integration_page
            show_md_integration_page()
            
        elif page == "Simulation":
            from pages.simulation_page import show_simulation_page
            show_simulation_page()
            
        else:
            st.error(f"Page '{page}' not found.")
            
    except ImportError as e:
        st.error(f"Error importing page module: {e}")
        st.info("Please ensure all page modules are correctly installed in the 'pages/' directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.expander("Error Details").code(traceback.format_exc())

# Render the selected page
render_page()
