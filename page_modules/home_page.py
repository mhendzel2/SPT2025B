"""
Home Page Module
Landing page with quick access and recent analyses
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
from page_modules import register_page
from ui_components.navigation import create_navigation_button, create_page_header

@register_page("Home")
def render():
    """Render the home page."""
    create_page_header("SPT Analysis", "Welcome to the Single Particle Tracking Analysis Platform")
    
    # Main welcome content
    st.markdown("""
    This application provides comprehensive tools for analyzing single particle tracking data.
    Use the navigation menu on the left to access different tools:

    - **Data Loading**: Load your tracking data or microscopy images
    - **Tracking**: Detect and track particles in microscopy images
    - **Analysis**: Analyze tracks with various methods
    - **Visualization**: Visualize tracks and analysis results
    - **Advanced Analysis**: Access specialized analysis modules
    - **Project Management**: Organize experiments and conditions
    """)
    
    # Quick access section
    st.header("Quick Start")
    
    # Two columns for quick access
    col1, col2 = st.columns(2)
    
    with col1:
        _render_quick_upload()
    
    with col2:
        _render_recent_analyses()
        _render_quick_links()


def _render_quick_upload():
    """Render the quick upload section."""
    st.subheader("Load Data")
    
    # Import loading functions
    from data_loader import load_image_file, load_tracks_file, format_track_data
    
    # File uploader for quick loading
    uploaded_file = st.file_uploader(
        "Upload image or track data", 
        type=["tif", "tiff", "png", "jpg", "jpeg", "csv", "xlsx", "h5", "json", "ims", "uic", "xml", "mvd2", "aisf", "aiix"]
    )
    
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Process depending on file type
        if file_extension in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            _load_image_file(uploaded_file)
        elif file_extension in ['.csv', '.xlsx', '.h5', '.json']:
            _load_track_file(uploaded_file)
        elif file_extension in ['.ims', '.xml']:
            _load_imaris_file(uploaded_file)
        elif file_extension in ['.uic', '.aisf', '.aiix']:
            _load_volocity_file(uploaded_file)
        elif file_extension in ['.mvd2']:
            _load_mvd2_file(uploaded_file)


def _load_image_file(uploaded_file):
    """Load and process image file."""
    from data_loader import load_image_file
    from ui_components.navigation import navigate_to
    
    try:
        image_data = load_image_file(uploaded_file)
        
        # Store in session state using centralized state manager
        if 'app_state' in st.session_state:
            st.session_state.app_state.load_image_data(image_data, {'filename': uploaded_file.name})
        else:
            st.session_state.image_data = image_data
            
        st.success(f"Image loaded successfully: {uploaded_file.name}")
        
        # Display preview
        if isinstance(image_data, list) and len(image_data) > 0:
            display_image = image_data[0].copy()
        else:
            display_image = image_data.copy()
            
        if display_image.dtype != np.uint8:
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype(np.uint8)
            else:
                display_image = ((display_image - display_image.min()) / 
                               (display_image.max() - display_image.min()) * 255).astype(np.uint8)
        st.image(display_image, caption="Image preview", use_container_width=True)
        
        # Navigate to tracking
        navigate_to("Tracking")
        st.rerun()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")


def _load_track_file(uploaded_file):
    """Load and process track data file."""
    from data_loader import load_tracks_file
    from utils import calculate_track_statistics
    from ui_components.navigation import navigate_to
    
    try:
        tracks_data = load_tracks_file(uploaded_file)
        
        # Store in session state
        if 'app_state' in st.session_state:
            st.session_state.app_state.load_tracks(tracks_data, source=uploaded_file.name)
        else:
            st.session_state.tracks_data = tracks_data
            
        # Calculate track statistics
        if 'analysis_manager' in st.session_state:
            st.session_state.analysis_manager.calculate_track_statistics()
        else:
            st.session_state.track_statistics = calculate_track_statistics(tracks_data)
            
        st.success(f"Track data loaded successfully: {uploaded_file.name}")
        st.dataframe(tracks_data.head())
        
        # Navigate to analysis
        navigate_to("Analysis")
        st.rerun()
    except Exception as e:
        st.error(f"Error loading track data: {str(e)}")


def _load_imaris_file(uploaded_file):
    """Load Imaris file."""
    from special_file_handlers import load_imaris_file
    from utils import calculate_track_statistics
    from ui_components.navigation import navigate_to
    
    try:
        st.session_state.tracks_data = load_imaris_file(uploaded_file)
        st.success(f"Imaris file loaded successfully: {uploaded_file.name}")
        st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
        navigate_to("Analysis")
        st.rerun()
    except Exception as e:
        st.error(f"Error loading Imaris file: {str(e)}")


def _load_volocity_file(uploaded_file):
    """Load Volocity file."""
    from volocity_handler import load_volocity_file
    from utils import calculate_track_statistics
    from ui_components.navigation import navigate_to
    
    try:
        st.session_state.tracks_data = load_volocity_file(uploaded_file)
        st.success(f"Volocity file loaded successfully: {uploaded_file.name}")
        st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
        navigate_to("Analysis")
        st.rerun()
    except Exception as e:
        st.error(f"Error loading Volocity file: {str(e)}")


def _load_mvd2_file(uploaded_file):
    """Load MVD2 file."""
    from mvd2_handler import load_mvd2_file
    from utils import calculate_track_statistics
    from ui_components.navigation import navigate_to
    
    try:
        st.session_state.tracks_data = load_mvd2_file(uploaded_file)
        st.success(f"MVD2 file loaded successfully: {uploaded_file.name}")
        st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
        navigate_to("Analysis")
        st.rerun()
    except Exception as e:
        st.error(f"Error loading MVD2 file: {str(e)}")


def _render_recent_analyses():
    """Render recent analyses section."""
    st.subheader("Recent Analysis")
    
    # Check if there are any recent analyses
    if 'recent_analyses' in st.session_state and st.session_state.recent_analyses:
        for analysis in st.session_state.recent_analyses[-3:]:  # Show last 3
            with st.expander(f"{analysis['name']} - {analysis['date']}"):
                st.write(f"Type: {analysis['type']}")
                st.write(f"Parameters: {analysis['parameters']}")
                
                # Button to open the analysis
                if st.button("Open", key=f"open_{analysis['id']}"):
                    st.session_state.selected_analysis = analysis['id']
                    from ui_components.navigation import navigate_to
                    navigate_to("Analysis")
                    st.rerun()
    else:
        st.info("No recent analyses available.")


def _render_quick_links():
    """Render quick links section."""
    from ui_components.navigation import navigate_to
    from data_loader import format_track_data
    from utils import calculate_track_statistics
    
    st.subheader("Quick Links")
    quick_links = st.selectbox(
        "Select a task",
        ["Select a task...", "New tracking", "New analysis", "Load sample data"]
    )
    
    if quick_links == "New tracking":
        navigate_to("Tracking")
        st.rerun()
    elif quick_links == "New analysis":
        navigate_to("Analysis")
        st.rerun()
    elif quick_links == "Load sample data":
        try:
            # Load first available sample dataset
            sample_data_dir = "sample data"
            sample_file_path = None
            
            if os.path.exists(sample_data_dir):
                for subdir in os.listdir(sample_data_dir):
                    subdir_path = os.path.join(sample_data_dir, subdir)
                    if os.path.isdir(subdir_path):
                        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                        if csv_files:
                            sample_file_path = os.path.join(subdir_path, csv_files[0])
                            break
            
            if sample_file_path and os.path.exists(sample_file_path):
                st.session_state.tracks_data = pd.read_csv(sample_file_path)
                # Format to standard format
                st.session_state.tracks_data = format_track_data(st.session_state.tracks_data)
                # Calculate track statistics
                st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                st.success(f"Sample data loaded: {os.path.basename(sample_file_path)}")
                navigate_to("Analysis")
                st.rerun()
            else:
                st.warning("Sample data file not found. Check 'sample data' folder.")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
