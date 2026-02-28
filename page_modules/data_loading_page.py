"""
Data Loading Page Module

Handles all data import functionality including track data, images, and settings.
"""

import streamlit as st
from page_modules import register_page

# Import required utilities
try:
    from data_loader import load_image_file, load_tracks_file
    from utils import calculate_track_statistics
    from secure_file_validator import get_file_validator
    from data_quality_checker import DataQualityChecker
    from ui_components import navigate_to
    DATA_LOADING_AVAILABLE = True
except ImportError as e:
    DATA_LOADING_AVAILABLE = False
    _import_error = str(e)


@register_page("Data Loading")
def render():
    """
    Render the data loading page.
    
    Features 5 tabs:
    - Image Settings: Global pixel size and frame interval
    - Track Data: Upload pre-tracked data
    - Images for Tracking: Upload images for particle detection
    - Images for Mask Generation: Upload images for segmentation
    - Sample Data: Load test datasets
    """
    if not DATA_LOADING_AVAILABLE:
        st.error("Data loading module dependencies not available")
        st.error(f"Import error: {_import_error}")
        return
    
    st.title("Data Loading")
    
    # Create tabs for different data loading options
    tabs = st.tabs([
        "Image Settings",
        "Track Data", 
        "Images for Tracking",
        "Images for Mask Generation",
        "Sample Data"
    ])
    
    with tabs[0]:
        _render_image_settings()
    
    with tabs[1]:
        _render_track_data_upload()
    
    with tabs[2]:
        _render_tracking_image_upload()
    
    with tabs[3]:
        _render_mask_image_upload()
    
    with tabs[4]:
        _render_sample_data()


def _render_image_settings():
    """Render global image settings tab."""
    st.header("Image Settings")
    st.info("Configure global parameters that will be used throughout the application for all analyses")
    
    st.markdown("### Global Image Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Spatial Calibration")
        if 'global_pixel_size' not in st.session_state:
            st.session_state.global_pixel_size = 0.1
        
        global_pixel_size = st.number_input(
            "Pixel Size (µm)",
            min_value=0.001,
            max_value=10.0,
            value=st.session_state.global_pixel_size,
            step=0.001,
            format="%.3f",
            help="Physical size of one pixel in micrometers",
            key="global_pixel_settings"
        )
    
    with col2:
        st.markdown("#### Temporal Calibration")
        if 'global_frame_interval' not in st.session_state:
            st.session_state.global_frame_interval = 0.1
        
        global_frame_interval = st.number_input(
            "Frame Interval (s)",
            min_value=0.001,
            max_value=3600.0,
            value=st.session_state.global_frame_interval,
            step=0.001,
            format="%.3f",
            help="Time between consecutive frames in seconds",
            key="global_frame_settings"
        )
    
    # Store values
    st.session_state.current_pixel_size = global_pixel_size
    st.session_state.current_frame_interval = global_frame_interval
    st.session_state.pixel_size = global_pixel_size
    st.session_state.frame_interval = global_frame_interval
    
    st.markdown("---")
    
    # Track coordinate units
    st.markdown("### Track Data Coordinate Units")
    if 'track_coordinates_in_microns' not in st.session_state:
        st.session_state.track_coordinates_in_microns = False
    
    track_coords_in_microns = st.checkbox(
        "Track data coordinates are already in microns",
        value=st.session_state.track_coordinates_in_microns,
        key="track_coords_units",
        help="Check if track coordinates are in micrometers rather than pixels"
    )
    st.session_state.track_coordinates_in_microns = track_coords_in_microns
    
    st.markdown("---")
    
    # Settings summary
    st.markdown("### Current Settings Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Pixel Size", f"{global_pixel_size:.3f} µm")
    with col2:
        st.metric("Frame Interval", f"{global_frame_interval:.3f} s")
    
    if st.button("Reset to Default Values"):
        st.session_state.global_pixel_size = 0.1
        st.session_state.global_frame_interval = 0.1
        st.session_state.pixel_size = 0.1
        st.session_state.frame_interval = 0.1
        st.success("Settings reset to default values")
        st.rerun()


def _render_track_data_upload():
    """Render track data upload tab."""
    st.header("Upload Track Data")
    st.info("Load pre-processed track data from other tracking software")
    
    track_file = st.file_uploader(
        "Upload track data file",
        type=["csv", "xlsx", "h5", "json", "ims", "xml", "uic", "mvd2", "aisf", "aiix"],
        help="Upload track data in various formats"
    )
    
    if track_file is not None:
        try:
            # Validate file
            file_validator = get_file_validator()
            validation_result = file_validator.validate_file(
                track_file, 
                file_type='track_data',
                check_content=True
            )
            
            if not validation_result['valid']:
                st.error("⚠️ File validation failed!")
                for error in validation_result['errors']:
                    st.error(f"❌ {error}")
                return
            
            # Load tracks
            st.session_state.tracks_data = load_tracks_file(track_file)
            st.success(f"Track data loaded successfully: {track_file.name}")
            
            # Calculate statistics
            with st.spinner("Calculating track statistics..."):
                st.session_state.track_statistics = calculate_track_statistics(
                    st.session_state.tracks_data
                )
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.tracks_data.head())
            
            # Basic statistics
            st.subheader("Basic Statistics")
            num_tracks = st.session_state.tracks_data['track_id'].nunique()
            num_points = len(st.session_state.tracks_data)
            st.write(f"Number of tracks: {num_tracks}")
            st.write(f"Number of points: {num_points}")
            st.write(f"Average track length: {num_points / num_tracks:.1f} points")
            
            # Navigation button
            st.button("Proceed to Analysis", on_click=navigate_to, args=("Analysis",))
            
        except Exception as e:
            st.error(f"Error loading track data: {e}")


def _render_tracking_image_upload():
    """Render images for tracking upload tab."""
    st.header("Upload Images for Tracking")
    st.info("Load microscopy images to perform particle detection and tracking")
    
    # Display current settings
    st.markdown("### Current Global Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pixel Size", f"{st.session_state.get('pixel_size', 0.1):.3f} µm")
    with col2:
        st.metric("Frame Interval", f"{st.session_state.get('frame_interval', 0.1):.3f} s")
    
    tracking_image_file = st.file_uploader(
        "Upload microscopy images",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="Upload images for particle detection and tracking",
        key="tracking_images_uploader"
    )
    
    if tracking_image_file is not None:
        try:
            st.session_state.image_data = load_image_file(tracking_image_file)
            st.success(f"Image loaded successfully: {tracking_image_file.name}")
            
            # Display preview
            st.subheader("Image Preview")
            preview_image = (st.session_state.image_data[0] 
                           if isinstance(st.session_state.image_data, list)
                           else st.session_state.image_data)
            
            if preview_image is not None:
                from image_processing_utils import normalize_image_for_display
                st.image(
                    normalize_image_for_display(preview_image),
                    caption="Image for tracking",
                    use_container_width=True
                )
                st.write(f"Image dimensions: {preview_image.shape}")
            
            st.button("Proceed to Tracking", on_click=navigate_to, args=("Tracking",))
            
        except Exception as e:
            st.error(f"Error loading image: {e}")


def _render_mask_image_upload():
    """Render mask image upload tab."""
    st.header("Upload Images for Mask Generation")
    st.info("Load images to generate masks for boundary crossing and spatial analysis")
    
    mask_image_file = st.file_uploader(
        "Upload images for mask generation",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="Upload images to create masks for nuclear boundaries or ROIs",
        key="mask_image_uploader"
    )
    
    if mask_image_file is not None:
        try:
            st.session_state.mask_images = load_image_file(mask_image_file)
            st.success(f"Mask images loaded successfully: {mask_image_file.name}")
            
            # Display preview
            st.subheader("Image Preview")
            preview_image = (st.session_state.mask_images[0]
                           if isinstance(st.session_state.mask_images, list)
                           else st.session_state.mask_images)
            
            if preview_image is not None:
                from image_processing_utils import normalize_image_for_display
                st.image(
                    normalize_image_for_display(preview_image),
                    caption="Mask image",
                    use_container_width=True
                )
                st.write(f"Image dimensions: {preview_image.shape}")
            
            st.button(
                "Proceed to Image Processing",
                on_click=navigate_to,
                args=("Image Processing",),
                key="proceed_to_image_processing"
            )
            
        except Exception as e:
            st.error(f"Error loading mask images: {e}")


def _render_sample_data():
    """Render sample data loading tab."""
    st.header("Sample Data")
    st.info("Load sample datasets to test the application features")
    
    st.subheader("Available Sample Datasets")
    
    sample_datasets = [
        "Sample Track Data (CSV)",
        "Sample Microscopy Images",
        "Sample MD Simulation Data"
    ]
    
    selected_sample = st.selectbox("Choose a sample dataset", sample_datasets)
    
    if st.button("Load Sample Data", key="load_sample_data_btn"):
        try:
            if selected_sample == "Sample Track Data (CSV)":
                from test_data_generator import generate_sample_tracks
                st.session_state.tracks_data = generate_sample_tracks()
                st.success("Sample track data loaded successfully!")
                
            elif selected_sample == "Sample Microscopy Images":
                from test_data_generator import generate_sample_image
                st.session_state.image_data = generate_sample_image()
                st.success("Sample microscopy image loaded successfully!")
                
            elif selected_sample == "Sample MD Simulation Data":
                from test_data_generator import generate_sample_md_data
                st.session_state.md_data = generate_sample_md_data()
                st.success("Sample MD simulation data loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
