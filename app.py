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
import pandas as pd
import numpy as np
import os
import io
import time
import uuid
import json
from pathlib import Path
import tempfile  # ADDED: For temporary file handling
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import project_management as pm
from scipy.stats import linregress

# Import logging configuration
from logging_config import get_logger

# Initialize logger for app.py
logger = get_logger(__name__)


# Import report generation module
try:
    # CORRECTED: Import the EnhancedSPTReportGenerator class from the correct file
    # Import both the report generator class and the helper function that
    # displays the Streamlit interface.
    from enhanced_report_generator import (
        EnhancedSPTReportGenerator,
        show_enhanced_report_generator,
    )
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False

# Import utility modules
from data_loader import load_image_file, load_tracks_file
from constants import DEFAULT_PIXEL_SIZE, DEFAULT_FRAME_INTERVAL
from anomaly_detection import AnomalyDetector
from anomaly_visualization import AnomalyVisualizer
from rheology import MicrorheologyAnalyzer, create_rheology_plots, display_rheology_summary
from trajectory_heatmap import create_streamlit_heatmap_interface
from state_manager import get_state_manager
from analysis_manager import AnalysisManager
from config_manager import get_config_manager
from channel_manager import channel_manager
from data_access_utils import get_track_data, get_units

# Import unified settings panel and progress utilities
from settings_panel import get_settings_panel, get_global_units
from progress_utils import AnalysisProgress, MultiStepProgress, SimpleProgress

# Import new page modules for multi-page architecture
import importlib

# Attempt to dynamically import optional page modules; if unavailable, mark as not available
PAGES_AVAILABLE = True
try:
    # Import pages package and individual modules dynamically to avoid static import errors
    _mod = importlib.import_module("pages.data_loading")
    show_data_loading_page = getattr(_mod, "show_data_loading_page", None)
    detect_file_format = getattr(_mod, "detect_file_format", None)

    _mod = importlib.import_module("pages.analysis")
    show_analysis_page = getattr(_mod, "show_analysis_page", None)

    _mod = importlib.import_module("pages.visualization")
    show_visualization_page = getattr(_mod, "show_visualization_page", None)

    _mod = importlib.import_module("pages.tracking")
    show_tracking_page = getattr(_mod, "show_tracking_page", None)

    PAGES_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Optional pages module not present — set fallbacks so rest of the app can run
    PAGES_AVAILABLE = False
    show_data_loading_page = None
    detect_file_format = None
    show_analysis_page = None
    show_visualization_page = None
    show_tracking_page = None
from utils import initialize_session_state, calculate_track_statistics, format_track_data, create_analysis_record, sync_global_parameters, get_global_pixel_size, get_global_frame_interval
from special_file_handlers import load_ms2_spots_file, load_imaris_file
from image_processing_utils import (
    apply_noise_reduction, apply_ai_noise_reduction, 
    normalize_image_for_display, create_timepoint_preview, 
    get_image_statistics
)
from visualization import plot_tracks, plot_tracks_3d, plot_track_statistics, plot_motion_analysis, plot_diffusion_coefficients
from segmentation import (
    segment_image_channel_otsu, 
    segment_image_channel_simple_threshold,
    segment_image_channel_adaptive_threshold,
    convert_compartments_to_boundary_crossing_format,
    convert_compartments_to_dwell_time_regions,
    classify_particles_by_compartment,
    analyze_compartment_statistics,
    density_map_threshold,
    enhanced_threshold_image,
    analyze_density_segmentation,
    visualize_density_segmentation,
    gaussian_mixture_segmentation,
    bayesian_gaussian_mixture_segmentation,
    compare_segmentation_methods
)
from multi_channel_analysis import (
    analyze_channel_colocalization,
    analyze_compartment_occupancy,
    compare_channel_dynamics,
    create_multi_channel_visualization
)
from analysis import (
    calculate_msd, analyze_diffusion, analyze_motion, analyze_clustering,
    analyze_dwell_time, load_precalculated_dwell_events, analyze_gel_structure, analyze_diffusion_population,
    analyze_crowding, analyze_active_transport, analyze_boundary_crossing,
    analyze_polymer_physics
)
from ornstein_uhlenbeck_analyzer import analyze_ornstein_uhlenbeck
try:
    from hmm_analysis import fit_hmm, ensure_hmmlearn
except RuntimeError as e:
    fit_hmm = None
    _hmm_err = str(e)
    def _hmm_warning():
        return f"HMM features disabled: {_hmm_err}"
from intensity_analysis import (
    extract_intensity_channels, calculate_movement_metrics,
    correlate_intensity_movement, create_intensity_movement_plots,
    intensity_based_segmentation
)
from unit_converter import UnitConverter
from md_integration import MDSimulation, load_md_file
from biophysics_tab import show_advanced_biophysical_metrics, show_biophysical_models
from simulation import show_simulation_page
from data_quality_checker import DataQualityChecker
from secure_file_validator import get_file_validator
from logic import (
    calculate_population_metrics,
    perform_class_based_analysis,
    apply_mask_to_tracks,
    apply_mask_to_image_analysis,
    plot_density_map,
    convert_coordinates_to_microns,
    normalize_image_for_display,
    process_image_data,
)
from spt2025b.ui.dual_mode import (
    BiologyPreset,
    GuidedInputs,
    TrafficLightStatus,
    UIMode,
    apply_guided_inputs_to_state,
    chatbot_prompt_for_mode,
    deploy_as_guided_protocol,
    eject_to_expert_workspace,
    get_ui_mode,
    get_universal_config,
    init_dual_mode_state,
    load_custom_guided_protocol,
    set_ui_mode,
    set_universal_config,
)

# Import advanced segmentation module
try:
    from advanced_segmentation import integrate_advanced_segmentation_with_app
    ADVANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ADVANCED_SEGMENTATION_AVAILABLE = False

# Import tracking module with proper error handling
try:
    from tracking import detect_particles, link_particles
    TRACKING_MODULE_AVAILABLE = True
except ImportError:
    TRACKING_MODULE_AVAILABLE = False

# Import advanced analysis modules
try:
    from correlative_analysis import CorrelativeAnalyzer, MultiChannelAnalyzer, TemporalCrossCorrelator
    from changepoint_detection import ChangePointDetector
    CORRELATIVE_ANALYSIS_AVAILABLE = True
    CHANGEPOINT_DETECTION_AVAILABLE = True
except ImportError:
    CORRELATIVE_ANALYSIS_AVAILABLE = False
    CHANGEPOINT_DETECTION_AVAILABLE = False

# Import advanced tracking module
try:
    from advanced_tracking import ParticleFilter, AdvancedTracking, bayesian_detection_refinement
    ADVANCED_TRACKING_AVAILABLE = True
except ImportError:
    ADVANCED_TRACKING_AVAILABLE = False

# Import intensity analysis module
try:
    from intensity_analysis import (
        extract_intensity_channels,
        correlate_intensity_movement,
        create_intensity_movement_plots,
        analyze_intensity_profiles,
        classify_intensity_behavior
    )
    INTENSITY_ANALYSIS_AVAILABLE = True
except ImportError:
    INTENSITY_ANALYSIS_AVAILABLE = False

# Import report generation module
try:
    # CORRECTED: Import the EnhancedSPTReportGenerator class from the correct file
    # Import both the report generator class and the helper function that
    # displays the Streamlit interface.
    from enhanced_report_generator import (
        EnhancedSPTReportGenerator,
        show_enhanced_report_generator,
    )
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False

# Remove redundant placeholder functions - these are handled by data_loader.py and special_file_handlers.py

def initialize_mask_tracking():
    """Initialize mask tracking in session state."""
    if 'available_masks' not in st.session_state:
        st.session_state.available_masks = {}
    if 'mask_metadata' not in st.session_state:
        st.session_state.mask_metadata = {}


def store_mask(mask_name: str, mask_data: np.ndarray, mask_type: str, description: str = ""):
    """Store a generated mask for later use in analysis."""
    initialize_mask_tracking()
    
    st.session_state.available_masks[mask_name] = mask_data
    st.session_state.mask_metadata[mask_name] = {
        'type': mask_type,
        'description': description,
        'shape': mask_data.shape,
        'classes': np.unique(mask_data).tolist(),
        'n_classes': len(np.unique(mask_data))
    }


def get_available_masks():
    """Get dictionary of available masks for analysis."""
    initialize_mask_tracking()
    return st.session_state.available_masks


# Removed redundant analyze_boundary_crossings function - using analyze_boundary_crossing from analysis.py instead



def _combine_channels(multichannel_img: np.ndarray, channel_indices: List[int], mode: str = "average", weights: Optional[List[float]] = None) -> np.ndarray:
    """Combine multiple channels from a HxWxC image into a single 2D image.

    Args:
        multichannel_img: 3D array (H, W, C)
        channel_indices: indices of channels to combine
        mode: one of ["average", "max", "min", "sum", "weighted"]
        weights: optional weights for weighted mode (same length as channel_indices)

    Returns:
        2D numpy array (H, W)
    """
    if multichannel_img is None or multichannel_img.ndim != 3 or multichannel_img.shape[2] == 0:
        raise ValueError("Expected a multichannel image of shape (H, W, C)")
    if not channel_indices:
        # default to first channel
        channel_indices = [0]
    # Clip invalid indices
    c = multichannel_img.shape[2]
    channel_indices = [ci for ci in channel_indices if 0 <= ci < c]
    if not channel_indices:
        channel_indices = [0]
    # Extract channels
    stack = np.stack([multichannel_img[:, :, ci] for ci in channel_indices], axis=2)
    if stack.shape[2] == 1:
        return stack[:, :, 0]

    mode = (mode or "average").lower()
    if mode == "max":
        return np.max(stack, axis=2)
    if mode == "min":
        return np.min(stack, axis=2)
    if mode == "sum":
        return np.sum(stack, axis=2)
    if mode == "weighted":
        if not weights or len(weights) != stack.shape[2]:
            # default equal weights
            weights = [1.0 / stack.shape[2]] * stack.shape[2]
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) if np.sum(w) != 0 else 1.0)
        return np.tensordot(stack, w, axes=([2], [0]))
    # default average
    return np.mean(stack, axis=2)


def create_mask_selection_ui(analysis_type: str = ""):
    """Create UI for selecting segmentation method and analyzing all classes from that method."""
    available_masks = get_available_masks()
    
    if not available_masks:
        st.info("No masks available. Generate masks in the Image Processing tab first.")
        return None, [], None
    
    st.markdown("#### Region-Based Analysis")
    
    # Analysis region selection
    analysis_region = st.radio(
        "Analysis Region",
        ["Whole Image", "Segmentation-Based Analysis"],
        help="Choose whether to analyze the entire image or use segmentation regions",
        key=f"analysis_region_{analysis_type}"
    )
    
    if analysis_region == "Whole Image":
        return None, [], None
    
    st.subheader("Segmentation Method Selection")
    
    # Categorize available masks by type
    simple_masks = []
    two_step_masks = []
    density_masks = []
    
    for mask_name in available_masks.keys():
        mask_metadata = st.session_state.mask_metadata[mask_name]
        mask_type = mask_metadata.get('type', 'unknown').lower()
        
        if 'density' in mask_type or 'nuclear density' in mask_type:
            density_masks.append(mask_name)
        elif mask_metadata['n_classes'] >= 3:
            two_step_masks.append(mask_name)
        else:
            simple_masks.append(mask_name)
    
    # Select segmentation method
    segmentation_methods = []
    if simple_masks:
        segmentation_methods.append("Simple Segmentation (Binary)")
    if two_step_masks:
        segmentation_methods.append("Two-Step Segmentation (3 Classes)")
    if density_masks:
        segmentation_methods.append("Nuclear Density Mapping")
    
    if not segmentation_methods:
        st.error("No compatible segmentation masks found.")
        return None, [], None
    
    selected_method = st.selectbox(
        "Choose Segmentation Method",
        segmentation_methods,
        help="Select the type of segmentation to use for analysis. All classes from this method will be analyzed.",
        key=f"segmentation_method_{analysis_type}"
    )
    
    # Select specific mask based on method
    if selected_method == "Simple Segmentation (Binary)":
        available_for_method = simple_masks
        expected_classes = [0, 1]  # Background, Nucleus
        class_names = ["Background", "Nucleus"]
    elif selected_method == "Two-Step Segmentation (3 Classes)":
        available_for_method = two_step_masks
        
        # Add analysis options for Two-Step Segmentation
        st.subheader("Two-Step Segmentation Analysis Options")
        analysis_option = st.selectbox(
            "Select 2-Step Segmentation Analysis Method",
            [
                "Analyze all three classes separately",
                "Analyze classes separately, then combine Class 1 and 2"
            ],
            help="Choose how to analyze the three segmentation classes",
            key=f"two_step_analysis_option_{analysis_type}"
        )
        
        if analysis_option == "Analyze all three classes separately":
            expected_classes = [0, 1, 2]  # Background, Class 1, Class 2
            class_names = ["Background", "Class 1", "Class 2"]
        else:  # "Analyze classes separately, then combine Class 1 and 2"
            expected_classes = [0, 1, 2]  # Still analyze all classes first
            class_names = ["Background", "Class 1", "Class 2", "Combined Class 1+2"]
    else:  # Nuclear Density Mapping
        available_for_method = density_masks
        expected_classes = [0, 1, 2]  # Background, Low Density, High Density
        class_names = ["Background", "Low Density", "High Density"]
    
    # Automatically select the first available mask for the method
    if not available_for_method:
        st.error(f"No masks available for {selected_method}")
        return None, [], None
    
    selected_mask = available_for_method[0]  # Use the first (and typically only) available mask
    
    # Only show dropdown if multiple masks are available for this method
    if len(available_for_method) > 1:
        selected_mask = st.selectbox(
            f"Select {selected_method} Mask",
            available_for_method,
            help=f"Choose the specific mask for {selected_method.lower()}",
            key=f"specific_mask_{analysis_type}"
        )
    else:
        # Show which mask is being used automatically
        st.info(f"Using mask: **{selected_mask}**")
    
    # Show mask information
    mask_metadata = st.session_state.mask_metadata[selected_mask]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Type:** {mask_metadata['type']}")
    with col2:
        st.write(f"**Classes:** {mask_metadata['n_classes']}")
    with col3:
        st.write(f"**Shape:** {mask_metadata['shape']}")
    
    if mask_metadata['description']:
        st.write(f"**Description:** {mask_metadata['description']}")
    
    # Display analysis plan
    st.info(f"Analysis will be performed separately for each class: {', '.join(class_names)}")
    
    # Return all classes for comprehensive analysis
    return [selected_mask], {selected_mask: expected_classes}, selected_method






def detection_parameter_tuning_section():
    """Interactive detection parameter tuning with real-time threshold visualization"""
    st.subheader("🔍 Particle Detection Parameter Tuning")

    if 'image_data' in st.session_state and st.session_state.image_data:
        num_frames = len(st.session_state.image_data)

        if num_frames > 0:
            # Frame selection for parameter testing
            selected_frame_index = 0
            if num_frames > 1:
                selected_frame_index = st.slider(
                    "Test Frame",
                    min_value=0,
                    max_value=num_frames - 1,
                    value=st.session_state.get('detection_test_frame', 0),
                    key="detection_test_frame_slider",
                    help=f"Select frame to test detection parameters (total: {num_frames} frames)"
                )
                st.session_state.detection_test_frame = selected_frame_index

            # Get the test frame
            test_frame = st.session_state.image_data[selected_frame_index]
            
            # Detection parameter controls
            st.markdown("### Detection Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                # Intensity threshold with real-time adjustment
                frame_min, frame_max = np.min(test_frame), np.max(test_frame)
                frame_mean = np.mean(test_frame)
                
                threshold_value = st.slider(
                    "Intensity Threshold",
                    min_value=int(frame_min),
                    max_value=int(frame_max),
                    value=int(frame_mean + np.std(test_frame)),
                    help=f"""Intensity cutoff for particle detection:
• Low values: Detect dim particles but include more noise
• High values: Detect only bright particles, may miss dim ones
• Frame range: {frame_min:.1f} - {frame_max:.1f}
• Default (mean + std): {int(frame_mean + np.std(test_frame))}

Adjust based on particle brightness relative to background."""
                )
                
                min_particle_size = st.slider("Min Particle Size (pixels)", 1, 50, 5,
                                             help="""Minimum area for detected particles:
• 1-3: Very small particles, single pixels (may include noise)
• 4-10: Small particles, typical for high-resolution imaging
• 11-20: Medium particles, standard for most applications
• 21+: Large particles or low-resolution imaging

Filters out noise and artifacts smaller than expected particle size.""")
                
            with col2:
                max_particle_size = st.slider("Max Particle Size (pixels)", 10, 500, 100,
                                             help="""Maximum area for detected particles:
• 10-50: Small particles, prevents detection of large artifacts
• 51-150: Medium particles, standard range for most applications
• 151-300: Large particles or aggregates
• 300+: Very large objects, may include cell boundaries or debris

Filters out oversized objects that are unlikely to be genuine particles.""")
                detection_method = st.selectbox("Detection Method", 
                                               ["Intensity Threshold", "Local Maxima", "Blob Detection (DoG)", "Adaptive Threshold"],
                                               help="""Algorithm for particle detection:
• Intensity Threshold: Simple cutoff based on pixel intensity (fast, good for bright particles)
• Local Maxima: Finds intensity peaks (good for well-separated particles)
• Blob Detection (DoG): Difference of Gaussians, detects blob-like structures (robust for various sizes)
• Adaptive Threshold: Adjusts threshold locally (good for uneven illumination)

DoG is recommended for most SPT applications due to robustness.""")
                
                noise_reduction_strength = st.slider("Noise Reduction", 0.0, 2.0, 0.5, 0.1,
                                                    help="""Gaussian smoothing strength for noise reduction:
• 0.0: No smoothing, preserves all details but may include noise
• 0.1-0.5: Light smoothing, reduces noise while preserving particle features
• 0.6-1.0: Moderate smoothing, good balance for most images
• 1.1-2.0: Heavy smoothing, removes significant noise but may blur particles

Higher values reduce noise but may merge nearby particles.""")

            # Apply threshold visualization
            st.markdown("### Threshold Preview")
            
            # Create thresholded preview
            try:
                # Apply noise reduction if selected
                preview_frame = test_frame.copy()
                if noise_reduction_strength > 0:
                    from image_processing_utils import apply_noise_reduction
                    preview_frame = apply_noise_reduction([preview_frame], 'gaussian', {'sigma': noise_reduction_strength})[0]
                
                # Apply threshold
                thresholded = preview_frame > threshold_value
                
                # Create side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Frame**")
                    display_original = normalize_image_for_display(preview_frame)
                    st.image(display_original, caption=f"Frame {selected_frame_index}", use_container_width=True)
                    
                with col2:
                    st.write("**Threshold Result**")
                    # Convert boolean to uint8 for display
                    threshold_display = (thresholded * 255).astype(np.uint8)
                    st.image(threshold_display, caption=f"Threshold: {threshold_value}", use_container_width=True)
                
                # Display detection statistics
                particle_pixels = np.sum(thresholded)
                total_pixels = thresholded.size
                detection_percentage = (particle_pixels / total_pixels) * 100
                
                st.markdown("### Detection Statistics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Particle Pixels", f"{particle_pixels:,}")
                with metric_col2:
                    st.metric("Detection %", f"{detection_percentage:.2f}%")
                with metric_col3:
                    # Estimate potential particle count (rough approximation)
                    if min_particle_size > 0:
                        est_particles = particle_pixels // (min_particle_size ** 2)
                        st.metric("Est. Particles", f"{est_particles}")
                
                # Frame statistics
                with st.expander("Frame Statistics", expanded=False):
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.write(f"**Size:** {test_frame.shape}")
                        st.write(f"**Data type:** {test_frame.dtype}")
                        st.write(f"**Min intensity:** {frame_min:.2f}")
                        st.write(f"**Max intensity:** {frame_max:.2f}")
                    with stat_col2:
                        st.write(f"**Mean intensity:** {frame_mean:.2f}")
                        st.write(f"**Std intensity:** {np.std(test_frame):.2f}")
                        st.write(f"**Median intensity:** {np.median(test_frame):.2f}")
                        st.write(f"**99th percentile:** {np.percentile(test_frame, 99):.2f}")

            except Exception as e:
                st.error(f"Error in threshold preview: {str(e)}")
                # Fallback to simple display
                display_image = normalize_image_for_display(test_frame)
                st.image(display_image, caption=f"Frame {selected_frame_index}", use_container_width=True)

        else:
            st.warning("No image frames are currently loaded or available for preview.")
    else:
        st.info("Load an image sequence to start detection parameter tuning.")


def preprocessing_options_ui():
    """UI for image preprocessing options including noise reduction"""
    images_loaded = 'image_data' in st.session_state and st.session_state.image_data is not None
    
    st.sidebar.subheader("🔧 Image Preprocessing")

    if not images_loaded:
        st.sidebar.info("Load images to enable preprocessing options.")
        return None, None, None, None, None, None

    # --- Image Statistics ---
    if st.sidebar.checkbox("Show Image Statistics"):
        with st.sidebar.expander("Image Info"):
            stats = get_image_statistics(st.session_state.image_data)
            if stats:
                st.sidebar.write(f"Frames: {stats['num_frames']}")
                if stats['frame_shapes']:
                    st.sidebar.write(f"Shape: {stats['frame_shapes'][0]}")
                st.sidebar.write(f"Data types: {set(stats['data_types'])}")

    # --- Standard Noise Reduction ---
    apply_std_denoising = st.sidebar.checkbox("Apply Standard Noise Reduction", key="apply_std_denoising_cb")
    std_denoising_method = None
    std_denoising_params = {}

    if apply_std_denoising:
        std_denoising_method = st.sidebar.selectbox(
            "Method",
            ['gaussian', 'median', 'nl_means'],
            key="std_denoising_method_select"
        )

        if std_denoising_method == 'gaussian':
            gauss_sigma = st.sidebar.slider("Gaussian Sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="gauss_sigma_slider")
            std_denoising_params = {'sigma': gauss_sigma}
            
        elif std_denoising_method == 'median':
            median_radius = st.sidebar.slider("Median Filter Disk Radius", min_value=1, max_value=10, value=2, step=1, key="median_radius_slider")
            std_denoising_params = {'disk_radius': median_radius}
            
        elif std_denoising_method == 'nl_means':
            nl_h = st.sidebar.slider("NL Means `h` (filter strength)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="nl_h_slider")
            nl_sigma = st.sidebar.slider("NL Means `sigma` (noise std dev)", min_value=0.01, max_value=0.5, value=0.08, step=0.01, key="nl_sigma_slider")
            nl_patch_size = st.sidebar.number_input("NL Means Patch Size", min_value=3, max_value=15, value=5, step=2, key="nl_psize_input", help="Must be odd.")
            nl_patch_distance = st.sidebar.number_input("NL Means Patch Distance", min_value=1, max_value=21, value=6, step=1, key="nl_pdist_input")
            
            # Ensure patch_size is odd
            if nl_patch_size % 2 == 0:
                nl_patch_size += 1
                st.sidebar.caption(f"Patch size adjusted to {nl_patch_size} (must be odd).")
            std_denoising_params = {'h': nl_h, 'sigma': nl_sigma, 'patch_size': nl_patch_size, 'patch_distance': nl_patch_distance}

    # --- AI-based Noise Reduction ---
    apply_ai_denoising = st.sidebar.checkbox("Apply AI Noise Reduction (Experimental)", key="apply_ai_denoising_cb")
    ai_denoising_choice = None
    ai_denoising_params = {}

    if apply_ai_denoising:
        st.sidebar.info("AI models require pre-training and specific configurations.")
        ai_denoising_choice = st.sidebar.selectbox(
            "AI Model",
            ["noise2void", "care"], 
            key="ai_denoising_model_select"
        )

    return apply_std_denoising, std_denoising_method, std_denoising_params, apply_ai_denoising, ai_denoising_choice, ai_denoising_params


def apply_preprocessing_pipeline():
    """Apply the selected preprocessing steps to loaded images"""
    if 'image_data' not in st.session_state or not st.session_state.image_data:
        return
    
    # Get preprocessing settings
    std_denoise, std_method, std_params, ai_denoise, ai_model, ai_params = preprocessing_options_ui()
    
    processed_images = st.session_state.image_data.copy()
    
    # Apply standard noise reduction
    if std_denoise and std_method:
        with st.spinner(f"Applying {std_method} noise reduction..."):
            params = std_params if std_params is not None else {}
            processed_images = apply_noise_reduction(processed_images, method=std_method, params=params)
            st.session_state.denoised_std_images = processed_images
        st.success(f"{std_method} noise reduction applied.")
    
    # Apply AI-based noise reduction
    if ai_denoise and ai_model:
        with st.spinner(f"Applying AI noise reduction ({ai_model})..."):
            params = ai_params if ai_params is not None else {}
            processed_images = apply_ai_noise_reduction(processed_images, model_choice=ai_model, params=params)
            st.session_state.denoised_ai_images = processed_images
    
    # Store final processed images
    st.session_state.images_for_tracking = processed_images
    


# Try to import biophysical models
try:
        BIOPHYSICAL_MODELS_AVAILABLE = True
except ImportError:
    BIOPHYSICAL_MODELS_AVAILABLE = False

# Try to import change point detection
try:
        CHANGEPOINT_DETECTION_AVAILABLE = True
except ImportError:
    CHANGEPOINT_DETECTION_AVAILABLE = False

# Try to import correlative analysis
try:
        CORRELATIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    CORRELATIVE_ANALYSIS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="SPT Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
initialize_session_state()

# Initialize unit converter
if 'unit_converter' not in st.session_state:
    st.session_state.unit_converter = UnitConverter()
    # Set default values
    st.session_state.unit_converter.set_pixel_size(0.1)  # 0.1 μm/pixel
    st.session_state.unit_converter.set_frame_interval(0.03)  # 0.03 s/frame
    
# Initialize unit values for consistent access across all analyses
if 'current_pixel_size' not in st.session_state:
    st.session_state.current_pixel_size = st.session_state.unit_converter.pixel_size
if 'current_frame_interval' not in st.session_state:
    st.session_state.current_frame_interval = st.session_state.unit_converter.frame_interval

# Helper function to get current unit settings 


def get_current_units():
    """Get the currently set unit values for use in all analyses."""
    return {
        'pixel_size': st.session_state.current_pixel_size,
        'frame_interval': st.session_state.current_frame_interval
    }


def handle_track_upload(uploaded_file):
    """
    Wrapper for file upload -> load -> persist.
    Call wherever the upload widget processes a new file.
    """
    if not uploaded_file:
        return None
    import io, os
    name = getattr(uploaded_file, "name", "uploaded_tracks")
    path_hint = name
    # Detect Excel vs CSV by extension
    data_bytes = uploaded_file.read()
    bio = io.BytesIO(data_bytes)
    if name.lower().endswith((".xls", ".xlsx")):
        import pandas as pd
        df = pd.read_excel(bio, engine="openpyxl")
    else:
        import pandas as pd
        df = pd.read_csv(bio)
    # Reuse loader cleaning / persistence pipeline
    from data_loader import load_tracks_file
    # Provide a pseudo path string for metadata
    df_clean = load_tracks_file(path_hint, persist=True, state_manager=state_manager, raw_df=df)
    return df_clean

def get_active_tracks():
    sm = state_manager
    return sm.get_tracks_or_none()


def _generate_batch_html_report(report_results: Dict, 
                                condition_datasets: Dict[str, pd.DataFrame],
                                pixel_size: float,
                                frame_interval: float,
                                interactive: bool = True) -> bytes:
    """
    Generate HTML report for batch condition analysis.
    
    Parameters
    ----------
    report_results : Dict
        Results from generate_condition_reports
    condition_datasets : Dict[str, pd.DataFrame]
        Original condition datasets
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Frame interval in seconds
    interactive : bool
        If True, include interactive Plotly figures; if False, convert to static images
        
    Returns
    -------
    bytes
        HTML report as bytes
    """
    import html
    import plotly.io as pio
    from datetime import datetime as _dt
    
    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>SPT Batch Analysis Report</title>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    
    # Enhanced CSS styling
    parts.append("""
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
        h3 { color: #7f8c8d; }
        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metadata ul { list-style: none; padding: 0; }
        .metadata li { padding: 5px 0; }
        .condition-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }
        .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .metrics-table th { background: #3498db; color: white; padding: 10px; text-align: left; }
        .metrics-table td { padding: 8px; border-bottom: 1px solid #ddd; }
        .metrics-table tr:hover { background: #f0f0f0; }
        .figure { margin: 20px 0; text-align: center; }
        .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: #27ae60; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        .summary-card { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); min-width: 200px; }
        .summary-card h4 { margin: 0 0 10px 0; color: #3498db; }
        .summary-card .value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .comparison-section { background: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }
        code { background: #f7f7f7; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
    </style>
    """)
    
    if interactive:
        parts.append("<script src='https://cdn.plot.ly/plotly-2.18.0.min.js'></script>")
    
    parts.append("</head><body><div class='container'>")
    
    # Header
    parts.append(f"<h1>📊 SPT Batch Analysis Report</h1>")
    
    # Metadata section
    parts.append("<div class='metadata'>")
    parts.append("<h3>Report Metadata</h3>")
    parts.append("<ul>")
    parts.append(f"<li><b>Generated:</b> {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}</li>")
    parts.append(f"<li><b>Pixel Size:</b> {pixel_size} µm</li>")
    parts.append(f"<li><b>Frame Interval:</b> {frame_interval} s</li>")
    parts.append(f"<li><b>Number of Conditions:</b> {len(condition_datasets)}</li>")
    parts.append(f"<li><b>Report Type:</b> {'Interactive' if interactive else 'Static'} HTML</li>")
    parts.append("</ul>")
    parts.append("</div>")
    
    # Summary overview
    parts.append("<h2>📈 Summary Overview</h2>")
    parts.append("<div>")
    for cond_name, tracks_df in condition_datasets.items():
        n_tracks = tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 0
        n_points = len(tracks_df)
        parts.append(f"""
        <div class='summary-card'>
            <h4>{html.escape(cond_name)}</h4>
            <div><b>Tracks:</b> <span class='value'>{n_tracks}</span></div>
            <div><b>Data Points:</b> {n_points:,}</div>
        </div>
        """)
    parts.append("</div><div style='clear:both;'></div>")
    
    # Individual condition results
    parts.append("<h2>🔬 Condition Analysis Results</h2>")
    for cond_name, cond_result in report_results.get('conditions', {}).items():
        parts.append(f"<div class='condition-section'>")
        parts.append(f"<h3>Condition: {html.escape(cond_name)}</h3>")
        
        if cond_result.get('success', False):
            parts.append(f"<p class='success'>✅ Analysis completed successfully</p>")
            parts.append(f"<p><b>Analyses performed:</b> {len(cond_result.get('analysis_results', {}))}</p>")
            parts.append(f"<p><b>Figures generated:</b> {len(cond_result.get('figures', {}))}</p>")
            
            # Display figures
            for analysis_key, fig in cond_result.get('figures', {}).items():
                if fig:
                    parts.append(f"<div class='figure'>")
                    parts.append(f"<h4>{html.escape(analysis_key.replace('_', ' ').title())}</h4>")
                    try:
                        if interactive:
                            # Include interactive Plotly figure
                            fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
                            parts.append(fig_html)
                        else:
                            # Convert to static image
                            import base64
                            img_bytes = pio.to_image(fig, format='png', width=1000, height=600)
                            b64 = base64.b64encode(img_bytes).decode('utf-8')
                            parts.append(f"<img src='data:image/png;base64,{b64}' alt='{analysis_key}'>")
                    except Exception as e:
                        parts.append(f"<p class='error'>Error rendering figure: {html.escape(str(e))}</p>")
                    parts.append("</div>")
        else:
            parts.append(f"<p class='error'>❌ Analysis failed: {html.escape(cond_result.get('error', 'Unknown error'))}</p>")
        
        parts.append("</div>")
    
    # Comparison results
    comparisons = report_results.get('comparisons', {})
    if comparisons and comparisons.get('success', False) and len(condition_datasets) >= 2:
        parts.append("<div class='comparison-section'>")
        parts.append("<h2>📊 Statistical Comparisons</h2>")
        
        # Metrics table
        if 'metrics' in comparisons:
            parts.append("<h3>Summary Metrics by Condition</h3>")
            parts.append("<table class='metrics-table'>")
            parts.append("<tr><th>Condition</th><th>Mean Track Length</th><th>Mean Displacement (µm)</th><th>Mean Velocity (µm/s)</th></tr>")
            for cond_name, metrics in comparisons['metrics'].items():
                parts.append(f"<tr>")
                parts.append(f"<td><b>{html.escape(cond_name)}</b></td>")
                parts.append(f"<td>{metrics.get('mean_track_length', 0):.2f}</td>")
                parts.append(f"<td>{metrics.get('mean_displacement', 0):.4f}</td>")
                parts.append(f"<td>{metrics.get('mean_velocity', 0):.4f}</td>")
                parts.append(f"</tr>")
            parts.append("</table>")
        
        # Statistical tests
        if 'statistical_tests' in comparisons and comparisons['statistical_tests']:
            parts.append("<h3>Pairwise Statistical Tests</h3>")
            for comparison, tests in comparisons['statistical_tests'].items():
                parts.append(f"<h4>{html.escape(comparison)}</h4>")
                parts.append("<table class='metrics-table'>")
                parts.append("<tr><th>Metric</th><th>t-test p-value</th><th>Mann-Whitney p-value</th><th>Significant?</th></tr>")
                for metric, test_results in tests.items():
                    t_test_p = test_results.get('t_test', {}).get('p_value', 'N/A')
                    mw_p = test_results.get('mann_whitney', {}).get('p_value', 'N/A')
                    significant = test_results.get('significant', False)
                    sig_text = "✅ Yes" if significant else "❌ No"
                    
                    parts.append(f"<tr>")
                    parts.append(f"<td>{html.escape(metric.replace('_', ' ').title())}</td>")
                    parts.append(f"<td>{t_test_p if isinstance(t_test_p, str) else f'{t_test_p:.4f}'}</td>")
                    parts.append(f"<td>{mw_p if isinstance(mw_p, str) else f'{mw_p:.4f}'}</td>")
                    parts.append(f"<td>{sig_text}</td>")
                    parts.append(f"</tr>")
                parts.append("</table>")
        
        # Comparison figures
        if 'figures' in comparisons and comparisons['figures'].get('comparison_boxplots'):
            parts.append("<h3>Comparison Visualizations</h3>")
            parts.append("<div class='figure'>")
            try:
                fig = comparisons['figures']['comparison_boxplots']
                if interactive:
                    fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
                    parts.append(fig_html)
                else:
                    import base64
                    img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    parts.append(f"<img src='data:image/png;base64,{b64}' alt='Comparison Boxplots'>")
            except Exception as e:
                parts.append(f"<p class='error'>Error rendering comparison figure: {html.escape(str(e))}</p>")
            parts.append("</div>")
        
        parts.append("</div>")
    
    # Footer
    parts.append("<hr style='margin-top: 40px;'>")
    parts.append("<p style='text-align: center; color: #7f8c8d;'>")
    parts.append("Generated by SPT2025B Analysis Platform | ")
    parts.append(f"{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("</p>")
    
    parts.append("</div></body></html>")
    
    html_str = "".join(parts)
    return html_str.encode('utf-8')


# Define navigation function
def navigate_to(page):
    st.session_state.active_page = page
    # Note: st.rerun() not needed here - Streamlit automatically reruns after callback


def _traffic_light_from_quality_score(score: float) -> TrafficLightStatus:
    """Map quality score (0-100) to traffic-light categories."""
    if score >= 80.0:
        return TrafficLightStatus.GREEN
    if score >= 60.0:
        return TrafficLightStatus.YELLOW
    return TrafficLightStatus.RED


def _traffic_light_icon(status: TrafficLightStatus) -> str:
    return {
        TrafficLightStatus.GREEN: "🟢",
        TrafficLightStatus.YELLOW: "🟡",
        TrafficLightStatus.RED: "🔴",
    }[status]


def _run_guided_quality_check() -> Optional[Dict[str, Any]]:
    """Run quality checks on loaded tracks and return a compact traffic-light result."""
    tracks_df, has_data = get_track_data()
    if not has_data or tracks_df is None or tracks_df.empty:
        return None

    checker = DataQualityChecker()
    pixel_size = float(st.session_state.get("pixel_size", DEFAULT_PIXEL_SIZE))
    frame_interval = float(st.session_state.get("frame_interval", DEFAULT_FRAME_INTERVAL))
    report = checker.run_all_checks(
        tracks_df=tracks_df,
        pixel_size=pixel_size,
        frame_interval=frame_interval,
    )

    status = _traffic_light_from_quality_score(float(report.overall_score))
    status_message = {
        TrafficLightStatus.GREEN: "Excellent signal quality and track continuity.",
        TrafficLightStatus.YELLOW: "Moderate quality issues detected; noise correction is recommended.",
        TrafficLightStatus.RED: "Low data quality detected; expert review is recommended before batch runs.",
    }[status]

    return {
        "status": status.value,
        "score": float(report.overall_score),
        "message": status_message,
        "failed_checks": int(report.failed_checks),
        "warnings": int(report.warnings),
        "timestamp": datetime.now().isoformat(),
    }


def _sync_runtime_units(pixel_size_um: float, frame_interval_s: float) -> None:
    """Keep legacy unit keys and global settings aligned with shared config units."""
    st.session_state.pixel_size = float(pixel_size_um)
    st.session_state.frame_interval = float(frame_interval_s)
    st.session_state.current_pixel_size = float(pixel_size_um)
    st.session_state.current_frame_interval = float(frame_interval_s)
    if "unit_converter" in st.session_state:
        st.session_state.unit_converter.set_pixel_size(float(pixel_size_um))
        st.session_state.unit_converter.set_frame_interval(float(frame_interval_s))

    global_settings = st.session_state.get("global_settings")
    if global_settings is not None:
        try:
            global_settings.pixel_size = float(pixel_size_um)
            global_settings.frame_interval = float(frame_interval_s)
        except Exception:
            pass


def _show_dual_mode_sidebar() -> None:
    """Render guided/expert controls that share one backend configuration."""
    init_dual_mode_state(st.session_state)
    current_mode = get_ui_mode(st.session_state)

    mode_labels = {
        UIMode.GUIDED: "🎓 Guided Mode",
        UIMode.EXPERT: "⚙️ Expert Workspace",
    }
    ordered_modes = [UIMode.GUIDED, UIMode.EXPERT]
    selected_label = st.sidebar.radio(
        "Interface",
        [mode_labels[m] for m in ordered_modes],
        index=ordered_modes.index(current_mode),
        key="dual_mode_selector",
        help="Guided mode uses biology presets; Expert mode gives direct parameter control.",
    )
    selected_mode = ordered_modes[[mode_labels[m] for m in ordered_modes].index(selected_label)]
    if selected_mode != current_mode:
        set_ui_mode(selected_mode, st.session_state)
        st.rerun()

    st.session_state["chatbot_system_prompt"] = chatbot_prompt_for_mode(selected_mode)

    try:
        config = get_universal_config(st.session_state)
    except Exception:
        config = set_universal_config({}, st.session_state)
    guided_cache = st.session_state.get("guided_inputs", {})

    if selected_mode == UIMode.GUIDED:
        st.sidebar.caption("Guided wizard: map biology context to validated expert defaults.")

        preset_labels = {
            "Membrane Receptors": BiologyPreset.MEMBRANE_RECEPTOR,
            "Chromatin/DNA Binding": BiologyPreset.CHROMATIN_DNA_BINDING,
            "Cytosolic Proteins": BiologyPreset.CYTOSOLIC_PROTEIN,
        }
        reverse_preset = {v.value: k for k, v in preset_labels.items()}
        default_preset_key = guided_cache.get(
            "biology_preset", BiologyPreset.MEMBRANE_RECEPTOR.value
        )
        default_preset_label = reverse_preset.get(default_preset_key, "Membrane Receptors")
        chosen_label = st.sidebar.selectbox(
            "What are you imaging?",
            options=list(preset_labels.keys()),
            index=list(preset_labels.keys()).index(default_preset_label),
            key="guided_preset_select",
        )
        chosen_preset = preset_labels[chosen_label]

        if st.sidebar.button("Run Traffic-Light Check", key="guided_quality_check_btn"):
            quality_summary = _run_guided_quality_check()
            if quality_summary is None:
                st.sidebar.warning("Load track data first to run quality checks.")
            else:
                st.session_state["guided_quality_summary"] = quality_summary

        quality_summary = st.session_state.get("guided_quality_summary")
        auto_traffic = TrafficLightStatus.GREEN
        if isinstance(quality_summary, dict) and "status" in quality_summary:
            auto_traffic = TrafficLightStatus(str(quality_summary["status"]))
            st.sidebar.info(
                f"{_traffic_light_icon(auto_traffic)} Score {quality_summary.get('score', 0.0):.1f}/100\n\n"
                f"{quality_summary.get('message', '')}"
            )

        traffic_options = {
            "🟢 Green (Excellent)": TrafficLightStatus.GREEN,
            "🟡 Yellow (Use Noise Correction)": TrafficLightStatus.YELLOW,
            "🔴 Red (Needs Expert Review)": TrafficLightStatus.RED,
        }
        selected_traffic = st.sidebar.selectbox(
            "Data quality status",
            options=list(traffic_options.keys()),
            index=list(traffic_options.values()).index(auto_traffic),
            key="guided_traffic_select",
        )
        traffic_status = traffic_options[selected_traffic]

        if st.sidebar.button("Apply Guided Preset", type="primary", key="apply_guided_preset_btn"):
            guided = GuidedInputs(
                biology_preset=chosen_preset,
                traffic_light=traffic_status,
                pixel_size_um=float(st.session_state.get("pixel_size", DEFAULT_PIXEL_SIZE)),
                frame_interval_s=float(st.session_state.get("frame_interval", DEFAULT_FRAME_INTERVAL)),
            )
            config = apply_guided_inputs_to_state(guided, st.session_state)
            _sync_runtime_units(config.pixel_size_um, config.frame_interval_s)
            st.sidebar.success("Guided preset applied to shared expert configuration.")

        custom_protocol_dir = (
            Path(__file__).resolve().parent / "spt2025b" / "ui" / "protocols" / "custom"
        )
        custom_protocols = sorted(custom_protocol_dir.glob("*.json")) if custom_protocol_dir.exists() else []
        if custom_protocols:
            selected_protocol = st.sidebar.selectbox(
                "Load Custom Protocol",
                options=custom_protocols,
                format_func=lambda p: p.name,
                key="guided_custom_protocol_select",
            )
            if st.sidebar.button("Load Custom Protocol", key="guided_load_custom_protocol_btn"):
                loaded_cfg = load_custom_guided_protocol(selected_protocol)
                set_universal_config(loaded_cfg, st.session_state)
                _sync_runtime_units(loaded_cfg.pixel_size_um, loaded_cfg.frame_interval_s)
                st.session_state["guided_inputs"] = {
                    "biology_preset": chosen_preset.value,
                    "traffic_light": traffic_status.value,
                    "pixel_size_um": loaded_cfg.pixel_size_um,
                    "frame_interval_s": loaded_cfg.frame_interval_s,
                    "custom_protocol": str(selected_protocol),
                }
                st.sidebar.success(f"Loaded protocol: {selected_protocol.name}")

        if st.sidebar.button("Eject to Expert Workspace", key="guided_eject_to_expert_btn"):
            eject_to_expert_workspace(st.session_state)
            st.rerun()

    else:
        st.sidebar.caption("Expert controls are writing directly to the shared backend state.")
        with st.sidebar.expander("Expert Quick Controls", expanded=False):
            msd_fit_fraction = st.number_input(
                "MSD Fit Fraction",
                min_value=0.01,
                max_value=1.0,
                value=float(config.msd_fit_fraction),
                step=0.01,
                key="expert_msd_fit_fraction",
                help="Fraction of each trajectory used for MSD fitting.",
            )
            hmm_state_count = st.number_input(
                "HMM States",
                min_value=1,
                max_value=10,
                value=int(config.hmm_state_count),
                step=1,
                key="expert_hmm_state_count",
            )
            search_radius_px = st.number_input(
                "Search Radius (px)",
                min_value=0.1,
                max_value=100.0,
                value=float(config.search_radius_px),
                step=0.1,
                key="expert_search_radius_px",
            )
            if st.button("Update Expert Config", key="expert_update_config_btn"):
                config.msd_fit_fraction = float(msd_fit_fraction)
                config.hmm_state_count = int(hmm_state_count)
                config.search_radius_px = float(search_radius_px)
                set_universal_config(config, st.session_state)
                st.sidebar.success("Expert parameters updated.")

        with st.sidebar.expander("Deploy to Guided Protocol", expanded=False):
            default_protocol_name = f"custom_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            protocol_name = st.text_input(
                "Protocol Name",
                value=default_protocol_name,
                key="expert_protocol_name_input",
            )
            protocol_description = st.text_area(
                "Description",
                value="Expert-tuned protocol exported from Expert Workspace.",
                key="expert_protocol_desc_input",
            )
            if st.button("Save as Guided Protocol", key="expert_save_guided_protocol_btn"):
                saved_path = deploy_as_guided_protocol(
                    get_universal_config(st.session_state),
                    name=protocol_name.strip() or default_protocol_name,
                    description=protocol_description.strip(),
                )
                st.sidebar.success(f"Saved: {saved_path}")

    with st.sidebar.expander("Shared Config Snapshot", expanded=False):
        cfg_dict = get_universal_config(st.session_state).to_dict()
        summary_keys = [
            "protocol_name",
            "dimensionality",
            "msd_fit_fraction",
            "hmm_state_count",
            "search_radius_px",
            "max_jump_um",
            "pixel_size_um",
            "frame_interval_s",
        ]
        st.json({k: cfg_dict.get(k) for k in summary_keys})


# Initialize the centralized state and analysis managers
state_manager = get_state_manager()
analysis_manager = AnalysisManager()

# Expose managers in session state for components that expect them
if 'app_state' not in st.session_state:
    st.session_state.app_state = state_manager
if 'analysis_manager' not in st.session_state:
    st.session_state.analysis_manager = analysis_manager

# Sidebar navigation
st.sidebar.title("SPT Analysis")
#st.sidebar.image("generated-icon.png", width=100)
_show_dual_mode_sidebar()
st.sidebar.divider()

# Main navigation menu - Updated for multi-page architecture
nav_options = [
    "Home", "Project Management", "Data Loading", "Image Processing", "Tracking",
    "Analysis", "Visualization", "Advanced Analysis", "AI Anomaly Detection", "Report Generation", "MD Integration", "Simulation"
]

# Set the index based on active_page to sync programmatic navigation with sidebar
try:
    default_index = nav_options.index(st.session_state.active_page) if st.session_state.active_page in nav_options else 0
except (ValueError, AttributeError):
    default_index = 0

nav_option = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=default_index
)

# Update session state based on navigation
if nav_option != st.session_state.active_page:
    st.session_state.active_page = nav_option

# Initialize Project Management features
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False
if "confirm_delete_condition" not in st.session_state:
    st.session_state.confirm_delete_condition = {}
if "confirm_delete_file" not in st.session_state:
    st.session_state.confirm_delete_file = False

# Ensure detection-related session keys exist to avoid AttributeError before first use
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = {}
if 'all_detections' not in st.session_state:
    st.session_state.all_detections = {}

# Load sample data option in sidebar
with st.sidebar.expander("Sample Data"):
    # Scan for available sample datasets
    sample_data_dir = "sample data"
    sample_datasets = {}
    
    if os.path.exists(sample_data_dir):
        for subdir in os.listdir(sample_data_dir):
            subdir_path = os.path.join(sample_data_dir, subdir)
            if os.path.isdir(subdir_path):
                csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    display_name = f"{subdir}/{csv_file}"
                    sample_datasets[display_name] = os.path.join(subdir_path, csv_file)
    
    if sample_datasets:
        selected_sample = st.selectbox(
            "Select Sample Dataset",
            options=list(sample_datasets.keys()),
            help="Choose from available sample datasets in the repository"
        )
        
        if st.button("Load Selected Sample"):
            try:
                sample_file_path = sample_datasets[selected_sample]
                if os.path.exists(sample_file_path):
                    st.session_state.tracks_data = pd.read_csv(sample_file_path)
                    # Format to standard format
                    st.session_state.tracks_data = format_track_data(st.session_state.tracks_data)
                    # Calculate track statistics
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.sidebar.success(f"Loaded: {selected_sample}")
                else:
                    st.sidebar.warning("Sample data file not found.")
            except Exception as e:
                st.sidebar.error(f"Error loading sample data: {str(e)}")
    else:
        st.sidebar.info("No sample data found in 'sample data' folder")

# Display data status in sidebar
with st.sidebar.expander("Data Status"):
    if st.session_state.tracks_data is not None:
        try:
            if isinstance(st.session_state.tracks_data, pd.DataFrame) and not st.session_state.tracks_data.empty:
                if 'track_id' in st.session_state.tracks_data.columns:
                    st.write(f"Tracks loaded: {len(st.session_state.tracks_data['track_id'].unique())} tracks, {len(st.session_state.tracks_data)} points")
                else:
                    st.write(f"Data loaded: {len(st.session_state.tracks_data)} rows")
                    st.write("Available columns: " + ", ".join(st.session_state.tracks_data.columns.tolist()))
            else:
                st.write("Track data format not recognized")
        except Exception as e:
            st.write(f"Track data present but format issue: {str(e)}")
    else:
        st.write("No track data loaded.")
        
    if st.session_state.image_data is not None:
        st.write(f"Images loaded: {len(st.session_state.image_data)} frames")
    else:
        st.write("No image data loaded.")

# Initialize global parameters if not set
if "pixel_size" not in st.session_state:
    st.session_state.pixel_size = DEFAULT_PIXEL_SIZE
if "frame_interval" not in st.session_state:
    st.session_state.frame_interval = DEFAULT_FRAME_INTERVAL

# Unified Settings Panel (replaces redundant unit controls)
try:
    settings_panel = get_settings_panel()
    settings_panel.show_compact_sidebar()
    
    # Sync settings with session state for backward compatibility
    global_units = get_global_units()
    st.session_state.pixel_size = global_units['pixel_size']
    st.session_state.frame_interval = global_units['frame_interval']
    st.session_state.unit_converter.set_pixel_size(st.session_state.pixel_size)
    st.session_state.unit_converter.set_frame_interval(st.session_state.frame_interval)
except Exception as e:
    # Fallback to manual unit settings if settings panel fails
    st.sidebar.warning(f"Settings panel unavailable: {e}")
    with st.sidebar.expander("Unit Settings"):
        st.session_state.pixel_size = st.number_input(
            "Pixel Size (μm)", 
            min_value=0.001, 
            max_value=10.0, 
            value=st.session_state.pixel_size,
            step=0.01
        )
        st.session_state.frame_interval = st.number_input(
            "Frame Interval (s)", 
            min_value=0.001, 
            max_value=10.0, 
            value=st.session_state.frame_interval,
            step=0.01
        )
        st.session_state.unit_converter.set_pixel_size(st.session_state.pixel_size)
        st.session_state.unit_converter.set_frame_interval(st.session_state.frame_interval)

# Show system limits for clarity across machines
with st.sidebar.expander("System Limits"):
    # Upload limit from Streamlit config
    upload_limit_mb = None
    try:
        from streamlit import config as st_config  # type: ignore
        upload_limit_mb = st_config.get_option("server.maxUploadSize")
    except Exception:
        try:
            import toml  # type: ignore
            cfg_path = os.path.join(".streamlit", "config.toml")
            if os.path.exists(cfg_path):
                cfg = toml.load(cfg_path)
                upload_limit_mb = cfg.get("server", {}).get("maxUploadSize")
        except Exception:
            upload_limit_mb = None

    cfg_mgr = get_config_manager()
    processing_limit_mb = cfg_mgr.get("file_handling", "max_file_size_mb", None)
    chunk_size = cfg_mgr.get("file_handling", "chunk_size", None)

    st.write(f"Upload limit: {upload_limit_mb if upload_limit_mb is not None else 'unknown'} MB")
    if processing_limit_mb is not None:
        st.write(f"Processing limit: {processing_limit_mb} MB")
    if chunk_size is not None:
        st.write(f"Chunk size: {chunk_size} rows")

# Main content area based on active page
if st.session_state.active_page == "MD Integration":
    st.title("Molecular Dynamics Integration")
    
    # Create tabs for different MD functions
    md_tabs = st.tabs(["Load Simulation", "Analyze Simulation", "Compare with SPT"])
    
    # Load Simulation tab
    with md_tabs[0]:
        st.header("Load Molecular Dynamics Simulation")
        
        # File uploader for simulation data
        md_file = st.file_uploader(
            "Upload MD simulation file", 
            type=["gro", "pdb", "xtc", "dcd", "trr", "csv", "xyz"],
            help="Upload molecular dynamics simulation data in various formats"
        )
        
        if md_file is not None:
            try:
                # Load the simulation file
                with st.spinner("Loading simulation data..."):
                    md_sim = load_md_file(md_file)
                    st.session_state.md_simulation = md_sim
                
                # Display simulation information
                st.subheader("Simulation Information")
                
                info = md_sim.simulation_info
                if info:
                    # Display key information
                    cols = st.columns(3)
                    
                    with cols[0]:
                        st.metric("Format", info.get('format', 'Unknown'))
                    
                    with cols[1]:
                        if 'particles' in info:
                            st.metric("Particles", info['particles'])
                    
                    with cols[2]:
                        if 'frames' in info:
                            st.metric("Frames", info['frames'])
                    
                    # Additional information
                    st.subheader("Details")
                    for key, value in info.items():
                        if key not in ['format', 'particles', 'frames', 'loaded']:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Convert to tracks format if trajectory is available
                if md_sim.trajectory is not None:
                    if st.button("Convert to SPT Format"):
                        with st.spinner("Converting simulation to tracks format..."):
                            # Convert the first 10 particles by default, or fewer if there are fewer particles
                            n_particles = min(10, md_sim.trajectory.shape[1])
                            selected_particles = list(range(n_particles))
                            
                            md_tracks = md_sim.convert_to_tracks_format(selected_particles)
                            st.session_state.md_tracks = md_tracks
                            
                            st.success(f"Converted {len(selected_particles)} particles to SPT format")
                            
                            # Preview the tracks
                            st.subheader("Preview")
                            st.dataframe(md_tracks.head())
                
                # Visualize some particles
                if md_sim.trajectory is not None:
                    st.subheader("Visualization")
                    
                    # Visualization options
                    viz_cols = st.columns(3)
                    
                    with viz_cols[0]:
                        n_particles = min(5, md_sim.trajectory.shape[1])
                        num_particles = st.slider("Number of particles", 1, n_particles, 3)
                    
                    with viz_cols[1]:
                        max_frames = md_sim.trajectory.shape[0]
                        num_frames = st.slider("Number of frames", 10, max_frames, min(100, max_frames))
                    
                    with viz_cols[2]:
                        plot_mode = st.selectbox("Plot mode", ["3D", "2D"])
                    
                    # Plot the trajectories
                    if st.button("Plot Trajectories"):
                        with st.spinner("Generating plot..."):
                            # Select random particles
                            selected_particles = np.random.choice(md_sim.trajectory.shape[1], num_particles, replace=False)
                            
                            # Generate the plot
                            fig = md_sim.plot_trajectory(
                                particles=selected_particles,
                                num_frames=num_frames,
                                mode='3d' if plot_mode == '3D' else '2d'
                            )
                            
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing simulation file: {str(e)}")
    
    # Analyze Simulation tab
    with md_tabs[1]:
        st.header("Analyze Molecular Dynamics Simulation")
        
        if st.session_state.md_simulation is None or st.session_state.md_simulation.trajectory is None:
            st.warning("No simulation data loaded. Please load a simulation file first.")
            
            # Button to navigate to the load tab
            if st.button("Go to Load Simulation"):
                md_tabs[0].active = True
        else:
            # Get the loaded simulation
            md_sim = st.session_state.md_simulation
            
            # Analysis options
            st.subheader("Analysis Options")
            
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Mean Squared Displacement", "Diffusion Coefficient", "Trajectory Statistics"]
            )
            
            if analysis_type == "Mean Squared Displacement":
                # MSD analysis
                st.write("Calculate Mean Squared Displacement from the simulation")
                
                # Analysis parameters
                cols = st.columns(2)
                
                with cols[0]:
                    max_frames = md_sim.trajectory.shape[0]
                    max_lag = st.slider("Maximum lag time", 5, max_frames // 2, min(20, max_frames // 2))
                
                with cols[1]:
                    n_particles = md_sim.trajectory.shape[1]
                    particle_sample = st.slider("Number of particles to sample", 1, min(100, n_particles), min(20, n_particles))
                
                # Calculate MSD
                if st.button("Calculate MSD"):
                    with st.spinner("Calculating Mean Squared Displacement..."):
                        # Select random particles
                        selected_particles = np.random.choice(n_particles, particle_sample, replace=False)
                        
                        # Calculate MSD
                        msd_result = md_sim.calculate_msd(selected_particles=selected_particles, max_lag=max_lag)
                        
                        # Plot the MSD curve
                        fig = md_sim.plot_msd(msd_result=msd_result, with_fit=True)
                        
                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display MSD data
                        st.subheader("MSD Data")
                        msd_df = pd.DataFrame({
                            'Lag Time': msd_result['lag_time'],
                            'MSD': msd_result['msd']
                        })
                        st.dataframe(msd_df)
            
            elif analysis_type == "Diffusion Coefficient":
                # Diffusion coefficient analysis
                st.write("Calculate Diffusion Coefficient from the simulation")
                
                # Analysis parameters
                cols = st.columns(2)
                
                with cols[0]:
                    n_particles = md_sim.trajectory.shape[1]
                    particle_sample = st.slider("Number of particles to sample", 1, min(100, n_particles), min(20, n_particles), key="diff_particles")
                
                with cols[1]:
                    fit_points = st.slider("Points for linear fit", 3, 20, 5)
                
                # Calculate diffusion coefficient
                if st.button("Calculate Diffusion Coefficient"):
                    with st.spinner("Calculating Diffusion Coefficient..."):
                        # Select random particles
                        selected_particles = np.random.choice(n_particles, particle_sample, replace=False)
                        
                        # Calculate MSD
                        msd_result = md_sim.calculate_msd(selected_particles=selected_particles)
                        
                        # Calculate diffusion coefficient
                        D = md_sim.calculate_diffusion_coefficient(msd_result=msd_result, fit_points=fit_points)
                        
                        # Display the result
                        st.subheader("Results")
                        st.metric("Diffusion Coefficient (μm²/s)", f"{D:.6f}")
                        
                        # Plot the MSD curve with fit
                        fig = md_sim.plot_msd(msd_result=msd_result, with_fit=True, fit_points=fit_points)
                        st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Trajectory Statistics":
                # Trajectory statistics
                st.write("Calculate statistical properties of trajectories")
                
                # Convert to tracks format if not already done
                if st.session_state.md_tracks is None:
                    if st.button("Generate Tracks"):
                        with st.spinner("Converting simulation to tracks format..."):
                            # Convert the first 10 particles by default, or fewer if there are fewer particles
                            n_particles = min(10, md_sim.trajectory.shape[1])
                            selected_particles = list(range(n_particles))
                            
                            md_tracks = md_sim.convert_to_tracks_format(selected_particles)
                            st.session_state.md_tracks = md_tracks
                            
                            st.success(f"Converted {len(selected_particles)} particles to SPT format")
                
                if st.session_state.md_tracks is not None:
                    # Calculate track statistics
                    with st.spinner("Calculating track statistics..."):
                        track_stats = calculate_track_statistics(st.session_state.md_tracks)
                        
                        # Display the statistics
                        st.subheader("Track Statistics")
                        st.dataframe(track_stats)
                        
                        # Generate plots
                        st.subheader("Statistics Plots")
                        plots = plot_track_statistics(track_stats)
                        
                        for name, fig in plots.items():
                            st.subheader(name.replace("_", " ").title())
                            st.plotly_chart(fig, use_container_width=True)
    
    # Compare with SPT tab
    with md_tabs[2]:
        st.header("Compare MD Simulation with SPT Data")
        
        # Check if both MD and SPT data are available
        md_available = st.session_state.md_simulation is not None and st.session_state.md_simulation.trajectory is not None
        spt_available = st.session_state.tracks_data is not None
        
        if not md_available:
            st.warning("No MD simulation data loaded. Please load a simulation file first.")
            
            # Button to navigate to the load tab
            if st.button("Go to Load Simulation", key="goto_load_sim"):
                md_tabs[0].active = True
        
        elif not spt_available:
            st.warning("No SPT data loaded. Please load SPT data first.")
            
            # Button to navigate to the data loading page
            if st.button("Go to Data Loading", key="goto_data_loading"):
                st.session_state.active_page = "Data Loading"
                st.rerun()
        
        else:
            # Both data types are available
            st.subheader("Comparison Options")
            
            comparison_type = st.selectbox(
                "Comparison Type",
                ["Diffusion Coefficient", "MSD Curves", "Trajectories"]
            )
            
            md_sim = st.session_state.md_simulation
            
            if comparison_type == "Diffusion Coefficient":
                # Compare diffusion coefficients
                st.write("Compare diffusion coefficients between MD simulation and SPT data")
                
                # Check if diffusion analysis has been performed on SPT data
                if "diffusion" not in st.session_state.analysis_results:
                    st.warning("No diffusion analysis results available for SPT data. Please run diffusion analysis first.")
                    
                    # Button to navigate to the analysis page
                    if st.button("Go to Diffusion Analysis"):
                        st.session_state.active_page = "Analysis"
                        st.rerun()
                else:
                    # Get SPT diffusion results
                    spt_results = st.session_state.analysis_results["diffusion"]
                    
                    # Calculate MD diffusion coefficient
                    if st.button("Compare Diffusion Coefficients"):
                        with st.spinner("Comparing diffusion coefficients..."):
                            try:
                                # Calculate diffusion coefficient from MD
                                md_diffusion = md_sim.calculate_diffusion_coefficient()
                                
                                # Get diffusion coefficient from SPT data
                                spt_diffusion = spt_results["ensemble_results"]["mean_diffusion_coefficient"]
                                
                                # Calculate ratio and difference
                                ratio = md_diffusion / spt_diffusion if spt_diffusion != 0 else float('inf')
                                difference = md_diffusion - spt_diffusion
                                
                                # Display the results
                                st.subheader("Comparison Results")
                                
                                cols = st.columns(2)
                                
                                with cols[0]:
                                    st.metric("MD Diffusion Coefficient (μm²/s)", f"{md_diffusion:.6f}")
                                    st.metric("SPT Diffusion Coefficient (μm²/s)", f"{spt_diffusion:.6f}")
                                
                                with cols[1]:
                                    st.metric("Ratio (MD/SPT)", f"{ratio:.2f}")
                                    st.metric("Difference (MD-SPT)", f"{difference:.6f}")
                                
                                # Plot a comparison bar chart
                                data = pd.DataFrame({
                                    'Source': ['MD Simulation', 'SPT Data'],
                                    'Diffusion Coefficient': [md_diffusion, spt_diffusion]
                                })
                                
                                fig = px.bar(
                                    data,
                                    x='Source',
                                    y='Diffusion Coefficient',
                                    title="Diffusion Coefficient Comparison",
                                    labels={'Diffusion Coefficient': 'Diffusion Coefficient (μm²/s)'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error comparing diffusion coefficients: {str(e)}")
            
            elif comparison_type == "MSD Curves":
                # Compare MSD curves
                st.write("Compare Mean Squared Displacement curves between MD simulation and SPT data")
                
                # Check if MSD data is available for SPT data
                if "diffusion" not in st.session_state.analysis_results or "msd_data" not in st.session_state.analysis_results["diffusion"]:
                    st.warning("No MSD data available for SPT data. Please run diffusion analysis first.")
                    
                    # Button to navigate to the analysis page
                    if st.button("Go to Diffusion Analysis", key="goto_diff_analysis"):
                        st.session_state.active_page = "Analysis"
                        st.rerun()
                else:
                    # Get SPT MSD data
                    spt_msd_data = st.session_state.analysis_results["diffusion"]["msd_data"]
                    
                    # Calculate MD MSD data
                    if st.button("Compare MSD Curves"):
                        with st.spinner("Comparing MSD curves..."):
                            try:
                                # Calculate MSD from MD
                                md_msd = md_sim.calculate_msd()
                                
                                # Create a plot comparing the MSD curves
                                fig = go.Figure()
                                
                                # Add MD MSD curve
                                fig.add_trace(go.Scatter(
                                    x=md_msd['lag_time'],
                                    y=md_msd['msd'],
                                    mode='lines+markers',
                                    name='MD Simulation'
                                ))
                                
                                # Add SPT MSD curve (average over all tracks)
                                avg_spt_msd = spt_msd_data.groupby('lag_time')['msd'].mean().reset_index()
                                
                                fig.add_trace(go.Scatter(
                                    x=avg_spt_msd['lag_time'],
                                    y=avg_spt_msd['msd'],
                                    mode='lines+markers',
                                    name='SPT Data'
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title="MSD Curve Comparison",
                                    xaxis_title="Lag Time (s)",
                                    yaxis_title="MSD (μm²)",
                                    legend=dict(
                                        x=0,
                                        y=1,
                                        bgcolor='rgba(255, 255, 255, 0.5)'
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error comparing MSD curves: {str(e)}")
            
            elif comparison_type == "Trajectories":
                # Compare trajectories
                st.write("Compare trajectory patterns between MD simulation and SPT data")
                
                # Convert MD to tracks format if not already done
                if st.session_state.md_tracks is None:
                    if st.button("Generate MD Tracks"):
                        with st.spinner("Converting simulation to tracks format..."):
                            # Convert the first 10 particles by default, or fewer if there are fewer particles
                            n_particles = min(10, md_sim.trajectory.shape[1])
                            selected_particles = list(range(n_particles))
                            
                            md_tracks = md_sim.convert_to_tracks_format(selected_particles)
                            st.session_state.md_tracks = md_tracks
                            
                            st.success(f"Converted {len(selected_particles)} particles to SPT format")
                
                if st.session_state.md_tracks is not None:
                    # Visualization options
                    st.subheader("Visualization Options")
                    
                    viz_type = st.radio("Display Type", ["Side by Side", "Overlay"])
                    
                    # Number of tracks to show
                    max_md_tracks = min(10, st.session_state.md_tracks['track_id'].nunique())
                    max_spt_tracks = min(10, st.session_state.tracks_data['track_id'].nunique())
                    
                    cols = st.columns(2)
                    
                    with cols[0]:
                        md_tracks_to_show = st.slider("MD Tracks to Show", 1, max_md_tracks, 3)
                    
                    with cols[1]:
                        spt_tracks_to_show = st.slider("SPT Tracks to Show", 1, max_spt_tracks, 3)
                    
                    # Create visualizations
                    if st.button("Compare Trajectories"):
                        with st.spinner("Generating trajectory comparison..."):
                            try:
                                if viz_type == "Side by Side":
                                    # Create side-by-side plots
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("MD Simulation Trajectories")
                                        
                                        # Select random MD tracks
                                        md_track_ids = np.random.choice(
                                            st.session_state.md_tracks['track_id'].unique(),
                                            md_tracks_to_show,
                                            replace=False
                                        )
                                        
                                        md_subset = st.session_state.md_tracks[st.session_state.md_tracks['track_id'].isin(md_track_ids)]
                                        
                                        # Plot MD tracks
                                        fig_md = plot_tracks(md_subset, color_by='track_id')
                                        st.plotly_chart(fig_md, use_container_width=True)
                                    
                                    with col2:
                                        st.subheader("SPT Data Trajectories")
                                        
                                        # Select random SPT tracks
                                        spt_track_ids = np.random.choice(
                                            st.session_state.tracks_data['track_id'].unique(),
                                            spt_tracks_to_show,
                                            replace=False
                                        )
                                        
                                        spt_subset = st.session_state.tracks_data[st.session_state.tracks_data['track_id'].isin(spt_track_ids)]
                                        
                                        # Plot SPT tracks
                                        fig_spt = plot_tracks(spt_subset, color_by='track_id')
                                        st.plotly_chart(fig_spt, use_container_width=True)
                                
                                else:  # Overlay
                                    st.subheader("Overlaid Trajectories")
                                    
                                    # Add a distinguishing column to each dataset
                                    md_subset = st.session_state.md_tracks.copy()
                                    md_subset['source'] = 'MD'
                                    
                                    spt_subset = st.session_state.tracks_data.copy()
                                    spt_subset['source'] = 'SPT'
                                    
                                    # Select random tracks
                                    md_track_ids = np.random.choice(
                                        md_subset['track_id'].unique(),
                                        md_tracks_to_show,
                                        replace=False
                                    )
                                    
                                    spt_track_ids = np.random.choice(
                                        spt_subset['track_id'].unique(),
                                        spt_tracks_to_show,
                                        replace=False
                                    )
                                    
                                    # Filter to selected tracks
                                    md_filtered = md_subset[md_subset['track_id'].isin(md_track_ids)]
                                    spt_filtered = spt_subset[spt_subset['track_id'].isin(spt_track_ids)]
                                    
                                    # Combine the data
                                    combined = pd.concat([md_filtered, spt_filtered])
                                    
                                    # Create a custom combined plot
                                    fig = px.scatter(
                                        combined,
                                        x='x',
                                        y='y',
                                        color='source',
                                        title="MD and SPT Trajectory Comparison",
                                        labels={'x': 'X Position', 'y': 'Y Position'},
                                        color_discrete_map={'MD': 'blue', 'SPT': 'red'}
                                    )
                                    
                                    # Add lines for tracks grouped by source and track_id
                                    for source, source_group in combined.groupby('source'):
                                        for track_id, track_df in source_group.groupby('track_id'):
                                            # Sort by frame
                                            sorted_track = track_df.sort_values('frame')
                                            
                                            # Add trace for this track
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=sorted_track['x'],
                                                    y=sorted_track['y'],
                                                    mode='lines',
                                                    showlegend=False,
                                                    line=dict(
                                                        color='blue' if source == 'MD' else 'red',
                                                        width=1
                                                    )
                                                )
                                            )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error comparing trajectories: {str(e)}")

# Simulation page
elif st.session_state.active_page == "Simulation":
    try:
        show_simulation_page()
    except Exception as e:
        st.error(f"Simulation page error: {e}")

# Home page                
elif st.session_state.active_page == "Home":
    # Application title
    st.title("SPT Analysis")
    
    # Main page content
    st.markdown("""
    ## Welcome to SPT Analysis

    This application provides comprehensive tools for analyzing single particle tracking data.
    Use the navigation menu on the left to access different tools:

    - **Data Loading**: Load your tracking data or microscopy images
    - **Tracking**: Detect and track particles in microscopy images
    - **Analysis**: Analyze tracks with various methods
    - **Visualization**: Visualize tracks and analysis results
    - **Advanced Analysis**: Access specialized analysis modules
    - **Comparative Analysis**: Compare results across different datasets
    """)
    
    # Quick access section
    st.header("Quick Start")
    
    # Two columns for quick access
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Load Data")
        
        # File uploader for quick loading
        uploaded_file = st.file_uploader(
            "Upload image or track data", 
            type=["tif", "tiff", "png", "jpg", "jpeg", "csv", "xlsx", "h5", "json", "ims", "uic", "xml", "mvd2", "aisf", "aiix"]
        )
        
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Process depending on file type
            if file_extension in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                # Image file handling using centralized state
                try:
                    image_data = load_image_file(uploaded_file)
                    st.session_state.app_state.load_image_data(image_data, {'filename': uploaded_file.name})
                    st.success(f"Image loaded successfully: {uploaded_file.name}")
                    st.session_state.active_page = "Tracking"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    
            elif file_extension in ['.csv', '.xlsx', '.h5', '.json']:
                # Track data handling using centralized state
                try:
                    tracks_data = load_tracks_file(uploaded_file)
                    st.session_state.app_state.load_tracks(tracks_data, source=uploaded_file.name)
                    # Automatically calculate track statistics using analysis manager
                    st.session_state.analysis_manager.calculate_track_statistics()
                    st.success(f"Track data loaded successfully: {uploaded_file.name}")
                    st.session_state.active_page = "Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading track data: {str(e)}")
                    
            elif file_extension in ['.ims', '.xml']:
                # Imaris file handling
                try:
                    st.session_state.tracks_data = load_imaris_file(uploaded_file)
                    st.success(f"Imaris file loaded successfully: {uploaded_file.name}")
                    # Calculate track statistics
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.session_state.active_page = "Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading Imaris file: {str(e)}")
                    
            elif file_extension in ['.uic', '.aisf', '.aiix']:
                # Volocity file handling
                try:
                    from volocity_handler import load_volocity_file
                    st.session_state.tracks_data = load_volocity_file(uploaded_file)
                    st.success(f"Volocity file loaded successfully: {uploaded_file.name}")
                    # Calculate track statistics
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.session_state.active_page = "Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading Volocity file: {str(e)}")
                    
            elif file_extension in ['.mvd2']:
                # MVD2 file handling
                try:
                    from mvd2_handler import load_mvd2_file
                    st.session_state.tracks_data = load_mvd2_file(uploaded_file)
                    st.success(f"MVD2 file loaded successfully: {uploaded_file.name}")
                    # Calculate track statistics
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.session_state.active_page = "Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading MVD2 file: {str(e)}")
                    
            # Display a preview if data is loaded
            if st.session_state.image_data is not None:
                # Normalize image for display
                display_image = st.session_state.image_data[0].copy()
                if display_image.dtype != np.uint8:
                    if display_image.max() <= 1.0:
                        display_image = (display_image * 255).astype(np.uint8)
                    else:
                        display_image = ((display_image - display_image.min()) / 
                                       (display_image.max() - display_image.min()) * 255).astype(np.uint8)
                st.image(display_image, caption="Image preview", use_container_width=True)
                
            if st.session_state.tracks_data is not None:
                st.dataframe(st.session_state.tracks_data.head())
    
    with col2:
        st.subheader("Recent Analysis")
        
        # Check if there are any recent analyses
        if st.session_state.recent_analyses:
            for analysis in st.session_state.recent_analyses[-3:]:  # Show last 3
                with st.expander(f"{analysis['name']} - {analysis['date']}"):
                    st.write(f"Type: {analysis['type']}")
                    st.write(f"Parameters: {analysis['parameters']}")
                    
                    # Button to open the analysis
                    if st.button("Open", key=f"open_{analysis['id']}"):
                        st.session_state.selected_analysis = analysis['id']
                        st.session_state.active_page = "Analysis"
                        st.rerun()
        else:
            st.info("No recent analyses available.")
            
        # Quick links to common tasks
        st.subheader("Quick Links")
        quick_links = st.selectbox(
            "Select a task",
            ["Select a task...", "New tracking", "New analysis", "Load sample data", "Comparative analysis"]
        )
        
        if quick_links == "New tracking":
            st.session_state.active_page = "Tracking"
            st.rerun()
        elif quick_links == "New analysis":
            st.session_state.active_page = "Analysis"
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
                    st.session_state.active_page = "Analysis"
                    st.rerun()
                else:
                    st.warning("Sample data file not found. Check 'sample data' folder.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        elif quick_links == "Comparative analysis":
            st.session_state.active_page = "Comparative Analysis"
            st.rerun()
    
    # Experimental Planning Section
    st.markdown("---")
    with st.expander("🔬 Experimental Planning - Acquisition Parameter Optimizer", expanded=False):
        st.markdown("""
        ### Design Optimal Imaging Experiments
        
        Calculate recommended **frame interval** and **exposure time** based on your expected diffusion 
        coefficient and localization precision. This optimizer balances competing factors:
        - **Short exposure** → less motion blur → reduced bias
        - **Long exposure** → more photons → better localization
        - **Short intervals** → better temporal resolution
        - **Long intervals** → larger displacements → better SNR
        """)
        
        try:
            from acquisition_advisor import AcquisitionOptimizer
            
            optimizer = AcquisitionOptimizer()
            
            # Create input columns
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                expected_D = st.number_input(
                    "Expected Diffusion Coefficient (µm²/s)",
                    min_value=0.001,
                    max_value=100.0,
                    value=0.5,
                    step=0.1,
                    format="%.3f",
                    help="Approximate D you expect for your particle/molecule"
                )
                
                localization_precision = st.number_input(
                    "Localization Precision (nm)",
                    min_value=5.0,
                    max_value=200.0,
                    value=30.0,
                    step=5.0,
                    help="Expected 1σ localization precision (typically 20-50 nm)"
                )
                
                pixel_size_nm = st.number_input(
                    "Pixel Size (nm)",
                    min_value=10.0,
                    max_value=500.0,
                    value=100.0,
                    step=10.0,
                    help="Camera pixel size in object space"
                )
            
            with input_col2:
                target_error = st.slider(
                    "Target Measurement Error (%)",
                    min_value=5.0,
                    max_value=30.0,
                    value=10.0,
                    step=1.0,
                    help="Acceptable relative error in D estimation"
                )
                
                track_length = st.slider(
                    "Typical Track Length (frames)",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Expected number of frames per track"
                )
                
                # Optional camera constraints
                st.markdown("**Camera Limits (optional)**")
                use_camera_limits = st.checkbox("Specify camera constraints", value=False)
                
                camera_min_interval = None
                camera_max_interval = None
                
                if use_camera_limits:
                    camera_max_fps = st.number_input(
                        "Max Frame Rate (fps)",
                        min_value=1.0,
                        max_value=10000.0,
                        value=100.0,
                        help="Maximum frames per second your camera can achieve"
                    )
                    camera_min_interval = 1.0 / camera_max_fps
                    
                    camera_max_interval = st.number_input(
                        "Max Frame Interval (s)",
                        min_value=0.001,
                        max_value=10.0,
                        value=1.0,
                        help="Longest frame interval you want to consider"
                    )
            
            # Calculate button
            if st.button("🎯 Calculate Optimal Parameters", type="primary"):
                with st.spinner("Optimizing acquisition parameters..."):
                    result = optimizer.calculate_optimal_parameters(
                        expected_D=expected_D,
                        localization_precision=localization_precision,
                        target_error_pct=target_error,
                        pixel_size=pixel_size_nm / 1000.0,  # Convert to µm
                        typical_track_length_frames=track_length,
                        camera_min_interval=camera_min_interval,
                        camera_max_interval=camera_max_interval
                    )
                    
                    # Store results in session state
                    st.session_state.acquisition_optimizer_result = result
            
            # Display results if available
            if 'acquisition_optimizer_result' in st.session_state:
                result = st.session_state.acquisition_optimizer_result
                
                st.markdown("---")
                st.markdown("### 📊 Recommended Parameters")
                
                # Main results in columns
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    st.metric(
                        "Frame Interval",
                        f"{result['frame_interval']*1000:.2f} ms",
                        help="Recommended time between frames"
                    )
                    st.caption(f"= {1/result['frame_interval']:.1f} fps")
                
                with rec_col2:
                    st.metric(
                        "Exposure Time",
                        f"{result['exposure_time']*1000:.2f} ms",
                        help="Recommended camera exposure duration"
                    )
                    st.caption(f"{result['blur_fraction']*100:.0f}% of frame interval")
                
                with rec_col3:
                    st.metric(
                        "Expected Error",
                        f"{result['expected_error_pct']:.1f}%",
                        delta=f"{result['expected_error_pct']-target_error:+.1f}%",
                        delta_color="inverse",
                        help="Predicted measurement uncertainty"
                    )
                
                # Performance metrics
                st.markdown("### 📈 Expected Performance")
                
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    st.metric(
                        "Displacement per Frame",
                        f"{result['displacement_per_frame']*1000:.1f} nm",
                        help="Expected RMS displacement between frames"
                    )
                    
                    st.metric(
                        "Spatial SNR",
                        f"{result['displacement_to_noise_ratio']:.2f}",
                        help="Ratio of displacement to localization noise"
                    )
                    
                    if result['displacement_to_noise_ratio'] < 1.5:
                        st.warning("⚠️ Low SNR - motion barely exceeds noise!")
                    elif result['displacement_to_noise_ratio'] > 3.0:
                        st.success("✅ Excellent SNR for reliable tracking")
                
                with perf_col2:
                    st.metric(
                        "Frames Needed",
                        f"{result['steps_needed']} frames",
                        help="Minimum track length for target error"
                    )
                    
                    st.metric(
                        "Acquisition Time",
                        f"{result['acquisition_time']:.2f} s",
                        help="Total time for typical track"
                    )
                    
                    st.metric(
                        "Feasibility",
                        result['feasibility'].upper(),
                        help="Overall experiment feasibility assessment"
                    )
                
                # Bias analysis
                with st.expander("🔍 Bias Analysis", expanded=False):
                    bias_col1, bias_col2 = st.columns(2)
                    
                    with bias_col1:
                        st.metric(
                            "Localization Noise Bias",
                            f"+{result['noise_bias_pct']:.1f}%",
                            help="Overestimation due to localization noise"
                        )
                    
                    with bias_col2:
                        st.metric(
                            "Motion Blur Bias",
                            f"{result['blur_bias_pct']:.1f}%",
                            help="Underestimation due to motion blur"
                        )
                    
                    st.info(
                        f"**Net bias**: ~{abs(result['noise_bias_pct'] + result['blur_bias_pct']):.1f}% "
                        f"(noise {'dominates' if abs(result['noise_bias_pct']) > abs(result['blur_bias_pct']) else 'vs blur'}). "
                        "Use CVE or MLE estimators for bias correction."
                    )
                
                # Warnings
                if result['warnings']:
                    st.markdown("### ⚠️ Warnings")
                    for warning in result['warnings']:
                        st.warning(warning)
                
                # Recommendations
                if result['recommendations']:
                    st.markdown("### 💡 Recommendations")
                    for rec in result['recommendations']:
                        st.info(rec)
                
                # Quick comparison with optimal
                with st.expander("📐 Comparison with Theoretical Optimal", expanded=False):
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.write("**Frame Interval:**")
                        st.write(f"Recommended: {result['frame_interval']*1000:.2f} ms")
                        st.write(f"Theoretical Optimal: {result['optimal_interval']*1000:.2f} ms")
                        deviation_interval = abs(result['frame_interval'] - result['optimal_interval']) / result['optimal_interval'] * 100
                        st.write(f"Deviation: {deviation_interval:.1f}%")
                    
                    with comp_col2:
                        st.write("**Exposure Time:**")
                        st.write(f"Recommended: {result['exposure_time']*1000:.2f} ms")
                        st.write(f"Theoretical Optimal: {result['optimal_exposure']*1000:.2f} ms")
                        deviation_exposure = abs(result['exposure_time'] - result['optimal_exposure']) / result['optimal_exposure'] * 100
                        st.write(f"Deviation: {deviation_exposure:.1f}%")
                    
                    if deviation_interval < 10 and deviation_exposure < 10:
                        st.success("✅ Recommended parameters are very close to theoretical optimum!")
                    elif deviation_interval < 50 and deviation_exposure < 50:
                        st.info("Parameters deviate from optimum due to camera constraints but are acceptable.")
                    else:
                        st.warning("Significant deviation from optimal - consider upgrading camera or adjusting expectations.")
        
        except ImportError:
            st.error("Acquisition optimizer module not available. Please check installation.")
        except Exception as e:
            st.error(f"Error in acquisition optimizer: {str(e)}")

# Project Management Page
elif st.session_state.active_page == "Project Management":
    st.title("Project Management: Group cells into experimental conditions")
    pmgr = pm.ProjectManager()
    if "pm_current" not in st.session_state:
        st.session_state.pm_current = None  # holds pm.Project
    # Project selector/creator
    with st.expander("Project Selection", expanded=True):
        existing = pmgr.list_projects()
        options = [f"{p['name']} ({p['id'][:8]})" for p in existing]
        sel = st.selectbox("Select a project", options + ["<New Project>"])
        if sel == "<New Project>":
            c1, c2 = st.columns([2,1])
            with c1:
                new_name = st.text_input("Project name", value="My Experiment")
            with c2:
                if st.button("Create Project", use_container_width=True):
                    proj = pmgr.create_project(new_name)
                    st.session_state.pm_current = proj
                    st.success("Project created.")
                    st.rerun()
        else:
            idx = options.index(sel)
            meta = existing[idx]
            st.session_state.pm_current = pmgr.get_project(meta['id'])

    proj = st.session_state.pm_current
    if proj is None:
        st.info("Create or select a project to manage conditions and files.")
    else:
        st.subheader(f"Project: {proj.name}")

        if st.button("Delete Project", key="pm_delete_project"):
            st.session_state.confirm_delete = True

        if st.session_state.get("confirm_delete"):
            st.warning(f"Are you sure you want to delete project '{proj.name}'? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete it", type="primary"):
                    pmgr.delete_project(proj.id)
                    st.session_state.pm_current = None
                    st.session_state.confirm_delete = False
                    st.success("Project deleted.")
                    st.rerun()
            with c2:
                if st.button("Cancel"):
                    st.session_state.confirm_delete = False
                    st.rerun()

        # Add condition
        with st.expander("Add Condition", expanded=True):
            cname = st.text_input("Condition name", key="pm_new_cond_name")
            cdesc = st.text_input("Description", key="pm_new_cond_desc")
            if st.button("Add Condition", key="pm_add_condition_btn"):
                if cname and cname.strip():
                    pmgr.add_condition(proj, cname.strip(), cdesc.strip())
                    pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                    st.success("Condition added.")
                    st.rerun()
                else:
                    st.error("Please enter a condition name.")

        # List conditions with file upload per condition
        for cond in list(proj.conditions):
            with st.expander(f"Condition: {cond.name} ({len(cond.files)} files)", expanded=True):

                # Delete condition button
                if st.button("Delete Condition", key=f"delete_cond_{cond.id}"):
                    if "confirm_delete_condition" not in st.session_state:
                        st.session_state.confirm_delete_condition = {}
                    st.session_state.confirm_delete_condition[cond.id] = True

                if st.session_state.get("confirm_delete_condition", {}).get(cond.id):
                    st.warning(f"Are you sure you want to delete condition '{cond.name}'?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes, delete", type="primary", key=f"confirm_delete_cond_{cond.id}"):
                            pmgr.remove_condition(proj, cond.id)
                            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                            st.session_state.confirm_delete_condition[cond.id] = False
                            st.success("Condition deleted.")
                            st.rerun()
                    with c2:
                        if st.button("Cancel", key=f"cancel_delete_cond_{cond.id}"):
                            st.session_state.confirm_delete_condition[cond.id] = False
                            st.rerun()

                uploaded = st.file_uploader(
                    "Add cell files (CSV, Excel, XML)", 
                    type=["csv", "xlsx", "xls", "xml"], 
                    accept_multiple_files=True, 
                    key=f"pm_up_{cond.id}",
                    help="Upload track data in CSV, Excel, or XML (TrackMate) format"
                )
                
                # Track which files have been processed to avoid duplicates
                upload_key = f"pm_upload_processed_{cond.id}"
                if upload_key not in st.session_state:
                    st.session_state[upload_key] = set()
                
                if uploaded:
                    # Check if these are new files (not already processed)
                    new_files = []
                    for uf in uploaded:
                        file_id = f"{uf.name}_{uf.size}"
                        if file_id not in st.session_state[upload_key]:
                            new_files.append((uf, file_id))
                    
                    # Only process new files
                    if new_files:
                        for uf, file_id in new_files:
                            try:
                                file_extension = os.path.splitext(uf.name)[1].lower()
                                
                                # Handle different file types
                                if file_extension == '.csv':
                                    import pandas as _pd
                                    df = _pd.read_csv(uf)
                                elif file_extension in ['.xlsx', '.xls']:
                                    # Use the existing load_tracks_file function for Excel
                                    from data_loader import load_tracks_file
                                    df = load_tracks_file(uf)
                                    if df is None or df.empty:
                                        st.warning(f"No track data found in {uf.name}")
                                        continue
                                elif file_extension == '.xml':
                                    # Use the existing load_tracks_file function for XML
                                    from data_loader import load_tracks_file
                                    df = load_tracks_file(uf)
                                    if df is None or df.empty:
                                        st.warning(f"No track data found in {uf.name}")
                                        continue
                                else:
                                    st.warning(f"Unsupported file type: {file_extension}")
                                    continue
                                
                                pmgr.add_file_to_condition(proj, cond.id, uf.name, df)
                                st.session_state[upload_key].add(file_id)
                            except Exception as e:
                                st.warning(f"Failed to add {uf.name}: {e}")
                        pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                        st.success(f"{len(new_files)} file(s) added.")
                        st.rerun()

                # Show files and remove option
                if cond.files:
                    for f in list(cond.files):
                        fname = f.get('name') or f.get('file_name') or f.get('id')
                        cols = st.columns([6,2,2])
                        cols[0].write(fname)
                        if cols[1].button("Preview", key=f"pv_{cond.id}_{f.get('id')}"):
                            try:
                                import pandas as _pd, io as _io, os as _os
                                if f.get('data'):
                                    df = _pd.read_csv(_io.BytesIO(f['data']))
                                elif f.get('data_path') and _os.path.exists(f['data_path']):
                                    df = _pd.read_csv(f['data_path'])
                                else:
                                    df = None
                                if df is not None:
                                    st.dataframe(df.head())
                                else:
                                    st.info("No data available for preview.")
                            except Exception as e:
                                st.warning(f"Preview failed: {e}")
                        if cols[2].button("Remove", key=f"rm_{cond.id}_{f.get('id')}"):
                            pmgr.remove_file_from_project(proj, cond.id, f.get('id'))
                            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                            st.rerun()
                    
                    # Clear All Files button
                    st.write("")  # Spacing
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("🗑️ Clear All Files", key=f"clear_all_{cond.id}", type="secondary"):
                            # Remove all files from this condition
                            for f in list(cond.files):
                                pmgr.remove_file_from_project(proj, cond.id, f.get('id'))
                            # Clear the processed tracking for this condition
                            upload_key = f"pm_upload_processed_{cond.id}"
                            if upload_key in st.session_state:
                                del st.session_state[upload_key]
                            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                            st.success(f"All files removed from '{cond.name}'")
                            st.rerun()

                # Subpopulation detection and pooling workflow
                st.write("---")
                st.write("**Analysis Workflow:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Step 1: Detect Subpopulations**")
                    if st.button("🔬 Detect Subpopulations", key=f"subpop_{cond.id}", type="primary"):
                        with st.spinner("Analyzing single-cell heterogeneity..."):
                            try:
                                # Strategy: Each file represents one cell/nucleus
                                # Analyze track-level variation within each file to characterize cells
                                
                                st.info(f"Analyzing {len(cond.files)} cells (files) in condition '{cond.name}'")
                                
                                # Collect per-cell (per-file) features
                                cell_features_list = []
                                all_tracks_with_cell_id = []
                                
                                for file_idx, file_info in enumerate(cond.files):
                                    try:
                                        # Load this cell's data
                                        import pandas as pd_local, io as io_local
                                        
                                        # Try multiple ways to get the data
                                        df = None
                                        if file_info.get('data'):
                                            # Data stored as bytes
                                            df = pd_local.read_csv(io_local.BytesIO(file_info['data']))
                                        elif file_info.get('data_path') and os.path.exists(file_info['data_path']):
                                            # Data stored in file path
                                            df = pd_local.read_csv(file_info['data_path'])
                                        elif file_info.get('path') and os.path.exists(file_info['path']):
                                            # Legacy path field
                                            from data_loader import load_tracks_file
                                            df = load_tracks_file(file_info['path'])
                                        
                                        if df is None or df.empty:
                                            st.warning(f"Could not load data from {file_info.get('name', 'unknown file')}")
                                            continue
                                        
                                        # Assign cell_id based on file
                                        cell_id = f"cell_{file_idx}"
                                        df['cell_id'] = cell_id
                                        df['source_file'] = file_info['name']
                                        all_tracks_with_cell_id.append(df)
                                        
                                        # Calculate per-cell features from track variation
                                        n_tracks = df['track_id'].nunique()
                                        
                                        if n_tracks < 5:  # Need minimum tracks for meaningful statistics
                                            continue
                                        
                                        # Get track-level statistics
                                        track_stats = df.groupby('track_id').agg({
                                            'x': ['mean', 'std'],
                                            'y': ['mean', 'std'],
                                            'frame': ['min', 'max', 'count']
                                        }).reset_index()
                                        
                                        track_stats.columns = ['track_id', 'x_mean', 'x_std', 'y_mean', 'y_std', 
                                                              'frame_min', 'frame_max', 'track_length']
                                        
                                        # Calculate displacements per track
                                        track_displacements = []
                                        for tid, track_df in df.groupby('track_id'):
                                            if len(track_df) < 2:
                                                continue
                                            dx = np.diff(track_df['x'].values)
                                            dy = np.diff(track_df['y'].values)
                                            disp = np.sqrt(dx**2 + dy**2)
                                            track_displacements.append({
                                                'track_id': tid,
                                                'mean_displacement': np.mean(disp),
                                                'total_displacement': np.sum(disp)
                                            })
                                        
                                        disp_df = pd.DataFrame(track_displacements)
                                        
                                        # Aggregate to cell-level features
                                        cell_features = {
                                            'cell_id': cell_id,
                                            'source_file': file_info['name'],
                                            'n_tracks': n_tracks,
                                            
                                            # Track length statistics (measures track stability/lifetime)
                                            'mean_track_length': track_stats['track_length'].mean(),
                                            'std_track_length': track_stats['track_length'].std(),
                                            'cv_track_length': track_stats['track_length'].std() / track_stats['track_length'].mean() if track_stats['track_length'].mean() > 0 else 0,
                                            
                                            # Displacement statistics (measures mobility)
                                            'mean_displacement_per_track': disp_df['mean_displacement'].mean() if not disp_df.empty else 0,
                                            'std_displacement_per_track': disp_df['mean_displacement'].std() if not disp_df.empty else 0,
                                            'cv_displacement': (disp_df['mean_displacement'].std() / disp_df['mean_displacement'].mean()) if not disp_df.empty and disp_df['mean_displacement'].mean() > 0 else 0,
                                            
                                            # Spatial characteristics
                                            'spatial_extent_x': df['x'].max() - df['x'].min(),
                                            'spatial_extent_y': df['y'].max() - df['y'].min(),
                                            
                                            # Temporal characteristics
                                            'total_frames': df['frame'].max() - df['frame'].min() + 1,
                                            
                                            # Heterogeneity measures (variation within cell)
                                            'track_length_heterogeneity': track_stats['track_length'].std() / track_stats['track_length'].mean() if track_stats['track_length'].mean() > 0 else 0,
                                            'displacement_heterogeneity': disp_df['mean_displacement'].std() / disp_df['mean_displacement'].mean() if not disp_df.empty and disp_df['mean_displacement'].mean() > 0 else 0
                                        }
                                        
                                        cell_features_list.append(cell_features)
                                        
                                    except Exception as e:
                                        st.warning(f"Could not analyze file {file_info['name']}: {str(e)}")
                                        continue
                                
                                if len(cell_features_list) < 5:
                                    st.error(f"Insufficient cells for analysis. Found {len(cell_features_list)} cells, need at least 5.")
                                    st.info("Each file should represent one cell with multiple tracks.")
                                    continue
                                
                                # Create cell-level dataframe
                                cell_df = pd.DataFrame(cell_features_list)
                                
                                # Prepare features for clustering
                                feature_cols = [
                                    'mean_track_length', 'cv_track_length',
                                    'mean_displacement_per_track', 'cv_displacement',
                                    'track_length_heterogeneity', 'displacement_heterogeneity'
                                ]
                                
                                available_features = [f for f in feature_cols if f in cell_df.columns]
                                X = cell_df[available_features].fillna(0)
                                
                                # Standardize features
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Try different numbers of clusters
                                from sklearn.cluster import KMeans
                                from sklearn.metrics import silhouette_score
                                
                                best_k = 2
                                best_score = -1
                                
                                for k in range(2, min(5, len(cell_df))):
                                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                    labels = kmeans.fit_predict(X_scaled)
                                    
                                    if len(np.unique(labels)) > 1:
                                        score = silhouette_score(X_scaled, labels)
                                        if score > best_score:
                                            best_score = score
                                            best_k = k
                                
                                # Final clustering with optimal k
                                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                                cell_df['subpopulation'] = kmeans.fit_predict(X_scaled)
                                
                                # Check if subpopulations are meaningful (silhouette > 0.25)
                                subpops_detected = best_score > 0.25
                                
                                # Create result structure
                                result = {
                                    'success': True,
                                    'n_cells_total': len(cell_df),
                                    'n_subpopulations': best_k if subpops_detected else 1,
                                    'subpopulations_detected': subpops_detected,
                                    'clustering_method': 'kmeans',
                                    'silhouette_score': best_score,
                                    'cell_level_data': cell_df,
                                    'features_used': available_features
                                }
                                
                                # Characterize subpopulations
                                if subpops_detected:
                                    subpop_chars = {}
                                    for subpop_id in range(best_k):
                                        subpop_cells = cell_df[cell_df['subpopulation'] == subpop_id]
                                        subpop_chars[f'subpop_{subpop_id}'] = {
                                            'subpopulation_id': int(subpop_id),
                                            'n_cells': len(subpop_cells),
                                            'fraction_of_total': len(subpop_cells) / len(cell_df),
                                            'feature_means': {f: float(subpop_cells[f].mean()) for f in available_features}
                                        }
                                    result['subpopulation_characteristics'] = subpop_chars
                                
                                # Concatenate all tracks with cell_id for later use
                                if all_tracks_with_cell_id:
                                    pooled = pd.concat(all_tracks_with_cell_id, ignore_index=True)
                                    # Map subpopulation labels
                                    cell_to_subpop = dict(zip(cell_df['cell_id'], cell_df['subpopulation']))
                                    pooled['subpopulation'] = pooled['cell_id'].map(cell_to_subpop)
                                    result['pooled_tracks'] = pooled
                                
                                # Store results
                                if 'subpopulation_results' not in st.session_state:
                                    st.session_state.subpopulation_results = {}
                                st.session_state.subpopulation_results[cond.id] = result
                                
                                if result['subpopulations_detected']:
                                    st.success(f"✓ Detected {result['n_subpopulations']} subpopulations in '{cond.name}'")
                                    st.info(f"📊 Analyzed {result['n_cells_total']} cells (files) with silhouette score: {result['silhouette_score']:.3f}")
                                    
                                    # Show subpopulation breakdown
                                    subpop_chars = result.get('subpopulation_characteristics', {})
                                    if subpop_chars:
                                        st.write("**Subpopulation Distribution:**")
                                        for subpop_name, chars in subpop_chars.items():
                                            st.write(f"- Subpop {chars['subpopulation_id']}: {chars['n_cells']} cells ({chars['fraction_of_total']:.1%})")
                                else:
                                    st.info(f"'{cond.name}' appears homogeneous (silhouette score: {result['silhouette_score']:.3f} < 0.25)")
                                    
                            except ImportError as e:
                                st.error(f"Required module not available: {str(e)}")
                                st.info("Install required packages: pip install scikit-learn scipy numpy")
                            except Exception as e:
                                st.error(f"Error in subpopulation analysis: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                
                with col2:
                    st.write("**Step 2: Pool by Subpopulation**")
                    
                    # Check if subpopulation analysis has been run
                    has_subpop_results = (
                        'subpopulation_results' in st.session_state and 
                        cond.id in st.session_state.subpopulation_results
                    )
                    
                    if has_subpop_results:
                        result = st.session_state.subpopulation_results[cond.id]
                        
                        if result.get('subpopulations_detected'):
                            # Show pooling options
                            pool_option = st.radio(
                                "Pooling Strategy:",
                                ["Pool all data together", "Pool by subpopulation"],
                                key=f"pool_option_{cond.id}"
                            )
                            
                            if st.button("📊 Pool & Load", key=f"pool_{cond.id}"):
                                cell_df = result.get('cell_level_data')
                                
                                if pool_option == "Pool by subpopulation":
                                    # Pool each subpopulation separately - use pre-loaded data with cell_ids
                                    st.info("Loading data with subpopulation assignments...")
                                    
                                    # Use the pooled tracks from subpopulation detection (already has cell_id and subpopulation labels)
                                    pooled = result.get('pooled_tracks')
                                    
                                    if pooled is not None and not pooled.empty and cell_df is not None:
                                        if 'cell_id' in pooled.columns and 'subpopulation' in pooled.columns:
                                            pooled['group'] = cond.name
                                            
                                            st.session_state.tracks_data = pooled
                                            try:
                                                st.session_state.track_statistics = calculate_track_statistics(pooled)
                                            except Exception:
                                                pass
                                            
                                            st.success(f"✓ Loaded {len(pooled)} tracks with subpopulation labels")
                                            
                                            # Show distribution
                                            subpop_dist = pooled['subpopulation'].value_counts()
                                            st.write("**Track distribution by subpopulation:**")
                                            for subpop_id, count in subpop_dist.items():
                                                st.write(f"- Subpopulation {int(subpop_id)}: {count} tracks")
                                        else:
                                            st.error("cell_id or subpopulation column not found in pooled data")
                                    else:
                                        st.error("Failed to load pooled tracks from subpopulation detection")
                                else:
                                    # Pool all together
                                    pooled_result = cond.pool_tracks()
                                    if isinstance(pooled_result, tuple):
                                        pooled, errors = pooled_result
                                    else:
                                        pooled = pooled_result
                                        errors = []
                                    
                                    if pooled is not None and not pooled.empty:
                                        st.session_state.tracks_data = pooled
                                        try:
                                            st.session_state.track_statistics = calculate_track_statistics(pooled)
                                        except Exception:
                                            pass
                                        st.success(f"✓ Pooled {len(pooled)} rows (ignoring subpopulations)")
                                        if errors:
                                            st.warning(f"Encountered {len(errors)} errors during pooling")
                        else:
                            # Homogeneous - just pool normally
                            st.info("Condition is homogeneous")
                            if st.button("📊 Pool Data", key=f"pool_{cond.id}"):
                                pooled_result = cond.pool_tracks()
                                if isinstance(pooled_result, tuple):
                                    pooled, errors = pooled_result
                                else:
                                    pooled = pooled_result
                                    errors = []
                                
                                if pooled is not None and not pooled.empty:
                                    st.session_state.tracks_data = pooled
                                    try:
                                        st.session_state.track_statistics = calculate_track_statistics(pooled)
                                    except Exception:
                                        pass
                                    st.success(f"✓ Pooled {len(pooled)} rows")
                                    if errors:
                                        st.warning(f"Encountered {len(errors)} errors")
                    else:
                        st.info("👈 Run subpopulation detection first")
                        st.caption("Or click below to pool without subpopulation analysis:")
                        if st.button("⚡ Quick Pool (Skip Analysis)", key=f"quick_pool_{cond.id}"):
                            pooled_result = cond.pool_tracks()
                            if isinstance(pooled_result, tuple):
                                pooled, errors = pooled_result
                            else:
                                pooled = pooled_result
                                errors = []
                            
                            if pooled is not None and not pooled.empty:
                                st.session_state.tracks_data = pooled
                                try:
                                    st.session_state.track_statistics = calculate_track_statistics(pooled)
                                except Exception:
                                    pass
                                st.success(f"✓ Pooled {len(pooled)} rows")
                                if errors:
                                    st.warning(f"Encountered {len(errors)} errors")

        # Batch Analysis Section
        st.divider()
        st.header("📊 Batch Analysis & Comparison")
        
        if proj.conditions and any(cond.files for cond in proj.conditions):
            st.info("Analyze and compare data across all conditions in this project")
            
            # Analysis options
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.subheader("Select Conditions to Compare")
                conditions_to_analyze = []
                for cond in proj.conditions:
                    if cond.files:
                        if st.checkbox(f"{cond.name} ({len(cond.files)} files)", 
                                     value=True, 
                                     key=f"analyze_cond_{cond.id}"):
                            conditions_to_analyze.append(cond)
            
            with analysis_col2:
                st.subheader("Analysis Options")
                
                # Select analyses to run
                with st.expander("📋 Select Analyses", expanded=False):
                    st.write("**Choose analyses to run on pooled condition data:**")
                    
                    if REPORT_GENERATOR_AVAILABLE:
                        from enhanced_report_generator import EnhancedSPTReportGenerator
                        temp_gen = EnhancedSPTReportGenerator(pd.DataFrame(), 0.1, 0.1)
                        
                        # Group analyses by category
                        analyses_by_category = {}
                        for key, analysis in temp_gen.available_analyses.items():
                            category = analysis.get('category', 'Other')
                            if category not in analyses_by_category:
                                analyses_by_category[category] = []
                            analyses_by_category[category].append((key, analysis['name']))
                        
                        selected_analyses = []
                        for category, analyses in sorted(analyses_by_category.items()):
                            st.write(f"**{category}**")
                            for key, name in analyses:
                                if st.checkbox(name, key=f"batch_analysis_{key}", value=key in ['basic_statistics', 'diffusion_analysis']):
                                    selected_analyses.append(key)
                    else:
                        selected_analyses = []
                        st.info("Enhanced Report Generator not available")
                
                # Quick analysis buttons
                if st.button("📊 Generate Individual Reports", type="primary"):
                    if len(conditions_to_analyze) < 1:
                        st.error("Select at least one condition to analyze")
                    elif not REPORT_GENERATOR_AVAILABLE:
                        st.error("Enhanced Report Generator not available")
                    else:
                        with st.spinner("Generating reports for each condition..."):
                            try:
                                # Pool data from each condition, breaking down by subpopulation if detected
                                condition_datasets = {}
                                pooling_errors = {}
                                subpop_info = {}
                                
                                for cond in conditions_to_analyze:
                                    # Check if subpopulation results exist for this condition
                                    has_subpop = ('subpopulation_results' in st.session_state and 
                                                 cond.id in st.session_state.subpopulation_results)
                                    
                                    if has_subpop:
                                        result = st.session_state.subpopulation_results[cond.id]
                                        
                                        if result.get('subpopulations_detected'):
                                            # Use pre-loaded pooled data with subpopulation labels
                                            pooled_df = result.get('pooled_tracks')
                                            
                                            if pooled_df is not None and not pooled_df.empty and 'subpopulation' in pooled_df.columns:
                                                # Split into separate datasets by subpopulation
                                                n_subpops = result['n_subpopulations']
                                                subpop_info[cond.name] = {'n_subpopulations': n_subpops}
                                                
                                                for subpop_id in range(n_subpops):
                                                    subpop_df = pooled_df[pooled_df['subpopulation'] == subpop_id].copy()
                                                    if not subpop_df.empty:
                                                        dataset_name = f"{cond.name} - Subpop {subpop_id}"
                                                        condition_datasets[dataset_name] = subpop_df
                                                continue
                                    
                                    # No subpopulations detected or no subpopulation analysis - pool normally
                                    pooled_result = cond.pool_tracks()
                                    if isinstance(pooled_result, tuple):
                                        pooled_df, errors = pooled_result
                                        if errors:
                                            pooling_errors[cond.name] = errors
                                    else:
                                        pooled_df = pooled_result
                                    
                                    if pooled_df is not None and not pooled_df.empty:
                                        condition_datasets[cond.name] = pooled_df
                                
                                if not condition_datasets:
                                    st.error("No valid data in selected conditions")
                                else:
                                    # Show pooling summary
                                    if subpop_info:
                                        st.success(f"✅ Pooled data from {len(conditions_to_analyze)} conditions ({len(condition_datasets)} datasets including subpopulations)")
                                        with st.expander("ℹ️ Subpopulation Breakdown", expanded=False):
                                            for cond_name, info in subpop_info.items():
                                                st.write(f"**{cond_name}:** {info['n_subpopulations']} subpopulations detected")
                                    else:
                                        st.success(f"✅ Pooled data from {len(condition_datasets)} conditions")
                                    
                                    if pooling_errors:
                                        with st.expander("⚠️ Pooling Warnings", expanded=False):
                                            for cond_name, errors in pooling_errors.items():
                                                st.warning(f"**{cond_name}:** {len(errors)} files had errors")
                                    
                                    # Get units
                                    pixel_size = st.session_state.get('pixel_size', 0.1)
                                    frame_interval = st.session_state.get('frame_interval', 0.1)
                                    
                                    # Generate reports
                                    from enhanced_report_generator import EnhancedSPTReportGenerator
                                    generator = EnhancedSPTReportGenerator(pd.DataFrame(), pixel_size, frame_interval)
                                    
                                    analyses_to_run = selected_analyses if selected_analyses else ['basic_statistics', 'diffusion_analysis']
                                    
                                    report_results = generator.generate_condition_reports(
                                        condition_datasets,
                                        analyses_to_run,
                                        pixel_size,
                                        frame_interval
                                    )
                                    
                                    # Store results in session state
                                    st.session_state['batch_report_results'] = report_results
                                    
                                    # Display results
                                    st.subheader("📊 Analysis Results")
                                    
                                    # Summary table
                                    summary_data = []
                                    for name, df in condition_datasets.items():
                                        n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
                                        n_frames = df['frame'].nunique() if 'frame' in df.columns else 0
                                        n_points = len(df)
                                        
                                        cond_results = report_results['conditions'].get(name, {})
                                        
                                        if cond_results.get('success', False):
                                            status = "✅ Success"
                                            error_msg = ""
                                        else:
                                            status = "❌ Failed"
                                            error_msg = cond_results.get('error', 'Unknown error')
                                        
                                        summary_row = {
                                            'Condition': name,
                                            'Status': status,
                                            'Tracks': n_tracks,
                                            'Frames': n_frames,
                                            'Data Points': n_points
                                        }
                                        
                                        if error_msg:
                                            summary_row['Error'] = error_msg
                                        
                                        summary_data.append(summary_row)
                                    
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    # Show individual condition results
                                    for cond_name, cond_result in report_results['conditions'].items():
                                        with st.expander(f"📈 {cond_name} - Detailed Results", expanded=False):
                                            if cond_result.get('success', False):
                                                st.write(f"**Analyses completed:** {len(cond_result.get('analysis_results', {}))}")
                                                st.write(f"**Figures generated:** {len(cond_result.get('figures', {}))}")
                                                
                                                # Show individual analysis statuses
                                                analysis_results = cond_result.get('analysis_results', {})
                                                if analysis_results:
                                                    st.write("**Analysis Status:**")
                                                    for analysis_key, analysis_result in analysis_results.items():
                                                        if isinstance(analysis_result, dict):
                                                            if analysis_result.get('success', True) and 'error' not in analysis_result:
                                                                st.write(f"- ✅ {analysis_key}")
                                                            else:
                                                                st.write(f"- ❌ {analysis_key}: {analysis_result.get('error', 'Unknown error')}")
                                                        else:
                                                            st.write(f"- ✅ {analysis_key}")
                                                
                                                # Show figures - handle both Plotly and Matplotlib
                                                for analysis_key, fig in cond_result.get('figures', {}).items():
                                                    if fig:
                                                        try:
                                                            # Check if it's a matplotlib figure
                                                            import matplotlib.figure
                                                            if isinstance(fig, matplotlib.figure.Figure):
                                                                st.pyplot(fig, use_container_width=True, key=f"fig_{cond_name}_{analysis_key}")
                                                            else:
                                                                # Assume it's a Plotly figure
                                                                st.plotly_chart(fig, use_container_width=True, key=f"fig_{cond_name}_{analysis_key}")
                                                        except Exception as e:
                                                            st.warning(f"Could not display figure for {analysis_key}: {e}")
                                            else:
                                                st.error(f"Analysis failed: {cond_result.get('error', 'Unknown error')}")
                                    
                                    # Show comparison results if available
                                    if report_results.get('comparisons') and len(condition_datasets) >= 2:
                                        st.divider()
                                        st.subheader("🔬 Statistical Comparisons")
                                        
                                        comparisons = report_results['comparisons']
                                        
                                        if comparisons.get('success', False):
                                            # Show metrics summary
                                            if 'metrics' in comparisons:
                                                st.write("**Summary Metrics:**")
                                                metrics_df = pd.DataFrame(comparisons['metrics']).T
                                                st.dataframe(metrics_df, use_container_width=True)
                                            
                                            # Show statistical tests
                                            if 'statistical_tests' in comparisons and comparisons['statistical_tests']:
                                                st.write("**Pairwise Statistical Tests:**")
                                                for comparison, tests in comparisons['statistical_tests'].items():
                                                    with st.expander(f"📊 {comparison}", expanded=False):
                                                        for metric, test_results in tests.items():
                                                            st.write(f"**{metric.replace('_', ' ').title()}:**")
                                                            if 't_test' in test_results:
                                                                p_val = test_results['t_test']['p_value']
                                                                significant = test_results.get('significant', False)
                                                                sig_text = "✅ Significant" if significant else "❌ Not significant"
                                                                st.write(f"- t-test p-value: {p_val:.4f} ({sig_text})")
                                                            if 'mann_whitney' in test_results:
                                                                p_val = test_results['mann_whitney']['p_value']
                                                                st.write(f"- Mann-Whitney p-value: {p_val:.4f}")
                                            
                                            # Show comparison figures
                                            if 'figures' in comparisons and comparisons['figures'].get('comparison_boxplots'):
                                                fig = comparisons['figures']['comparison_boxplots']
                                                try:
                                                    # Check if it's a matplotlib figure
                                                    import matplotlib.figure
                                                    if isinstance(fig, matplotlib.figure.Figure):
                                                        st.pyplot(fig, use_container_width=True, key="comparison_boxplots")
                                                    else:
                                                        # Assume it's a Plotly figure
                                                        st.plotly_chart(fig, use_container_width=True, key="comparison_boxplots")
                                                except Exception as e:
                                                    st.warning(f"Could not display comparison figure: {e}")
                                        else:
                                            st.warning(f"Comparison analysis failed: {comparisons.get('error', 'Unknown error')}")
                                    
                                    # Download options
                                    st.divider()
                                    st.subheader("💾 Export Results")
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        # JSON export
                                        report_json = json.dumps(report_results, indent=2, default=str)
                                        st.download_button(
                                            "📄 JSON Report",
                                            data=report_json,
                                            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )
                                    
                                    with col2:
                                        # CSV export of metrics
                                        if report_results.get('comparisons', {}).get('metrics'):
                                            metrics_csv = pd.DataFrame(report_results['comparisons']['metrics']).T.to_csv()
                                            st.download_button(
                                                "📊 Metrics CSV",
                                                data=metrics_csv,
                                                file_name=f"batch_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv"
                                            )
                                    
                                    with col3:
                                        # HTML export (static)
                                        try:
                                            html_report = _generate_batch_html_report(
                                                report_results, 
                                                condition_datasets, 
                                                pixel_size, 
                                                frame_interval,
                                                interactive=False
                                            )
                                            st.download_button(
                                                "📰 HTML Report",
                                                data=html_report,
                                                file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                                mime="text/html"
                                            )
                                        except Exception as e:
                                            st.error(f"HTML export error: {e}")
                                    
                                    with col4:
                                        # Interactive HTML export
                                        try:
                                            interactive_html = _generate_batch_html_report(
                                                report_results, 
                                                condition_datasets, 
                                                pixel_size, 
                                                frame_interval,
                                                interactive=True
                                            )
                                            st.download_button(
                                                "🎨 Interactive HTML",
                                                data=interactive_html,
                                                file_name=f"batch_report_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                                mime="text/html"
                                            )
                                        except Exception as e:
                                            st.error(f"Interactive HTML export error: {e}")
                                    
                            except Exception as e:
                                st.error(f"Error generating reports: {e}")
                                import traceback
                                with st.expander("🐛 Error Details"):
                                    st.code(traceback.format_exc())
                
                st.divider()
                
                # Legacy comparative report button (simpler version)
                if st.button("📈 Generate Comparative Report", type="secondary"):
                    if len(conditions_to_analyze) < 1:
                        st.error("Select at least one condition to analyze")
                    else:
                        with st.spinner("Generating comparative report..."):
                            try:
                                # Pool data from each condition, breaking down by subpopulation if detected
                                condition_datasets = {}
                                subpop_info_comp = {}
                                
                                for cond in conditions_to_analyze:
                                    # Check if subpopulation results exist for this condition
                                    has_subpop = ('subpopulation_results' in st.session_state and 
                                                 cond.id in st.session_state.subpopulation_results)
                                    
                                    if has_subpop:
                                        result = st.session_state.subpopulation_results[cond.id]
                                        
                                        if result.get('subpopulations_detected'):
                                            # Use pre-loaded pooled data with subpopulation labels
                                            pooled_df = result.get('pooled_tracks')
                                            
                                            if pooled_df is not None and not pooled_df.empty and 'subpopulation' in pooled_df.columns:
                                                # Split into separate datasets by subpopulation
                                                n_subpops = result['n_subpopulations']
                                                subpop_info_comp[cond.name] = {'n_subpopulations': n_subpops}
                                                
                                                for subpop_id in range(n_subpops):
                                                    subpop_df = pooled_df[pooled_df['subpopulation'] == subpop_id].copy()
                                                    if not subpop_df.empty:
                                                        dataset_name = f"{cond.name} - Subpop {subpop_id}"
                                                        condition_datasets[dataset_name] = subpop_df
                                                continue
                                    
                                    # No subpopulations detected or no subpopulation analysis - pool normally
                                    pooled_result = cond.pool_tracks()
                                    if isinstance(pooled_result, tuple):
                                        pooled_df, _ = pooled_result
                                    else:
                                        pooled_df = pooled_result
                                    if pooled_df is not None and not pooled_df.empty:
                                        condition_datasets[cond.name] = pooled_df
                                
                                if not condition_datasets:
                                    st.error("No valid data in selected conditions")
                                else:
                                    if subpop_info_comp:
                                        st.success(f"✅ Pooled data from {len(conditions_to_analyze)} conditions ({len(condition_datasets)} datasets including subpopulations)")
                                        with st.expander("ℹ️ Subpopulation Breakdown", expanded=False):
                                            for cond_name, info in subpop_info_comp.items():
                                                st.write(f"**{cond_name}:** {info['n_subpopulations']} subpopulations detected")
                                    else:
                                        st.success(f"✅ Pooled data from {len(condition_datasets)} conditions")
                                    
                                    # Show summary
                                    st.subheader("Condition Summaries")
                                    summary_data = []
                                    for name, df in condition_datasets.items():
                                        n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
                                        n_frames = df['frame'].nunique() if 'frame' in df.columns else 0
                                        n_points = len(df)
                                        summary_data.append({
                                            'Condition': name,
                                            'Tracks': n_tracks,
                                            'Frames': n_frames,
                                            'Data Points': n_points
                                        })
                                    
                                    import pandas as pd
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    # Option to run enhanced report generator on each condition
                                    st.subheader("Advanced Analysis")
                                    if REPORT_GENERATOR_AVAILABLE:
                                        st.info("💡 Use the Enhanced Report Generator tab to run detailed analyses on individual conditions")
                                        
                                        # Quick access to load condition into main analysis
                                        st.write("**Load condition for detailed analysis:**")
                                        selected_cond_name = st.selectbox(
                                            "Select condition to load into main workspace",
                                            options=list(condition_datasets.keys()),
                                            key="load_cond_to_workspace"
                                        )
                                        
                                        if st.button("Load Selected Condition", key="load_cond_btn"):
                                            st.session_state.tracks_data = condition_datasets[selected_cond_name]
                                            try:
                                                st.session_state.track_statistics = calculate_track_statistics(
                                                    condition_datasets[selected_cond_name]
                                                )
                                            except Exception:
                                                pass
                                            st.success(f"✅ Loaded '{selected_cond_name}' into main workspace. Go to 'Enhanced Report Generator' tab to run analyses.")
                                    else:
                                        st.warning("Enhanced Report Generator not available")
                                    
                            except Exception as e:
                                st.error(f"Error generating report: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                
                # Export options
                st.divider()
                if st.button("💾 Export All Condition Data"):
                    try:
                        export_data = {}
                        for cond in conditions_to_analyze:
                            pooled_result = cond.pool_tracks()
                            if isinstance(pooled_result, tuple):
                                pooled_df, _ = pooled_result
                            else:
                                pooled_df = pooled_result
                            if pooled_df is not None and not pooled_df.empty:
                                export_data[cond.name] = pooled_df
                        
                        if export_data:
                            # Create a combined CSV with condition labels
                            combined_rows = []
                            for cond_name, df in export_data.items():
                                df_copy = df.copy()
                                df_copy['condition'] = cond_name
                                combined_rows.append(df_copy)
                            
                            import pandas as pd
                            combined_df = pd.concat(combined_rows, ignore_index=True)
                            
                            csv = combined_df.to_csv(index=False)
                            st.download_button(
                                label="Download Combined CSV",
                                data=csv,
                                file_name=f"{proj.name}_all_conditions.csv",
                                mime="text/csv"
                            )
                            st.success(f"✅ Prepared {len(combined_df)} rows from {len(export_data)} conditions")
                        else:
                            st.error("No data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        else:
            st.info("Add files to conditions to enable batch analysis")

        # Save project explicitly
        if st.button("Save Project"):
            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
            st.success("Project saved.")

# Image Processing Page
elif st.session_state.active_page == "Image Processing":
    st.title("Image Processing & Nuclear Density Analysis")
    
    # Image Processing tabs
    img_tabs = st.tabs([
        "Segmentation",
        "Nuclear Density Analysis", 
        "Advanced Segmentation",  # Added advanced segmentation tab
        "Export Results"
    ])
    
    # Tab 1: Segmentation
    with img_tabs[0]:
        st.subheader("Image Segmentation")
        
        # Check for mask images from Data Loading tab
        if 'mask_images' not in st.session_state or st.session_state.mask_images is None:
            st.warning("Upload mask images in the Data Loading tab first to perform segmentation.")
            st.info("Go to Data Loading → 'Images for Mask Generation' to upload images for processing.")
        else:
            # Handle multichannel images
            mask_image_data = st.session_state.mask_images
            
            # Log mask image data information
            logger.debug(f"Mask image data type: {type(mask_image_data)}")
            if isinstance(mask_image_data, np.ndarray):
                logger.debug(f"Mask shape: {mask_image_data.shape}")
            elif isinstance(mask_image_data, list):
                logger.debug(f"Mask list length: {len(mask_image_data)}")
                if len(mask_image_data) > 0:
                    logger.debug(f"First mask item shape: {mask_image_data[0].shape}")
            
            # Check different multichannel scenarios
            multichannel_detected = False
            
            # Case 1: Direct numpy array with channels
            if isinstance(mask_image_data, np.ndarray) and len(mask_image_data.shape) == 3 and mask_image_data.shape[2] > 1:
                multichannel_detected = True
                num_channels = mask_image_data.shape[2]
                multichannel_data = mask_image_data
                
            # Case 2: List with single multichannel array
            elif isinstance(mask_image_data, list) and len(mask_image_data) == 1 and len(mask_image_data[0].shape) == 3 and mask_image_data[0].shape[2] > 1:
                multichannel_detected = True
                num_channels = mask_image_data[0].shape[2]
                multichannel_data = mask_image_data[0]
                
            if multichannel_detected:
                # Multichannel image detected
                st.info(f"Multichannel image detected with {num_channels} channels")
                
                # Channel selection for segmentation (support multi-channel)
                st.subheader("Channel Selection for Segmentation")

                # Channel labeling helpers
                seg_label_key = "mask_channel_labels"
                if seg_label_key not in st.session_state:
                    st.session_state[seg_label_key] = {}
                # Build display labels with saved or suggested names
                def _seg_label(idx: int) -> str:
                    saved = st.session_state[seg_label_key].get(idx)
                    return f"Channel {idx + 1}" if not saved else f"{saved} (C{idx + 1})"

                with st.expander("Name channels (optional)", expanded=False):
                    common = channel_manager.get_common_names()
                    cols = st.columns(min(4, num_channels))
                    for i in range(num_channels):
                        with cols[i % len(cols)]:
                            current = st.session_state[seg_label_key].get(i, "")
                            new_name = st.text_input(
                                f"Channel {i+1} name",
                                value=current,
                                key=f"seg_label_input_{i}",
                                placeholder="e.g., DAPI, GFP"
                            )
                            if new_name and new_name != current:
                                is_valid, validated = channel_manager.validate_name(new_name)
                                if is_valid:
                                    st.session_state[seg_label_key][i] = validated
                                    channel_manager.add_channel_name(validated)
                                else:
                                    st.warning(f"Invalid name for C{i+1}: {validated}")
                default_sel = st.session_state.get("segmentation_channels", [0])
                segmentation_channels = st.multiselect(
                    "Choose channel(s) for segmentation:",
                    options=list(range(num_channels)),
                    default=[ci for ci in default_sel if 0 <= ci < num_channels] or [0],
                    format_func=lambda x: _seg_label(x),
                    key="segmentation_channels_select",
                    help="Select one or more channels to build the segmentation image"
                )

                fusion_mode = st.selectbox(
                    "Fusion mode",
                    ["average", "max", "min", "sum", "weighted"],
                    index=["average", "max", "min", "sum", "weighted"].index(
                        st.session_state.get("segmentation_fusion_mode", "average")
                    ),
                    help="How to combine multiple channels into a single image for segmentation"
                )

                weights = None
                if fusion_mode == "weighted" and len(segmentation_channels) > 1:
                    weights_text = st.text_input(
                        "Weights (comma-separated)",
                        value=st.session_state.get("segmentation_weights", ",".join(["1"] * len(segmentation_channels))),
                        help="Provide one weight per selected channel, e.g., 1,2,1"
                    )
                    try:
                        weights = [float(x.strip()) for x in weights_text.split(",") if x.strip() != ""]
                    except Exception:
                        weights = None

                # Display channel previews in tabs
                channel_tabs = st.tabs([f"Channel {i+1}" for i in range(num_channels)])
                for i in range(num_channels):
                    with channel_tabs[i]:
                        channel_image = multichannel_data[:, :, i]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.image(normalize_image_for_display(channel_image), caption=f"Channel {i + 1}", use_container_width=True)
                        with col2:
                            st.write("**Statistics:**")
                            st.metric("Min", f"{channel_image.min():.1f}")
                            st.metric("Max", f"{channel_image.max():.1f}")
                            st.metric("Mean", f"{channel_image.mean():.1f}")
                            st.metric("Std", f"{channel_image.std():.1f}")
                            if i in segmentation_channels:
                                st.success("Selected")

                # Use selected channels combined
                try:
                    current_image = _combine_channels(multichannel_data, segmentation_channels or [0], fusion_mode, weights)
                except Exception as _e:
                    st.warning(f"Combining channels failed ({_e}); falling back to channel 1")
                    current_image = multichannel_data[:, :, 0]

                # Persist choices
                st.session_state.segmentation_channels = segmentation_channels or [0]
                st.session_state.segmentation_fusion_mode = fusion_mode
                if weights is not None:
                    st.session_state.segmentation_weights = ",".join(map(str, weights))
                
                sel_label = ", ".join([f"C{c+1}" for c in (segmentation_channels or [0])])
                st.subheader(f"Using {sel_label} with '{fusion_mode}' fusion for Segmentation")
                
            elif isinstance(mask_image_data, list) and len(mask_image_data) > 0:
                # List of single-channel images
                current_image = mask_image_data[0]
                st.info("Using first image from the loaded sequence")
            else:
                # Single channel image
                current_image = mask_image_data
                st.info("Single channel image loaded for segmentation")
            
            # Display selected image for segmentation (ensure single channel)
            if len(current_image.shape) == 3:
                # If still multichannel, take first channel
                current_image = current_image[:, :, 0] if current_image.shape[2] > 1 else current_image.squeeze()
            
            display_image = current_image.astype(np.float32)
            if display_image.max() > display_image.min():
                display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
            st.image(normalize_image_for_display(display_image), caption="Image Selected for Segmentation", use_container_width=True)
            
            # Segmentation approach selection
            seg_approach = st.selectbox(
                "Segmentation Approach",
                ["Simple Segmentation", "Nuclear Segmentation (Two-Step)"],
                help="Choose segmentation approach: Simple (binary) or Nuclear (two-step process)"
            )
            
            if seg_approach == "Nuclear Segmentation (Two-Step)":
                st.markdown("### Two-Step Nuclear Segmentation")
                st.info("Choose methods and preprocessing that will be applied to the original image for both nuclear boundary detection and interior segmentation")
                
                # Method Selection
                st.markdown("#### Method Selection")
                col1_method, col2_method = st.columns(2)
                
                with col1_method:
                    st.markdown("**Nuclear Boundary Method**")
                    nuclear_method = st.selectbox(
                        "Nuclear Boundary Detection",
                        ["Otsu", "Triangle", "Manual"],
                        help="Method for detecting nuclear boundaries",
                        key="nuclear_method_select"
                    )
                    
                with col2_method:
                    st.markdown("**Interior Segmentation Method**")
                    internal_method = st.selectbox(
                        "Interior Segmentation",
                        ["Otsu", "Triangle", "Manual", "Density Map"],
                        help="Method for segmenting within nuclear boundaries",
                        key="internal_method_select"
                    )
                
                # Image Preprocessing (applied to original image)
                st.markdown("#### Image Preprocessing")
                st.info("Preprocessing will be applied to the original image before both nuclear and interior segmentation")
                
                col1_preprocess, col2_preprocess = st.columns(2)
                with col1_preprocess:
                    apply_preprocessing = st.checkbox("Apply Image Preprocessing", value=False, key="apply_preprocessing_nuclear")
                    if apply_preprocessing:
                        preprocessing_method = st.selectbox("Preprocessing Method", ["Median Filter", "Gaussian Filter"], key="preprocessing_method_nuclear")
                
                with col2_preprocess:
                    if apply_preprocessing:
                        if preprocessing_method == "Median Filter":
                            preprocessing_size = st.slider("Median Filter Size", 3, 15, 5, 2, key="preprocessing_size_nuclear")
                        else:  # Gaussian
                            preprocessing_sigma = st.slider("Gaussian Sigma", 0.5, 3.0, 1.0, 0.1, key="preprocessing_sigma_nuclear")
                
                # Nuclear Boundary Parameters
                st.markdown("#### Nuclear Boundary Parameters")
                col1_nuclear, col2_nuclear = st.columns(2)
                with col1_nuclear:
                    if nuclear_method == "Manual":
                        nuclear_threshold = st.slider(
                            "Nuclear Threshold", 
                            float(current_image.min()), 
                            float(current_image.max()), 
                            float((current_image.min() + current_image.max()) / 2),
                            key="nuclear_threshold_new"
                        )
                    min_nuclear_size = st.slider("Minimum Nuclear Size (pixels)", 100, 10000, 1000, 100, key="min_nuc_size_new")
                
                with col2_nuclear:
                    nuclear_closing = st.slider("Nuclear Closing Size", 1, 20, 5, 1, key="nuc_closing_new")
                    nuclear_smooth = st.checkbox("Smooth Nuclear Boundaries", value=True, key="nuc_smooth_new")
                
                # Interior Segmentation Parameters
                st.markdown("#### Interior Segmentation Parameters")
                
                if internal_method == "Manual":
                    internal_threshold = st.slider(
                        "Internal Threshold", 
                        float(current_image.min()), 
                        float(current_image.max()), 
                        float((current_image.min() + current_image.max()) / 2),
                        key="internal_threshold_new"
                    )
                elif internal_method == "Density Map":
                    col1_int, col2_int = st.columns(2)
                    with col1_int:
                        sigma_hp = st.slider("High-pass Sigma", 0.5, 10.0, 2.0, 0.1, key="internal_sigma_new")
                        disk_radius = st.slider("Background Disk Radius", 5, 50, 10, 1, key="internal_disk_new")
                    with col2_int:
                        pcutoff_in = st.slider("Interior Threshold", 0.01, 0.5, 0.10, 0.01, key="internal_pcutoff_in_new")
                        pcutoff_out = st.slider("Background Threshold", 0.01, 0.5, 0.10, 0.01, key="internal_pcutoff_out_new")
                
                # Initialize default values for parameters not set
                if internal_method != "Density Map":
                    sigma_hp = 2.0
                    disk_radius = 10
                    pcutoff_in = 0.10
                    pcutoff_out = 0.10
                
                if internal_method != "Manual":
                    internal_threshold = (current_image.min() + current_image.max()) / 2
                
                if nuclear_method != "Manual":
                    nuclear_threshold = (current_image.min() + current_image.max()) * 0.6
                
                # Initialize preprocessing defaults
                if 'apply_preprocessing' not in locals():
                    apply_preprocessing = False
                    preprocessing_method = "Median Filter"
                    preprocessing_size = 5
                    preprocessing_sigma = 1.0
                
                if st.button("Apply Nuclear Segmentation", key="nuclear_seg"):
                    with st.spinner("Performing two-step nuclear segmentation..."):
                        try:
                            from segmentation import apply_nuclear_segmentation_with_preprocessing
                            
                            # Preprocessing parameters (applied to original image)
                            preprocessing_params = {
                                'apply_preprocessing': apply_preprocessing
                            }
                            if apply_preprocessing:
                                preprocessing_params.update({
                                    'method': preprocessing_method,
                                    'size': preprocessing_size if preprocessing_method == "Median Filter" else None,
                                    'sigma': preprocessing_sigma if preprocessing_method == "Gaussian Filter" else None
                                })
                            
                            # Parameters for nuclear boundary detection
                            nuclear_params = {
                                'method': nuclear_method,
                                'min_size': min_nuclear_size,
                                'closing_size': nuclear_closing,
                                'smooth_boundaries': nuclear_smooth
                            }
                            
                            if nuclear_method == "Manual":
                                nuclear_params['threshold'] = nuclear_threshold
                            
                            # Parameters for internal segmentation
                            internal_params = {'method': internal_method}
                            if internal_method == "Manual":
                                internal_params['threshold'] = internal_threshold
                            elif internal_method == "Density Map":
                                internal_params.update({
                                    'sigma_hp': sigma_hp,
                                    'disk_radius': disk_radius,
                                    'pcutoff_in': pcutoff_in,
                                    'pcutoff_out': pcutoff_out
                                })
                            
                            # Apply nuclear segmentation with preprocessing
                            nuclear_mask, internal_classes, combined_result = apply_nuclear_segmentation_with_preprocessing(
                                current_image, preprocessing_params, nuclear_params, internal_params
                            )
                            
                            # Display results
                            st.success("Nuclear segmentation completed!")
                            
                            col1_res, col2_res, col3_res = st.columns(3)
                            
                            with col1_res:
                                st.subheader("Nuclear Mask")
                                st.image(nuclear_mask.astype(np.uint8) * 255, 
                                        caption="Nuclear boundary detection", 
                                        use_container_width=True, clamp=True)
                            
                            with col2_res:
                                st.subheader("Internal Classes")
                                # Color-code internal classes
                                display_internal = np.zeros_like(internal_classes, dtype=np.uint8)
                                display_internal[internal_classes == 1] = 127
                                display_internal[internal_classes == 2] = 255
                                st.image(display_internal, 
                                        caption="Internal segmentation", 
                                        use_container_width=True, clamp=True)
                            
                            with col3_res:
                                st.subheader("Combined Result")
                                # Color-code combined result: 0=background, 1=nuclear class 1, 2=nuclear class 2
                                display_combined = np.zeros_like(combined_result, dtype=np.uint8)
                                display_combined[combined_result == 1] = 127
                                display_combined[combined_result == 2] = 255
                                st.image(display_combined, 
                                        caption="Final segmentation (0=background, 1=class1, 2=class2)", 
                                        use_container_width=True, clamp=True)
                            
                            # Statistics
                            st.subheader("Segmentation Statistics")
                            col1_stats, col2_stats, col3_stats = st.columns(3)
                            
                            with col1_stats:
                                background_pixels = np.sum(combined_result == 0)
                                st.metric("Background pixels", f"{background_pixels:,}")
                            
                            with col2_stats:
                                class1_pixels = np.sum(combined_result == 1)
                                st.metric("Nuclear class 1 pixels", f"{class1_pixels:,}")
                            
                            with col3_stats:
                                class2_pixels = np.sum(combined_result == 2)
                                st.metric("Nuclear class 2 pixels", f"{class2_pixels:,}")
                            
                            # Create complete three-class segmentation (background, class 1, class 2)
                            three_class_result = np.zeros_like(combined_result)
                            three_class_result[combined_result == 1] = 1  # Nuclear class 1
                            three_class_result[combined_result == 2] = 2  # Nuclear class 2
                            # Background remains 0
                            
                            # Compose trace info for segmentation channels/fusion
                            seg_chs = st.session_state.get('segmentation_channels', [0])
                            seg_mode = st.session_state.get('segmentation_fusion_mode', 'average')
                            seg_lbls = st.session_state.get('mask_channel_labels', {})
                            ch_label_list = [seg_lbls.get(ci, f"C{ci+1}") for ci in seg_chs]
                            trace = f" | channels: {', '.join(ch_label_list)} | fusion: {seg_mode}"

                            # Store generated masks for analysis with trace
                            store_mask("Nuclear_Boundaries", nuclear_mask, "Nuclear Boundary", 
                                     f"Nuclear boundaries detected using {nuclear_method}{trace}")
                            store_mask("Nuclear_Internal_Classes", internal_classes, "Nuclear Internal", 
                                     f"Internal nuclear segmentation using {internal_method}{trace}")
                            store_mask("Nuclear_Combined", combined_result, "Nuclear Combined", 
                                     f"Combined nuclear segmentation: {nuclear_method} + {internal_method}{trace}")
                            store_mask("Nuclear_Three_Class", three_class_result, "Nuclear Three-Class", 
                                     f"Complete three-class nuclear segmentation: 0=Background, 1={nuclear_method}_Class1, 2={nuclear_method}_Class2{trace}")
                            
                            # Store results in session state
                            st.session_state.nuclear_segmentation_result = combined_result
                            st.session_state.nuclear_mask = nuclear_mask
                            st.session_state.internal_classes = internal_classes
                            
                            # Show generated masks info
                            available_masks = get_available_masks()
                            if len(available_masks) >= 3:
                                st.info(f"Generated masks stored: {', '.join(available_masks[-3:])}")
                            
                        except Exception as e:
                            st.error(f"Nuclear segmentation failed: {str(e)}")
                            st.info("This feature requires the segmentation module to be properly configured")
            
            else:
                # Simple segmentation method selection
                seg_method = st.selectbox(
                    "Segmentation Method",
                    ["Otsu", "Triangle", "Manual", "Density Map"],
                    help="Choose the segmentation method for your image"
                )
                
                if seg_method == "Density Map":
                    st.markdown("### Density Map Segmentation Parameters")
                    col1, col2 = st.columns(2)
                
                    with col1:
                        sigma_hp = st.slider("High-pass Gaussian Sigma", 0.5, 10.0, 2.0, 0.1, 
                                           help="Sigma for initial smoothing and background subtraction")
                        disk_radius = st.slider("Background Disk Radius", 5, 50, 10, 1,
                                              help="Size of morphological disk for background estimation")
                    
                    with col2:
                        pcutoff_in = st.slider("Interior Threshold", 0.01, 0.5, 0.10, 0.01,
                                             help="Threshold for detecting interior regions")
                        pcutoff_out = st.slider("Background Threshold", 0.01, 0.5, 0.10, 0.01,
                                              help="Threshold for detecting background regions")
                    
                    if st.button("Apply Density Map Segmentation", key="density_seg"):
                        with st.spinner("Applying density map segmentation..."):
                            try:
                                density_results = density_map_threshold(
                                    current_image,
                                    gaussian_sigma_hp=sigma_hp,
                                    disk_radius=disk_radius,
                                    pcutoff_in=pcutoff_in,
                                    pcutoff_out=pcutoff_out
                                )
                                
                                mask = density_results['mask_in']
                                mask_out = density_results.get('mask_out', np.zeros_like(mask))
                                
                                # Create three-class density map (0=background, 1=interior, 2=exterior/boundary)
                                density_three_class = np.zeros_like(mask, dtype=np.uint8)
                                density_three_class[mask] = 1  # Interior regions
                                density_three_class[mask_out] = 2  # Exterior/boundary regions
                                # Background remains 0
                                
                                st.session_state.segmentation_mask = mask
                                
                                # Trace details
                                seg_chs = st.session_state.get('segmentation_channels', [0])
                                seg_mode = st.session_state.get('segmentation_fusion_mode', 'average')
                                seg_lbls = st.session_state.get('mask_channel_labels', {})
                                ch_label_list = [seg_lbls.get(ci, f"C{ci+1}") for ci in seg_chs]
                                trace = f" | channels: {', '.join(ch_label_list)} | fusion: {seg_mode}"

                                # Store generated masks for analysis
                                store_mask("Simple_Density_Map", mask.astype(np.uint8), "Density Map", 
                                         f"Density map segmentation (sigma_hp={sigma_hp}, disk_radius={disk_radius}){trace}")
                                store_mask("Density_Three_Class", density_three_class, "Density Three-Class", 
                                         f"Complete density map classification: 0=Background, 1=Interior, 2=Exterior (sigma_hp={sigma_hp}){trace}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Original Image**")
                                    st.image(display_image, use_container_width=True)
                                
                                with col2:
                                    st.write("**Density Map Segmentation**")
                                    mask_display = (mask * 255).astype(np.uint8)
                                    st.image(mask_display, use_container_width=True)
                                
                                # Additional density map info
                                st.write("**High-pass Filtered Image**")
                                st.image(density_results['hp'], caption="Normalized High-pass", use_container_width=True)
                                
                                # Analysis
                                analysis = analyze_density_segmentation(mask)
                                st.write(f"Objects detected: {analysis['num_objects']}")
                                st.write(f"Total area: {analysis['total_area']} pixels")
                                
                                st.success("Density map mask stored for analysis!")
                                available_masks = get_available_masks()
                                if available_masks:
                                    st.info(f"Mask '{available_masks[-1]}' ready for region-based analysis")
                                
                            except Exception as e:
                                st.error(f"Error in density map segmentation: {str(e)}")
                
                elif seg_method == "Manual":
                    threshold_value = st.slider(
                        "Threshold Value", 
                        float(current_image.min()), 
                        float(current_image.max()), 
                        float((current_image.min() + current_image.max()) / 2)
                    )
                    mask, info = enhanced_threshold_image(current_image, method='manual', manual_threshold=threshold_value)
                elif seg_method == "Otsu":
                    mask, info = enhanced_threshold_image(current_image, method='otsu')
                elif seg_method == "Triangle":
                    mask, info = enhanced_threshold_image(current_image, method='triangle')
                
                # Display segmentation result for simple methods
                if seg_method in ["Manual", "Otsu", "Triangle"] and 'mask' in locals():
                    st.session_state.segmentation_mask = mask
                    
                    # Trace details
                    seg_chs = st.session_state.get('segmentation_channels', [0])
                    seg_mode = st.session_state.get('segmentation_fusion_mode', 'average')
                    seg_lbls = st.session_state.get('mask_channel_labels', {})
                    ch_label_list = [seg_lbls.get(ci, f"C{ci+1}") for ci in seg_chs]
                    trace = f" | channels: {', '.join(ch_label_list)} | fusion: {seg_mode}"

                    # Store generated mask for analysis
                    store_mask(f"Simple_{seg_method}", mask.astype(np.uint8), f"Simple {seg_method}", 
                             f"{seg_method} threshold segmentation{trace}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Image**")
                        st.image(display_image, use_container_width=True)
                    
                    with col2:
                        st.write("**Segmentation Result**")
                        mask_display = (mask * 255).astype(np.uint8)
                        st.image(mask_display, use_container_width=True)
                    
                    # Analysis
                    analysis = analyze_density_segmentation(mask)
                    st.write(f"Objects detected: {analysis['num_objects']}")
                    st.write(f"Total area: {analysis['total_area']} pixels")
                    
                    st.success(f"{seg_method} mask stored for analysis!")
                    available_masks = get_available_masks()
                    if available_masks:
                        mask_names = list(available_masks.keys())
                        st.info(f"Mask '{mask_names[-1]}' ready for region-based analysis")
    
    # Tab 2: Nuclear Density Analysis
    with img_tabs[1]:
        st.subheader("Nuclear Density Mapping & Classification")
        
        if 'mask_images' not in st.session_state or st.session_state.mask_images is None:
            st.warning("Upload mask images in the Data Loading tab first to perform nuclear density analysis.")
            st.info("Go to Data Loading → 'Images for Mask Generation' to upload images for processing.")
        else:
            # Handle multichannel images
            mask_image_data = st.session_state.mask_images
            
            # Check different multichannel scenarios
            multichannel_detected = False
            
            # Case 1: Direct numpy array with channels
            if isinstance(mask_image_data, np.ndarray) and len(mask_image_data.shape) == 3 and mask_image_data.shape[2] > 1:
                multichannel_detected = True
                num_channels = mask_image_data.shape[2]
                multichannel_data = mask_image_data
                
            # Case 2: List with single multichannel array
            elif isinstance(mask_image_data, list) and len(mask_image_data) == 1 and len(mask_image_data[0].shape) == 3 and mask_image_data[0].shape[2] > 1:
                multichannel_detected = True
                num_channels = mask_image_data[0].shape[2]
                multichannel_data = mask_image_data[0]
                
            if multichannel_detected:
                # Multichannel image detected
                st.info(f"Multichannel image detected with {num_channels} channels")
                
                # Channel selection for nuclear density analysis
                st.subheader("Channel Selection for Nuclear Density Analysis")
                analysis_channel = st.radio(
                    "Choose channel for nuclear density analysis:",
                    range(num_channels),
                    format_func=lambda x: f"Channel {x + 1}",
                    key="nuclear_density_channel_radio",
                    horizontal=True,
                    help="Select which channel to use for nuclear density mapping and classification"
                )
                
                # Display channel previews in tabs
                channel_tabs = st.tabs([f"Channel {i+1}" for i in range(num_channels)])
                
                for i in range(num_channels):
                    with channel_tabs[i]:
                        channel_image = multichannel_data[:, :, i]
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Use normalized display function
                            st.image(normalize_image_for_display(channel_image), caption=f"Channel {i + 1}", use_container_width=True)
                        
                        with col2:
                            st.write("**Statistics:**")
                            st.metric("Min", f"{channel_image.min():.1f}")
                            st.metric("Max", f"{channel_image.max():.1f}")
                            st.metric("Mean", f"{channel_image.mean():.1f}")
                            st.metric("Std", f"{channel_image.std():.1f}")
                            
                            if i == analysis_channel:
                                st.success("Selected for analysis")
                
                # Use selected channel
                current_image = multichannel_data[:, :, analysis_channel]
                st.subheader(f"Using Channel {analysis_channel + 1} for Nuclear Density Analysis")
                
            elif isinstance(mask_image_data, list) and len(mask_image_data) > 0:
                # List of single-channel images
                current_image = mask_image_data[0]
                st.info("Using first image from the loaded sequence")
            else:
                # Single channel image
                current_image = mask_image_data
                st.info("Single channel image loaded for nuclear density analysis")
            
            # Display selected image for nuclear density analysis (ensure single channel)
            if len(current_image.shape) == 3:
                # If still multichannel, take first channel
                current_image = current_image[:, :, 0] if current_image.shape[2] > 1 else current_image.squeeze()
            
            display_image = current_image.astype(np.float32)
            if display_image.max() > display_image.min():
                display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
            st.image(normalize_image_for_display(display_image), caption="Image Selected for Nuclear Density Analysis", use_container_width=True)
            
            # Nuclear segmentation parameters
            st.markdown("### Step 1: Nuclear Boundary Detection")
            col1, col2 = st.columns(2)
            
            with col1:
                threshold_method = st.selectbox(
                    "Threshold Method",
                    ["Otsu", "Triangle", "Manual"],
                    help="""Automatic threshold selection method for nuclear boundary detection:
• Otsu: Assumes bimodal intensity distribution (nucleus vs background). Works well for clear nuclear boundaries.
• Triangle: Good for images with skewed intensity distributions or when nucleus occupies small area.
• Manual: Set threshold manually for precise control or when automatic methods fail.

Otsu is recommended for most fluorescent nuclear images with good contrast."""
                )
                
                if threshold_method == "Manual":
                    manual_threshold = st.slider(
                        "Manual Threshold", 
                        float(current_image.min()), 
                        float(current_image.max()), 
                        float((current_image.min() + current_image.max()) / 2)
                    )
            
            with col2:
                min_nuclear_size = st.slider("Minimum Nuclear Size (pixels)", 100, 10000, 1000, 100,
                                            help="""Minimum area (in pixels) for detected nuclear regions:
• 100-500: Small nuclei or high magnification images
• 1000-3000: Typical nuclear sizes for most microscopy setups
• 3000-5000: Large nuclei or low magnification images
• 5000+: Very large nuclei or when multiple nuclei might be connected

Filters out small artifacts and noise while preserving genuine nuclear regions.""")
                closing_size = st.slider("Morphological Closing Size", 1, 20, 5, 1,
                                        help="""Size of structuring element for morphological closing:
• 1-3: Minimal gap filling, preserves fine details
• 4-7: Standard gap filling for typical nuclear boundaries
• 8-12: Aggressive gap filling for fragmented nuclei
• 13+: Very aggressive, may alter nuclear shape

Closes small gaps in nuclear boundaries while preserving overall shape.""")
            
            # Advanced boundary processing options
            with st.expander("Advanced Boundary Processing"):
                col1_adv, col2_adv = st.columns(2)
                
                with col1_adv:
                    smooth_boundary = st.checkbox("Smooth Boundaries", value=True,
                                                help="""Apply morphological smoothing to nuclear boundaries:
• Checked: Removes rough edges and small protrusions for cleaner nuclear outlines
• Unchecked: Preserves original boundary details but may include noise artifacts

Recommended for most applications to ensure clean density analysis regions.""")
                    if smooth_boundary:
                        smoothing_iterations = st.slider("Smoothing Iterations", 1, 5, 2, 1,
                                                        help="""Number of erosion-dilation cycles for boundary smoothing:
• 1: Minimal smoothing, preserves most boundary details
• 2-3: Standard smoothing for typical nuclear boundaries
• 4-5: Heavy smoothing for very rough or noisy boundaries

More iterations create smoother boundaries but may alter nuclear shape.""")
                    else:
                        smoothing_iterations = 2
                
                with col2_adv:
                    largest_only = st.checkbox("Select Largest Object Only", value=False,
                                             help="""Controls object selection after nuclear detection:
• Checked: Keeps only the largest detected region (useful for single-nucleus analysis)
• Unchecked: Keeps all regions above minimum size threshold

Use when image contains one main nucleus to exclude small artifacts or secondary objects.""")
                    if largest_only:
                        st.info("Will select only the largest connected component")
            
            # Nuclear segmentation
            if st.button("Detect Nuclear Boundaries", key="detect_nucleus"):
                with st.spinner("Detecting nuclear boundaries..."):
                    try:
                        # Perform nuclear segmentation using enhanced_threshold_image
                        if threshold_method == "Manual":
                            nucleus_mask, info = enhanced_threshold_image(
                                current_image, 
                                method='manual', 
                                manual_threshold=manual_threshold,
                                min_size=min_nuclear_size,
                                closing_disk_size=closing_size,
                                smooth_boundary=smooth_boundary,
                                smoothing_iterations=smoothing_iterations,
                                largest_object_only=largest_only
                            )
                        else:
                            nucleus_mask, info = enhanced_threshold_image(
                                current_image, 
                                method=threshold_method.lower(),
                                min_size=min_nuclear_size,
                                closing_disk_size=closing_size,
                                smooth_boundary=smooth_boundary,
                                smoothing_iterations=smoothing_iterations,
                                largest_object_only=largest_only
                            )
                        
                        st.session_state.nucleus_mask = nucleus_mask
                        st.session_state.nucleus_threshold_info = info
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Image**")
                            st.image(display_image, caption="Original Image", use_container_width=True)
                        
                        with col2:
                            st.write("**Nuclear Mask**")
                            mask_display = (nucleus_mask * 255).astype(np.uint8)
                            st.image(mask_display, caption="Detected Nuclear Region", use_container_width=True)
                        
                        # Nucleus statistics
                        nucleus_area = np.sum(nucleus_mask)
                        total_area = nucleus_mask.size
                        nuclear_fraction = nucleus_area / total_area
                        
                        st.success(f"Nuclear region detected: {nucleus_area:,} pixels ({nuclear_fraction:.1%} of image)")
                        
                        # Display processing information
                        col1_info, col2_info, col3_info = st.columns(3)
                        with col1_info:
                            st.metric("Threshold Value", f"{info.get('threshold_value', 'N/A'):.3f}" if isinstance(info.get('threshold_value'), (int, float)) else "N/A")
                        with col2_info:
                            st.metric("Final Objects", info.get('final_objects', 'N/A'))
                        with col3_info:
                            processing_status = []
                            if info.get('boundary_smoothed', False):
                                processing_status.append(f"Smoothed ({info.get('smoothing_iterations', 0)}x)")
                            if info.get('largest_object_selected', False):
                                processing_status.append("Largest only")
                            st.metric("Processing", ", ".join(processing_status) if processing_status else "Basic")
                        
                    except Exception as e:
                        st.error(f"Error in nuclear detection: {str(e)}")
            
            # Density classification section
            if 'nucleus_mask' in st.session_state and st.session_state.nucleus_mask is not None:
                st.markdown("### Step 2: Nuclear Density Classification")
                
                # Segmentation method selection
                segmentation_method = st.selectbox(
                    "Density Classification Method",
                    ["Manual Classes", "Gaussian Mixture Model (Auto)", "Bayesian GMM (Auto)", "Compare Methods"],
                    help="Choose how to determine nuclear density classes"
                )
                
                # Initialize default values
                n_classes = 5
                sigma_density = 3.0
                binning_method = 'equal'
                include_background = True
                max_components = 8
                criterion = 'bic'
                weight_prior = 0.01
                random_seed = 42
                covariance_type = 'full'
                
                if segmentation_method == "Manual Classes":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_classes = st.slider("Number of Density Classes", 2, 10, 5, 1,
                                            help="""Number of nuclear density classes to create:
• 2-3: Basic classification (e.g., heterochromatin vs euchromatin)
• 4-5: Standard nuclear organization analysis, captures major density domains
• 6-7: Detailed classification for complex nuclear structures
• 8-10: Very fine-grained analysis, may include noise artifacts

Most biological studies use 3-5 classes for interpretable results.""")
                        sigma_density = st.slider("Density Smoothing Sigma", 0.5, 10.0, 3.0, 0.1,
                                                 help="""Gaussian smoothing applied before density classification:
• 0.5-1.5: Minimal smoothing, preserves fine details but may include noise
• 2.0-4.0: Balanced smoothing, good for most nuclear density analysis
• 5.0-7.0: Heavy smoothing, creates broader density regions
• 8.0-10.0: Very heavy smoothing, may lose important structural details

Higher values create smoother, more continuous density regions but may blur boundaries.""")
                    
                    with col2:
                        binning_method = st.selectbox("Binning Method", ['equal', 'quantile'],
                                                    help="""Method for dividing nuclear density into classes:
• Equal: Divides intensity range into equal-sized bins. Good when density is uniformly distributed.
• Quantile: Each class contains equal number of pixels. Better for handling outliers and skewed distributions.

Quantile binning is generally recommended for nuclear density analysis as it ensures balanced representation of all density levels.""")
                        
                        include_background = st.checkbox("Show Background as Class 0", value=True,
                                                       help="""Controls background pixel labeling:
• Checked: Pixels outside nucleus are labeled as class 0 (background), nuclear classes start from 1
• Unchecked: Only nuclear pixels are classified, background pixels remain unlabeled

Recommended to keep checked for clear distinction between nuclear and background regions.""")
                        
                elif segmentation_method in ["Gaussian Mixture Model (Auto)", "Bayesian GMM (Auto)"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        max_components = st.slider("Maximum Components to Test", 2, 15, 8, 1,
                                                 help="""Maximum number of density classes to test during automatic selection:
• 2-5: Conservative, good for simple nuclear organization
• 6-8: Balanced, works well for most cell types
• 9-12: Detailed classification for complex nuclear structures
• 13-15: Very fine-grained, may overfit to noise
Higher values increase computation time but may capture more subtle density variations.""")
                        if segmentation_method == "Gaussian Mixture Model (Auto)":
                            criterion = st.selectbox("Model Selection Criterion", ['bic', 'aic'], 
                                                    help="""Statistical criteria for selecting optimal number of density classes:
• BIC (Bayesian Information Criterion): Penalizes complex models more heavily, tends to select fewer classes. Best for avoiding overfitting.
• AIC (Akaike Information Criterion): Less conservative penalty, may select more classes. Better for capturing subtle variations.

BIC is generally recommended for nuclear density analysis as it provides more stable, interpretable results.""")
                        else:
                            weight_prior = st.slider("Weight Concentration Prior", 0.001, 0.1, 0.01, 0.001,
                                                    help="""Controls automatic component pruning in Bayesian GMM:
• 0.001-0.005: Very conservative, strongly favors fewer density classes. Good for avoiding over-segmentation.
• 0.005-0.02: Balanced approach, automatically selects appropriate number of classes for most nuclear structures.
• 0.02-0.05: Less conservative, may detect more subtle density variations.
• 0.05-0.1: Liberal, may create many small classes. Use only for very complex nuclear organization.

Lower values are generally recommended as they produce more stable, biologically meaningful segmentations.""")
                    
                    with col2:
                        random_seed = st.number_input("Random Seed", value=42, min_value=0, 
                                                    help="""Random number generator seed for reproducible results:
• Same seed + same parameters = identical segmentation results
• Different seeds may produce slightly different class assignments
• Use the same seed (e.g., 42) for comparing methods or parameters
• Change seed if results seem unstable to test robustness

Set to any integer value; 42 is commonly used as a default.""")
                        covariance_type = st.selectbox("Covariance Type", ['full', 'tied', 'diag', 'spherical'],
                                                     help="""Controls how density classes model pixel intensity variations:
• Full: Each class has its own covariance matrix. Most flexible, best for complex nuclear patterns.
• Tied: All classes share the same covariance matrix. Good balance of flexibility and stability.
• Diag: Diagonal covariance matrices. Assumes independent pixel variations. Faster computation.
• Spherical: Circular/spherical classes. Most constrained, good for simple, round density regions.

'Full' is recommended for nuclear density analysis as it captures the complex shapes of chromatin domains.""")
                        
                elif segmentation_method == "Compare Methods":
                    max_components = st.slider("Maximum Components to Test", 2, 10, 6, 1,
                                             help="""Maximum number of density classes for method comparison:
• 2-4: Conservative range, good for comparing basic nuclear organization patterns
• 5-7: Balanced range, suitable for most nuclear density analysis comparisons
• 8-10: Detailed comparison, captures subtle differences between methods

Lower values provide faster comparison but may miss fine differences between methods.""")
                    st.info("This will run all methods and compare their results")
                
                if st.button("Generate Density Classes", key="generate_density_classes"):
                    with st.spinner("Generating nuclear density classes..."):
                        try:
                            # Ensure we have a valid nuclear mask
                            nuclear_mask = st.session_state.nucleus_mask
                            if nuclear_mask is None:
                                st.error("No nuclear mask found. Please run Step 1: Nuclear Boundary Detection first.")
                            else:
                                st.info(f"Performing density classification within {np.sum(nuclear_mask):,} nuclear pixels")
                                
                                if segmentation_method == "Manual Classes":
                                    # Original manual classification method
                                    density_results = density_map_threshold(
                                        current_image,
                                        gaussian_sigma_hp=2.0,  # Default value for high-pass filtering
                                        gaussian_sigma_density=sigma_density,
                                        disk_radius=10,  # Default disk radius
                                        pcutoff_in=0.1,  # These won't be used since we provide roi_mask
                                        pcutoff_out=0.1,
                                        roi_mask=nuclear_mask,
                                        num_classes=n_classes,
                                        binning=binning_method
                                    )
                                    method_info = f"Manual classification with {n_classes} classes using {binning_method} binning"
                                    
                                elif segmentation_method == "Gaussian Mixture Model (Auto)":
                                    # Automatic GMM classification
                                    density_results = gaussian_mixture_segmentation(
                                        current_image,
                                        roi_mask=nuclear_mask,
                                        max_components=max_components,
                                        criterion=criterion,
                                        random_state=int(random_seed),
                                        covariance_type=covariance_type
                                    )
                                    optimal_n = density_results['optimal_n_components']
                                    method_info = f"GMM auto-detected {optimal_n} optimal classes using {criterion.upper()} criterion"
                                    
                                elif segmentation_method == "Bayesian GMM (Auto)":
                                    # Bayesian GMM classification with automatic component pruning
                                    density_results = bayesian_gaussian_mixture_segmentation(
                                        current_image,
                                        roi_mask=nuclear_mask,
                                        max_components=max_components,
                                        weight_concentration_prior=weight_prior,
                                        random_state=int(random_seed),
                                        covariance_type=covariance_type
                                    )
                                    effective_n = density_results['n_effective_components']
                                    method_info = f"Bayesian GMM found {effective_n} effective classes from {max_components} tested"
                                    
                                elif segmentation_method == "Compare Methods":
                                    # Compare all methods
                                    comparison_results = compare_segmentation_methods(
                                        current_image,
                                        roi_mask=nuclear_mask,
                                        max_components=max_components
                                    )
                                    
                                    # Use the GMM result as primary if available
                                    if 'gmm_bic' in comparison_results and 'error' not in comparison_results['gmm_bic']:
                                        density_results = comparison_results['gmm_bic']
                                        optimal_n = density_results['optimal_n_components']
                                        method_info = f"Method comparison: GMM selected {optimal_n} classes"
                                    elif 'bayesian_gmm' in comparison_results and 'error' not in comparison_results['bayesian_gmm']:
                                        density_results = comparison_results['bayesian_gmm']
                                        effective_n = density_results['n_effective_components']
                                        method_info = f"Method comparison: Bayesian GMM selected {effective_n} classes"
                                    else:
                                        # Fallback to equal binning
                                        density_results = comparison_results.get('equal_binning', {})
                                        if 'error' not in density_results:
                                            method_info = f"Method comparison: Using equal binning with {density_results.get('n_classes', 5)} classes"
                                        else:
                                            raise ValueError("All methods failed in comparison")
                                    
                                    # Store comparison for display
                                    st.session_state['method_comparison'] = comparison_results
                                
                                st.session_state['density_classes'] = density_results['classes']
                                st.session_state['density_results'] = density_results
                                st.session_state['segmentation_method_used'] = segmentation_method
                                
                                # Success message with method info
                                st.success(f"✓ {method_info}")
                                
                                # Analyze class distribution
                                if segmentation_method == "Manual Classes":
                                    class_analysis = analyze_density_segmentation(
                                        st.session_state.nucleus_mask, 
                                        density_results['classes']
                                    )
                                else:
                                    # For GMM methods, we already have component statistics
                                    class_analysis = {
                                        'num_objects': 1,  # Nuclear region
                                        'total_area': np.sum(st.session_state.nucleus_mask),
                                        'component_stats': density_results.get('component_stats', [])
                                    }
                                
                                # Display density map
                                st.write("**Nuclear Density Classification Map**")
                                
                                # Create colored density map
                                classes = density_results['classes']
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                # Determine number of classes from segmentation results
                                if segmentation_method == "Bayesian GMM (Auto)" and 'n_effective_components' in density_results:
                                    # Use actual number of effective components for Bayesian GMM
                                    max_class = density_results['n_effective_components']
                                elif segmentation_method == "Gaussian Mixture Model (Auto)" and 'optimal_n_components' in density_results:
                                    # Use optimal components for standard GMM
                                    max_class = density_results['optimal_n_components']
                                else:
                                    # Fallback to maximum class value for other methods
                                    max_class = int(classes.max()) if classes.max() > 0 else 1
                                
                                # Use viridis colormap, setting background (class 0) to black
                                cmap = plt.cm.viridis.copy()
                                cmap.set_under('black')  # Background color
                                
                                im = ax.imshow(classes, cmap=cmap, vmin=0.5, vmax=max_class)
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('Density Class')
                                ax.set_title(f'Nuclear Density Classification ({max_class} classes + background)')
                                ax.axis('off')
                                
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Display side-by-side comparison for Compare Methods
                                if segmentation_method == "Compare Methods" and 'method_comparison' in st.session_state:
                                    st.markdown("### Method Comparison: Side-by-Side Density Maps")
                                    
                                    comparison_results = st.session_state['method_comparison']
                                    
                                    # Create a figure with subplots for each method
                                    methods_to_show = []
                                    method_names = []
                                    
                                    # Check which methods succeeded
                                    if 'gmm_bic' in comparison_results and 'error' not in comparison_results['gmm_bic']:
                                        methods_to_show.append(comparison_results['gmm_bic'])
                                        method_names.append("GMM (BIC)")
                                    
                                    if 'gmm_aic' in comparison_results and 'error' not in comparison_results['gmm_aic']:
                                        methods_to_show.append(comparison_results['gmm_aic'])
                                        method_names.append("GMM (AIC)")
                                    
                                    if 'bayesian_gmm' in comparison_results and 'error' not in comparison_results['bayesian_gmm']:
                                        methods_to_show.append(comparison_results['bayesian_gmm'])
                                        method_names.append("Bayesian GMM")
                                    
                                    if 'equal_binning' in comparison_results and 'error' not in comparison_results['equal_binning']:
                                        methods_to_show.append(comparison_results['equal_binning'])
                                        method_names.append("Equal Binning")
                                    
                                    if 'quantile_binning' in comparison_results and 'error' not in comparison_results['quantile_binning']:
                                        methods_to_show.append(comparison_results['quantile_binning'])
                                        method_names.append("Quantile Binning")
                                    
                                    if methods_to_show:
                                        # Calculate grid layout
                                        n_methods = len(methods_to_show)
                                        cols = min(3, n_methods)  # Max 3 columns
                                        rows = (n_methods + cols - 1) // cols  # Ceiling division
                                        
                                        fig_comp, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                                        if n_methods == 1:
                                            axes = [axes]
                                        elif rows == 1:
                                            axes = axes.reshape(1, -1)
                                        
                                        # Flatten axes for easier indexing
                                        axes_flat = axes.flatten() if n_methods > 1 else axes
                                        
                                        # Plot each method
                                        for i, (method_result, method_name) in enumerate(zip(methods_to_show, method_names)):
                                            ax = axes_flat[i]
                                            method_classes = method_result['classes']
                                            
                                            # Determine colormap range for this method
                                            if 'n_effective_components' in method_result:
                                                max_class = method_result['n_effective_components']
                                            elif 'optimal_n_components' in method_result:
                                                max_class = method_result['optimal_n_components']
                                            elif 'n_classes' in method_result:
                                                max_class = method_result['n_classes']
                                            else:
                                                max_class = int(method_classes.max()) if method_classes.max() > 0 else 1
                                            
                                            # Use consistent colormap
                                            cmap = plt.cm.viridis.copy()
                                            cmap.set_under('black')
                                            
                                            im = ax.imshow(method_classes, cmap=cmap, vmin=0.5, vmax=max_class)
                                            ax.set_title(f'{method_name}\n({max_class} classes)')
                                            ax.axis('off')
                                            
                                            # Add colorbar
                                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                        
                                        # Hide unused subplots
                                        for i in range(n_methods, len(axes_flat)):
                                            axes_flat[i].set_visible(False)
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig_comp)
                                        plt.close(fig_comp)
                                        
                                        # Method comparison summary table
                                        st.markdown("#### Method Performance Summary")
                                        
                                        summary_data = []
                                        for method_result, method_name in zip(methods_to_show, method_names):
                                            summary_row = {
                                                'Method': method_name,
                                                'Classes': method_result.get('optimal_n_components', 
                                                         method_result.get('n_effective_components', 
                                                         method_result.get('n_classes', 'N/A')))
                                            }
                                            
                                            # Add method-specific metrics
                                            if 'model_info' in method_result:
                                                model_info = method_result['model_info']
                                                if 'bic' in model_info:
                                                    summary_row['BIC Score'] = f"{model_info['bic']:.2f}"
                                                if 'aic' in model_info:
                                                    summary_row['AIC Score'] = f"{model_info['aic']:.2f}"
                                                if 'lower_bound' in model_info:
                                                    summary_row['Log Likelihood'] = f"{model_info['lower_bound']:.2f}"
                                            
                                            summary_data.append(summary_row)
                                        
                                        if summary_data:
                                            summary_df = pd.DataFrame(summary_data)
                                            st.dataframe(summary_df, use_container_width=True)
                                
                                # Display histogram with fitted components for GMM methods
                                if segmentation_method in ["Gaussian Mixture Model (Auto)", "Bayesian GMM (Auto)"]:
                                    if 'roi_pixels' in density_results and 'optimal_model' in density_results:
                                        st.markdown("### Intensity Histogram with Fitted Components")
                                        
                                        # Create histogram plot
                                        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                                        
                                        # Plot histogram of ROI pixels
                                        roi_pixels = density_results['roi_pixels']
                                        n_bins = min(50, len(np.unique(roi_pixels)))
                                        counts, bins, patches = ax_hist.hist(roi_pixels, bins=n_bins, alpha=0.7, 
                                                                            density=True, color='lightblue', 
                                                                            edgecolor='black', label='Data')
                                        
                                        # Plot fitted GMM components
                                        model = density_results['optimal_model']
                                        x_range = np.linspace(roi_pixels.min(), roi_pixels.max(), 300)
                                        
                                        # Overall fitted distribution
                                        total_pdf = np.zeros_like(x_range)
                                        
                                        # Individual components
                                        colors = plt.cm.Set3(np.linspace(0, 1, len(model.means_)))
                                        for i, (mean, cov, weight) in enumerate(zip(model.means_.flatten(), 
                                                                                   model.covariances_.flatten(), 
                                                                                   model.weights_)):
                                            component_pdf = weight * (1/np.sqrt(2*np.pi*cov)) * np.exp(-0.5*(x_range - mean)**2/cov)
                                            total_pdf += component_pdf
                                            
                                            ax_hist.plot(x_range, component_pdf, 
                                                       color=colors[i], linewidth=2, linestyle='--',
                                                       label=f'Component {i+1} (μ={mean:.1f}, σ={np.sqrt(cov):.1f})')
                                        
                                        # Plot total fitted distribution
                                        ax_hist.plot(x_range, total_pdf, 'r-', linewidth=3, label='Total Fit')
                                        
                                        ax_hist.set_xlabel('Pixel Intensity')
                                        ax_hist.set_ylabel('Density')
                                        ax_hist.set_title(f'GMM Fit: {len(model.means_)} Components')
                                        ax_hist.legend()
                                        ax_hist.grid(True, alpha=0.3)
                                        
                                        st.pyplot(fig_hist)
                                        plt.close(fig_hist)
                                
                                # Display class statistics
                                st.markdown("### Density Class Statistics")
                                
                                # Display GMM component statistics for automatic methods
                                if segmentation_method in ["Gaussian Mixture Model (Auto)", "Bayesian GMM (Auto)"]:
                                    if 'component_stats' in density_results and density_results['component_stats']:
                                        st.markdown("#### Model Selection and Component Analysis")
                                        
                                        # Model selection info
                                        if 'model_info' in density_results:
                                            model_info = density_results['model_info']
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Optimal Components", model_info.get('n_components', 'N/A'))
                                            with col2:
                                                if 'bic' in model_info:
                                                    st.metric("BIC Score", f"{model_info['bic']:.2f}")
                                                elif 'lower_bound' in model_info:
                                                    st.metric("Log Likelihood", f"{model_info['lower_bound']:.2f}")
                                            with col3:
                                                if 'aic' in model_info:
                                                    st.metric("AIC Score", f"{model_info['aic']:.2f}")
                                                elif 'n_components_used' in model_info:
                                                    st.metric("Components Used", model_info['n_components_used'])
                                        
                                        # Component statistics table
                                        st.markdown("#### Component Characteristics")
                                        comp_stats = density_results['component_stats']
                                        
                                        if comp_stats:
                                            # Create formatted dataframe with proper column names
                                            formatted_stats = []
                                            total_pixels = sum(stat['pixel_count'] for stat in comp_stats)
                                            
                                            for stat in comp_stats:
                                                area_fraction = (stat['pixel_count'] / total_pixels * 100) if total_pixels > 0 else 0
                                                formatted_stats.append({
                                                    'Component': stat['component'],
                                                    'Weight': f"{stat['weight']:.3f}",
                                                    'Mean Intensity': f"{stat['mean_intensity']:.2f}",
                                                    'Std Intensity': f"{stat['std_intensity']:.2f}",
                                                    'Pixel Count': f"{stat['pixel_count']:,}",
                                                    'Area Fraction': f"{area_fraction:.1f}%"
                                                })
                                            
                                            comp_df = pd.DataFrame(formatted_stats)
                                            st.dataframe(comp_df, use_container_width=True)
                                
                                # Log Bayesian GMM classification info
                                if segmentation_method == "Bayesian GMM (Auto)":
                                    unique_classes = np.unique(classes)
                                    logger.debug(f"Bayesian GMM unique classes: {unique_classes}")
                                    logger.debug(f"Bayesian GMM class counts: {[(c, np.sum(classes == c)) for c in unique_classes]}")
                                    # if 'component_to_brightness_order' in locals():
                                    #     logger.debug(f"Component mapping: {component_to_brightness_order}")
                                
                                # Display class distribution
                                st.markdown("#### Class Distribution Summary")
                                
                                # Create unified class statistics
                                class_stats = []
                                total_pixels = classes.size
                                background_pixels = np.sum(classes == 0)
                                nuclear_pixels = np.sum(classes > 0)
                                
                                # Background class (always present for GMM methods when using ROI)
                                if segmentation_method in ["Gaussian Mixture Model (Auto)", "Bayesian GMM (Auto)"]:
                                    has_background = density_results.get('has_background', True)
                                else:
                                    has_background = include_background
                                
                                if has_background and background_pixels > 0:
                                    class_stats.append({
                                        'Class': 0,
                                        'Description': 'Background (outside nucleus)',
                                        'Pixels': background_pixels,
                                        'Percentage': f"{background_pixels / total_pixels * 100:.1f}%"
                                    })
                                
                                # Nuclear density classes
                                if segmentation_method in ["Gaussian Mixture Model (Auto)", "Bayesian GMM (Auto)"]:
                                    # For GMM methods, use component stats
                                    if 'component_stats' in density_results and density_results['component_stats']:
                                        for stat in density_results['component_stats']:
                                            class_stats.append({
                                                'Class': stat['component'],
                                                'Description': f'Nuclear density class {stat["component"]} (μ={stat["mean_intensity"]:.1f})',
                                                'Pixels': stat['pixel_count'],
                                                'Percentage': f"{stat['pixel_count'] / nuclear_pixels * 100:.1f}%" if nuclear_pixels > 0 else "0%"
                                            })
                                else:
                                    # For manual method, use class distribution
                                    if 'class_distribution' in class_analysis:
                                        class_dist = class_analysis['class_distribution']
                                        for class_id, pixel_count in class_dist.items():
                                            percentage = pixel_count / nuclear_pixels * 100 if nuclear_pixels > 0 else 0
                                            class_stats.append({
                                                'Class': int(class_id),
                                                'Description': f'Nuclear density class {class_id}',
                                                'Pixels': pixel_count,
                                                'Percentage': f"{percentage:.1f}%"
                                            })
                                
                                if class_stats:
                                    # Sort by class number for consistent display
                                    class_stats.sort(key=lambda x: x['Class'])
                                    class_df = pd.DataFrame(class_stats)
                                    st.dataframe(class_df, use_container_width=True)
                                
                                # Summary metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    total_nuclear = class_analysis.get('total_area', np.sum(st.session_state.nucleus_mask))
                                    st.metric("Total Nuclear Pixels", f"{total_nuclear:,}")
                                with col2:
                                    bg_pixels = class_analysis.get('background_pixels', 0)
                                    st.metric("Background Pixels", f"{bg_pixels:,}")
                                with col3:
                                    actual_classes = max_class if segmentation_method != "Manual Classes" else n_classes
                                    st.metric("Density Classes", actual_classes)
                                
                                # Display method comparison results if available
                                if 'method_comparison' in st.session_state and st.session_state['method_comparison']:
                                    st.markdown("### Method Comparison Results")
                                    
                                    comparison = st.session_state['method_comparison']
                                    
                                    # Summary table
                                    comp_summary = []
                                    for method, results in comparison.items():
                                        if 'model_info' in results:
                                            model_info = results['model_info']
                                            comp_summary.append({
                                                'Method': method,
                                                'Components': model_info.get('n_components', 'N/A'),
                                                'Score': model_info.get('bic', model_info.get('lower_bound', 'N/A')),
                                                'Score Type': 'BIC' if 'bic' in model_info else 'Log Likelihood'
                                            })
                                    
                                    if comp_summary:
                                        comp_df = pd.DataFrame(comp_summary)
                                        st.dataframe(comp_df, use_container_width=True)
                                        
                                        # Recommendation
                                        if len(comp_summary) >= 2:
                                            st.markdown("**Recommendation:** The method with the lowest BIC score or highest log likelihood typically provides the best balance between model fit and complexity.")
                                
                                st.success("Density classification completed!")
                            
                        except Exception as e:
                            st.error(f"Error in density classification: {str(e)}")
            
            # Integration with particle tracking
            if 'density_classes' in st.session_state and st.session_state.density_classes is not None and st.session_state.tracks_data is not None:
                st.markdown("### Step 3: Integrate with Particle Tracking")
                
                if st.button("Assign Density Classes to Tracks", key="assign_density"):
                    with st.spinner("Assigning density classes to particle tracks..."):
                        try:
                            tracks_df = st.session_state.tracks_data.copy()
                            classes = st.session_state.density_classes
                            
                            # Assign density class to each track position
                            density_assignments = []
                            
                            # Get current pixel size for coordinate conversion
                            pixel_size = st.session_state.get('global_pixel_size', st.session_state.get('current_pixel_size', 0.1))
                            track_coords_in_microns = st.session_state.get('track_coordinates_in_microns', False)
                            
                            for _, row in tracks_df.iterrows():
                                # Convert coordinates to pixel indices for class lookup
                                if track_coords_in_microns:
                                    # Track coordinates are in microns - convert to pixels for class indexing
                                    x_coord = int(row['x'] / pixel_size)
                                    y_coord = int(row['y'] / pixel_size)
                                else:
                                    # Track coordinates are already in pixels
                                    x_coord = int(row['x'])
                                    y_coord = int(row['y'])
                                
                                # Check bounds and assign class
                                if 0 <= x_coord < classes.shape[1] and 0 <= y_coord < classes.shape[0]:
                                    density_class = classes[y_coord, x_coord]
                                else:
                                    density_class = 0  # Outside image bounds = background
                                
                                density_assignments.append(density_class)
                            
                            tracks_df['nuclear_density_class'] = density_assignments
                            st.session_state.tracks_data = tracks_df
                            
                            # Display summary
                            class_summary = tracks_df['nuclear_density_class'].value_counts().sort_index()
                            
                            st.success("Density classes assigned to tracks!")
                            st.write("**Track Distribution by Density Class:**")
                            
                            summary_df = pd.DataFrame({
                                'Density Class': class_summary.index,
                                'Number of Tracks': class_summary.values,
                                'Percentage': (class_summary.values / len(tracks_df) * 100).round(1)
                            })
                            
                            st.dataframe(summary_df, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error assigning density classes: {str(e)}")
    
    # Tab 3: Advanced Segmentation
    with img_tabs[2]:
        st.subheader("Advanced Segmentation Methods")
        
        if 'mask_images' not in st.session_state or st.session_state.mask_images is None:
            st.warning("Upload mask images in the Data Loading tab first to use advanced segmentation.")
            st.info("Go to Data Loading → 'Images for Mask Generation' to upload images for processing.")
        else:
            if ADVANCED_SEGMENTATION_AVAILABLE:
                # Call the integration function from advanced_segmentation module
                integrate_advanced_segmentation_with_app()
            else:
                st.error("Advanced segmentation module not available.")
                st.info("Please install required packages to use advanced segmentation features.")
                
                with st.expander("Installation Instructions"):
                    st.markdown("""
                    ### Required packages:
                    
                    ```
                    pip install torch torchvision segment-anything cellpose opencv-python
                    ```
                    """)
    
    # Tab 4: Export Results
    with img_tabs[3]:
        st.subheader("Export Results")
        
        if 'nucleus_mask' in st.session_state and st.session_state.nucleus_mask is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Nuclear Mask", key="export_nucleus"):
                    # Convert mask to uint8
                    mask_export = (st.session_state.nucleus_mask * 255).astype(np.uint8)
                    
                    # Create download
                    from PIL import Image
                    import io
                    img_buffer = io.BytesIO()
                    Image.fromarray(mask_export).save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Nuclear Mask (PNG)",
                        data=img_buffer.getvalue(),
                        file_name="nuclear_mask.png",
                        mime="image/png"
                    )
            
            with col2:
                if 'density_classes' in st.session_state and st.session_state.density_classes is not None:
                    if st.button("Export Density Classes", key="export_density"):
                        # Export as CSV with coordinates and class assignments
                        height, width = st.session_state.density_classes.shape
                        
                        # Create coordinate grid
                        y_coords, x_coords = np.mgrid[0:height, 0:width]
                        
                        export_data = pd.DataFrame({
                            'x': x_coords.flatten(),
                            'y': y_coords.flatten(),
                            'density_class': st.session_state.density_classes.flatten()
                        })
                        
                        # Remove NaN values if any
                        export_data = export_data.dropna()
                        
                        csv_buffer = io.StringIO()
                        export_data.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download Density Classes (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="nuclear_density_classes.csv",
                            mime="text/csv"
                        )
        else:
            st.info("Process images first to enable export options.")

# Data Loading Page
elif st.session_state.active_page == "Data Loading":
    st.title("Data Loading")
    
    # Create tabs for different data loading options
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Image Settings", "Track Data", "Images for Tracking", "Images for Mask Generation", "Sample Data"])
    
    with tab1:
        st.header("Image Settings")
        st.info("Configure global parameters that will be used throughout the application for all analyses")
        
        st.markdown("### Global Image Parameters")
        st.markdown("These settings will be applied to all analyses in the application unless overridden.")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Spatial Calibration")
            # Initialize default values if not in session state
            if 'global_pixel_size' not in st.session_state:
                st.session_state.global_pixel_size = 0.1
            
            global_pixel_size = st.number_input(
                "Pixel Size (µm)", 
                min_value=0.001, 
                max_value=10.0, 
                value=st.session_state.global_pixel_size, 
                step=0.001,
                format="%.3f",
                help="Physical size of one pixel in micrometers. This value will be used for converting pixel coordinates to real distances in all analyses.",
                key="global_pixel_settings"
            )
            
        with col2:
            st.markdown("#### Temporal Calibration")
            # Initialize default values if not in session state
            if 'global_frame_interval' not in st.session_state:
                st.session_state.global_frame_interval = 0.1
            
            global_frame_interval = st.number_input(
                "Frame Interval (s)", 
                min_value=0.001, 
                max_value=3600.0, 
                value=st.session_state.global_frame_interval, 
                step=0.001,
                format="%.3f",
                help="Time between consecutive frames in seconds. This value will be used for calculating velocities and time-dependent properties in all analyses.",
                key="global_frame_settings"
            )
        
        # Store current values for use in analysis modules
        st.session_state.current_pixel_size = global_pixel_size
        st.session_state.current_frame_interval = global_frame_interval
        
        # Update legacy keys for compatibility with existing modules
        st.session_state.pixel_size = global_pixel_size
        st.session_state.frame_interval = global_frame_interval
        
        st.markdown("---")
        
        # Track coordinate units setting
        st.markdown("### Track Data Coordinate Units")
        
        # Initialize track coordinate units if not set
        if 'track_coordinates_in_microns' not in st.session_state:
            st.session_state.track_coordinates_in_microns = False
        
        track_coords_in_microns = st.checkbox(
            "Track data coordinates are already in microns",
            value=st.session_state.track_coordinates_in_microns,
            key="track_coords_units",
            help="Check this if your imported track data coordinates are already in micrometers rather than pixels. This affects coordinate conversion in analyses."
        )
        
        # Update session state
        st.session_state.track_coordinates_in_microns = track_coords_in_microns
        
        # Show explanation
        if track_coords_in_microns:
            st.info("✓ Track coordinates are in microns - only image pixel coordinates will be converted to microns for comparison")
        else:
            st.info("• Track coordinates are in pixels - both track and image coordinates will be converted to microns")
        
        st.markdown("---")
        
        # Display current settings summary
        st.markdown("### Current Settings Summary")
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            st.metric(
                label="Pixel Size", 
                value=f"{global_pixel_size:.3f} µm",
                help="Current pixel size setting that will be used in analyses"
            )
            
        with settings_col2:
            st.metric(
                label="Frame Interval", 
                value=f"{global_frame_interval:.3f} s",
                help="Current frame interval setting that will be used in analyses"
            )
        
        # Units information
        st.markdown("### Units Information")
        st.info("""
        **How these settings are used:**
        - **Pixel Size**: Converts pixel measurements to micrometers (µm) for spatial analyses
        - **Frame Interval**: Converts frame numbers to time (seconds) for temporal analyses
        - **Diffusion Analysis**: Uses both for calculating diffusion coefficients in µm²/s
        - **Velocity Analysis**: Uses both for calculating velocities in µm/s
        - **MSD Analysis**: Uses both for proper time and distance scaling
        """)
        
        # Reset to defaults button
        if st.button("Reset to Default Values", help="Reset pixel size to 0.1 µm and frame interval to 0.1 s"):
            st.session_state.global_pixel_size = 0.1
            st.session_state.global_frame_interval = 0.1
            st.session_state.pixel_size = 0.1
            st.session_state.frame_interval = 0.1
            st.success("Settings reset to default values")
            st.rerun()
    
    with tab2:
        st.header("Upload Track Data")
        st.info("Load pre-processed track data from other tracking software")
        
        track_file = st.file_uploader(
            "Upload track data file", 
            type=["csv", "xlsx", "h5", "json", "ims", "xml", "uic", "mvd2", "aisf", "aiix"],
            help="Upload your track data in CSV, Excel, HDF5, JSON, XML (TrackMate), Imaris (IMS), Volocity (UIC/AISF/AIIX), or MVD2 format"
        )
        
        if track_file is not None:
            # Validate file before processing
            file_validator = get_file_validator()
            validation_result = file_validator.validate_file(track_file, file_type='track_data', check_content=True)
            
            if not validation_result['valid']:
                st.error("⚠️ File validation failed!")
                for error in validation_result['errors']:
                    st.error(f"❌ {error}")
                for warning in validation_result['warnings']:
                    st.warning(f"⚠️ {warning}")
                st.info("Please check your file and try again.")
                st.stop()
            
            # Show warnings if any (but allow processing)
            if validation_result['warnings']:
                with st.expander("⚠️ File Validation Warnings", expanded=False):
                    for warning in validation_result['warnings']:
                        st.warning(warning)
            
            # Show file info
            st.success(f"✅ File validation passed: {validation_result['filename']} ({validation_result['file_size'] / 1024 / 1024:.1f} MB)")
            
            try:
                st.session_state.tracks_data = load_tracks_file(track_file)
                st.success(f"Track data loaded successfully: {track_file.name}")
                
                # Calculate track statistics
                with st.spinner("Calculating track statistics..."):
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(st.session_state.tracks_data.head())
                
                # Display basic statistics
                st.subheader("Basic Statistics")
                st.write(f"Number of tracks: {st.session_state.tracks_data['track_id'].nunique()}")
                st.write(f"Number of points: {len(st.session_state.tracks_data)}")
                st.write(f"Average track length: {len(st.session_state.tracks_data) / st.session_state.tracks_data['track_id'].nunique():.1f} points")
                
                # Run automatic data quality checks
                st.divider()
                st.subheader("🔍 Data Quality Assessment")
                
                # Add option to run quality check
                if st.button("Run Quality Check", type="primary", help="Assess the quality and reliability of loaded track data"):
                    with st.spinner("Running comprehensive quality checks..."):
                        quality_checker = DataQualityChecker()
                        
                        # Get current parameters
                        pixel_size = st.session_state.get('current_pixel_size', st.session_state.get('pixel_size', 0.1))
                        frame_interval = st.session_state.get('current_frame_interval', st.session_state.get('frame_interval', 0.1))
                        
                        # Run quality assessment
                        quality_report = quality_checker.run_all_checks(
                            st.session_state.tracks_data,
                            pixel_size=pixel_size,
                            frame_interval=frame_interval
                        )
                        
                        # Store report in session state
                        st.session_state.quality_report = quality_report
                        
                        # Display quality score
                        score = quality_report.overall_score
                        if score >= 80:
                            score_color = "green"
                            score_emoji = "✅"
                        elif score >= 60:
                            score_color = "orange"
                            score_emoji = "⚠️"
                        else:
                            score_color = "red"
                            score_emoji = "❌"
                        
                        st.markdown(f"### {score_emoji} Overall Quality Score: :{score_color}[{score:.1f}/100]")
                        
                        # Display quality summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Passed Checks", quality_report.passed_checks)
                        with col2:
                            st.metric("Warnings", quality_report.warnings)
                        with col3:
                            st.metric("Failed Checks", quality_report.failed_checks)
                        
                        # Display checks by category
                        st.divider()
                        st.markdown("#### Quality Check Details")
                        
                        # Group checks by category
                        critical_checks = [c for c in quality_report.checks if c.category == 'critical']
                        warning_checks = [c for c in quality_report.checks if c.category == 'warning']
                        info_checks = [c for c in quality_report.checks if c.category == 'info']
                        
                        # Critical checks
                        if critical_checks:
                            with st.expander("🔴 Critical Issues", expanded=not all(c.passed for c in critical_checks)):
                                for check in critical_checks:
                                    icon = "✅" if check.passed else "❌"
                                    st.markdown(f"{icon} **{check.check_name}**: {check.message} (Score: {check.score:.0f}/100)")
                        
                        # Warning checks
                        if warning_checks:
                            with st.expander("🟡 Warnings", expanded=False):
                                for check in warning_checks:
                                    icon = "✅" if check.passed else "⚠️"
                                    st.markdown(f"{icon} **{check.check_name}**: {check.message} (Score: {check.score:.0f}/100)")
                        
                        # Info checks
                        if info_checks:
                            with st.expander("ℹ️ Information", expanded=False):
                                for check in info_checks:
                                    icon = "✅" if check.passed else "ℹ️"
                                    st.markdown(f"{icon} **{check.check_name}**: {check.message} (Score: {check.score:.0f}/100)")
                        
                        # Display recommendations
                        if quality_report.recommendations:
                            st.divider()
                            st.markdown("#### 💡 Recommendations")
                            for i, rec in enumerate(quality_report.recommendations, 1):
                                st.info(f"{i}. {rec}")
                        
                        # Display track statistics
                        if quality_report.track_statistics:
                            st.divider()
                            with st.expander("📊 Detailed Track Statistics"):
                                stats = quality_report.track_statistics
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                
                                with stat_col1:
                                    st.metric("Total Tracks", stats.get('n_tracks', 'N/A'))
                                    st.metric("Total Points", stats.get('n_points', 'N/A'))
                                    st.metric("Avg Track Length", f"{stats.get('mean_track_length', 0):.1f}")
                                
                                with stat_col2:
                                    st.metric("Median Track Length", f"{stats.get('median_track_length', 0):.1f}")
                                    st.metric("Frame Range", f"{stats.get('frame_range', [0, 0])[0]} - {stats.get('frame_range', [0, 0])[1]}")
                                    st.metric("Spatial Extent (X)", f"{stats.get('x_range', [0, 0])[1] - stats.get('x_range', [0, 0])[0]:.1f} µm")
                                
                                with stat_col3:
                                    st.metric("Spatial Extent (Y)", f"{stats.get('y_range', [0, 0])[1] - stats.get('y_range', [0, 0])[0]:.1f} µm")
                                    st.metric("Avg Displacement", f"{stats.get('mean_displacement', 0):.2f} µm/frame")
                                    st.metric("Avg Velocity", f"{stats.get('mean_velocity', 0):.2f} µm/s")
                
                # Show stored quality report if available
                elif 'quality_report' in st.session_state:
                    score = st.session_state.quality_report.overall_score
                    if score >= 80:
                        score_emoji = "✅"
                    elif score >= 60:
                        score_emoji = "⚠️"
                    else:
                        score_emoji = "❌"
                    
                    st.info(f"{score_emoji} Quality Score: {score:.1f}/100 - Click 'Run Quality Check' to see details")
                
                st.divider()
                
                # Display Imaris metadata if available
                if track_file.name.endswith('.ims') and 'imaris_metadata' in st.session_state and track_file.name in st.session_state.imaris_metadata:
                    with st.expander("Imaris File Information", expanded=True):
                        imaris_info = st.session_state.imaris_metadata[track_file.name]
                        
                        # Display thumbnail if available
                        if 'thumbnail' in imaris_info and imaris_info['thumbnail'] is not None:
                            thumb_col1, thumb_col2 = st.columns([1, 1])
                            with thumb_col1:
                                st.subheader("Image Preview")
                                # Display the thumbnail
                                thumbnail = imaris_info['thumbnail']
                                if thumbnail.ndim == 2:
                                    # Grayscale image
                                    st.image(thumbnail, caption="Image Preview", use_container_width=True)
                                elif thumbnail.ndim == 3 and thumbnail.shape[2] == 3:
                                    # RGB image
                                    st.image(thumbnail, caption="Image Preview", use_container_width=True)
                                else:
                                    st.write("Preview not available in correct format")
                        
                        # Display basic metadata
                        st.subheader("File Metadata")
                        metadata = imaris_info['metadata']
                        
                        # Extract relevant metadata fields
                        meta_col1, meta_col2 = st.columns(2)
                        
                        with meta_col1:
                            if 'X_Resolution' in metadata:
                                st.write(f"X Resolution: {metadata['X_Resolution']:.3f} µm/pixel")
                            if 'Y_Resolution' in metadata:
                                st.write(f"Y Resolution: {metadata['Y_Resolution']:.3f} µm/pixel")
                            if 'Z_Resolution' in metadata:
                                st.write(f"Z Resolution: {metadata['Z_Resolution']:.3f} µm/pixel")
                                
                        with meta_col2:
                            if 'Image_Name' in metadata:
                                st.write(f"Image Name: {metadata['Image_Name']}")
                            if 'DateTime' in metadata:
                                st.write(f"Date: {metadata['DateTime']}")
                            if 'Creator' in metadata:
                                st.write(f"Creator: {metadata['Creator']}")
                        
                        # Display image information
                        if imaris_info['image_info']:
                            st.subheader("Image Information")
                            img_info = imaris_info['image_info']
                            
                            img_col1, img_col2 = st.columns(2)
                            
                            with img_col1:
                                if 'channels' in img_info:
                                    st.write(f"Number of channels: {len(img_info['channels'])}")
                                    if len(img_info['channels']) > 0:
                                        st.write("Channel names:")
                                        for channel in img_info['channels']:
                                            st.write(f"- {channel['name']}")
                            
                            with img_col2:
                                if 'time_points' in img_info:
                                    st.write(f"Time points: {img_info['time_points']}")
                                if 'time_interval' in img_info:
                                    st.write(f"Time interval: {img_info['time_interval']:.3f} s")
                                if 'dimensions' in img_info and img_info['dimensions']:
                                    st.write(f"Dimensions: {' x '.join(map(str, img_info['dimensions']))}")
                
                # Navigation buttons
                st.button("Proceed to Analysis", on_click=navigate_to, args=("Analysis",))
                
            except Exception as e:
                st.error(f"Error loading track data: {str(e)}")
    

    with tab3:
        st.header("Upload Images for Tracking")
        st.info("Load microscopy images to perform particle detection and tracking in this application")
        
        # Display current global settings
        st.markdown("### Current Global Settings")
        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            st.metric("Pixel Size", f"{st.session_state.get('pixel_size', 0.1):.3f} µm")
        with settings_col2:
            st.metric("Frame Interval", f"{st.session_state.get('frame_interval', 0.1):.3f} s")
        
        st.info("These values from the Image Settings tab will be used for tracking analysis. To change them, go to the Image Settings tab.")
        
        tracking_image_file = st.file_uploader(
            "Upload microscopy images", 
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            help="Upload your microscopy images for particle detection and tracking. Supports single images or multi-frame TIFF files.",
            key="tracking_images_uploader"
        )
        
        if tracking_image_file is not None:
            try:
                st.session_state.image_data = load_image_file(tracking_image_file)
                st.success(f"Image loaded successfully: {tracking_image_file.name}")
                
                # Display image preview
                if isinstance(st.session_state.image_data, list) and len(st.session_state.image_data) > 0:
                    preview_image = st.session_state.image_data[0]
                else:
                    preview_image = st.session_state.image_data
                
                if preview_image is not None:
                    st.subheader("Image Preview")
                    if len(preview_image.shape) == 3 and preview_image.shape[2] > 1:
                        # Multichannel image
                        num_channels = preview_image.shape[2]
                        st.info(f"Multichannel image detected with {num_channels} channels")
                        st.image(normalize_image_for_display(preview_image[:, :, 0]), 
                                caption=f"Preview (Channel 1 of {num_channels})", 
                                use_container_width=True)
                    else:
                        st.image(normalize_image_for_display(preview_image), 
                                caption="Image for tracking analysis", 
                                use_container_width=True)
                    
                    st.write(f"Image dimensions: {preview_image.shape}")
                
                # Navigation button
                if st.button("Proceed to Tracking", key="proceed_tracking_btn"):
                    st.session_state.active_page = "Tracking"
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with tab4:
        st.header("Upload Images for Mask Generation")
        st.info("Load images to generate masks for boundary crossing and spatial analysis")
        
        mask_image_file = st.file_uploader(
            "Upload images for mask generation", 
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            help="Upload images to create masks for nuclear boundaries, ROIs, or other spatial features. These will be used in the Image Processing tab.",
            key="mask_image_uploader"
        )
        
        # Use global pixel size instead of separate mask pixel size
        st.subheader("Mask Image Parameters")
        st.info("Using global pixel size from Image Settings tab")
        
        # Display current global pixel size
        if 'pixel_size' in st.session_state:
            st.metric("Current Pixel Size", f"{st.session_state.pixel_size:.3f} µm")
        else:
            st.warning("Please set global pixel size in Image Settings tab first")
        
        if mask_image_file is not None:
            try:
                st.session_state.mask_images = load_image_file(mask_image_file)
                st.success(f"Mask images loaded successfully: {mask_image_file.name}")
                
                # Check if we loaded multiple frames that could be interpreted as channels
                if (isinstance(st.session_state.mask_images, list) and 
                    len(st.session_state.mask_images) > 1 and 
                    all(len(frame.shape) == 2 for frame in st.session_state.mask_images)):
                    
                    st.info(f"Loaded {len(st.session_state.mask_images)} frames - these could represent different channels or time points")
                    
                    # Channel selection for multichannel masks
                    channel_options = [f"Channel {i+1}" for i in range(len(st.session_state.mask_images))]
                    selected_channel = st.selectbox(
                        "Select channel for mask generation",
                        options=range(len(st.session_state.mask_images)),
                        format_func=lambda x: channel_options[x],
                        help="Choose which channel to use for mask generation"
                    )
                    
                    preview_image = st.session_state.mask_images[selected_channel]
                else:
                    preview_image = st.session_state.mask_images[0] if isinstance(st.session_state.mask_images, list) else st.session_state.mask_images
                
                # Display preview
                st.subheader("Image Preview")
                if len(preview_image.shape) == 3 and preview_image.shape[2] > 1:
                    # Multichannel image - show RGB composite
                    st.image(normalize_image_for_display(preview_image), caption="Mask Image Preview (RGB composite)", use_container_width=True)
                else:
                    # Single channel - use normalize_image_for_display function
                    st.image(normalize_image_for_display(preview_image), caption="Mask Image Preview", use_container_width=True)
                
                # Display basic statistics
                st.write(f"Number of frames: {len(st.session_state.mask_images) if isinstance(st.session_state.mask_images, list) else 1}")
                st.write(f"Image dimensions: {preview_image.shape}")
                if len(preview_image.shape) == 3 and preview_image.shape[2] > 1:
                    st.write(f"Number of channels: {preview_image.shape[2]}")
            
                # Navigation buttons
                if st.button("Proceed to Image Processing", key="proceed_to_image_processing"):
                    st.session_state.active_page = "Image Processing"
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error loading mask images: {str(e)}")
    
    with tab5:
        st.header("Sample Data")
        st.info("Load sample datasets to test the application features")
        
        st.subheader("Available Sample Datasets")
        
        sample_datasets = [
            "Sample Track Data (CSV)",
            "Sample Microscopy Images", 
            "Sample MD Simulation Data"
        ]
        
        selected_sample = st.selectbox("Choose a sample dataset", sample_datasets)
        
        if st.button("Load Sample Data", key="load_sample_data_tab5"):
            try:
                if selected_sample == "Sample Track Data (CSV)":
                    # Generate sample track data
                    from test_data_generator import generate_sample_tracks
                    st.session_state.tracks_data = generate_sample_tracks()
                    st.success("Sample track data loaded successfully!")
                    
                elif selected_sample == "Sample Microscopy Images":
                    # Generate sample image data
                    from test_data_generator import generate_sample_image
                    st.session_state.image_data = generate_sample_image()
                    st.success("Sample microscopy image loaded successfully!")
                    
                elif selected_sample == "Sample MD Simulation Data":
                    # Generate sample MD data
                    from test_data_generator import generate_sample_md_data
                    st.session_state.md_data = generate_sample_md_data()
                    st.success("Sample MD simulation data loaded successfully!")
                    
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

# Tracking page
elif st.session_state.active_page == "Tracking":
    st.title("Particle Detection and Tracking")
    
    # Check if image_data exists and is not None
    if 'image_data' not in st.session_state or st.session_state.image_data is None:
        st.warning("No image data loaded. Please upload images first.")
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
    else:
        # Create tabs for tracking workflow
        tab1, tab2, tab3 = st.tabs(["Particle Detection", "Particle Linking", "Track Results"])
        
        with tab1:
            st.header("Particle Detection")
            
            # Mask selection for region-specific detection
            mask_selection_ui = create_mask_selection_ui("tracking")
            selected_mask, selected_classes, segmentation_method = mask_selection_ui
            
            if selected_mask:
                st.info(f"Detection will be performed on mask: {selected_mask}")
                if selected_classes:
                    st.info(f"Using mask classes: {', '.join(map(str, selected_classes))}")
            
            st.divider()
            
            # If multichannel images, allow selecting channels for tracking
            try:
                _first_frame = st.session_state.image_data[0]
                _is_mc = isinstance(_first_frame, np.ndarray) and _first_frame.ndim == 3 and _first_frame.shape[2] > 1
                _num_ch = int(_first_frame.shape[2]) if _is_mc else 1
            except Exception:
                _is_mc, _num_ch = False, 1

            if _is_mc:
                st.subheader("Tracking Channel Selection")
                # Channel labeling for tracking
                track_label_key = "tracking_channel_labels"
                if track_label_key not in st.session_state:
                    st.session_state[track_label_key] = {}
                def _track_label(idx: int) -> str:
                    saved = st.session_state[track_label_key].get(idx)
                    return f"Channel {idx + 1}" if not saved else f"{saved} (C{idx + 1})"

                with st.expander("Name channels (optional)", expanded=False):
                    common = channel_manager.get_common_names()
                    cols = st.columns(min(4, _num_ch))
                    for i in range(_num_ch):
                        with cols[i % len(cols)]:
                            current = st.session_state[track_label_key].get(i, "")
                            new_name = st.text_input(
                                f"Channel {i+1} name",
                                value=current,
                                key=f"track_label_input_{i}",
                                placeholder="e.g., DAPI, GFP"
                            )
                            if new_name and new_name != current:
                                is_valid, validated = channel_manager.validate_name(new_name)
                                if is_valid:
                                    st.session_state[track_label_key][i] = validated
                                    channel_manager.add_channel_name(validated)
                                else:
                                    st.warning(f"Invalid name for C{i+1}: {validated}")
                default_track_sel = st.session_state.get("tracking_channels", [0])
                tracking_channels = st.multiselect(
                    "Choose channel(s) for tracking:",
                    options=list(range(_num_ch)),
                    default=[ci for ci in default_track_sel if 0 <= ci < _num_ch] or [0],
                    format_func=lambda x: _track_label(x),
                    key="tracking_channels_select",
                    help="Select one or two channels to detect particles on"
                )
                tracking_fusion_mode = st.selectbox(
                    "Fusion mode",
                    ["average", "max", "min", "sum", "weighted"],
                    index=["average", "max", "min", "sum", "weighted"].index(
                        st.session_state.get("tracking_fusion_mode", "average")
                    ),
                    help="How to combine the selected channels into a single detection image"
                )
                tracking_weights = None
                if tracking_fusion_mode == "weighted" and len(tracking_channels) > 1:
                    tw_text = st.text_input(
                        "Weights (comma-separated)",
                        value=st.session_state.get("tracking_weights", ",".join(["1"] * len(tracking_channels))),
                        help="Provide one weight per selected channel, e.g., 1,1 or 1,2"
                    )
                    try:
                        tracking_weights = [float(x.strip()) for x in tw_text.split(",") if x.strip() != ""]
                    except Exception:
                        tracking_weights = None
                # persist
                st.session_state.tracking_channels = tracking_channels or [0]
                st.session_state.tracking_fusion_mode = tracking_fusion_mode
                if tracking_weights is not None:
                    st.session_state.tracking_weights = ",".join(map(str, tracking_weights))
                sel_label = ", ".join([f"C{c+1}" for c in (tracking_channels or [0])])
                st.info(f"Tracking will use {sel_label} with '{tracking_fusion_mode}' fusion.")

            # Detection parameters with real-time tuning
            st.subheader("Detection Parameters")

            # Optional ROI restriction using segmentation masks
            with st.expander("Restrict detection to segmentation classes (optional)", expanded=False):
                roi_enabled = st.checkbox("Enable ROI restriction", value=False, key="tracking_roi_enabled")
                roi_mask = None
                if roi_enabled:
                    # Reuse segmentation mask selection utility
                    available_masks = get_available_masks()
                    if not available_masks:
                        st.info("No masks available. Generate masks in Image Processing.")
                    else:
                        mask_names = list(available_masks.keys())
                        sel_mask_name = st.selectbox("Mask", mask_names, index=0, key="tracking_roi_mask_name")
                        sel_mask = available_masks[sel_mask_name]
                        unique_classes = sorted(list(np.unique(sel_mask)))
                        sel_classes = st.multiselect(
                            "Classes to include",
                            options=unique_classes,
                            default=[c for c in unique_classes if c != 0],
                            key="tracking_roi_classes",
                            help="Only pixels in these classes will be considered during detection"
                        )
                        # Build boolean ROI from chosen classes
                        roi_mask = np.isin(sel_mask, sel_classes)
                        st.caption(f"ROI built from mask '{sel_mask_name}' with classes {sel_classes}")
            
            # Add real-time detection tuning section
            with st.expander("🔍 Real-time Detection Tuning", expanded=False):
                st.write("**Preview detection settings on a test frame before running full detection**")
                
                # Frame selection for testing
                num_frames = len(st.session_state.image_data)
                if num_frames > 1:
                    test_frame_idx = st.slider(
                        "Test Frame",
                        min_value=0,
                        max_value=num_frames - 1,
                        value=min(5, num_frames - 1),  # Default to frame 5 or last frame
                        key="tracking_test_frame"
                    )
                else:
                    test_frame_idx = 0
                
                # Get test frame for preview (apply tracking channel fusion if multichannel)
                raw_test = st.session_state.image_data[test_frame_idx]
                if isinstance(raw_test, np.ndarray) and raw_test.ndim == 3 and raw_test.shape[2] > 1:
                    # combine selected channels
                    t_channels = st.session_state.get("tracking_channels", [0])
                    t_mode = st.session_state.get("tracking_fusion_mode", "average")
                    t_weights = None
                    if t_mode == "weighted":
                        try:
                            t_weights = [float(x) for x in st.session_state.get("tracking_weights", "").split(",") if x.strip() != ""]
                        except Exception:
                            t_weights = None
                    try:
                        test_frame = _combine_channels(raw_test, t_channels, t_mode, t_weights)
                    except Exception:
                        test_frame = raw_test[:, :, 0]
                else:
                    test_frame = raw_test
                # If ROI restriction enabled, ignore out-of-ROI pixels for stats
                if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                    roi_vals = test_frame[roi_mask]
                    if roi_vals.size > 0:
                        frame_min, frame_max = float(np.min(roi_vals)), float(np.max(roi_vals))
                        frame_mean = float(np.mean(roi_vals))
                        frame_std = float(np.std(roi_vals))
                    else:
                        frame_min, frame_max = float(np.min(test_frame)), float(np.max(test_frame))
                        frame_mean = float(np.mean(test_frame))
                        frame_std = float(np.std(test_frame))
                else:
                    frame_min, frame_max = float(np.min(test_frame)), float(np.max(test_frame))
                    frame_mean = float(np.mean(test_frame))
                    frame_std = float(np.std(test_frame))
                
                # Ensure we have a valid range for the slider
                if frame_max <= frame_min:
                    frame_max = frame_min + 1.0
                
                # Calculate a safe step size based on the range
                range_size = frame_max - frame_min
                if range_size > 0:
                    step_size = max(0.01, range_size / 100.0)  # Ensure reasonable granularity
                else:
                    step_size = 0.01
                
                # Quick threshold testing
                preview_col1, preview_col2 = st.columns(2)
                
                with preview_col1:
                    # Interactive threshold slider based on frame statistics
                    # Ensure the default value is within the min/max range
                    default_threshold = max(frame_min, min(frame_max, frame_mean + frame_std))
                    
                    quick_threshold = st.slider(
                        "Quick Intensity Threshold",
                        min_value=frame_min,
                        max_value=frame_max,
                        value=default_threshold,
                        step=step_size,
                        help=f"Frame range: {frame_min:.1f} - {frame_max:.1f}"
                    )
                    
                    # Add noise reduction controls for tracking
                    st.write("**Noise Reduction**")
                    noise_reduction_enabled = st.checkbox("Enable Noise Reduction", value=False)
                    
                    if noise_reduction_enabled:
                        noise_method = st.selectbox(
                            "Noise Reduction Method",
                            ["Gaussian", "Median", "Non-local Means"],
                            help="Choose noise reduction algorithm"
                        )
                        
                        if noise_method == "Gaussian":
                            noise_strength = st.slider("Gaussian Sigma", 0.1, 3.0, 0.5, 0.1)
                        elif noise_method == "Median":
                            noise_strength = st.slider("Median Disk Size", 1, 10, 3, 1)
                        else:  # Non-local Means
                            noise_strength = st.slider("Denoising Strength", 0.1, 2.0, 0.3, 0.1)
                    
                    # Apply noise reduction if enabled
                    processed_frame = test_frame.copy()
                    if noise_reduction_enabled:
                        try:
                            from image_processing_utils import apply_noise_reduction
                            if noise_method == "Gaussian":
                                processed_frame = apply_noise_reduction([processed_frame], 'gaussian', {'sigma': noise_strength})[0]
                            elif noise_method == "Median":
                                processed_frame = apply_noise_reduction([processed_frame], 'median', {'disk_size': int(noise_strength)})[0]
                            else:  # Non-local Means
                                processed_frame = apply_noise_reduction([processed_frame], 'nlm', {'h': noise_strength})[0]
                        except Exception as e:
                            st.warning(f"Noise reduction failed: {str(e)}")
                            processed_frame = test_frame.copy()
                    
                    # Simple threshold preview on processed frame
                    threshold_mask = processed_frame > quick_threshold
                    # Apply ROI if requested
                    if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                        if roi_mask.shape == threshold_mask.shape:
                            threshold_mask = threshold_mask & roi_mask
                    detected_pixels = int(np.sum(threshold_mask))
                    total_pixels = threshold_mask.size
                    detection_percent = (detected_pixels / total_pixels) * 100
                    
                    st.metric("Detection Coverage", f"{detection_percent:.1f}%")
                    st.metric("Detected Pixels", f"{detected_pixels:,}")
                
                with preview_col2:
                    # Display threshold preview
                    if st.checkbox("Show Threshold Preview", value=True):
                        # Create side-by-side preview
                        preview_img = normalize_image_for_display(processed_frame)
                        threshold_img = (threshold_mask * 255).astype(np.uint8)
                        
                        st.write("**Original vs Threshold**")
                        img_col1, img_col2 = st.columns(2)
                        with img_col1:
                            st.image(preview_img, caption="Original", use_container_width=True)
                        with img_col2:
                            st.image(threshold_img, caption="Threshold", use_container_width=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Apply optimal parameters if requested
                default_method = "LoG"
                default_size = 3.0
                
                # Initialize optimal parameter defaults
                optimal_min_sigma = 0.5
                optimal_max_sigma = 3.0
                optimal_threshold = 0.05
                optimal_sigma1 = 1.0
                optimal_sigma2 = 2.0
                optimal_percentile = 90.0
                optimal_intensity_percentile = 95.0
                optimal_wavelet_type = "mexican_hat"
                optimal_wavelet_threshold = 1.0
                
                if 'apply_optimal' in st.session_state and st.session_state.apply_optimal and 'optimal_params' in st.session_state:
                    optimal = st.session_state['optimal_params']
                    default_method = optimal.get('method', 'LoG')
                    default_size = optimal.get('particle_size', 1.25)
                    
                    # Apply method-specific optimal parameters
                    if optimal['method'] == 'LoG':
                        optimal_min_sigma = optimal.get('min_sigma', 0.5)
                        optimal_max_sigma = optimal.get('max_sigma', 3.0)
                        optimal_threshold = optimal.get('threshold', 0.05)
                    elif optimal['method'] == 'DoG':
                        optimal_sigma1 = optimal.get('sigma1', 1.0)
                        optimal_sigma2 = optimal.get('sigma2', 2.0)
                        # DoG uses 'threshold' but should be clamped to valid percentile range
                        raw_threshold = optimal.get('threshold', 90.0)
                        optimal_percentile = max(5.0, min(99.0, raw_threshold))
                    elif optimal['method'] == 'Intensity':
                        optimal_intensity_percentile = optimal.get('percentile_thresh', 95.0)
                    elif optimal['method'] == 'Wavelet':
                        optimal_wavelet_threshold = optimal.get('threshold_factor', 1.0)
                        optimal_wavelet_type = optimal.get('wavelet_type', 'mexican_hat')
                    
                    st.session_state.apply_optimal = False  # Reset flag
                
                detection_method = st.selectbox(
                    "Detection Method",
                    ["LoG", "DoG", "Wavelet", "Intensity", "CellSAM", "Cellpose"],
                    index=["LoG", "DoG", "Wavelet", "Intensity", "CellSAM", "Cellpose"].index(default_method) if default_method in ["LoG", "DoG", "Wavelet", "Intensity", "CellSAM", "Cellpose"] else 0,
                    help="Method for detecting particles. CellSAM and Cellpose are AI-based methods for advanced segmentation."
                )
                
                particle_size = st.number_input(
                    "Particle Size (pixels)",
                    min_value=0.1,
                    max_value=50.0,
                    value=default_size,
                    step=0.1,
                    format="%.1f",
                    help="Approximate size of particles - use decimal values for fine tuning"
                )
                
            with col2:
                # Method-specific parameters with optimal values
                if detection_method == "LoG":
                    min_sigma = st.number_input(
                        "Min Sigma",
                        min_value=0.1,
                        max_value=5.0,
                        value=optimal_min_sigma,
                        step=0.1,
                        help="Minimum sigma for LoG detection"
                    )
                    max_sigma = st.number_input(
                        "Max Sigma", 
                        min_value=0.5,
                        max_value=10.0,
                        value=optimal_max_sigma,
                        step=0.1,
                        help="Maximum sigma for LoG detection"
                    )
                    threshold_factor = st.number_input(
                        "Threshold",
                        min_value=0.001,
                        max_value=1.0,
                        value=optimal_threshold,
                        step=0.001,
                        format="%.3f",
                        help="Detection threshold for LoG"
                    )
                    
                elif detection_method == "DoG":
                    sigma1 = st.number_input(
                        "Sigma 1",
                        min_value=0.1,
                        max_value=5.0,
                        value=optimal_sigma1,
                        step=0.1,
                        help="First sigma for DoG detection"
                    )
                    sigma2 = st.number_input(
                        "Sigma 2",
                        min_value=0.5,
                        max_value=10.0,
                        value=optimal_sigma2,
                        step=0.1,
                        help="Second sigma for DoG detection"
                    )
                    threshold_factor = st.number_input(
                        "Percentile Threshold",
                        min_value=5.0,
                        max_value=99.0,
                        value=max(5.0, min(99.0, optimal_percentile)),
                        step=1.0,
                        help="Percentile threshold for DoG detection"
                    )
                    
                elif detection_method == "Intensity":
                    threshold_factor = st.number_input(
                        "Percentile Threshold",
                        min_value=80.0,
                        max_value=99.5,
                        value=optimal_intensity_percentile,
                        step=0.5,
                        help="Percentile threshold for intensity detection"
                    )
                    
                elif detection_method == "Wavelet":
                    wavelet_type = st.selectbox(
                        "Wavelet Type",
                        ["mexican_hat", "morlet", "ricker"],
                        index=["mexican_hat", "morlet", "ricker"].index(optimal_wavelet_type),
                        help="Type of wavelet for detection"
                    )
                    threshold_factor = st.number_input(
                        "Threshold Factor",
                        min_value=0.001,
                        max_value=10.0,
                        value=optimal_wavelet_threshold,
                        step=0.001,
                        format="%.3f",
                        help="Threshold factor for wavelet detection"
                    )
                
                min_distance = st.number_input(
                    "Minimum Distance (pixels)",
                    min_value=0.1,
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                    format="%.1f",
                    help="Minimum distance between particles"
                )
            
            # Set default advanced parameters
            num_sigma = 5
            overlap_threshold = 0.5
            threshold_mode = "Auto (Otsu)"
            block_size = 11
            manual_threshold = 0.1
            
            # Advanced parameters in an expander
            with st.expander("Advanced Detection Parameters", expanded=False):
                st.write("**Fine-tune detection algorithms for better results**")
                
                if detection_method == "LoG":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        num_sigma = st.slider(
                            "Number of Sigma Steps",
                            min_value=1,
                            max_value=20,
                            value=5,
                            help="More steps = better detection but slower"
                        )
                    with adv_col2:
                        overlap_threshold = st.slider(
                            "Overlap Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.05,
                            help="Higher values = fewer overlapping detections"
                        )
                
                elif detection_method == "DoG":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        threshold_mode = st.selectbox(
                            "Threshold Mode",
                            ["Auto (Otsu)", "Manual", "Adaptive"],
                            help="How to determine detection threshold"
                        )
                    with adv_col2:
                        if threshold_mode == "Manual":
                            manual_threshold = st.slider(
                                "Manual Threshold",
                                min_value=0.01,
                                max_value=2.0,
                                value=0.1,
                                step=0.01,
                                format="%.2f",
                                help="Manual threshold value for DoG detection"
                            )
                        elif threshold_mode == "Adaptive":
                            block_size = st.slider(
                                "Adaptive Block Size",
                                min_value=3,
                                max_value=51,
                                value=11,
                                step=2,
                                help="Size of neighborhood for adaptive threshold"
                            )
                
                elif detection_method == "Wavelet":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        wavelet_type = st.selectbox(
                            "Wavelet Type",
                            ["sym8", "db4", "haar", "coif2", "bior2.2"],
                            help="Type of wavelet for decomposition"
                        )
                        wavelet_levels = st.slider(
                            "Decomposition Levels",
                            min_value=1,
                            max_value=6,
                            value=int(np.log2(particle_size)) + 2,
                            help="Number of wavelet decomposition levels"
                        )
                    with adv_col2:
                        detail_enhancement = st.slider(
                            "Detail Enhancement",
                            min_value=0.1,
                            max_value=5.0,
                            value=1.0,
                            step=0.1,
                            help="Enhance detail coefficients"
                        )
                        noise_reduction = st.slider(
                            "Noise Reduction",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.01,
                            help="Remove small coefficients (noise)"
                        )
                
                elif detection_method == "Intensity":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        intensity_mode = st.selectbox(
                            "Threshold Mode",
                            ["Auto (Otsu)", "Manual", "Percentile", "Mean + Std"],
                            help="How to determine intensity threshold"
                        )
                        if intensity_mode == "Percentile":
                            percentile_value = st.slider(
                                "Intensity Percentile",
                                min_value=50,
                                max_value=99,
                                value=90,
                                help="Use this percentile as threshold"
                            )
                        elif intensity_mode == "Mean + Std":
                            std_multiplier = st.number_input(
                                "Std Dev Multiplier",
                                min_value=0.1,
                                max_value=10.0,
                                value=2.0,
                                step=0.1,
                                help="Threshold = mean + (this × std dev)"
                            )
                    with adv_col2:
                        morphology_cleanup = st.checkbox(
                            "Morphological Cleanup",
                            value=True,
                            help="Clean up detected regions"
                        )
                        if morphology_cleanup:
                            erosion_size = st.slider(
                                "Erosion Size",
                                min_value=0,
                                max_value=5,
                                value=1,
                                help="""Size of erosion operation for morphological cleanup:
• 0: No erosion, preserves all detected features
• 1-2: Light erosion, removes thin connections and noise
• 3-4: Moderate erosion, separates touching particles
• 5: Heavy erosion, may reduce particle size significantly

Erosion shrinks detected particles to separate touching objects."""
                            )
                            dilation_size = st.slider(
                                "Dilation Size",
                                min_value=0,
                                max_value=5,
                                value=1,
                                help="""Size of dilation operation for morphological cleanup:
• 0: No dilation, particles may appear smaller
• 1-2: Light dilation, restores original particle size after erosion
• 3-4: Moderate dilation, fills gaps and smooths boundaries
• 5: Heavy dilation, may merge nearby particles

Dilation expands detected particles to restore size after erosion."""
                            )
            
            # Display image and interactive detector
            if len(st.session_state.image_data) > 1:
                frame_idx = st.slider(
                    "Frame to preview",
                    min_value=0,
                    max_value=len(st.session_state.image_data) - 1,
                    value=0
                )
            else:
                frame_idx = 0
                st.info("Single frame loaded")
            
            # Detection buttons
            col1, col2 = st.columns(2)
            with col1:
                single_frame_button = st.button("🔍 Run Detection (Current Frame)", type="secondary")
            with col2:
                all_frames_button = st.button("🚀 Run Detection (All Frames)", type="primary")
            
            # Show detection on button click to avoid auto-computation
            if single_frame_button:
                with st.spinner("Detecting particles on current frame..."):
                    try:
                        # Import the detect_particles function from tracking module
                        from tracking import detect_particles
                        
                        # Get the current frame for detection (apply tracking fusion)
                        _raw = st.session_state.image_data[frame_idx]
                        if isinstance(_raw, np.ndarray) and _raw.ndim == 3 and _raw.shape[2] > 1:
                            t_channels = st.session_state.get("tracking_channels", [0])
                            t_mode = st.session_state.get("tracking_fusion_mode", "average")
                            t_weights = None
                            if t_mode == "weighted":
                                try:
                                    t_weights = [float(x) for x in st.session_state.get("tracking_weights", "").split(",") if x.strip() != ""]
                                except Exception:
                                    t_weights = None
                            try:
                                current_frame = _combine_channels(_raw, t_channels, t_mode, t_weights)
                            except Exception:
                                current_frame = _raw[:, :, 0]
                        else:
                            current_frame = _raw.copy()
                        # Apply ROI
                        if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                            if roi_mask.shape == current_frame.shape:
                                # Zero-out non-ROI to avoid detections there
                                current_frame = current_frame * roi_mask
                        
                        # Use advanced detection with fine-tuned parameters
                        if detection_method == "LoG":
                            from skimage.feature import blob_log
                            # Use LoG parameters - get from main parameters section
                            blobs = blob_log(
                                current_frame, 
                                min_sigma=min_sigma,
                                max_sigma=max_sigma,
                                num_sigma=num_sigma if 'num_sigma' in locals() else 5,
                                threshold=threshold_factor,
                                overlap=overlap_threshold if 'overlap_threshold' in locals() else 0.5
                            )
                            particles_list = []
                            for blob in blobs:
                                y, x, sigma = blob
                                intensity = current_frame[int(y), int(x)]
                                particles_list.append({
                                    'y': y, 'x': x, 
                                    'sigma': sigma, 
                                    'intensity': intensity,
                                    'SNR': intensity / np.std(current_frame)
                                })
                            particles_df = pd.DataFrame(particles_list)
                            
                        elif detection_method == "DoG":
                            from skimage import filters, measure
                            # Apply Difference of Gaussians
                            dog = filters.gaussian(current_frame, sigma1) - filters.gaussian(current_frame, sigma2)
                            
                            # Apply threshold based on mode
                            if threshold_mode == "Auto (Otsu)":
                                thresh = filters.threshold_otsu(dog)
                            elif threshold_mode == "Manual":
                                # Use a more sensitive manual threshold based on std deviation
                                thresh = manual_threshold * np.std(dog)
                            elif threshold_mode == "Adaptive":
                                from skimage.filters import threshold_local
                                thresh = threshold_local(dog, block_size=block_size)
                                mask = dog > thresh
                            else:
                                mask = dog > (threshold_factor * np.mean(dog))
                            
                            if threshold_mode != "Adaptive":
                                mask = dog > thresh
                            
                            labeled_mask = measure.label(mask)
                            props = measure.regionprops(labeled_mask)
                            
                            min_area = int(particle_size * particle_size * 0.5)
                            particles_list = []
                            for prop in props:
                                if prop.area >= min_area:
                                    y, x = prop.centroid
                                    intensity = current_frame[int(y), int(x)]
                                    particles_list.append({
                                        'y': y, 'x': x, 
                                        'sigma': np.sqrt(prop.area / np.pi), 
                                        'intensity': intensity,
                                        'SNR': intensity / np.std(current_frame)
                                    })
                            particles_df = pd.DataFrame(particles_list)
                            
                        elif detection_method == "Wavelet":
                            import pywt
                            from skimage import measure
                            
                            # Perform wavelet decomposition
                            coeffs = pywt.wavedec2(current_frame, wavelet_type, level=wavelet_levels)
                            
                            # Enhance detail coefficients and reduce noise
                            coeffs_enhanced = list(coeffs)
                            for i in range(1, len(coeffs)):
                                if isinstance(coeffs[i], tuple):
                                    enhanced_details = []
                                    for detail in coeffs[i]:
                                        # Enhance details
                                        enhanced = detail * detail_enhancement
                                        # Reduce noise
                                        enhanced[np.abs(enhanced) < noise_reduction * np.std(enhanced)] = 0
                                        enhanced_details.append(enhanced)
                                    coeffs_enhanced[i] = tuple(enhanced_details)
                            
                            # Reconstruct enhanced image
                            enhanced = pywt.waverec2(coeffs_enhanced, wavelet_type)
                            
                            # Apply threshold and find particles
                            thresh = threshold_factor * np.std(enhanced)
                            mask = enhanced > thresh
                            
                            labeled_mask = measure.label(mask)
                            props = measure.regionprops(labeled_mask)
                            
                            min_area = int(particle_size * particle_size * 0.5)
                            particles_list = []
                            for prop in props:
                                if prop.area >= min_area:
                                    y, x = prop.centroid
                                    intensity = current_frame[int(y), int(x)]
                                    particles_list.append({
                                        'y': y, 'x': x, 
                                        'sigma': np.sqrt(prop.area / np.pi), 
                                        'intensity': intensity,
                                        'SNR': intensity / np.std(current_frame)
                                    })
                            particles_df = pd.DataFrame(particles_list)
                            
                        elif detection_method == "Intensity":
                            from skimage import filters, measure, morphology
                            
                            # Apply threshold based on mode
                            if intensity_mode == "Auto (Otsu)":
                                thresh = filters.threshold_otsu(current_frame)
                            elif intensity_mode == "Manual":
                                thresh = threshold_factor * np.mean(current_frame)
                            elif intensity_mode == "Percentile":
                                thresh = np.percentile(current_frame, percentile_value)
                            elif intensity_mode == "Mean + Std":
                                thresh = np.mean(current_frame) + std_multiplier * np.std(current_frame)
                            
                            # Create binary mask
                            mask = current_frame > thresh
                            
                            # Apply morphological operations if requested
                            if morphology_cleanup:
                                if erosion_size > 0:
                                    mask = morphology.erosion(mask, morphology.disk(erosion_size))
                                if dilation_size > 0:
                                    mask = morphology.dilation(mask, morphology.disk(dilation_size))
                            
                            # Label connected components
                            labeled_mask = measure.label(mask)
                            props = measure.regionprops(labeled_mask)
                            
                            # Filter by size and extract centroids
                            min_area = int(particle_size * particle_size * 0.5)
                            particles_list = []
                            for prop in props:
                                if prop.area >= min_area:
                                    y, x = prop.centroid
                                    intensity = current_frame[int(y), int(x)]
                                    particles_list.append({
                                        'y': y, 'x': x, 
                                        'sigma': np.sqrt(prop.area / np.pi), 
                                        'intensity': intensity,
                                        'SNR': intensity / np.std(current_frame)
                                    })
                            particles_df = pd.DataFrame(particles_list)
                        
                        elif detection_method == "CellSAM":
                            try:
                                from advanced_segmentation import CellSAMSegmentation
                                st.info("Using CellSAM (Segment Anything Model) for particle detection...")
                                
                                cellsam = CellSAMSegmentation()
                                success = cellsam.load_model()
                                
                                if success:
                                    masks, scores = cellsam.segment_automatic(current_frame)
                                    
                                    particles_list = []
                                    for mask in masks:
                                        props = measure.regionprops(mask.astype(int))
                                        if props:
                                            prop = props[0]
                                            min_area = int(particle_size * particle_size * 0.5)
                                            if prop.area >= min_area:
                                                y, x = prop.centroid
                                                intensity = current_frame[int(y), int(x)]
                                                particles_list.append({
                                                    'y': y, 'x': x,
                                                    'sigma': np.sqrt(prop.area / np.pi),
                                                    'intensity': intensity,
                                                    'SNR': intensity / np.std(current_frame)
                                                })
                                    particles_df = pd.DataFrame(particles_list)
                                else:
                                    st.error("Failed to load CellSAM model")
                                    particles_df = pd.DataFrame()
                            except Exception as e:
                                st.error(f"CellSAM detection failed: {e}")
                                particles_df = pd.DataFrame()
                        
                        elif detection_method == "Cellpose":
                            try:
                                from advanced_segmentation import CellposeSegmentation
                                st.info("Using Cellpose for particle detection...")
                                
                                cellpose = CellposeSegmentation()
                                success = cellpose.load_model()
                                
                                if success:
                                    masks, flows, _ = cellpose.segment_image(current_frame)
                                    
                                    particles_list = []
                                    for region_id in np.unique(masks):
                                        if region_id == 0:  # Skip background
                                            continue
                                        mask = masks == region_id
                                        props = measure.regionprops(mask.astype(int))
                                        if props:
                                            prop = props[0]
                                            min_area = int(particle_size * particle_size * 0.5)
                                            if prop.area >= min_area:
                                                y, x = prop.centroid
                                                intensity = current_frame[int(y), int(x)]
                                                particles_list.append({
                                                    'y': y, 'x': x,
                                                    'sigma': np.sqrt(prop.area / np.pi),
                                                    'intensity': intensity,
                                                    'SNR': intensity / np.std(current_frame)
                                                })
                                    particles_df = pd.DataFrame(particles_list)
                                else:
                                    st.error("Failed to load Cellpose model")
                                    particles_df = pd.DataFrame()
                            except Exception as e:
                                st.error(f"Cellpose detection failed: {e}")
                                particles_df = pd.DataFrame()
                        
                        # Convert DataFrame to numpy array format for visualization
                        if not particles_df.empty:
                            particles = particles_df[['x', 'y']].values
                            particle_info = particles_df.to_dict('records')
                        else:
                            particles = np.array([]).reshape(0, 2)
                            particle_info = []
                        
                        # Store detection results for current frame
                        st.session_state.detection_results = {
                            "frame": frame_idx,
                            "detected": len(particles),
                            "particles": particles,
                            "particle_details": particle_info,
                            "method": detection_method,
                            "parameters": {
                                "particle_size": particle_size,
                                "threshold_factor": threshold_factor,
                                "min_distance": min_distance
                            }
                        }
                        
                        # Also store in accumulated detections for linking
                        if 'all_detections' not in st.session_state:
                            st.session_state.all_detections = {}
                        
                        # Store particles with intensity and SNR information
                        enhanced_particles = []
                        for i, particle in enumerate(particles):
                            enhanced_particle = [
                                particle[0],  # x
                                particle[1],  # y
                                particle_info[i]['intensity'],  # intensity
                                particle_info[i]['SNR']  # SNR
                            ]
                            enhanced_particles.append(enhanced_particle)
                        
                        st.session_state.all_detections[frame_idx] = enhanced_particles
                        
                        st.success(f"Detected {len(particles)} particles using {detection_method}")
                        
                        # Show additional details if particles were found
                        if len(particles) > 0:
                            avg_intensity = np.mean([p['intensity'] for p in particle_info])
                            avg_snr = np.mean([p['SNR'] for p in particle_info])
                            st.info(f"Average intensity: {avg_intensity:.2f}, Average SNR: {avg_snr:.2f}")
                        
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        st.session_state.detection_results = {"frame": frame_idx, "detected": 0}
            
            # Run detection on all frames
            if all_frames_button:
                total_frames = len(st.session_state.image_data)
                
                with st.spinner(f"Running detection on all {total_frames} frames..."):
                    # Initialize progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Clear previous all_detections
                    st.session_state.all_detections = {}
                    total_particles_detected = 0
                    
                    try:
                        from tracking import detect_particles
                        
                        for current_frame_idx in range(total_frames):
                            # Update progress
                            progress = (current_frame_idx + 1) / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {current_frame_idx + 1}/{total_frames}...")
                            
                            _rawf = st.session_state.image_data[current_frame_idx]
                            if isinstance(_rawf, np.ndarray) and _rawf.ndim == 3 and _rawf.shape[2] > 1:
                                t_channels = st.session_state.get("tracking_channels", [0])
                                t_mode = st.session_state.get("tracking_fusion_mode", "average")
                                t_weights = None
                                if t_mode == "weighted":
                                    try:
                                        t_weights = [float(x) for x in st.session_state.get("tracking_weights", "").split(",") if x.strip() != ""]
                                    except Exception:
                                        t_weights = None
                                try:
                                    current_frame = _combine_channels(_rawf, t_channels, t_mode, t_weights)
                                except Exception:
                                    current_frame = _rawf[:, :, 0]
                            else:
                                current_frame = _rawf
                            # Apply ROI
                            if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                                if roi_mask.shape == current_frame.shape:
                                    current_frame = current_frame * roi_mask
                            
                            # Run the same detection logic as the single frame version
                            if detection_method == "LoG":
                                from skimage import feature
                                blobs = feature.blob_log(current_frame, min_sigma=min_sigma, max_sigma=max_sigma, 
                                                       num_sigma=num_sigma, threshold=threshold_factor)
                                if len(blobs) > 0:
                                    particles = blobs[:, :2]  # x, y coordinates
                                    particle_info = []
                                    for blob in blobs:
                                        y, x, sigma = blob
                                        intensity = current_frame[int(y), int(x)]
                                        particle_info.append({
                                            'x': x, 'y': y, 'sigma': sigma, 
                                            'intensity': intensity,
                                            'SNR': intensity / np.std(current_frame)
                                        })
                                else:
                                    particles = np.array([]).reshape(0, 2)
                                    particle_info = []
                            
                            elif detection_method == "DoG":
                                from skimage import filters, measure
                                dog = filters.gaussian(current_frame, sigma1) - filters.gaussian(current_frame, sigma2)
                                
                                if threshold_mode == "Auto (Otsu)":
                                    thresh = filters.threshold_otsu(dog)
                                elif threshold_mode == "Manual":
                                    thresh = manual_threshold * np.std(dog)
                                elif threshold_mode == "Adaptive":
                                    from skimage.filters import threshold_local
                                    thresh = threshold_local(dog, block_size=block_size)
                                    mask = dog > thresh
                                else:
                                    mask = dog > (threshold_factor * np.mean(dog))
                                
                                if threshold_mode != "Adaptive":
                                    mask = dog > thresh
                                
                                labeled_mask = measure.label(mask)
                                props = measure.regionprops(labeled_mask)
                                
                                min_area = int(particle_size * particle_size * 0.5)
                                max_area = int(particle_size * particle_size * 2.0)
                                
                                particles_list = []
                                particle_info = []
                                for prop in props:
                                    if min_area <= prop.area <= max_area:
                                        y, x = prop.centroid
                                        intensity = current_frame[int(y), int(x)]
                                        particles_list.append([x, y])
                                        particle_info.append({
                                            'x': x, 'y': y, 
                                            'area': prop.area,
                                            'sigma': np.sqrt(prop.area / np.pi), 
                                            'intensity': intensity,
                                            'SNR': intensity / np.std(current_frame)
                                        })
                                
                                if particles_list:
                                    particles = np.array(particles_list)
                                else:
                                    particles = np.array([]).reshape(0, 2)
                            
                            elif detection_method == "Wavelet":
                                import pywt
                                from skimage import measure
                                
                                # Perform wavelet decomposition
                                coeffs = pywt.wavedec2(current_frame, wavelet_type, level=wavelet_levels)
                                
                                # Enhance detail coefficients and reduce noise
                                coeffs_enhanced = list(coeffs)
                                for i in range(1, len(coeffs)):
                                    if isinstance(coeffs[i], tuple):
                                        enhanced_details = []
                                        for detail in coeffs[i]:
                                            enhanced = detail * detail_enhancement
                                            enhanced[np.abs(enhanced) < noise_reduction * np.std(enhanced)] = 0
                                            enhanced_details.append(enhanced)
                                        coeffs_enhanced[i] = tuple(enhanced_details)
                                    else:
                                        enhanced = coeffs[i] * detail_enhancement
                                        enhanced[np.abs(enhanced) < noise_reduction * np.std(enhanced)] = 0
                                        coeffs_enhanced[i] = enhanced
                                
                                # Reconstruct the enhanced image
                                enhanced_image = pywt.waverec2(coeffs_enhanced, wavelet_type)
                                enhanced_image = np.abs(enhanced_image)
                                
                                # Apply threshold
                                thresh = threshold_factor * np.mean(enhanced_image)
                                binary = enhanced_image > thresh
                                
                                # Label connected components
                                labeled = measure.label(binary)
                                props = measure.regionprops(labeled, intensity_image=current_frame)
                                
                                particles_list = []
                                particle_info = []
                                min_area = int(particle_size * particle_size * 0.5)
                                max_area = int(particle_size * particle_size * 2.0)
                                
                                for prop in props:
                                    if min_area <= prop.area <= max_area:
                                        y, x = prop.centroid
                                        intensity = prop.mean_intensity
                                        particles_list.append([x, y])
                                        particle_info.append({
                                            'x': x, 'y': y, 
                                            'area': prop.area,
                                            'sigma': np.sqrt(prop.area / np.pi), 
                                            'intensity': intensity,
                                            'SNR': intensity / np.std(current_frame)
                                        })
                                
                                if particles_list:
                                    particles = np.array(particles_list)
                                else:
                                    particles = np.array([]).reshape(0, 2)
                            
                            elif detection_method == "Intensity":
                                from skimage import measure, morphology, filters
                                
                                # Apply intensity threshold based on mode
                                if intensity_mode == "Auto (Otsu)":
                                    thresh = filters.threshold_otsu(current_frame)
                                elif intensity_mode == "Manual":
                                    thresh = threshold_factor * np.mean(current_frame)
                                elif intensity_mode == "Percentile":
                                    thresh = np.percentile(current_frame, percentile_value)
                                elif intensity_mode == "Mean + Std":
                                    thresh = np.mean(current_frame) + std_multiplier * np.std(current_frame)
                                else:
                                    thresh = threshold_factor * np.mean(current_frame)
                                
                                # Create binary mask
                                binary = current_frame > thresh
                                
                                # Apply morphological cleanup if enabled
                                if morphology_cleanup:
                                    if erosion_size > 0:
                                        binary = morphology.binary_erosion(binary, morphology.disk(erosion_size))
                                    if dilation_size > 0:
                                        binary = morphology.binary_dilation(binary, morphology.disk(dilation_size))
                                
                                # Label connected components
                                labeled = measure.label(binary)
                                props = measure.regionprops(labeled, intensity_image=current_frame)
                                
                                particles_list = []
                                particle_info = []
                                min_area = int(particle_size * particle_size * 0.5)
                                max_area = int(particle_size * particle_size * 2.0)
                                
                                for prop in props:
                                    if min_area <= prop.area <= max_area:
                                        y, x = prop.centroid
                                        intensity = prop.mean_intensity
                                        particles_list.append([x, y])
                                        particle_info.append({
                                            'x': x, 'y': y, 
                                            'area': prop.area,
                                            'sigma': np.sqrt(prop.area / np.pi), 
                                            'intensity': intensity,
                                            'SNR': intensity / np.std(current_frame)
                                        })
                                
                                if particles_list:
                                    particles = np.array(particles_list)
                                else:
                                    particles = np.array([]).reshape(0, 2)
                            
                            else:
                                # Fallback - no particles detected
                                particles = np.array([]).reshape(0, 2)
                                particle_info = []
                            
                            # Store enhanced particles for linking
                            enhanced_particles = []
                            for i, particle in enumerate(particles):
                                enhanced_particle = [
                                    particle[0],  # x
                                    particle[1],  # y
                                    particle_info[i]['intensity'],  # intensity
                                    particle_info[i]['SNR']  # SNR
                                ]
                                enhanced_particles.append(enhanced_particle)
                            
                            st.session_state.all_detections[current_frame_idx] = enhanced_particles
                            total_particles_detected += len(enhanced_particles)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show success message
                        st.success(f"🎉 Detection completed on all {total_frames} frames!")
                        st.info(f"📊 Total particles detected: **{total_particles_detected}**")
                        
                        # Show frame summary
                        with st.expander("📋 Detection Summary by Frame"):
                            for frame_num in sorted(st.session_state.all_detections.keys()):
                                count = len(st.session_state.all_detections[frame_num])
                                st.write(f"Frame {frame_num}: {count} particles")
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error during batch detection: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
            
            # Display current frame
            display_image = st.session_state.image_data[frame_idx].copy()
            
            # Handle 3D images (z-stacks or timelapses incorrectly loaded as single frame)
            if len(display_image.shape) == 3:
                # Check if third dimension is too large to be channels
                if display_image.shape[2] > 4:
                    # Likely a z-stack loaded incorrectly - take max projection
                    display_image = np.max(display_image, axis=2)
                    st.info(f"⚠️ 3D stack detected ({display_image.shape[2]} slices). Showing max projection.")
                elif display_image.shape[2] == 3 or display_image.shape[2] == 4:
                    # Valid RGB or RGBA image - keep as is
                    pass
                elif display_image.shape[2] == 2:
                    # Two channels - take first channel
                    display_image = display_image[:, :, 0]
                else:
                    # Single channel with extra dimension
                    display_image = display_image.squeeze()
            
            # Ensure 2D for grayscale or keep 3D for RGB
            if len(display_image.shape) > 2 and display_image.shape[2] not in [3, 4]:
                display_image = display_image[:, :, 0] if display_image.shape[2] > 0 else display_image.squeeze()
            
            # Normalize image data to 0-255 range for display
            if display_image.dtype != np.uint8:
                # Handle different data types and ranges
                if display_image.max() <= 1.0:
                    # Assuming 0-1 range, scale to 0-255
                    display_image = (display_image * 255).astype(np.uint8)
                else:
                    # Scale to 0-255 from current range
                    display_image = ((display_image - display_image.min()) / 
                                   (display_image.max() - display_image.min()) * 255).astype(np.uint8)
            
            # Display image with particles overlaid if detection results available
            if (st.session_state.detection_results and 
                st.session_state.detection_results['frame'] == frame_idx and 
                'particles' in st.session_state.detection_results and
                len(st.session_state.detection_results['particles']) > 0):
                
                # Create image with particle overlay
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from matplotlib.patches import Circle
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(display_image, cmap='gray', aspect='equal')
                
                # Overlay detected particles
                particles = st.session_state.detection_results['particles']
                for particle in particles:
                    x, y = particle[0], particle[1]
                    circle = Circle((x, y), radius=particle_size/2, fill=False, 
                                  color='red', linewidth=2, alpha=0.8)
                    ax.add_patch(circle)
                
                ax.set_title(f"Frame {frame_idx} - {len(particles)} particles detected")
                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                
                # Remove ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                st.pyplot(fig)
                plt.close()
                
                # Display detection summary
                st.success(f"✓ Detected {len(particles)} particles using {st.session_state.detection_results['method']}")
                
                # Show detection parameters used
                with st.expander("Detection Parameters Used"):
                    params = st.session_state.detection_results['parameters']
                    st.write(f"**Method:** {st.session_state.detection_results['method']}")
                    st.write(f"**Particle Size:** {params['particle_size']} pixels")
                    st.write(f"**Threshold Factor:** {params['threshold_factor']}")
                    st.write(f"**Minimum Distance:** {params['min_distance']} pixels")
                
                # Option to save detected particles
                if st.button("Save Particle Coordinates"):
                    import pandas as pd
                    particle_df = pd.DataFrame(particles, columns=['x', 'y'])
                    particle_df['frame'] = frame_idx
                    
                    # Convert to CSV for download
                    csv = particle_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"particles_frame_{frame_idx}.csv",
                        mime="text/csv"
                    )
                    st.success("Particle coordinates ready for download!")
            else:
                # Display image without overlay
                st.image(display_image, caption=f"Frame {frame_idx}", use_container_width=True)
                
                # Show detection status
                if st.session_state.detection_results:
                    if st.session_state.detection_results['frame'] != frame_idx:
                        st.info(f"Detection results available for frame {st.session_state.detection_results['frame']}. Click 'Run Detection' to detect particles in current frame.")
                    elif ('particles' in st.session_state.detection_results and 
                          len(st.session_state.detection_results['particles']) == 0):
                        st.warning("No particles detected in this frame. Try adjusting the detection parameters.")
                else:
                    st.info("Click 'Run Detection' to detect particles in this frame.")
            
        with tab2:
            st.header("Particle Linking")
            
            # Show detection status first
            st.subheader("Detection Status")
            
            if hasattr(st.session_state, 'all_detections') and st.session_state.all_detections:
                detected_frames = sorted(st.session_state.all_detections.keys())
                total_particles = sum(len(particles) for particles in st.session_state.all_detections.values())
                
                st.success(f"✅ Detection completed on {len(detected_frames)} frames with {total_particles} total particles")
                
                # Show frame-by-frame summary
                with st.expander("Frame-by-frame Detection Summary"):
                    for frame in detected_frames:
                        particle_count = len(st.session_state.all_detections[frame])
                        st.write(f"**Frame {frame}:** {particle_count} particles")
                
                # Button to clear all detections
                if st.button("🗑️ Clear All Detections"):
                    st.session_state.all_detections = {}
                    st.rerun()
                    
            else:
                st.info("👆 Run particle detection on multiple frames first before linking.")
                st.write("**Tip:** Use the Detection tab above to detect particles frame by frame, then return here to link them into tracks.")
            
            st.subheader("Linking Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                linking_method = st.selectbox(
                    "Linking Method",
                    ["Trackpy", "btrack", "Particle Filter", "Hungarian", "LAP"],
                    help="Method for linking particles between frames. Trackpy uses nearest-neighbor with gap closing, btrack uses Bayesian tracking, Particle Filter uses probabilistic tracking."
                )
                
                max_distance = st.slider(
                    "Maximum Distance (pixels)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    help="Maximum allowed linking distance"
                )
                
            with col2:
                max_frame_gap = st.slider(
                    "Maximum Frame Gap",
                    min_value=0,
                    max_value=10,
                    value=0,
                    help="Maximum allowed gap in frames"
                )
                
                min_track_length = st.slider(
                    "Minimum Track Length",
                    min_value=2,
                    max_value=20,
                    value=5,
                    help="Minimum length of tracks to retain"
                )
            
            # Run linking on button click
            if st.button("Run Linking", type="primary"):
                if not hasattr(st.session_state, 'all_detections') or not st.session_state.all_detections:
                    st.warning("⚠️ No detection results found. Please run particle detection on multiple frames first.")
                else:
                    with st.spinner("Running particle linking algorithm..."):
                        try:
                            # Use the proper tracking module for linking
                            from tracking import link_particles
                            
                            # Convert detections format for tracking module
                            detection_dict = {}
                            for frame_num, particles in st.session_state.all_detections.items():
                                if len(particles) > 0:
                                    df_data = []
                                    for particle in particles:
                                        df_data.append({
                                            'x': particle[0],
                                            'y': particle[1], 
                                            'intensity': particle[2] if len(particle) > 2 else 1.0,
                                            'SNR': particle[3] if len(particle) > 3 else 1.0,
                                            'sigma': 1.0
                                        })
                                    detection_dict[frame_num] = pd.DataFrame(df_data)
                                else:
                                    detection_dict[frame_num] = pd.DataFrame(columns=['x', 'y', 'intensity', 'SNR', 'sigma'])
                            
                            # Link particles using the tracking module
                            if linking_method == "btrack":
                                from advanced_tracking import AdvancedTracking
                                advanced_tracker = AdvancedTracking()
                                # btrack expects a single dataframe with a 't' column for the frame
                                all_detections_list = []
                                for frame, detections in st.session_state.all_detections.items():
                                    if detections:
                                        df = pd.DataFrame(detections, columns=['x', 'y', 'intensity', 'SNR'])
                                        df['t'] = frame
                                        df['z'] = 0
                                        df['label'] = 0
                                        all_detections_list.append(df)
                                if all_detections_list:
                                    all_detections_df = pd.concat(all_detections_list, ignore_index=True)
                                    tracks_data = advanced_tracker.track_particles_btrack(all_detections_df, min_track_length)
                                else:
                                    tracks_data = pd.DataFrame()
                                    
                            elif linking_method == "Particle Filter":
                                from advanced_tracking import AdvancedTracking
                                advanced_tracker = AdvancedTracking()
                                # Particle filter expects dict of frame -> DataFrame
                                tracks_data = advanced_tracker.track_particles(
                                    detection_dict,
                                    max_search_radius=max_distance,
                                    motion_std=5.0,
                                    measurement_std=2.0,
                                    min_track_length=min_track_length
                                )
                                
                            elif linking_method in ["Trackpy", "Hungarian", "LAP"]:
                                # Use the standard link_particles function (trackpy-based)
                                linked_tracks_df = link_particles(
                                    detection_dict,
                                    max_distance=max_distance,
                                    memory=max_frame_gap,
                                    min_track_length=min_track_length
                                )
                                if not linked_tracks_df.empty:
                                    tracks_data = linked_tracks_df
                                else:
                                    tracks_data = pd.DataFrame()
                                    
                            else:
                                # Default to trackpy
                                linked_tracks_df = link_particles(
                                    detection_dict,
                                    max_distance=max_distance,
                                    memory=max_frame_gap,
                                    min_track_length=min_track_length
                                )
                                if not linked_tracks_df.empty:
                                    tracks_data = linked_tracks_df
                                else:
                                    tracks_data = pd.DataFrame()

                            # Filter by minimum track length and store directly
                            if not tracks_data.empty:
                                # Store the filtered DataFrame directly
                                st.session_state.tracks_data = tracks_data
                                
                                # Data is already in DataFrame format, just store results
                                st.session_state.track_results = {
                                    "n_tracks": tracks_data['track_id'].nunique(),
                                    "total_points": len(tracks_data),
                                    "method": linking_method
                                }
                                
                                st.success(f"✅ Successfully linked {tracks_data['track_id'].nunique()} tracks with {len(tracks_data)} total points!")
                                st.balloons()
                                
                            else:
                                st.warning("No tracks found with the current parameters. Try adjusting the linking settings.")
                                
                        except Exception as e:
                            st.error(f"Error during linking: {str(e)}")
                            st.session_state.track_results = {"n_tracks": 0}
            
            # Display linking results if available
            if hasattr(st.session_state, 'track_results') and st.session_state.track_results:
                st.success(f"Linked {st.session_state.track_results['n_tracks']} tracks.")
        
        with tab3:
            st.header("Track Results")
            
            if st.session_state.tracks_data is not None:
                # Display track statistics
                st.subheader("Track Statistics")
                if st.session_state.track_statistics is not None:
                    st.dataframe(st.session_state.track_statistics)
                else:
                    # Calculate statistics if not already done
                    with st.spinner("Calculating track statistics..."):
                        st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.dataframe(st.session_state.track_statistics)
                
                # Preview tracks
                st.subheader("Track Visualization")
                max_tracks_to_display = min(20, st.session_state.tracks_data['track_id'].nunique())
                fig = plot_tracks(
                    st.session_state.tracks_data.query(f"track_id < {max_tracks_to_display}"), 
                    color_by='track_id'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Navigation button
                st.button("Proceed to Analysis", on_click=navigate_to, args=("Analysis",))
            else:
                st.info("No track data available. Complete detection and linking or upload track data.")
                st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))

# Analysis page with multiple modules
elif st.session_state.active_page == "Analysis":
    st.title("Track Analysis")
    
    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
    else:
        # Create tabs for different analysis modules
        tabs = st.tabs([
            "Overview", 
            "Diffusion Analysis", 
            "Motion Analysis", 
            "Clustering Analysis",
            "Dwell Time Analysis",
            "Boundary Crossing Analysis",
            "Multi-Channel Analysis",
            "Advanced Analysis"
        ])
        
        # Overview tab
        with tabs[0]:
            st.header("Track Overview")
            
            # Display track statistics
            st.subheader("Track Statistics")
            if not hasattr(st.session_state, 'track_statistics') or st.session_state.track_statistics is None:
                with st.spinner("Calculating track statistics..."):
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
            
            st.dataframe(st.session_state.track_statistics)
            
            # Basic visualization
            st.subheader("Track Visualization")
            
            viz_type = st.radio("Visualization Type", ["2D Tracks", "3D Tracks (time as Z)", "Statistics"])
            
            if viz_type == "2D Tracks":
                color_by = st.selectbox("Color tracks by", ["track_id", "track_length", "mean_speed", "straightness"])
                
                # For color_by options that are in track_statistics, add that information to tracks_data
                if color_by != "track_id" and color_by in st.session_state.track_statistics.columns:
                    # Create a mapping from track_id to the selected color_by value
                    color_map = st.session_state.track_statistics.set_index('track_id')[color_by].to_dict()
                    
                    # Add a temporary column for coloring
                    temp_df = st.session_state.tracks_data.copy()
                    temp_df[color_by] = temp_df['track_id'].map(color_map)
                    
                    fig = plot_tracks(temp_df, color_by=color_by)
                else:
                    fig = plot_tracks(st.session_state.tracks_data, color_by=color_by)
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "3D Tracks (time as Z)":
                fig = plot_tracks_3d(st.session_state.tracks_data)
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Statistics
                if st.session_state.track_statistics is not None:
                    figs = plot_track_statistics(st.session_state.track_statistics)
                    
                    for name, fig in figs.items():
                        st.subheader(name.replace("_", " ").title())
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No track statistics available.")
            
        # Diffusion Analysis tab
        with tabs[1]:
            st.header("Diffusion Analysis")
            
            # Primary analysis choice: whole image vs segmentation-based vs subpopulation
            analysis_type = st.radio(
                "Analysis Type",
                ["Whole Image Analysis", "Segmentation-Based Analysis", "Subpopulation Analysis (by Cell)"],
                help="Choose analysis approach: whole image, segmentation-based, or detect subpopulations within groups"
            )
            
            selected_mask = None
            selected_classes = None
            segmentation_method = None
            tracks_with_classes = None
            analysis_option = None
            
            if analysis_type == "Segmentation-Based Analysis":
                # Get available masks
                available_masks = get_available_masks()
                
                if not available_masks:
                    st.error("No segmentation masks available. Please create masks in the Image Processing tab first.")
                else:
                    # Categorize masks by type using metadata
                    simple_masks = []
                    two_step_masks = []
                    density_masks = []
                    
                    for mask_name in available_masks.keys():
                        if mask_name in st.session_state.mask_metadata:
                            mask_meta = st.session_state.mask_metadata[mask_name]
                            mask_type = mask_meta.get('type', 'unknown').lower()
                            
                            if 'density' in mask_type or 'nuclear_density' in mask_type:
                                density_masks.append(mask_name)
                            elif mask_meta.get('n_classes', 2) >= 3:
                                two_step_masks.append(mask_name)
                            else:
                                simple_masks.append(mask_name)
                    
                    # Build segmentation options
                    segmentation_options = []
                    if simple_masks:
                        segmentation_options.append("Simple Segmentation (Binary)")
                    if two_step_masks:
                        segmentation_options.append("Two-Step Segmentation (3 Classes)")
                    if density_masks:
                        segmentation_options.append("Nuclear Density Mapping")
                    
                    if segmentation_options:
                        segmentation_method = st.selectbox(
                            "Choose Segmentation Method",
                            segmentation_options,
                            help="Select the type of segmentation to use for analysis"
                        )
                        
                        # Select specific mask and assign tracks to classes
                        if segmentation_method == "Simple Segmentation (Binary)":
                            if simple_masks:
                                selected_mask = simple_masks[0] if len(simple_masks) == 1 else st.selectbox("Select Mask", simple_masks)
                                st.info(f"Using mask: **{selected_mask}**")
                                
                                # Assign tracks to classes dynamically
                                tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, [0, 1])
                                if 'class' in tracks_with_classes.columns:
                                    class_summary = tracks_with_classes['class'].value_counts().sort_index()
                                    st.write("**Track Distribution:**")
                                    for class_id, count in class_summary.items():
                                        class_name = ["Background", "Nucleus"][int(class_id)]
                                        st.write(f"- {class_name}: {count} track points")
                                    
                        elif segmentation_method == "Two-Step Segmentation (3 Classes)":
                            if two_step_masks:
                                selected_mask = two_step_masks[0] if len(two_step_masks) == 1 else st.selectbox("Select Mask", two_step_masks)
                                st.info(f"Using mask: **{selected_mask}**")
                                
                                # Analysis options for Two-Step Segmentation
                                analysis_option = st.selectbox(
                                    "Two-Step Analysis Method",
                                    [
                                        "Analyze all three classes separately",
                                        "Analyze classes separately, then combine Class 1 and 2"
                                    ]
                                )
                                
                                # Assign tracks to classes dynamically
                                tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, [0, 1, 2])
                                if 'class' in tracks_with_classes.columns:
                                    class_summary = tracks_with_classes['class'].value_counts().sort_index()
                                    st.write("**Track Distribution:**")
                                    for class_id, count in class_summary.items():
                                        class_name = ["Background", "Class 1", "Class 2"][int(class_id)]
                                        st.write(f"- {class_name}: {count} track points")
                                
                        elif segmentation_method == "Nuclear Density Mapping":
                            if density_masks:
                                selected_mask = density_masks[0] if len(density_masks) == 1 else st.selectbox("Select Mask", density_masks)
                                st.info(f"Using mask: **{selected_mask}**")
                                
                                # Get mask metadata to determine number of classes
                                mask_metadata = st.session_state.mask_metadata[selected_mask]
                                n_classes = mask_metadata['n_classes']
                                class_list = list(range(n_classes))
                                
                                # Assign tracks to classes dynamically
                                tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, class_list)
                                if 'class' in tracks_with_classes.columns:
                                    class_summary = tracks_with_classes['class'].value_counts().sort_index()
                                    st.write("**Track Distribution:**")
                                    for class_id, count in class_summary.items():
                                        if int(class_id) == 0:
                                            class_name = "Background"
                                        elif int(class_id) == 1:
                                            class_name = "Low Density"
                                        elif int(class_id) == 2:
                                            class_name = "High Density"
                                        else:
                                            class_name = f"Class {int(class_id)}"
                                        st.write(f"- {class_name}: {count} track points")
                    else:
                        st.error("No compatible segmentation masks found.")
            
            if selected_mask:
                mask_name = selected_mask[0] if isinstance(selected_mask, list) else selected_mask
                st.info(f"Analysis will be performed on mask: {mask_name}")
                if selected_classes:
                    mask_classes = selected_classes.get(mask_name, []) if isinstance(selected_classes, dict) else selected_classes
                    st.info(f"Using mask classes: {', '.join(map(str, mask_classes))}")
            
            # Parameters for diffusion analysis
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_lag = st.slider(
                    "Maximum Lag Time (frames)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Maximum lag time for MSD calculation"
                )
                
                # Sync and use global settings from Image Settings tab
                sync_global_parameters()
                pixel_size = get_global_pixel_size()
                frame_interval = get_global_frame_interval()
                
                # Log session state values for debugging
                logger.debug(f"Session state - global_pixel_size: {st.session_state.get('global_pixel_size', 'NOT FOUND')}")
                logger.debug(f"Session state - current_pixel_size: {st.session_state.get('current_pixel_size', 'NOT FOUND')}")
                logger.debug(f"Session state - global_frame_interval: {st.session_state.get('global_frame_interval', 'NOT FOUND')}")
                logger.debug(f"Session state - current_frame_interval: {st.session_state.get('current_frame_interval', 'NOT FOUND')}")
                logger.debug(f"Final pixel_size: {pixel_size}")
                logger.debug(f"Final frame_interval: {frame_interval}")
                
                # Optional: Still show debug expander for user visibility
                with st.expander("Debug: Session State Values", expanded=False):
                    st.write("global_pixel_size:", st.session_state.get('global_pixel_size', 'NOT FOUND'))
                    st.write("current_pixel_size:", st.session_state.get('current_pixel_size', 'NOT FOUND'))
                    st.write("global_frame_interval:", st.session_state.get('global_frame_interval', 'NOT FOUND'))
                    st.write("current_frame_interval:", st.session_state.get('current_frame_interval', 'NOT FOUND'))
                    st.write("Final pixel_size:", pixel_size)
                    st.write("Final frame_interval:", frame_interval)
                
                st.info(f"Using global settings: Pixel Size = {pixel_size:.3f} µm, Frame Interval = {frame_interval:.3f} s")
                st.info("To change these values, use the Image Settings tab in Data Loading section.")
                
            with col2:
                min_track_length = st.slider(
                    "Minimum Track Length",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="Minimum track length to include in analysis"
                )
                
                fit_method = st.selectbox(
                    "Fitting Method",
                    ["linear", "weighted", "nonlinear"],
                    help="Method for fitting MSD curves"
                )
                
                analysis_options = st.multiselect(
                    "Analysis Options",
                    ["Anomalous Diffusion", "Confined Diffusion"],
                    default=["Anomalous Diffusion", "Confined Diffusion"],
                    help="Additional analysis options"
                )
            
            # Run analysis on button click
            if st.button("Run Diffusion Analysis"):
                with st.spinner("Running diffusion analysis..."):
                    try:
                        analyze_anomalous = "Anomalous Diffusion" in analysis_options
                        check_confinement = "Confined Diffusion" in analysis_options
                        
                        if analysis_type == "Whole Image Analysis":
                            # Standard whole-image diffusion analysis
                            st.subheader("Whole Image Diffusion Analysis")
                            result = analyze_diffusion(
                                st.session_state.tracks_data,
                                max_lag=max_lag,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                min_track_length=min_track_length,
                                fit_method=fit_method,
                                analyze_anomalous=analyze_anomalous,
                                check_confinement=check_confinement
                            )
                            # Extract result dict or error
                            if result.get('success', False):
                                diffusion_results = result['result']
                                diffusion_results['success'] = True
                            else:
                                diffusion_results = {
                                    'success': False,
                                    'error': result.get('error', 'Unknown error')
                                }
                        elif analysis_type == "Segmentation-Based Analysis" and selected_mask and segmentation_method:
                            # Perform class-based diffusion analysis using tracks already assigned to classes
                            st.subheader(f"Class-Based Diffusion Analysis - {segmentation_method}")
                            
                            # Check if tracks_with_classes is available from mask selection
                            if tracks_with_classes is not None and 'class' in tracks_with_classes.columns:
                                # Get class names based on segmentation method
                                if segmentation_method == "Simple Segmentation (Binary)":
                                    class_names = {0: "Background", 1: "Nucleus"}
                                    mask_classes = [0, 1]
                                elif segmentation_method == "Two-Step Segmentation (3 Classes)":
                                    class_names = {0: "Background", 1: "Class 1", 2: "Class 2"}
                                    mask_classes = [0, 1, 2]
                                else:  # Nuclear Density Mapping
                                    # Dynamic class names based on actual classes in data
                                    unique_classes = sorted(tracks_with_classes['class'].unique())
                                    mask_classes = unique_classes
                                    class_names = {}
                                    for class_id in unique_classes:
                                        if int(class_id) == 0:
                                            class_names[class_id] = "Background"
                                        elif int(class_id) == 1:
                                            class_names[class_id] = "Low Density"
                                        elif int(class_id) == 2:
                                            class_names[class_id] = "High Density"
                                        else:
                                            class_names[class_id] = f"Class {int(class_id)}"
                                # Perform class-based analysis
                                diffusion_results = {}
                                
                                for class_id in mask_classes:
                                    class_tracks = tracks_with_classes[tracks_with_classes['class'] == class_id]
                                    
                                    if len(class_tracks) < min_track_length:
                                        st.warning(f"Insufficient tracks for {class_names.get(class_id, f'Class {class_id}')} (only {len(class_tracks)} points)")
                                        continue
                                    
                                    st.write(f"**Analyzing {class_names.get(class_id, f'Class {class_id}')}** ({len(class_tracks)} track points)")
                                    
                                    class_result = analyze_diffusion(
                                        class_tracks,
                                        max_lag=max_lag,
                                        pixel_size=pixel_size,
                                        frame_interval=frame_interval,
                                        min_track_length=min_track_length,
                                        fit_method=fit_method,
                                        analyze_anomalous=analyze_anomalous,
                                        check_confinement=check_confinement
                                    )
                                    
                                    # Extract result dict if success, otherwise store error info
                                    if class_result.get('success', False):
                                        diffusion_results[class_names.get(class_id, f'Class {class_id}')] = class_result['result']
                                        diffusion_results[class_names.get(class_id, f'Class {class_id}')]['success'] = True
                                    else:
                                        diffusion_results[class_names.get(class_id, f'Class {class_id}')] = {
                                            'success': False,
                                            'error': class_result.get('error', 'Unknown error')
                                        }
                                # Handle Two-Step Segmentation combined analysis if selected
                                if (segmentation_method == "Two-Step Segmentation (3 Classes)" and 
                                    analysis_option == "Analyze classes separately, then combine Class 1 and 2"):
                                    st.write("**Performing Combined Class 1+2 Analysis**")
                                    
                                    # Combine Class 1 and Class 2 tracks
                                    combined_tracks = tracks_with_classes[tracks_with_classes['class'].isin([1, 2])]
                                    
                                    if len(combined_tracks) >= min_track_length:
                                        combined_result = analyze_diffusion(
                                            combined_tracks,
                                            max_lag=max_lag,
                                            pixel_size=pixel_size,
                                            frame_interval=frame_interval,
                                            min_track_length=min_track_length,
                                            fit_method=fit_method,
                                            analyze_anomalous=analyze_anomalous,
                                            check_confinement=check_confinement
                                        )
                                        # Extract result dict if success
                                        if combined_result.get('success', False):
                                            diffusion_results["Combined Class 1+2"] = combined_result['result']
                                            diffusion_results["Combined Class 1+2"]['success'] = True
                                        else:
                                            diffusion_results["Combined Class 1+2"] = {
                                                'success': False,
                                                'error': combined_result.get('error', 'Unknown error')
                                            }
                                    else:
                                        st.warning("Insufficient tracks for combined Class 1+2 analysis")
                            else:
                                st.error("Track classification failed. Please select a segmentation method first.")
                                diffusion_results = {"error": "Classification failed"}
                        else:
                            st.error("Please select analysis type and configure segmentation if needed.")
                            diffusion_results = {"error": "Invalid configuration"}
                        
                        # Store results in session state
                        st.session_state.analysis_results["diffusion"] = diffusion_results
                        
                        # Create a record of this analysis
                        analysis_record = create_analysis_record(
                            name="Diffusion Analysis",
                            analysis_type="diffusion",
                            parameters={
                                "max_lag": max_lag,
                                "pixel_size": pixel_size,
                                "frame_interval": frame_interval,
                                "min_track_length": min_track_length,
                                "fit_method": fit_method,
                                "analyze_anomalous": analyze_anomalous,
                                "check_confinement": check_confinement
                            }
                        )
                        
                        # Add to recent analyses
                        st.session_state.recent_analyses.append(analysis_record)
                        
                        st.success("Diffusion analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running diffusion analysis: {str(e)}")
            
            # Display results if available
            if "diffusion" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["diffusion"]
                
                # Check if this is class-based analysis (dictionary of class results)
                if isinstance(results, dict) and any(key in ["Background", "Nucleus", "Class 1", "Class 2", "Combined Class 1+2", "Low Density", "High Density"] for key in results.keys()):
                    st.subheader("Class-Based Diffusion Analysis Results")
                    
                    # Display results for each class
                    for class_name, class_results in results.items():
                        st.write(f"### {class_name}")
                        
                        if "error" in class_results:
                            st.error(f"{class_name}: {class_results['error']}")
                            continue
                        
                        if class_results.get("success", False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if "diffusion_coefficient" in class_results:
                                    st.metric("Diffusion Coefficient", f"{class_results['diffusion_coefficient']:.4f}")
                                if "n_tracks" in class_results:
                                    st.metric("Number of Tracks", class_results["n_tracks"])
                            
                            with col2:
                                if "alpha" in class_results:
                                    st.metric("Anomalous Exponent (α)", f"{class_results['alpha']:.3f}")
                                if "confinement_radius" in class_results:
                                    st.metric("Confinement Radius", f"{class_results['confinement_radius']:.2f}")
                            
                            with col3:
                                if "fitting_quality" in class_results:
                                    st.metric("R² (Fit Quality)", f"{class_results['fitting_quality']:.3f}")
                            
                            # MSD plot for this class
                            if "msd_data" in class_results and not class_results["msd_data"].empty:
                                st.write(f"**{class_name} - Mean Squared Displacement**")
                                
                                fig = px.scatter(
                                    class_results["msd_data"], 
                                    x="lag_time", 
                                    y="msd",
                                    title=f"MSD vs Lag Time - {class_name}",
                                    labels={"lag_time": "Lag Time (s)", "msd": "MSD (μm²)"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("---")
                    
                    # Summary comparison between classes
                    st.subheader("Class Comparison")
                    comparison_data = []
                    for class_name, class_results in results.items():
                        if class_results.get("success", False) and "diffusion_coefficient" in class_results:
                            comparison_data.append({
                                "Class": class_name,
                                "Diffusion Coefficient": class_results["diffusion_coefficient"],
                                "Alpha (α)": class_results.get("alpha", "N/A"),
                                "Number of Tracks": class_results.get("n_tracks", "N/A"),
                                "R²": class_results.get("fitting_quality", "N/A")
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                        
                        # Bar chart comparison
                        fig = px.bar(
                            comparison_df, 
                            x="Class", 
                            y="Diffusion Coefficient",
                            title="Diffusion Coefficient by Class"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif results.get("success", False):
                    # Standard whole-image analysis results
                    st.subheader("Diffusion Analysis Results")
                    
                    # Extract result dict if it's nested
                    display_results = results.get('result', results) if 'result' in results else results
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Check ensemble_results first for mean values
                        if 'ensemble_results' in display_results and 'mean_diffusion_coefficient' in display_results['ensemble_results']:
                            st.metric("Diffusion Coefficient", f"{display_results['ensemble_results']['mean_diffusion_coefficient']:.4f}")
                        elif 'diffusion_coefficient' in display_results:
                            st.metric("Diffusion Coefficient", f"{display_results['diffusion_coefficient']:.4f}")
                        elif 'mean_diffusion_coefficient' in display_results:
                            st.metric("Diffusion Coefficient", f"{display_results['mean_diffusion_coefficient']:.4f}")
                        else:
                            st.metric("Diffusion Coefficient", "N/A")
                    with col2:
                        if 'ensemble_results' in display_results and 'mean_alpha' in display_results['ensemble_results']:
                            st.metric("Anomalous Exponent (α)", f"{display_results['ensemble_results']['mean_alpha']:.3f}")
                        elif 'alpha' in display_results:
                            st.metric("Anomalous Exponent (α)", f"{display_results['alpha']:.3f}")
                        else:
                            st.metric("Anomalous Exponent (α)", "N/A")
                    with col3:
                        if 'fitting_quality' in display_results:
                            st.metric("R² (Fit Quality)", f"{display_results['fitting_quality']:.3f}")
                        else:
                            st.metric("R² (Fit Quality)", "N/A")
                    
                    # MSD data
                    if "msd_data" in display_results and display_results["msd_data"] is not None:
                        st.subheader("Mean Squared Displacement")
                        st.dataframe(display_results["msd_data"].head())
                    
                    # Track results
                    if "track_results" in display_results and display_results["track_results"] is not None:
                        st.subheader("Diffusion Results by Track")
                        if isinstance(display_results["track_results"], pd.DataFrame) and not display_results["track_results"].empty:
                            st.dataframe(display_results["track_results"])
                        else:
                            st.info("No track-level results available")
                    
                    # Ensemble results
                    if "ensemble_results" in display_results and display_results["ensemble_results"]:
                        st.subheader("Ensemble Statistics")
                        for key, value in display_results["ensemble_results"].items():
                            st.text(f"{key}: {value}")
                    
                    # Display visualizations
                    if 'msd_data' in display_results and display_results['msd_data'] is not None:
                        if isinstance(display_results['msd_data'], pd.DataFrame) and not display_results['msd_data'].empty:
                            # Define plot_msd_curves if not already defined or import from your utilities
                            def plot_msd_curves(msd_data):
                                import plotly.express as px
                                if isinstance(msd_data, dict) and 'lag_time' in msd_data and 'msd' in msd_data:
                                    df = pd.DataFrame({'lag_time': msd_data['lag_time'], 'msd': msd_data['msd']})
                                elif isinstance(msd_data, pd.DataFrame):
                                    df = msd_data
                                else:
                                    return px.scatter(title="No MSD data available")
                                fig = px.scatter(df, x="lag_time", y="msd", title="MSD vs Lag Time", labels={"lag_time": "Lag Time (s)", "msd": "MSD (μm²)"})
                                fig.update_traces(mode='lines+markers')
                                return fig

                            msd_fig = plot_msd_curves(display_results['msd_data'])
                            st.plotly_chart(msd_fig, use_container_width=True)
                    
                    if 'track_results' in display_results and display_results['track_results'] is not None:
                        if isinstance(display_results['track_results'], pd.DataFrame) and not display_results['track_results'].empty:
                            diff_fig = plot_diffusion_coefficients(display_results)
                            st.plotly_chart(diff_fig, use_container_width=True)
                else:
                    st.warning(f"Analysis was not successful: {results.get('error', 'Unknown error')}")
            
            # Subpopulation Analysis Section
            elif analysis_type == "Subpopulation Analysis (by Cell)":
                st.subheader("🔬 Single-Cell Subpopulation Analysis")
                
                st.markdown("""
                **Detect heterogeneous subpopulations within your data:**
                - Identifies distinct subgroups based on tracking behavior
                - Useful for detecting cell cycle stages, metabolic states, or treatment responses
                - Requires cell identifiers in your tracking data
                """)
                
                # Check if cell_id column exists
                if 'cell_id' not in st.session_state.tracks_data.columns:
                    st.error("""
                    **Cell identifiers required!**
                    
                    Your tracking data must include a 'cell_id' column to perform subpopulation analysis.
                    Each track should be associated with a specific cell/nucleus.
                    
                    **Options:**
                    1. Load data with cell identifiers already included
                    2. Use segmentation masks to assign tracks to cells
                    3. Manually add cell_id column based on spatial proximity
                    """)
                    
                    # Offer to assign cell IDs based on spatial clustering
                    if st.checkbox("Auto-assign cell IDs based on spatial clustering"):
                        st.info("This will group nearby tracks into cells based on spatial proximity.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            clustering_radius = st.number_input(
                                "Clustering Radius (pixels)",
                                min_value=5.0,
                                max_value=200.0,
                                value=50.0,
                                help="Maximum distance for tracks to be considered part of the same cell"
                            )
                        with col2:
                            min_tracks_per_cell = st.number_input(
                                "Min Tracks per Cell",
                                min_value=1,
                                max_value=50,
                                value=5,
                                help="Minimum tracks required to form a cell group"
                            )
                        
                        if st.button("Assign Cell IDs"):
                            with st.spinner("Clustering tracks into cells..."):
                                try:
                                    from sklearn.cluster import DBSCAN
                                    
                                    # Get mean position per track
                                    track_positions = st.session_state.tracks_data.groupby('track_id')[['x', 'y']].mean()
                                    
                                    # Perform DBSCAN clustering
                                    clustering = DBSCAN(eps=clustering_radius, min_samples=min_tracks_per_cell)
                                    cell_labels = clustering.fit_predict(track_positions)
                                    
                                    # Map track_id to cell_id
                                    track_to_cell = dict(zip(track_positions.index, cell_labels))
                                    st.session_state.tracks_data['cell_id'] = st.session_state.tracks_data['track_id'].map(track_to_cell)
                                    
                                    # Filter out noise points (-1)
                                    st.session_state.tracks_data = st.session_state.tracks_data[st.session_state.tracks_data['cell_id'] != -1]
                                    
                                    n_cells = len(st.session_state.tracks_data['cell_id'].unique())
                                    st.success(f"✓ Assigned tracks to {n_cells} cells")
                                    st.rerun()
                                    
                                except ImportError:
                                    st.error("scikit-learn required for spatial clustering. Please install: pip install scikit-learn")
                                except Exception as e:
                                    st.error(f"Error assigning cell IDs: {str(e)}")
                else:
                    # Cell IDs are available - proceed with analysis
                    n_cells = st.session_state.tracks_data['cell_id'].nunique()
                    st.success(f"✓ Found {n_cells} cells in tracking data")
                    
                    # Configuration parameters
                    st.subheader("Analysis Configuration")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_tracks_per_cell = st.slider(
                            "Min Tracks/Cell",
                            min_value=5,
                            max_value=50,
                            value=10,
                            help="Minimum tracks required to include a cell in analysis"
                        )
                    
                    with col2:
                        clustering_method = st.selectbox(
                            "Clustering Method",
                            ["kmeans", "gmm", "hierarchical"],
                            help="Algorithm for detecting subpopulations"
                        )
                    
                    with col3:
                        max_clusters = st.slider(
                            "Max Clusters",
                            min_value=2,
                            max_value=6,
                            value=4,
                            help="Maximum subpopulations to test"
                        )
                    
                    # Group selection (if multiple groups/conditions exist)
                    st.subheader("Group Selection")
                    
                    if 'group' in st.session_state.tracks_data.columns or 'condition' in st.session_state.tracks_data.columns:
                        group_column = 'group' if 'group' in st.session_state.tracks_data.columns else 'condition'
                        groups = st.session_state.tracks_data[group_column].unique()
                        
                        analyze_all_groups = st.checkbox("Analyze all groups separately", value=True)
                        
                        if analyze_all_groups:
                            selected_groups = groups.tolist()
                        else:
                            selected_groups = st.multiselect(
                                "Select Groups to Analyze",
                                groups.tolist(),
                                default=[groups[0]] if len(groups) > 0 else []
                            )
                    else:
                        st.info("No group/condition column found. Analyzing all data as one group.")
                        selected_groups = ["All Data"]
                        group_column = None
                    
                    # Run analysis
                    if st.button("🔬 Detect Subpopulations", type="primary"):
                        if not selected_groups:
                            st.warning("Please select at least one group to analyze.")
                        else:
                            with st.spinner("Analyzing single-cell heterogeneity..."):
                                try:
                                    from subpopulation_analysis import (
                                        SubpopulationAnalyzer,
                                        SubpopulationConfig,
                                        create_subpopulation_visualizations
                                    )
                                    
                                    # Initialize analyzer
                                    config = SubpopulationConfig(
                                        min_tracks_per_cell=min_tracks_per_cell,
                                        clustering_methods=[clustering_method],
                                        n_clusters_range=(2, max_clusters),
                                        use_pca=True
                                    )
                                    
                                    analyzer = SubpopulationAnalyzer(config)
                                    
                                    # Analyze each group
                                    all_results = {}
                                    
                                    for group_name in selected_groups:
                                        st.write(f"### Analyzing: {group_name}")
                                        
                                        # Filter data for this group
                                        if group_column is not None:
                                            group_data = st.session_state.tracks_data[
                                                st.session_state.tracks_data[group_column] == group_name
                                            ].copy()
                                        else:
                                            group_data = st.session_state.tracks_data.copy()
                                        
                                        # Run analysis
                                        result = analyzer.analyze_group(group_data, group_name, cell_id_column='cell_id')
                                        all_results[group_name] = result
                                        
                                        if result['success']:
                                            if result['subpopulations_detected']:
                                                st.success(f"✓ Detected {result['n_subpopulations']} subpopulations in {group_name}")
                                            else:
                                                st.info(f"No distinct subpopulations detected in {group_name} (appears homogeneous)")
                                        else:
                                            st.error(f"Analysis failed for {group_name}: {result.get('error', 'Unknown error')}")
                                    
                                    # Store results
                                    st.session_state.analysis_results['subpopulation'] = all_results
                                    
                                    # Generate comparison
                                    comparison = analyzer.compare_groups_with_subpopulations(all_results)
                                    st.session_state.analysis_results['subpopulation_comparison'] = comparison
                                    
                                    st.success("✓ Subpopulation analysis complete!")
                                    
                                except ImportError as e:
                                    st.error(f"Required module not available: {str(e)}")
                                    st.info("Make sure scikit-learn is installed: pip install scikit-learn scipy")
                                except Exception as e:
                                    st.error(f"Error in subpopulation analysis: {str(e)}")
                                    import traceback
                                    with st.expander("Error Details"):
                                        st.code(traceback.format_exc())
                    
                    # Display results if available
                    if 'subpopulation' in st.session_state.get('analysis_results', {}):
                        st.divider()
                        st.header("📊 Subpopulation Analysis Results")
                        
                        results = st.session_state.analysis_results['subpopulation']
                        comparison = st.session_state.analysis_results.get('subpopulation_comparison', {})
                        
                        # Summary
                        if comparison:
                            st.subheader("Summary")
                            summary = comparison.get('summary', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Groups", summary.get('total_groups', 0))
                            with col2:
                                st.metric("Heterogeneous", summary.get('heterogeneous_groups', 0))
                            with col3:
                                st.metric("Homogeneous", summary.get('homogeneous_groups', 0))
                            
                            st.info(f"**Recommendation:** {summary.get('recommendation', 'N/A')}")
                        
                        # Detailed results per group
                        for group_name, group_result in results.items():
                            st.divider()
                            st.subheader(f"📍 {group_name}")
                            
                            if not group_result.get('success'):
                                st.error(f"Analysis failed: {group_result.get('error', 'Unknown error')}")
                                continue
                            
                            # Show metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Cells", group_result.get('n_cells_total', 0))
                            with col2:
                                st.metric("Subpopulations", group_result.get('n_subpopulations', 0))
                            with col3:
                                st.metric("Method", group_result.get('clustering_method', 'N/A'))
                            with col4:
                                if 'pca_explained_variance' in group_result:
                                    st.metric("PCA Variance", f"{group_result['pca_explained_variance']:.1%}")
                            
                            if group_result.get('subpopulations_detected'):
                                # Show subpopulation characteristics
                                st.write("**Subpopulation Characteristics:**")
                                
                                subpop_chars = group_result.get('subpopulation_characteristics', {})
                                
                                # Create comparison table
                                comparison_data = []
                                for subpop_name, chars in subpop_chars.items():
                                    row = {
                                        'Subpopulation': f"Subpop {chars['subpopulation_id']}",
                                        'N Cells': chars['n_cells'],
                                        'Fraction': f"{chars['fraction_of_total']:.1%}"
                                    }
                                    
                                    # Add key features
                                    feature_means = chars.get('feature_means', {})
                                    for feature, value in list(feature_means.items())[:3]:  # Top 3 features
                                        row[feature.replace('_', ' ').title()] = f"{value:.4f}"
                                    
                                    comparison_data.append(row)
                                
                                if comparison_data:
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df, use_container_width=True)
                                
                                # Visualizations
                                try:
                                    from subpopulation_analysis import create_subpopulation_visualizations
                                    
                                    figs = create_subpopulation_visualizations(group_result, group_name)
                                    
                                    if figs:
                                        with st.expander(f"📈 Visualizations for {group_name}", expanded=True):
                                            for fig_name, fig in figs.items():
                                                if fig is not None and hasattr(fig, 'update_layout'):
                                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not generate visualizations: {str(e)}")
                                
                                # Download cell-level data with subpopulation assignments
                                if 'cell_level_data' in group_result:
                                    cell_data = group_result['cell_level_data']
                                    csv = cell_data.to_csv(index=False)
                                    
                                    st.download_button(
                                        label=f"📥 Download {group_name} Cell Data with Assignments",
                                        data=csv,
                                        file_name=f"{group_name}_subpopulations.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.info(f"{group_name} appears homogeneous - no distinct subpopulations detected.")
        
        # Motion Analysis tab
        with tabs[2]:
            st.header("Motion Analysis")
            
            # Parameters for motion analysis
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                window_size = st.slider(
                    "Window Size (frames)",
                    min_value=3,
                    max_value=20,
                    value=5,
                    help="Window size for calculating local properties"
                )
                
                # Use current unit settings as default values
                units = get_current_units()
                
                pixel_size = st.number_input(
                    "Pixel Size (µm)",
                    min_value=0.01,
                    max_value=10.0,
                    value=units['pixel_size'],
                    step=0.01,
                    key="motion_pixel_size",
                    help="Pixel size in micrometers"
                )
                
                frame_interval = st.number_input(
                    "Frame Interval (s)",
                    min_value=0.001,
                    max_value=60.0,
                    value=units['frame_interval'],
                    step=0.001,
                    key="motion_frame_interval",
                    help="Time between frames in seconds"
                )
                
            with col2:
                min_track_length = st.slider(
                    "Minimum Track Length",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="motion_min_track_length",
                    help="Minimum track length to include in analysis"
                )
                
                motion_classification = st.selectbox(
                    "Motion Classification",
                    ["none", "basic", "advanced"],
                    help="Method for classifying motion"
                )
                
                analysis_options = st.multiselect(
                    "Analysis Options",
                    ["Velocity Autocorrelation", "Directional Persistence"],
                    default=["Velocity Autocorrelation", "Directional Persistence"],
                    help="Additional analysis options"
                )
            
            # Run analysis on button click
            if st.button("Run Motion Analysis"):
                with st.spinner("Running motion analysis..."):
                    try:
                        analyze_velocity_autocorr = "Velocity Autocorrelation" in analysis_options
                        analyze_persistence = "Directional Persistence" in analysis_options
                        
                        motion_results = analyze_motion(
                            st.session_state.tracks_data,
                            window_size=window_size,
                            analyze_velocity_autocorr=analyze_velocity_autocorr,
                            analyze_persistence=analyze_persistence,
                            motion_classification=motion_classification,
                            min_track_length=min_track_length,
                            pixel_size=pixel_size,
                            frame_interval=frame_interval
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results["motion"] = motion_results
                        
                        # Create a record of this analysis
                        analysis_record = create_analysis_record(
                            name="Motion Analysis",
                            analysis_type="motion",
                            parameters={
                                "window_size": window_size,
                                "pixel_size": pixel_size,
                                "frame_interval": frame_interval,
                                "min_track_length": min_track_length,
                                "motion_classification": motion_classification,
                                "analyze_velocity_autocorr": analyze_velocity_autocorr,
                                "analyze_persistence": analyze_persistence
                            }
                        )
                        
                        # Add to recent analyses
                        st.session_state.recent_analyses.append(analysis_record)
                        
                        st.success("Motion analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running motion analysis: {str(e)}")
            
            # Display results if available
            if "motion" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["motion"]
                
                # Track results
                st.subheader("Motion Analysis Results")
                if 'track_results' in results:
                    st.dataframe(results['track_results'])
                    
                    # Generate motion analysis visualizations
                    try:
                        motion_vis = plot_motion_analysis(results)
                        # plot_motion_analysis may return a Matplotlib Figure or a dict of Plotly figures
                        from matplotlib.figure import Figure as MplFigure
                        if isinstance(motion_vis, MplFigure):
                            st.subheader("Motion Analysis Visualization")
                            st.pyplot(motion_vis, use_container_width=True)
                        elif isinstance(motion_vis, dict):
                            if motion_vis and not motion_vis.get("empty"):
                                st.subheader("Motion Analysis Visualizations")
                                for plot_name, fig in motion_vis.items():
                                    if fig and plot_name != "empty":
                                        st.subheader(plot_name.replace("_", " ").title())
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No visualization data available for motion analysis.")
                        else:
                            st.info("No visualization data available for motion analysis.")
                    except Exception as e:
                        st.warning(f"Could not generate motion visualizations: {str(e)}")
                else:
                    st.warning("No results available to display.")
        
        # Clustering Analysis tab
        with tabs[3]:
            st.header("Clustering Analysis")
            
            # Parameters for clustering analysis
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                clustering_method = st.selectbox(
                    "Clustering Method",
                    ["DBSCAN", "OPTICS", "Hierarchical", "Density-based"],
                    help="Method for clustering particles"
                )
                
                epsilon = st.slider(
                    "Epsilon (µm)",
                    min_value=0.1,
                    max_value=10.0,
                    value=0.5,
                    step=0.1,
                    help="Maximum distance between points in a cluster"
                )
                
                min_samples = st.slider(
                    "Minimum Samples",
                    min_value=2,
                    max_value=20,
                    value=3,
                    help="Minimum number of points to form a cluster"
                )
                
            with col2:
                # Use current unit settings as default values
                units = get_current_units()
                
                pixel_size = st.number_input(
                    "Pixel Size (µm)",
                    min_value=0.01,
                    max_value=10.0,
                    value=units['pixel_size'],
                    step=0.01,
                    key="clustering_pixel_size",
                    help="Pixel size in micrometers"
                )
                
                analysis_options = st.multiselect(
                    "Analysis Options",
                    ["Track Clusters", "Analyze Dynamics"],
                    default=["Track Clusters"],
                    help="Additional analysis options"
                )
            
            # Run analysis on button click
            if st.button("Run Clustering Analysis"):
                with st.spinner("Running clustering analysis..."):
                    try:
                        track_clusters = "Track Clusters" in analysis_options
                        analyze_dynamics = "Analyze Dynamics" in analysis_options
                        
                        clustering_results = analyze_clustering(
                            st.session_state.tracks_data,
                            method=clustering_method,
                            epsilon=epsilon,
                            min_samples=min_samples,
                            track_clusters=track_clusters,
                            analyze_dynamics=analyze_dynamics,
                            pixel_size=pixel_size
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results["clustering"] = clustering_results
                        
                        # Create a record of this analysis
                        analysis_record = create_analysis_record(
                            name="Clustering Analysis",
                            analysis_type="clustering",
                            parameters={
                                "method": clustering_method,
                                "epsilon": epsilon,
                                "min_samples": min_samples,
                                "track_clusters": track_clusters,
                                "analyze_dynamics": analyze_dynamics,
                                "pixel_size": pixel_size
                            }
                        )
                        
                        # Add to recent analyses
                        st.session_state.recent_analyses.append(analysis_record)
                        
                        st.success("Clustering analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running clustering analysis: {str(e)}")
            
            # Display results if available
            if "clustering" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["clustering"]
                
                # Cluster statistics
                st.subheader("Cluster Statistics")
                if 'frame_results' in results and results['frame_results']:
                    # Create a dataframe from the cluster stats in the first frame with clusters
                    for frame_result in results['frame_results']:
                        if isinstance(frame_result, dict) and 'cluster_stats' in frame_result and not frame_result['cluster_stats'].empty:
                            st.dataframe(frame_result['cluster_stats'])
                            break
                    else:
                        # If no frame has cluster stats
                        st.warning("No cluster statistics available to display.")
                    
                    # Show ensemble statistics
                    if 'ensemble_results' in results:
                        st.subheader("Ensemble Statistics")
                        for key, value in results['ensemble_results'].items():
                            st.text(f"{key}: {value}")
                            
                    # Display clustering visualizations
                    if 'spatial_clusters' in results:
                        # Define a simple plot_spatial_clustering function if not already defined
                        def plot_spatial_clustering(results):
                            import plotly.express as px
                            if 'cluster_tracks' in results and not results['cluster_tracks'].empty:
                                df = results['cluster_tracks']
                                fig = px.scatter(
                                    df, x='x', y='y', color='cluster_id',
                                    title="Spatial Clustering of Tracks",
                                    labels={'x': 'X Position (µm)', 'y': 'Y Position (µm)', 'cluster_id': 'Cluster'}
                                )
                                fig.update_yaxes(autorange="reversed")
                                return fig
                            else:
                                return px.scatter(title="No clustering data available")
                        spatial_fig = plot_spatial_clustering(results) if 'clustering_results' in results else st.warning('No clustering results available')
                        st.plotly_chart(spatial_fig, use_container_width=True)
                    
                    if 'cluster_summary' in results:
                        st.subheader("Cluster Summary")
                        st.dataframe(results['cluster_summary'])
                else:
                    st.warning("No cluster statistics available to display.")
                    
                # Points with cluster assignments
                st.subheader("Cluster Assignments")
                if 'cluster_tracks' in results and not results['cluster_tracks'].empty:
                    st.dataframe(results['cluster_tracks'].head(20))
                else:
                    st.warning("No cluster assignments available to display.")

        # Dwell Time Analysis tab
        with tabs[4]:
            st.header("Dwell Time Analysis")
            
            # Debug info about loaded data
            if 'tracks_data' in st.session_state and st.session_state.tracks_data is not None:
                with st.expander("ℹ️ Data Information", expanded=False):
                    st.write(f"**Data shape:** {st.session_state.tracks_data.shape}")
                    st.write(f"**Columns:** {', '.join(st.session_state.tracks_data.columns.tolist())}")
                    st.write(f"**Number of unique tracks:** {st.session_state.tracks_data['track_id'].nunique() if 'track_id' in st.session_state.tracks_data.columns else 'N/A'}")
            
            # Check if the loaded data contains pre-calculated dwell events
            has_precalculated = False
            if 'tracks_data' in st.session_state and st.session_state.tracks_data is not None:
                dwell_cols = ['dwell_time', 'dwell_frames', 'start_frame', 'end_frame']
                available_dwell_cols = [col for col in dwell_cols if col in st.session_state.tracks_data.columns]
                
                if len(available_dwell_cols) >= 2:  # Has at least 2 dwell-related columns
                    has_precalculated = True
                    st.success("📊 Detected pre-calculated dwell event data in your file!")
                    st.write(f"**Available dwell columns:** {', '.join(available_dwell_cols)}")
                    
                    analysis_mode = st.radio(
                        "Analysis Mode:",
                        ["Use Pre-calculated Data", "Re-analyze from Scratch"],
                        help="Pre-calculated: Use existing dwell statistics from your data. Re-analyze: Detect dwell events from scratch.",
                        key="dwell_analysis_mode"
                    )
                    
                    if analysis_mode == "Use Pre-calculated Data":
                        # Add button to load pre-calculated dwell events
                        if st.button("Load Pre-calculated Dwell Events"):
                            with st.spinner("Loading dwell event statistics..."):
                                try:
                                    frame_interval = st.session_state.get('frame_interval', 0.1)
                                    dwell_results = load_precalculated_dwell_events(
                                        st.session_state.tracks_data,
                                        frame_interval=frame_interval
                                    )
                                    
                                    if dwell_results.get('success'):
                                        st.session_state.analysis_results["dwell_time"] = dwell_results
                                        st.success(f"✅ Loaded {dwell_results['ensemble_results']['n_dwell_events']} dwell events from {dwell_results['ensemble_results']['n_tracks_analyzed']} tracks!")
                                    else:
                                        st.error(f"Failed to load: {dwell_results.get('error', 'Unknown error')}")
                                except Exception as e:
                                    st.error(f"Error loading pre-calculated dwell events: {str(e)}")
                        
                        # Display results if loaded
                        if "dwell_time" in st.session_state.analysis_results:
                            results = st.session_state.analysis_results["dwell_time"]
                            
                            # Show statistics
                            st.subheader("Dwell Statistics")
                            if 'dwell_stats' in results:
                                for key, value in results['dwell_stats'].items():
                                    st.text(f"{key}: {value}")
                            
                            # Show dwell events table
                            st.subheader("Dwell Events")
                            if 'dwell_events' in results and not results['dwell_events'].empty:
                                st.dataframe(results['dwell_events'], use_container_width=True)
                            
                            # Show track results
                            if 'track_results' in results and not results['track_results'].empty:
                                st.subheader("Track Statistics")
                                st.dataframe(results['track_results'], use_container_width=True)
                            
                            # Show region stats if available
                            if 'region_stats' in results:
                                st.subheader("Region Statistics")
                                st.dataframe(results['region_stats'], use_container_width=True)
                        
                        st.stop()  # Skip the rest of the standard analysis UI
                else:
                    # No pre-calculated data detected
                    st.warning("⚠️ No pre-calculated dwell event data detected in your file.")
                    st.info("Your data needs columns like 'dwell_time', 'dwell_frames', 'start_frame', 'end_frame' to use pre-calculated mode. Otherwise, use the standard analysis below to detect dwell events from x, y coordinates.")
            
            # Parameters for dwell time analysis
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                threshold_distance = st.slider(
                    "Threshold Distance (µm)",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    help="Maximum distance to consider a particle as dwelling"
                )
                
                min_dwell_frames = st.slider(
                    "Minimum Dwell Frames",
                    min_value=2,
                    max_value=20,
                    value=3,
                    help="Minimum number of frames to consider a dwell event"
                )
                
            with col2:
                # Use current unit settings as default values
                units = get_current_units()
                
                # Auto-detect if data is already in micrometers
                data_in_micrometers = False
                if st.session_state.tracks_data is not None:
                    try:
                        if isinstance(st.session_state.tracks_data, pd.DataFrame) and 'x' in st.session_state.tracks_data.columns and 'y' in st.session_state.tracks_data.columns:
                            # Check if data appears to be in micrometers (typical range 0-100 µm)
                            x_range = st.session_state.tracks_data['x'].max() - st.session_state.tracks_data['x'].min()
                            y_range = st.session_state.tracks_data['y'].max() - st.session_state.tracks_data['y'].min()
                            if x_range < 1000 and y_range < 1000:  # Likely already in micrometers
                                data_in_micrometers = True
                    except (KeyError, TypeError, AttributeError):
                        pass  # Skip micrometers detection if data format is invalid
                
                if data_in_micrometers:
                    st.info("📏 Data appears to already be in micrometers. Setting pixel size to 1.0.")
                    pixel_size = 1.0
                else:
                    pixel_size = st.number_input(
                        "Pixel Size (µm)",
                        min_value=0.01,
                        max_value=10.0,
                        value=units['pixel_size'],
                        step=0.01,
                        key="dwell_pixel_size",
                        help="Pixel size in micrometers. Set to 1.0 if data is already in µm."
                    )
                
                frame_interval = st.number_input(
                    "Frame Interval (s)",
                    min_value=0.001,
                    max_value=60.0,
                    value=units['frame_interval'],
                    step=0.001,
                    key="dwell_frame_interval",
                    help="Time between frames in seconds"
                )
                
                use_regions = st.checkbox(
                    "Define Regions of Interest",
                    value=False,
                    help="Define specific regions for dwell time analysis"
                )
            
            # Region definition (if enabled)
            regions = None
            if use_regions:
                st.subheader("Regions of Interest")
                
                num_regions = st.slider(
                    "Number of Regions",
                    min_value=1,
                    max_value=10,
                    value=1
                )
                
                regions = []
                for i in range(num_regions):
                    with st.expander(f"Region {i+1}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x = st.number_input(f"X center (Region {i+1})", value=0.0)
                        with col2:
                            y = st.number_input(f"Y center (Region {i+1})", value=0.0)
                        with col3:
                            radius = st.number_input(f"Radius (Region {i+1})", value=1.0, min_value=0.1)
                        
                        regions.append({'x': x, 'y': y, 'radius': radius})
            
            # Run analysis on button click
            if st.button("Run Dwell Time Analysis"):
                with st.spinner("Running dwell time analysis..."):
                    try:
                        dwell_results = analyze_dwell_time(
                            st.session_state.tracks_data,
                            regions=regions,
                            threshold_distance=threshold_distance,
                            min_dwell_frames=min_dwell_frames,
                            pixel_size=pixel_size,
                            frame_interval=frame_interval
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results["dwell_time"] = dwell_results
                        
                        # Create a record of this analysis
                        analysis_record = create_analysis_record(
                            name="Dwell Time Analysis",
                            analysis_type="dwell_time",
                            parameters={
                                "threshold_distance": threshold_distance,
                                "min_dwell_frames": min_dwell_frames,
                                "pixel_size": pixel_size,
                                "frame_interval": frame_interval,
                                "use_regions": use_regions,
                                "num_regions": len(regions) if regions else 0
                            }
                        )
                        
                        # Add to recent analyses
                        st.session_state.recent_analyses.append(analysis_record)
                        
                        st.success("Dwell time analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running dwell time analysis: {str(e)}")
            
            # Display results if available
            if "dwell_time" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["dwell_time"]
                
                # Dwell events
                st.subheader("Dwell Events")
                if 'dwell_events' in results:
                    st.dataframe(results['dwell_events'])
                    
                    # Display dwell time visualizations
                    if 'dwell_times' in results:
                        def plot_dwell_time_analysis(results):
                            import plotly.express as px
                            if 'dwell_times' in results and isinstance(results['dwell_times'], dict) and results['dwell_times']:
                                dwell_data = []
                                for class_val, stats in results['dwell_times'].items():
                                    dwell_data.append({
                                        'Class': int(class_val),
                                        'Mean Dwell Time (frames)': stats['mean'],
                                        'Std Dev (frames)': stats['std'],
                                        'Min (frames)': stats['min'],
                                        'Max (frames)': stats['max'],
                                        'Count': stats['count']
                                    })
                                df = pd.DataFrame(dwell_data)
                                fig = px.bar(df, x='Class', y='Mean Dwell Time (frames)', error_y='Std Dev (frames)', title='Mean Dwell Time by Class')
                                return fig
                            else:
                                return px.scatter(title="No dwell time data available")
                        dwell_fig = plot_dwell_time_analysis(results)
                        st.plotly_chart(dwell_fig, use_container_width=True)
                    
                    if 'region_stats' in results:
                        st.subheader("Region Statistics")
                        st.dataframe(results['region_stats'])
                else:
                    st.warning("No dwell events available to display.")
                    
                # Dwell statistics
                st.subheader("Dwell Statistics")
                if 'dwell_stats' in results and results['dwell_stats']:
                    for key, value in results['dwell_stats'].items():
                        st.text(f"{key}: {value}")
                else:
                    # Display default statistics when no dwell events are detected
                    st.text("Total tracks analyzed: " + str(len(st.session_state.tracks_data['track_id'].unique())))
                    st.text("Tracks with dwell events: 0")
                    st.text("Total dwell events: 0")
                    st.text("Mean dwell time: 0.0 s")
                    st.text("Median dwell time: 0.0 s")
                    st.info("No dwell events were detected with the current parameters. Try adjusting the threshold distance or minimum dwell frames.")
        
        # Boundary Crossing Analysis tab
        with tabs[5]:
            st.header("Boundary Crossing Analysis")

            # Run boundary crossing analysis
            if st.button("Analyze Boundary Crossings"):
                with st.spinner("Analyzing boundary crossings..."):
                    try:
                        # Add class information to tracks
                        tracks_with_classes = apply_mask_to_tracks(
                            st.session_state.tracks_data,
                            selected_mask,
                            []  # Don't filter, get all classes
                        )
            
                        # Filter by minimum track length
                        track_lengths = tracks_with_classes.groupby('track_id').size()
                        valid_tracks = track_lengths[track_lengths >= min_track_length].index
                        filtered_tracks = tracks_with_classes[tracks_with_classes['track_id'].isin(valid_tracks)]

                        # --- FIX STARTS HERE ---
                        # Convert the selected segmentation mask into the correct boundary format
                        from segmentation import convert_compartments_to_boundary_crossing_format
            
                        # Create a list of dictionaries for each compartment region
                        compartments_for_conversion = []
                        for class_id in filtered_tracks['class'].unique():
                            if class_id != 'none':
                                # Get the mask data properly
                                if selected_mask and selected_mask[0] in st.session_state.available_masks:
                                    mask_data = st.session_state.available_masks[selected_mask[0]]
                                    comp_mask = (mask_data == class_id)
                                    props = measure.regionprops(comp_mask.astype(int))
                                else:
                                    props = []
                    
                                # Process ALL regions for this class, not just the first one
                                for i, prop in enumerate(props):
                                    min_row, min_col, max_row, max_col = prop.bbox
                                    compartments_for_conversion.append({
                                        'id': f'class_{class_id}_region_{i}',
                                        'bbox_um': {
                                            'x1': min_col * get_global_pixel_size(),
                                            'y1': min_row * get_global_pixel_size(),
                                            'x2': max_col * get_global_pixel_size(),
                                            'y2': max_row * get_global_pixel_size()
                                         }
                                    })
            
                        boundaries = convert_compartments_to_boundary_crossing_format(compartments_for_conversion)
                        # --- FIX ENDS HERE ---

                        # Analyze boundary crossings using the correctly formatted boundaries
                        boundary_stats = analyze_boundary_crossing(
                            filtered_tracks,
                            boundaries=boundaries,  # Pass the corrected boundaries
                            pixel_size=get_global_pixel_size(),
                            frame_interval=get_global_frame_interval(),
                            min_track_length=min_track_length
                        )
            
                        if "error" not in boundary_stats:
                            st.session_state.analysis_results["boundary_crossing"] = boundary_stats
                            st.success("Boundary crossing analysis completed!")
                        else:
                            st.error(boundary_stats["error"])
        
                    except Exception as e:
                        st.error(f"Error during boundary crossing analysis: {str(e)}")
                
            # Display results if available
            if "boundary_crossing" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["boundary_crossing"]
                
                # Overview statistics
                st.subheader("Crossing Statistics Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Tracks", results["total_tracks"])
                
                with col2:
                    st.metric("Tracks with Crossings", results["tracks_with_crossings"])
                
                with col3:
                    st.metric("Total Crossings", results["total_crossings"])
                
                with col4:
                    crossing_rate = (results["tracks_with_crossings"] / results["total_tracks"] * 100) if results["total_tracks"] > 0 else 0
                    st.metric("Crossing Rate (%)", f"{crossing_rate:.1f}")
                
                # Class transition matrix
                st.subheader("Class Transition Analysis")
                
                if results["class_transitions"]:
                    # Create transition dataframe
                    transitions_data = []
                    for transition, count in results["class_transitions"].items():
                        from_class, to_class = transition.split('->')
                        transitions_data.append({
                            'From Class': int(from_class),
                            'To Class': int(to_class),
                            'Count': count,
                            'Transition': transition
                        })
                    
                    transitions_df = pd.DataFrame(transitions_data)
                    st.dataframe(transitions_df)
                    
                    # Transition visualization
                    if len(transitions_df) > 0:
                        fig = px.bar(transitions_df, 
                                    x='Transition', 
                                    y='Count',
                                    title='Class Transition Frequency',
                                    labels={'Count': 'Number of Transitions'})
                        fig.update_xaxes(title='Transition Type')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No class transitions detected.")
                
                # Dwell time analysis
                st.subheader("Dwell Time Analysis")
                
                if results["dwell_times"]:
                    dwell_summary = []
                    for class_val, stats in results["dwell_times"].items():
                        dwell_summary.append({
                            'Class': int(class_val),
                            'Mean Dwell Time (frames)': f"{stats['mean']:.2f}",
                            'Std Dev (frames)': f"{stats['std']:.2f}",
                            'Min (frames)': stats['min'],
                            'Max (frames)': stats['max'],
                            'Count': stats['count']
                        })
                    
                    dwell_df = pd.DataFrame(dwell_summary)
                    st.dataframe(dwell_df)
                    
                    # Convert to time units if frame interval is available
                    if frame_interval > 0:
                        st.subheader("Dwell Times in Time Units")
                        time_dwell_summary = []
                        for class_val, stats in results["dwell_times"].items():
                            time_dwell_summary.append({
                                'Class': int(class_val),
                                'Mean Dwell Time (s)': f"{stats['mean'] * frame_interval:.3f}",
                                'Std Dev (s)': f"{stats['std'] * frame_interval:.3f}",
                                'Min (s)': f"{stats['min'] * frame_interval:.3f}",
                                'Max (s)': f"{stats['max'] * frame_interval:.3f}",
                                'Count': stats['count']
                            })
                        
                        time_dwell_df = pd.DataFrame(time_dwell_summary)
                        st.dataframe(time_dwell_df)
                else:
                    st.info("No dwell time data available.")
                
                # Individual crossing tracks
                st.subheader("Tracks with Boundary Crossings")
                
                if results["crossing_tracks"]:
                    # Show summary of crossing tracks
                    crossing_summary = []
                    for track_info in results["crossing_tracks"]:
                        crossing_summary.append({
                            'Track ID': track_info['track_id'],
                            'Number of Crossings': track_info['num_crossings'],
                            'Crossing Details': f"{len(track_info['crossings'])} transitions"
                        })
                    
                    crossing_summary_df = pd.DataFrame(crossing_summary)
                    st.dataframe(crossing_summary_df)
                    
                    # Option to view detailed crossing information
                    if st.checkbox("Show Detailed Crossing Information"):
                        selected_track = st.selectbox(
                            "Select track to view details:",
                            options=[track['track_id'] for track in results["crossing_tracks"]]
                        )
                        
                        if selected_track:
                            track_info = next(t for t in results["crossing_tracks"] if t['track_id'] == selected_track)
                            
                            st.write(f"**Track {selected_track} Crossing Details:**")
                            
                            crossing_details = []
                            for crossing in track_info['crossings']:
                                crossing_details.append({
                                    'Frame': crossing['frame'],
                                    'From Class': crossing['from_class'],
                                    'To Class': crossing['to_class'],
                                    'Position (X, Y)': f"({crossing['position'][0]:.1f}, {crossing['position'][1]:.1f})"
                                })
                            
                            crossing_details_df = pd.DataFrame(crossing_details)
                            st.dataframe(crossing_details_df)
                else:
                    st.info("No tracks with boundary crossings found.")
            
            else:
                st.info("Please select a segmentation mask from the Image Processing tab to perform boundary crossing analysis.")
        
        # Multi-Channel Analysis tab  
        with tabs[6]:
            st.header("Multi-Channel Analysis")
            
            # Check if multi-channel analysis is available
            if CORRELATIVE_ANALYSIS_AVAILABLE:
                # Check for existing intensity channels in primary data
                existing_channels = None
                if st.session_state.tracks_data is not None:
                    from intensity_analysis import extract_intensity_channels
                    existing_channels = extract_intensity_channels(st.session_state.tracks_data)
                
                # Display detected channels
                if existing_channels and len(existing_channels) > 0:
                    st.success(f"📊 Detected {len(existing_channels)} intensity channel(s) in your tracking data!")
                    with st.expander("ℹ️ Available Intensity Channels", expanded=False):
                        for ch_name, ch_cols in existing_channels.items():
                            st.write(f"**{ch_name.upper()}**: {', '.join(ch_cols)}")
                
                # Data source selection
                st.subheader("Secondary Channel Data Source")
                
                if existing_channels and len(existing_channels) >= 2:
                    data_source = st.radio(
                        "Choose data source for secondary channel:",
                        ["Use Existing Intensity Data", "Upload Separate Tracking File"],
                        help="Use intensity columns already in your data, or upload a separate file"
                    )
                else:
                    st.info("💡 Your tracking data contains intensity information for multiple channels. You can analyze channel interactions without uploading additional files!")
                    data_source = "Upload Separate Tracking File"
                
                # Handle existing intensity data
                if existing_channels and len(existing_channels) >= 2 and data_source == "Use Existing Intensity Data":
                    st.subheader("Select Channels for Analysis")
                    
                    channel_list = list(existing_channels.keys())
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        primary_channel = st.selectbox(
                            "Primary Channel",
                            options=channel_list,
                            index=0,
                            key="primary_channel_select"
                        )
                        primary_channel_name = st.text_input("Primary Channel Name", value=primary_channel.upper(), key="primary_ch_name_existing")
                        primary_color = st.color_picker("Primary Channel Color", value="#FF4B4B", key="primary_ch_color_existing")
                    
                    with col2:
                        # Filter out primary channel from secondary options
                        secondary_options = [ch for ch in channel_list if ch != primary_channel]
                        if secondary_options:
                            secondary_channel = st.selectbox(
                                "Secondary Channel",
                                options=secondary_options,
                                index=0,
                                key="secondary_channel_select"
                            )
                            secondary_channel_name = st.text_input("Secondary Channel Name", value=secondary_channel.upper(), key="secondary_ch_name_existing")
                            secondary_color = st.color_picker("Secondary Channel Color", value="#4B70FF", key="secondary_ch_color_existing")
                        else:
                            st.warning("Need at least 2 channels for multi-channel analysis")
                            secondary_channel = None
                    
                    # Select intensity columns to use
                    if secondary_channel:
                        st.subheader("Select Intensity Metrics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            primary_intensity_col = st.selectbox(
                                f"Intensity metric for {primary_channel_name}",
                                options=existing_channels[primary_channel],
                                key="primary_intensity_metric"
                            )
                        
                        with col2:
                            secondary_intensity_col = st.selectbox(
                                f"Intensity metric for {secondary_channel_name}",
                                options=existing_channels[secondary_channel],
                                key="secondary_intensity_metric"
                            )
                        
                        # Create virtual "channel2_data" using intensity columns from primary data
                        # This allows the existing analysis code to work without modification
                        channel2_data_virtual = st.session_state.tracks_data[['track_id', 'frame', 'x', 'y']].copy()
                        if 'z' in st.session_state.tracks_data.columns:
                            channel2_data_virtual['z'] = st.session_state.tracks_data['z']
                        
                        # Add intensity as the "primary" metric for this virtual channel
                        channel2_data_virtual['intensity'] = st.session_state.tracks_data[secondary_intensity_col]
                        
                        # Also add intensity to primary data for correlation
                        tracks_with_intensity = st.session_state.tracks_data.copy()
                        if 'intensity' not in tracks_with_intensity.columns:
                            tracks_with_intensity['intensity'] = tracks_with_intensity[primary_intensity_col]
                        
                        st.session_state.channel2_data = channel2_data_virtual
                        st.session_state.tracks_data_with_intensity = tracks_with_intensity
                        
                        st.info(f"✓ Using {primary_intensity_col} from primary data and {secondary_intensity_col} as secondary channel")
                
                else:
                    # Original file upload workflow
                    st.subheader("Load Secondary Channel Data")
                    
                    # File uploader for second channel
                    channel2_file = st.file_uploader(
                        "Upload secondary channel track data",
                        type=["csv", "txt", "xls", "xlsx", "h5", "json"],
                        key="channel2_uploader",
                        help="Upload track data for the second channel to analyze interactions"
                    )
                    
                    if channel2_file is not None:
                        try:
                            # Load the second channel data
                            with st.spinner("Loading secondary channel data..."):
                                channel2_data = load_tracks_file(channel2_file)
                                
                                # Format to standard format if needed
                                channel2_data = format_track_data(channel2_data)
                                
                                # Display preview
                                st.subheader("Secondary Channel Data Preview")
                                st.dataframe(channel2_data.head())
                                
                                # Display basic statistics
                                st.metric("Tracks", len(channel2_data['track_id'].unique()))
                                st.metric("Total Points", len(channel2_data))
                                
                                # Set session state for secondary channel
                                st.session_state.channel2_data = channel2_data
                                
                        except Exception as e:
                            st.error(f"Error loading secondary channel data: {str(e)}")
                            st.session_state.channel2_data = None
                    else:
                        if not (existing_channels and len(existing_channels) >= 2):
                            st.info("Please upload data for the secondary channel to perform multi-channel analysis.")
                        st.session_state.channel2_data = None
                
                # Configure the Analysis settings
                if st.session_state.tracks_data is not None and st.session_state.channel2_data is not None:
                    # Check if we're using existing intensity data or separate files
                    using_existing_intensity = (existing_channels and len(existing_channels) >= 2 and 
                                               data_source == "Use Existing Intensity Data")
                    
                    if not using_existing_intensity:
                        # Only show these settings if NOT using existing intensity data
                        # (to avoid duplicate inputs)
                        st.subheader("Multi-Channel Analysis Settings")
                        
                        # Organize parameters in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Primary Channel Settings")
                            primary_channel_name = st.text_input("Primary Channel Name", value="Channel 1", key="primary_ch_name")
                            primary_color = st.color_picker("Primary Channel Color", value="#FF4B4B", key="primary_ch_color")
                        
                        with col2:
                            st.write("Secondary Channel Settings")
                            secondary_channel_name = st.text_input("Secondary Channel Name", value="Channel 2", key="secondary_ch_name")
                            secondary_color = st.color_picker("Secondary Channel Color", value="#4B70FF", key="secondary_ch_color")
                    else:
                        # Settings already defined above for existing intensity mode
                        pass
                    
                    # Analysis parameters
                    st.subheader("Interaction Parameters")
                    
                    distance_threshold = st.slider(
                        "Maximum Interaction Distance (μm)", 
                        min_value=0.1, 
                        max_value=10.0, 
                        value=2.0,
                        step=0.1,
                        help="Maximum distance to consider particles as interacting"
                    )
                    
                    max_time_difference = st.slider(
                        "Maximum Time Difference (frames)", 
                        min_value=0, 
                        max_value=10, 
                        value=1,
                        help="Maximum time difference between frames to consider for interactions"
                    )
                    
                    # Button to run the analysis
                    if st.button("Run Multi-Channel Analysis"):
                        with st.spinner("Analyzing channel interactions..."):
                            try:
                                # Determine which data to use for primary channel
                                primary_data = (st.session_state.tracks_data_with_intensity 
                                              if 'tracks_data_with_intensity' in st.session_state 
                                              else st.session_state.tracks_data)
                                
                                # Create analyzer
                                analyzer = MultiChannelAnalyzer()
                                
                                # Add channels
                                analyzer.add_channel(primary_data, primary_channel_name)
                                analyzer.add_channel(st.session_state.channel2_data, secondary_channel_name)
                                
                                # Calculate colocalization
                                coloc_results = analyzer.calculate_colocalization_statistics(
                                    primary_channel_name, 
                                    secondary_channel_name,
                                    distance_threshold=distance_threshold,
                                    max_time_difference=max_time_difference
                                )
                                
                                # Save results to session state
                                st.session_state.multi_channel_results = coloc_results
                                
                                # Create analysis record
                                create_analysis_record(
                                    name="Multi-Channel Analysis",
                                    analysis_type="multi_channel",
                                    parameters={
                                        "distance_threshold": distance_threshold,
                                        "max_time_difference": max_time_difference,
                                        "primary_channel": primary_channel_name,
                                        "secondary_channel": secondary_channel_name
                                    },
                                    results=coloc_results
                                )
                                
                                # Display results
                                st.success("Multi-channel analysis completed successfully!")
                                
                                # Show results in an organized way
                                st.subheader("Colocalization Results")
                                
                                # Display key metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Colocalization Events", coloc_results['n_colocalization_events'])
                                
                                with col2:
                                    st.metric("Colocalized Tracks (Ch1)", coloc_results['n_tracks_colocalized_ch1'])
                                
                                with col3:
                                    st.metric("Colocalized Tracks (Ch2)", coloc_results['n_tracks_colocalized_ch2'])
                                
                                # Display colocalization plots if available
                                if 'spatial_distribution' in coloc_results:
                                    st.subheader("Spatial Distribution of Colocalization Events")
                                    st.plotly_chart(coloc_results['spatial_distribution'], use_container_width=True)
                                
                                if 'temporal_profile' in coloc_results:
                                    st.subheader("Temporal Profile of Colocalization")
                                    st.plotly_chart(coloc_results['temporal_profile'], use_container_width=True)
                                
                                # Display colocalization events as a table
                                if 'colocalization_events' in coloc_results:
                                    st.subheader("Colocalization Events")
                                    st.dataframe(coloc_results['colocalization_events'])
                                
                                # Display statistics table
                                if 'statistics' in coloc_results:
                                    st.subheader("Colocalization Statistics")
                                    stats_df = pd.DataFrame(coloc_results['statistics'], index=[0]).T
                                    stats_df.columns = ['Value']
                                    st.dataframe(stats_df)
                                    
                            except Exception as e:
                                st.error(f"Error in multi-channel analysis: {str(e)}")
                                st.error("Please check that both datasets are compatible and try again.")
                else:
                    st.warning("Please ensure both primary and secondary channel data are loaded.")
            else:
                st.warning("Multi-Channel Analysis module is not available. Please ensure the 'correlative_analysis.py' file is properly installed.")
                
        # Advanced Analysis tab
        with tabs[6]:
            st.header("Advanced Analysis")
            
            # Create subtabs for different advanced analysis types
            adv_tabs = st.tabs([
                "Gel Structure", 
                "Diffusion Population", 
                "Active Transport", 
                "Boundary Crossing", 
                "Crowding",
                "Polymer Physics",
                "Biophysical Models",
                "Microrheology"
            ])
            
            # Gel Structure Analysis
            with adv_tabs[0]:
                st.header("Gel Structure Analysis")
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    min_confinement_time = st.slider(
                        "Minimum Confinement Time (frames)",
                        min_value=3,
                        max_value=20,
                        value=5,
                        help="Minimum frames for detecting confinement"
                    )
                    
                    diffusion_threshold = st.slider(
                        "Diffusion Threshold (µm²/s)",
                        min_value=0.01,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        help="Threshold for identifying confined diffusion"
                    )
                    
                with col2:
                    # Use current unit settings as default values
                    units = get_current_units()
                    
                    pixel_size = st.number_input(
                        "Pixel Size (µm)",
                        min_value=0.01,
                        max_value=10.0,
                        value=units['pixel_size'],
                        step=0.01,
                        key="gel_pixel_size",
                        help="Pixel size in micrometers"
                    )
                    
                    frame_interval = st.number_input(
                        "Frame Interval (s)",
                        min_value=0.001,
                        max_value=60.0,
                        value=units['frame_interval'],
                        step=0.001,
                        key="gel_frame_interval",
                        help="Time between frames in seconds"
                    )
                
                # Run analysis on button click
                if st.button("Run Gel Structure Analysis"):
                    with st.spinner("Running gel structure analysis..."):
                        try:
                            gel_results = analyze_gel_structure(
                                st.session_state.tracks_data,
                                min_confinement_time=min_confinement_time,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                diffusion_threshold=diffusion_threshold
                            )
                            
                            # Store results in session state
                            st.session_state.analysis_results["gel_structure"] = gel_results
                            
                            # Create a record of this analysis
                            analysis_record = create_analysis_record(
                                name="Gel Structure Analysis",
                                analysis_type="gel_structure",
                                parameters={
                                    "min_confinement_time": min_confinement_time,
                                    "pixel_size": pixel_size,
                                    "frame_interval": frame_interval,
                                    "diffusion_threshold": diffusion_threshold
                                }
                            )
                            
                            # Add to recent analyses
                            st.session_state.recent_analyses.append(analysis_record)
                            
                            st.success("Gel structure analysis completed successfully!")
                        except Exception as e:
                            st.error(f"Error running gel structure analysis: {str(e)}")
                
                # Display results if available
                if "gel_structure" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["gel_structure"]
                    
                    # Show ensemble statistics first if available
                    if 'ensemble_results' in results:
                        st.subheader("Structural Statistics")
                        for key, value in results['ensemble_results'].items():
                            st.text(f"{key}: {value}")
                    
                    # Mesh properties
                    st.subheader("Mesh Properties")
                    if 'mesh_properties' in results and results['mesh_properties'] is not None:
                        for key, value in results['mesh_properties'].items():
                            st.text(f"{key}: {value}")
                    else:
                        st.warning("No mesh properties available to display.")
                        
                    # Confinement regions
                    st.subheader("Confinement Regions")
                    if 'confined_regions' in results and isinstance(results['confined_regions'], pd.DataFrame) and not results['confined_regions'].empty:
                        st.dataframe(results['confined_regions'])
                    else:
                        # Display an empty dataframe with the proper columns
                        st.dataframe(pd.DataFrame(columns=['track_id', 'region_id', 'start_frame', 'end_frame', 'duration', 'center_x', 'center_y', 'radius', 'diffusion_coeff']))
                        st.info("No confinement regions were detected with the current parameters. Try adjusting the diffusion threshold or minimum confinement time.")
                    
                    # Track analysis results
                    st.subheader("Track Analysis")
                    if 'track_results' in results and results['track_results'] is not None:
                        if isinstance(results['track_results'], pd.DataFrame) and not results['track_results'].empty:
                            st.dataframe(results['track_results'].head(20))
                        elif isinstance(results['track_results'], list) and results['track_results']:
                            st.dataframe(pd.DataFrame(results['track_results']).head(20))
                        else:
                            st.warning("No track analysis results found.")
                    else:
                        st.warning("No track analysis results available to display.")
            
            # Diffusion Population Analysis
            with adv_tabs[1]:
                st.header("Diffusion Population Analysis")
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    max_lag = st.slider(
                        "Maximum Lag Time (frames)",
                        min_value=5,
                        max_value=50,
                        value=20,
                        key="diffpop_max_lag",
                        help="Maximum lag time for MSD calculation"
                    )
                    
                    min_track_length = st.slider(
                        "Minimum Track Length",
                        min_value=5,
                        max_value=50,
                        value=10,
                        key="diffpop_min_track_length",
                        help="Minimum track length to include in analysis"
                    )
                    
                    n_populations = st.slider(
                        "Number of Populations",
                        min_value=2,
                        max_value=5,
                        value=2,
                        help="Number of diffusion populations to identify"
                    )
                    
                with col2:
                    # Use current unit settings as default values
                    units = get_current_units()
                    
                    pixel_size = st.number_input(
                        "Pixel Size (µm)",
                        min_value=0.01,
                        max_value=10.0,
                        value=units['pixel_size'],
                        step=0.01,
                        key="diffpop_pixel_size",
                        help="Pixel size in micrometers"
                    )
                    
                    frame_interval = st.number_input(
                        "Frame Interval (s)",
                        min_value=0.001,
                        max_value=60.0,
                        value=units['frame_interval'],
                        step=0.001,
                        key="diffpop_frame_interval",
                        help="Time between frames in seconds"
                    )
                
                # Run analysis on button click
                if st.button("Run Diffusion Population Analysis"):
                    with st.spinner("Running diffusion population analysis..."):
                        try:
                            diffpop_results = analyze_diffusion_population(
                                st.session_state.tracks_data,
                                max_lag=max_lag,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                min_track_length=min_track_length,
                                n_populations=n_populations
                            )
                            
                            # Store results in session state
                            st.session_state.analysis_results["diffusion_population"] = diffpop_results
                            
                            # Create a record of this analysis
                            analysis_record = create_analysis_record(
                                name="Diffusion Population Analysis",
                                analysis_type="diffusion_population",
                                parameters={
                                    "max_lag": max_lag,
                                    "pixel_size": pixel_size,
                                    "frame_interval": frame_interval,
                                    "min_track_length": min_track_length,
                                    "n_populations": n_populations
                                }
                            )
                            
                            # Add to recent analyses
                            st.session_state.recent_analyses.append(analysis_record)
                            
                            st.success("Diffusion population analysis completed successfully!")
                        except Exception as e:
                            st.error(f"Error running diffusion population analysis: {str(e)}")
                
                # Display results if available
                if "diffusion_population" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["diffusion_population"]
                    
                    # First show ensemble results if available
                    if 'ensemble_results' in results:
                        st.subheader("Population Statistics")
                        for key, value in results['ensemble_results'].items():
                            st.text(f"{key}: {value}")
                    
                    # Population parameters
                    st.subheader("Population Parameters")
                    if 'populations' in results and isinstance(results['populations'], pd.DataFrame) and not results['populations'].empty:
                        st.dataframe(results['populations'])
                    else:
                        # Display an empty dataframe with proper columns
                        empty_df = pd.DataFrame(columns=['population_id', 'weight', 'mean_diffusion_coeff', 'std_diffusion_coeff', 'log_mean', 'log_std'])
                        st.dataframe(empty_df)
                        st.info("No diffusion populations were detected. Try adjusting the number of populations or analyzing more tracks.")
                        
                    # Track classifications
                    st.subheader("Track Classifications")
                    if 'track_assignments' in results and isinstance(results['track_assignments'], pd.DataFrame) and not results['track_assignments'].empty:
                        st.dataframe(results['track_assignments'].head(20))
                    else:
                        # Display an empty dataframe with proper columns
                        empty_df = pd.DataFrame(columns=['track_id', 'diffusion_coeff', 'log_diffusion_coeff', 'population_id', 'population_name'])
                        st.dataframe(empty_df)
                        st.info("No track classifications were detected. Try adjusting the analysis parameters or ensure there are enough tracks with valid diffusion coefficients.")
                    
                    # MSD data
                    st.subheader("Mean Squared Displacement Data")
                    msd_data = results.get('msd_data')
                    if 'raw_diffusion_data' in results and msd_data is not None and isinstance(msd_data, pd.DataFrame) and not msd_data.empty:
                        st.dataframe(msd_data.head(20))
                    else:
                        # Display an empty dataframe with the proper columns
                        empty_df = pd.DataFrame(columns=['track_id', 'lag_time', 'msd', 'stderr'])
                        st.dataframe(empty_df)
                        st.info("No MSD data was available. Try adjusting the analysis parameters or check the track data quality.")
            
            # Active Transport Analysis
            with adv_tabs[2]:
                st.header("Active Transport Analysis")
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    window_size = st.slider(
                        "Window Size (frames)",
                        min_value=3,
                        max_value=20,
                        value=5,
                        key="active_window_size",
                        help="Window size for detecting directed segments"
                    )
                    
                    min_track_length = st.slider(
                        "Minimum Track Length",
                        min_value=5,
                        max_value=50,
                        value=10,
                        key="active_min_track_length",
                        help="Minimum track length to include in analysis"
                    )
                    
                    straightness_threshold = st.slider(
                        "Straightness Threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        help="Threshold for straightness to identify directed motion"
                    )
                    
                    min_segment_length = st.slider(
                        "Minimum Segment Length (frames)",
                        min_value=3,
                        max_value=20,
                        value=5,
                        help="Minimum length of directed motion segment"
                    )
                    
                with col2:
                    # Use current unit settings as default values
                    units = get_current_units()
                    
                    pixel_size = st.number_input(
                        "Pixel Size (µm)",
                        min_value=0.01,
                        max_value=10.0,
                        value=units['pixel_size'],
                        step=0.01,
                        key="active_pixel_size",
                        help="Pixel size in micrometers"
                    )
                    
                    frame_interval = st.number_input(
                        "Frame Interval (s)",
                        min_value=0.001,
                        max_value=60.0,
                        value=units['frame_interval'],
                        step=0.001,
                        key="active_frame_interval",
                        help="Time between frames in seconds"
                    )
                
                # Run analysis on button click
                if st.button("Run Active Transport Analysis"):
                    with st.spinner("Running active transport analysis..."):
                        try:
                            active_results = analyze_active_transport(
                                st.session_state.tracks_data,
                                window_size=window_size,
                                min_track_length=min_track_length,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                straightness_threshold=straightness_threshold,
                                min_segment_length=min_segment_length
                            )
                            
                            # Store results in session state
                            st.session_state.analysis_results["active_transport"] = active_results
                            
                            # Create a record of this analysis
                            analysis_record = create_analysis_record(
                                name="Active Transport Analysis",
                                analysis_type="active_transport",
                                parameters={
                                    "window_size": window_size,
                                    "min_track_length": min_track_length,
                                    "pixel_size": pixel_size,
                                    "frame_interval": frame_interval,
                                    "straightness_threshold": straightness_threshold,
                                    "min_segment_length": min_segment_length
                                }
                            )
                            
                            # Add to recent analyses
                            st.session_state.recent_analyses.append(analysis_record)
                            
                            st.success("Active transport analysis completed successfully!")
                        except Exception as e:
                            st.error(f"Error running active transport analysis: {str(e)}")
                
                # Display results if available
                if "active_transport" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["active_transport"]
                    
                    # Directed segments
                    st.subheader("Directed Segments")
                    if 'directed_segments' in results and isinstance(results['directed_segments'], pd.DataFrame) and not results['directed_segments'].empty:
                        st.dataframe(results['directed_segments'])
                    else:
                        # Display an empty dataframe with the proper columns
                        empty_df = pd.DataFrame(columns=['track_id', 'segment_id', 'start_frame', 'end_frame', 'duration', 'length', 'speed', 'straightness'])
                        st.dataframe(empty_df)
                        st.info("No directed motion segments were detected with the current parameters. Try adjusting the straightness threshold or window size.")
                        
                    # Transport statistics
                    st.subheader("Transport Statistics")
                    ensemble_stats = results.get('ensemble_results', {})
                    if ensemble_stats:
                        st.text(f"Mean Transport Velocity: {ensemble_stats.get('mean_transport_velocity', 0.0):.3f} µm/s")
                        st.text(f"Median Transport Velocity: {ensemble_stats.get('median_transport_velocity', 0.0):.3f} µm/s")
                        st.text(f"Mean Segment Duration: {ensemble_stats.get('mean_segment_duration', 0.0):.3f} s")
                        st.text(f"Percent of Tracks with Directed Motion: {ensemble_stats.get('percent_directed_tracks', 0.0):.1f}%")
                        st.text(f"Mean Straightness: {ensemble_stats.get('mean_straightness', 0.0):.3f}")
                    else:
                        # Display default statistics
                        st.text("Mean segment speed: 0.0 µm/s")
                        st.text("Mean segment duration: 0.0 s")
                        st.text("Percent of tracks with directed motion: 0.0%")
                        st.text("Mean straightness: 0.0")
                        st.info("No transport statistics were calculated. This may occur when no directed motion is detected in the dataset.")
            
            # Boundary Crossing Analysis
            with adv_tabs[3]:
                st.header("Boundary Crossing Analysis")
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    min_track_length = st.slider(
                        "Minimum Track Length",
                        min_value=5,
                        max_value=50,
                        value=5,
                        key="boundary_min_track_length",
                        help="Minimum track length to include in analysis"
                    )
                    
                    use_boundaries = st.checkbox(
                        "Define Boundaries",
                        value=False,
                        help="Define specific boundaries manually"
                    )
                    
                with col2:
                    # Use current unit settings as default values
                    units = get_current_units()
                    
                    pixel_size = st.number_input(
                        "Pixel Size (µm)",
                        min_value=0.01,
                        max_value=10.0,
                        value=units['pixel_size'],
                        step=0.01,
                        key="boundary_pixel_size",
                        help="Pixel size in micrometers"
                    )
                    
                    frame_interval = st.number_input(
                        "Frame Interval (s)",
                        min_value=0.001,
                        max_value=60.0,
                        value=units['frame_interval'],
                        step=0.001,
                        key="advanced_boundary_frame_interval",
                        help="Time between frames in seconds"
                    )
                
                # Boundary definition (if enabled)
                boundaries = None
                if use_boundaries:
                    st.subheader("Boundary Definitions")
                    
                    num_boundaries = st.slider(
                        "Number of Boundaries",
                        min_value=1,
                        max_value=5,
                        value=1
                    )
                    
                    boundaries = []
                    boundary_type = st.selectbox(
                        "Boundary Type",
                        ["Line", "Circle", "Rectangle"],
                        help="Type of boundary to define"
                    )
                    
                    for i in range(num_boundaries):
                        with st.expander(f"Boundary {i+1}"):
                            if boundary_type == "Line":
                                col1, col2 = st.columns(2)
                                with col1:
                                    x1 = st.number_input(f"X1 (Boundary {i+1})", value=0.0)
                                    y1 = st.number_input(f"Y1 (Boundary {i+1})", value=0.0)
                                with col2:
                                    x2 = st.number_input(f"X2 (Boundary {i+1})", value=10.0)
                                    y2 = st.number_input(f"Y2 (Boundary {i+1})", value=10.0)
                                
                                boundaries.append({
                                    'type': 'line',
                                    'x1': x1, 'y1': y1,
                                    'x2': x2, 'y2': y2
                                })
                            
                            elif boundary_type == "Circle":
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    x = st.number_input(f"X center (Boundary {i+1})", value=0.0)
                                with col2:
                                    y = st.number_input(f"Y center (Boundary {i+1})", value=0.0)
                                with col3:
                                    radius = st.number_input(f"Radius (Boundary {i+1})", value=5.0, min_value=0.1)
                                
                                boundaries.append({
                                    'type': 'circle',
                                    'x': x, 'y': y,
                                    'radius': radius
                                })
                            
                            elif boundary_type == "Rectangle":
                                col1, col2 = st.columns(2)
                                with col1:
                                    x1 = st.number_input(f"X1 (Boundary {i+1})", value=0.0, key=f"rect_x1_{i}")
                                    y1 = st.number_input(f"Y1 (Boundary {i+1})", value=0.0, key=f"rect_y1_{i}")
                                with col2:
                                    x2 = st.number_input(f"X2 (Boundary {i+1})", value=10.0, key=f"rect_x2_{i}")
                                    y2 = st.number_input(f"Y2 (Boundary {i+1})", value=10.0, key=f"rect_y2_{i}")
                                
                                boundaries.append({
                                    'type': 'rectangle',
                                    'x1': x1, 'y1': y1,
                                    'x2': x2, 'y2': y2
                                })
                
                # Run analysis on button click
                if st.button("Run Boundary Crossing Analysis"):
                    with st.spinner("Running boundary crossing analysis..."):
                        try:
                            boundary_results = analyze_boundary_crossing(
                                st.session_state.tracks_data,
                                boundaries=boundaries,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                min_track_length=min_track_length
                            )
                            
                            # Store results in session state
                            st.session_state.analysis_results["boundary_crossing"] = boundary_results
                            
                            # Create a record of this analysis
                            analysis_record = create_analysis_record(
                                name="Boundary Crossing Analysis",
                                analysis_type="boundary_crossing",
                                parameters={
                                    "pixel_size": pixel_size,
                                    "frame_interval": frame_interval,
                                    "min_track_length": min_track_length,
                                    "use_boundaries": use_boundaries,
                                    "num_boundaries": len(boundaries) if boundaries else 0
                                }
                            )
                            
                            # Add to recent analyses
                            st.session_state.recent_analyses.append(analysis_record)
                            
                            st.success("Boundary crossing analysis completed successfully!")
                        except Exception as e:
                            st.error(f"Error running boundary crossing analysis: {str(e)}")
                
                # Display results if available
                if "boundary_crossing" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["boundary_crossing"]
                    
                    # Crossing events
                    st.subheader("Crossing Events")
                    if 'crossing_events' in results:
                        st.dataframe(results['crossing_events'])
                    else:
                        st.warning("No crossing events available to display.")
                        
                    # Crossing statistics
                    st.subheader("Crossing Statistics")
                    if 'crossing_stats' in results and results['crossing_stats']:
                        for key, value in results['crossing_stats'].items():
                            st.text(f"{key}: {value}")
                    else:
                        st.warning("No crossing statistics available to display.")
            
            # Crowding Analysis
            with adv_tabs[4]:
                st.header("Crowding Analysis")
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    radius_of_influence = st.slider(
                        "Radius of Influence (µm)",
                        min_value=0.1,
                        max_value=10.0,
                        value=2.0,
                        step=0.1,
                        help="Radius to consider for density calculation"
                    )
                    
                    min_track_length = st.slider(
                        "Minimum Track Length",
                        min_value=3,
                        max_value=50,
                        value=5,
                        key="crowding_min_track_length",
                        help="Minimum track length to include in analysis"
                    )
                    
                with col2:
                    # Use current unit settings as default values
                    units = get_current_units()
                    
                    pixel_size = st.number_input(
                        "Pixel Size (µm)",
                        min_value=0.01,
                        max_value=10.0,
                        value=units['pixel_size'],
                        step=0.01,
                        key="crowding_pixel_size",
                        help="Pixel size in micrometers"
                    )
                
                # Run analysis on button click
                if st.button("Run Crowding Analysis"):
                    with st.spinner("Running crowding analysis..."):
                        try:
                            crowding_results = analyze_crowding(
                                st.session_state.tracks_data,
                                radius_of_influence=radius_of_influence,
                                pixel_size=pixel_size,
                                min_track_length=min_track_length
                            )
                            
                            # Store results in session state
                            st.session_state.analysis_results["crowding"] = crowding_results
                            
                            # Create a record of this analysis
                            analysis_record = create_analysis_record(
                                name="Crowding Analysis",
                                analysis_type="crowding",
                                parameters={
                                    "radius_of_influence": radius_of_influence,
                                    "pixel_size": pixel_size,
                                    "min_track_length": min_track_length
                                }
                            )
                            
                            # Add to recent analyses
                            st.session_state.recent_analyses.append(analysis_record)
                            
                            st.success("Crowding analysis completed successfully!")
                        except Exception as e:
                            st.error(f"Error running crowding analysis: {str(e)}")
                
                # Display results if available
                if "crowding" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["crowding"]
                    
                    # Show ensemble statistics first
                    st.subheader("Crowding Statistics")
                    if 'ensemble_results' in results and isinstance(results['ensemble_results'], dict) and results['ensemble_results']:
                        for key, value in results['ensemble_results'].items():
                            st.text(f"{key}: {value}")
                    else:
                        # Display default statistics
                        st.text("Mean local density: 0.0 particles/µm²")
                        st.text("Median local density: 0.0 particles/µm²") 
                        st.text("Density-diffusion correlation: 0.0")
                        st.info("No crowding statistics were calculated. Try adjusting the radius of influence or check that your dataset contains multiple particles in proximity.")
                    
                    # Density map
                    st.subheader("Density Map")
                    if 'crowding_data' in results and results.get('density_map') is not None:
                        try:
                            # If it's an array, display it as a heatmap
                            if isinstance(results['density_map'], np.ndarray):
                                fig = px.imshow(results['density_map'], 
                                              labels=dict(x="X Position", y="Y Position", color="Particle Density"),
                                              color_continuous_scale='viridis')
                                fig.update_layout(title="Particle Density Heatmap")
                                st.plotly_chart(fig, use_container_width=True)
                            # If it's a dataframe with x,y,density columns
                            elif isinstance(results['density_map'], pd.DataFrame) and not results['density_map'].empty:
                                fig = px.scatter(results['density_map'], 
                                               x='x', y='y', 
                                               color='density',
                                               color_continuous_scale='viridis',
                                               size='density',
                                               labels={'x': 'X Position (µm)', 'y': 'Y Position (µm)', 'density': 'Particle Density'},
                                               title="Particle Density Map")
                                fig.update_yaxes(autorange="reversed")  # Invert Y-axis for image-like coordinates
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Density map is available but in an unsupported format.")
                        except Exception as e:
                            st.error(f"Error generating density map visualization: {str(e)}")
                    else:
                        st.info("No density map was generated. A density map requires multiple particles within the radius of influence.")
                        
                    # Track results with density
                    st.subheader("Track Results with Local Density")
                    if 'track_results' in results and isinstance(results['track_results'], pd.DataFrame) and not results['track_results'].empty:
                        st.dataframe(results['track_results'].head(20))
                    else:
                        # Display an empty dataframe with the proper columns
                        empty_df = pd.DataFrame(columns=['track_id', 'mean_density', 'max_density', 'min_density', 'diffusion_coeff', 'mobility_factor'])
                        st.dataframe(empty_df)
                        st.info("No track density results were calculated. This may occur when particles are too far apart or the radius of influence is too small.")
                        
                    # Density-mobility correlation
                    st.subheader("Density-Mobility Correlation")
                    if 'density_displacement_correlation' in results and results['density_displacement_correlation'] is not None:
                        if isinstance(results['density_displacement_correlation'], pd.DataFrame):
                            st.dataframe(results['density_displacement_correlation'])
                        else:
                            for key, value in results['density_displacement_correlation'].items():
                                st.text(f"{key}: {value}")
                    else:
                        # Display an empty dataframe with proper columns
                        empty_df = pd.DataFrame(columns=['density_bin', 'mean_diffusion', 'count', 'std_diffusion'])
                        st.dataframe(empty_df)
                        st.info("No density-mobility correlation was calculated. This requires sufficient data points across different density regions.")
            
            # Biophysical Models Analysis
            # Polymer Physics Analysis
            with adv_tabs[5]:
                st.header("Polymer Physics Analysis")
                
                st.markdown("""
                Analyze polymer dynamics and determine the regime (Rouse, Zimm, Reptation) based on 
                the scaling exponent of mean squared displacement: MSD ~ t^α
                
                **Polymer Regimes:**
                - **Rouse (unentangled):** α ≈ 0.5 - subdiffusive motion without hydrodynamic interactions
                - **Zimm (with hydrodynamics):** α ≈ 0.6 - includes solvent effects
                - **Reptation (entangled):** α ≈ 0.25 - strong subdiffusion due to topological constraints
                - **Free diffusion:** α ≈ 1.0 - normal diffusive behavior
                """)
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    pixel_size = get_global_pixel_size()
                    st.metric("Pixel Size", f"{pixel_size:.4f} µm")
                    
                    frame_interval = get_global_frame_interval()
                    st.metric("Frame Interval", f"{frame_interval:.4f} s")
                    
                with col2:
                    min_track_length = st.slider(
                        "Minimum Track Length",
                        min_value=5,
                        max_value=50,
                        value=10,
                        help="Minimum number of frames required for polymer analysis"
                    )
                
                # Run analysis
                if st.button("Run Polymer Physics Analysis", key="polymer_physics_btn"):
                    with st.spinner("Analyzing polymer dynamics..."):
                        try:
                            polymer_results = analyze_polymer_physics(
                                st.session_state.tracks_data,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                min_track_length=min_track_length
                            )
                            
                            if polymer_results.get('success'):
                                st.session_state.analysis_results['polymer_physics'] = polymer_results
                                st.success("✓ Polymer physics analysis completed!")
                                
                                # Display results
                                st.subheader("Results")
                                
                                # Key metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Scaling Exponent (α)", 
                                        f"{polymer_results['scaling_exponent']:.3f} ± {polymer_results.get('scaling_exponent_error', 0):.3f}"
                                    )
                                
                                with col2:
                                    st.metric("Regime", polymer_results['regime'])
                                
                                with col3:
                                    st.metric(
                                        "Fit Quality (R²)", 
                                        f"{polymer_results['r_squared']:.3f}"
                                    )
                                
                                # Interpretation
                                st.subheader("Interpretation")
                                interp = polymer_results.get('interpretation', {})
                                st.info(f"**{interp.get('description', 'N/A')}**")
                                st.write(interp.get('properties', ''))
                                
                                # MSD vs Time Plot (log-log)
                                st.subheader("MSD vs Time (Log-Log Plot)")
                                
                                import plotly.graph_objects as go
                                
                                fig = go.Figure()
                                
                                # Data points
                                fig.add_trace(go.Scatter(
                                    x=polymer_results['lag_times'],
                                    y=polymer_results['msd_data'],
                                    mode='markers',
                                    name='Data',
                                    marker=dict(size=8, color='blue')
                                ))
                                
                                # Power law fit
                                fig.add_trace(go.Scatter(
                                    x=polymer_results['lag_times'],
                                    y=polymer_results['fitted_msd'],
                                    mode='lines',
                                    name=f'Fit: MSD = A·t^{polymer_results["scaling_exponent"]:.3f}',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                                
                                fig.update_xaxes(
                                    title="Lag Time (s)",
                                    type="log",
                                    showgrid=True,
                                    gridcolor='lightgray'
                                )
                                fig.update_yaxes(
                                    title="MSD (µm²)",
                                    type="log",
                                    showgrid=True,
                                    gridcolor='lightgray'
                                )
                                fig.update_layout(
                                    title=f"Polymer Regime: {polymer_results['regime']}",
                                    hovermode='closest',
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Statistical details
                                with st.expander("📊 Statistical Details"):
                                    st.write("**Ensemble Results:**")
                                    for key, value in polymer_results['ensemble_results'].items():
                                        if isinstance(value, float):
                                            st.write(f"- {key.replace('_', ' ').title()}: {value:.4f}")
                                        else:
                                            st.write(f"- {key.replace('_', ' ').title()}: {value}")
                                
                            else:
                                st.error(f"Analysis failed: {polymer_results.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error in polymer physics analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Display previous results if available
                elif 'polymer_physics' in st.session_state.get('analysis_results', {}):
                    st.info("Previous analysis results available. Click 'Run Polymer Physics Analysis' to recompute.")
                    
                    polymer_results = st.session_state.analysis_results['polymer_physics']
                    
                    if polymer_results.get('success'):
                        st.subheader("Previous Results")
                        
                        # Check if expected keys exist
                        if 'scaling_exponent' in polymer_results and 'regime' in polymer_results:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Scaling Exponent (α)", 
                                    f"{polymer_results['scaling_exponent']:.3f}"
                                )
                            
                            with col2:
                                st.metric("Regime", polymer_results['regime'])
                            
                            with col3:
                                if 'r_squared' in polymer_results:
                                    st.metric(
                                        "Fit Quality (R²)", 
                                        f"{polymer_results['r_squared']:.3f}"
                                    )
                                else:
                                    st.metric("Fit Quality (R²)", "N/A")
                        else:
                            st.warning("Previous results are in an incompatible format. Please rerun the analysis.")
                    else:
                        st.error(f"Previous analysis failed: {polymer_results.get('error', 'Unknown error')}")
            
            # Biophysical Models Analysis
            with adv_tabs[6]:
                st.header("Biophysical Models Analysis")
                
                if st.session_state.tracks_data is not None:
                    # Create subtabs for different biophysical analyses
                    bio_tabs = st.tabs([
                        "Biophysical Models",
                        "Advanced Biophysical Metrics"
                    ])
                    
                    with bio_tabs[0]:
                        show_biophysical_models()
                    
                    with bio_tabs[1]:
                        show_advanced_biophysical_metrics()
                
                else:
                    st.warning("Please load tracking data first to perform biophysical analysis.")

# Visualization page
elif st.session_state.active_page == "Visualization":
    st.title("Visualization Tools")
    
    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
    else:
        # Create tabs for different visualization types
        viz_tabs = st.tabs([
            "Track Visualization", 
            "Diffusion Visualization", 
            "Motion Visualization", 
            "3D Visualization",
            "Trajectory Heatmaps",
            "Custom Visualization"
        ])
        
        # Track Visualization tab
        with viz_tabs[0]:
            st.header("Track Visualization")
            
            # Visualization options
            st.subheader("Visualization Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                color_by = st.selectbox(
                    "Color By",
                    ["track_id", "frame", "x", "y", "Quality"],
                    help="Property to use for coloring tracks"
                )
                
                colormap = st.selectbox(
                    "Color Map",
                    ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "rainbow"],
                    help="Colormap for tracks"
                )
                
            with col2:
                max_tracks = st.slider(
                    "Max Tracks to Display",
                    min_value=1,
                    max_value=min(100, st.session_state.tracks_data['track_id'].nunique()),
                    value=min(20, st.session_state.tracks_data['track_id'].nunique()),
                    help="Maximum number of tracks to display (for better performance)"
                )
                
                plot_type = st.selectbox(
                    "Plot Type",
                    ["plotly", "matplotlib"],
                    help="Backend to use for plotting"
                )
            
            # Filter tracks if needed
            if max_tracks < st.session_state.tracks_data['track_id'].nunique():
                unique_tracks = st.session_state.tracks_data['track_id'].unique()[:max_tracks]
                filtered_tracks = st.session_state.tracks_data[st.session_state.tracks_data['track_id'].isin(unique_tracks)]
            else:
                filtered_tracks = st.session_state.tracks_data
            
            # Generate plot
            with st.spinner("Generating track visualization..."):
                try:
                    fig = plot_tracks(
                        filtered_tracks,
                        color_by=color_by,
                        colormap=colormap,
                        plot_type=plot_type
                    )
                    
                    # Display the plot
                    if plot_type == "plotly":
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating track visualization: {str(e)}")
        
        # Diffusion Visualization tab
        with viz_tabs[1]:
            st.header("Diffusion Visualization")
            
            # Check if diffusion analysis results are available
            if "diffusion" not in st.session_state.analysis_results:
                st.warning("No diffusion analysis results available. Please run diffusion analysis first.")
                st.button("Go to Diffusion Analysis", on_click=navigate_to, args=("Analysis",))
            else:
                # Visualization options
                st.subheader("Visualization Options")
                
                viz_type = st.radio(
                    "Visualization Type",
                    ["MSD Curves", "Diffusion Coefficients", "Anomalous Exponents", "Spatial Map"]
                )
                
                # Fetch diffusion results
                diffusion_results = st.session_state.analysis_results["diffusion"]
                
                # Generate appropriate visualization based on selection
                if viz_type == "MSD Curves":
                    if 'msd_data' in diffusion_results and isinstance(diffusion_results['msd_data'], pd.DataFrame) and not diffusion_results['msd_data'].empty:
                        # Check which lag column name is used in this dataset
                        lag_column = 'lag_time' if 'lag_time' in diffusion_results['msd_data'].columns else 'lag'
                        fig = px.scatter(diffusion_results['msd_data'], x=lag_column, y='msd', color='track_id', 
                                        labels={lag_column: 'Lag Time (frames)', 'msd': 'MSD (µm²)'},
                                        title='Mean Squared Displacement Curves')
                        
                        # Add linear fits if available
                        if 'track_fits' in diffusion_results and isinstance(diffusion_results['track_fits'], dict):
                            for track_id, fit in diffusion_results['track_fits'].items():
                                if 'fit_x' in fit and 'fit_y' in fit:
                                    fig.add_scatter(x=fit['fit_x'], y=fit['fit_y'], mode='lines', 
                                                  name=f'Fit: Track {track_id}', line=dict(dash='dash'))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No MSD data available. Run diffusion analysis with more tracks to generate MSD curves.")
                        
                elif viz_type == "Diffusion Coefficients":
                    if 'track_results' in diffusion_results and isinstance(diffusion_results['track_results'], pd.DataFrame) and not diffusion_results['track_results'].empty:
                        if 'diffusion_coeff' in diffusion_results['track_results'].columns:
                            fig = px.histogram(diffusion_results['track_results'], x='diffusion_coeff', 
                                              title='Distribution of Diffusion Coefficients',
                                              labels={'diffusion_coeff': 'Diffusion Coefficient (µm²/s)'},
                                              nbins=20)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Also show log-scale version which is often more informative
                            log_fig = px.histogram(diffusion_results['track_results'], 
                                                 x=np.log10(diffusion_results['track_results']['diffusion_coeff'].replace(0, np.nan)), 
                                                 title='Log Distribution of Diffusion Coefficients',
                                                 labels={'x': 'Log10 Diffusion Coefficient (µm²/s)'},
                                                 nbins=20)
                            st.plotly_chart(log_fig, use_container_width=True)
                        else:
                            st.info("Diffusion coefficient column not found in the track results.")
                    else:
                        st.info("No diffusion coefficient data available. Run diffusion analysis to generate this visualization.")
                
                elif viz_type == "Anomalous Exponents":
                    if 'track_results' in diffusion_results and isinstance(diffusion_results['track_results'], pd.DataFrame) and not diffusion_results['track_results'].empty:
                        if 'alpha' in diffusion_results['track_results'].columns:
                            fig = px.histogram(diffusion_results['track_results'], x='alpha', 
                                              title='Distribution of Anomalous Exponents',
                                              labels={'alpha': 'Anomalous Exponent (α)'},
                                              nbins=20)
                            
                            # Add reference line for α=1 (normal diffusion)
                            fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                         annotation_text="Normal Diffusion (α=1)")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show scatter plot of alpha vs diffusion coefficient
                            # Check for either column name version for compatibility
                            d_col = 'diffusion_coefficient'
                            if d_col not in diffusion_results['track_results'].columns:
                                d_col = 'diffusion_coeff'  # try alternative name
                                
                            if d_col in diffusion_results['track_results'].columns and 'alpha' in diffusion_results['track_results'].columns:
                                scatter_fig = px.scatter(diffusion_results['track_results'], 
                                                      x=d_col, y='alpha',
                                                      title='Diffusion Coefficient vs Anomalous Exponent',
                                                      labels={d_col: 'Diffusion Coefficient (µm²/s)', 
                                                             'alpha': 'Anomalous Exponent (α)'})
                                st.plotly_chart(scatter_fig, use_container_width=True)
                        else:
                            st.info("Anomalous exponent column not found in the track results. Make sure you selected 'anomalous' analysis in the diffusion settings.")
                    else:
                        st.info("No anomalous exponent data available. Run diffusion analysis with anomalous diffusion enabled to generate this visualization.")
                
                elif viz_type == "Spatial Map":
                    if 'track_results' in diffusion_results and isinstance(diffusion_results['track_results'], pd.DataFrame) and not diffusion_results['track_results'].empty:
                        # Merge track results with position data
                        if 'diffusion_coeff' in diffusion_results['track_results'].columns and not st.session_state.tracks_data.empty:
                            # Get the mean position of each track
                            track_positions = st.session_state.tracks_data.groupby('track_id')[['x', 'y']].mean().reset_index()
                            merged_data = pd.merge(track_positions, diffusion_results['track_results'], on='track_id')
                            
                            # Create a spatial map colored by diffusion coefficient
                            fig = px.scatter(merged_data, x='x', y='y', color='diffusion_coeff', 
                                           color_continuous_scale='viridis', 
                                           title='Spatial Map of Diffusion Coefficients',
                                           labels={'x': 'X Position (µm)', 'y': 'Y Position (µm)', 
                                                  'diffusion_coeff': 'Diffusion Coeff (µm²/s)'})
                            
                            fig.update_yaxes(autorange="reversed")  # Invert Y-axis for image-like coordinates
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Diffusion coefficient column not found or position data is missing.")
                    else:
                        st.info("No diffusion data available. Run diffusion analysis to generate the spatial map.")
        
        # Motion Visualization tab
        with viz_tabs[2]:
            st.header("Motion Visualization")
            
            # Check if motion analysis results are available
            if "motion" not in st.session_state.analysis_results:
                st.warning("No motion analysis results available. Please run motion analysis first.")
                st.button("Go to Motion Analysis", on_click=navigate_to, args=("Analysis",))
            else:
                # Visualization options
                st.subheader("Visualization Options")
                
                viz_type = st.radio(
                    "Visualization Type",
                    ["Speed Distribution", "Angle Changes", "Velocity Autocorrelation", "Motion Types"]
                )
                
                # Fetch motion results
                motion_results = st.session_state.analysis_results["motion"]
                
                # Generate appropriate visualization based on selection
                if viz_type == "Speed Distribution":
                    if 'speed_data' in motion_results and isinstance(motion_results['speed_data'], pd.DataFrame) and not motion_results['speed_data'].empty:
                        fig = px.histogram(motion_results['speed_data'], x='speed', 
                                          title='Speed Distribution',
                                          labels={'speed': 'Speed (µm/s)'},
                                          nbins=25)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics
                        if 'mean_speed' in motion_results:
                            st.text(f"Mean Speed: {motion_results['mean_speed']:.4f} µm/s")
                        if 'median_speed' in motion_results:
                            st.text(f"Median Speed: {motion_results['median_speed']:.4f} µm/s")
                        if 'max_speed' in motion_results:
                            st.text(f"Maximum Speed: {motion_results['max_speed']:.4f} µm/s")
                    else:
                        st.info("No speed data available. Run motion analysis with a valid dataset to generate this visualization.")
                
                elif viz_type == "Angle Changes":
                    if 'angle_data' in motion_results and isinstance(motion_results['angle_data'], pd.DataFrame) and not motion_results['angle_data'].empty:
                        # Create a radial histogram (rose plot) of angles
                        fig = px.bar_polar(motion_results['angle_data'], r='count', theta='angle_bin',
                                         title='Distribution of Direction Changes',
                                         labels={'r': 'Frequency', 'theta': 'Angle (degrees)'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also show a regular histogram for easier interpretation
                        hist_fig = px.histogram(motion_results['angle_data'], x='angle', 
                                              title='Histogram of Direction Changes',
                                              labels={'angle': 'Angle Change (degrees)'},
                                              nbins=36)  # 10-degree bins
                        st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        # Check if raw data is available to calculate angles
                        st.info("No angle change data available. Make sure your tracks have sufficient length to calculate direction changes.")
                
                elif viz_type == "Velocity Autocorrelation":
                    if 'velocity_autocorr' in motion_results and isinstance(motion_results['velocity_autocorr'], dict):
                        # Create dataframe from autocorrelation data
                        lags = motion_results['velocity_autocorr'].get('lags', [])
                        autocorr = motion_results['velocity_autocorr'].get('autocorr', [])
                        
                        if len(lags) == len(autocorr) and len(lags) > 0:
                            autocorr_df = pd.DataFrame({'lag': lags, 'autocorr': autocorr})
                            
                            # Use a more flexible approach to handle different column naming
                            lag_column = 'lag' if 'lag' in autocorr_df.columns else 'lag_time'
                            fig = px.line(autocorr_df, x=lag_column, y='autocorr', 
                                         title='Velocity Autocorrelation',
                                         labels={lag_column: 'Lag Time (frames)', 'autocorr': 'Autocorrelation'})
                            
                            # Add zero reference line
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add interpretation
                            if autocorr[1] > 0.7:
                                st.info("Strong positive autocorrelation at short time scales suggests persistent motion.")
                            elif autocorr[1] < 0.3:
                                st.info("Weak autocorrelation suggests random or constrained motion.")
                            
                            # Find where autocorrelation crosses zero
                            try:
                                for i in range(1, len(autocorr)):
                                    if autocorr[i-1] > 0 and autocorr[i] <= 0:
                                        st.text(f"Velocity correlation time: approximately {lags[i]:.1f} frames")
                                        break
                            except:
                                pass
                        else:
                            st.info("Velocity autocorrelation data appears to be incomplete.")
                    else:
                        st.info("No velocity autocorrelation data available. Make sure you enabled this option in the motion analysis.")
                
                elif viz_type == "Motion Types":
                    if 'motion_types' in motion_results and isinstance(motion_results['motion_types'], pd.DataFrame) and not motion_results['motion_types'].empty:
                        # Plot distribution of motion types
                        fig = px.pie(motion_results['motion_types'], names='motion_type', values='count',
                                    title='Distribution of Motion Types')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also show a table with statistics for each motion type
                        if 'motion_type_stats' in motion_results and isinstance(motion_results['motion_type_stats'], pd.DataFrame):
                            st.subheader("Motion Type Statistics")
                            st.dataframe(motion_results['motion_type_stats'])
                        
                        # If track-level classifications are available, create a spatial map
                        if 'track_motion_types' in motion_results and not st.session_state.tracks_data.empty:
                            st.subheader("Spatial Distribution of Motion Types")
                            try:
                                # Get the mean position of each track
                                track_positions = st.session_state.tracks_data.groupby('track_id')[['x', 'y']].mean().reset_index()
                                motion_track_data = pd.merge(track_positions, motion_results['track_motion_types'], on='track_id')
                                
                                # Create a spatial map colored by motion type
                                spatial_fig = px.scatter(motion_track_data, x='x', y='y', color='motion_type',
                                                      title='Spatial Map of Motion Types',
                                                      labels={'x': 'X Position (µm)', 'y': 'Y Position (µm)'})
                                
                                spatial_fig.update_yaxes(autorange="reversed")  # Invert Y-axis for image-like coordinates
                                st.plotly_chart(spatial_fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating motion type spatial map: {str(e)}")
                    else:
                        st.info("No motion type classification available. Make sure you selected a motion classification method in the analysis.")
        
        # 3D Visualization tab
        with viz_tabs[3]:
            st.header("3D Visualization")
            
            # 3D visualization options
            st.subheader("Visualization Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                color_by = st.selectbox(
                    "Color By",
                    ["track_id", "frame", "x", "y", "Quality"],
                    key="3d_color_by",
                    help="Property to use for coloring tracks"
                )
                
                colormap = st.selectbox(
                    "Color Map",
                    ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "rainbow"],
                    key="3d_colormap",
                    help="Colormap for tracks"
                )
                
            with col2:
                max_tracks = st.slider(
                    "Max Tracks to Display",
                    min_value=1,
                    max_value=min(50, st.session_state.tracks_data['track_id'].nunique()),
                    value=min(20, st.session_state.tracks_data['track_id'].nunique()),
                    key="3d_max_tracks",
                    help="Maximum number of tracks to display (for better performance)"
                )
            
            # Filter tracks if needed
            if max_tracks < st.session_state.tracks_data['track_id'].nunique():
                unique_tracks = st.session_state.tracks_data['track_id'].unique()[:max_tracks]
                filtered_tracks = st.session_state.tracks_data[st.session_state.tracks_data['track_id'].isin(unique_tracks)]
            else:
                filtered_tracks = st.session_state.tracks_data
            
            # Generate 3D plot
            with st.spinner("Generating 3D visualization..."):
                try:
                    fig = plot_tracks_3d(
                        filtered_tracks,
                        color_by=color_by,
                        colormap=colormap
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating 3D visualization: {str(e)}")
        
        # Trajectory Heatmaps tab
        with viz_tabs[4]:
            create_streamlit_heatmap_interface()
        
        # Custom Visualization tab
        with viz_tabs[5]:
            st.header("Custom Visualization")
            
            st.write("Create a custom visualization by selecting data and plot type.")
            
            # Custom visualization options
            st.subheader("Data Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "X-axis",
                    ["track_id", "frame", "x", "y", "Quality"]
                )
                
                y_axis = st.selectbox(
                    "Y-axis",
                    ["y", "x", "frame", "track_id", "Quality"]
                )
                
            with col2:
                color_by = st.selectbox(
                    "Color By",
                    ["track_id", "frame", "x", "y", "Quality"],
                    key="custom_color_by"
                )
                
                plot_kind = st.selectbox(
                    "Plot Type",
                    ["scatter", "line", "histogram", "box", "violin", "heatmap"]
                )
            
            # Generate custom plot
            with st.spinner("Generating custom visualization..."):
                try:
                    # Create custom plot using Plotly
                    if plot_kind == "scatter":
                        fig = px.scatter(
                            st.session_state.tracks_data,
                            x=x_axis,
                            y=y_axis,
                            color=color_by,
                            title=f"{y_axis} vs {x_axis}",
                            labels={x_axis: x_axis, y_axis: y_axis}
                        )
                    elif plot_kind == "line":
                        fig = px.line(
                            st.session_state.tracks_data,
                            x=x_axis,
                            y=y_axis,
                            color=color_by,
                            title=f"{y_axis} vs {x_axis}",
                            labels={x_axis: x_axis, y_axis: y_axis}
                        )
                    elif plot_kind == "histogram":
                        fig = px.histogram(
                            st.session_state.tracks_data,
                            x=x_axis,
                            color=color_by,
                            title=f"Histogram of {x_axis}",
                            labels={x_axis: x_axis}
                        )
                    elif plot_kind == "box":
                        fig = px.box(
                            st.session_state.tracks_data,
                            x=x_axis,
                            y=y_axis,
                            color=color_by,
                            title=f"Box plot of {y_axis} by {x_axis}",
                            labels={x_axis: x_axis, y_axis: y_axis}
                        )
                    elif plot_kind == "violin":
                        fig = px.violin(
                            st.session_state.tracks_data,
                            x=x_axis,
                            y=y_axis,
                            color=color_by,
                            title=f"Violin plot of {y_axis} by {x_axis}",
                            labels={x_axis: x_axis, y_axis: y_axis}
                        )
                    elif plot_kind == "heatmap":
                        # Create 2D histogram for heatmap
                        fig = px.density_heatmap(
                            st.session_state.tracks_data,
                            x=x_axis,
                            y=y_axis,
                            title=f"Heatmap of {y_axis} vs {x_axis}",
                            labels={x_axis: x_axis, y_axis: y_axis}
                        )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating custom visualization: {str(e)}")

# Advanced Analysis page
elif st.session_state.active_page == "Advanced Analysis":
    st.title("Advanced Analysis")
    
    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
    else:
        # Create tabs for different advanced analysis modules
        adv_tabs = st.tabs([
            "Biophysical Models",
            "Advanced Biophysical",
            "Changepoint Detection", 
            "Correlative Analysis",
            "Advanced Tracking",
            "Intensity Analysis",
            "Microrheology",
            "CTRW Analysis",
            "FBM Analysis",
            "Advanced Metrics",
            "Statistical Tests",
            "Ornstein-Uhlenbeck",
            "HMM Analysis",
            "DDM (Tracking-Free)"
        ])
        
        # HMM Analysis tab
        with adv_tabs[12]:
            st.header("Hidden Markov Model (HMM) Analysis")
            st.write("Model track dynamics using a Hidden Markov Model to identify distinct movement states.")

            if st.session_state.tracks_data is not None:
                if fit_hmm is None:
                    st.warning(_hmm_warning())
                else:
                    # proceed with HMM analysis using fit_hmm
                    pass  # ...existing code...
                # Parameters
                st.subheader("HMM Parameters")
                n_states = st.slider("Number of Hidden States", 2, 10, 3, 1)
                n_iter = st.slider("Number of Iterations", 10, 200, 100, 10)

                if st.button("Run HMM Analysis"):
                    with st.spinner("Fitting HMM..."):
                        model, predictions = fit_hmm(st.session_state.tracks_data, n_states, n_iter)
                        st.session_state.analysis_results["hmm"] = {
                            "model": model,
                            "predictions": predictions
                        }
                        st.success("HMM analysis complete!")

                if "hmm" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["hmm"]
                    model = results["model"]
                    predictions = results["predictions"]

                    st.subheader("HMM Results")
                    st.write("Fitted Model Parameters:")
                    st.write("Means (dx, dy):")
                    st.write(model.means_)
                    st.write("Covariances:")
                    covars = getattr(model, "covars_", None)
                    cov_type = getattr(model, "covariance_type", "full")
                    if covars is None:
                        st.info("No covariance matrix available.")
                    else:
                        try:
                            if cov_type == "full":
                                # Shape: (n_components, n_features, n_features)
                                n_components = covars.shape[0]
                                for i in range(n_components):
                                    st.caption(f"State {i} covariance")
                                    df_cov = pd.DataFrame(covars[i])
                                    st.dataframe(df_cov)
                            elif cov_type == "tied":
                                # Shape: (n_features, n_features)
                                st.dataframe(pd.DataFrame(covars))
                            elif cov_type == "diag":
                                # Shape: (n_components, n_features)
                                df_diag = pd.DataFrame(covars, columns=[f"f{j}" for j in range(covars.shape[1])])
                                df_diag.index = [f"state_{i}" for i in range(covars.shape[0])]
                                st.dataframe(df_diag)
                            elif cov_type == "spherical":
                                # Shape: (n_components,)
                                df_sph = pd.DataFrame({"variance": covars})
                                df_sph.index = [f"state_{i}" for i in range(len(covars))]
                                st.dataframe(df_sph)
                            else:
                                # Fallback: display as text/array
                                st.write(covars)
                        except Exception:
                            # As a last resort, show a readable string
                            import numpy as _np
                            st.text(_np.array2string(_np.asarray(covars), precision=4, suppress_small=True))
                    st.write("Transition Matrix:")
                    st.write(model.transmat_)

                    st.subheader("State Predictions")
                    # Add state predictions to the tracks dataframe for visualization
                    tracks_with_states = st.session_state.tracks_data.copy()
                    state_assignments = []
                    for i, row in tracks_with_states.iterrows():
                        track_id = row['track_id']
                        frame = row['frame']
                        if track_id in predictions:
                            # The predictions are for displacements, so we need to align them with the frames
                            # The first frame of a track has no displacement, so we can assign it the state of the first displacement
                            track_predictions = predictions[track_id]
                            if frame > tracks_with_states[tracks_with_states['track_id'] == track_id]['frame'].min():
                                state_idx = frame - tracks_with_states[tracks_with_states['track_id'] == track_id]['frame'].min() -1
                                if state_idx < len(track_predictions):
                                    state_assignments.append(track_predictions[state_idx])
                                else:
                                    state_assignments.append(np.nan) # Or some other indicator for missing state
                            else:
                                state_assignments.append(track_predictions[0])
                        else:
                            state_assignments.append(np.nan)

                    # This part is tricky because the number of states is not the same as the number of points
                    # I will simplify and just show the predictions for the first few tracks
                    st.write("State sequences for the first 5 tracks:")
                    for i, (track_id, states) in enumerate(predictions.items()):
                        if i >= 5:
                            break
                        st.write(f"Track {track_id}: {states}")
            
            # iHMM (Infinite HMM) Section
            st.divider()
            with st.expander("🔮 iHMM (Infinite HMM) - Automatic State Selection", expanded=False):
                st.markdown("""
                **Blur-Aware iHMM with Automatic State Detection**
                
                Enhanced HMM that:
                - ✅ **Corrects for motion blur** from finite exposure time
                - ✅ **Automatically determines** optimal number of states (BIC/AIC)
                - ✅ **Classifies states** as Bound/Diffusive/Fast based on D values
                - ✅ **Accounts for localization noise** in diffusion estimates
                
                **When to use iHMM:**
                - Unknown number of distinct states
                - High exposure time (R = exposure/interval > 0.3)
                - Precise diffusion coefficient estimation needed
                - Multi-state dynamics (e.g., chromatin binding, receptor switching)
                """)
                
                try:
                    from ihmm_analysis import analyze_track_with_ihmm, InfiniteHMM
                    
                    st.subheader("iHMM Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Get current units
                        units = get_units()
                        pixel_size = st.number_input(
                            f"Pixel Size ({units['length']})",
                            min_value=0.001,
                            max_value=10.0,
                            value=0.1,
                            step=0.01,
                            format="%.3f",
                            key="ihmm_pixel_size"
                        )
                        
                        frame_interval = st.number_input(
                            f"Frame Interval ({units['time']})",
                            min_value=0.001,
                            max_value=10.0,
                            value=0.1,
                            step=0.01,
                            format="%.3f",
                            key="ihmm_frame_interval"
                        )
                        
                        exposure_time = st.number_input(
                            f"Exposure Time ({units['time']})",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.08,
                            step=0.01,
                            format="%.3f",
                            help="Camera exposure time. Set to 0 for no blur correction.",
                            key="ihmm_exposure_time"
                        )
                    
                    with col2:
                        localization_error = st.number_input(
                            f"Localization Precision ({units['length']})",
                            min_value=0.001,
                            max_value=1.0,
                            value=0.03,
                            step=0.005,
                            format="%.3f",
                            help="Static localization error (σ_loc)",
                            key="ihmm_loc_error"
                        )
                        
                        min_states = st.number_input(
                            "Minimum States",
                            min_value=2,
                            max_value=5,
                            value=2,
                            step=1,
                            key="ihmm_min_states"
                        )
                        
                        max_states = st.number_input(
                            "Maximum States",
                            min_value=2,
                            max_value=10,
                            value=4,
                            step=1,
                            key="ihmm_max_states"
                        )
                        
                        selection_method = st.selectbox(
                            "Model Selection",
                            options=["BIC", "AIC"],
                            index=0,
                            help="BIC penalizes complexity more than AIC",
                            key="ihmm_method"
                        )
                    
                    # Calculate blur fraction
                    R = exposure_time / frame_interval if frame_interval > 0 else 0
                    blur_correction_factor = 1.0 - (R ** 2) / 12.0
                    
                    st.info(f"📊 Blur fraction R = {R:.2f} | Correction factor = {blur_correction_factor:.3f}")
                    
                    if R > 0.5:
                        st.warning("⚠️ High blur fraction (R > 0.5) may affect accuracy. Consider shorter exposure time.")
                    
                    # Track selection
                    st.subheader("Track Selection")
                    
                    tracks_df, has_data = get_track_data()
                    
                    if not has_data:
                        st.error("No track data available.")
                    else:
                        track_ids = sorted(tracks_df['track_id'].unique())
                        
                        selected_track = st.selectbox(
                            "Select Track",
                            options=track_ids,
                            key="ihmm_track_selection"
                        )
                        
                        track_data = tracks_df[tracks_df['track_id'] == selected_track]
                        st.caption(f"Track length: {len(track_data)} frames")
                        
                        if len(track_data) < 10:
                            st.error("Track too short (need ≥10 frames)")
                        else:
                            if st.button("🚀 Run iHMM Analysis", type="primary", key="run_ihmm"):
                                with st.spinner("Fitting iHMM with automatic state selection..."):
                                    try:
                                        result = analyze_track_with_ihmm(
                                            track=track_data,
                                            pixel_size=pixel_size,
                                            frame_interval=frame_interval,
                                            exposure_time=exposure_time if exposure_time > 0 else None,
                                            localization_error=localization_error,
                                            min_states=int(min_states),
                                            max_states=int(max_states),
                                            method=selection_method
                                        )
                                        
                                        if result['success']:
                                            st.session_state.ihmm_result = result
                                            st.success(f"✅ iHMM complete! Selected {result['best_n_states']} states.")
                                        else:
                                            st.error(f"❌ iHMM failed: {result.get('error', 'Unknown error')}")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Error during iHMM: {str(e)}")
                                        import traceback
                                        st.code(traceback.format_exc())
                    
                    # Display results if available
                    if hasattr(st.session_state, 'ihmm_result') and st.session_state.ihmm_result is not None:
                        result = st.session_state.ihmm_result
                        
                        st.divider()
                        st.subheader("📊 iHMM Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Selected States", result['best_n_states'])
                        with col2:
                            st.metric("Track Length", result['track_length'])
                        with col3:
                            st.metric("Log-Likelihood", f"{result['log_likelihood']:.1f}")
                        
                        # State classification table
                        st.subheader("State Classification")
                        
                        state_data = []
                        D_values = result['diffusion_coefficients']
                        state_labels = result['state_labels']
                        
                        for i, (D, label) in enumerate(zip(D_values, state_labels)):
                            state_data.append({
                                'State': i,
                                f'D ({units["diffusion"]})': f"{D:.4f}",
                                'Classification': label
                            })
                        
                        state_df = pd.DataFrame(state_data)
                        st.dataframe(state_df, use_container_width=True)
                        
                        # Transition matrix
                        st.subheader("State Transition Matrix")
                        transition_df = pd.DataFrame(
                            result['transition_matrix'],
                            columns=[f"→ State {i}" for i in range(result['best_n_states'])],
                            index=[f"State {i} →" for i in range(result['best_n_states'])]
                        )
                        st.dataframe(transition_df.style.format("{:.3f}"), use_container_width=True)
                        
                        # Model selection scores
                        st.subheader("Model Selection Scores")
                        
                        score_data = []
                        for n_states, score_info in result['model_scores'].items():
                            score_data.append({
                                'n_states': n_states,
                                f'{selection_method} Score': score_info['score'],
                                'Log-Likelihood': score_info['log_likelihood'],
                                'Selected': '✓' if n_states == result['best_n_states'] else ''
                            })
                        
                        score_df = pd.DataFrame(score_data)
                        st.dataframe(score_df, use_container_width=True)
                        
                        # Plot: D values comparison
                        import plotly.graph_objects as go
                        
                        fig_D = go.Figure()
                        
                        for n_states, score_info in result['model_scores'].items():
                            D_vals = score_info['D_values']
                            fig_D.add_trace(go.Scatter(
                                x=list(range(n_states)),
                                y=D_vals,
                                mode='lines+markers',
                                name=f"{n_states} states" + (" (selected)" if n_states == result['best_n_states'] else ""),
                                line=dict(width=3) if n_states == result['best_n_states'] else dict(width=1)
                            ))
                        
                        fig_D.update_layout(
                            title="Diffusion Coefficients vs. Model Complexity",
                            xaxis_title="State Index",
                            yaxis_title=f"D ({units['diffusion']})",
                            yaxis_type="log",
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig_D, use_container_width=True)
                        
                        # Plot: State trajectory
                        state_sequence = result['state_sequence']
                        
                        # Get positions for the track
                        track_data_viz = tracks_df[tracks_df['track_id'] == result.get('track_id', selected_track)]
                        positions = track_data_viz[['x', 'y']].values * pixel_size
                        
                        # Color by state (skip first position since state sequence is for displacements)
                        fig_traj = go.Figure()
                        
                        # Plot positions colored by state
                        for state_idx in range(result['best_n_states']):
                            # Find segments in this state (state_sequence corresponds to displacements)
                            state_mask = (state_sequence == state_idx)
                            
                            # Extract positions for this state (add 1 to index since states are for displacements)
                            state_positions_x = []
                            state_positions_y = []
                            
                            for i, is_state in enumerate(state_mask):
                                if is_state:
                                    # Displacement i connects position i to position i+1
                                    state_positions_x.extend([positions[i, 0], positions[i+1, 0], None])
                                    state_positions_y.extend([positions[i, 1], positions[i+1, 1], None])
                            
                            if len(state_positions_x) > 0:
                                fig_traj.add_trace(go.Scatter(
                                    x=state_positions_x,
                                    y=state_positions_y,
                                    mode='lines+markers',
                                    name=f"State {state_idx} ({state_labels[state_idx]})",
                                    line=dict(width=2),
                                    marker=dict(size=4)
                                ))
                        
                        fig_traj.update_layout(
                            title="Track Trajectory Colored by State",
                            xaxis_title=f"X ({units['length']})",
                            yaxis_title=f"Y ({units['length']})",
                            hovermode="closest"
                        )
                        
                        st.plotly_chart(fig_traj, use_container_width=True)
                        
                        # State duration histogram
                        st.subheader("State Duration Distribution")
                        
                        # Calculate run lengths for each state
                        state_durations = {i: [] for i in range(result['best_n_states'])}
                        
                        current_state = state_sequence[0]
                        current_duration = 1
                        
                        for i in range(1, len(state_sequence)):
                            if state_sequence[i] == current_state:
                                current_duration += 1
                            else:
                                state_durations[current_state].append(current_duration * frame_interval)
                                current_state = state_sequence[i]
                                current_duration = 1
                        
                        # Add last run
                        state_durations[current_state].append(current_duration * frame_interval)
                        
                        # Plot duration histograms
                        fig_duration = go.Figure()
                        
                        for state_idx in range(result['best_n_states']):
                            if len(state_durations[state_idx]) > 0:
                                fig_duration.add_trace(go.Histogram(
                                    x=state_durations[state_idx],
                                    name=f"State {state_idx}",
                                    opacity=0.7
                                ))
                        
                        fig_duration.update_layout(
                            title="State Duration Distribution",
                            xaxis_title=f"Duration ({units['time']})",
                            yaxis_title="Count",
                            barmode='overlay'
                        )
                        
                        st.plotly_chart(fig_duration, use_container_width=True)
                
                except ImportError:
                    st.error("❌ iHMM module not available. Ensure ihmm_analysis.py is in the project directory.")
                except Exception as e:
                    st.error(f"❌ Error loading iHMM interface: {str(e)}")

        # DDM (Tracking-Free) tab
        with adv_tabs[13]:
            st.header("🔬 Differential Dynamic Microscopy (DDM)")
            st.markdown("""
            **Tracking-free diffusion analysis** from microscopy images.
            
            DDM extracts dynamics by analyzing image fluctuations in Fourier space, without detecting or tracking individual particles.
            
            **When to use DDM:**
            - Dense samples where particles overlap
            - High concentration (>10 particles/field)
            - Validation of tracking-based methods
            - Fast dynamics below tracking resolution
            """)
            
            try:
                from ddm_analyzer import DDMAnalyzer
                
                # Check if images are loaded
                if st.session_state.image_data is None:
                    st.info("📁 Load microscopy images to perform DDM analysis.")
                    
                    # File uploader
                    ddm_uploaded_file = st.file_uploader(
                        "Upload image stack (TIFF, PNG sequence)",
                        type=["tif", "tiff"],
                        key="ddm_image_upload",
                        help="Upload a multi-frame TIFF or image sequence"
                    )
                    
                    if ddm_uploaded_file is not None:
                        try:
                            from data_loader import load_image_file
                            images = load_image_file(ddm_uploaded_file)
                            st.session_state.ddm_images = images
                            st.success(f"✅ Loaded {len(images)} frames")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading images: {str(e)}")
                else:
                    # Use currently loaded images
                    if st.button("Use Currently Loaded Images"):
                        st.session_state.ddm_images = st.session_state.image_data
                        st.success(f"✅ Using {len(st.session_state.image_data)} loaded frames")
                        st.rerun()
                
                # If images are available, show analysis interface
                if 'ddm_images' in st.session_state and st.session_state.ddm_images is not None:
                    images = st.session_state.ddm_images
                    n_frames = len(images) if isinstance(images, list) else images.shape[0]
                    
                    st.success(f"📊 Image stack loaded: {n_frames} frames")
                    
                    # Convert to numpy array if needed
                    if isinstance(images, list):
                        images = np.array(images)
                    
                    # Display image info
                    with st.expander("📸 Image Stack Information", expanded=False):
                        st.write(f"**Frames:** {images.shape[0]}")
                        st.write(f"**Size:** {images.shape[1]} × {images.shape[2]} pixels")
                        st.write(f"**Data type:** {images.dtype}")
                        st.write(f"**Intensity range:** [{images.min():.1f}, {images.max():.1f}]")
                        
                        # Show preview
                        col_prev1, col_prev2 = st.columns(2)
                        with col_prev1:
                            st.image(images[0], caption="First frame", use_container_width=True)
                        with col_prev2:
                            mid_frame = images[images.shape[0] // 2]
                            st.image(mid_frame, caption="Middle frame", use_container_width=True)
                    
                    # Parameters
                    st.subheader("⚙️ Analysis Parameters")
                    
                    param_col1, param_col2 = st.columns(2)
                    
                    with param_col1:
                        ddm_pixel_size = st.number_input(
                            "Pixel Size (µm)",
                            min_value=0.01,
                            max_value=1.0,
                            value=float(st.session_state.get('pixel_size', 0.1)),
                            step=0.01,
                            format="%.3f",
                            key="ddm_pixel_size",
                            help="Pixel size in micrometers"
                        )
                        
                        ddm_frame_interval = st.number_input(
                            "Frame Interval (s)",
                            min_value=0.001,
                            max_value=10.0,
                            value=float(st.session_state.get('frame_interval', 0.1)),
                            step=0.01,
                            format="%.3f",
                            key="ddm_frame_interval",
                            help="Time between frames in seconds"
                        )
                        
                        ddm_particle_radius = st.number_input(
                            "Particle Radius (µm)",
                            min_value=0.01,
                            max_value=5.0,
                            value=0.5,
                            step=0.1,
                            format="%.2f",
                            key="ddm_particle_radius",
                            help="Approximate particle radius for rheology calculations"
                        )
                    
                    with param_col2:
                        # Lag times
                        max_lag_suggestion = min(n_frames // 4, 100)
                        ddm_max_lag = st.slider(
                            "Maximum Lag Time (frames)",
                            min_value=5,
                            max_value=min(n_frames // 2, 200),
                            value=min(50, max_lag_suggestion),
                            step=5,
                            key="ddm_max_lag",
                            help="Maximum time lag for structure function"
                        )
                        
                        ddm_n_lag_times = st.slider(
                            "Number of Lag Times",
                            min_value=10,
                            max_value=50,
                            value=20,
                            step=5,
                            key="ddm_n_lag_times",
                            help="Number of logarithmically-spaced lag times"
                        )
                        
                        ddm_chunk_size = st.slider(
                            "Chunk Size (memory management)",
                            min_value=10,
                            max_value=200,
                            value=50,
                            step=10,
                            key="ddm_chunk_size",
                            help="Process frames in chunks to manage memory"
                        )
                    
                    # Advanced options
                    with st.expander("🔧 Advanced Options", expanded=False):
                        ddm_subtract_bg = st.checkbox(
                            "Subtract Background",
                            value=True,
                            key="ddm_subtract_bg",
                            help="Subtract temporal median background before analysis"
                        )
                        
                        ddm_q_range_auto = st.checkbox(
                            "Auto q-range",
                            value=True,
                            key="ddm_q_range_auto",
                            help="Automatically determine wave vector range"
                        )
                        
                        if not ddm_q_range_auto:
                            q_min = st.number_input("Min q (µm⁻¹)", value=0.5, step=0.1)
                            q_max = st.number_input("Max q (µm⁻¹)", value=10.0, step=0.5)
                            ddm_q_range = (q_min, q_max)
                        else:
                            ddm_q_range = None
                    
                    # Run analysis
                    if st.button("🚀 Run DDM Analysis", type="primary", key="run_ddm"):
                        with st.spinner("Computing image structure function D(q, Δt)..."):
                            try:
                                # Initialize analyzer
                                analyzer = DDMAnalyzer(
                                    pixel_size_um=ddm_pixel_size,
                                    frame_interval_s=ddm_frame_interval
                                )
                                
                                # Generate lag times
                                lag_frames = np.unique(
                                    np.logspace(0, np.log10(ddm_max_lag), ddm_n_lag_times).astype(int)
                                )
                                lag_frames = lag_frames[lag_frames > 0]
                                
                                # Run analysis
                                with st.status("DDM Analysis Pipeline", expanded=True) as status:
                                    st.write("1️⃣ Computing structure function...")
                                    ddm_result = analyzer.compute_image_structure_function(
                                        image_stack=images,
                                        lag_frames=lag_frames.tolist(),
                                        q_range_um_inv=ddm_q_range,
                                        subtract_background=ddm_subtract_bg
                                    )
                                    
                                    if not ddm_result['success']:
                                        st.error(f"❌ DDM failed: {ddm_result.get('error', 'Unknown error')}")
                                        status.update(label="DDM Analysis Failed", state="error")
                                    else:
                                        st.write("2️⃣ Extracting MSD from structure function...")
                                        msd_result = analyzer.extract_msd_from_structure_function(ddm_result)
                                        
                                        st.write("3️⃣ Computing rheology (if applicable)...")
                                        rheology_result = analyzer.compute_rheology_from_ddm(
                                            msd_result,
                                            particle_radius_um=ddm_particle_radius
                                        )
                                        
                                        # Store results
                                        st.session_state.ddm_results = {
                                            'ddm': ddm_result,
                                            'msd': msd_result,
                                            'rheology': rheology_result,
                                            'analyzer': analyzer
                                        }
                                        
                                        status.update(label="✅ DDM Analysis Complete!", state="complete")
                                        st.success("✅ DDM analysis completed successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Error during DDM analysis: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    # Display results if available
                    if 'ddm_results' in st.session_state and st.session_state.ddm_results is not None:
                        results = st.session_state.ddm_results
                        ddm_res = results['ddm']
                        msd_res = results['msd']
                        rheo_res = results.get('rheology', {})
                        
                        st.markdown("---")
                        st.subheader("📊 DDM Results")
                        
                        # Summary metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            D_val = msd_res.get('diffusion_coeff_um2_s', np.nan)
                            if not np.isnan(D_val):
                                st.metric(
                                    "Diffusion Coefficient",
                                    f"{D_val:.4f} µm²/s",
                                    help="Extracted from MSD vs time"
                                )
                            else:
                                st.metric("Diffusion Coefficient", "N/A")
                        
                        with metric_col2:
                            alpha_val = msd_res.get('alpha_exponent', np.nan)
                            if not np.isnan(alpha_val):
                                st.metric(
                                    "Anomalous Exponent α",
                                    f"{alpha_val:.2f}",
                                    help="α=1: Brownian, α<1: subdiffusive, α>1: superdiffusive"
                                )
                            else:
                                st.metric("Anomalous Exponent α", "N/A")
                        
                        with metric_col3:
                            n_analyzed = ddm_res.get('n_frames_analyzed', 0)
                            st.metric(
                                "Frames Analyzed",
                                f"{n_analyzed}",
                                help="Number of frames used in analysis"
                            )
                        
                        # Visualization tabs
                        ddm_viz_tabs = st.tabs([
                            "Structure Function D(q,τ)",
                            "MSD vs Time",
                            "Rheology"
                        ])
                        
                        # Structure function visualization
                        with ddm_viz_tabs[0]:
                            st.subheader("Image Structure Function D(q, τ)")
                            
                            try:
                                import plotly.graph_objects as go
                                
                                D_q_tau = ddm_res['D_q_tau']
                                q_values = ddm_res['q_values_um_inv']
                                lag_times = ddm_res['lag_times_s']
                                
                                fig = go.Figure(data=go.Heatmap(
                                    x=q_values,
                                    y=lag_times,
                                    z=D_q_tau,
                                    colorscale='Viridis',
                                    colorbar=dict(title='D(q,τ)')
                                ))
                                
                                fig.update_layout(
                                    title='Image Structure Function',
                                    xaxis_title='Wave vector q (µm⁻¹)',
                                    yaxis_title='Lag time τ (s)',
                                    xaxis_type='log',
                                    yaxis_type='log',
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Visualization error: {str(e)}")
                        
                        # MSD visualization
                        with ddm_viz_tabs[1]:
                            st.subheader("Mean Square Displacement from DDM")
                            
                            if 'msd_vs_time' in msd_res:
                                try:
                                    msd_data = msd_res['msd_vs_time']
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=msd_data['lag_times_s'],
                                        y=msd_data['msd_um2'],
                                        mode='markers',
                                        name='DDM MSD',
                                        marker=dict(size=8)
                                    ))
                                    
                                    # Add fit if available
                                    if 'msd_fit_um2' in msd_data:
                                        fig.add_trace(go.Scatter(
                                            x=msd_data['lag_times_s'],
                                            y=msd_data['msd_fit_um2'],
                                            mode='lines',
                                            name=f'Fit: MSD = {D_val:.4f}·t^{alpha_val:.2f}',
                                            line=dict(color='red', dash='dash')
                                        ))
                                    
                                    fig.update_layout(
                                        title='MSD from DDM Analysis',
                                        xaxis_title='Lag time τ (s)',
                                        yaxis_title='MSD (µm²)',
                                        xaxis_type='log',
                                        yaxis_type='log',
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show fit quality
                                    if 'r_squared' in msd_res:
                                        st.info(f"Fit quality: R² = {msd_res['r_squared']:.3f}")
                                    
                                except Exception as e:
                                    st.error(f"MSD visualization error: {str(e)}")
                            else:
                                st.warning("MSD data not available")
                        
                        # Rheology visualization
                        with ddm_viz_tabs[2]:
                            st.subheader("Microrheology from DDM")
                            
                            if rheo_res.get('success'):
                                try:
                                    freq = rheo_res['frequency_rad_s']
                                    G_prime = rheo_res['G_prime_Pa']
                                    G_double_prime = rheo_res['G_double_prime_Pa']
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=freq,
                                        y=G_prime,
                                        mode='lines+markers',
                                        name="G' (elastic)",
                                        line=dict(color='blue')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=freq,
                                        y=G_double_prime,
                                        mode='lines+markers',
                                        name='G" (viscous)',
                                        line=dict(color='red')
                                    ))
                                    
                                    fig.update_layout(
                                        title='Complex Modulus from DDM',
                                        xaxis_title='Frequency ω (rad/s)',
                                        yaxis_title='Modulus (Pa)',
                                        xaxis_type='log',
                                        yaxis_type='log',
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"Rheology visualization error: {str(e)}")
                            else:
                                st.info("Rheology analysis not available or failed")
                        
                        # Export options
                        with st.expander("💾 Export Results", expanded=False):
                            if st.button("Export DDM Data as CSV"):
                                try:
                                    # Create DataFrame with key results
                                    export_df = pd.DataFrame({
                                        'q_um_inv': ddm_res['q_values_um_inv'],
                                        'lag_time_s': ddm_res['lag_times_s'][:len(ddm_res['q_values_um_inv'])],
                                    })
                                    
                                    csv = export_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="ddm_results.csv",
                                        mime="text/csv"
                                    )
                                except Exception as e:
                                    st.error(f"Export error: {str(e)}")
            
            except ImportError:
                st.error("❌ DDM analyzer module not available. Check installation.")
            except Exception as e:
                st.error(f"❌ Error in DDM interface: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # Biophysical Models tab
        with adv_tabs[0]:
            st.header("Biophysical Models")
            
            if BIOPHYSICAL_MODELS_AVAILABLE:
                # Create subtabs for different biophysical models
                model_tabs = st.tabs([
                    "Polymer Physics", 
                    "Active Transport", 
                    "Energy Landscape",
                    "Percolation Analysis",
                    "CTRW & FBM"
                ])
                
                # Polymer Physics Model
                with model_tabs[0]:
                    st.header("Polymer Physics Model")
                    
                    # Model selection
                    st.subheader("Model Selection")
                    polymer_model_type = st.selectbox(
                        "Select Polymer Model",
                        ["Rouse (α = 0.5)", "Zimm (α = 2/3)", "Reptation (α = 0.25-0.5)", "Auto-fit α"],
                        index=3,
                        help="Choose which polymer physics model to fit"
                    )
                    
                    # Parameters
                    st.subheader("Model Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "Reptation" in polymer_model_type:
                            tube_diameter = st.number_input(
                                "Tube Diameter (nm)",
                                min_value=10.0,
                                max_value=500.0,
                                value=100.0,
                                step=10.0,
                                help="Tube diameter for reptation model"
                            )
                            
                            contour_length = st.number_input(
                                "Contour Length (nm)",
                                min_value=100.0,
                                max_value=10000.0,
                                value=1000.0,
                                step=100.0,
                                help="Polymer contour length in nanometers"
                            )
                        elif "Zimm" in polymer_model_type:
                            solvent_viscosity = st.number_input(
                                "Solvent Viscosity (mPa·s)",
                                min_value=0.1,
                                max_value=10.0,
                                value=1.0,
                                step=0.1,
                                help="Solvent viscosity (1.0 = water at 25°C)"
                            )
                            
                            hydrodynamic_radius = st.number_input(
                                "Hydrodynamic Radius (nm)",
                                min_value=1.0,
                                max_value=100.0,
                                value=5.0,
                                step=1.0,
                                help="Hydrodynamic radius of polymer segment"
                            )
                        else:
                            persistence_length = st.number_input(
                                "Persistence Length (nm)",
                                min_value=1.0,
                                max_value=1000.0,
                                value=50.0,
                                step=1.0,
                                help="Persistence length in nanometers"
                            )
                            
                            contour_length = st.number_input(
                                "Contour Length (nm)",
                                min_value=10.0,
                                max_value=10000.0,
                                value=1000.0,
                                step=10.0,
                                help="Contour length in nanometers"
                            )
                        
                    with col2:
                        temperature = st.number_input(
                            "Temperature (K)",
                            min_value=250.0,
                            max_value=350.0,
                            value=300.0,
                            step=5.0,
                            help="Temperature in Kelvin"
                        )
                        
                        fit_alpha = st.checkbox(
                            "Fit α exponent",
                            value=True if "Auto-fit" in polymer_model_type else False,
                            help="If checked, fit the scaling exponent α. Otherwise use theoretical value."
                        )
                    
                    # Additional analyses
                    st.subheader("Additional Analyses")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        calculate_fractal = st.checkbox(
                            "Calculate Fractal Dimension",
                            value=True,
                            help="Analyze trajectory fractal dimension (Df) using box-counting"
                        )
                        
                        apply_crowding = st.checkbox(
                            "Correct for Macromolecular Crowding",
                            value=False,
                            help="Apply crowding corrections to estimate free diffusion coefficient"
                        )
                    
                    with col4:
                        if apply_crowding:
                            phi_crowding = st.slider(
                                "Crowding Volume Fraction (φ)",
                                min_value=0.1,
                                max_value=0.5,
                                value=0.3,
                                step=0.05,
                                help="Typical nuclear: 0.2-0.4"
                            )
                        else:
                            phi_crowding = 0.3
                    
                    # Run analysis on button click
                    if st.button("Run Polymer Physics Analysis"):
                        with st.spinner("Running polymer physics analysis..."):
                            try:
                                from biophysical_models import PolymerPhysicsModel
                                from analysis import calculate_msd
                                
                                # Get track data
                                tracks_df, has_data = get_track_data()
                                if not has_data:
                                    st.error("No track data available")
                                else:
                                    units = get_units()
                                    pixel_size_um = units.get('pixel_size', 0.1)
                                    frame_interval_s = units.get('frame_interval', 0.1)
                                    
                                    # Calculate MSD first
                                    msd_results = calculate_msd(
                                        tracks_df,
                                        pixel_size=pixel_size_um,
                                        frame_interval=frame_interval_s
                                    )
                                    
                                    # Handle different return types from calculate_msd
                                    if isinstance(msd_results, dict):
                                        if not msd_results.get('success'):
                                            st.error(f"MSD calculation failed: {msd_results.get('error')}")
                                        else:
                                            msd_df = msd_results.get('ensemble_msd')
                                    else:
                                        # Direct DataFrame return
                                        msd_df = msd_results
                                    
                                    if msd_df is not None and not (isinstance(msd_df, pd.DataFrame) and msd_df.empty):
                                        polymer_model = PolymerPhysicsModel(
                                            msd_data=msd_df,
                                            pixel_size=pixel_size_um,
                                            frame_interval=frame_interval_s
                                        )
                                        
                                        # Select and fit appropriate model
                                        if "Reptation" in polymer_model_type:
                                            model_results = polymer_model.fit_reptation_model(
                                                temperature=temperature,
                                                tube_diameter=tube_diameter * 1e-9,  # Convert nm to m
                                                contour_length=contour_length * 1e-9  # Convert nm to m
                                            )
                                            model_name = "Reptation"
                                        elif "Zimm" in polymer_model_type:
                                            model_results = polymer_model.fit_zimm_model(
                                                fit_alpha=fit_alpha,
                                                solvent_viscosity=solvent_viscosity * 0.001,  # Convert mPa·s to Pa·s
                                                hydrodynamic_radius=hydrodynamic_radius * 1e-9,  # Convert nm to m
                                                temperature=temperature
                                            )
                                            model_name = "Zimm"
                                        else:  # Rouse or Auto-fit
                                            model_results = polymer_model.fit_rouse_model(fit_alpha=fit_alpha)
                                            model_name = "Rouse"
                                        
                                        if model_results.get('success'):
                                            st.success(f"✓ {model_name} model analysis completed")
                                            
                                            # Display results
                                            params = model_results.get('parameters', {})
                                            
                                            # Show model-specific interpretation
                                            if 'interpretation' in model_results:
                                                st.info(f"**Interpretation**: {model_results['interpretation']}")
                                            elif 'regime' in params:
                                                st.info(f"**Regime**: {params['regime']}")
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                if 'alpha' in params:
                                                    st.metric("Alpha (Exponent)", f"{params['alpha']:.3f}")
                                                if 'K_rouse' in params:
                                                    st.metric("K (Rouse)", f"{params['K_rouse']:.6f}")
                                                if 'K_zimm' in params:
                                                    st.metric("K (Zimm)", f"{params['K_zimm']:.6f}")
                                                if 'K_reptation' in params:
                                                    st.metric("K (Reptation)", f"{params['K_reptation']:.6f}")
                                            with col2:
                                                if 'D_zimm_theory' in params:
                                                    st.metric("D (Zimm Theory)", f"{params['D_zimm_theory']:.3e} m²/s")
                                                if 'tube_diameter_estimated' in params:
                                                    st.metric("Tube Diameter (Est.)", f"{params['tube_diameter_estimated']*1e9:.1f} nm")
                                                if 'reptation_time' in params and params['reptation_time']:
                                                    st.metric("Reptation Time", f"{params['reptation_time']:.3e} s")
                                            
                                            # Store results
                                            if 'analysis_results' not in st.session_state:
                                                st.session_state.analysis_results = {}
                                            st.session_state.analysis_results['polymer_physics'] = model_results
                                            st.session_state.analysis_results['polymer_model_type'] = model_name
                                            
                                            # Show fitted curve if available
                                            if 'fitted_curve' in model_results:
                                                st.subheader("Model Fit")
                                                fig = go.Figure()
                                                
                                                # Original MSD
                                                fig.add_trace(go.Scatter(
                                                    x=msd_df['lag_time'],
                                                    y=msd_df['msd'],
                                                    mode='markers',
                                                    name='Measured MSD',
                                                    marker=dict(size=8, color='blue')
                                                ))
                                                
                                                # Fitted curve
                                                fit_data = model_results['fitted_curve']
                                                fig.add_trace(go.Scatter(
                                                    x=fit_data['lag_time'],
                                                    y=fit_data['msd_fit'],
                                                    mode='lines',
                                                    name=f'{model_name} Model Fit',
                                                    line=dict(color='red', width=2)
                                                ))
                                                
                                                fig.update_layout(
                                                    title=f"{model_name} Model Fit to MSD",
                                                    xaxis_title="Lag Time (s)",
                                                    yaxis_title="MSD (μm²)",
                                                    hovermode='closest'
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Fractal Dimension Analysis
                                            if calculate_fractal:
                                                st.subheader("Fractal Dimension Analysis")
                                                try:
                                                    fractal_results = polymer_model.analyze_fractal_dimension()
                                                    
                                                    if fractal_results.get('success'):
                                                        Df = fractal_results.get('fractal_dimension', 0)
                                                        
                                                        col_f1, col_f2 = st.columns(2)
                                                        with col_f1:
                                                            st.metric("Fractal Dimension (Df)", f"{Df:.3f}")
                                                        with col_f2:
                                                            st.info(fractal_results.get('interpretation', ''))
                                                        
                                                        # Store results
                                                        st.session_state.analysis_results['fractal_dimension'] = fractal_results
                                                    else:
                                                        st.warning(f"Fractal dimension calculation: {fractal_results.get('error', 'Unknown error')}")
                                                except Exception as e:
                                                    st.warning(f"Could not calculate fractal dimension: {str(e)}")
                                            
                                            # Crowding Correction
                                            if apply_crowding and 'alpha' in params:
                                                st.subheader("Macromolecular Crowding Correction")
                                                try:
                                                    # Extract D from fitted model
                                                    if 'D_eff' in params:
                                                        D_measured = params['D_eff']
                                                    elif 'K_rouse' in params:
                                                        # Estimate D from K: MSD = K*t^alpha => D ~ K/(4*t^(alpha-1))
                                                        D_measured = params['K_rouse'] / (4.0 * frame_interval_s**(params['alpha']-1))
                                                    else:
                                                        D_measured = 0.1  # Default guess
                                                    
                                                    crowding_results = polymer_model.correct_for_crowding(
                                                        D_measured=D_measured,
                                                        phi_crowding=phi_crowding
                                                    )
                                                    
                                                    if crowding_results.get('success'):
                                                        col_c1, col_c2, col_c3 = st.columns(3)
                                                        with col_c1:
                                                            st.metric("D (Measured)", f"{crowding_results['D_measured']:.3e} μm²/s")
                                                        with col_c2:
                                                            st.metric("D (Free)", f"{crowding_results['D_free']:.3e} μm²/s")
                                                        with col_c3:
                                                            st.metric("Crowding Factor", f"{crowding_results['crowding_factor']:.2%}")
                                                        
                                                        st.info(crowding_results.get('interpretation', ''))
                                                        
                                                        # Store results
                                                        st.session_state.analysis_results['crowding_correction'] = crowding_results
                                                    else:
                                                        st.warning(f"Crowding correction failed: {crowding_results.get('error')}")
                                                except Exception as e:
                                                    st.warning(f"Could not apply crowding correction: {str(e)}")
                                        else:
                                            st.error(f"Polymer analysis failed: {model_results.get('error', 'Unknown error')}")
                                    else:
                                        st.error("MSD calculation failed. Cannot proceed with polymer analysis.")
                            except Exception as e:
                                st.error(f"Error in polymer physics analysis: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                
                # Active Transport Model
                with model_tabs[1]:
                    st.header("Active Transport Analyzer")
                    
                    # Parameters
                    st.subheader("Model Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        speed_threshold = st.number_input(
                            "Speed Threshold (μm/s)",
                            min_value=0.01,
                            max_value=10.0,
                            value=0.5,
                            step=0.1,
                            help="Minimum speed to classify as active transport"
                        )
                        
                        straightness_threshold = st.number_input(
                            "Straightness Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.7,
                            step=0.05,
                            help="Minimum straightness (0-1) for directed motion"
                        )
                    
                    with col2:
                        min_track_length = st.number_input(
                            "Minimum Track Length",
                            min_value=5,
                            max_value=100,
                            value=10,
                            step=1,
                            help="Minimum number of frames for analysis"
                        )
                        
                        velocity_window = st.number_input(
                            "Velocity Window (frames)",
                            min_value=2,
                            max_value=20,
                            value=5,
                            step=1,
                            help="Window size for velocity calculations"
                        )
                    
                    # Run analysis
                    if st.button("Analyze Active Transport"):
                        with st.spinner("Analyzing active transport..."):
                            try:
                                from biophysical_models import ActiveTransportAnalyzer
                                
                                # Get track data
                                tracks_df, has_data = get_track_data()
                                if not has_data:
                                    st.error("No track data available")
                                else:
                                    units = get_units()
                                    pixel_size_um = units.get('pixel_size', 0.1)
                                    frame_interval_s = units.get('frame_interval', 0.1)
                                    
                                    # Initialize analyzer
                                    analyzer = ActiveTransportAnalyzer(
                                        tracks_df=tracks_df,
                                        pixel_size=pixel_size_um,
                                        frame_interval=frame_interval_s
                                    )
                                    
                                    # Detect active transport
                                    results = analyzer.detect_active_transport(
                                        speed_threshold=speed_threshold,
                                        straightness_threshold=straightness_threshold,
                                        min_track_length=min_track_length
                                    )
                                    
                                    if results.get('success'):
                                        st.success("✓ Active transport analysis completed")
                                        
                                        # Display summary
                                        st.subheader("Summary")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Total Tracks",
                                                results['summary']['total_tracks']
                                            )
                                        with col2:
                                            st.metric(
                                                "Active Tracks",
                                                results['summary']['active_tracks']
                                            )
                                        with col3:
                                            active_pct = (results['summary']['active_tracks'] / 
                                                        results['summary']['total_tracks'] * 100)
                                            st.metric(
                                                "Active %",
                                                f"{active_pct:.1f}%"
                                            )
                                        
                                        # Display statistics
                                        if 'statistics' in results:
                                            st.subheader("Active Transport Statistics")
                                            stats = results['statistics']
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("Mean Speed (μm/s)", 
                                                        f"{stats.get('mean_speed', 0):.3f}")
                                                st.metric("Mean Straightness", 
                                                        f"{stats.get('mean_straightness', 0):.3f}")
                                            with col2:
                                                st.metric("Max Speed (μm/s)", 
                                                        f"{stats.get('max_speed', 0):.3f}")
                                                st.metric("Mean Path Length (μm)", 
                                                        f"{stats.get('mean_path_length', 0):.2f}")
                                        
                                        # Store results
                                        if 'analysis_results' not in st.session_state:
                                            st.session_state.analysis_results = {}
                                        st.session_state.analysis_results['active_transport'] = results
                                        
                                        # Show classification plot if available
                                        if 'classified_tracks' in results:
                                            st.subheader("Track Classification")
                                            classified_df = results['classified_tracks']
                                            
                                            # Create scatter plot
                                            fig = go.Figure()
                                            
                                            # Passive tracks
                                            passive = classified_df[classified_df['is_active'] == False]
                                            fig.add_trace(go.Scatter(
                                                x=passive['mean_speed'],
                                                y=passive['straightness'],
                                                mode='markers',
                                                name='Passive',
                                                marker=dict(size=8, color='blue', opacity=0.6)
                                            ))
                                            
                                            # Active tracks
                                            active = classified_df[classified_df['is_active'] == True]
                                            fig.add_trace(go.Scatter(
                                                x=active['mean_speed'],
                                                y=active['straightness'],
                                                mode='markers',
                                                name='Active',
                                                marker=dict(size=8, color='red', opacity=0.6)
                                            ))
                                            
                                            # Add threshold lines
                                            fig.add_hline(
                                                y=straightness_threshold,
                                                line_dash="dash",
                                                line_color="gray",
                                                annotation_text="Straightness Threshold"
                                            )
                                            fig.add_vline(
                                                x=speed_threshold,
                                                line_dash="dash",
                                                line_color="gray",
                                                annotation_text="Speed Threshold"
                                            )
                                            
                                            fig.update_layout(
                                                title="Track Classification: Speed vs Straightness",
                                                xaxis_title="Mean Speed (μm/s)",
                                                yaxis_title="Straightness",
                                                hovermode='closest'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error in active transport analysis: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                
                # Energy Landscape Model
                with model_tabs[2]:
                    st.header("Energy Landscape Mapper")
                    
                    # Parameters
                    st.subheader("Model Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        resolution = st.slider(
                            "Grid Resolution",
                            min_value=10,
                            max_value=50,
                            value=20,
                            step=5,
                            help="Number of bins in each dimension for energy landscape"
                        )
                        
                        method = st.selectbox(
                            "Calculation Method",
                            ["boltzmann", "drift", "kramers"],
                            index=0,
                            help="Method for calculating energy landscape"
                        )
                    
                    with col2:
                        smoothing = st.slider(
                            "Smoothing Factor",
                            min_value=0.0,
                            max_value=2.0,
                            value=0.5,
                            step=0.1,
                            help="Gaussian smoothing applied to landscape"
                        )
                        
                        temperature = st.number_input(
                            "Temperature (K)",
                            min_value=250.0,
                            max_value=350.0,
                            value=300.0,
                            step=5.0,
                            help="Temperature in Kelvin for energy calculations"
                        )
                    
                    normalize = st.checkbox(
                        "Normalize to kBT units",
                        value=True,
                        help="Express energies in units of thermal energy (kBT)"
                    )
                    
                    # Run analysis
                    if st.button("Map Energy Landscape"):
                        with st.spinner("Mapping energy landscape..."):
                            try:
                                from biophysical_models import EnergyLandscapeMapper
                                
                                # Get track data
                                tracks_df, has_data = get_track_data()
                                if not has_data:
                                    st.error("No track data available")
                                else:
                                    units = get_units()
                                    pixel_size_um = units.get('pixel_size', 0.1)
                                    
                                    # Initialize mapper
                                    mapper = EnergyLandscapeMapper(
                                        tracks_df=tracks_df,
                                        pixel_size=pixel_size_um,
                                        temperature=temperature
                                    )
                                    
                                    # Map landscape
                                    results = mapper.map_energy_landscape(
                                        resolution=resolution,
                                        method=method,
                                        smoothing=smoothing,
                                        normalize=normalize
                                    )
                                    
                                    if results.get('success'):
                                        st.success("✓ Energy landscape mapping completed")
                                        
                                        # Display statistics
                                        st.subheader("Landscape Statistics")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        stats = results.get('statistics', {})
                                        with col1:
                                            st.metric(
                                                "Min Energy",
                                                f"{stats.get('min_energy', 0):.2f} kBT"
                                            )
                                        with col2:
                                            st.metric(
                                                "Max Energy",
                                                f"{stats.get('max_energy', 0):.2f} kBT"
                                            )
                                        with col3:
                                            st.metric(
                                                "Energy Range",
                                                f"{stats.get('energy_range', 0):.2f} kBT"
                                            )
                                        
                                        # Store results
                                        if 'analysis_results' not in st.session_state:
                                            st.session_state.analysis_results = {}
                                        st.session_state.analysis_results['energy_landscape'] = results
                                        
                                        # Visualize energy landscape
                                        if 'energy_map' in results:
                                            st.subheader("Energy Landscape")
                                            
                                            energy_map = results['energy_map']
                                            x_coords = results['x_coords']
                                            y_coords = results['y_coords']
                                            
                                            # Create heatmap
                                            fig = go.Figure(data=go.Heatmap(
                                                z=energy_map.T,
                                                x=x_coords,
                                                y=y_coords,
                                                colorscale='Viridis',
                                                colorbar=dict(
                                                    title="Energy (kBT)" if normalize else "Energy (J)"
                                                )
                                            ))
                                            
                                            fig.update_layout(
                                                title=f"Energy Landscape ({method} method)",
                                                xaxis_title="X Position (μm)",
                                                yaxis_title="Y Position (μm)",
                                                hovermode='closest'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                        # Show force field if available
                                        if 'force_field' in results:
                                            st.subheader("Force Field")
                                            
                                            force_x = results['force_field']['fx']
                                            force_y = results['force_field']['fy']
                                            x_coords = results['x_coords']
                                            y_coords = results['y_coords']
                                            
                                            # Create quiver plot (subsample for visibility)
                                            step = max(1, resolution // 10)
                                            fig = go.Figure()
                                            
                                            for i in range(0, len(x_coords), step):
                                                for j in range(0, len(y_coords), step):
                                                    fig.add_trace(go.Scatter(
                                                        x=[x_coords[i], x_coords[i] + force_x[i, j] * 0.1],
                                                        y=[y_coords[j], y_coords[j] + force_y[i, j] * 0.1],
                                                        mode='lines',
                                                        line=dict(color='red', width=1),
                                                        showlegend=False,
                                                        hoverinfo='skip'
                                                    ))
                                            
                                            fig.update_layout(
                                                title="Force Field (Negative Gradient of Energy)",
                                                xaxis_title="X Position (μm)",
                                                yaxis_title="Y Position (μm)",
                                                hovermode='closest'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.error(f"Mapping failed: {results.get('error', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error in energy landscape mapping: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                    
                    # Local Diffusion Map D(x,y)
                    st.markdown("---")
                    st.subheader("Local Diffusion Coefficient Map D(x,y)")
                    st.markdown("""
                    Map spatially-resolved diffusion coefficients to identify heterogeneous diffusion environments.
                    This complements the energy landscape by revealing local mobility variations.
                    """)
                    
                    show_diffusion_map = st.checkbox(
                        "Show Local Diffusion Map",
                        value=False,
                        help="Calculate and display D(x,y) map overlaid with energy landscape"
                    )
                    
                    if show_diffusion_map:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            diff_map_resolution = st.slider(
                                "D(x,y) Grid Resolution",
                                min_value=5,
                                max_value=20,
                                value=10,
                                step=1,
                                help="Number of grid cells in each dimension"
                            )
                        
                        with col2:
                            min_points_per_cell = st.slider(
                                "Min Points per Cell",
                                min_value=5,
                                max_value=50,
                                value=10,
                                step=5,
                                help="Minimum number of points required for reliable D calculation"
                            )
                        
                        if st.button("Calculate Local Diffusion Map"):
                            with st.spinner("Calculating local diffusion coefficients..."):
                                try:
                                    from biophysical_models import PolymerPhysicsModel
                                    
                                    tracks_df, has_data = get_track_data()
                                    if not has_data:
                                        st.error("No track data available")
                                    else:
                                        units = get_units()
                                        pixel_size_um = units.get('pixel_size', 0.1)
                                        frame_interval_s = units.get('frame_interval', 0.1)
                                        
                                        # Calculate D(x,y) map
                                        model = PolymerPhysicsModel(
                                            pixel_size=pixel_size_um,
                                            frame_interval=frame_interval_s
                                        )
                                        
                                        diffusion_map_results = model.calculate_local_diffusion_map(
                                            tracks_df=tracks_df,
                                            grid_resolution=diff_map_resolution,
                                            min_points=min_points_per_cell
                                        )
                                        
                                        if diffusion_map_results['success']:
                                            st.success("✓ Local diffusion map calculated")
                                            
                                            D_map = diffusion_map_results['D_map']
                                            confidence_map = diffusion_map_results['confidence_map']
                                            x_centers = diffusion_map_results['x_coords']
                                            y_centers = diffusion_map_results['y_coords']
                                            
                                            # Statistics
                                            st.subheader("Diffusion Map Statistics")
                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            valid_D = D_map[~np.isnan(D_map)]
                                            if len(valid_D) > 0:
                                                with col1:
                                                    st.metric("Min D", f"{np.min(valid_D):.3e} μm²/s")
                                                with col2:
                                                    st.metric("Max D", f"{np.max(valid_D):.3e} μm²/s")
                                                with col3:
                                                    st.metric("Mean D", f"{np.mean(valid_D):.3e} μm²/s")
                                                with col4:
                                                    st.metric("Std D", f"{np.std(valid_D):.3e} μm²/s")
                                            
                                            # Visualize D(x,y) map
                                            st.subheader("Diffusion Coefficient Map")
                                            
                                            fig = go.Figure()
                                            
                                            # D(x,y) heatmap
                                            fig.add_trace(go.Heatmap(
                                                z=D_map.T,
                                                x=x_centers,
                                                y=y_centers,
                                                colorscale='Plasma',
                                                colorbar=dict(
                                                    title="D (μm²/s)",
                                                    x=1.15
                                                ),
                                                hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>D: %{z:.2e} μm²/s<extra></extra>'
                                            ))
                                            
                                            fig.update_layout(
                                                title="Local Diffusion Coefficient Map D(x,y)",
                                                xaxis_title="X Position (μm)",
                                                yaxis_title="Y Position (μm)",
                                                hovermode='closest',
                                                height=600
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Confidence map
                                            st.subheader("Confidence Map")
                                            st.markdown("Shows the reliability of D estimates based on number of data points per cell")
                                            
                                            fig_conf = go.Figure()
                                            fig_conf.add_trace(go.Heatmap(
                                                z=confidence_map.T,
                                                x=x_centers,
                                                y=y_centers,
                                                colorscale='Greens',
                                                colorbar=dict(title="Confidence Score"),
                                                hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>Confidence: %{z:.2f}<extra></extra>'
                                            ))
                                            
                                            fig_conf.update_layout(
                                                title="Confidence Map",
                                                xaxis_title="X Position (μm)",
                                                yaxis_title="Y Position (μm)",
                                                hovermode='closest',
                                                height=500
                                            )
                                            st.plotly_chart(fig_conf, use_container_width=True)
                                            
                                            # Store results
                                            if 'analysis_results' not in st.session_state:
                                                st.session_state.analysis_results = {}
                                            st.session_state.analysis_results['local_diffusion_map'] = diffusion_map_results
                                            
                                        else:
                                            st.error(f"Diffusion map calculation failed: {diffusion_map_results.get('error', 'Unknown error')}")
                                
                                except Exception as e:
                                    st.error(f"Error calculating local diffusion map: {str(e)}")
                                    import traceback
                                    st.text(traceback.format_exc())
                
                # Percolation Analysis
                with model_tabs[3]:
                    st.header("Percolation Analysis")
                    st.write("Analyze connectivity and phase transitions in particle distributions.")
                    
                    # Parameters
                    st.subheader("Analysis Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        percolation_method = st.selectbox(
                            "Estimation Method",
                            ["Connectivity", "Density", "MSD Transition"],
                            help="Method for estimating percolation threshold"
                        )
                        
                        distance_threshold = st.number_input(
                            "Connection Distance (μm)",
                            min_value=0.01,
                            max_value=10.0,
                            value=0.5,
                            step=0.1,
                            help="Distance below which particles are considered connected (auto if None)"
                        )
                        
                        use_auto_distance = st.checkbox(
                            "Auto-estimate distance",
                            value=True,
                            help="Automatically estimate connection distance from data"
                        )
                    
                    with col2:
                        show_network = st.checkbox(
                            "Show Connection Network",
                            value=True,
                            help="Visualize connectivity network"
                        )
                        
                        color_by = st.selectbox(
                            "Color Nodes By",
                            ["Cluster", "Degree"],
                            help="Node coloring scheme"
                        )
                    
                    if st.button("Run Percolation Analysis"):
                        with st.spinner("Analyzing percolation properties..."):
                            try:
                                from percolation_analyzer import PercolationAnalyzer
                                
                                # Get track data
                                tracks_df, has_data = get_track_data()
                                if not has_data:
                                    st.error("No track data available")
                                else:
                                    units = get_units()
                                    pixel_size_um = units.get('pixel_size', 0.1)
                                    
                                    analyzer = PercolationAnalyzer(tracks_df, pixel_size=pixel_size_um)
                                    
                                    # Estimate percolation threshold
                                    st.subheader("Percolation Threshold Estimation")
                                    
                                    method_map = {
                                        "Connectivity": "connectivity",
                                        "Density": "density",
                                        "MSD Transition": "msd_transition"
                                    }
                                    
                                    dist_threshold = None if use_auto_distance else distance_threshold
                                    
                                    threshold_results = analyzer.estimate_percolation_threshold(
                                        method=method_map[percolation_method],
                                        distance_threshold=dist_threshold
                                    )
                                    
                                    # Display results
                                    if threshold_results['is_percolating']:
                                        st.success("✓ System is PERCOLATING (above threshold)")
                                    else:
                                        st.warning("System is NOT percolating (below threshold)")
                                    
                                    col_p1, col_p2, col_p3 = st.columns(3)
                                    with col_p1:
                                        st.metric("Density", f"{threshold_results['density']:.2f} particles/μm²")
                                    with col_p2:
                                        st.metric("P(percolation)", f"{threshold_results['percolation_probability']:.2f}")
                                    with col_p3:
                                        st.metric("Confidence", threshold_results['confidence'].upper())
                                    
                                    # Connectivity network analysis
                                    st.subheader("Connectivity Network")
                                    
                                    network_results = analyzer.analyze_connectivity_network(
                                        distance_threshold=dist_threshold
                                    )
                                    
                                    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                                    with col_n1:
                                        st.metric("Nodes", network_results['num_nodes'])
                                    with col_n2:
                                        st.metric("Edges", network_results['num_edges'])
                                    with col_n3:
                                        st.metric("Clusters", network_results['num_clusters'])
                                    with col_n4:
                                        st.metric("Largest Cluster", network_results['largest_cluster_size'])
                                    
                                    if network_results['spanning_cluster']:
                                        st.success("✓ Spanning cluster detected!")
                                    
                                    # Cluster size distribution
                                    st.subheader("Cluster Size Distribution")
                                    
                                    cluster_dist = analyzer.calculate_cluster_size_distribution(
                                        distance_threshold=dist_threshold
                                    )
                                    
                                    if not np.isnan(cluster_dist['tau_exponent']):
                                        st.info(f"Power-law exponent τ = {cluster_dist['tau_exponent']:.2f} " +
                                               f"(Theory: 2D={cluster_dist['theoretical_tau_2d']}, " +
                                               f"3D={cluster_dist['theoretical_tau_3d']})")
                                    
                                    # Plot cluster size distribution
                                    fig_dist = go.Figure()
                                    fig_dist.add_trace(go.Scatter(
                                        x=cluster_dist['cluster_sizes'],
                                        y=cluster_dist['probabilities'],
                                        mode='markers',
                                        name='Data',
                                        marker=dict(size=8, color='blue')
                                    ))
                                    
                                    fig_dist.update_layout(
                                        title="Cluster Size Distribution",
                                        xaxis_title="Cluster Size (s)",
                                        yaxis_title="Probability P(s)",
                                        xaxis_type="log",
                                        yaxis_type="log",
                                        hovermode='closest'
                                    )
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                    
                                    # Visualize network
                                    if show_network:
                                        st.subheader("Connectivity Network Visualization")
                                        
                                        fig_network = analyzer.visualize_percolation_map(
                                            distance_threshold=dist_threshold,
                                            show_connections=True,
                                            color_by=color_by.lower()
                                        )
                                        st.plotly_chart(fig_network, use_container_width=True)
                                    
                                    # Store results
                                    if 'analysis_results' not in st.session_state:
                                        st.session_state.analysis_results = {}
                                    st.session_state.analysis_results['percolation'] = {
                                        'threshold': threshold_results,
                                        'network': network_results,
                                        'cluster_distribution': cluster_dist
                                    }
                            
                            except Exception as e:
                                st.error(f"Error in percolation analysis: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                
                # CTRW & FBM Analysis
                with model_tabs[4]:
                    st.header("CTRW & Fractional Brownian Motion")
                    st.write("Analyze Continuous Time Random Walk properties and Fractional Brownian Motion.")
                    
                    analysis_type = st.radio(
                        "Analysis Type",
                        ["CTRW Analysis", "FBM Fitting"],
                        horizontal=True
                    )
                    
                    if analysis_type == "CTRW Analysis":
                        st.subheader("Continuous Time Random Walk Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            min_pause_threshold = st.number_input(
                                "Pause Threshold (μm)",
                                min_value=0.001,
                                max_value=1.0,
                                value=0.01,
                                step=0.01,
                                format="%.3f",
                                help="Minimum displacement considered as movement"
                            )
                        
                        with col2:
                            fit_distribution = st.selectbox(
                                "Distribution Type",
                                ["Auto", "Exponential", "Power-law"],
                                help="Type of distribution to fit"
                            )
                        
                        if st.button("Run CTRW Analysis"):
                            with st.spinner("Analyzing CTRW properties..."):
                                try:
                                    from advanced_diffusion_models import CTRWAnalyzer
                                    
                                    # Get track data
                                    tracks_df, has_data = get_track_data()
                                    if not has_data:
                                        st.error("No track data available")
                                    else:
                                        units = get_units()
                                        pixel_size_um = units.get('pixel_size', 0.1)
                                        frame_interval_s = units.get('frame_interval', 0.1)
                                        
                                        analyzer = CTRWAnalyzer(tracks_df, pixel_size_um, frame_interval_s)
                                        
                                        # Waiting time analysis
                                        st.subheader("Waiting Time Distribution")
                                        
                                        wait_results = analyzer.analyze_waiting_time_distribution(
                                            min_pause_threshold=min_pause_threshold,
                                            fit_distribution=fit_distribution.lower()
                                        )
                                        
                                        col_w1, col_w2, col_w3 = st.columns(3)
                                        with col_w1:
                                            st.metric("Distribution Type", wait_results['distribution_type'].capitalize())
                                        with col_w2:
                                            st.metric("Mean Wait Time", f"{wait_results['mean_waiting_time']:.3f} s")
                                        with col_w3:
                                            if not np.isnan(wait_results['alpha_exponent']):
                                                st.metric("Power-law α", f"{wait_results['alpha_exponent']:.2f}")
                                        
                                        if wait_results['is_heavy_tailed']:
                                            st.warning("⚠ Heavy-tailed distribution detected (α < 3) → CTRW behavior")
                                        
                                        # Plot waiting time distribution
                                        if len(wait_results['waiting_times']) > 0:
                                            fig_wait = go.Figure()
                                            
                                            # Histogram
                                            fig_wait.add_trace(go.Histogram(
                                                x=wait_results['waiting_times'],
                                                nbinsx=50,
                                                name='Waiting Times',
                                                histnorm='probability'
                                            ))
                                            
                                            fig_wait.update_layout(
                                                title="Waiting Time Distribution",
                                                xaxis_title="Waiting Time (s)",
                                                yaxis_title="Probability",
                                                hovermode='closest'
                                            )
                                            st.plotly_chart(fig_wait, use_container_width=True)
                                        
                                        # Jump length analysis
                                        st.subheader("Jump Length Distribution")
                                        
                                        jump_results = analyzer.analyze_jump_length_distribution(
                                            fit_distribution=fit_distribution.lower()
                                        )
                                        
                                        col_j1, col_j2, col_j3 = st.columns(3)
                                        with col_j1:
                                            st.metric("Distribution Type", jump_results['distribution_type'].capitalize())
                                        with col_j2:
                                            st.metric("Mean Jump Length", f"{jump_results['mean_jump_length']:.3f} μm")
                                        with col_j3:
                                            if not np.isnan(jump_results['levy_exponent']):
                                                st.metric("Levy β", f"{jump_results['levy_exponent']:.2f}")
                                        
                                        if jump_results['is_levy_flight']:
                                            st.warning("⚠ Levy flight detected (0 < β < 2)")
                                        
                                        # Plot jump length distribution
                                        if len(jump_results['jump_lengths']) > 0:
                                            fig_jump = go.Figure()
                                            
                                            fig_jump.add_trace(go.Histogram(
                                                x=jump_results['jump_lengths'],
                                                nbinsx=50,
                                                name='Jump Lengths',
                                                histnorm='probability'
                                            ))
                                            
                                            fig_jump.update_layout(
                                                title="Jump Length Distribution",
                                                xaxis_title="Jump Length (μm)",
                                                yaxis_title="Probability",
                                                hovermode='closest'
                                            )
                                            st.plotly_chart(fig_jump, use_container_width=True)
                                        
                                        # Ergodicity test
                                        st.subheader("Ergodicity Test")
                                        
                                        ergodicity_results = analyzer.test_ergodicity(n_segments=4)
                                        
                                        if ergodicity_results['is_ergodic']:
                                            st.success("✓ System is ERGODIC")
                                        else:
                                            st.warning("⚠ System is NON-ERGODIC (aging detected)")
                                        
                                        col_e1, col_e2 = st.columns(2)
                                        with col_e1:
                                            st.metric("EB Parameter", f"{ergodicity_results['ergodicity_breaking_parameter']:.3f}")
                                        with col_e2:
                                            st.metric("Aging Coefficient", f"{ergodicity_results['aging_coefficient']:.3f}")
                                        
                                        # Coupling analysis
                                        st.subheader("Wait-Jump Coupling")
                                        
                                        coupling_results = analyzer.analyze_coupling(
                                            min_pause_threshold=min_pause_threshold
                                        )
                                        
                                        if coupling_results['is_coupled']:
                                            st.warning(f"⚠ Coupling detected (r={coupling_results['correlation_coefficient']:.2f}, p={coupling_results['p_value']:.3f})")
                                        else:
                                            st.info(f"No significant coupling (r={coupling_results['correlation_coefficient']:.2f}, p={coupling_results['p_value']:.3f})")
                                        
                                        # Store results
                                        if 'analysis_results' not in st.session_state:
                                            st.session_state.analysis_results = {}
                                        st.session_state.analysis_results['ctrw'] = {
                                            'waiting_times': wait_results,
                                            'jump_lengths': jump_results,
                                            'ergodicity': ergodicity_results,
                                            'coupling': coupling_results
                                        }
                                
                                except Exception as e:
                                    st.error(f"Error in CTRW analysis: {str(e)}")
                                    import traceback
                                    st.text(traceback.format_exc())
                    
                    else:  # FBM Fitting
                        st.subheader("Fractional Brownian Motion Fitting")
                        
                        st.write("Fit FBM model to characterize long-range correlations via Hurst exponent.")
                        
                        if st.button("Fit FBM Model"):
                            with st.spinner("Fitting FBM model..."):
                                try:
                                    from advanced_diffusion_models import fit_fbm_model
                                    
                                    # Get track data
                                    tracks_df, has_data = get_track_data()
                                    if not has_data:
                                        st.error("No track data available")
                                    else:
                                        units = get_units()
                                        pixel_size_um = units.get('pixel_size', 0.1)
                                        frame_interval_s = units.get('frame_interval', 0.1)
                                        
                                        fbm_results = fit_fbm_model(
                                            tracks_df,
                                            pixel_size=pixel_size_um,
                                            frame_interval=frame_interval_s
                                        )
                                        
                                        if fbm_results.get('success'):
                                            st.success("✓ FBM model fitted successfully")
                                            
                                            col_f1, col_f2, col_f3 = st.columns(3)
                                            with col_f1:
                                                st.metric("Hurst Exponent (H)", f"{fbm_results['hurst_exponent']:.3f}")
                                            with col_f2:
                                                st.metric("Diffusion Coeff.", f"{fbm_results['diffusion_coefficient']:.3e} μm²/s")
                                            with col_f3:
                                                st.metric("R²", f"{fbm_results['r_squared']:.3f}")
                                            
                                            st.info(f"**{fbm_results['persistence_type']}**: {fbm_results['interpretation']}")
                                            
                                            # Interpretation guide
                                            with st.expander("Hurst Exponent Interpretation"):
                                                st.write("""
                                                - **H = 0.5**: Standard Brownian motion (no memory)
                                                - **H < 0.5**: Anti-persistent (motion reverses, negative correlation)
                                                - **H > 0.5**: Persistent (motion continues, positive correlation)
                                                - **MSD scaling**: MSD ~ t^(2H)
                                                """)
                                            
                                            # Store results
                                            if 'analysis_results' not in st.session_state:
                                                st.session_state.analysis_results = {}
                                            st.session_state.analysis_results['fbm'] = fbm_results
                                        else:
                                            st.error(f"FBM fitting failed: {fbm_results.get('error', 'Unknown error')}")
                                
                                except Exception as e:
                                    st.error(f"Error in FBM fitting: {str(e)}")
                                    import traceback
                                    st.text(traceback.format_exc())
            else:
                st.warning("Biophysical models module is not available. Make sure the appropriate files are in the correct location.")
        
        # Microrheology Analysis tab
        with adv_tabs[3]:
            st.header("Microrheology Analysis")
            st.write("Calculate G' (storage modulus), G\" (loss modulus), and effective viscosity from particle tracking data.")
            
            if st.session_state.tracks_data is not None:
                # Parameters section
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    particle_radius_nm = st.number_input(
                        "Particle Radius (nm)", 
                        min_value=1.0, 
                        max_value=1000.0, 
                        value=50.0, 
                        step=1.0,
                        help="Radius of tracked particles in nanometers"
                    )
                    
                    temperature_K = st.number_input(
                        "Temperature (K)", 
                        min_value=200.0, 
                        max_value=400.0, 
                        value=300.0, 
                        step=1.0,
                        help="Temperature in Kelvin (default: 300K = 27°C)"
                    )
                    
                    max_lag_frames = st.slider(
                        "Max Lag Frames", 
                        min_value=10, 
                        max_value=100, 
                        value=50, 
                        help="Maximum lag time for MSD calculation"
                    )
                
                with col2:
                    units = get_current_units()
                    pixel_size_um = st.number_input(
                        "Pixel Size (µm)", 
                        min_value=0.01, 
                        max_value=10.0, 
                        value=units['pixel_size'], 
                        step=0.01,
                        key="rheology_pixel_size"
                    )
                    
                    frame_interval_s = st.number_input(
                        "Frame Interval (s)", 
                        min_value=0.001, 
                        max_value=10.0, 
                        value=units['frame_interval'], 
                        step=0.001,
                        key="rheology_frame_interval"
                    )
                    
                    use_multi_dataset = st.checkbox(
                        "Multi-Dataset Analysis", 
                        value=False,
                        help="Use multiple datasets with different sampling rates for comprehensive frequency-dependent analysis"
                    )
                
                # Multi-dataset parameters
                if use_multi_dataset:
                    st.subheader("Multi-Dataset Parameters")
                    st.info("📁 Upload multiple track files from the same sample with different sampling rates for comprehensive microrheology analysis.")
                    
                    # File uploader for multiple datasets
                    multi_files = st.file_uploader(
                        "Upload Additional Track Files",
                        type=['csv', 'xlsx', 'xml'],
                        accept_multiple_files=True,
                        help="Upload track files with different sampling rates from the same sample"
                    )
                    
                    # Frame intervals for each dataset
                    if multi_files:
                        st.write("**Frame Intervals for Each Dataset:**")
                        
                        # Include current dataset
                        all_datasets = ["Current Dataset"] + [f.name for f in multi_files]
                        frame_intervals = []
                        
                        for i, dataset_name in enumerate(all_datasets):
                            if i == 0:
                                # Current dataset uses the frame interval from above
                                st.write(f"**{dataset_name}:** {frame_interval_s:.3f} s")
                                frame_intervals.append(frame_interval_s)
                            else:
                                interval = st.number_input(
                                    f"Frame Interval for {dataset_name} (s)",
                                    min_value=0.001,
                                    max_value=10.0,
                                    value=0.1 if i == 1 else 0.01,
                                    step=0.001,
                                    key=f"interval_{i}",
                                    help=f"Frame interval for {dataset_name}"
                                )
                                frame_intervals.append(interval)
                    else:
                        st.warning("Please upload additional track files to enable multi-dataset analysis.")
                
                # Analysis button
                if st.button("Run Microrheology Analysis", key="rheology_analysis_btn_1"):
                    with st.spinner("Performing microrheology analysis..."):
                        try:
                            # Initialize analyzer
                            particle_radius_m = particle_radius_nm * 1e-9  # Convert to meters
                            analyzer = MicrorheologyAnalyzer(
                                particle_radius_m=particle_radius_m,
                                temperature_K=temperature_K
                            )
                            
                            if use_multi_dataset and 'multi_files' in locals() and multi_files:
                                # Multi-dataset analysis with different sampling rates
                                try:
                                    # Load additional datasets
                                    track_datasets = [st.session_state.tracks_data]  # Include current dataset
                                    
                                    for uploaded_file in multi_files:
                                        # Load each additional file
                                        additional_tracks = load_tracks_file(uploaded_file)
                                        if additional_tracks is not None and not additional_tracks.empty:
                                            track_datasets.append(additional_tracks)
                                        else:
                                            st.warning(f"Could not load tracks from {uploaded_file.name}")
                                    
                                    if len(track_datasets) > 1:
                                        # Run multi-dataset analysis
                                        analysis_results = analyzer.multi_dataset_analysis(
                                            track_datasets=track_datasets,
                                            frame_intervals_s=frame_intervals,
                                            pixel_size_um=pixel_size_um
                                        )
                                    else:
                                        st.error("Need at least 2 datasets for multi-dataset analysis")
                                        analysis_results = {'success': False, 'error': 'Insufficient datasets'}
                                        
                                except Exception as e:
                                    st.error(f"Error loading additional datasets: {str(e)}")
                                    analysis_results = {'success': False, 'error': f'Dataset loading error: {str(e)}'}
                            else:
                                # Single dataset analysis - use the proper analysis method
                                analysis_results = analyzer.analyze_microrheology(
                                    st.session_state.tracks_data,
                                    pixel_size_um=pixel_size_um,
                                    frame_interval_s=frame_interval_s,
                                    max_lag=max_lag_frames
                                )
                            
                            # Store results
                            st.session_state.analysis_results["microrheology"] = analysis_results
                            
                            # Create analysis record
                            analysis_record = create_analysis_record(
                                name="Microrheology Analysis",
                                analysis_type="microrheology",
                                parameters={
                                    "particle_radius_nm": particle_radius_nm,
                                    "temperature_K": temperature_K,
                                    "pixel_size_um": pixel_size_um,
                                    "frame_interval_s": frame_interval_s,
                                    "multi_dataset": use_multi_dataset,
                                    "num_datasets": len(track_datasets) if use_multi_dataset and 'track_datasets' in locals() else 1
                                }
                            )
                            
                            # Add to recent analyses
                            st.session_state.recent_analyses.append(analysis_record)
                            
                            if analysis_results.get('success', False):
                                st.success("Microrheology analysis completed successfully!")
                            else:
                                st.error(f"Analysis failed: {analysis_results.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Error in microrheology analysis: {str(e)}")
                
                # Display results
                if "microrheology" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["microrheology"]
                    
                    if results.get('success', False):
                        # Display summary
                        display_rheology_summary(results)
                        
                        # Create and display plots
                        try:
                            figures = create_rheology_plots(results)
                            
                            if 'msd_comparison' in figures:
                                st.subheader("Mean Squared Displacement")
                                st.plotly_chart(figures['msd_comparison'], use_container_width=True)
                            
                            if 'frequency_response' in figures:
                                st.subheader("Combined Frequency Response")
                                st.plotly_chart(figures['frequency_response'], use_container_width=True)
                            
                            if 'individual_frequency_response' in figures:
                                st.subheader("Individual Dataset Frequency Responses")
                                st.plotly_chart(figures['individual_frequency_response'], use_container_width=True)
                                
                        except Exception as e:
                            st.warning(f"Could not generate plots: {str(e)}")
                        
                        # Show detailed results
                        with st.expander("Detailed Results"):
                            if 'msd_data' in results:
                                st.subheader("MSD Data")
                                st.dataframe(results['msd_data'])
                            
                            if 'frequency_response' in results:
                                st.subheader("Frequency Response Data")
                                freq_data = results['frequency_response']
                                freq_df = pd.DataFrame({
                                    'Frequency (Hz)': freq_data.get('frequencies_hz', []),
                                    'G\' (Pa)': freq_data.get('g_prime_pa', []),
                                    'G" (Pa)': freq_data.get('g_double_prime_pa', [])
                                })
                                st.dataframe(freq_df)
                    else:
                        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
            else:
                st.warning("No track data available. Please load track data first.")
        
        # Advanced Biophysical Analysis tab
        with adv_tabs[1]:
            st.header("🔬 Advanced Biophysical Analysis")
            st.write("Comprehensive biophysical characterization: polymer dynamics, viscoelasticity, and active matter analysis")
            
            try:
                from advanced_biophysical import show_advanced_biophysical
                show_advanced_biophysical()
            except ImportError as e:
                st.error(f"Advanced biophysical module not available: {e}")
                st.info("This module requires additional dependencies for polymer dynamics and viscoelasticity analysis.")
            except Exception as e:
                st.error(f"Error loading advanced biophysical analysis: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
        # Changepoint Detection tab
        with adv_tabs[2]:
            st.header("Changepoint Detection")
            
            if CHANGEPOINT_DETECTION_AVAILABLE:
                # Parameters
                st.subheader("Detection Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    method = st.selectbox(
                        "Detection Method",
                        ["pelt", "window", "binary"],
                        help="Method for detecting changepoints"
                    )
                    
                    feature = st.selectbox(
                        "Feature for Detection",
                        ["step_size", "angle_change", "msd"],
                        help="Feature to use for changepoint detection"
                    )
                    
                with col2:
                    window_size = st.slider(
                        "Window Size",
                        min_value=5,
                        max_value=50,
                        value=10,
                        help="Window size for calculating local motion statistics"
                    )
                    
                    significance_level = st.slider(
                        "Significance Level",
                        min_value=0.01,
                        max_value=0.10,
                        value=0.05,
                        step=0.01,
                        help="Statistical significance level for changepoint detection"
                    )
                
                min_segment_length = st.slider(
                    "Minimum Segment Length",
                    min_value=3,
                    max_value=20,
                    value=5,
                    key="cp_min_segment_length",
                    help="Minimum length of segments between changepoints"
                )
                
                # Run analysis on button click
                if st.button("Run Changepoint Detection"):
                    with st.spinner("Running changepoint detection..."):
                        try:
                            from changepoint_detection import ChangePointDetector
                            
                            # Get track data
                            tracks_df, has_data = get_track_data()
                            if not has_data:
                                st.error("No track data available")
                            else:
                                # Initialize detector
                                detector = ChangePointDetector()
                                
                                # Run detection with user parameters
                                changepoint_results = detector.detect_motion_regime_changes(
                                    tracks_df=tracks_df,
                                    window_size=window_size,
                                    min_segment_length=min_segment_length,
                                    significance_level=significance_level
                                )
                                
                                if changepoint_results.get('success', True):
                                    st.success("✓ Changepoint detection completed")
                                    
                                    # Display motion segments
                                    if 'motion_segments' in changepoint_results:
                                        st.subheader("Motion Segments")
                                        segments_data = changepoint_results['motion_segments']
                                        if isinstance(segments_data, pd.DataFrame) and not segments_data.empty:
                                            st.dataframe(segments_data.head(20))
                                            
                                            # Summary statistics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Total Segments", len(segments_data))
                                            with col2:
                                                st.metric("Tracks Analyzed", segments_data['track_id'].nunique())
                                            with col3:
                                                st.metric("Avg Segment Length", f"{segments_data['duration'].mean():.1f} frames")
                                        else:
                                            st.info("No motion segments detected.")
                                    
                                    # Display changepoints
                                    if 'changepoints' in changepoint_results:
                                        st.subheader("Detected Changepoints")
                                        cp_data = changepoint_results['changepoints']
                                        if isinstance(cp_data, pd.DataFrame) and not cp_data.empty:
                                            st.dataframe(cp_data)
                                        else:
                                            st.info("No changepoints detected.")
                                    
                                    # Display regime classification
                                    if 'regime_classification' in changepoint_results:
                                        regime_data = changepoint_results['regime_classification']
                                        if isinstance(regime_data, dict) and regime_data.get('success'):
                                            st.subheader("Motion Regime Classification")
                                            if 'regime_counts' in regime_data:
                                                st.bar_chart(regime_data['regime_counts'])
                                    
                                    # Store results in session state
                                    if 'analysis_results' not in st.session_state:
                                        st.session_state.analysis_results = {}
                                    st.session_state.analysis_results['changepoints'] = changepoint_results
                                else:
                                    st.error(f"Changepoint detection failed: {changepoint_results.get('error', 'Unknown error')}")
                        except ImportError:
                            st.error("Changepoint detection module not available.")
                        except Exception as e:
                            st.error(f"Error in changepoint detection: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
            else:
                st.warning("Changepoint detection module is not available. Make sure the appropriate files are in the correct location.")
        
        # Correlative Analysis tab
        with adv_tabs[3]:
            st.header("Correlative Analysis")
            
            if CORRELATIVE_ANALYSIS_AVAILABLE:
                # Channel labeling section
                st.subheader("Channel Configuration")
                
                # Detect intensity columns
                intensity_columns = [col for col in st.session_state.tracks_data.columns 
                                   if 'intensity' in col.lower() or 'ch' in col.lower()]
                
                if intensity_columns:
                    st.write(f"Found {len(intensity_columns)} intensity channels:")
                    
                    # Initialize channel labels in session state if not exists
                    if 'channel_labels' not in st.session_state:
                        st.session_state.channel_labels = {}
                    
                    # Create input fields for each channel
                    col1, col2 = st.columns(2)
                    
                    for i, channel in enumerate(intensity_columns):
                        with col1 if i % 2 == 0 else col2:
                            current_label = st.session_state.channel_labels.get(channel, "")
                            label = st.text_input(
                                f"Label for {channel}",
                                value=current_label,
                                placeholder="e.g., DNA, splicing factor, protein X",
                                key=f"label_{channel}",
                                help=f"Enter a descriptive name for {channel}"
                            )
                            if label != current_label:
                                st.session_state.channel_labels[channel] = label
                    
                    # Show current labeling
                    if any(st.session_state.channel_labels.values()):
                        st.write("**Current Channel Labels:**")
                        for channel, label in st.session_state.channel_labels.items():
                            if label:
                                st.write(f"• {channel} → {label}")
                    
                    st.divider()
                else:
                    st.warning("No intensity channels detected in the data.")
                
                # Analysis parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                with col1:
                    lag_range = st.slider(
                        "Lag Range (frames)",
                        min_value=1,
                        max_value=20,
                        value=5,
                        help="Range of time lags for temporal correlation analysis"
                    )
                
                with col2:
                    min_track_length = st.number_input(
                        "Minimum Track Length",
                        min_value=5,
                        max_value=100,
                        value=10,
                        help="Minimum number of frames for analysis"
                    )
                
                # Run correlative analysis
                if st.button("Run Correlative Analysis", type="primary"):
                    with st.spinner("Running correlative analysis..."):
                        try:
                            from correlative_analysis import CorrelativeAnalyzer
                            
                            # Get track data
                            tracks_df, has_data = get_track_data()
                            if not has_data:
                                st.error("No track data available")
                            else:
                                # Check if we have intensity data
                                intensity_columns = [col for col in tracks_df.columns 
                                                   if 'intensity' in col.lower() or 'ch' in col.lower()]
                                
                                if not intensity_columns:
                                    st.warning("No intensity channels detected in the data. Load data with intensity information.")
                                else:
                                    # Initialize analyzer
                                    analyzer = CorrelativeAnalyzer()
                                    
                                    # Run analysis
                                    corr_results = analyzer.analyze_intensity_motion_coupling(
                                        tracks_df=tracks_df,
                                        intensity_columns=intensity_columns,
                                        lag_range=lag_range
                                    )
                                    
                                    st.success("✓ Correlative analysis completed")
                                    
                                    # Display track coupling results
                                    if 'track_coupling' in corr_results and corr_results['track_coupling']:
                                        st.subheader("Track Coupling Analysis")
                                        coupling_df = pd.DataFrame(corr_results['track_coupling'])
                                        
                                        # Apply custom channel labels if available
                                        channel_labels = st.session_state.get('channel_labels', {})
                                        if channel_labels:
                                            new_columns = []
                                            for col in coupling_df.columns:
                                                new_col = col
                                                for original_channel, label in channel_labels.items():
                                                    if original_channel in col and label:
                                                        new_col = new_col.replace(original_channel, label)
                                                new_columns.append(new_col)
                                            coupling_df.columns = new_columns
                                        
                                        st.dataframe(coupling_df.head(20))
                                        
                                        # Summary statistics
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Tracks Analyzed", len(coupling_df))
                                        with col2:
                                            corr_cols = [c for c in coupling_df.columns if 'correlation' in c.lower()]
                                            if corr_cols:
                                                mean_corr = coupling_df[corr_cols[0]].mean()
                                                st.metric("Mean Correlation", f"{mean_corr:.3f}")
                                        with col3:
                                            st.metric("Intensity Channels", len(intensity_columns))
                                    
                                    # Display ensemble correlations
                                    if 'ensemble_correlations' in corr_results and corr_results['ensemble_correlations']:
                                        st.subheader("Ensemble Correlations")
                                        
                                        for channel, correlation in corr_results['ensemble_correlations'].items():
                                            display_name = st.session_state.get('channel_labels', {}).get(channel, channel)
                                            st.metric(f"{display_name}", f"{correlation:.3f}")
                                    
                                    # Display lag correlations
                                    if 'lag_correlations' in corr_results and corr_results['lag_correlations']:
                                        st.subheader("Temporal Cross-Correlation")
                                        
                                        import plotly.graph_objects as go
                                        fig = go.Figure()
                                        
                                        for channel, lag_data in corr_results['lag_correlations'].items():
                                            if lag_data:
                                                display_name = st.session_state.get('channel_labels', {}).get(channel, channel)
                                                fig.add_trace(go.Scatter(
                                                    x=list(lag_data.keys()),
                                                    y=list(lag_data.values()),
                                                    mode='lines+markers',
                                                    name=display_name
                                                ))
                                        
                                        fig.update_layout(
                                            title="Intensity-Motion Cross-Correlation vs Lag",
                                            xaxis_title="Lag (frames)",
                                            yaxis_title="Correlation Coefficient",
                                            hovermode='x unified'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Store results
                                    if 'analysis_results' not in st.session_state:
                                        st.session_state.analysis_results = {}
                                    st.session_state.analysis_results['correlative'] = corr_results
                        
                        except ImportError:
                            st.error("Correlative analysis module not available.")
                        except Exception as e:
                            st.error(f"Error in correlative analysis: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
            else:
                st.warning("Correlative analysis module is not available. Make sure the appropriate files are in the correct location.")
        
        # Advanced Tracking tab
        with adv_tabs[4]:
            st.header("Advanced Tracking")
            st.write("Advanced particle tracking using Bayesian inference and particle filtering.")
            
            if ADVANCED_TRACKING_AVAILABLE:
                # Tracking method selection
                st.subheader("Tracking Method")
                tracking_method = st.selectbox(
                    "Select Method",
                    ["Particle Filter", "Bayesian Detection Refinement"],
                    help="Choose advanced tracking algorithm"
                )
                
                if tracking_method == "Particle Filter":
                    st.subheader("Particle Filter Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        n_particles = st.slider(
                            "Number of Particles",
                            min_value=50,
                            max_value=1000,
                            value=200,
                            step=50,
                            help="Number of particles in the filter"
                        )
                        
                        motion_std = st.number_input(
                            "Motion Noise (σ)",
                            min_value=0.1,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            help="Standard deviation of motion model noise"
                        )
                    
                    with col2:
                        measurement_std = st.number_input(
                            "Measurement Noise (σ)",
                            min_value=0.1,
                            max_value=10.0,
                            value=0.5,
                            step=0.1,
                            help="Standard deviation of measurement noise"
                        )
                        
                        max_distance = st.number_input(
                            "Max Linking Distance (pixels)",
                            min_value=1.0,
                            max_value=50.0,
                            value=10.0,
                            step=1.0,
                            help="Maximum distance for particle linking"
                        )
                    
                    if st.button("Run Particle Filter Tracking"):
                        with st.spinner("Running particle filter tracking..."):
                            try:
                                from advanced_tracking import ParticleFilter, AdvancedTracking
                                
                                # Get track data
                                tracks_df, has_data = get_track_data()
                                if not has_data:
                                    st.error("No track data available")
                                else:
                                    # Initialize tracking
                                    tracker = AdvancedTracking()
                                    
                                    # Create detections from track data
                                    detections = tracks_df[['frame', 'x', 'y']].copy()
                                    
                                    # Run particle filter
                                    pf = ParticleFilter(
                                        n_particles=n_particles,
                                        motion_noise=motion_std,
                                        measurement_noise=measurement_std
                                    )
                                    
                                    # Track particles
                                    results = tracker.track_with_particle_filter(
                                        detections=detections,
                                        n_particles=n_particles,
                                        motion_std=motion_std,
                                        measurement_std=measurement_std,
                                        max_distance=max_distance
                                    )
                                    
                                    if results.get('success', False):
                                        st.success("✓ Particle filter tracking completed")
                                        
                                        # Display statistics
                                        if 'statistics' in results:
                                            st.subheader("Tracking Statistics")
                                            stats = results['statistics']
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Total Tracks", stats.get('n_tracks', 0))
                                            with col2:
                                                st.metric("Total Detections", stats.get('n_detections', 0))
                                            with col3:
                                                st.metric("Mean Track Length", f"{stats.get('mean_track_length', 0):.1f}")
                                        
                                        # Store results
                                        if 'analysis_results' not in st.session_state:
                                            st.session_state.analysis_results = {}
                                        st.session_state.analysis_results['particle_filter'] = results
                                        
                                        # Display tracks if available
                                        if 'tracks' in results:
                                            st.subheader("Refined Tracks")
                                            st.dataframe(results['tracks'].head(20))
                                    else:
                                        st.error(f"Tracking failed: {results.get('error', 'Unknown error')}")
                            
                            except ImportError:
                                st.error("Advanced tracking module not available.")
                            except Exception as e:
                                st.error(f"Error in particle filter tracking: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                
                else:  # Bayesian Detection Refinement
                    st.subheader("Bayesian Refinement Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        prior_weight = st.slider(
                            "Prior Weight",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                            step=0.05,
                            help="Weight of prior information in Bayesian update"
                        )
                        
                        localization_precision = st.number_input(
                            "Localization Precision (nm)",
                            min_value=1.0,
                            max_value=100.0,
                            value=20.0,
                            step=1.0,
                            help="Expected localization precision in nanometers"
                        )
                    
                    with col2:
                        refinement_iterations = st.slider(
                            "Refinement Iterations",
                            min_value=1,
                            max_value=10,
                            value=3,
                            help="Number of Bayesian refinement iterations"
                        )
                    
                    if st.button("Run Bayesian Refinement"):
                        with st.spinner("Running Bayesian detection refinement..."):
                            try:
                                from advanced_tracking import bayesian_detection_refinement
                                
                                # Get track data
                                tracks_df, has_data = get_track_data()
                                if not has_data:
                                    st.error("No track data available")
                                else:
                                    # Create detections dataframe
                                    detections = tracks_df[['frame', 'x', 'y']].copy()
                                    
                                    # Run refinement
                                    refined_detections = bayesian_detection_refinement(
                                        detections=detections,
                                        prior_weight=prior_weight,
                                        localization_precision=localization_precision,
                                        n_iterations=refinement_iterations
                                    )
                                    
                                    st.success("✓ Bayesian refinement completed")
                                    
                                    # Display refined detections
                                    st.subheader("Refined Detections")
                                    st.dataframe(refined_detections.head(20))
                                    
                                    # Calculate improvement statistics
                                    if 'x_refined' in refined_detections.columns:
                                        dx = refined_detections['x'] - refined_detections['x_refined']
                                        dy = refined_detections['y'] - refined_detections['y_refined']
                                        displacement = np.sqrt(dx**2 + dy**2)
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Mean Adjustment", f"{displacement.mean():.3f} px")
                                        with col2:
                                            st.metric("Max Adjustment", f"{displacement.max():.3f} px")
                                        with col3:
                                            st.metric("Detections Refined", len(refined_detections))
                                    
                                    # Store results
                                    if 'analysis_results' not in st.session_state:
                                        st.session_state.analysis_results = {}
                                    st.session_state.analysis_results['bayesian_refinement'] = {
                                        'refined_detections': refined_detections,
                                        'parameters': {
                                            'prior_weight': prior_weight,
                                            'localization_precision': localization_precision,
                                            'refinement_iterations': refinement_iterations
                                        }
                                    }
                            
                            except ImportError:
                                st.error("Advanced tracking module not available.")
                            except Exception as e:
                                st.error(f"Error in Bayesian refinement: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
            else:
                st.warning("Advanced tracking module is not available.")
        
        # Intensity Analysis tab
        with adv_tabs[5]:
            st.header("Intensity Analysis")
            st.write("Analyze intensity-motion correlations and intensity-based behavior classification.")
            
            if INTENSITY_ANALYSIS_AVAILABLE:
                # Get track data
                tracks_df, has_data = get_track_data()
                
                if not has_data:
                    st.warning("No track data available. Please load data first.")
                else:
                    # Detect intensity channels
                    intensity_columns = [col for col in tracks_df.columns 
                                       if 'intensity' in col.lower() or 'ch' in col.lower()]
                    
                    if not intensity_columns:
                        st.warning("No intensity channels detected. Load data with intensity information.")
                    else:
                        st.success(f"Found {len(intensity_columns)} intensity channel(s)")
                        
                        # Analysis selection
                        st.subheader("Analysis Type")
                        analysis_type = st.selectbox(
                            "Select Analysis",
                            ["Intensity-Movement Correlation", "Intensity Profiles", "Intensity Behavior Classification"],
                            help="Choose intensity analysis method"
                        )
                        
                        if analysis_type == "Intensity-Movement Correlation":
                            st.subheader("Correlation Parameters")
                            
                            selected_channels = st.multiselect(
                                "Select Intensity Channels",
                                intensity_columns,
                                default=intensity_columns[:2] if len(intensity_columns) >= 2 else intensity_columns
                            )
                            
                            if st.button("Run Intensity-Movement Correlation"):
                                with st.spinner("Analyzing intensity-movement correlation..."):
                                    try:
                                        from intensity_analysis import correlate_intensity_movement, create_intensity_movement_plots
                                        
                                        units = get_units()
                                        
                                        # Run correlation analysis
                                        results = correlate_intensity_movement(
                                            tracks_df=tracks_df,
                                            intensity_channels=selected_channels,
                                            pixel_size=units.get('pixel_size', 0.1),
                                            frame_interval=units.get('frame_interval', 0.1)
                                        )
                                        
                                        if results.get('success', True):
                                            st.success("✓ Correlation analysis completed")
                                            
                                            # Display correlation statistics
                                            if 'correlation_coefficients' in results:
                                                st.subheader("Correlation Coefficients")
                                                corr_df = pd.DataFrame([results['correlation_coefficients']])
                                                st.dataframe(corr_df)
                                            
                                            # Create and display plots
                                            plots = create_intensity_movement_plots(results)
                                            
                                            if 'correlation_heatmap' in plots:
                                                st.subheader("Correlation Heatmap")
                                                st.plotly_chart(plots['correlation_heatmap'], use_container_width=True)
                                            
                                            if 'time_series' in plots:
                                                st.subheader("Time Series")
                                                st.plotly_chart(plots['time_series'], use_container_width=True)
                                            
                                            # Store results
                                            if 'analysis_results' not in st.session_state:
                                                st.session_state.analysis_results = {}
                                            st.session_state.analysis_results['intensity_movement'] = results
                                        else:
                                            st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
                                    
                                    except Exception as e:
                                        st.error(f"Error in intensity-movement correlation: {str(e)}")
                                        import traceback
                                        st.text(traceback.format_exc())
                        
                        elif analysis_type == "Intensity Profiles":
                            st.subheader("Profile Parameters")
                            
                            normalize = st.checkbox(
                                "Normalize Intensities",
                                value=True,
                                help="Normalize intensities to [0, 1] range"
                            )
                            
                            if st.button("Analyze Intensity Profiles"):
                                with st.spinner("Analyzing intensity profiles..."):
                                    try:
                                        from intensity_analysis import analyze_intensity_profiles
                                        
                                        units = get_units()
                                        
                                        results = analyze_intensity_profiles(
                                            tracks_df=tracks_df,
                                            intensity_channels=intensity_columns,
                                            normalize=normalize
                                        )
                                        
                                        if results.get('success', True):
                                            st.success("✓ Profile analysis completed")
                                            
                                            # Display profile statistics
                                            if 'profile_stats' in results:
                                                st.subheader("Profile Statistics")
                                                st.dataframe(pd.DataFrame(results['profile_stats']))
                                            
                                            # Display profiles
                                            if 'profiles' in results:
                                                st.subheader("Intensity Profiles")
                                                st.line_chart(results['profiles'])
                                            
                                            # Store results
                                            if 'analysis_results' not in st.session_state:
                                                st.session_state.analysis_results = {}
                                            st.session_state.analysis_results['intensity_profiles'] = results
                                        else:
                                            st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
                                    
                                    except Exception as e:
                                        st.error(f"Error in intensity profile analysis: {str(e)}")
                                        import traceback
                                        st.text(traceback.format_exc())
                        
                        else:  # Intensity Behavior Classification
                            st.subheader("Classification Parameters")
                            
                            n_clusters = st.slider(
                                "Number of Behavior Classes",
                                min_value=2,
                                max_value=5,
                                value=3,
                                help="Number of distinct intensity behaviors to identify"
                            )
                            
                            if st.button("Classify Intensity Behavior"):
                                with st.spinner("Classifying intensity behavior..."):
                                    try:
                                        from intensity_analysis import classify_intensity_behavior
                                        
                                        results = classify_intensity_behavior(
                                            tracks_df=tracks_df,
                                            intensity_channels=intensity_columns,
                                            n_clusters=n_clusters
                                        )
                                        
                                        if results.get('success', True):
                                            st.success("✓ Classification completed")
                                            
                                            # Display classification results
                                            if 'classified_tracks' in results:
                                                st.subheader("Classified Tracks")
                                                classified_df = results['classified_tracks']
                                                st.dataframe(classified_df.head(20))
                                                
                                                # Show distribution
                                                if 'cluster' in classified_df.columns:
                                                    st.subheader("Behavior Distribution")
                                                    cluster_counts = classified_df['cluster'].value_counts()
                                                    st.bar_chart(cluster_counts)
                                            
                                            # Display cluster centers
                                            if 'cluster_centers' in results:
                                                st.subheader("Cluster Centers")
                                                st.dataframe(pd.DataFrame(results['cluster_centers']))
                                            
                                            # Store results
                                            if 'analysis_results' not in st.session_state:
                                                st.session_state.analysis_results = {}
                                            st.session_state.analysis_results['intensity_classification'] = results
                                        else:
                                            st.error(f"Classification failed: {results.get('error', 'Unknown error')}")
                                    
                                    except Exception as e:
                                        st.error(f"Error in intensity behavior classification: {str(e)}")
                                        import traceback
                                        st.text(traceback.format_exc())
            else:
                st.warning("Intensity analysis module is not available.")
        
        # Microrheology Analysis tab
        with adv_tabs[6]:
            st.header("Microrheology Analysis")
            
            # Parameters
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                particle_radius_nm = st.number_input(
                    "Particle Radius (nm)",
                    min_value=10.0,
                    max_value=1000.0,
                    value=100.0,
                    step=10.0,
                    help="Radius of tracked particles in nanometers"
                )
                
                temperature_K = st.number_input(
                    "Temperature (K)",
                    min_value=200.0,
                    max_value=400.0,
                    value=298.15,
                    step=1.0,
                    help="Temperature in Kelvin (default: 298.15K = 25°C)"
                )
            
            with col2:
                pixel_size_um = st.number_input(
                    "Pixel Size (µm)",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.1,
                    step=0.001,
                    format="%.3f",
                    help="Pixel size in micrometers"
                )
                
                frame_interval_s = st.number_input(
                    "Frame Interval (s)",
                    min_value=0.001,
                    max_value=10.0,
                    value=0.1,
                    step=0.001,
                    format="%.3f",
                    help="Time between frames in seconds"
                )
            
            # Multi-dataset analysis option
            st.subheader("Multi-Dataset Analysis")
            use_multi_dataset = st.checkbox(
                "Enable Multi-Dataset Analysis",
                value=False,
                help="Combine multiple files with different sampling rates for comprehensive frequency analysis"
            )
            
            if use_multi_dataset:
                st.info("Upload additional track files to combine different sampling rates for broader frequency range analysis.")
                multi_files = st.file_uploader(
                    "Additional Track Files",
                    type=['csv', 'xlsx', 'xml'],
                    accept_multiple_files=True,
                    help="Upload additional tracking files from the same sample with different sampling rates"
                )
                
                if multi_files:
                    st.write(f"Selected {len(multi_files)} additional files:")
                    frame_intervals = [frame_interval_s]  # Include current dataset
                    
                    for i, uploaded_file in enumerate(multi_files):
                        dataset_name = uploaded_file.name
                        st.write(f"• {dataset_name}")
                        
                        # Get frame interval for each dataset
                        interval = st.number_input(
                            f"Frame Interval for {dataset_name} (s)",
                            min_value=0.001,
                            max_value=10.0,
                            value=0.1,
                            step=0.001,
                            format="%.3f",
                            key=f"interval_{i}",
                            help=f"Frame interval for {dataset_name}"
                        )
                        frame_intervals.append(interval)
                else:
                    st.warning("Please upload additional track files to enable multi-dataset analysis.")
            
            # Analysis button
            if st.button("Run Microrheology Analysis", key="rheology_analysis_btn_2"):
                with st.spinner("Performing microrheology analysis..."):
                    try:
                        # Initialize analyzer
                        particle_radius_m = particle_radius_nm * 1e-9  # Convert to meters
                        analyzer = MicrorheologyAnalyzer(
                            particle_radius_m=particle_radius_m,
                            temperature_K=temperature_K
                        )
                        
                        if use_multi_dataset and 'multi_files' in locals() and multi_files:
                            # Multi-dataset analysis with different sampling rates
                            try:
                                # Load additional datasets
                                track_datasets = [st.session_state.tracks_data]  # Include current dataset
                                
                                for uploaded_file in multi_files:
                                    # Load each additional file
                                    from data_loader import load_tracks_file
                                    additional_tracks = load_tracks_file(uploaded_file)
                                    if additional_tracks is not None and not additional_tracks.empty:
                                        track_datasets.append(additional_tracks)
                                
                                # Perform multi-dataset analysis
                                analysis_results = analyzer.analyze_multi_dataset_rheology(
                                    track_datasets=track_datasets,
                                    pixel_sizes=[pixel_size_um] * len(track_datasets),
                                    frame_intervals=frame_intervals,
                                    max_lag=20
                                )
                                
                            except Exception as e:
                                st.error(f"Error in multi-dataset analysis: {str(e)}")
                                # Fallback to single dataset
                                analysis_results = analyzer.analyze_microrheology(
                                    st.session_state.tracks_data,
                                    pixel_size_um=pixel_size_um,
                                    frame_interval_s=frame_interval_s,
                                    max_lag=20
                                )
                        else:
                            # Single dataset analysis
                            analysis_results = analyzer.analyze_microrheology(
                                st.session_state.tracks_data,
                                pixel_size_um=pixel_size_um,
                                frame_interval_s=frame_interval_s,
                                max_lag=20
                            )
                        
                        # Store results
                        st.session_state.analysis_results["microrheology"] = analysis_results
                        
                        # Create analysis record
                        analysis_record = create_analysis_record(
                            name="Microrheology Analysis",
                            analysis_type="microrheology",
                            parameters={
                                "particle_radius_nm": particle_radius_nm,
                                "temperature_K": temperature_K,
                                "pixel_size_um": pixel_size_um,
                                "frame_interval_s": frame_interval_s,
                                "multi_dataset": use_multi_dataset,
                                "num_datasets": len(track_datasets) if use_multi_dataset and 'track_datasets' in locals() else 1
                            }
                        )
                        
                        # Add to recent analyses
                        st.session_state.recent_analyses.append(analysis_record)
                        
                        if analysis_results.get('success', False):
                            st.success("Microrheology analysis completed successfully!")
                        else:
                            st.error(f"Analysis failed: {analysis_results.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"Error in microrheology analysis: {str(e)}")
            
            # Display results
            if "microrheology" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["microrheology"]
                
                if results.get('success', False):
                    # Display summary
                    display_rheology_summary(results)
                    
                    # Create and display plots
                    try:
                        figures = create_rheology_plots(results)
                        
                        if 'msd_comparison' in figures:
                            st.subheader("Mean Squared Displacement")
                            st.plotly_chart(figures['msd_comparison'], use_container_width=True)
                        
                        if 'frequency_response' in figures:
                            st.subheader("Combined Frequency Response")
                            st.plotly_chart(figures['frequency_response'], use_container_width=True)
                        
                        if 'individual_frequency_response' in figures:
                            st.subheader("Individual Dataset Frequency Responses")
                            st.plotly_chart(figures['individual_frequency_response'], use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Could not generate plots: {str(e)}")
                    
                    # Show detailed results
                    with st.expander("Detailed Results"):
                        if 'msd_data' in results:
                            st.subheader("MSD Data")
                            st.dataframe(results['msd_data'])
                        
                        if 'rheology_data' in results:
                            st.subheader("Rheological Properties")
                            st.dataframe(results['rheology_data'])
                        
                        if 'frequency_sweep' in results:
                            st.subheader("Frequency Sweep Data")
                            st.dataframe(results['frequency_sweep'])

            # CTRW Analysis tab
            with adv_tabs[7]:
                st.header("Continuous Time Random Walk (CTRW) Analysis")
                
                st.markdown("""
                **Continuous Time Random Walk (CTRW)** models characterize anomalous diffusion through:
                - **Waiting Time Distribution** ψ(t): Time between movement events
                - **Jump Length Distribution** λ(r): Size of spatial jumps
                
                Heavy-tailed distributions indicate CTRW behavior often seen in biological systems.
                """)
                
                # Parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    min_pause_threshold = st.number_input(
                        "Pause Threshold (μm)",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.01,
                        step=0.001,
                        format="%.3f",
                        help="Minimum displacement considered as movement"
                    )
                    
                with col2:
                    fit_distribution = st.selectbox(
                        "Distribution Model",
                        ["auto", "exponential", "powerlaw"],
                        help="Distribution to fit to waiting times"
                    )
                
                if st.button("Run CTRW Analysis", key="ctrw_btn"):
                    with st.spinner("Analyzing CTRW properties..."):
                        try:
                            from advanced_diffusion_models import CTRWAnalyzer
                            
                            units = get_current_units()
                            
                            analyzer = CTRWAnalyzer(
                                st.session_state.tracks_data,
                                pixel_size=units['pixel_size'],
                                frame_interval=units['frame_interval']
                            )
                            
                            # Waiting time analysis
                            waiting_time_results = analyzer.analyze_waiting_time_distribution(
                                min_pause_threshold=min_pause_threshold,
                                fit_distribution=fit_distribution
                            )
                            
                            # Jump length analysis
                            jump_length_results = analyzer.analyze_jump_length_distribution()
                            
                            # Coupling analysis
                            coupling_results = analyzer.analyze_coupling()
                            
                            ctrw_results = {
                                'success': True,
                                'waiting_times': waiting_time_results,
                                'jump_lengths': jump_length_results,
                                'coupling': coupling_results
                            }
                            
                            st.session_state.analysis_results['ctrw'] = ctrw_results
                            
                            # Display results
                            st.success("✓ CTRW analysis completed!")
                            
                            # Waiting time results
                            st.subheader("Waiting Time Distribution")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Distribution Type", waiting_time_results['distribution_type'])
                            with col2:
                                st.metric("Mean Waiting Time", f"{waiting_time_results['mean_waiting_time']:.3f} s")
                            with col3:
                                if waiting_time_results['is_heavy_tailed']:
                                    st.metric("Heavy-Tailed", "✓ Yes")
                                else:
                                    st.metric("Heavy-Tailed", "✗ No")
                            
                            # Plot waiting times
                            if len(waiting_time_results['waiting_times']) > 0:
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=waiting_time_results['waiting_times'],
                                    name='Waiting Times',
                                    nbinsx=50
                                ))
                                fig.update_layout(
                                    title="Waiting Time Distribution",
                                    xaxis_title="Time (s)",
                                    yaxis_title="Count",
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Jump length results
                            st.subheader("Jump Length Distribution")
                            
                            if 'mean_jump_length' in jump_length_results:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Mean Jump Length", f"{jump_length_results['mean_jump_length']:.3f} μm")
                                with col2:
                                    st.metric("Std Jump Length", f"{jump_length_results.get('std_jump_length', 0):.3f} μm")
                            
                            # Coupling analysis
                            if coupling_results:
                                st.subheader("Waiting Time-Jump Length Coupling")
                                st.info(f"Correlation: {coupling_results.get('correlation', 0):.3f}")
                        
                        except Exception as e:
                            st.error(f"Error in CTRW analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # FBM Analysis tab
            with adv_tabs[8]:
                st.header("Fractional Brownian Motion (FBM) Analysis")
                
                st.markdown("""
                **Fractional Brownian Motion** characterizes anomalous diffusion through the **Hurst exponent** H:
                - H = 0.5: Normal Brownian motion
                - H < 0.5: Subdiffusive (antipersistent)
                - H > 0.5: Superdiffusive (persistent)
                """)
                
                # Parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    min_track_length_fbm = st.slider(
                        "Minimum Track Length",
                        min_value=10,
                        max_value=100,
                        value=20,
                        help="Minimum number of points for FBM fitting"
                    )
                
                with col2:
                    confidence_level = st.slider(
                        "Confidence Level",
                        min_value=0.90,
                        max_value=0.99,
                        value=0.95,
                        step=0.01,
                        help="Confidence level for parameter estimation"
                    )
                
                if st.button("Run FBM Analysis", key="fbm_btn"):
                    with st.spinner("Fitting FBM models..."):
                        try:
                            from advanced_diffusion_models import fit_fbm_model
                            
                            units = get_current_units()
                            
                            # Filter tracks by length
                            track_groups = st.session_state.tracks_data.groupby('track_id')
                            long_tracks = [g for _, g in track_groups if len(g) >= min_track_length_fbm]
                            
                            if len(long_tracks) == 0:
                                st.error(f"No tracks longer than {min_track_length_fbm} frames found.")
                            else:
                                # Fit FBM to each track
                                fbm_results_list = []
                                
                                progress_bar = st.progress(0)
                                for idx, track in enumerate(long_tracks):
                                    track_result = fit_fbm_model(
                                        track,
                                        pixel_size=units['pixel_size'],
                                        frame_interval=units['frame_interval']
                                    )
                                    if track_result['success']:
                                        fbm_results_list.append(track_result)
                                    progress_bar.progress((idx + 1) / len(long_tracks))
                                
                                progress_bar.empty()
                                
                                if len(fbm_results_list) > 0:
                                    st.success(f"✓ FBM analysis completed for {len(fbm_results_list)} tracks!")
                                    
                                    # Aggregate results
                                    hurst_exponents = [r['hurst_exponent'] for r in fbm_results_list]
                                    diffusion_coeffs = [r['diffusion_coefficient'] for r in fbm_results_list]
                                    
                                    fbm_results = {
                                        'success': True,
                                        'track_results': fbm_results_list,
                                        'mean_hurst': np.mean(hurst_exponents),
                                        'std_hurst': np.std(hurst_exponents),
                                        'mean_D': np.mean(diffusion_coeffs),
                                        'hurst_distribution': hurst_exponents,
                                        'D_distribution': diffusion_coeffs
                                    }
                                    
                                    st.session_state.analysis_results['fbm'] = fbm_results
                                    
                                    # Display ensemble results
                                    st.subheader("Ensemble Results")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Mean Hurst Exponent", f"{fbm_results['mean_hurst']:.3f} ± {fbm_results['std_hurst']:.3f}")
                                    with col2:
                                        st.metric("Mean D", f"{fbm_results['mean_D']:.3e} μm²/s")
                                    with col3:
                                        st.metric("N Tracks Analyzed", len(fbm_results_list))
                                    
                                    # Interpret mean Hurst
                                    H_mean = fbm_results['mean_hurst']
                                    if H_mean < 0.45:
                                        regime = "Subdiffusive (antipersistent)"
                                    elif H_mean < 0.55:
                                        regime = "Normal Brownian motion"
                                    else:
                                        regime = "Superdiffusive (persistent)"
                                    
                                    st.info(f"**Diffusion Regime:** {regime}")
                                    
                                    # Plot Hurst distribution
                                    import plotly.graph_objects as go
                                    fig = go.Figure()
                                    fig.add_trace(go.Histogram(
                                        x=hurst_exponents,
                                        name='Hurst Exponent',
                                        nbinsx=30
                                    ))
                                    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                                                 annotation_text="H=0.5 (Brownian)")
                                    fig.update_layout(
                                        title="Hurst Exponent Distribution",
                                        xaxis_title="Hurst Exponent (H)",
                                        yaxis_title="Count"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("FBM fitting failed for all tracks.")
                        
                        except Exception as e:
                            st.error(f"Error in FBM analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Advanced Metrics tab
            with adv_tabs[9]:
                st.header("Advanced Biophysical Metrics")
                
                st.markdown("""
                Comprehensive set of advanced metrics:
                - **TAMSD/EAMSD**: Time-averaged vs Ensemble-averaged MSD
                - **Ergodicity Breaking (EB)**: Measure of non-ergodicity
                - **NGP**: Non-Gaussian Parameter (deviation from Brownian)
                - **VACF**: Velocity Autocorrelation Function
                - **van Hove Distribution**: Displacement probability distribution
                """)
                
                # Parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    max_lag_metrics = st.slider(
                        "Maximum Lag Time",
                        min_value=5,
                        max_value=50,
                        value=20,
                        help="Maximum lag for MSD and correlation calculations"
                    )
                    
                with col2:
                    min_track_length_metrics = st.slider(
                        "Minimum Track Length",
                        min_value=5,
                        max_value=50,
                        value=10,
                        help="Minimum points per track"
                    )
                
                if st.button("Calculate Advanced Metrics", key="advanced_metrics_btn"):
                    with st.spinner("Computing advanced metrics..."):
                        try:
                            from advanced_biophysical_metrics import AdvancedMetricsAnalyzer, MetricConfig
                            
                            units = get_current_units()
                            
                            config = MetricConfig(
                                pixel_size=units['pixel_size'],
                                frame_interval=units['frame_interval'],
                                min_track_length=min_track_length_metrics,
                                max_lag=max_lag_metrics,
                                log_lag=True,
                                n_bootstrap=500
                            )
                            
                            analyzer = AdvancedMetricsAnalyzer(st.session_state.tracks_data, config)
                            
                            if analyzer.df.empty:
                                st.error("No tracks meet the minimum length requirement.")
                            else:
                                # Calculate TAMSD/EAMSD
                                tamsd_df, eamsd_df = analyzer.tamsd_eamsd()
                                
                                # Calculate ergodicity
                                ergodicity_df = analyzer.ergodicity_measures(tamsd_df, eamsd_df)
                                
                                # Calculate NGP
                                ngp_results = []
                                for lag in analyzer.lags:
                                    dx, dy, dr, n_pairs = analyzer._collect_displacements(lag)
                                    if n_pairs > 0:
                                        from advanced_metrics import non_gaussian_parameter_2d
                                        ngp = non_gaussian_parameter_2d(dr)
                                        ngp_results.append({
                                            'lag': lag,
                                            'tau_s': lag * units['frame_interval'],
                                            'ngp': ngp,
                                            'n_pairs': n_pairs
                                        })
                                
                                ngp_df = pd.DataFrame(ngp_results)
                                
                                # Calculate VACF
                                vacf_results = analyzer.vacf()
                                
                                metrics_results = {
                                    'success': True,
                                    'tamsd': tamsd_df,
                                    'eamsd': eamsd_df,
                                    'ergodicity': ergodicity_df,
                                    'ngp': ngp_df,
                                    'vacf': vacf_results
                                }
                                
                                st.session_state.analysis_results['advanced_metrics'] = metrics_results
                                
                                st.success("✓ Advanced metrics calculated!")
                                
                                # Display results
                                st.subheader("Ergodicity Breaking (EB) Ratio")
                                
                                if not ergodicity_df.empty:
                                    mean_eb = ergodicity_df['EB_ratio'].mean()
                                    st.metric("Mean EB Ratio", f"{mean_eb:.3f}")
                                    
                                    if mean_eb < 0.8:
                                        st.warning("⚠️ Strong ergodicity breaking detected (EB < 0.8)")
                                    elif mean_eb < 1.2:
                                        st.info("ℹ️ Weakly ergodic system (0.8 < EB < 1.2)")
                                    else:
                                        st.success("✓ Ergodic system (EB ≈ 1)")
                                    
                                    # Plot EB ratio vs lag
                                    import plotly.graph_objects as go
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=ergodicity_df['tau_s'],
                                        y=ergodicity_df['EB_ratio'],
                                        mode='lines+markers',
                                        name='EB Ratio'
                                    ))
                                    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                                                 annotation_text="Ergodic (EB=1)")
                                    fig.update_layout(
                                        title="Ergodicity Breaking Ratio vs Time",
                                        xaxis_title="Lag Time (s)",
                                        yaxis_title="EB Ratio",
                                        xaxis_type="log",
                                        yaxis_type="log"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # NGP Plot
                                st.subheader("Non-Gaussian Parameter (NGP)")
                                
                                if not ngp_df.empty:
                                    fig_ngp = go.Figure()
                                    fig_ngp.add_trace(go.Scatter(
                                        x=ngp_df['tau_s'],
                                        y=ngp_df['ngp'],
                                        mode='lines+markers',
                                        name='NGP'
                                    ))
                                    fig_ngp.add_hline(y=0.0, line_dash="dash", line_color="red",
                                                     annotation_text="Gaussian (NGP=0)")
                                    fig_ngp.update_layout(
                                        title="Non-Gaussian Parameter vs Time",
                                        xaxis_title="Lag Time (s)",
                                        yaxis_title="NGP",
                                        xaxis_type="log"
                                    )
                                    st.plotly_chart(fig_ngp, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error calculating advanced metrics: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Statistical Tests tab
            with adv_tabs[10]:
                st.header("Statistical Tests & Model Validation")
                
                st.markdown("""
                Rigorous statistical validation tools:
                - **Goodness-of-Fit Tests**: Chi-squared, Kolmogorov-Smirnov, Anderson-Darling
                - **Model Selection**: AIC/BIC criteria
                - **Bootstrap Confidence Intervals**
                - **Non-parametric Tests**: Mann-Whitney U, Kruskal-Wallis, Wilcoxon
                """)
                
                # Test selection
                test_type = st.selectbox(
                    "Select Test Type",
                    [
                        "Goodness-of-Fit (Chi-squared)",
                        "Kolmogorov-Smirnov Test",
                        "Bootstrap CI",
                        "Mann-Whitney U Test",
                        "Model Comparison (AIC/BIC)"
                    ]
                )
                
                if "Goodness-of-Fit" in test_type:
                    st.subheader("Chi-Squared Goodness-of-Fit Test")
                    
                    st.info("This test will be applied to diffusion analysis results if available.")
                    
                    if 'diffusion' in st.session_state.get('analysis_results', {}):
                        if st.button("Run Chi-Squared Test"):
                            try:
                                from advanced_statistical_tests import chi_squared_goodness_of_fit, validate_model_fit
                                
                                diff_results = st.session_state.analysis_results['diffusion']
                                
                                # Get MSD data
                                if 'msd_data' in diff_results:
                                    observed_msd = np.array(diff_results['msd_data'])
                                    lag_times = np.array(diff_results['lag_times'])
                                    
                                    # Fit linear model
                                    from scipy.stats import linregress
                                    slope, intercept, r_value, p_value, std_err = linregress(lag_times, observed_msd)
                                    expected_msd = slope * lag_times + intercept
                                    
                                    # Chi-squared test
                                    chi2_results = chi_squared_goodness_of_fit(observed_msd, expected_msd, n_params=2)
                                    
                                    if chi2_results['success']:
                                        st.success("✓ Chi-squared test completed")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("χ² Statistic", f"{chi2_results['statistic']:.3f}")
                                        with col2:
                                            st.metric("p-value", f"{chi2_results['p_value']:.4f}")
                                        with col3:
                                            st.metric("DOF", chi2_results['dof'])
                                        
                                        st.info(f"**Conclusion:** {chi2_results['conclusion']}")
                                    else:
                                        st.error(chi2_results['error'])
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Run Diffusion Analysis first to use this test.")
                
                elif "Bootstrap" in test_type:
                    st.subheader("Bootstrap Confidence Intervals")
                    
                    n_bootstrap = st.slider("Number of Bootstrap Samples", 100, 10000, 1000, 100)
                    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
                    
                    if st.button("Calculate Bootstrap CI"):
                        try:
                            from advanced_statistical_tests import bootstrap_confidence_interval
                            
                            # Use track lengths as example
                            track_lengths = st.session_state.tracks_data.groupby('track_id').size().values
                            
                            ci_result = bootstrap_confidence_interval(
                                track_lengths,
                                statistic_func=np.mean,
                                n_bootstrap=n_bootstrap,
                                confidence_level=confidence_level
                            )
                            
                            st.success("✓ Bootstrap CI calculated")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Observed Mean", f"{ci_result['point_estimate']:.2f}")
                            with col2:
                                st.metric("Lower CI", f"{ci_result['ci_lower']:.2f}")
                            with col3:
                                st.metric("Upper CI", f"{ci_result['ci_upper']:.2f}")
                            
                            st.info(f"{int(confidence_level*100)}% CI: [{ci_result['ci_lower']:.2f}, {ci_result['ci_upper']:.2f}]")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Ornstein-Uhlenbeck Analysis tab
            with adv_tabs[11]:
                st.header("Ornstein-Uhlenbeck Analysis")
                st.write("Analyze tracks assuming an Ornstein-Uhlenbeck process, which models a particle in a harmonic potential.")

                if st.button("Run Ornstein-Uhlenbeck Analysis"):
                    with st.spinner("Running Ornstein-Uhlenbeck analysis..."):
                        units = get_current_units()
                        ou_results = analyze_ornstein_uhlenbeck(
                            st.session_state.tracks_data,
                            pixel_size=units['pixel_size'],
                            frame_interval=units['frame_interval']
                        )
                        st.session_state.analysis_results["ornstein_uhlenbeck"] = ou_results
                        st.success("Ornstein-Uhlenbeck analysis completed!")

                if "ornstein_uhlenbeck" in st.session_state.analysis_results:
                    results = st.session_state.analysis_results["ornstein_uhlenbeck"]
                    if results['success']:
                        st.subheader("OU Parameters")
                        st.dataframe(results['ou_parameters'])

                        st.subheader("Velocity Autocorrelation Function (VACF)")
                        fig = px.line(results['vacf_data'], x='lag', y='vacf', color='track_id', title="VACF per Track")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")

# AI Anomaly Detection page
elif st.session_state.active_page == "AI Anomaly Detection":
    st.title("AI-Powered Anomaly Detection")
    
    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
    else:
        st.write("Detect unusual particle behaviors using advanced machine learning algorithms.")
        
        # Parameters for anomaly detection
        st.subheader("Detection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            contamination = st.slider(
                "Expected Anomaly Rate (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Expected percentage of anomalous tracks in your data"
            ) / 100.0
            
            z_threshold = st.number_input(
                "Velocity Change Threshold",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Z-score threshold for detecting sudden velocity changes"
            )
            
            expansion_threshold = st.number_input(
                "Confinement Violation Threshold",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Threshold for detecting sudden expansion beyond normal confinement"
            )
        
        with col2:
            reversal_threshold = st.number_input(
                "Directional Change Threshold (radians)",
                min_value=1.0,
                max_value=3.14,
                value=2.5,
                step=0.1,
                help="Threshold for detecting significant directional changes"
            )
            
            clustering_eps = st.number_input(
                "Spatial Clustering Parameter",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                help="Distance parameter for spatial clustering analysis"
            )
            
            min_track_length = st.slider(
                "Minimum Track Length",
                min_value=5,
                max_value=50,
                value=10,
                help="Minimum track length for anomaly analysis"
            )
        
        # Run anomaly detection
        if st.button("Run Anomaly Detection"):
            with st.spinner("Analyzing particle behavior patterns..."):
                try:
                    # Initialize anomaly detector
                    detector = AnomalyDetector(contamination=contamination)
                    
                    # Filter tracks by minimum length
                    filtered_tracks = st.session_state.tracks_data.groupby('track_id').filter(
                        lambda x: len(x) >= min_track_length
                    )
                    
                    if len(filtered_tracks) > 0:
                        # Run comprehensive anomaly detection
                        anomaly_results = detector.comprehensive_anomaly_detection(
                            filtered_tracks
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results["anomaly_detection"] = {
                            'results': anomaly_results,
                            'detector': detector,
                            'parameters': {
                                'contamination': contamination,
                                'z_threshold': z_threshold,
                                'expansion_threshold': expansion_threshold,
                                'reversal_threshold': reversal_threshold,
                                'clustering_eps': clustering_eps,
                                'min_track_length': min_track_length
                            }
                        }
                        
                        st.success("Anomaly detection completed successfully!")
                    else:
                        st.warning("No tracks meet the minimum length requirement for analysis.")
                        
                except Exception as e:
                    st.error(f"Error during anomaly detection: {str(e)}")
        
        # Display results if available
        if "anomaly_detection" in st.session_state.analysis_results:
            results = st.session_state.analysis_results["anomaly_detection"]
            anomaly_results = results['results']
            
            # Initialize visualizer and create dashboard
            visualizer = AnomalyVisualizer()
            
            # Filter tracks for visualization
            filtered_tracks = st.session_state.tracks_data.groupby('track_id').filter(
                lambda x: len(x) >= results['parameters']['min_track_length']
            )
            
            # Create comprehensive anomaly dashboard
            visualizer.create_anomaly_dashboard(filtered_tracks, anomaly_results)

# Report Generation Page
elif st.session_state.active_page == "Report Generation":
    st.title("Analysis Report Generation")
    if REPORT_GENERATOR_AVAILABLE:
        show_enhanced_report_generator()
    else:
        st.error("Report generator module not available.")

# Footer
st.markdown("---")
st.markdown("SPT Analysis Tool - For analyzing single particle tracking data")
