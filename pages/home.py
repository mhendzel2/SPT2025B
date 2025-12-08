import streamlit as st
import os
import pandas as pd
import numpy as np
from utils import format_track_data, calculate_track_statistics
from data_loader import load_image_file, load_tracks_file
from special_file_handlers import load_ms2_spots_file, load_imaris_file
from ui_utils import navigate_to
from state_manager import get_state_manager

def show_home_page():
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

            state_manager = get_state_manager()

            # Process depending on file type
            if file_extension in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                # Image file handling using centralized state
                try:
                    image_data = load_image_file(uploaded_file)
                    state_manager.load_image_data(image_data, {'filename': uploaded_file.name})
                    st.success(f"Image loaded successfully: {uploaded_file.name}")
                    navigate_to("Tracking")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")

            elif file_extension in ['.csv', '.xlsx', '.h5', '.json']:
                # Track data handling using centralized state
                try:
                    tracks_data = load_tracks_file(uploaded_file)
                    state_manager.load_tracks(tracks_data, source=uploaded_file.name)
                    # Automatically calculate track statistics using analysis manager
                    # (Assuming analysis_manager is available in session state or we instantiate it)
                    if 'analysis_manager' in st.session_state:
                        st.session_state.analysis_manager.calculate_track_statistics()
                    st.success(f"Track data loaded successfully: {uploaded_file.name}")
                    navigate_to("Analysis")
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
                    navigate_to("Analysis")
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
                    navigate_to("Analysis")
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
                    navigate_to("Analysis")
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
                        navigate_to("Analysis")
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
        elif quick_links == "Comparative analysis":
            navigate_to("Comparative Analysis")
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
