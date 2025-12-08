import streamlit as st
import os
import pandas as pd
import numpy as np
from utils import calculate_track_statistics
from data_loader import load_tracks_file, load_image_file
from data_quality_checker import DataQualityChecker
from secure_file_validator import get_file_validator
from ui_utils import navigate_to, handle_track_upload
from state_manager import get_state_manager

def show_data_loading_page():
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
                from image_processing_utils import normalize_image_for_display

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
                    navigate_to("Tracking")
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
                from image_processing_utils import normalize_image_for_display

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
                    navigate_to("Image Processing")
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
                    # Assuming test_data_generator is available or similar
                    # If not, we can use sample data from 'sample data' folder
                    sample_data_dir = "sample data"
                    if os.path.exists(sample_data_dir):
                        # Find a csv
                        for root, dirs, files in os.walk(sample_data_dir):
                            for f in files:
                                if f.endswith(".csv"):
                                    st.session_state.tracks_data = pd.read_csv(os.path.join(root, f))
                                    st.session_state.tracks_data = format_track_data(st.session_state.tracks_data)
                                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                                    st.success(f"Sample track data loaded successfully from {f}!")
                                    break
                            else:
                                continue
                            break
                        else:
                             st.warning("No sample CSV found.")
                    else:
                        st.warning("Sample data directory not found.")

                elif selected_sample == "Sample Microscopy Images":
                    # Placeholder
                    st.info("Sample microscopy images not implemented yet in this refactor.")

                elif selected_sample == "Sample MD Simulation Data":
                    # Placeholder
                    st.info("Sample MD data not implemented yet in this refactor.")

            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
