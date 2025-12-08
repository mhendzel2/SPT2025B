import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import plotly.express as px
from ui_utils import create_mask_selection_ui, get_available_masks
from image_processing_utils import normalize_image_for_display, apply_noise_reduction
from visualization import plot_tracks
from tracking import detect_particles, link_particles
from advanced_tracking import ParticleFilter, AdvancedTracking, bayesian_detection_refinement
from channel_manager import channel_manager

def _combine_channels(multichannel_img: np.ndarray, channel_indices: list[int], mode: str = "average", weights: list[float] = None) -> np.ndarray:
    """Combine multiple channels from a HxWxC image into a single 2D image."""
    if multichannel_img is None or multichannel_img.ndim != 3 or multichannel_img.shape[2] == 0:
        raise ValueError("Expected a multichannel image of shape (H, W, C)")
    if not channel_indices:
        channel_indices = [0]
    c = multichannel_img.shape[2]
    channel_indices = [ci for ci in channel_indices if 0 <= ci < c]
    if not channel_indices:
        channel_indices = [0]
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
            weights = [1.0 / stack.shape[2]] * stack.shape[2]
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) if np.sum(w) != 0 else 1.0)
        return np.tensordot(stack, w, axes=([2], [0]))
    return np.mean(stack, axis=2)

def show_tracking_page():
    st.title("Particle Detection and Tracking")

    if 'image_data' not in st.session_state or st.session_state.image_data is None:
        st.warning("No image data loaded. Please upload images first.")
        if st.button("Go to Data Loading"):
            st.session_state.active_page = "Data Loading"
            st.rerun()
    else:
        tab1, tab2, tab3 = st.tabs(["Particle Detection", "Particle Linking", "Track Results"])

        with tab1:
            st.header("Particle Detection")

            mask_selection_ui = create_mask_selection_ui("tracking")
            selected_mask, selected_classes, segmentation_method = mask_selection_ui

            if selected_mask:
                st.info(f"Detection will be performed on mask: {selected_mask}")
                if selected_classes:
                    st.info(f"Using mask classes: {', '.join(map(str, selected_classes))}")

            st.divider()

            try:
                _first_frame = st.session_state.image_data[0]
                _is_mc = isinstance(_first_frame, np.ndarray) and _first_frame.ndim == 3 and _first_frame.shape[2] > 1
                _num_ch = int(_first_frame.shape[2]) if _is_mc else 1
            except Exception:
                _is_mc, _num_ch = False, 1

            if _is_mc:
                st.subheader("Tracking Channel Selection")
                track_label_key = "tracking_channel_labels"
                if track_label_key not in st.session_state:
                    st.session_state[track_label_key] = {}
                def _track_label(idx: int) -> str:
                    saved = st.session_state[track_label_key].get(idx)
                    return f"Channel {idx + 1}" if not saved else f"{saved} (C{idx + 1})"

                with st.expander("Name channels (optional)", expanded=False):
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
                st.session_state.tracking_channels = tracking_channels or [0]
                st.session_state.tracking_fusion_mode = tracking_fusion_mode
                if tracking_weights is not None:
                    st.session_state.tracking_weights = ",".join(map(str, tracking_weights))
                sel_label = ", ".join([f"C{c+1}" for c in (tracking_channels or [0])])
                st.info(f"Tracking will use {sel_label} with '{tracking_fusion_mode}' fusion.")

            st.subheader("Detection Parameters")

            with st.expander("Restrict detection to segmentation classes (optional)", expanded=False):
                roi_enabled = st.checkbox("Enable ROI restriction", value=False, key="tracking_roi_enabled")
                roi_mask = None
                if roi_enabled:
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
                        roi_mask = np.isin(sel_mask, sel_classes)
                        st.caption(f"ROI built from mask '{sel_mask_name}' with classes {sel_classes}")

            with st.expander("🔍 Real-time Detection Tuning", expanded=False):
                st.write("**Preview detection settings on a test frame before running full detection**")

                num_frames = len(st.session_state.image_data)
                if num_frames > 1:
                    test_frame_idx = st.slider(
                        "Test Frame",
                        min_value=0,
                        max_value=num_frames - 1,
                        value=min(5, num_frames - 1),
                        key="tracking_test_frame"
                    )
                else:
                    test_frame_idx = 0

                raw_test = st.session_state.image_data[test_frame_idx]
                if isinstance(raw_test, np.ndarray) and raw_test.ndim == 3 and raw_test.shape[2] > 1:
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

                if frame_max <= frame_min:
                    frame_max = frame_min + 1.0

                range_size = frame_max - frame_min
                if range_size > 0:
                    step_size = max(0.01, range_size / 100.0)
                else:
                    step_size = 0.01

                preview_col1, preview_col2 = st.columns(2)

                with preview_col1:
                    default_threshold = max(frame_min, min(frame_max, frame_mean + frame_std))

                    quick_threshold = st.slider(
                        "Quick Intensity Threshold",
                        min_value=frame_min,
                        max_value=frame_max,
                        value=default_threshold,
                        step=step_size,
                        help=f"Frame range: {frame_min:.1f} - {frame_max:.1f}"
                    )

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
                        else:
                            noise_strength = st.slider("Denoising Strength", 0.1, 2.0, 0.3, 0.1)

                    processed_frame = test_frame.copy()
                    if noise_reduction_enabled:
                        try:
                            if noise_method == "Gaussian":
                                processed_frame = apply_noise_reduction([processed_frame], 'gaussian', {'sigma': noise_strength})[0]
                            elif noise_method == "Median":
                                processed_frame = apply_noise_reduction([processed_frame], 'median', {'disk_size': int(noise_strength)})[0]
                            else:
                                processed_frame = apply_noise_reduction([processed_frame], 'nlm', {'h': noise_strength})[0]
                        except Exception as e:
                            st.warning(f"Noise reduction failed: {str(e)}")
                            processed_frame = test_frame.copy()

                    threshold_mask = processed_frame > quick_threshold
                    if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                        if roi_mask.shape == threshold_mask.shape:
                            threshold_mask = threshold_mask & roi_mask
                    detected_pixels = int(np.sum(threshold_mask))
                    total_pixels = threshold_mask.size
                    detection_percent = (detected_pixels / total_pixels) * 100

                    st.metric("Detection Coverage", f"{detection_percent:.1f}%")
                    st.metric("Detected Pixels", f"{detected_pixels:,}")

                with preview_col2:
                    if st.checkbox("Show Threshold Preview", value=True):
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
                default_method = "LoG"
                default_size = 3.0

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

                    if optimal['method'] == 'LoG':
                        optimal_min_sigma = optimal.get('min_sigma', 0.5)
                        optimal_max_sigma = optimal.get('max_sigma', 3.0)
                        optimal_threshold = optimal.get('threshold', 0.05)
                    elif optimal['method'] == 'DoG':
                        optimal_sigma1 = optimal.get('sigma1', 1.0)
                        optimal_sigma2 = optimal.get('sigma2', 2.0)
                        raw_threshold = optimal.get('threshold', 90.0)
                        optimal_percentile = max(5.0, min(99.0, raw_threshold))
                    elif optimal['method'] == 'Intensity':
                        optimal_intensity_percentile = optimal.get('percentile_thresh', 95.0)
                    elif optimal['method'] == 'Wavelet':
                        optimal_wavelet_threshold = optimal.get('threshold_factor', 1.0)
                        optimal_wavelet_type = optimal.get('wavelet_type', 'mexican_hat')

                    st.session_state.apply_optimal = False

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
                if detection_method == "LoG":
                    min_sigma = st.number_input("Min Sigma", min_value=0.1, max_value=5.0, value=optimal_min_sigma, step=0.1)
                    max_sigma = st.number_input("Max Sigma", min_value=0.5, max_value=10.0, value=optimal_max_sigma, step=0.1)
                    threshold_factor = st.number_input("Threshold", min_value=0.001, max_value=1.0, value=optimal_threshold, step=0.001, format="%.3f")

                elif detection_method == "DoG":
                    sigma1 = st.number_input("Sigma 1", min_value=0.1, max_value=5.0, value=optimal_sigma1, step=0.1)
                    sigma2 = st.number_input("Sigma 2", min_value=0.5, max_value=10.0, value=optimal_sigma2, step=0.1)
                    threshold_factor = st.number_input("Percentile Threshold", min_value=5.0, max_value=99.0, value=max(5.0, min(99.0, optimal_percentile)), step=1.0)

                elif detection_method == "Intensity":
                    threshold_factor = st.number_input("Percentile Threshold", min_value=80.0, max_value=99.5, value=optimal_intensity_percentile, step=0.5)

                elif detection_method == "Wavelet":
                    wavelet_type = st.selectbox("Wavelet Type", ["mexican_hat", "morlet", "ricker"], index=["mexican_hat", "morlet", "ricker"].index(optimal_wavelet_type))
                    threshold_factor = st.number_input("Threshold Factor", min_value=0.001, max_value=10.0, value=optimal_wavelet_threshold, step=0.001, format="%.3f")

                min_distance = st.number_input("Minimum Distance (pixels)", min_value=0.1, max_value=100.0, value=5.0, step=0.1, format="%.1f")

            num_sigma = 5
            overlap_threshold = 0.5
            threshold_mode = "Auto (Otsu)"
            block_size = 11
            manual_threshold = 0.1

            with st.expander("Advanced Detection Parameters", expanded=False):
                if detection_method == "LoG":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        num_sigma = st.slider("Number of Sigma Steps", 1, 20, 5)
                    with adv_col2:
                        overlap_threshold = st.slider("Overlap Threshold", 0.0, 1.0, 0.5, 0.05)

                elif detection_method == "DoG":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        threshold_mode = st.selectbox("Threshold Mode", ["Auto (Otsu)", "Manual", "Adaptive"])
                    with adv_col2:
                        if threshold_mode == "Manual":
                            manual_threshold = st.slider("Manual Threshold", 0.01, 2.0, 0.1, 0.01, format="%.2f")
                        elif threshold_mode == "Adaptive":
                            block_size = st.slider("Adaptive Block Size", 3, 51, 11, 2)

                elif detection_method == "Wavelet":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        wavelet_type = st.selectbox("Wavelet Type", ["sym8", "db4", "haar", "coif2", "bior2.2"])
                        wavelet_levels = st.slider("Decomposition Levels", 1, 6, int(np.log2(particle_size)) + 2)
                    with adv_col2:
                        detail_enhancement = st.slider("Detail Enhancement", 0.1, 5.0, 1.0, 0.1)
                        noise_reduction = st.slider("Noise Reduction", 0.0, 1.0, 0.1, 0.01)

                elif detection_method == "Intensity":
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        intensity_mode = st.selectbox("Threshold Mode", ["Auto (Otsu)", "Manual", "Percentile", "Mean + Std"])
                        if intensity_mode == "Percentile":
                            percentile_value = st.slider("Intensity Percentile", 50, 99, 90)
                        elif intensity_mode == "Mean + Std":
                            std_multiplier = st.number_input("Std Dev Multiplier", 0.1, 10.0, 2.0, 0.1)
                    with adv_col2:
                        morphology_cleanup = st.checkbox("Morphological Cleanup", value=True)
                        if morphology_cleanup:
                            erosion_size = st.slider("Erosion Size", 0, 5, 1)
                            dilation_size = st.slider("Dilation Size", 0, 5, 1)

            if len(st.session_state.image_data) > 1:
                frame_idx = st.slider("Frame to preview", 0, len(st.session_state.image_data) - 1, 0)
            else:
                frame_idx = 0
                st.info("Single frame loaded")

            col1, col2 = st.columns(2)
            with col1:
                single_frame_button = st.button("🔍 Run Detection (Current Frame)", type="secondary")
            with col2:
                all_frames_button = st.button("🚀 Run Detection (All Frames)", type="primary")

            if single_frame_button:
                with st.spinner("Detecting particles on current frame..."):
                    try:
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
                        if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                            if roi_mask.shape == current_frame.shape:
                                current_frame = current_frame * roi_mask

                        if detection_method == "LoG":
                            from skimage.feature import blob_log
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
                            coeffs = pywt.wavedec2(current_frame, wavelet_type, level=wavelet_levels)
                            coeffs_enhanced = list(coeffs)
                            for i in range(1, len(coeffs)):
                                if isinstance(coeffs[i], tuple):
                                    enhanced_details = []
                                    for detail in coeffs[i]:
                                        enhanced = detail * detail_enhancement
                                        enhanced[np.abs(enhanced) < noise_reduction * np.std(enhanced)] = 0
                                        enhanced_details.append(enhanced)
                                    coeffs_enhanced[i] = tuple(enhanced_details)
                            enhanced = pywt.waverec2(coeffs_enhanced, wavelet_type)
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
                            if intensity_mode == "Auto (Otsu)":
                                thresh = filters.threshold_otsu(current_frame)
                            elif intensity_mode == "Manual":
                                thresh = threshold_factor * np.mean(current_frame)
                            elif intensity_mode == "Percentile":
                                thresh = np.percentile(current_frame, percentile_value)
                            elif intensity_mode == "Mean + Std":
                                thresh = np.mean(current_frame) + std_multiplier * np.std(current_frame)
                            mask = current_frame > thresh
                            if morphology_cleanup:
                                if erosion_size > 0:
                                    mask = morphology.erosion(mask, morphology.disk(erosion_size))
                                if dilation_size > 0:
                                    mask = morphology.dilation(mask, morphology.disk(dilation_size))
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
                                        if region_id == 0:
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

                        if not particles_df.empty:
                            particles = particles_df[['x', 'y']].values
                            particle_info = particles_df.to_dict('records')
                        else:
                            particles = np.array([]).reshape(0, 2)
                            particle_info = []

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

                        if 'all_detections' not in st.session_state:
                            st.session_state.all_detections = {}

                        enhanced_particles = []
                        for i, particle in enumerate(particles):
                            enhanced_particle = [
                                particle[0],
                                particle[1],
                                particle_info[i]['intensity'],
                                particle_info[i]['SNR']
                            ]
                            enhanced_particles.append(enhanced_particle)

                        st.session_state.all_detections[frame_idx] = enhanced_particles
                        st.success(f"Detected {len(particles)} particles using {detection_method}")

                        if len(particles) > 0:
                            avg_intensity = np.mean([p['intensity'] for p in particle_info])
                            avg_snr = np.mean([p['SNR'] for p in particle_info])
                            st.info(f"Average intensity: {avg_intensity:.2f}, Average SNR: {avg_snr:.2f}")

                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        st.session_state.detection_results = {"frame": frame_idx, "detected": 0}

            if all_frames_button:
                total_frames = len(st.session_state.image_data)
                with st.spinner(f"Running detection on all {total_frames} frames..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    st.session_state.all_detections = {}
                    total_particles_detected = 0

                    try:
                        from tracking import detect_particles
                        for current_frame_idx in range(total_frames):
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
                            if 'tracking_roi_enabled' in st.session_state and st.session_state.tracking_roi_enabled and roi_mask is not None:
                                if roi_mask.shape == current_frame.shape:
                                    current_frame = current_frame * roi_mask

                            if detection_method == "LoG":
                                from skimage import feature
                                blobs = feature.blob_log(current_frame, min_sigma=min_sigma, max_sigma=max_sigma,
                                                       num_sigma=num_sigma, threshold=threshold_factor)
                                if len(blobs) > 0:
                                    particles = blobs[:, :2]
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
                                coeffs = pywt.wavedec2(current_frame, wavelet_type, level=wavelet_levels)
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
                                enhanced_image = pywt.waverec2(coeffs_enhanced, wavelet_type)
                                enhanced_image = np.abs(enhanced_image)
                                thresh = threshold_factor * np.mean(enhanced_image)
                                binary = enhanced_image > thresh
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
                                binary = current_frame > thresh
                                if morphology_cleanup:
                                    if erosion_size > 0:
                                        binary = morphology.binary_erosion(binary, morphology.disk(erosion_size))
                                    if dilation_size > 0:
                                        binary = morphology.binary_dilation(binary, morphology.disk(dilation_size))
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
                                particles = np.array([]).reshape(0, 2)
                                particle_info = []

                            enhanced_particles = []
                            for i, particle in enumerate(particles):
                                enhanced_particle = [
                                    particle[0],
                                    particle[1],
                                    particle_info[i]['intensity'],
                                    particle_info[i]['SNR']
                                ]
                                enhanced_particles.append(enhanced_particle)

                            st.session_state.all_detections[current_frame_idx] = enhanced_particles
                            total_particles_detected += len(enhanced_particles)

                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"🎉 Detection completed on all {total_frames} frames!")
                        st.info(f"📊 Total particles detected: **{total_particles_detected}**")
                        with st.expander("📋 Detection Summary by Frame"):
                            for frame_num in sorted(st.session_state.all_detections.keys()):
                                count = len(st.session_state.all_detections[frame_num])
                                st.write(f"Frame {frame_num}: {count} particles")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error during batch detection: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

            display_image = st.session_state.image_data[frame_idx].copy()
            if len(display_image.shape) == 3:
                if display_image.shape[2] > 4:
                    display_image = np.max(display_image, axis=2)
                    st.info(f"⚠️ 3D stack detected ({display_image.shape[2]} slices). Showing max projection.")
                elif display_image.shape[2] == 3 or display_image.shape[2] == 4:
                    pass
                elif display_image.shape[2] == 2:
                    display_image = display_image[:, :, 0]
                else:
                    display_image = display_image.squeeze()

            if len(display_image.shape) > 2 and display_image.shape[2] not in [3, 4]:
                display_image = display_image[:, :, 0] if display_image.shape[2] > 0 else display_image.squeeze()

            if display_image.dtype != np.uint8:
                if display_image.max() <= 1.0:
                    display_image = (display_image * 255).astype(np.uint8)
                else:
                    display_image = ((display_image - display_image.min()) /
                                   (display_image.max() - display_image.min()) * 255).astype(np.uint8)

            if (st.session_state.detection_results and
                st.session_state.detection_results['frame'] == frame_idx and
                'particles' in st.session_state.detection_results and
                len(st.session_state.detection_results['particles']) > 0):

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(display_image, cmap='gray', aspect='equal')
                particles = st.session_state.detection_results['particles']
                for particle in particles:
                    x, y = particle[0], particle[1]
                    circle = Circle((x, y), radius=particle_size/2, fill=False,
                                  color='red', linewidth=2, alpha=0.8)
                    ax.add_patch(circle)
                ax.set_title(f"Frame {frame_idx} - {len(particles)} particles detected")
                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                ax.set_xticks([])
                ax.set_yticks([])
                st.pyplot(fig)
                plt.close()

                st.success(f"✓ Detected {len(particles)} particles using {st.session_state.detection_results['method']}")
                with st.expander("Detection Parameters Used"):
                    params = st.session_state.detection_results['parameters']
                    st.write(f"**Method:** {st.session_state.detection_results['method']}")
                    st.write(f"**Particle Size:** {params['particle_size']} pixels")
                    st.write(f"**Threshold Factor:** {params['threshold_factor']}")
                    st.write(f"**Minimum Distance:** {params['min_distance']} pixels")

                if st.button("Save Particle Coordinates"):
                    particle_df = pd.DataFrame(particles, columns=['x', 'y'])
                    particle_df['frame'] = frame_idx
                    csv = particle_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"particles_frame_{frame_idx}.csv",
                        mime="text/csv"
                    )
                    st.success("Particle coordinates ready for download!")
            else:
                st.image(display_image, caption=f"Frame {frame_idx}", use_container_width=True)
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
            st.subheader("Detection Status")

            if hasattr(st.session_state, 'all_detections') and st.session_state.all_detections:
                detected_frames = sorted(st.session_state.all_detections.keys())
                total_particles = sum(len(particles) for particles in st.session_state.all_detections.values())
                st.success(f"✅ Detection completed on {len(detected_frames)} frames with {total_particles} total particles")
                with st.expander("Frame-by-frame Detection Summary"):
                    for frame in detected_frames:
                        particle_count = len(st.session_state.all_detections[frame])
                        st.write(f"**Frame {frame}:** {particle_count} particles")
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
                    help="Method for linking particles between frames."
                )
                max_distance = st.slider("Maximum Distance (pixels)", 1.0, 50.0, 10.0, 0.5)
            with col2:
                max_frame_gap = st.slider("Maximum Frame Gap", 0, 10, 0)
                min_track_length = st.slider("Minimum Track Length", 2, 20, 5)

            if st.button("Run Linking", type="primary"):
                if not hasattr(st.session_state, 'all_detections') or not st.session_state.all_detections:
                    st.warning("⚠️ No detection results found. Please run particle detection on multiple frames first.")
                else:
                    with st.spinner("Running particle linking algorithm..."):
                        try:
                            from tracking import link_particles
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

                            if linking_method == "btrack":
                                from advanced_tracking import AdvancedTracking
                                advanced_tracker = AdvancedTracking()
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
                                tracks_data = advanced_tracker.track_particles(
                                    detection_dict,
                                    max_search_radius=max_distance,
                                    motion_std=5.0,
                                    measurement_std=2.0,
                                    min_track_length=min_track_length
                                )

                            elif linking_method in ["Trackpy", "Hungarian", "LAP"]:
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

                            if not tracks_data.empty:
                                st.session_state.tracks_data = tracks_data
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

            if hasattr(st.session_state, 'track_results') and st.session_state.track_results:
                st.success(f"Linked {st.session_state.track_results['n_tracks']} tracks.")

        with tab3:
            st.header("Track Results")
            if st.session_state.tracks_data is not None:
                st.subheader("Track Statistics")
                if st.session_state.track_statistics is not None:
                    st.dataframe(st.session_state.track_statistics)
                else:
                    with st.spinner("Calculating track statistics..."):
                        from utils import calculate_track_statistics
                        st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)
                    st.dataframe(st.session_state.track_statistics)

                st.subheader("Track Visualization")
                max_tracks_to_display = min(20, st.session_state.tracks_data['track_id'].nunique())
                fig = plot_tracks(
                    st.session_state.tracks_data.query(f"track_id < {max_tracks_to_display}"),
                    color_by='track_id'
                )
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Proceed to Analysis"):
                    st.session_state.active_page = "Analysis"
                    st.rerun()
            else:
                st.info("No track data available. Complete detection and linking or upload track data.")
                if st.button("Go to Data Loading"):
                    st.session_state.active_page = "Data Loading"
                    st.rerun()
