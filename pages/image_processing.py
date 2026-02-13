import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from PIL import Image
from ui_utils import get_available_masks, store_mask
from image_processing_utils import normalize_image_for_display, apply_noise_reduction
from segmentation import (
    density_map_threshold, enhanced_threshold_image,
    analyze_density_segmentation, gaussian_mixture_segmentation,
    bayesian_gaussian_mixture_segmentation, compare_segmentation_methods
)
from channel_manager import channel_manager

# Check for advanced segmentation availability
try:
    from advanced_segmentation import integrate_advanced_segmentation_with_app
    ADVANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ADVANCED_SEGMENTATION_AVAILABLE = False

def _combine_channels(multichannel_img: np.ndarray, channel_indices: list[int], mode: str = "average", weights: list[float] = None) -> np.ndarray:
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

def show_image_processing_page():
    st.title("Image Processing & Nuclear Density Analysis")

    img_tabs = st.tabs([
        "Segmentation",
        "Nuclear Density Analysis",
        "Advanced Segmentation",
        "Export Results"
    ])

    with img_tabs[0]:
        st.subheader("Image Segmentation")
        if 'mask_images' not in st.session_state or st.session_state.mask_images is None:
            st.warning("Upload mask images in the Data Loading tab first.")
        else:
            mask_image_data = st.session_state.mask_images
            multichannel_detected = False

            if isinstance(mask_image_data, np.ndarray) and len(mask_image_data.shape) == 3 and mask_image_data.shape[2] > 1:
                multichannel_detected = True
                num_channels = mask_image_data.shape[2]
                multichannel_data = mask_image_data
            elif isinstance(mask_image_data, list) and len(mask_image_data) == 1 and len(mask_image_data[0].shape) == 3 and mask_image_data[0].shape[2] > 1:
                multichannel_detected = True
                num_channels = mask_image_data[0].shape[2]
                multichannel_data = mask_image_data[0]

            if multichannel_detected:
                st.info(f"Multichannel image detected with {num_channels} channels")
                default_sel = st.session_state.get("segmentation_channels", [0])
                segmentation_channels = st.multiselect("Choose channel(s)", list(range(num_channels)), default=default_sel)
                fusion_mode = st.selectbox("Fusion mode", ["average", "max", "min", "sum"])
                current_image = _combine_channels(multichannel_data, segmentation_channels, fusion_mode)
            elif isinstance(mask_image_data, list) and len(mask_image_data) > 0:
                current_image = mask_image_data[0]
            else:
                current_image = mask_image_data

            st.image(normalize_image_for_display(current_image), caption="Selected Image")

            seg_method = st.selectbox("Segmentation Method", ["Otsu", "Triangle", "Manual", "Density Map"])

            if st.button("Apply Segmentation"):
                if seg_method == "Otsu":
                    mask, _ = enhanced_threshold_image(current_image, method='otsu')
                elif seg_method == "Triangle":
                    mask, _ = enhanced_threshold_image(current_image, method='triangle')
                elif seg_method == "Manual":
                    mask, _ = enhanced_threshold_image(current_image, method='manual', manual_threshold=np.mean(current_image))
                elif seg_method == "Density Map":
                    res = density_map_threshold(current_image)
                    mask = res['mask_in']

                st.session_state.segmentation_mask = mask
                store_mask(f"Simple_{seg_method}", mask.astype(np.uint8), "Simple", f"{seg_method} segmentation")
                st.image((mask * 255).astype(np.uint8), caption="Result")
                st.success("Mask generated!")

    with img_tabs[1]:
        st.subheader("Nuclear Density Analysis")
        if 'nucleus_mask' not in st.session_state:
            st.info("Detect nuclear boundaries first (Segmentation tab).")
        else:
            if st.button("Run Density Classification"):
                # Placeholder logic - refer to original app.py for full implementation
                st.success("Density classification complete (Placeholder)")

    with img_tabs[2]:
        st.subheader("Advanced Segmentation")
        if ADVANCED_SEGMENTATION_AVAILABLE:
            integrate_advanced_segmentation_with_app()
        else:
            st.error("Advanced segmentation module not available.")

    with img_tabs[3]:
        st.subheader("Export")
        if 'nucleus_mask' in st.session_state:
            if st.button("Export Mask"):
                mask_img = Image.fromarray((st.session_state.nucleus_mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                st.download_button("Download PNG", buf.getvalue(), "mask.png", "image/png")
