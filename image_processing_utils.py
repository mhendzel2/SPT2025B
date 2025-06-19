"""
Image Processing Utilities for SPT Analysis Application.
Provides noise reduction and preprocessing capabilities for microscopy images.
"""

import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Any
import warnings

# Core image processing imports
try:
    from skimage.restoration import denoise_nl_means
    from skimage.filters import gaussian, median
    from skimage.morphology import disk
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    st.warning("scikit-image not available. Image processing features will be limited.")

warnings.filterwarnings('ignore')


def apply_noise_reduction(image_frames: List[np.ndarray], method: str = 'gaussian', params: Dict[str, Any] = None) -> List[np.ndarray]:
    """
    Apply noise reduction to a list of image frames.

    Parameters
    ----------
    image_frames : list
        List of 2D NumPy arrays (image frames).
    method : str, optional
        Noise reduction method: 'gaussian', 'median', 'nl_means'.
        Defaults to 'gaussian'.
    params : dict, optional
        Parameters for the chosen method.
        - gaussian: {'sigma': float (e.g., 1)}
        - median: {'disk_radius': int (e.g., 2)}
        - nl_means: {'h': float (controls filter strength, e.g. 0.1), 
                     'sigma': float (estimated noise std dev, e.g., 0.08),
                     'patch_size': int (e.g., 5),
                     'patch_distance': int (e.g., 6)}

    Returns
    -------
    list
        List of processed 2D NumPy arrays.
    """
    if not image_frames:
        return []
    
    if not SKIMAGE_AVAILABLE:
        st.error("scikit-image is required for noise reduction. Please install it: pip install scikit-image")
        return image_frames

    processed_frames = []
    if params is None:
        params = {}

    for i, frame in enumerate(image_frames):
        original_dtype = frame.dtype
        
        # Convert to float in [0, 1] range for most scikit-image filters
        frame_float = frame.astype(np.float64)
        min_val, max_val = np.min(frame_float), np.max(frame_float)
        if max_val > min_val:
            frame_norm = (frame_float - min_val) / (max_val - min_val)
        else:  # Handle constant image
            frame_norm = np.zeros_like(frame_float) if min_val == 0 else frame_float / (max_val if max_val != 0 else 1.0)

        denoised_frame_norm = None

        try:
            if method == 'gaussian':
                sigma = params.get('sigma', 1)
                denoised_frame_norm = gaussian(frame_norm, sigma=sigma, preserve_range=True, channel_axis=None)
                
            elif method == 'median':
                disk_radius = params.get('disk_radius', 2)
                if np.issubdtype(original_dtype, np.integer):
                    temp_int_frame = (frame_norm * np.iinfo(original_dtype).max).astype(original_dtype)
                    denoised_int_frame = median(temp_int_frame, footprint=disk(disk_radius))
                    denoised_frame_norm = denoised_int_frame.astype(np.float64) / np.iinfo(original_dtype).max
                else:
                    temp_u8_frame = (frame_norm * 255).astype(np.uint8)
                    denoised_u8_frame = median(temp_u8_frame, footprint=disk(disk_radius))
                    denoised_frame_norm = denoised_u8_frame.astype(np.float64) / 255.0

            elif method == 'nl_means':
                h_param = params.get('h', 0.1)
                sigma_param = params.get('sigma', 0.08)
                patch_size = params.get('patch_size', 5)
                patch_distance = params.get('patch_distance', 6)
                denoised_frame_norm = denoise_nl_means(
                    frame_norm,
                    h=h_param * sigma_param,
                    sigma=sigma_param,
                    patch_size=patch_size,
                    patch_distance=patch_distance,
                    channel_axis=None
                )
            else:
                denoised_frame_norm = frame_norm  # No processing for unknown method

        except Exception as e:
            st.warning(f"Error applying {method} to frame {i}: {str(e)}")
            denoised_frame_norm = frame_norm

        # Scale back to original range and type
        if denoised_frame_norm is not None:
            if max_val > min_val:
                denoised_frame_restored = denoised_frame_norm * (max_val - min_val) + min_val
            else:
                denoised_frame_restored = denoised_frame_norm * (max_val if max_val != 0 else 1.0) + min_val

            # Ensure clipping to original data type's limits if integer
            if np.issubdtype(original_dtype, np.integer):
                d_min, d_max = np.iinfo(original_dtype).min, np.iinfo(original_dtype).max
                denoised_frame_restored = np.clip(denoised_frame_restored, d_min, d_max)

            processed_frames.append(denoised_frame_restored.astype(original_dtype))
        else:
            processed_frames.append(frame)

    return processed_frames


def apply_ai_noise_reduction(image_frames: List[np.ndarray], model_choice: str = "default_model", params: Dict[str, Any] = None) -> List[np.ndarray]:
    """
    Apply AI-based noise reduction to a list of image frames.
    
    This is a framework for AI-based denoising. Actual implementation requires
    specific AI models and their dependencies.
    
    Parameters
    ----------
    image_frames : list
        List of 2D NumPy arrays (image frames)
    model_choice : str
        Choice of AI model ('noise2void', 'care', 'custom')
    params : dict, optional
        Model-specific parameters
        
    Returns
    -------
    list
        List of processed 2D NumPy arrays
    """
    if not image_frames:
        return []

    # Framework for AI denoising - requires specific model implementations
    if model_choice == "noise2void":
        try:
            # This would require the Noise2Void library
            # from n2v.models import N2V
            st.info("Noise2Void requires specific model files and configuration. Please ensure you have trained models available.")
            return image_frames
        except ImportError:
            st.error("Noise2Void (n2v) library not installed. Install with: pip install n2v")
            return image_frames
    
    elif model_choice == "care":
        try:
            # This would require the CSBDeep/CARE library
            # from csbdeep.models import CARE
            st.info("CARE requires specific model files and configuration. Please ensure you have trained models available.")
            return image_frames
        except ImportError:
            st.error("CARE (csbdeep) library not installed. Install with: pip install csbdeep")
            return image_frames
    
    else:
        st.warning(f"AI model '{model_choice}' not implemented. Returning original images.")
        return image_frames


def load_image_file(uploaded_file) -> Optional[np.ndarray]:
    """
    Load an image file from uploaded file object.
    
    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        The uploaded file object from Streamlit
        
    Returns
    -------
    np.ndarray or None
        Loaded image array or None if loading failed
    """
    try:
        # Import PIL for image loading
        from PIL import Image
        import io
        
        # Read file content
        file_content = uploaded_file.read()
        
        # Create BytesIO object
        image_buffer = io.BytesIO(file_content)
        
        # Load image using PIL
        pil_image = Image.open(image_buffer)
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Handle different image formats
        if len(image_array.shape) == 3 and image_array.shape[2] > 1:
            # Multi-channel image - convert to grayscale if needed
            if image_array.shape[2] == 3:  # RGB
                image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            elif image_array.shape[2] == 4:  # RGBA
                image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        return image_array.astype(np.float32)
        
    except Exception as e:
        st.error(f"Error loading image file {uploaded_file.name}: {str(e)}")
        return None

def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for display purposes.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array
        
    Returns
    -------
    np.ndarray
        Normalized image as uint8
    """
    display_image = image.copy()
    
    if display_image.dtype != np.uint8:
        min_val = np.min(display_image)
        max_val = np.max(display_image)
        
        if max_val > min_val:
            display_image = (display_image - min_val) / (max_val - min_val)
            display_image = (display_image * 255).astype(np.uint8)
        elif np.issubdtype(display_image.dtype, np.floating):
            display_image = (display_image * 255).astype(np.uint8) if max_val <= 1 else display_image.astype(np.uint8)
        else:
            display_image = display_image.astype(np.uint8)
    
    return display_image


def create_timepoint_preview(image_data: List[np.ndarray], selected_frame: int = 0) -> Optional[np.ndarray]:
    """
    Create a preview image for a specific timepoint.
    
    Parameters
    ----------
    image_data : list
        List of image frames
    selected_frame : int
        Frame index to display
        
    Returns
    -------
    np.ndarray or None
        Preview image or None if invalid frame
    """
    if not image_data or selected_frame < 0 or selected_frame >= len(image_data):
        return None
    
    preview_image = image_data[selected_frame]
    return normalize_image_for_display(preview_image)


def get_image_statistics(image_frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Calculate basic statistics for image frames.
    
    Parameters
    ----------
    image_frames : list
        List of image frames
        
    Returns
    -------
    dict
        Image statistics including dimensions, data types, intensity ranges
    """
    if not image_frames:
        return {}
    
    stats = {
        'num_frames': len(image_frames),
        'frame_shapes': [img.shape for img in image_frames],
        'data_types': [str(img.dtype) for img in image_frames],
        'intensity_ranges': [(np.min(img), np.max(img)) for img in image_frames],
        'mean_intensities': [np.mean(img) for img in image_frames],
        'std_intensities': [np.std(img) for img in image_frames]
    }
    
    return stats