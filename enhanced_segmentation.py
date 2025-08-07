"""
Enhanced Segmentation Module for SPT Analysis Application.
Provides advanced segmentation tools including watershed segmentation
and improved particle-compartment classification.
"""

import numpy as np
import pandas as pd
from skimage import filters, morphology, measure, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def segment_image_channel_watershed(image_channel: np.ndarray, 
                                   min_distance_peaks: int = 7, 
                                   min_object_size: int = 50, 
                                   pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Segments an image using watershed algorithm based on distance transform.
    Useful for separating touching or overlapping cells/compartments.
    
    Parameters
    ----------
    image_channel : np.ndarray
        2D image array to segment
    min_distance_peaks : int
        Minimum distance between peaks for watershed seeds
    min_object_size : int
        Minimum size of objects to keep (in pixels)
    pixel_size : float
        Pixel size in micrometers
        
    Returns
    -------
    List[Dict[str, Any]]
        List of segmented compartments with properties
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")

    try:
        # Pre-processing: binary mask from Otsu
        thresh = filters.threshold_otsu(image_channel)
        binary_mask = image_channel > thresh
    except ValueError:
        # Otsu fails on flat images
        return []

    # Calculate Euclidean distance to background
    distance = ndi.distance_transform_edt(binary_mask)
    
    # Find local maxima for markers (seeds for watershed)
    coords = peak_local_max(distance, min_distance=min_distance_peaks, indices=binary_mask)
    mask_markers = np.zeros(distance.shape, dtype=bool)
    mask_markers[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_markers)
    
    # Perform watershed
    labels = watershed(-distance, markers, mask=binary_mask)
    
    # Remove small objects from watershed labels
    labels = morphology.remove_small_objects(labels, min_size=min_object_size)
    
    # Relabel to ensure consecutive numbering
    relabeled_regions, _ = ndi.label(labels > 0)
    
    region_properties = measure.regionprops(relabeled_regions, intensity_image=image_channel)
    segmented_compartments = []

    for i, props in enumerate(region_properties):
        if props.area < min_object_size:
            continue
            
        # Extract contour
        mask = (relabeled_regions == props.label)
        contours = measure.find_contours(mask.astype(float), 0.5)
        
        if len(contours) > 0:
            contour = contours[0]  # Take the largest contour
            contour_um = contour * pixel_size
            
            # Calculate properties
            centroid_um = np.array(props.centroid) * pixel_size
            area_um2 = props.area * (pixel_size ** 2)
            
            # Bounding box in micrometers
            bbox_um = np.array(props.bbox) * pixel_size
            
            compartment = {
                'id': f'watershed_{i+1}',
                'method': 'watershed',
                'centroid_um': centroid_um,
                'area_um2': area_um2,
                'bbox_um': bbox_um,
                'contour_um': contour_um.tolist(),
                'mean_intensity': props.intensity_mean,
                'max_intensity': props.intensity_max,
                'eccentricity': props.eccentricity,
                'solidity': props.solidity
            }
            segmented_compartments.append(compartment)
    
    return segmented_compartments

def classify_particles_by_contour(tracks_df: pd.DataFrame, 
                                 segmented_compartments: List[Dict], 
                                 pixel_size: float = 1.0) -> pd.DataFrame:
    """
    Classify particles by compartment using accurate contour-based point-in-polygon test.
    More accurate than bounding box classification for irregular shapes.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with x, y coordinates
    segmented_compartments : List[Dict]
        List of compartments with contour information
    pixel_size : float
        Pixel size in micrometers
        
    Returns
    -------
    pd.DataFrame
        Enhanced tracks with compartment classification
    """
    try:
        from matplotlib.path import Path
    except ImportError:
        # Fallback to simple bounding box classification
        return _classify_particles_by_bbox(tracks_df, segmented_compartments, pixel_size)
    
    tracks_enhanced = tracks_df.copy()
    tracks_enhanced['compartment_id'] = 'none'
    tracks_enhanced['in_compartment'] = False
    
    # Convert coordinates to micrometers if needed
    x_um = tracks_enhanced['x'] * pixel_size
    y_um = tracks_enhanced['y'] * pixel_size
    
    for comp in segmented_compartments:
        if 'contour_um' not in comp or not comp['contour_um']:
            continue
            
        contour_um = np.array(comp['contour_um'])
        if contour_um.shape[0] < 3:
            continue  # Not a valid polygon
        
        # Create Path object for point-in-polygon test
        try:
            path = Path(contour_um)
            
            # Test which points are inside this compartment
            particle_coords = np.column_stack((x_um, y_um))
            inside_mask = path.contains_points(particle_coords)
            
            # Update tracks that fall within this compartment and haven't been assigned
            update_mask = inside_mask & (tracks_enhanced['compartment_id'] == 'none')
            tracks_enhanced.loc[update_mask, 'compartment_id'] = comp['id']
            tracks_enhanced.loc[update_mask, 'in_compartment'] = True
            
        except Exception:
            # Fallback to bounding box for this compartment
            continue
    
    return tracks_enhanced

def _classify_particles_by_bbox(tracks_df: pd.DataFrame, 
                               segmented_compartments: List[Dict], 
                               pixel_size: float) -> pd.DataFrame:
    """Fallback classification using bounding boxes."""
    tracks_enhanced = tracks_df.copy()
    tracks_enhanced['compartment_id'] = 'none'
    tracks_enhanced['in_compartment'] = False
    
    x_um = tracks_enhanced['x'] * pixel_size
    y_um = tracks_enhanced['y'] * pixel_size
    
    for comp in segmented_compartments:
        if 'bbox_um' not in comp:
            continue
            
        bbox = comp['bbox_um']
        # Assuming bbox format is [min_row, min_col, max_row, max_col]
        if len(bbox) >= 4:
            min_y, min_x, max_y, max_x = bbox[:4]
            
            inside_mask = (
                (x_um >= min_x) & (x_um <= max_x) &
                (y_um >= min_y) & (y_um <= max_y)
            )
            
            update_mask = inside_mask & (tracks_enhanced['compartment_id'] == 'none')
            tracks_enhanced.loc[update_mask, 'compartment_id'] = comp['id']
            tracks_enhanced.loc[update_mask, 'in_compartment'] = True
    
    return tracks_enhanced

def adaptive_threshold_segmentation(image_channel: np.ndarray, 
                                   block_size: int = 51, 
                                   offset: float = 0.01,
                                   min_object_size: int = 50,
                                   pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Segments an image using adaptive thresholding for varying illumination conditions.
    
    Parameters
    ----------
    image_channel : np.ndarray
        2D image array to segment
    block_size : int
        Size of the local neighborhood for adaptive thresholding
    offset : float
        Constant subtracted from the mean/median of neighborhood
    min_object_size : int
        Minimum size of objects to keep
    pixel_size : float
        Pixel size in micrometers
        
    Returns
    -------
    List[Dict[str, Any]]
        List of segmented compartments
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Normalize image
    image_norm = (image_channel - image_channel.min()) / (image_channel.max() - image_channel.min())
    
    # Adaptive thresholding
    try:
        adaptive_thresh = filters.threshold_local(image_norm, block_size=block_size, offset=offset)
        binary_mask = image_norm > adaptive_thresh
    except Exception:
        # Fallback to Otsu if adaptive fails
        try:
            thresh = filters.threshold_otsu(image_norm)
            binary_mask = image_norm > thresh
        except ValueError:
            return []
    
    # Clean up the mask
    binary_mask = morphology.binary_opening(binary_mask, morphology.disk(2))
    binary_mask = morphology.binary_closing(binary_mask, morphology.disk(3))
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_object_size)
    
    # Label connected components
    labeled_regions = measure.label(binary_mask)
    region_properties = measure.regionprops(labeled_regions, intensity_image=image_channel)
    
    segmented_compartments = []
    
    for i, props in enumerate(region_properties):
        if props.area < min_object_size:
            continue
            
        # Extract contour
        mask = (labeled_regions == props.label)
        contours = measure.find_contours(mask.astype(float), 0.5)
        
        if len(contours) > 0:
            contour = contours[0]
            contour_um = contour * pixel_size
            
            # Calculate properties
            centroid_um = np.array(props.centroid) * pixel_size
            area_um2 = props.area * (pixel_size ** 2)
            bbox_um = np.array(props.bbox) * pixel_size
            
            compartment = {
                'id': f'adaptive_{i+1}',
                'method': 'adaptive_threshold',
                'centroid_um': centroid_um,
                'area_um2': area_um2,
                'bbox_um': bbox_um,
                'contour_um': contour_um.tolist(),
                'mean_intensity': props.intensity_mean,
                'max_intensity': props.intensity_max,
                'eccentricity': props.eccentricity,
                'solidity': props.solidity
            }
            segmented_compartments.append(compartment)
    
    return segmented_compartments

def multi_scale_segmentation(image_channel: np.ndarray,
                            scales: List[float] = [1.0, 2.0, 4.0],
                            min_object_size: int = 50,
                            pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Performs multi-scale segmentation to detect objects of different sizes.
    
    Parameters
    ----------
    image_channel : np.ndarray
        2D image array to segment
    scales : List[float]
        Different scales (sigma values) for Gaussian filtering
    min_object_size : int
        Minimum size of objects to keep
    pixel_size : float
        Pixel size in micrometers
        
    Returns
    -------
    List[Dict[str, Any]]
        List of segmented compartments from all scales
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")
    
    all_compartments = []
    
    for scale_idx, sigma in enumerate(scales):
        # Apply Gaussian filter at this scale
        filtered_image = filters.gaussian(image_channel, sigma=sigma)
        
        # Threshold
        try:
            thresh = filters.threshold_otsu(filtered_image)
            binary_mask = filtered_image > thresh
        except ValueError:
            continue
            
        # Morphological operations
        binary_mask = morphology.binary_opening(binary_mask, morphology.disk(int(sigma)))
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_object_size)
        
        # Label and extract properties
        labeled_regions = measure.label(binary_mask)
        region_properties = measure.regionprops(labeled_regions, intensity_image=image_channel)
        
        for i, props in enumerate(region_properties):
            if props.area < min_object_size:
                continue
                
            # Extract contour
            mask = (labeled_regions == props.label)
            contours = measure.find_contours(mask.astype(float), 0.5)
            
            if len(contours) > 0:
                contour = contours[0]
                contour_um = contour * pixel_size
                
                # Calculate properties
                centroid_um = np.array(props.centroid) * pixel_size
                area_um2 = props.area * (pixel_size ** 2)
                bbox_um = np.array(props.bbox) * pixel_size
                
                compartment = {
                    'id': f'scale{scale_idx}_{i+1}',
                    'method': f'multi_scale_sigma_{sigma}',
                    'scale': sigma,
                    'centroid_um': centroid_um,
                    'area_um2': area_um2,
                    'bbox_um': bbox_um,
                    'contour_um': contour_um.tolist(),
                    'mean_intensity': props.intensity_mean,
                    'max_intensity': props.intensity_max,
                    'eccentricity': props.eccentricity,
                    'solidity': props.solidity
                }
                all_compartments.append(compartment)
    
    return all_compartments

def merge_overlapping_compartments(compartments: List[Dict[str, Any]], 
                                  overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge compartments that have significant overlap (useful for multi-scale results).
    
    Parameters
    ----------
    compartments : List[Dict[str, Any]]
        List of compartments to merge
    overlap_threshold : float
        Minimum overlap ratio to trigger merging
        
    Returns
    -------
    List[Dict[str, Any]]
        Merged compartments list
    """
    if not compartments:
        return []
    
    try:
        from matplotlib.path import Path
        has_path = True
    except ImportError:
        has_path = False
    
    merged_compartments = []
    used_indices = set()
    
    for i, comp1 in enumerate(compartments):
        if i in used_indices:
            continue
            
        # Start a new merged compartment
        merged_comp = comp1.copy()
        merged_areas = [comp1['area_um2']]
        merged_indices = [i]
        
        # Check for overlaps with remaining compartments
        for j, comp2 in enumerate(compartments[i+1:], start=i+1):
            if j in used_indices:
                continue
            
            # Calculate overlap between compartments
            overlap_score = _calculate_compartment_overlap(comp1, comp2, has_path)
            
            if overlap_score >= overlap_threshold:
                merged_areas.append(comp2['area_um2'])
                merged_indices.append(j)
                used_indices.add(j)
        
        # Update merged compartment properties
        if len(merged_areas) > 1:
            merged_comp['area_um2'] = sum(merged_areas)
            merged_comp['id'] = f"merged_{'_'.join([str(idx) for idx in merged_indices])}"
            merged_comp['method'] = 'merged'
            
            # Update centroid as weighted average
            total_area = sum(merged_areas)
            weighted_centroid = np.zeros(2)
            for idx, area in zip(merged_indices, merged_areas):
                comp_centroid = np.array(compartments[idx]['centroid_um'])
                weighted_centroid += comp_centroid * area
            merged_comp['centroid_um'] = (weighted_centroid / total_area).tolist()
        
        merged_compartments.append(merged_comp)
        used_indices.add(i)
    
    return merged_compartments

def _calculate_compartment_overlap(comp1: Dict[str, Any], comp2: Dict[str, Any], 
                                 has_path: bool = True) -> float:
    """Calculate overlap score between two compartments."""
    try:
        if has_path and 'contour_um' in comp1 and 'contour_um' in comp2:
            from matplotlib.path import Path
            
            contour1 = np.array(comp1['contour_um'])
            contour2 = np.array(comp2['contour_um'])
            
            if len(contour1) < 3 or len(contour2) < 3:
                return _bbox_overlap_score(comp1, comp2)
            
            path1 = Path(contour1)
            path2 = Path(contour2)
            
            # Simple overlap check: test if centroids are inside other compartments
            overlap_score = 0
            if path1.contains_point(comp2['centroid_um']):
                overlap_score += 0.5
            if path2.contains_point(comp1['centroid_um']):
                overlap_score += 0.5
            
            return overlap_score
        else:
            return _bbox_overlap_score(comp1, comp2)
            
    except Exception:
        return _bbox_overlap_score(comp1, comp2)

def _bbox_overlap_score(comp1: Dict[str, Any], comp2: Dict[str, Any]) -> float:
    """Calculate overlap score using bounding boxes."""
    try:
        bbox1 = comp1.get('bbox_um', [])
        bbox2 = comp2.get('bbox_um', [])
        
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # bbox format: [min_row, min_col, max_row, max_col]
        y1_min, x1_min, y1_max, x1_max = bbox1[:4]
        y2_min, x2_min, y2_max, x2_max = bbox2[:4]
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        if x_overlap == 0 or y_overlap == 0:
            return 0.0
        
        intersection_area = x_overlap * y_overlap
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
        
    except Exception:
        return 0.0

def enhanced_watershed_segmentation(image_channel: np.ndarray,
                                   gaussian_sigma: float = 1.0,
                                   min_distance_peaks: int = 7,
                                   min_object_size: int = 50,
                                   pixel_size: float = 1.0,
                                   use_gradient: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced watershed segmentation with preprocessing options.
    
    Parameters
    ----------
    image_channel : np.ndarray
        2D image array to segment
    gaussian_sigma : float
        Sigma for Gaussian smoothing preprocessing
    min_distance_peaks : int
        Minimum distance between watershed seeds
    min_object_size : int
        Minimum size of objects to keep
    pixel_size : float
        Pixel size in micrometers
    use_gradient : bool
        Whether to use gradient magnitude for watershed
        
    Returns
    -------
    List[Dict[str, Any]]
        List of segmented compartments
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")
    
    # Preprocessing
    if gaussian_sigma > 0:
        smoothed = filters.gaussian(image_channel, sigma=gaussian_sigma)
    else:
        smoothed = image_channel.copy()
    
    # Create binary mask
    try:
        thresh = filters.threshold_otsu(smoothed)
        binary_mask = smoothed > thresh
    except ValueError:
        return []
    
    # Prepare image for watershed
    if use_gradient:
        # Use gradient magnitude
        gradient = filters.sobel(smoothed)
        watershed_image = gradient
    else:
        # Use distance transform (more common for cell segmentation)
        distance = ndi.distance_transform_edt(binary_mask)
        watershed_image = -distance
    
    # Find markers
    if use_gradient:
        # For gradient, find local minima
        local_minima = morphology.h_minima(watershed_image, h=0.1 * watershed_image.max())
        markers = measure.label(local_minima)
    else:
        # For distance transform, find local maxima
        coords = peak_local_max(
            distance, 
            min_distance=min_distance_peaks, 
            indices=binary_mask,
            threshold_abs=0.1 * distance.max()
        )
        mask_markers = np.zeros(watershed_image.shape, dtype=bool)
        if len(coords) > 0:
            mask_markers[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_markers)
    
    # Perform watershed
    if markers.max() > 0:
        labels = watershed(watershed_image, markers, mask=binary_mask)
    else:
        # No markers found, return empty list
        return []
    
    # Remove small objects
    labels = morphology.remove_small_objects(labels, min_size=min_object_size)
    
    # Extract properties
    region_properties = measure.regionprops(labels, intensity_image=image_channel)
    segmented_compartments = []
    
    for i, props in enumerate(region_properties):
        if props.area < min_object_size:
            continue
        
        # Extract contour
        mask = (labels == props.label)
        contours = measure.find_contours(mask.astype(float), 0.5)
        
        if len(contours) > 0:
            contour = contours[0]
            contour_um = contour * pixel_size
            
            # Calculate properties
            centroid_um = np.array(props.centroid) * pixel_size
            area_um2 = props.area * (pixel_size ** 2)
            bbox_um = np.array(props.bbox) * pixel_size
            
            compartment = {
                'id': f'enhanced_watershed_{i+1}',
                'method': 'enhanced_watershed',
                'centroid_um': centroid_um,
                'area_um2': area_um2,
                'bbox_um': bbox_um,
                'contour_um': contour_um.tolist(),
                'mean_intensity': props.intensity_mean,
                'max_intensity': props.intensity_max,
                'eccentricity': props.eccentricity,
                'solidity': props.solidity,
                'perimeter_um': props.perimeter * pixel_size
            }
            segmented_compartments.append(compartment)
    
    return segmented_compartments

def create_segmentation_summary(compartments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics for segmentation results.
    
    Parameters
    ----------
    compartments : List[Dict[str, Any]]
        List of segmented compartments
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics
    """
    if not compartments:
        return {
            'total_compartments': 0,
            'total_area_um2': 0,
            'mean_area_um2': 0,
            'median_area_um2': 0,
            'area_std_um2': 0,
            'methods_used': []
        }
    
    # Extract metrics
    areas = [comp['area_um2'] for comp in compartments]
    methods = [comp.get('method', 'unknown') for comp in compartments]
    
    # Calculate statistics
    total_area = sum(areas)
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    area_std = np.std(areas)
    
    # Count by method
    method_counts = {}
    for method in methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    summary = {
        'total_compartments': len(compartments),
        'total_area_um2': total_area,
        'mean_area_um2': mean_area,
        'median_area_um2': median_area,
        'area_std_um2': area_std,
        'min_area_um2': min(areas),
        'max_area_um2': max(areas),
        'methods_used': list(method_counts.keys()),
        'method_counts': method_counts
    }
    
    # Add additional statistics if available
    if any('mean_intensity' in comp for comp in compartments):
        intensities = [comp['mean_intensity'] for comp in compartments if 'mean_intensity' in comp]
        summary.update({
            'mean_intensity': np.mean(intensities),
            'median_intensity': np.median(intensities),
            'intensity_std': np.std(intensities)
        })
    
    if any('eccentricity' in comp for comp in compartments):
        eccentricities = [comp['eccentricity'] for comp in compartments if 'eccentricity' in comp]
        summary.update({
            'mean_eccentricity': np.mean(eccentricities),
            'median_eccentricity': np.median(eccentricities)
        })
    
    if any('solidity' in comp for comp in compartments):
        solidities = [comp['solidity'] for comp in compartments if 'solidity' in comp]
        summary.update({
            'mean_solidity': np.mean(solidities),
            'median_solidity': np.median(solidities)
        })
    
    return summary

def integrate_advanced_segmentation_with_app():
    """
    Integration function for advanced segmentation with Streamlit app.
    This function provides the UI for advanced segmentation methods.
    """
    import streamlit as st
    
    st.subheader("Advanced Segmentation Methods")
    st.info("Advanced segmentation tools for complex image analysis")
    
    # Check if mask images are available
    if 'mask_images' not in st.session_state or st.session_state.mask_images is None:
        st.warning("Upload mask images in the Data Loading tab first to use advanced segmentation.")
        return
    
    # Get the image data
    mask_image_data = st.session_state.mask_images
    
    # Handle different image formats
    if isinstance(mask_image_data, list) and len(mask_image_data) > 0:
        current_image = mask_image_data[0]
    else:
        current_image = mask_image_data
    
    # Ensure single channel
    if len(current_image.shape) == 3:
        if current_image.shape[2] > 1:
            st.info(f"Multichannel image detected. Using channel 1 of {current_image.shape[2]}")
            current_image = current_image[:, :, 0]
        else:
            current_image = current_image.squeeze()
    
    # Method selection
    segmentation_method = st.selectbox(
        "Advanced Segmentation Method",
        [
            "Watershed Segmentation",
            "Enhanced Watershed", 
            "Adaptive Threshold",
            "Multi-Scale Segmentation"
        ],
        help="Choose advanced segmentation algorithm"
    )
    
    # Get pixel size
    pixel_size = st.session_state.get('pixel_size', 0.1)
    
    # Method-specific parameters
    if segmentation_method == "Watershed Segmentation":
        st.markdown("#### Watershed Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_distance_peaks = st.slider(
                "Min Distance Between Peaks", 
                min_value=3, 
                max_value=20, 
                value=7,
                help="Minimum distance between watershed seeds"
            )
            
        with col2:
            min_object_size = st.slider(
                "Min Object Size (pixels)", 
                min_value=10, 
                max_value=500, 
                value=50,
                help="Minimum size of detected objects"
            )
        
        if st.button("Run Watershed Segmentation"):
            with st.spinner("Running watershed segmentation..."):
                try:
                    compartments = segment_image_channel_watershed(
                        current_image,
                        min_distance_peaks=min_distance_peaks,
                        min_object_size=min_object_size,
                        pixel_size=pixel_size
                    )
                    
                    if compartments:
                        st.success(f"✓ Detected {len(compartments)} compartments using watershed")
                        
                        # Store results
                        st.session_state.advanced_segmentation_results = compartments
                        
                        # Display summary
                        summary = create_segmentation_summary(compartments)
                        display_segmentation_results(compartments, summary, current_image)
                        
                    else:
                        st.warning("No compartments detected. Try adjusting parameters.")
                        
                except Exception as e:
                    st.error(f"Watershed segmentation failed: {str(e)}")
    
    elif segmentation_method == "Enhanced Watershed":
        st.markdown("#### Enhanced Watershed Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            gaussian_sigma = st.slider(
                "Gaussian Smoothing Sigma", 
                min_value=0.0, 
                max_value=5.0, 
                value=1.0, 
                step=0.1,
                help="Preprocessing smoothing strength"
            )
            
            min_distance_peaks = st.slider(
                "Min Distance Between Peaks", 
                min_value=3, 
                max_value=20, 
                value=7
            )
            
        with col2:
            min_object_size = st.slider(
                "Min Object Size (pixels)", 
                min_value=10, 
                max_value=500, 
                value=50
            )
            
            use_gradient = st.checkbox(
                "Use Gradient for Watershed", 
                value=False,
                help="Use gradient magnitude instead of distance transform"
            )
        
        if st.button("Run Enhanced Watershed"):
            with st.spinner("Running enhanced watershed segmentation..."):
                try:
                    compartments = enhanced_watershed_segmentation(
                        current_image,
                        gaussian_sigma=gaussian_sigma,
                        min_distance_peaks=min_distance_peaks,
                        min_object_size=min_object_size,
                        pixel_size=pixel_size,
                        use_gradient=use_gradient
                    )
                    
                    if compartments:
                        st.success(f"✓ Detected {len(compartments)} compartments using enhanced watershed")
                        
                        # Store results
                        st.session_state.advanced_segmentation_results = compartments
                        
                        # Display summary
                        summary = create_segmentation_summary(compartments)
                        display_segmentation_results(compartments, summary, current_image)
                        
                    else:
                        st.warning("No compartments detected. Try adjusting parameters.")
                        
                except Exception as e:
                    st.error(f"Enhanced watershed segmentation failed: {str(e)}")
    
    elif segmentation_method == "Adaptive Threshold":
        st.markdown("#### Adaptive Threshold Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            block_size = st.slider(
                "Block Size", 
                min_value=11, 
                max_value=101, 
                value=51, 
                step=2,
                help="Size of local neighborhood (must be odd)"
            )
            
        with col2:
            offset = st.slider(
                "Threshold Offset", 
                min_value=-0.1, 
                max_value=0.1, 
                value=0.01, 
                step=0.001,
                help="Constant subtracted from local mean"
            )
            
            min_object_size = st.slider(
                "Min Object Size (pixels)", 
                min_value=10, 
                max_value=500, 
                value=50
            )
        
        if st.button("Run Adaptive Threshold Segmentation"):
            with st.spinner("Running adaptive threshold segmentation..."):
                try:
                    compartments = adaptive_threshold_segmentation(
                        current_image,
                        block_size=block_size,
                        offset=offset,
                        min_object_size=min_object_size,
                        pixel_size=pixel_size
                    )
                    
                    if compartments:
                        st.success(f"✓ Detected {len(compartments)} compartments using adaptive threshold")
                        
                        # Store results
                        st.session_state.advanced_segmentation_results = compartments
                        
                        # Display summary
                        summary = create_segmentation_summary(compartments)
                        display_segmentation_results(compartments, summary, current_image)
                        
                    else:
                        st.warning("No compartments detected. Try adjusting parameters.")
                        
                except Exception as e:
                    st.error(f"Adaptive threshold segmentation failed: {str(e)}")
    
    elif segmentation_method == "Multi-Scale Segmentation":
        st.markdown("#### Multi-Scale Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            scale_min = st.slider("Min Scale (Sigma)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            scale_max = st.slider("Max Scale (Sigma)", min_value=2.0, max_value=8.0, value=4.0, step=0.5)
            num_scales = st.slider("Number of Scales", min_value=2, max_value=6, value=3)
            
        with col2:
            min_object_size = st.slider("Min Object Size (pixels)", min_value=10, max_value=500, value=50)
            merge_overlaps = st.checkbox("Merge Overlapping Objects", value=True)
            
            if merge_overlaps:
                overlap_threshold = st.slider(
                    "Overlap Threshold", 
                    min_value=0.1, 
                    max_value=0.9, 
                    value=0.5,
                    help="Minimum overlap to trigger merging"
                )
        
        if st.button("Run Multi-Scale Segmentation"):
            with st.spinner("Running multi-scale segmentation..."):
                try:
                    # Generate scale values
                    scales = np.linspace(scale_min, scale_max, num_scales)
                    
                    compartments = multi_scale_segmentation(
                        current_image,
                        scales=scales.tolist(),
                        min_object_size=min_object_size,
                        pixel_size=pixel_size
                    )
                    
                    # Merge overlapping compartments if requested
                    if merge_overlaps and compartments:
                        compartments = merge_overlapping_compartments(
                            compartments, 
                            overlap_threshold=overlap_threshold
                        )
                    
                    if compartments:
                        st.success(f"✓ Detected {len(compartments)} compartments using multi-scale method")
                        
                        # Store results
                        st.session_state.advanced_segmentation_results = compartments
                        
                        # Display summary
                        summary = create_segmentation_summary(compartments)
                        display_segmentation_results(compartments, summary, current_image)
                        
                    else:
                        st.warning("No compartments detected. Try adjusting parameters.")
                        
                except Exception as e:
                    st.error(f"Multi-scale segmentation failed: {str(e)}")

def display_segmentation_results(compartments, summary, original_image):
    """Display segmentation results in Streamlit interface."""
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Display summary statistics
    st.subheader("Segmentation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Compartments", summary['total_compartments'])
    with col2:
        st.metric("Total Area", f"{summary['total_area_um2']:.1f} μm²")
    with col3:
        st.metric("Mean Area", f"{summary['mean_area_um2']:.1f} μm²")
    with col4:
        st.metric("Area Std", f"{summary['area_std_um2']:.1f} μm²")
    
    # Visualization
    st.subheader("Segmentation Visualization")
    
    # Create overlay visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Overlay compartments
    ax2.imshow(original_image, cmap='gray', alpha=0.7)
    
    # Plot compartment contours and centroids
    colors = plt.cm.Set3(np.linspace(0, 1, len(compartments)))
    
    for i, comp in enumerate(compartments):
        if 'contour_um' in comp and comp['contour_um']:
            contour = np.array(comp['contour_um'])
            if len(contour) > 0:
                # Convert from micrometers back to pixels for display
                pixel_size = st.session_state.get('pixel_size', 0.1)
                contour_pixels = contour / pixel_size
                
                ax2.plot(contour_pixels[:, 1], contour_pixels[:, 0], 
                        color=colors[i], linewidth=2, alpha=0.8)
                
                # Plot centroid
                centroid_pixels = np.array(comp['centroid_um']) / pixel_size
                ax2.plot(centroid_pixels[1], centroid_pixels[0], 
                        'o', color=colors[i], markersize=8, markeredgecolor='white')
    
    ax2.set_title(f'Segmentation Results ({len(compartments)} compartments)')
    ax2.axis('off')
    
    st.pyplot(fig)
    plt.close()
    
    # Detailed statistics
    with st.expander("Detailed Statistics"):
        if summary['methods_used']:
            st.write("**Methods Used:**")
            for method, count in summary['method_counts'].items():
                st.write(f"  - {method}: {count} compartments")
        
        st.write(f"**Area Statistics (μm²):**")
        st.write(f"  - Min: {summary['min_area_um2']:.2f}")
        st.write(f"  - Max: {summary['max_area_um2']:.2f}")
        st.write(f"  - Median: {summary['median_area_um2']:.2f}")
        
        if 'mean_intensity' in summary:
            st.write(f"**Intensity Statistics:**")
            st.write(f"  - Mean: {summary['mean_intensity']:.2f}")
            st.write(f"  - Median: {summary['median_intensity']:.2f}")
            st.write(f"  - Std: {summary['intensity_std']:.2f}")
        
        if 'mean_eccentricity' in summary:
            st.write(f"**Shape Statistics:**")
            st.write(f"  - Mean Eccentricity: {summary['mean_eccentricity']:.3f}")
            st.write(f"  - Mean Solidity: {summary['mean_solidity']:.3f}")
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Compartment Data"):
            # Convert compartments to DataFrame
            export_data = []
            for comp in compartments:
                row = {
                    'id': comp['id'],
                    'method': comp['method'],
                    'centroid_x_um': comp['centroid_um'][1] if len(comp['centroid_um']) > 1 else 0,
                    'centroid_y_um': comp['centroid_um'][0] if len(comp['centroid_um']) > 0 else 0,
                    'area_um2': comp['area_um2'],
                    'mean_intensity': comp.get('mean_intensity', 0),
                    'max_intensity': comp.get('max_intensity', 0),
                    'eccentricity': comp.get('eccentricity', 0),
                    'solidity': comp.get('solidity', 0)
                }
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Compartment Data (CSV)",
                data=csv,
                file_name="advanced_segmentation_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Apply to Track Classification"):
            if 'tracks_data' in st.session_state and st.session_state.tracks_data is not None:
                try:
                    # Classify particles using the segmented compartments
                    pixel_size = st.session_state.get('pixel_size', 0.1)
                    
                    classified_tracks = classify_particles_by_contour(
                        st.session_state.tracks_data,
                        compartments,
                        pixel_size=pixel_size
                    )
                    
                    # Update session state
                    st.session_state.tracks_data = classified_tracks
                    
                    # Show classification summary
                    classification_summary = classified_tracks['compartment_id'].value_counts()
                    
                    st.success("✓ Tracks classified by compartments!")
                    st.write("**Classification Summary:**")
                    for comp_id, count in classification_summary.items():
                        st.write(f"  - {comp_id}: {count} particles")
                    
                except Exception as e:
                    st.error(f"Track classification failed: {str(e)}")
            else:
                st.warning("No track data available for classification.")