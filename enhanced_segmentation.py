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
    coords = peak_local_max(distance, min_distance=min_distance_peaks, labels=binary_mask)
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
        coords = peak_local_max(distance, min_distance=min_distance_peaks, labels=binary_mask)
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