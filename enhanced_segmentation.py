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
    from matplotlib.path import Path
    
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
        path = Path(contour_um)
        
        # Test which points are inside this compartment
        particle_coords = np.column_stack((x_um, y_um))
        inside_mask = path.contains_points(particle_coords)
        
        # Update tracks that fall within this compartment and haven't been assigned
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
    
    # Normalize image
    image_norm = (image_channel - image_channel.min()) / (image_channel.max() - image_channel.min())
    
    # Adaptive thresholding
    adaptive_thresh = filters.threshold_local(image_norm, block_size=block_size, offset=offset)
    binary_mask = image_norm > adaptive_thresh
    
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
    
    from matplotlib.path import Path
    
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
            try:
                contour1 = np.array(comp1['contour_um'])
                contour2 = np.array(comp2['contour_um'])
                
                if len(contour1) < 3 or len(contour2) < 3:
                    continue
                
                path1 = Path(contour1)
                path2 = Path(contour2)
                
                # Simple overlap check: test if centroids are inside other compartments
                overlap_score = 0
                if path1.contains_point(comp2['centroid_um']):
                    overlap_score += 0.5
                if path2.contains_point(comp1['centroid_um']):
                    overlap_score += 0.5
                
                if overlap_score >= overlap_threshold:
                    merged_areas.append(comp2['area_um2'])
                    merged_indices.append(j)
                    used_indices.add(j)
                    
            except Exception:
                continue
        
        # Update merged compartment properties
        if len(merged_areas) > 1:
            merged_comp['area_um2'] = sum(merged_areas)
            merged_comp['id'] = f"merged_{'_'.join([str(idx) for idx in merged_indices])}"
            merged_comp['method'] = 'merged'
        
        merged_compartments.append(merged_comp)
        used_indices.add(i)
    
    return merged_compartments