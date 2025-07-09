"""
Image segmentation module for compartment identification and boundary definition.
Provides tools for segmenting image channels to define compartment boundaries
for multi-channel single particle tracking analysis.

Enhanced with density map segmentation for nuclear density classification.
"""

import numpy as np
import pandas as pd
from skimage import measure, filters, morphology, draw
from skimage.filters import threshold_otsu, threshold_triangle, gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk, opening, binary_erosion, binary_dilation, label
from skimage.segmentation import find_boundaries
from skimage.draw import polygon
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import warnings
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation
from typing import List, Dict, Tuple, Any, Optional
import streamlit as st

def segment_image_channel_otsu(image_channel: np.ndarray, min_object_size: int = 50, 
                              pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Segments a 2D image channel using Otsu's thresholding to define compartments.

    Parameters
    ----------
    image_channel : np.ndarray
        2D numpy array representing the image channel.
    min_object_size : int
        Minimum size (in pixels) for an object to be considered a compartment.
    pixel_size : float
        Physical size of a pixel (e.g., in micrometers), used for reporting region properties.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a segmented compartment
        and contains keys like 'id', 'contour_pixels', 'contour_um', 'bbox_pixels', 
        'bbox_um', 'centroid_pixels', 'centroid_um', 'area_pixels', 'area_um2'.
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")

    try:
        thresh = threshold_otsu(image_channel)
        binary_mask = image_channel > thresh
    except ValueError:        
        # Otsu fails on flat images
        return []

    # Clean up mask
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_object_size)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=min_object_size)
    
    # Label regions
    labeled_regions = measure.label(cleaned_mask)
    region_properties = measure.regionprops(labeled_regions, intensity_image=image_channel)

    segmented_compartments = []
    for i, props in enumerate(region_properties):
        # Extract contour
        mask = (labeled_regions == props.label)
        contours = measure.find_contours(mask.astype(float), 0.5)
        
        if len(contours) > 0:
            contour = contours[0]  # Take the largest contour
            contour_um = contour * pixel_size
            
            # Calculate properties
            centroid_um = np.array(props.centroid) * pixel_size
            area_um2 = props.area * (pixel_size ** 2)
            bbox_um = np.array(props.bbox) * pixel_size
            
            compartment = {
                'id': f'otsu_{i+1}',
                'method': 'otsu',
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

def segment_image_channel_simple_threshold(image_channel: np.ndarray, threshold_value: float, 
                                         min_object_size: int = 50, pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Segments a 2D image channel using a simple global threshold.
    (Identical return structure to segment_image_channel_otsu)
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")

    binary_mask = image_channel > threshold_value
    
    # Clean up mask
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_object_size)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=min_object_size)
    
    labeled_regions = measure.label(cleaned_mask)
    region_properties = measure.regionprops(labeled_regions, intensity_image=image_channel)

    segmented_compartments = []
    for i, props in enumerate(region_properties):
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
                'id': f'threshold_{i+1}',
                'method': 'simple_threshold',
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

def segment_image_channel_adaptive_threshold(image_channel: np.ndarray, block_size: int = 51, 
                                           offset: float = 0, min_object_size: int = 50, 
                                           pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Segments a 2D image channel using adaptive thresholding for uneven illumination.
    """
    if image_channel.ndim != 2:
        raise ValueError("Image channel for segmentation must be 2D.")

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    try:
        binary_mask = filters.threshold_local(image_channel, block_size=block_size, offset=offset)
        binary_mask = image_channel > binary_mask
    except Exception:
        # Fallback to Otsu if adaptive fails
        return segment_image_channel_otsu(image_channel, min_object_size, pixel_size)
    
    # Clean up mask
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_object_size)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=min_object_size)
    
    labeled_regions = measure.label(cleaned_mask)
    region_properties = measure.regionprops(labeled_regions, intensity_image=image_channel)

    segmented_compartments = []
    for i, props in enumerate(region_properties):
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
                'id': f'compartment_{i}',
                'method': 'adaptive_threshold',
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

def convert_compartments_to_boundary_crossing_format(segmented_compartments: List[Dict[str, Any]], 
                                                   pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Converts segmented compartment data to a list of boundaries suitable for analyze_boundary_crossing.
    """
    formatted_boundaries = []
    for comp in segmented_compartments:
        comp_id = comp['id']
        bbox = comp['bbox_um']  # Use um coordinates
        
        # Create rectangular boundaries from bounding box
        formatted_boundaries.append({
            'id': f'{comp_id}_top', 
            'type': 'line', 
            'orientation': 'horizontal', 
            'y': bbox['y1'], 
            'x_min': bbox['x1'], 
            'x_max': bbox['x2']
        })
        formatted_boundaries.append({
            'id': f'{comp_id}_bottom', 
            'type': 'line', 
            'orientation': 'horizontal', 
            'y': bbox['y2'], 
            'x_min': bbox['x1'], 
            'x_max': bbox['x2']
        })
        formatted_boundaries.append({
            'id': f'{comp_id}_left', 
            'type': 'line', 
            'orientation': 'vertical', 
            'x': bbox['x1'], 
            'y_min': bbox['y1'], 
            'y_max': bbox['y2']
        })
        formatted_boundaries.append({
            'id': f'{comp_id}_right', 
            'type': 'line', 
            'orientation': 'vertical', 
            'x': bbox['x2'], 
            'y_min': bbox['y1'], 
            'y_max': bbox['y2']
        })
    return formatted_boundaries

def convert_compartments_to_dwell_time_regions(segmented_compartments: List[Dict[str, Any]], 
                                             pixel_size: float = 1.0) -> List[Dict[str, Any]]:
    """
    Converts segmented compartment data to regions suitable for analyze_dwell_time.
    Uses centroid and an effective radius derived from area.
    """
    regions = []
    for comp in segmented_compartments:
        centroid_um = comp['centroid_um']
        area_um2 = comp['area_um2']
        
        # Effective radius from area (Area = pi * r^2)
        effective_radius = np.sqrt(area_um2 / np.pi) if area_um2 > 0 else pixel_size
        
        regions.append({
            'id': comp['id'],
            'x': centroid_um['x'], 
            'y': centroid_um['y'], 
            'radius': effective_radius
        })
    return regions

def classify_particles_by_compartment(tracks_df: pd.DataFrame, 
                                    segmented_compartments: List[Dict[str, Any]], 
                                    pixel_size: float = 1.0) -> pd.DataFrame:
    """
    Classify particles/tracks based on which compartment they belong to.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with x, y coordinates
    segmented_compartments : List[Dict[str, Any]]
        List of segmented compartments
    pixel_size : float
        Pixel size for coordinate conversion
        
    Returns
    -------
    pd.DataFrame
        Enhanced tracks DataFrame with compartment classification
    """
    tracks_enhanced = tracks_df.copy()
    tracks_enhanced['compartment_id'] = 'none'
    tracks_enhanced['in_compartment'] = False
    
    # Convert track coordinates to micrometers if needed
    x_um = tracks_df['x'] * pixel_size
    y_um = tracks_df['y'] * pixel_size
    
    for comp in segmented_compartments:
        bbox = comp['bbox_um']
        
        # Simple bounding box classification
        in_compartment = (
            (x_um >= bbox['x1']) & (x_um <= bbox['x2']) &
            (y_um >= bbox['y1']) & (y_um <= bbox['y2'])
        )
        
        # Update tracks that fall within this compartment
        mask = in_compartment & (tracks_enhanced['compartment_id'] == 'none')
        tracks_enhanced.loc[mask, 'compartment_id'] = comp['id']
        tracks_enhanced.loc[mask, 'in_compartment'] = True
    
    return tracks_enhanced

def analyze_compartment_statistics(segmented_compartments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for segmented compartments.
    """
    if not segmented_compartments:
        return {
            'total_compartments': 0,
            'total_area_um2': 0,
            'mean_area_um2': 0,
            'median_area_um2': 0,
            'mean_perimeter_um': 0,
            'mean_solidity': 0,
            'mean_eccentricity': 0
        }
    
    areas = [comp['area_um2'] for comp in segmented_compartments]
    perimeters = [comp['perimeter_um'] for comp in segmented_compartments]
    solidities = [comp['solidity'] for comp in segmented_compartments]
    eccentricities = [comp['eccentricity'] for comp in segmented_compartments]
    
    return {
        'total_compartments': len(segmented_compartments),
        'total_area_um2': sum(areas),
        'mean_area_um2': np.mean(areas),
        'median_area_um2': np.median(areas),
        'std_area_um2': np.std(areas),
        'min_area_um2': min(areas),
        'max_area_um2': max(areas),
        'mean_perimeter_um': np.mean(perimeters),
        'mean_solidity': np.mean(solidities),
        'mean_eccentricity': np.mean(eccentricities)
    }


def density_map_threshold(
    image: np.ndarray,
    gaussian_sigma_hp: float = 2,
    gaussian_sigma_density: float = None,
    disk_radius: int = 10,
    pcutoff_in: float = 0.10,
    pcutoff_out: float = 0.10,
    roi_mask: np.ndarray = None,
    num_classes: int = None,
    binning: str = 'equal'
) -> Dict[str, Any]:
    """
    Density Map Mask strategy for nuclear segmentation and classification:
      1) Gaussian smoothing for mask (hp)
      2) Background subtraction on hp
      3) Normalize hp to [0,1]
      4) Global threshold in/out on hp
      5) Gaussian smoothing for density classes (optional separate sigma)
      6) Build density classes only within ROI
      7) Set everything outside nucleus to class 0

    Parameters
    ----------
    image : np.ndarray
        Input image for density mapping
    gaussian_sigma_hp : float
        Gaussian sigma for high-pass filtering
    gaussian_sigma_density : float, optional
        Separate sigma for density classification (if None, uses hp sigma)
    disk_radius : int
        Disk radius for morphological opening (background subtraction)
    pcutoff_in : float
        Percentile cutoff for inside nucleus threshold
    pcutoff_out : float
        Percentile cutoff for outside nucleus threshold
    roi_mask : np.ndarray, optional
        ROI mask to restrict classification area
    num_classes : int, optional
        Number of density classes to create
    binning : str
        Binning method ('equal' or 'quantile')

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys 'hp', 'mask_in', 'mask_out', and optional 'classes'
    """
    # if no separate density sigma specified, use hp sigma
    if gaussian_sigma_density is None:
        gaussian_sigma_density = gaussian_sigma_hp

    # 1) Build hp image for mask_in/mask_out
    smooth_hp = gaussian(image, sigma=gaussian_sigma_hp)
    background = morphology.opening(smooth_hp, morphology.disk(disk_radius))
    hp = smooth_hp - background

    # 2) Normalize hp to [0,1]
    maxv = hp.max() if hp.max() != 0 else 1.0
    hp = hp / maxv

    # 3) Global in/out thresholds on full-field hp
    mask_in = hp > pcutoff_in
    mask_out = hp < pcutoff_out

    result = {'hp': hp, 'mask_in': mask_in, 'mask_out': mask_out}

    # 4) Density classes (only within ROI), now with background subtraction
    if num_classes is not None and num_classes > 1:
        # Build density image with background subtraction
        smooth_density = gaussian(image, sigma=gaussian_sigma_density)
        background_density = morphology.opening(smooth_density, morphology.disk(disk_radius))
        density_image = smooth_density - background_density
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            density_image = density_image * roi_mask
        else:
            # Use mask_in as ROI
            density_image = density_image * mask_in
        
        # Get valid density values
        valid_mask = density_image > 0
        valid_densities = density_image[valid_mask]
        
        if len(valid_densities) > 0:
            # Create class boundaries
            if binning == 'equal':
                # Equal width bins
                min_val, max_val = valid_densities.min(), valid_densities.max()
                class_boundaries = np.linspace(min_val, max_val, num_classes + 1)
            else:  # quantile
                # Equal count bins
                class_boundaries = np.percentile(valid_densities, 
                                               np.linspace(0, 100, num_classes + 1))
            
            # Assign classes
            classes = np.zeros_like(image, dtype=int)
            for i in range(num_classes):
                lower = class_boundaries[i]
                upper = class_boundaries[i + 1]
                
                if i == num_classes - 1:  # Last class includes upper boundary
                    class_mask = (density_image >= lower) & (density_image <= upper)
                else:
                    class_mask = (density_image >= lower) & (density_image < upper)
                
                classes[class_mask] = i + 1
            
            result['classes'] = classes
            result['class_boundaries'] = class_boundaries

    return result


def smooth_binary_boundary(binary_mask: np.ndarray, smoothing_iterations: int = 2) -> np.ndarray:
    """
    Smooth the boundary of a binary mask using morphological operations.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask to smooth
    smoothing_iterations : int
        Number of erosion-dilation cycles for smoothing
        
    Returns
    -------
    np.ndarray
        Smoothed binary mask
    """
    smoothed = binary_mask.copy()
    
    for _ in range(smoothing_iterations):
        # Erosion followed by dilation to smooth boundaries
        smoothed = binary_erosion(smoothed, disk(1))
        smoothed = binary_dilation(smoothed, disk(1))
    
    # Fill holes that may have been created
    smoothed = binary_fill_holes(smoothed)
    
    return smoothed

def select_largest_object(binary_mask: np.ndarray) -> np.ndarray:
    """
    Select only the largest connected component from a binary mask.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask potentially containing multiple objects
        
    Returns
    -------
    np.ndarray
        Binary mask with only the largest object
    """
    # Label connected components
    labeled = label(binary_mask)
    
    if labeled.max() == 0:
        return binary_mask
    
    # Find properties of all regions
    regions = regionprops(labeled)
    
    if not regions:
        return binary_mask
    
    # Find the largest region by area
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create mask with only the largest object
    largest_mask = (labeled == largest_region.label)
    
    return largest_mask

def enhanced_threshold_image(
    image: np.ndarray,
    method: str = 'otsu',
    manual_threshold: float = None,
    min_size: int = 100,
    closing_disk_size: int = 3,
    smooth_boundary: bool = True,
    smoothing_iterations: int = 2,
    largest_object_only: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhanced threshold-based segmentation with density mapping support:
      - 'otsu', 'triangle', 'manual'
      - 'density_map'
    Returns mask and info dict
    """
    info = {}
    
    if method == 'otsu':
        threshold_val = threshold_otsu(image)
        mask = image > threshold_val
        info['threshold'] = threshold_val

    elif method == 'triangle':
        threshold_val = threshold_triangle(image)
        mask = image > threshold_val
        info['threshold'] = threshold_val

    elif method == 'manual':
        if manual_threshold is None:
            raise ValueError("manual_threshold must be provided for manual method")
        mask = image > manual_threshold
        info['threshold'] = manual_threshold

    elif method == 'density_map':
        density_params = {
            'gaussian_sigma_hp': kwargs.get('gaussian_sigma_hp', 2.0),
            'disk_radius': kwargs.get('disk_radius', 15),
            'pcutoff_in': kwargs.get('pcutoff_in', 0.15),
            'pcutoff_out': kwargs.get('pcutoff_out', 0.05)
        }
        
        density_result = density_map_threshold(image, **density_params)
        mask = density_result['mask_in']
        info.update(density_result)

    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Post-process all methods: close, remove small, fill holes
    mask = binary_closing(mask, disk(closing_disk_size))
    mask = remove_small_objects(mask, min_size=min_size)
    # fill any internal holes completely
    mask = binary_fill_holes(mask)
    
    # Apply largest object selection if requested
    if largest_object_only:
        mask = select_largest_object(mask)
    else:
        pass  # Keep all objects
    
    # Apply boundary smoothing if requested
    if smooth_boundary:
        mask = smooth_binary_boundary(mask, smoothing_iterations)

    return mask, info


def analyze_density_segmentation(mask: np.ndarray, classes: np.ndarray = None) -> Dict[str, Any]:
    """Analyze density-based segmentation mask and class properties"""
    lbl = measure.label(mask)
    props = measure.regionprops(lbl)
    sizes = [p.area for p in props]
    count = len(sizes)
    total = int(sum(sizes))
    mean = float(total / count) if count else 0.0
    
    result = {
        'num_objects': count,
        'total_area': total,
        'mean_object_size': mean,
        'object_sizes': sizes
    }
    
    if classes is not None:
        # Analyze class distribution
        unique_classes, class_counts = np.unique(classes[classes > 0], return_counts=True)
        class_distribution = dict(zip(unique_classes.astype(int), class_counts))
        
        result['class_distribution'] = class_distribution
        result['total_nuclear_pixels'] = int(np.sum(classes > 0))
        result['background_pixels'] = int(np.sum(classes == 0))
        
    return result


def visualize_density_segmentation(image: np.ndarray, mask: np.ndarray, classes: np.ndarray = None) -> np.ndarray:
    """Create RGB overlay visualization of density segmentation"""
    # Convert image to uint8 for RGB overlay
    if image.dtype != np.uint8:
        mi, ma = image.min(), image.max()
        img8 = ((image - mi) / (ma - mi) * 255).astype(np.uint8) if ma > mi else np.zeros_like(image, dtype=np.uint8)
    else:
        img8 = image
    
    rgb = np.stack([img8]*3, axis=-1)
    
    if classes is not None:
        # Color-code different density classes
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Create colormap for classes
        n_classes = int(classes.max()) if classes.max() > 0 else 1
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes + 1))
        
        for class_id in range(1, n_classes + 1):
            class_mask = classes == class_id
            if np.any(class_mask):
                color = (colors[class_id] * 255).astype(np.uint8)[:3]
                rgb[class_mask] = color
    else:
        # Just show boundaries for basic mask
        boundaries = find_boundaries(mask)
        rgb[boundaries] = [255, 0, 0]
    
    return rgb


def _fit_gmm_model(args):
    """Helper function for parallel GMM fitting"""
    n_components, X, random_state, covariance_type, n_init = args
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type=covariance_type,
            n_init=n_init
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        return n_components, gmm, bic, aic
    except Exception:
        return n_components, None, np.inf, np.inf

def gaussian_mixture_segmentation(
    image: np.ndarray,
    roi_mask: np.ndarray = None,
    max_components: int = 10,
    min_components: int = 2,
    criterion: str = 'bic',
    random_state: int = 42,
    covariance_type: str = 'full',
    n_init: int = 10
) -> Dict[str, Any]:
    """
    Automatic nuclear density segmentation using Gaussian Mixture Models.
    
    Uses model selection criteria (BIC/AIC) to automatically determine the optimal
    number of density classes based on the pixel intensity distribution.
    
    Parameters
    ----------
    image : np.ndarray
        Input image for density analysis
    roi_mask : np.ndarray, optional
        ROI mask to restrict analysis area (e.g., nuclear boundary)
    max_components : int
        Maximum number of GMM components to test
    min_components : int
        Minimum number of GMM components to test
    criterion : str
        Model selection criterion ('bic', 'aic', or 'both')
    random_state : int
        Random state for reproducibility
    covariance_type : str
        GMM covariance type ('full', 'tied', 'diag', 'spherical')
    n_init : int
        Number of initializations for GMM fitting
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with segmentation results, optimal model, and statistics
    """
    
    # Extract pixel intensities from ROI
    if roi_mask is not None:
        roi_pixels = image[roi_mask > 0]
    else:
        roi_pixels = image.flatten()
    
    # Remove any invalid values
    roi_pixels = roi_pixels[np.isfinite(roi_pixels)]
    
    if len(roi_pixels) < min_components * 10:
        raise ValueError(f"Insufficient pixels ({len(roi_pixels)}) for GMM analysis")
    
    # Reshape for sklearn
    X = roi_pixels.reshape(-1, 1)
    
    # Test different numbers of components
    n_components_range = range(min_components, max_components + 1)
    models = {}
    scores = {'bic': [], 'aic': [], 'n_components': list(n_components_range)}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Prepare arguments for parallel processing
        args_list = [
            (n, X, random_state, covariance_type, n_init)
            for n in n_components_range
        ]
        
        # Use ThreadPoolExecutor for parallel fitting
        max_workers = min(multiprocessing.cpu_count(), len(args_list))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_fit_gmm_model, args_list))
        
        # Process results
        models = {}
        for n_components, gmm, bic_score, aic_score in results:
            models[n_components] = gmm
            scores['bic'].append(bic_score)
            scores['aic'].append(aic_score)
    
    # Select optimal model based on criterion
    if criterion == 'bic':
        optimal_idx = np.argmin(scores['bic'])
        optimal_score = scores['bic'][optimal_idx]
    elif criterion == 'aic':
        optimal_idx = np.argmin(scores['aic'])
        optimal_score = scores['aic'][optimal_idx]
    else:  # both - use BIC as primary
        optimal_idx = np.argmin(scores['bic'])
        optimal_score = scores['bic'][optimal_idx]
    
    optimal_n_components = scores['n_components'][optimal_idx]
    optimal_model = models[optimal_n_components]
    
    if optimal_model is None:
        raise ValueError("Could not fit any valid GMM models")
    
    # Generate class assignments for the full image
    classes = np.zeros_like(image, dtype=np.int32)
    
    if roi_mask is not None:
        # Predict classes only within ROI
        roi_classes = optimal_model.predict(X) + 1  # Start from 1 (0 = background)
        classes[roi_mask > 0] = roi_classes
    else:
        # Predict for entire image
        all_pixels = image.flatten().reshape(-1, 1)
        all_classes = optimal_model.predict(all_pixels) + 1
        classes = all_classes.reshape(image.shape)
    
    # Calculate component statistics
    component_stats = []
    for i in range(optimal_n_components):
        component_pixels = roi_pixels[optimal_model.predict(X) == i]
        stats = {
            'component': i + 1,
            'mean_intensity': float(optimal_model.means_[i, 0]),
            'std_intensity': float(np.sqrt(optimal_model.covariances_[i, 0, 0])),
            'weight': float(optimal_model.weights_[i]),
            'pixel_count': len(component_pixels),
            'intensity_range': (float(component_pixels.min()), float(component_pixels.max())) if len(component_pixels) > 0 else (0.0, 0.0)
        }
        component_stats.append(stats)
    
    # Sort components by mean intensity for consistent ordering
    component_stats.sort(key=lambda x: x['mean_intensity'])
    
    # Reassign class labels based on intensity order (brightest = highest class number)
    sorted_indices = np.argsort([s['mean_intensity'] for s in component_stats])
    
    # Create mapping from original component indices to brightness-ordered indices
    component_to_brightness_order = {}
    for new_idx, old_component_idx in enumerate(sorted_indices):
        # Find the original GMM component index for this sorted component
        original_component = component_stats[old_component_idx]['component'] - 1
        component_to_brightness_order[original_component] = new_idx + 1
    
    # Remap classes based on brightness order, keeping background as 0
    classes_remapped = np.zeros_like(classes)
    if roi_mask is not None:
        # Only pixels within ROI get classified, rest remain 0 (background)
        roi_predictions = optimal_model.predict(X)
        for i, prediction in enumerate(roi_predictions):
            # Map prediction to brightness-ordered class
            brightness_class = component_to_brightness_order.get(prediction, 0)
            roi_indices = np.where(roi_mask.flatten())[0]
            if i < len(roi_indices):
                row, col = np.unravel_index(roi_indices[i], roi_mask.shape)
                classes_remapped[row, col] = brightness_class
    else:
        # For whole image classification
        all_predictions = optimal_model.predict(image.flatten().reshape(-1, 1))
        brightness_classes = np.array([component_to_brightness_order.get(pred, 0) for pred in all_predictions])
        classes_remapped = brightness_classes.reshape(image.shape)
    
    # Update component stats with brightness-ordered indices
    for i, stat in enumerate(component_stats):
        stat['component'] = i + 1  # Reorder component numbers by brightness
    
    # Create model info structure for display
    model_info = {
        'n_components': optimal_n_components,
        'criterion_used': criterion,
        'optimal_score': optimal_score
    }
    
    if criterion == 'bic':
        model_info['bic'] = optimal_score
        if len(scores['aic']) > optimal_idx:
            model_info['aic'] = scores['aic'][optimal_idx]
    elif criterion == 'aic':
        model_info['aic'] = optimal_score
        if len(scores['bic']) > optimal_idx:
            model_info['bic'] = scores['bic'][optimal_idx]
    else:  # both
        model_info['bic'] = optimal_score
        if len(scores['aic']) > optimal_idx:
            model_info['aic'] = scores['aic'][optimal_idx]
    
    return {
        'classes': classes_remapped,
        'optimal_n_components': optimal_n_components,
        'optimal_model': optimal_model,
        'component_stats': component_stats,
        'model_scores': scores,
        'model_info': model_info,
        'criterion_used': criterion,
        'optimal_score': optimal_score,
        'roi_pixel_count': len(roi_pixels),
        'total_pixels_classified': np.sum(classes_remapped > 0),
        'roi_pixels': roi_pixels,  # Include for histogram plotting
        'has_background': roi_mask is not None  # Background exists when using ROI mask
    }


def _fit_bayesian_gmm_model(args):
    """Helper function for parallel Bayesian GMM fitting"""
    max_components, X, weight_concentration_prior, random_state, covariance_type = args
    try:
        bgmm = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior=weight_concentration_prior,
            random_state=random_state,
            covariance_type=covariance_type,
            max_iter=200
        )
        bgmm.fit(X)
        lower_bound = bgmm.lower_bound_
        return bgmm, lower_bound
    except Exception:
        return None, -np.inf

def bayesian_gaussian_mixture_segmentation(
    image: np.ndarray,
    roi_mask: np.ndarray = None,
    max_components: int = 10,
    weight_concentration_prior: float = 1e-3,
    random_state: int = 42,
    covariance_type: str = 'full',
    criterion: str = 'lower_bound'
) -> Dict[str, Any]:
    """
    Automatic nuclear density segmentation using Bayesian Gaussian Mixture Models.
    
    Uses Dirichlet process GMM with optional BIC/AIC model selection for
    easier comparison with standard GMM methods.
    
    Parameters
    ----------
    image : np.ndarray
        Input image for density analysis
    roi_mask : np.ndarray, optional
        ROI mask to restrict analysis area
    max_components : int
        Maximum number of components to test
    weight_concentration_prior : float
        Controls sparsity - smaller values favor fewer components
    random_state : int
        Random state for reproducibility
    covariance_type : str
        GMM covariance type
    criterion : str
        Model selection criterion ('lower_bound', 'bic', 'aic', or 'both')
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with segmentation results and component analysis
    """
    
    # Extract pixel intensities from ROI
    if roi_mask is not None:
        roi_pixels = image[roi_mask > 0]
    else:
        roi_pixels = image.flatten()
    
    # Remove any invalid values
    roi_pixels = roi_pixels[np.isfinite(roi_pixels)]
    
    if len(roi_pixels) < 20:
        raise ValueError(f"Insufficient pixels ({len(roi_pixels)}) for Bayesian GMM analysis")
    
    # Reshape for sklearn
    X = roi_pixels.reshape(-1, 1)
    
    # Fit Bayesian GMM with model selection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Always use traditional Bayesian approach with automatic component pruning
        bgmm = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior=weight_concentration_prior,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=200,
            n_init=5
        )
        
        bgmm.fit(X)
        criterion_used = 'Automatic Pruning'
    
    # Identify effective components (those with non-negligible weights)
    weight_threshold = 1.0 / max_components / 10  # 10x smaller than uniform
    effective_components = np.where(bgmm.weights_ > weight_threshold)[0]
    n_effective = len(effective_components)
    
    # Generate class assignments with proper mapping
    classes = np.zeros_like(image, dtype=np.int32)
    
    if roi_mask is not None:
        roi_classes = bgmm.predict(X)
        # Map to effective components only
        class_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(effective_components)}
        roi_classes_mapped = np.array([class_map.get(cls, 0) for cls in roi_classes])
        classes[roi_mask > 0] = roi_classes_mapped
    else:
        # For full image processing
        all_pixels = image.flatten().reshape(-1, 1)
        all_classes = bgmm.predict(all_pixels)
        class_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(effective_components)}
        
        all_classes_mapped = np.array([class_map.get(cls, 0) for cls in all_classes])
        classes = all_classes_mapped.reshape(image.shape)
    
    # Calculate component statistics for effective components
    component_stats = []
    for new_idx, old_idx in enumerate(effective_components):
        component_pixels = roi_pixels[bgmm.predict(X) == old_idx]
        if len(component_pixels) > 0:
            stats = {
                'component': new_idx + 1,
                'mean_intensity': float(bgmm.means_[old_idx, 0]),
                'std_intensity': float(np.sqrt(bgmm.covariances_[old_idx, 0, 0])),
                'weight': float(bgmm.weights_[old_idx]),
                'pixel_count': len(component_pixels),
                'intensity_range': (float(component_pixels.min()), float(component_pixels.max()))
            }
            component_stats.append(stats)
    
    # Create mapping from original component indices to brightness-ordered indices BEFORE sorting
    original_component_order = [(i, stat['mean_intensity']) for i, stat in enumerate(component_stats)]
    original_component_order.sort(key=lambda x: x[1])  # Sort by mean intensity
    
    # Create mapping: original_component_index -> brightness_ordered_class
    component_to_brightness_order = {}
    for new_class, (original_idx, _) in enumerate(original_component_order):
        bgmm_component_idx = effective_components[original_idx]
        component_to_brightness_order[bgmm_component_idx] = new_class + 1
    
    # Remap classes based on brightness order, keeping background as 0
    classes_remapped = np.zeros_like(classes)
    if roi_mask is not None:
        roi_predictions = bgmm.predict(X)
        roi_indices = np.where(roi_mask.flatten())[0]
        
        for i, prediction in enumerate(roi_predictions):
            if i < len(roi_indices):
                brightness_class = component_to_brightness_order.get(prediction, prediction + 1)
                row, col = np.unravel_index(roi_indices[i], roi_mask.shape)
                classes_remapped[row, col] = brightness_class
    else:
        all_predictions = bgmm.predict(image.flatten().reshape(-1, 1))
        for i, prediction in enumerate(all_predictions):
            brightness_class = component_to_brightness_order.get(prediction, prediction + 1)
            row, col = np.unravel_index(i, image.shape)
            classes_remapped[row, col] = brightness_class
    
    # Sort component stats by mean intensity and update component numbers
    component_stats.sort(key=lambda x: x['mean_intensity'])
    for i, stat in enumerate(component_stats):
        stat['component'] = i + 1  # Reorder component numbers by brightness
    
    # Calculate BIC and log likelihood for comparison with other methods
    try:
        log_likelihood = bgmm.score(X) * len(X)
        
        # Calculate number of parameters for effective components
        # For 1D data: n_effective means + n_effective variances + (n_effective-1) weights
        n_params = n_effective + n_effective + (n_effective - 1) if n_effective > 1 else 2
        n_samples = len(X)
        
        # Calculate BIC and AIC for comparison
        bic_score = -2 * log_likelihood + n_params * np.log(n_samples)
        aic_score = -2 * log_likelihood + 2 * n_params
        
    except Exception:
        log_likelihood = float(bgmm.lower_bound_)
        bic_score = np.inf
        aic_score = np.inf
    
    # Create model info structure for display
    model_info = {
        'n_components': n_effective,
        'n_components_used': n_effective,
        'total_components_tested': max_components,
        'lower_bound': float(bgmm.lower_bound_),
        'log_likelihood': log_likelihood,
        'bic': bic_score,
        'aic': aic_score,
        'weight_threshold': weight_threshold,
        'method': 'Bayesian GMM',
        'criterion': criterion_used
    }
    
    return {
        'classes': classes_remapped,
        'optimal_n_components': n_effective,
        'optimal_model': bgmm,
        'n_effective_components': n_effective,
        'total_components_tested': max_components,
        'effective_component_indices': effective_components.tolist(),
        'component_stats': component_stats,
        'model_info': model_info,
        'model': bgmm,
        'weight_threshold': weight_threshold,
        'roi_pixel_count': len(roi_pixels),
        'total_pixels_classified': np.sum(classes_remapped > 0),
        'roi_pixels': roi_pixels,
        'has_background': roi_mask is not None
    }


def compare_segmentation_methods(
    image: np.ndarray,
    roi_mask: np.ndarray = None,
    max_components: int = 8
) -> Dict[str, Any]:
    """
    Compare different segmentation methods on the same image/ROI.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    roi_mask : np.ndarray, optional
        ROI mask
    max_components : int
        Maximum components for testing
        
    Returns
    -------
    Dict[str, Any]
        Comparison results from different methods
    """
    
    results = {}
    
    try:
        # Standard GMM with BIC selection
        gmm_result = gaussian_mixture_segmentation(
            image, roi_mask, max_components=max_components, criterion='bic'
        )
        results['gmm_bic'] = gmm_result
    except Exception as e:
        results['gmm_bic'] = {'error': str(e)}
    
    try:
        # Bayesian GMM
        bgmm_result = bayesian_gaussian_mixture_segmentation(
            image, roi_mask, max_components=max_components
        )
        results['bayesian_gmm'] = bgmm_result
    except Exception as e:
        results['bayesian_gmm'] = {'error': str(e)}
    
    try:
        # Equal binning (current method) for comparison
        if roi_mask is not None:
            roi_pixels = image[roi_mask > 0]
        else:
            roi_pixels = image.flatten()
        
        # Use median number of components from other methods as default
        n_comp_estimates = []
        if 'gmm_bic' in results and 'optimal_n_components' in results['gmm_bic']:
            n_comp_estimates.append(results['gmm_bic']['optimal_n_components'])
        if 'bayesian_gmm' in results and 'n_effective_components' in results['bayesian_gmm']:
            n_comp_estimates.append(results['bayesian_gmm']['n_effective_components'])
        
        n_classes = int(np.median(n_comp_estimates)) if n_comp_estimates else 5
        
        # Equal intensity binning
        min_val, max_val = roi_pixels.min(), roi_pixels.max()
        class_edges = np.linspace(min_val, max_val, n_classes + 1)
        
        equal_classes = np.zeros_like(image, dtype=np.int32)
        if roi_mask is not None:
            roi_digitized = np.digitize(roi_pixels, class_edges, right=False)
            roi_digitized = np.clip(roi_digitized, 1, n_classes)  # Ensure valid range
            equal_classes[roi_mask > 0] = roi_digitized
        else:
            all_digitized = np.digitize(image.flatten(), class_edges, right=False)
            all_digitized = np.clip(all_digitized, 1, n_classes)
            equal_classes = all_digitized.reshape(image.shape)
        
        results['equal_binning'] = {
            'classes': equal_classes,
            'n_classes': n_classes,
            'class_edges': class_edges.tolist(),
            'method': 'equal_intensity_binning'
        }
        
    except Exception as e:
        results['equal_binning'] = {'error': str(e)}
    
    return results


def apply_nuclear_segmentation_with_preprocessing(image: np.ndarray, 
                                                preprocessing_params: Dict[str, Any],
                                                nuclear_params: Dict[str, Any], 
                                                internal_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply two-step nuclear segmentation with preprocessing: preprocess original image, 
    then nuclear boundary detection + internal segmentation on processed image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    preprocessing_params : dict
        Parameters for image preprocessing
    nuclear_params : dict
        Parameters for nuclear boundary detection
    internal_params : dict
        Parameters for internal segmentation
        
    Returns
    -------
    tuple
        (nuclear_mask, internal_classes, combined_result)
    """
    # Step 0: Apply preprocessing to original image if requested
    processed_image = image.copy()
    
    if preprocessing_params.get('apply_preprocessing', False):
        method = preprocessing_params.get('method', 'Median Filter')
        
        if method == "Median Filter":
            filter_size = preprocessing_params.get('size', 5)
            from scipy.ndimage import median_filter
            processed_image = median_filter(processed_image, size=filter_size)
        elif method == "Gaussian Filter":
            sigma = preprocessing_params.get('sigma', 1.0)
            from skimage.filters import gaussian
            processed_image = gaussian(processed_image, sigma=sigma)
    
    # Step 1: Detect nuclear boundaries on processed image
    nuclear_mask = segment_nuclear_boundary(processed_image, nuclear_params)
    
    # Step 2: Segment within nuclear regions on processed image
    internal_classes = segment_within_nucleus(processed_image, nuclear_mask, internal_params)
    
    # Create combined result
    combined_result = np.zeros_like(image, dtype=np.uint8)
    
    # Apply internal classes only within nuclear regions
    nuclear_regions = nuclear_mask > 0
    combined_result[nuclear_regions & (internal_classes == 1)] = 1
    combined_result[nuclear_regions & (internal_classes == 2)] = 2
    
    return nuclear_mask, internal_classes, combined_result


def apply_nuclear_segmentation(image: np.ndarray, nuclear_params: Dict[str, Any], 
                             internal_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply two-step nuclear segmentation: first segment the nucleus, then segment within the nuclear boundary.
    
    Parameters
    ----------
    image : np.ndarray
        Input image for segmentation
    nuclear_params : dict
        Parameters for nuclear boundary detection
    internal_params : dict
        Parameters for internal segmentation within nucleus
        
    Returns
    -------
    tuple
        (nuclear_mask, internal_classes, combined_result)
        - nuclear_mask: Binary mask of nuclear boundaries
        - internal_classes: Classification within nuclear regions (1, 2)
        - combined_result: Final segmentation with 0=background, 1=nuclear_class1, 2=nuclear_class2
    """
    
    # Step 1: Nuclear boundary detection
    nuclear_mask = segment_nuclear_boundary(image, nuclear_params)
    
    # Step 2: Internal segmentation within nuclear regions
    internal_classes = segment_within_nucleus(image, nuclear_mask, internal_params)
    
    # Step 3: Combine results
    combined_result = np.zeros_like(image, dtype=np.uint8)
    
    # Apply internal classes only within nuclear regions
    nuclear_regions = nuclear_mask > 0
    combined_result[nuclear_regions & (internal_classes == 1)] = 1
    combined_result[nuclear_regions & (internal_classes == 2)] = 2
    
    return nuclear_mask, internal_classes, combined_result


def segment_nuclear_boundary(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Segment nuclear boundaries using specified method.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    params : dict
        Nuclear segmentation parameters
        
    Returns
    -------
    np.ndarray
        Binary nuclear mask
    """
    method = params.get('method', 'Otsu')
    min_size = params.get('min_size', 1000)
    closing_size = params.get('closing_size', 5)
    smooth_boundaries = params.get('smooth_boundaries', True)
    
    # Apply threshold
    if method == 'Otsu':
        threshold = threshold_otsu(image)
    elif method == 'Triangle':
        threshold = threshold_triangle(image)
    elif method == 'Manual':
        threshold = params.get('threshold', np.mean(image))
    else:
        threshold = threshold_otsu(image)
    
    # Create binary mask
    binary_mask = image > threshold
    
    # Apply morphological closing
    if closing_size > 0:
        binary_mask = binary_closing(binary_mask, disk(closing_size))
    
    # Remove small objects
    binary_mask = remove_small_objects(binary_mask, min_size=min_size)
    
    # Fill holes
    binary_mask = binary_fill_holes(binary_mask)
    
    # Smooth boundaries if requested
    if smooth_boundaries:
        # Apply erosion followed by dilation for smoothing
        smoothing_iterations = params.get('smoothing_iterations', 2)
        for _ in range(smoothing_iterations):
            binary_mask = binary_erosion(binary_mask, disk(1))
            binary_mask = binary_dilation(binary_mask, disk(1))
    
    return binary_mask.astype(np.uint8)


def segment_within_nucleus(image: np.ndarray, nuclear_mask: np.ndarray, 
                          params: Dict[str, Any]) -> np.ndarray:
    """
    Perform internal segmentation within nuclear regions.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    nuclear_mask : np.ndarray
        Binary nuclear mask
    params : dict
        Internal segmentation parameters
        
    Returns
    -------
    np.ndarray
        Internal classification (0=background, 1=class1, 2=class2)
    """
    method = params.get('method', 'Otsu')
    
    # Extract nuclear regions only
    nuclear_regions = nuclear_mask > 0
    nuclear_image = image.copy()
    nuclear_image[~nuclear_regions] = 0
    
    # Apply smoothing if requested for non-density map methods
    working_image = image.copy()
    if method != 'Density Map' and params.get('apply_smoothing', False):
        smoothing_method = params.get('smoothing_method', 'Median')
        if smoothing_method == 'Median':
            smoothing_size = params.get('smoothing_size', 5)
            from scipy.ndimage import median_filter
            working_image = median_filter(working_image, size=smoothing_size)
        elif smoothing_method == 'Gaussian':
            smoothing_sigma = params.get('smoothing_sigma', 1.0)
            from skimage.filters import gaussian
            working_image = gaussian(working_image, sigma=smoothing_sigma)
    
    # Apply internal segmentation method
    if method == 'Otsu':
        # Use Otsu on nuclear regions only
        nuclear_pixels = working_image[nuclear_regions]
        if len(nuclear_pixels) > 0:
            threshold = threshold_otsu(nuclear_pixels)
            internal_mask = (working_image > threshold) & nuclear_regions
        else:
            internal_mask = np.zeros_like(working_image, dtype=bool)
            
    elif method == 'Triangle':
        # Use Triangle on nuclear regions only
        nuclear_pixels = working_image[nuclear_regions]
        if len(nuclear_pixels) > 0:
            threshold = threshold_triangle(nuclear_pixels)
            internal_mask = (working_image > threshold) & nuclear_regions
        else:
            internal_mask = np.zeros_like(working_image, dtype=bool)
            
    elif method == 'Manual':
        threshold = params.get('threshold', np.mean(working_image))
        internal_mask = (working_image > threshold) & nuclear_regions
        
    elif method == 'Density Map':
        # Apply density map segmentation within nuclear regions
        internal_mask = apply_density_map_within_nucleus(image, nuclear_mask, params)
        
    else:
        # Default to Otsu
        nuclear_pixels = image[nuclear_regions]
        if len(nuclear_pixels) > 0:
            threshold = threshold_otsu(nuclear_pixels)
            internal_mask = (image > threshold) & nuclear_regions
        else:
            internal_mask = np.zeros_like(image, dtype=bool)
    
    # Create classification result
    internal_classes = np.zeros_like(image, dtype=np.uint8)
    
    # Class 1: Lower intensity nuclear regions
    class1_regions = nuclear_regions & ~internal_mask
    internal_classes[class1_regions] = 1
    
    # Class 2: Higher intensity nuclear regions
    class2_regions = nuclear_regions & internal_mask
    internal_classes[class2_regions] = 2
    
    return internal_classes


def apply_density_map_within_nucleus(image: np.ndarray, nuclear_mask: np.ndarray, 
                                   params: Dict[str, Any]) -> np.ndarray:
    """
    Apply density map segmentation within nuclear regions.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    nuclear_mask : np.ndarray
        Binary nuclear mask
    params : dict
        Density map parameters
        
    Returns
    -------
    np.ndarray
        Binary mask of high-density regions within nucleus
    """
    sigma_hp = params.get('sigma_hp', 2.0)
    disk_radius = params.get('disk_radius', 10)
    pcutoff_in = params.get('pcutoff_in', 0.10)
    pcutoff_out = params.get('pcutoff_out', 0.10)
    
    # Apply density map segmentation only to nuclear regions
    nuclear_regions = nuclear_mask > 0
    
    # Create masked image
    masked_image = image.copy()
    masked_image[~nuclear_regions] = 0
    
    try:
        # Apply density map algorithm within nuclear regions
        results = density_map_threshold(
            masked_image, 
            gaussian_sigma_hp=sigma_hp, 
            disk_radius=disk_radius, 
            pcutoff_in=pcutoff_in, 
            pcutoff_out=pcutoff_out
        )
        
        # Extract high-density regions
        if 'density_classes' in results:
            density_classes = results['density_classes']
            # High-density regions within nucleus
            high_density = (density_classes > 1) & nuclear_regions
        else:
            # Fallback to simple thresholding
            nuclear_pixels = image[nuclear_regions]
            if len(nuclear_pixels) > 0:
                threshold = np.percentile(nuclear_pixels, 75)  # Upper quartile
                high_density = (image > threshold) & nuclear_regions
            else:
                high_density = np.zeros_like(image, dtype=bool)
                
    except Exception:
        # Fallback to percentile-based thresholding
        nuclear_pixels = image[nuclear_regions]
        if len(nuclear_pixels) > 0:
            threshold = np.percentile(nuclear_pixels, 75)  # Upper quartile
            high_density = (image > threshold) & nuclear_regions
        else:
            high_density = np.zeros_like(image, dtype=bool)
    
    return high_density


def apply_three_class_segmentation(image: np.ndarray, cell_params: Dict[str, Any], 
                                  nuclear_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply three-class segmentation: Step 1 detects cells, Step 2 detects nuclei within cells, 
    Step 3 combines into background (0), cytoplasm (1), and nucleus (2).
    
    Parameters
    ----------
    image : np.ndarray
        Input image for segmentation
    cell_params : dict
        Parameters for cell detection
    nuclear_params : dict
        Parameters for nuclear detection within cells
        
    Returns
    -------
    tuple
        (cell_mask, nuclear_mask, three_class_result)
        - cell_mask: Binary mask of all cellular objects
        - nuclear_mask: Binary mask of nuclear regions within cells
        - three_class_result: Final segmentation with 0=background, 1=cytoplasm, 2=nucleus
    """
    
    # Step 1: Detect all cellular objects
    cell_mask = segment_cellular_objects(image, cell_params)
    
    # Step 2: Detect nuclear regions within cellular objects
    nuclear_mask = segment_nuclei_within_cells(image, cell_mask, nuclear_params)
    
    # Step 3: Combine into three classes
    three_class_result = np.zeros_like(image, dtype=np.uint8)
    
    # Class 1: Cytoplasm (cellular regions excluding nucleus)
    cytoplasm_regions = cell_mask & ~nuclear_mask
    three_class_result[cytoplasm_regions] = 1
    
    # Class 2: Nucleus (nuclear regions within cells)
    nuclear_regions = cell_mask & nuclear_mask
    three_class_result[nuclear_regions] = 2
    
    # Class 0: Background (everything else) remains 0
    
    return cell_mask, nuclear_mask, three_class_result


def segment_cellular_objects(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Segment cellular objects using specified method.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    params : dict
        Cell segmentation parameters
        
    Returns
    -------
    np.ndarray
        Binary mask of cellular objects
    """
    method = params.get('method', 'Otsu')
    min_size = params.get('min_size', 5000)
    closing_size = params.get('closing_size', 8)
    
    # Apply threshold based on method
    if method == 'Otsu':
        threshold = threshold_otsu(image)
    elif method == 'Triangle':
        threshold = threshold_triangle(image)
    elif method == 'Manual':
        threshold = params.get('threshold', np.mean(image))
    elif method == 'Density Map':
        # Apply density map segmentation for cell detection
        try:
            density_results = density_map_threshold(
                image,
                sigma_hp=params.get('sigma_hp', 2.0),
                disk_radius=params.get('disk_radius', 15),
                pcutoff_in=params.get('pcutoff_in', 0.15),
                pcutoff_out=params.get('pcutoff_out', 0.05)
            )
            
            if 'mask_in' in density_results:
                binary_mask = density_results['mask_in']
            else:
                # Fallback to simple thresholding
                threshold = threshold_otsu(image)
                binary_mask = image > threshold
        except Exception:
            # Fallback to Otsu
            threshold = threshold_otsu(image)
            binary_mask = image > threshold
    else:
        # Default to Otsu
        threshold = threshold_otsu(image)
        binary_mask = image > threshold
    
    # Create binary mask if not already created by density map
    if method != 'Density Map' or 'binary_mask' not in locals():
        binary_mask = image > threshold
    
    # Apply morphological closing to connect nearby regions
    if closing_size > 0:
        binary_mask = binary_closing(binary_mask, disk(closing_size))
    
    # Remove small objects
    binary_mask = remove_small_objects(binary_mask, min_size=min_size)
    
    # Fill holes in cellular objects
    binary_mask = binary_fill_holes(binary_mask)
    
    return binary_mask.astype(np.uint8)


def segment_nuclei_within_cells(image: np.ndarray, cell_mask: np.ndarray, 
                               params: Dict[str, Any]) -> np.ndarray:
    """
    Segment nuclear regions within cellular objects.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    cell_mask : np.ndarray
        Binary mask of cellular objects
    params : dict
        Nuclear segmentation parameters
        
    Returns
    -------
    np.ndarray
        Binary mask of nuclear regions
    """
    method = params.get('method', 'Otsu')
    min_size = params.get('min_size', 1500)
    closing_size = params.get('closing_size', 3)
    
    # Create masked image (only consider cellular regions)
    cellular_regions = cell_mask > 0
    masked_image = image.copy()
    masked_image[~cellular_regions] = 0
    
    # Apply nuclear segmentation method
    if method == 'Otsu':
        # Use Otsu on cellular regions only
        cellular_pixels = image[cellular_regions]
        if len(cellular_pixels) > 0:
            threshold = threshold_otsu(cellular_pixels)
            nuclear_mask = (image > threshold) & cellular_regions
        else:
            nuclear_mask = np.zeros_like(image, dtype=bool)
            
    elif method == 'Triangle':
        # Use Triangle on cellular regions only
        cellular_pixels = image[cellular_regions]
        if len(cellular_pixels) > 0:
            threshold = threshold_triangle(cellular_pixels)
            nuclear_mask = (image > threshold) & cellular_regions
        else:
            nuclear_mask = np.zeros_like(image, dtype=bool)
            
    elif method == 'Manual':
        threshold = params.get('threshold', np.mean(image))
        nuclear_mask = (image > threshold) & cellular_regions
        
    elif method == 'Density Map':
        # Apply density map segmentation within cellular regions
        try:
            density_results = density_map_threshold(
                masked_image,
                sigma_hp=params.get('sigma_hp', 1.5),
                disk_radius=params.get('disk_radius', 8),
                pcutoff_in=params.get('pcutoff_in', 0.20),
                pcutoff_out=params.get('pcutoff_out', 0.05)
            )
            
            if 'mask_in' in density_results:
                nuclear_mask = density_results['mask_in'] & cellular_regions
            else:
                # Fallback to percentile-based thresholding
                cellular_pixels = image[cellular_regions]
                if len(cellular_pixels) > 0:
                    threshold = np.percentile(cellular_pixels, 80)  # Upper 20%
                    nuclear_mask = (image > threshold) & cellular_regions
                else:
                    nuclear_mask = np.zeros_like(image, dtype=bool)
                    
        except Exception:
            # Fallback to percentile-based thresholding
            cellular_pixels = image[cellular_regions]
            if len(cellular_pixels) > 0:
                threshold = np.percentile(cellular_pixels, 80)  # Upper 20%
                nuclear_mask = (image > threshold) & cellular_regions
            else:
                nuclear_mask = np.zeros_like(image, dtype=bool)
                
    else:
        # Default to Otsu within cellular regions
        cellular_pixels = image[cellular_regions]
        if len(cellular_pixels) > 0:
            threshold = threshold_otsu(cellular_pixels)
            nuclear_mask = (image > threshold) & cellular_regions
        else:
            nuclear_mask = np.zeros_like(image, dtype=bool)
    
    # Apply morphological closing for nuclear regions
    if closing_size > 0:
        nuclear_mask = binary_closing(nuclear_mask, disk(closing_size))
    
    # Remove small nuclear objects
    nuclear_mask = remove_small_objects(nuclear_mask, min_size=min_size)
    
    # Fill holes in nuclear regions
    nuclear_mask = binary_fill_holes(nuclear_mask)
    
    # Ensure nuclei are only within cellular regions
    nuclear_mask = nuclear_mask & cellular_regions
    
    return nuclear_mask.astype(np.uint8)


def apply_nuclear_interior_segmentation(image: np.ndarray, boundary_params: Dict[str, Any], 
                                       internal_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply nuclear interior segmentation: 
    Step 1: Detect nuclear boundaries using standard methods
    Step 2: Apply density map segmentation within nuclear regions to identify internal structures
    Step 3: Create 3-class result: background (0) + 2 nuclear classes (1, 2)
    
    Parameters
    ----------
    image : np.ndarray
        Input image for segmentation
    boundary_params : dict
        Parameters for nuclear boundary detection
    internal_params : dict
        Parameters for internal density map segmentation
        
    Returns
    -------
    tuple
        (nuclear_boundary, internal_class1, internal_class2, three_class_result)
        - nuclear_boundary: Binary mask of nuclear boundaries
        - internal_class1: Binary mask of first internal nuclear class
        - internal_class2: Binary mask of second internal nuclear class
        - three_class_result: Final segmentation with 0=background, 1=nuclear class 1, 2=nuclear class 2
    """
    
    # Step 1: Detect nuclear boundaries
    nuclear_boundary = detect_nuclear_boundaries(image, boundary_params)
    
    # Step 2: Apply density map segmentation within nuclear regions
    internal_class1, internal_class2 = segment_nuclear_interior_with_density_map(
        image, nuclear_boundary, internal_params
    )
    
    # Step 3: Create three-class result
    three_class_result = np.zeros_like(image, dtype=np.uint8)
    
    # Class 1: First internal nuclear class
    three_class_result[internal_class1] = 1
    
    # Class 2: Second internal nuclear class  
    three_class_result[internal_class2] = 2
    
    # Class 0: Background (everything else) remains 0
    
    return nuclear_boundary, internal_class1, internal_class2, three_class_result


def detect_nuclear_boundaries(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Detect nuclear boundaries using standard thresholding methods.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    params : dict
        Nuclear boundary detection parameters
        
    Returns
    -------
    np.ndarray
        Binary mask of nuclear boundaries
    """
    method = params.get('method', 'Otsu')
    min_size = params.get('min_size', 8000)
    closing_size = params.get('closing_size', 5)
    
    # Apply threshold based on method
    if method == 'Otsu':
        threshold = threshold_otsu(image)
    elif method == 'Triangle':
        threshold = threshold_triangle(image)
    elif method == 'Manual':
        threshold = params.get('threshold', np.mean(image))
    else:
        threshold = threshold_otsu(image)
    
    # Create binary mask
    nuclear_mask = image > threshold
    
    # Apply morphological closing to connect nearby regions
    if closing_size > 0:
        nuclear_mask = binary_closing(nuclear_mask, disk(closing_size))
    
    # Remove small objects
    nuclear_mask = remove_small_objects(nuclear_mask, min_size=min_size)
    
    # Fill holes in nuclear regions
    nuclear_mask = binary_fill_holes(nuclear_mask)
    
    return nuclear_mask.astype(np.uint8)


def segment_nuclear_interior_with_density_map(image: np.ndarray, nuclear_boundary: np.ndarray, 
                                            params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply density map segmentation within nuclear regions to identify internal structures.
    This preserves the density map method that was working well before.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    nuclear_boundary : np.ndarray
        Binary mask of nuclear boundaries
    params : dict
        Density map segmentation parameters
        
    Returns
    -------
    tuple
        (internal_class1, internal_class2) - Two binary masks for internal nuclear classes
    """
    
    # Create masked image (only consider nuclear regions)
    nuclear_regions = nuclear_boundary > 0
    masked_image = image.copy()
    masked_image[~nuclear_regions] = 0
    
    # Apply density map segmentation within nuclear regions
    try:
        density_results = density_map_threshold(
            masked_image,
            gaussian_sigma_hp=params.get('sigma_hp', 2.0),
            disk_radius=params.get('disk_radius', 15),
            pcutoff_in=params.get('pcutoff_in', 0.15),
            pcutoff_out=params.get('pcutoff_out', 0.05)
        )
        
        # Extract the two internal classes from density map results
        if 'mask_in' in density_results and 'mask_out' in density_results:
            # Class 1: Interior objects (high density regions within nucleus)
            internal_class1 = density_results['mask_in'] & nuclear_regions
            
            # Class 2: Intermediate density regions (between interior and background)
            # This represents structures that are denser than background but less than interior
            mask_out = density_results['mask_out'] & nuclear_regions
            internal_class2 = mask_out & ~internal_class1
            
        else:
            # Fallback: Use simple thresholding within nuclear regions
            nuclear_pixels = image[nuclear_regions]
            if len(nuclear_pixels) > 0:
                # Use two thresholds to create two classes
                high_threshold = np.percentile(nuclear_pixels, 75)  # Upper 25% for class 2
                medium_threshold = np.percentile(nuclear_pixels, 50)  # Upper 50% for class 1
                
                internal_class2 = (image > high_threshold) & nuclear_regions  # Brightest structures
                internal_class1 = (image > medium_threshold) & nuclear_regions & ~internal_class2  # Medium brightness
            else:
                internal_class1 = np.zeros_like(image, dtype=bool)
                internal_class2 = np.zeros_like(image, dtype=bool)
                
    except Exception:
        # Fallback: Use percentile-based thresholding within nuclear regions
        nuclear_pixels = image[nuclear_regions]
        if len(nuclear_pixels) > 0:
            # Use two thresholds to create two classes
            high_threshold = np.percentile(nuclear_pixels, 75)  # Upper 25% for class 2
            medium_threshold = np.percentile(nuclear_pixels, 50)  # Upper 50% for class 1
            
            internal_class2 = (image > high_threshold) & nuclear_regions  # Brightest structures
            internal_class1 = (image > medium_threshold) & nuclear_regions & ~internal_class2  # Medium brightness
        else:
            internal_class1 = np.zeros_like(image, dtype=bool)
            internal_class2 = np.zeros_like(image, dtype=bool)
    
    # Apply minimal morphological operations to clean up results
    internal_class1 = binary_closing(internal_class1, disk(2))
    internal_class2 = binary_closing(internal_class2, disk(2))
    
    # Remove very small objects
    internal_class1 = remove_small_objects(internal_class1, min_size=50)
    internal_class2 = remove_small_objects(internal_class2, min_size=50)
    
    # Ensure classes are only within nuclear regions
    internal_class1 = internal_class1 & nuclear_regions
    internal_class2 = internal_class2 & nuclear_regions
    
    return internal_class1.astype(np.uint8), internal_class2.astype(np.uint8)
    internal_class2 = binary_closing(internal_class2, disk(2))
    
    # Remove very small objects
    internal_class1 = remove_small_objects(internal_class1, min_size=50)
    internal_class2 = remove_small_objects(internal_class2, min_size=50)
    
    # Ensure classes are only within nuclear regions
    internal_class1 = internal_class1 & nuclear_regions
    internal_class2 = internal_class2 & nuclear_regions
    
    return internal_class1.astype(np.uint8), internal_class2.astype(np.uint8)
    return internal_class1.astype(np.uint8), internal_class2.astype(np.uint8)
                
                internal_class2 = (image > high_threshold) & nuclear_regions  # Brightest structures
                internal_class1 = (image > medium_threshold) & nuclear_regions & ~internal_class2  # Medium brightness
            else:
                internal_class1 = np.zeros_like(image, dtype=bool)
                internal_class2 = np.zeros_like(image, dtype=bool)
                
    except Exception:
        # Fallback: Use percentile-based thresholding within nuclear regions
        nuclear_pixels = image[nuclear_regions]
        if len(nuclear_pixels) > 0:
            # Use two thresholds to create two classes
            high_threshold = np.percentile(nuclear_pixels, 75)  # Upper 25% for class 2
            medium_threshold = np.percentile(nuclear_pixels, 50)  # Upper 50% for class 1
            
            internal_class2 = (image > high_threshold) & nuclear_regions  # Brightest structures
            internal_class1 = (image > medium_threshold) & nuclear_regions & ~internal_class2  # Medium brightness
        else:
            internal_class1 = np.zeros_like(image, dtype=bool)
            internal_class2 = np.zeros_like(image, dtype=bool)
    
    # Apply minimal morphological operations to clean up results
    internal_class1 = binary_closing(internal_class1, disk(2))
    internal_class2 = binary_closing(internal_class2, disk(2))
    
    # Remove very small objects
    internal_class1 = remove_small_objects(internal_class1, min_size=50)
    internal_class2 = remove_small_objects(internal_class2, min_size=50)
    
    # Ensure classes are only within nuclear regions
    internal_class1 = internal_class1 & nuclear_regions
    internal_class2 = internal_class2 & nuclear_regions
    
    return internal_class1.astype(np.uint8), internal_class2.astype(np.uint8)


