"""
Intensity Analysis Module for SPT Analysis Application.
Provides enhanced intensity-based analysis tools for particle tracking data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore, linregress
import warnings

# Import intensity analysis constant
try:
    from constants import DEFAULT_INTENSITY_VARIATION_THRESHOLD
except ImportError:
    DEFAULT_INTENSITY_VARIATION_THRESHOLD = 0.1

warnings.filterwarnings('ignore')


def _safe_numeric_convert(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Safely convert a column to numeric, handling malformed string data.
    
    This handles cases where data may be concatenated strings like '29.027.031.048...'
    which can occur from CSV parsing errors or data corruption.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column
    column : str
        Name of the column to convert
        
    Returns
    -------
    pd.Series
        Numeric series with invalid values as NaN
    """
    if column not in df.columns:
        return pd.Series(dtype=float)
    
    col_data = df[column].copy()
    
    # If already numeric, return as-is
    if pd.api.types.is_numeric_dtype(col_data):
        return col_data
    
    # Try direct numeric conversion first
    converted = pd.to_numeric(col_data, errors='coerce')
    
    # Check for high NaN rate indicating parsing issues
    nan_rate = converted.isna().sum() / len(converted) if len(converted) > 0 else 0
    
    if nan_rate > 0.5:
        # Many values couldn't be converted - likely malformed data
        # Try to extract first valid number from each value
        def extract_first_number(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            try:
                # Convert to string and try to extract first number
                s = str(val)
                # Check for multiple decimal points (malformed like '29.027.031')
                if s.count('.') > 1:
                    # Extract first number before second decimal
                    parts = s.split('.')
                    if len(parts) >= 2:
                        try:
                            return float(f"{parts[0]}.{parts[1]}")
                        except ValueError:
                            return np.nan
                return float(s)
            except (ValueError, TypeError):
                return np.nan
        
        converted = col_data.apply(extract_first_number)
    
    return converted


def extract_intensity_channels(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extract available intensity channels from the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing track data with intensity information
        
    Returns
    -------
    dict
        Dictionary mapping channel names to available intensity metrics
    """
    channels = {}
    
    # Check for different channel naming conventions
    for i in range(1, 4):  # Support up to 3 channels
        ch_name = f"ch{i}"
        ch_columns = []
        
        # Look for different intensity metrics
        metrics = ['mean', 'median', 'min', 'max', 'total', 'std']
        for metric in metrics:
            col_variants = [
                f'{metric}_intensity_ch{i}',
                f'{metric}_ch{i}',
                f'{metric.upper()}_INTENSITY_CH{i}',
                f'Mean intensity ch{i}' if metric == 'mean' else f'{metric.capitalize()} intensity ch{i}'
            ]
            
            for variant in col_variants:
                if variant in df.columns:
                    ch_columns.append(variant)
                    break
        
        if ch_columns:
            channels[ch_name] = ch_columns
    
    # Also check for contrast and SNR data
    for i in range(1, 4):
        ch_name = f"ch{i}"
        if ch_name in channels:
            # Add contrast and SNR if available
            contrast_cols = [f'contrast_ch{i}', f'Contrast ch{i}', f'Ctrst ch{i}', f'CONTRAST_CH{i}']
            snr_cols = [f'snr_ch{i}', f'SNR ch{i}', f'Signal/Noise ratio ch{i}', f'SNR_CH{i}']
            
            for col in contrast_cols:
                if col in df.columns:
                    channels[ch_name].append(col)
                    break
                    
            for col in snr_cols:
                if col in df.columns:
                    channels[ch_name].append(col)
                    break
    
    return channels

def get_channel_labels_with_particle_default(
    tracks_df: pd.DataFrame,
    intensity_variation_threshold: float = DEFAULT_INTENSITY_VARIATION_THRESHOLD
) -> Dict[str, str]:
    """
    Determines descriptive labels for intensity channels, defaulting to "particle"
    for a single channel that exhibits low intensity variation.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with intensity columns.
    intensity_variation_threshold : float, optional
        Coefficient of variation (CV) threshold below which a channel is considered
        to have "low intensity variation".

    Returns
    -------
    Dict[str, str]
        A dictionary mapping original channel keys (e.g., 'ch1') to descriptive labels
        (e.g., 'ch1' or 'particle').
    """
    identified_channels_with_cols = extract_intensity_channels(tracks_df)
    
    if not identified_channels_with_cols:
        return {}

    # Initialize labels with original channel keys
    channel_labels = {key: key for key in identified_channels_with_cols.keys()}
    
    low_variation_channel_keys = []
    
    for ch_key, intensity_cols_for_channel in identified_channels_with_cols.items():
        if not intensity_cols_for_channel:
            continue

        primary_intensity_col_name = None
        # Prioritize metrics for variation calculation (e.g. mean or median intensity)
        preferred_metrics_for_variation = ['mean', 'median', 'total', 'sum', 'intensity'] 
        
        # Find the most representative intensity column for this channel to assess its variation
        for pref_metric in preferred_metrics_for_variation:
            for col_name in intensity_cols_for_channel:
                # Simple check if preferred metric is part of the column name
                if pref_metric.lower() in col_name.lower() and f'{ch_key}'.lower() in col_name.lower():
                    primary_intensity_col_name = col_name
                    break
            if primary_intensity_col_name:
                break
        
        # Fallback to the first column if no preferred metric column found
        if not primary_intensity_col_name and intensity_cols_for_channel:
            primary_intensity_col_name = intensity_cols_for_channel[0]

        if primary_intensity_col_name and primary_intensity_col_name in tracks_df.columns:
            # Ensure the column is numeric and non-empty before calculating CV
            if pd.api.types.is_numeric_dtype(tracks_df[primary_intensity_col_name]):
                intensity_values = tracks_df[primary_intensity_col_name].dropna()
                
                if len(intensity_values) > 1: # CV requires at least 2 points for stddev
                    mean_val = np.mean(intensity_values)
                    std_val = np.std(intensity_values)
                    
                    if mean_val != 0: # Avoid division by zero for CV calculation
                        cv = abs(std_val / mean_val) # Use absolute for CV
                        if cv < intensity_variation_threshold:
                            low_variation_channel_keys.append(ch_key)
                    elif std_val == 0: # If mean is 0 and std is also 0, it's constant (low variation)
                        low_variation_channel_keys.append(ch_key)
                elif len(intensity_values) == 1: # A single point track/intensity has zero variation
                     low_variation_channel_keys.append(ch_key)

    # Apply "particle" label if *exactly one* of the identified channels meets the low variation criteria
    if len(low_variation_channel_keys) == 1:
        particle_channel_key = low_variation_channel_keys[0]
        channel_labels[particle_channel_key] = "particle"
    
    return channel_labels

def calculate_movement_metrics(tracks_df: pd.DataFrame, 
                              intensity_columns: List[str] = None) -> Dict[str, Any]:
    """
    Calculate movement metrics correlated with intensity changes.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with intensity and position information
    intensity_columns : List[str], optional
        List of intensity columns to analyze
        
    Returns
    -------
    dict
        Dictionary containing movement-intensity correlation metrics
    """
    if tracks_df.empty:
        return {'error': 'No track data available'}
    
    if intensity_columns is None:
        intensity_columns = [col for col in tracks_df.columns if 'intensity' in col.lower()]
    
    if not intensity_columns:
        return {'error': 'No intensity columns found'}
    
    # Safe conversion of intensity columns to numeric
    working_df = tracks_df.copy()
    for int_col in intensity_columns:
        if int_col in working_df.columns:
            working_df[int_col] = pd.to_numeric(working_df[int_col], errors='coerce')
    
    results = {
        'track_metrics': {},
        'correlation_matrix': {},
        'movement_intensity_correlation': {}
    }
    
    for track_id in working_df['track_id'].unique():
        track_data = working_df[working_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 3:
            continue
        
        # Calculate displacement
        dx = np.diff(track_data['x'].values)
        dy = np.diff(track_data['y'].values)
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate velocity
        velocity = displacement if len(displacement) > 0 else np.array([0])
        
        track_metrics = {
            'mean_velocity': np.mean(velocity),
            'std_velocity': np.std(velocity),
            'max_velocity': np.max(velocity),
            'total_displacement': np.sum(displacement)
        }
        
        # Correlate with intensity changes
        for int_col in intensity_columns:
            if int_col in track_data.columns:
                intensities = track_data[int_col].values[:-1]  # Match displacement length
                if len(intensities) == len(velocity) and len(intensities) > 1:
                    correlation = np.corrcoef(intensities, velocity)[0, 1]
                    if not np.isnan(correlation):
                        track_metrics[f'{int_col}_velocity_correlation'] = correlation
        
        results['track_metrics'][track_id] = track_metrics
    
    return results

def correlate_intensity_movement(tracks_df: pd.DataFrame, 
                               intensity_column: str = 'intensity') -> Dict[str, Any]:
    """
    Correlate intensity changes with movement patterns.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with intensity and position information
    intensity_column : str
        Column containing intensity values
        
    Returns
    -------
    dict
        Dictionary containing correlation analysis results
    """
    if tracks_df.empty or intensity_column not in tracks_df.columns:
        return {'error': f'No {intensity_column} data available'}
    
    # Safely convert intensity column to numeric
    intensity_values = _safe_numeric_convert(tracks_df, intensity_column)
    
    # Check if conversion succeeded
    if intensity_values.isna().all():
        return {'error': f'Could not convert {intensity_column} to numeric values. Data may be malformed.'}
    
    # Create working copy with converted values
    working_df = tracks_df.copy()
    working_df[intensity_column] = intensity_values
    working_df = working_df.dropna(subset=[intensity_column, 'x', 'y', 'track_id', 'frame'])
    
    if working_df.empty:
        return {'error': f'No valid data after converting {intensity_column} to numeric'}
    
    correlations = []
    
    for track_id in working_df['track_id'].unique():
        track_data = working_df[working_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 3:
            continue
        
        # Calculate movement metrics
        dx = np.diff(track_data['x'].values)
        dy = np.diff(track_data['y'].values)
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Get intensity changes
        intensity_changes = np.diff(track_data[intensity_column].values)
        
        if len(displacement) == len(intensity_changes) and len(displacement) > 1:
            # Calculate correlation
            correlation = np.corrcoef(displacement, np.abs(intensity_changes))[0, 1]
            if not np.isnan(correlation):
                correlations.append({
                    'track_id': track_id,
                    'correlation': correlation,
                    'mean_displacement': np.mean(displacement),
                    'mean_intensity_change': np.mean(np.abs(intensity_changes))
                })
    
    if correlations:
        correlation_df = pd.DataFrame(correlations)
        return {
            'correlations': correlation_df,
            'mean_correlation': correlation_df['correlation'].mean(),
            'positive_correlations': len(correlation_df[correlation_df['correlation'] > 0]),
            'negative_correlations': len(correlation_df[correlation_df['correlation'] < 0])
        }
    else:
        return {'error': 'No valid correlations calculated'}

def create_intensity_movement_plots(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create plots for intensity-movement correlation analysis.
    
    Parameters
    ----------
    analysis_results : dict
        Results from intensity-movement correlation analysis
        
    Returns
    -------
    dict
        Dictionary containing plot data and configurations
    """
    if 'error' in analysis_results:
        return analysis_results
    
    plots = {
        'correlation_histogram': {},
        'scatter_plots': {},
        'time_series': {}
    }
    
    if 'correlations' in analysis_results:
        correlations = analysis_results['correlations']
        
        # Histogram of correlations
        plots['correlation_histogram'] = {
            'data': correlations['correlation'].values,
            'title': 'Distribution of Intensity-Movement Correlations',
            'xlabel': 'Correlation Coefficient',
            'ylabel': 'Frequency'
        }
        
        # Scatter plot
        plots['scatter_plots'] = {
            'x': correlations['mean_displacement'].values,
            'y': correlations['mean_intensity_change'].values,
            'title': 'Mean Displacement vs Mean Intensity Change',
            'xlabel': 'Mean Displacement (pixels)',
            'ylabel': 'Mean Intensity Change'
        }
    
    return plots

def intensity_based_segmentation(image: np.ndarray, 
                                method: str = 'otsu',
                                **kwargs) -> np.ndarray:
    """
    Perform intensity-based image segmentation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image for segmentation
    method : str
        Segmentation method ('otsu', 'adaptive', 'manual')
    **kwargs : dict
        Method-specific parameters
        
    Returns
    -------
    np.ndarray
        Binary segmentation mask
    """
    if method == 'otsu':
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(image)
        return image > threshold
    elif method == 'adaptive':
        from skimage.filters import threshold_local
        block_size = kwargs.get('block_size', 35)
        threshold = threshold_local(image, block_size)
        return image > threshold
    elif method == 'manual':
        threshold = kwargs.get('threshold', np.mean(image))
        return image > threshold
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return np.zeros_like(image, dtype=bool)

def analyze_intensity_profiles(tracks_df: pd.DataFrame, 
                              intensity_column: str = 'intensity',
                              smoothing_window: int = 5) -> Dict[str, Any]:
    """
    Analyze intensity profiles along particle tracks for photobleaching and blinking detection.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with intensity information
    intensity_column : str
        Column name containing intensity values
    smoothing_window : int
        Window size for smoothing intensity profiles
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing intensity analysis results
    """
    if tracks_df.empty or intensity_column not in tracks_df.columns:
        return {'error': f'No {intensity_column} data available'}
    
    # Safely convert intensity column to numeric
    intensity_values = _safe_numeric_convert(tracks_df, intensity_column)
    
    # Check if conversion succeeded
    if intensity_values.isna().all():
        return {'error': f'Could not convert {intensity_column} to numeric values. Data may be malformed.'}
    
    # Create working copy with converted values
    working_df = tracks_df.copy()
    working_df[intensity_column] = intensity_values
    working_df = working_df.dropna(subset=[intensity_column, 'track_id', 'frame'])
    
    if working_df.empty:
        return {'error': f'No valid data after converting {intensity_column} to numeric'}
    
    results = {
        'track_profiles': {},
        'photobleaching_detected': [],
        'blinking_events': {},
        'intensity_statistics': {}
    }
    
    for track_id in working_df['track_id'].unique():
        track_data = working_df[working_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 3:
            continue
            
        intensities = track_data[intensity_column].values.astype(float)
        frames = track_data['frame'].values
        
        # Skip if any NaN values remain
        if np.any(np.isnan(intensities)):
            continue
        
        # Smooth intensity profile
        if len(intensities) >= smoothing_window:
            smoothed_intensities = savgol_filter(intensities, smoothing_window, 2)
        else:
            smoothed_intensities = intensities
        
        # Detect photobleaching (monotonic decrease)
        photobleaching_score = np.corrcoef(frames, smoothed_intensities)[0, 1]
        if not np.isnan(photobleaching_score) and photobleaching_score < -0.7:  # Strong negative correlation
            results['photobleaching_detected'].append(track_id)
        
        # Detect blinking events (sudden intensity drops and recoveries)
        intensity_diff = np.diff(smoothed_intensities)
        if len(intensity_diff) > 1 and np.std(intensity_diff) > 0:
            z_scores = np.abs(zscore(intensity_diff))
            blinking_events = find_peaks(z_scores, height=2.0, distance=2)[0]
            
            if len(blinking_events) > 0:
                results['blinking_events'][track_id] = {
                    'event_frames': frames[blinking_events + 1].tolist(),
                    'event_count': len(blinking_events)
                }
        
        # Store profile data
        results['track_profiles'][track_id] = {
            'frames': frames.tolist(),
            'raw_intensities': intensities.tolist(),
            'smoothed_intensities': smoothed_intensities.tolist(),
            'photobleaching_score': float(photobleaching_score) if not np.isnan(photobleaching_score) else 0.0
        }
    
    # Overall statistics
    all_intensities = tracks_df[intensity_column].dropna().values
    results['intensity_statistics'] = {
        'mean_intensity': np.mean(all_intensities),
        'median_intensity': np.median(all_intensities),
        'std_intensity': np.std(all_intensities),
        'cv_intensity': np.std(all_intensities) / np.mean(all_intensities),
        'tracks_with_photobleaching': len(results['photobleaching_detected']),
        'tracks_with_blinking': len(results['blinking_events'])
    }
    
    return results

def classify_intensity_behavior(tracks_df: pd.DataFrame,
                               intensity_column: str = 'intensity') -> pd.DataFrame:
    """
    Classify tracks based on their intensity behavior patterns.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with intensity information
    intensity_column : str
        Column name containing intensity values
        
    Returns
    -------
    pd.DataFrame
        Enhanced tracks with intensity behavior classification
    """
    enhanced_tracks = tracks_df.copy()
    enhanced_tracks['intensity_behavior'] = 'stable'
    enhanced_tracks['photobleaching_rate'] = np.nan
    enhanced_tracks['blinking_frequency'] = 0
    
    if intensity_column not in tracks_df.columns:
        return enhanced_tracks
    
    # Safe conversion of intensity column to numeric
    working_df = _safe_intensity_to_numeric(tracks_df, intensity_column)
    if working_df is None:
        return enhanced_tracks  # Return with defaults if conversion fails
    
    for track_id in working_df['track_id'].unique():
        track_data = working_df[working_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 3:
            continue
            
        intensities = track_data[intensity_column].values
        frames = track_data['frame'].values
        
        # Calculate photobleaching rate
        if len(intensities) > 2:
            slope, _, r_value, _, _ = linregress(frames, intensities)
            photobleaching_rate = -slope  # Negative slope indicates photobleaching
            
            # Update tracks
            mask = enhanced_tracks['track_id'] == track_id
            enhanced_tracks.loc[mask, 'photobleaching_rate'] = photobleaching_rate
            
            # Classify behavior
            if r_value < -0.7 and slope < 0:
                enhanced_tracks.loc[mask, 'intensity_behavior'] = 'photobleaching'
            elif np.std(intensities) / np.mean(intensities) > 0.3:
                enhanced_tracks.loc[mask, 'intensity_behavior'] = 'variable'
            
            # Count blinking events
            intensity_diff = np.diff(intensities)
            z_scores = np.abs(zscore(intensity_diff))
            blinking_count = len(find_peaks(z_scores, height=2.0, distance=2)[0])
            enhanced_tracks.loc[mask, 'blinking_frequency'] = blinking_count / len(track_data)
            
            if blinking_count > 2:
                enhanced_tracks.loc[mask, 'intensity_behavior'] = 'blinking'
    
    return enhanced_tracks

def detect_colocalization_events(primary_tracks: pd.DataFrame,
                                secondary_tracks: pd.DataFrame,
                                max_distance: float = 5.0,
                                min_overlap_frames: int = 3) -> Dict[str, Any]:
    """
    Detect colocalization events between primary and secondary channel tracks.
    
    Parameters
    ----------
    primary_tracks : pd.DataFrame
        Primary channel track data
    secondary_tracks : pd.DataFrame
        Secondary channel track data
    max_distance : float
        Maximum distance for colocalization (micrometers)
    min_overlap_frames : int
        Minimum number of overlapping frames required
        
    Returns
    -------
    Dict[str, Any]
        Colocalization analysis results
    """
    from scipy.spatial.distance import cdist
    
    colocalization_events = []
    
    for primary_id in primary_tracks['track_id'].unique():
        primary_track = primary_tracks[primary_tracks['track_id'] == primary_id]
        
        for secondary_id in secondary_tracks['track_id'].unique():
            secondary_track = secondary_tracks[secondary_tracks['track_id'] == secondary_id]
            
            # Find overlapping frames
            primary_frames = set(primary_track['frame'])
            secondary_frames = set(secondary_track['frame'])
            overlap_frames = primary_frames.intersection(secondary_frames)
            
            if len(overlap_frames) < min_overlap_frames:
                continue
            
            # Calculate distances in overlapping frames
            distances = []
            for frame in overlap_frames:
                p1 = primary_track[primary_track['frame'] == frame][['x', 'y']].values
                p2 = secondary_track[secondary_track['frame'] == frame][['x', 'y']].values
                
                if len(p1) > 0 and len(p2) > 0:
                    dist = cdist(p1, p2)[0, 0]
                    distances.append(dist)
            
            if distances:
                mean_distance = np.mean(distances)
                if mean_distance <= max_distance:
                    colocalization_events.append({
                        'primary_track_id': primary_id,
                        'secondary_track_id': secondary_id,
                        'overlap_frames': len(overlap_frames),
                        'mean_distance': mean_distance,
                        'min_distance': np.min(distances),
                        'max_distance': np.max(distances),
                        'colocalization_duration': len(overlap_frames)
                    })
    
    # Calculate statistics
    results = {
        'colocalization_events': colocalization_events,
        'total_events': len(colocalization_events),
        'primary_tracks_colocalized': len(set([e['primary_track_id'] for e in colocalization_events])),
        'secondary_tracks_colocalized': len(set([e['secondary_track_id'] for e in colocalization_events]))
    }
    
    if colocalization_events:
        durations = [e['colocalization_duration'] for e in colocalization_events]
        distances = [e['mean_distance'] for e in colocalization_events]
        
        results['statistics'] = {
            'mean_colocalization_duration': np.mean(durations),
            'median_colocalization_duration': np.median(durations),
            'mean_colocalization_distance': np.mean(distances),
            'median_colocalization_distance': np.median(distances)
        }
    
    return results

def analyze_intensity_hotspots(tracks_df: pd.DataFrame,
                              intensity_column: str = 'intensity',
                              spatial_bin_size: float = 1.0,
                              intensity_threshold_percentile: float = 90) -> Dict[str, Any]:
    """
    Identify spatial regions with high intensity particles (hotspots).
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with intensity and position information
    intensity_column : str
        Column name containing intensity values
    spatial_bin_size : float
        Size of spatial bins for hotspot detection (micrometers)
    intensity_threshold_percentile : float
        Percentile threshold for high intensity particles
        
    Returns
    -------
    Dict[str, Any]
        Hotspot analysis results
    """
    if tracks_df.empty or intensity_column not in tracks_df.columns:
        return {'error': f'No {intensity_column} data available'}
    
    # Safe conversion of intensity column to numeric
    working_df = _safe_intensity_to_numeric(tracks_df, intensity_column)
    if working_df is None:
        return {'error': f'Could not convert {intensity_column} to numeric values'}
    
    # Calculate intensity threshold
    intensity_threshold = np.percentile(working_df[intensity_column].dropna(), intensity_threshold_percentile)
    
    # Filter high intensity particles
    high_intensity_tracks = working_df[working_df[intensity_column] >= intensity_threshold]
    
    if high_intensity_tracks.empty:
        return {'error': 'No high intensity particles found'}
    
    # Create spatial bins
    x_min, x_max = working_df['x'].min(), working_df['x'].max()
    y_min, y_max = working_df['y'].min(), working_df['y'].max()
    
    x_bins = np.arange(x_min, x_max + spatial_bin_size, spatial_bin_size)
    y_bins = np.arange(y_min, y_max + spatial_bin_size, spatial_bin_size)
    
    # Count high intensity particles in each bin
    hist, x_edges, y_edges = np.histogram2d(
        high_intensity_tracks['x'], 
        high_intensity_tracks['y'], 
        bins=[x_bins, y_bins]
    )
    
    # Find hotspot bins (top 10% of bins by particle count)
    hotspot_threshold = np.percentile(hist[hist > 0], 90)
    hotspot_indices = np.where(hist >= hotspot_threshold)
    
    hotspots = []
    for i, j in zip(hotspot_indices[0], hotspot_indices[1]):
        x_center = (x_edges[i] + x_edges[i+1]) / 2
        y_center = (y_edges[j] + y_edges[j+1]) / 2
        particle_count = int(hist[i, j])
        
        # Get particles in this hotspot
        particles_in_hotspot = high_intensity_tracks[
            (high_intensity_tracks['x'] >= x_edges[i]) &
            (high_intensity_tracks['x'] < x_edges[i+1]) &
            (high_intensity_tracks['y'] >= y_edges[j]) &
            (high_intensity_tracks['y'] < y_edges[j+1])
        ]
        
        hotspot = {
            'center_x': x_center,
            'center_y': y_center,
            'particle_count': particle_count,
            'mean_intensity': particles_in_hotspot[intensity_column].mean(),
            'max_intensity': particles_in_hotspot[intensity_column].max(),
            'area': spatial_bin_size ** 2
        }
        hotspots.append(hotspot)
    
    results = {
        'hotspots': hotspots,
        'total_hotspots': len(hotspots),
        'intensity_threshold': intensity_threshold,
        'spatial_bin_size': spatial_bin_size,
        'high_intensity_particle_count': len(high_intensity_tracks),
        'hotspot_statistics': {}
    }
    
    if hotspots:
        particle_counts = [h['particle_count'] for h in hotspots]
        intensities = [h['mean_intensity'] for h in hotspots]
        
        results['hotspot_statistics'] = {
            'mean_particles_per_hotspot': np.mean(particle_counts),
            'total_particles_in_hotspots': sum(particle_counts),
            'mean_hotspot_intensity': np.mean(intensities),
            'max_hotspot_intensity': np.max(intensities)
        }
    
    return results