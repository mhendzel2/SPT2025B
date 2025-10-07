"""
Track Quality Metrics Module
Provides comprehensive quality assessment for single particle tracking data.

Features:
- Signal-to-noise ratio (SNR) estimation
- Track completeness scoring
- Localization precision estimation
- Trajectory quality metrics
- Automated quality filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings


# ==================== SIGNAL-TO-NOISE RATIO ====================

def calculate_snr(tracks_df: pd.DataFrame, 
                 intensity_column: str = 'intensity',
                 background_column: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate signal-to-noise ratio for each track.
    
    SNR is calculated as:
    - If background provided: SNR = mean(signal) / std(background)
    - Otherwise: SNR = mean(intensity) / std(intensity)
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with columns: track_id, intensity
    intensity_column : str
        Column name containing intensity values
    background_column : str, optional
        Column name containing background values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: track_id, snr, mean_intensity, std_intensity
    """
    if intensity_column not in tracks_df.columns:
        warnings.warn(f"Column '{intensity_column}' not found. Returning empty results.")
        return pd.DataFrame(columns=['track_id', 'snr', 'mean_intensity', 'std_intensity'])
    
    results = []
    
    for track_id in tracks_df['track_id'].unique():
        track = tracks_df[tracks_df['track_id'] == track_id]
        
        intensities = track[intensity_column].values
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        if background_column and background_column in tracks_df.columns:
            background = track[background_column].values
            background_std = np.std(background)
            snr = mean_intensity / (background_std + 1e-10)
        else:
            # Use coefficient of variation as proxy
            snr = mean_intensity / (std_intensity + 1e-10)
        
        results.append({
            'track_id': track_id,
            'snr': snr,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity
        })
    
    return pd.DataFrame(results)


def estimate_localization_precision(tracks_df: pd.DataFrame,
                                   intensity_column: str = 'intensity',
                                   pixel_size: float = 0.1,
                                   method: str = 'thompson') -> pd.DataFrame:
    """
    Estimate localization precision using Thompson formula or nearest-neighbor method.
    
    Thompson et al. (2002): σ_loc ≈ √(s²/N + a²/12N + 8πs⁴b²/a²N²)
    where s = PSF width, N = photon count, a = pixel size, b = background
    
    Simplified: σ_loc ≈ s / √N  (for high SNR)
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with track data
    intensity_column : str
        Column containing intensity/photon counts
    pixel_size : float
        Pixel size in μm
    method : str
        'thompson' or 'nearest_neighbor'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: track_id, localization_precision (in μm)
    """
    if intensity_column not in tracks_df.columns:
        warnings.warn(f"Column '{intensity_column}' not found. Using default precision.")
        return pd.DataFrame({
            'track_id': tracks_df['track_id'].unique(),
            'localization_precision': [pixel_size / 10] * len(tracks_df['track_id'].unique())
        })
    
    results = []
    
    for track_id in tracks_df['track_id'].unique():
        track = tracks_df[tracks_df['track_id'] == track_id]
        
        if method == 'thompson':
            # Simplified Thompson formula
            intensities = track[intensity_column].values
            mean_intensity = np.mean(intensities)
            
            # Assume PSF width ≈ 2 * pixel_size
            psf_width = 2 * pixel_size
            
            # Localization precision (Thompson approximation)
            if mean_intensity > 0:
                precision = psf_width / np.sqrt(mean_intensity)
            else:
                precision = pixel_size
        
        elif method == 'nearest_neighbor':
            # Use variance of displacement as proxy
            coords = track[['x', 'y']].values * pixel_size
            if len(coords) > 1:
                displacements = np.diff(coords, axis=0)
                displacement_variance = np.var(np.linalg.norm(displacements, axis=1))
                # Precision ≈ √(variance / 2) for static particle
                precision = np.sqrt(displacement_variance / 2)
            else:
                precision = pixel_size / 10
        
        else:
            precision = pixel_size / 10
        
        results.append({
            'track_id': track_id,
            'localization_precision': precision
        })
    
    return pd.DataFrame(results)


# ==================== TRACK COMPLETENESS ====================

def calculate_track_completeness(tracks_df: pd.DataFrame,
                                expected_frames: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate track completeness metrics.
    
    Metrics:
    - Length: number of frames in track
    - Completeness: ratio of observed/expected frames
    - Gap ratio: fraction of frames with gaps
    - Max gap: largest gap in frames
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with columns: track_id, frame
    expected_frames : int, optional
        Expected number of frames (default: max frame observed)
        
    Returns
    -------
    pd.DataFrame
        Completeness metrics for each track
    """
    if expected_frames is None:
        expected_frames = tracks_df['frame'].max() - tracks_df['frame'].min() + 1
    
    results = []
    
    for track_id in tracks_df['track_id'].unique():
        track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        frames = track['frame'].values
        
        # Basic metrics
        track_length = len(frames)
        first_frame = frames[0]
        last_frame = frames[-1]
        duration = last_frame - first_frame + 1
        
        # Completeness
        completeness = track_length / duration if duration > 0 else 0
        
        # Gap analysis
        if len(frames) > 1:
            gaps = np.diff(frames) - 1
            n_gaps = np.sum(gaps > 0)
            max_gap = np.max(gaps) if len(gaps) > 0 else 0
            gap_ratio = n_gaps / (len(frames) - 1) if len(frames) > 1 else 0
        else:
            n_gaps = 0
            max_gap = 0
            gap_ratio = 0
        
        results.append({
            'track_id': track_id,
            'track_length': track_length,
            'duration': duration,
            'completeness': completeness,
            'n_gaps': n_gaps,
            'max_gap': max_gap,
            'gap_ratio': gap_ratio,
            'first_frame': first_frame,
            'last_frame': last_frame
        })
    
    return pd.DataFrame(results)


# ==================== TRAJECTORY QUALITY METRICS ====================

def calculate_trajectory_smoothness(tracks_df: pd.DataFrame,
                                   pixel_size: float = 0.1) -> pd.DataFrame:
    """
    Calculate trajectory smoothness metrics.
    
    Metrics:
    - Displacement variance
    - Turning angle variance
    - Straightness index
    - Wiggliness (path length / end-to-end distance)
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with columns: track_id, frame, x, y
    pixel_size : float
        Pixel size in μm
        
    Returns
    -------
    pd.DataFrame
        Smoothness metrics for each track
    """
    results = []
    
    for track_id in tracks_df['track_id'].unique():
        track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if len(track) < 3:
            continue
        
        coords = track[['x', 'y']].values * pixel_size
        
        # Displacements
        displacements = np.diff(coords, axis=0)
        displacement_lengths = np.linalg.norm(displacements, axis=1)
        
        # Displacement variance (normalized)
        displacement_var = np.var(displacement_lengths)
        displacement_cv = np.std(displacement_lengths) / (np.mean(displacement_lengths) + 1e-10)
        
        # Turning angles
        if len(displacements) > 1:
            angles = []
            for i in range(len(displacements) - 1):
                v1 = displacements[i]
                v2 = displacements[i + 1]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            
            angle_var = np.var(angles)
            mean_angle = np.mean(angles)
        else:
            angle_var = 0
            mean_angle = 0
        
        # Straightness
        end_to_end = np.linalg.norm(coords[-1] - coords[0])
        path_length = np.sum(displacement_lengths)
        straightness = end_to_end / (path_length + 1e-10)
        wiggliness = path_length / (end_to_end + 1e-10)
        
        results.append({
            'track_id': track_id,
            'displacement_variance': displacement_var,
            'displacement_cv': displacement_cv,
            'turning_angle_variance': angle_var,
            'mean_turning_angle': mean_angle,
            'straightness': straightness,
            'wiggliness': wiggliness,
            'path_length': path_length,
            'end_to_end_distance': end_to_end
        })
    
    return pd.DataFrame(results)


# ==================== COMPREHENSIVE QUALITY SCORE ====================

def calculate_quality_score(tracks_df: pd.DataFrame,
                           pixel_size: float = 0.1,
                           intensity_column: Optional[str] = 'intensity',
                           weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Calculate comprehensive quality score for each track.
    
    Combines multiple quality metrics into a single score (0-1).
    
    Quality factors:
    - Track length (longer is better)
    - Completeness (fewer gaps is better)
    - SNR (higher is better)
    - Localization precision (lower is better)
    - Trajectory smoothness (smoother is better)
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    intensity_column : str, optional
        Intensity column name
    weights : dict, optional
        Custom weights for different quality factors
        
    Returns
    -------
    pd.DataFrame
        Quality scores for each track
    """
    # Default weights
    if weights is None:
        weights = {
            'length': 0.25,
            'completeness': 0.25,
            'snr': 0.20,
            'precision': 0.15,
            'smoothness': 0.15
        }
    
    # Calculate individual metrics
    completeness_df = calculate_track_completeness(tracks_df)
    smoothness_df = calculate_trajectory_smoothness(tracks_df, pixel_size)
    
    # Optional metrics
    has_intensity = intensity_column and intensity_column in tracks_df.columns
    
    if has_intensity:
        snr_df = calculate_snr(tracks_df, intensity_column)
        precision_df = estimate_localization_precision(tracks_df, intensity_column, pixel_size)
    else:
        snr_df = None
        precision_df = None
    
    results = []
    
    for track_id in tracks_df['track_id'].unique():
        scores = {}
        
        # Length score (normalized by log scale)
        comp_row = completeness_df[completeness_df['track_id'] == track_id]
        if not comp_row.empty:
            track_length = comp_row.iloc[0]['track_length']
            length_score = min(1.0, np.log10(track_length + 1) / 2.5)  # Cap at ~300 frames
            completeness_score = comp_row.iloc[0]['completeness']
        else:
            length_score = 0
            completeness_score = 0
        
        scores['length'] = length_score
        scores['completeness'] = completeness_score
        
        # SNR score
        if snr_df is not None:
            snr_row = snr_df[snr_df['track_id'] == track_id]
            if not snr_row.empty:
                snr = snr_row.iloc[0]['snr']
                snr_score = min(1.0, snr / 20.0)  # Normalize to 0-1 (cap at SNR=20)
            else:
                snr_score = 0.5
        else:
            snr_score = 0.5  # Default if no intensity
        
        scores['snr'] = snr_score
        
        # Precision score (inverse - lower precision is better)
        if precision_df is not None:
            prec_row = precision_df[precision_df['track_id'] == track_id]
            if not prec_row.empty:
                precision = prec_row.iloc[0]['localization_precision']
                # Score: 1 for precision < 0.01 μm, 0 for precision > 0.1 μm
                precision_score = max(0, min(1.0, 1 - (precision - 0.01) / 0.09))
            else:
                precision_score = 0.5
        else:
            precision_score = 0.5
        
        scores['precision'] = precision_score
        
        # Smoothness score
        smooth_row = smoothness_df[smoothness_df['track_id'] == track_id]
        if not smooth_row.empty:
            # Use straightness and low displacement CV as smoothness indicators
            straightness = smooth_row.iloc[0]['straightness']
            disp_cv = smooth_row.iloc[0]['displacement_cv']
            
            # Score: high straightness OR low CV is good
            smoothness_score = 0.5 * straightness + 0.5 * max(0, 1 - disp_cv)
        else:
            smoothness_score = 0.5
        
        scores['smoothness'] = smoothness_score
        
        # Calculate weighted total
        total_score = sum(scores[key] * weights.get(key, 0) for key in scores.keys())
        
        results.append({
            'track_id': track_id,
            'quality_score': total_score,
            'length_score': scores['length'],
            'completeness_score': scores['completeness'],
            'snr_score': scores['snr'],
            'precision_score': scores['precision'],
            'smoothness_score': scores['smoothness']
        })
    
    return pd.DataFrame(results)


# ==================== QUALITY FILTERING ====================

def filter_tracks_by_quality(tracks_df: pd.DataFrame,
                             min_length: int = 10,
                             min_completeness: float = 0.7,
                             min_quality_score: float = 0.5,
                             pixel_size: float = 0.1,
                             intensity_column: Optional[str] = 'intensity') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter tracks based on quality criteria.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Input track data
    min_length : int
        Minimum track length (frames)
    min_completeness : float
        Minimum completeness ratio (0-1)
    min_quality_score : float
        Minimum overall quality score (0-1)
    pixel_size : float
        Pixel size in μm
    intensity_column : str, optional
        Intensity column name
        
    Returns
    -------
    filtered_df : pd.DataFrame
        Tracks passing quality criteria
    quality_report : pd.DataFrame
        Quality metrics for all tracks with pass/fail status
    """
    # Calculate all quality metrics
    completeness_df = calculate_track_completeness(tracks_df)
    quality_df = calculate_quality_score(tracks_df, pixel_size, intensity_column)
    
    # Merge metrics
    metrics_df = completeness_df.merge(quality_df, on='track_id')
    
    # Apply filters
    metrics_df['pass_length'] = metrics_df['track_length'] >= min_length
    metrics_df['pass_completeness'] = metrics_df['completeness'] >= min_completeness
    metrics_df['pass_quality'] = metrics_df['quality_score'] >= min_quality_score
    metrics_df['pass_all'] = (metrics_df['pass_length'] & 
                              metrics_df['pass_completeness'] & 
                              metrics_df['pass_quality'])
    
    # Filter tracks
    passing_track_ids = metrics_df[metrics_df['pass_all']]['track_id'].values
    filtered_df = tracks_df[tracks_df['track_id'].isin(passing_track_ids)].copy()
    
    return filtered_df, metrics_df


# ==================== QUALITY REPORT ====================

def generate_quality_report(tracks_df: pd.DataFrame,
                           pixel_size: float = 0.1,
                           intensity_column: Optional[str] = 'intensity') -> Dict[str, Any]:
    """
    Generate comprehensive quality report for entire dataset.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    intensity_column : str, optional
        Intensity column name
        
    Returns
    -------
    dict
        Comprehensive quality report with summary statistics
    """
    n_tracks = len(tracks_df['track_id'].unique())
    
    # Calculate metrics
    completeness_df = calculate_track_completeness(tracks_df)
    smoothness_df = calculate_trajectory_smoothness(tracks_df, pixel_size)
    quality_df = calculate_quality_score(tracks_df, pixel_size, intensity_column)
    
    has_intensity = intensity_column and intensity_column in tracks_df.columns
    if has_intensity:
        snr_df = calculate_snr(tracks_df, intensity_column)
        precision_df = estimate_localization_precision(tracks_df, intensity_column, pixel_size)
    
    # Summary statistics
    report = {
        'n_tracks': n_tracks,
        'track_length': {
            'mean': completeness_df['track_length'].mean(),
            'median': completeness_df['track_length'].median(),
            'min': completeness_df['track_length'].min(),
            'max': completeness_df['track_length'].max(),
            'std': completeness_df['track_length'].std()
        },
        'completeness': {
            'mean': completeness_df['completeness'].mean(),
            'median': completeness_df['completeness'].median(),
            'tracks_above_70pct': (completeness_df['completeness'] >= 0.7).sum(),
            'tracks_above_90pct': (completeness_df['completeness'] >= 0.9).sum()
        },
        'quality_score': {
            'mean': quality_df['quality_score'].mean(),
            'median': quality_df['quality_score'].median(),
            'tracks_above_0.5': (quality_df['quality_score'] >= 0.5).sum(),
            'tracks_above_0.7': (quality_df['quality_score'] >= 0.7).sum(),
            'tracks_above_0.9': (quality_df['quality_score'] >= 0.9).sum()
        },
        'smoothness': {
            'mean_straightness': smoothness_df['straightness'].mean(),
            'mean_wiggliness': smoothness_df['wiggliness'].mean(),
            'mean_displacement_cv': smoothness_df['displacement_cv'].mean()
        }
    }
    
    if has_intensity:
        report['snr'] = {
            'mean': snr_df['snr'].mean(),
            'median': snr_df['snr'].median(),
            'tracks_above_5': (snr_df['snr'] >= 5).sum(),
            'tracks_above_10': (snr_df['snr'] >= 10).sum()
        }
        report['localization_precision'] = {
            'mean': precision_df['localization_precision'].mean(),
            'median': precision_df['localization_precision'].median(),
            'unit': 'μm'
        }
    
    return report


# ==================== HIGH-LEVEL API ====================

def assess_track_quality(tracks_df: pd.DataFrame,
                        pixel_size: float = 0.1,
                        frame_interval: float = 0.1,
                        intensity_column: Optional[str] = 'intensity',
                        apply_filtering: bool = False,
                        filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Comprehensive track quality assessment (high-level API).
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Time between frames in seconds
    intensity_column : str, optional
        Intensity column name
    apply_filtering : bool
        Whether to filter tracks
    filter_criteria : dict, optional
        Custom filtering criteria
        
    Returns
    -------
    dict
        Complete quality assessment results
    """
    if filter_criteria is None:
        filter_criteria = {
            'min_length': 10,
            'min_completeness': 0.7,
            'min_quality_score': 0.5
        }
    
    results = {
        'success': True,
        'n_tracks_input': len(tracks_df['track_id'].unique()),
        'parameters': {
            'pixel_size': pixel_size,
            'frame_interval': frame_interval,
            'intensity_column': intensity_column
        }
    }
    
    try:
        # Generate quality report
        report = generate_quality_report(tracks_df, pixel_size, intensity_column)
        results['summary'] = report
        
        # Calculate detailed metrics
        results['completeness'] = calculate_track_completeness(tracks_df)
        results['smoothness'] = calculate_trajectory_smoothness(tracks_df, pixel_size)
        results['quality_scores'] = calculate_quality_score(tracks_df, pixel_size, intensity_column)
        
        # Optional: SNR and precision
        has_intensity = intensity_column and intensity_column in tracks_df.columns
        if has_intensity:
            results['snr'] = calculate_snr(tracks_df, intensity_column)
            results['localization_precision'] = estimate_localization_precision(
                tracks_df, intensity_column, pixel_size
            )
        
        # Optional: Apply filtering
        if apply_filtering:
            filtered_df, filter_report = filter_tracks_by_quality(
                tracks_df,
                min_length=filter_criteria['min_length'],
                min_completeness=filter_criteria['min_completeness'],
                min_quality_score=filter_criteria['min_quality_score'],
                pixel_size=pixel_size,
                intensity_column=intensity_column
            )
            
            results['filtered_tracks'] = filtered_df
            results['filter_report'] = filter_report
            results['n_tracks_passed'] = len(filtered_df['track_id'].unique())
            results['filter_pass_rate'] = results['n_tracks_passed'] / results['n_tracks_input']
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results
