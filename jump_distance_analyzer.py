"""
Jump Distance Analysis Module for Single Particle Tracking

This module provides comprehensive analysis of jump distances between consecutive
particle positions, which is essential for understanding particle motion dynamics
and identifying different diffusion regimes.

Key Features:
- Calculate jump distances between consecutive positions
- Analyze jump distance distributions
- Identify confined, free, and directed motion
- Statistical analysis of jump patterns
- Visualization of jump distance histograms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import curve_fit


def calculate_jump_distances(tracks_df: pd.DataFrame, 
                           pixel_size: float = 1.0, 
                           frame_interval: float = 1.0,
                           min_track_length: int = 5) -> pd.DataFrame:
    """
    Calculate jump distances between consecutive particle positions.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with columns 'track_id', 'frame', 'x', 'y'
    pixel_size : float, default=1.0
        Pixel size in micrometers
    frame_interval : float, default=1.0
        Frame interval in seconds
    min_track_length : int, default=5
        Minimum track length to include in analysis
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'track_id', 'frame', 'jump_distance', 'jump_angle', 
        'velocity', 'x_displacement', 'y_displacement'
    """
    # Validate input
    if tracks_df is None or tracks_df.empty:
        return pd.DataFrame(columns=['track_id', 'frame', 'jump_distance', 'jump_angle', 
                                   'velocity', 'x_displacement', 'y_displacement'])
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    if not all(col in tracks_df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")
    
    jump_results = []
    
    # Group by track_id and calculate jump distances
    for track_id, track_data in tracks_df.groupby('track_id'):
        # Sort by frame
        track = track_data.sort_values('frame').copy()
        
        # Skip short tracks
        if len(track) < min_track_length:
            continue
        
        # Convert to physical units
        x = track['x'].values * pixel_size
        y = track['y'].values * pixel_size
        frames = track['frame'].values
        
        # Calculate consecutive displacements
        for i in range(len(x) - 1):
            # Calculate displacement components
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            
            # Calculate jump distance
            jump_distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate jump angle
            jump_angle = np.arctan2(dy, dx)
            
            # Calculate velocity
            dt = (frames[i + 1] - frames[i]) * frame_interval
            velocity = jump_distance / dt if dt > 0 else 0
            
            jump_results.append({
                'track_id': track_id,
                'frame': frames[i + 1],  # Frame of end position
                'jump_distance': jump_distance,
                'jump_angle': jump_angle,
                'velocity': velocity,
                'x_displacement': dx,
                'y_displacement': dy
            })
    
    return pd.DataFrame(jump_results)


def analyze_jump_distribution(jump_distances: np.ndarray, 
                            method: str = 'rayleigh',
                            temperature: float = 300.0,
                            viscosity: float = 1e-3) -> Dict[str, Any]:
    """
    Analyze the distribution of jump distances to characterize motion.
    
    Parameters
    ----------
    jump_distances : np.ndarray
        Array of jump distances
    method : str, default='rayleigh'
        Distribution fitting method ('rayleigh', 'exponential', 'gamma', 'lognormal')
    temperature : float, default=300.0
        Temperature in Kelvin (for theoretical diffusion coefficient)
    viscosity : float, default=1e-3
        Viscosity in Pa·s (for theoretical diffusion coefficient)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing distribution parameters, goodness of fit, and derived metrics
    """
    if len(jump_distances) == 0:
        return {'error': 'No jump distances provided'}
    
    # Remove zero and negative values
    valid_jumps = jump_distances[jump_distances > 0]
    
    if len(valid_jumps) == 0:
        return {'error': 'No valid jump distances'}
    
    results = {
        'n_jumps': len(valid_jumps),
        'mean_jump': np.mean(valid_jumps),
        'std_jump': np.std(valid_jumps),
        'median_jump': np.median(valid_jumps),
        'max_jump': np.max(valid_jumps),
        'min_jump': np.min(valid_jumps)
    }
    
    # Fit different distributions
    try:
        if method == 'rayleigh':
            # Rayleigh distribution for 2D diffusion
            loc, scale = stats.rayleigh.fit(valid_jumps, floc=0)
            results['rayleigh_scale'] = scale
            results['rayleigh_loc'] = loc
            
            # Calculate diffusion coefficient from Rayleigh scale parameter
            # For 2D diffusion: <r²> = 4Dt, scale = sqrt(2Dt)
            D_rayleigh = scale**2 / 2  # Diffusion coefficient in μm²/frame
            results['diffusion_coeff_rayleigh'] = D_rayleigh
            
            # Goodness of fit
            ks_stat, p_value = stats.kstest(valid_jumps, 
                                          lambda x: stats.rayleigh.cdf(x, loc=loc, scale=scale))
            results['rayleigh_ks_stat'] = ks_stat
            results['rayleigh_p_value'] = p_value
            
        elif method == 'exponential':
            # Exponential distribution
            loc, scale = stats.expon.fit(valid_jumps, floc=0)
            results['exponential_scale'] = scale
            results['exponential_loc'] = loc
            
            # Goodness of fit
            ks_stat, p_value = stats.kstest(valid_jumps, 
                                          lambda x: stats.expon.cdf(x, loc=loc, scale=scale))
            results['exponential_ks_stat'] = ks_stat
            results['exponential_p_value'] = p_value
            
        elif method == 'gamma':
            # Gamma distribution
            shape, loc, scale = stats.gamma.fit(valid_jumps, floc=0)
            results['gamma_shape'] = shape
            results['gamma_scale'] = scale
            results['gamma_loc'] = loc
            
            # Goodness of fit
            ks_stat, p_value = stats.kstest(valid_jumps, 
                                          lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale))
            results['gamma_ks_stat'] = ks_stat
            results['gamma_p_value'] = p_value
            
        elif method == 'lognormal':
            # Log-normal distribution
            shape, loc, scale = stats.lognorm.fit(valid_jumps, floc=0)
            results['lognormal_shape'] = shape
            results['lognormal_scale'] = scale
            results['lognormal_loc'] = loc
            
            # Goodness of fit
            ks_stat, p_value = stats.kstest(valid_jumps, 
                                          lambda x: stats.lognorm.cdf(x, shape, loc=loc, scale=scale))
            results['lognormal_ks_stat'] = ks_stat
            results['lognormal_p_value'] = p_value
            
    except Exception as e:
        results['fit_error'] = str(e)
    
    return results


def classify_motion_from_jumps(jump_data: pd.DataFrame, 
                             percentile_threshold: float = 90.0,
                             confinement_ratio: float = 0.1) -> Dict[str, Any]:
    """
    Classify particle motion based on jump distance patterns.
    
    Parameters
    ----------
    jump_data : pd.DataFrame
        DataFrame from calculate_jump_distances
    percentile_threshold : float, default=90.0
        Percentile threshold for identifying large jumps
    confinement_ratio : float, default=0.1
        Ratio threshold for identifying confined motion
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing motion classification results
    """
    if jump_data.empty:
        return {'error': 'No jump data provided'}
    
    results = {}
    
    # Analyze each track separately
    track_classifications = []
    
    for track_id, track_jumps in jump_data.groupby('track_id'):
        jumps = track_jumps['jump_distance'].values
        
        if len(jumps) < 3:
            continue
        
        # Calculate basic statistics
        mean_jump = np.mean(jumps)
        std_jump = np.std(jumps)
        max_jump = np.max(jumps)
        
        # Calculate coefficient of variation
        cv = std_jump / mean_jump if mean_jump > 0 else np.inf
        
        # Identify large jumps
        threshold = np.percentile(jumps, percentile_threshold)
        large_jumps = jumps[jumps > threshold]
        large_jump_fraction = len(large_jumps) / len(jumps)
        
        # Calculate confinement ratio (ratio of largest to smallest jump)
        confinement_ratio_actual = np.min(jumps) / np.max(jumps) if np.max(jumps) > 0 else 0
        
        # Motion classification
        if confinement_ratio_actual > confinement_ratio and cv < 0.5:
            motion_type = 'confined'
        elif large_jump_fraction > 0.2 and cv > 1.0:
            motion_type = 'directed'
        elif 0.5 <= cv <= 1.0:
            motion_type = 'free_diffusion'
        elif cv > 1.5:
            motion_type = 'superdiffusive'
        else:
            motion_type = 'subdiffusive'
        
        track_classifications.append({
            'track_id': track_id,
            'motion_type': motion_type,
            'mean_jump': mean_jump,
            'cv_jump': cv,
            'large_jump_fraction': large_jump_fraction,
            'confinement_ratio': confinement_ratio_actual,
            'n_jumps': len(jumps)
        })
    
    results['track_classifications'] = pd.DataFrame(track_classifications)
    
    # Overall statistics
    if track_classifications:
        motion_types = [t['motion_type'] for t in track_classifications]
        results['motion_type_counts'] = pd.Series(motion_types).value_counts().to_dict()
        results['dominant_motion_type'] = max(results['motion_type_counts'], 
                                            key=results['motion_type_counts'].get)
    
    return results


def analyze_jump_autocorrelation(jump_data: pd.DataFrame, 
                                max_lag: int = 10) -> Dict[str, Any]:
    """
    Analyze autocorrelation in jump distances to detect memory effects.
    
    Parameters
    ----------
    jump_data : pd.DataFrame
        DataFrame from calculate_jump_distances
    max_lag : int, default=10
        Maximum lag for autocorrelation analysis
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing autocorrelation results
    """
    if jump_data.empty:
        return {'error': 'No jump data provided'}
    
    results = {}
    track_autocorrelations = []
    
    for track_id, track_jumps in jump_data.groupby('track_id'):
        jumps = track_jumps['jump_distance'].values
        
        if len(jumps) <= max_lag:
            continue
        
        # Calculate autocorrelation
        autocorr = []
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr.append(1.0)
            else:
                if len(jumps) > lag:
                    corr = np.corrcoef(jumps[:-lag], jumps[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0.0)
                else:
                    autocorr.append(0.0)
        
        # Find correlation time (lag where autocorr drops below 1/e)
        corr_time = None
        threshold = 1.0 / np.e
        for lag, corr in enumerate(autocorr[1:], 1):
            if corr < threshold:
                corr_time = lag
                break
        
        track_autocorrelations.append({
            'track_id': track_id,
            'autocorrelation': autocorr,
            'correlation_time': corr_time,
            'memory_strength': autocorr[1] if len(autocorr) > 1 else 0.0
        })
    
    results['track_autocorrelations'] = track_autocorrelations
    
    # Calculate average autocorrelation
    if track_autocorrelations:
        all_autocorr = [t['autocorrelation'] for t in track_autocorrelations]
        min_length = min(len(ac) for ac in all_autocorr)
        
        # Truncate to minimum length
        truncated_autocorr = [ac[:min_length] for ac in all_autocorr]
        
        # Calculate mean and std
        mean_autocorr = np.mean(truncated_autocorr, axis=0)
        std_autocorr = np.std(truncated_autocorr, axis=0)
        
        results['mean_autocorrelation'] = mean_autocorr
        results['std_autocorrelation'] = std_autocorr
        results['lags'] = list(range(min_length))
        
        # Overall memory strength
        results['overall_memory_strength'] = mean_autocorr[1] if len(mean_autocorr) > 1 else 0.0
    
    return results


def generate_jump_distance_report(tracks_df: pd.DataFrame, 
                                pixel_size: float = 1.0,
                                frame_interval: float = 1.0) -> Dict[str, Any]:
    """
    Generate a comprehensive jump distance analysis report.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data
    pixel_size : float, default=1.0
        Pixel size in micrometers
    frame_interval : float, default=1.0
        Frame interval in seconds
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis report
    """
    report = {'analysis_type': 'jump_distance_analysis'}
    
    try:
        # Calculate jump distances
        jump_data = calculate_jump_distances(tracks_df, pixel_size, frame_interval)
        
        if jump_data.empty:
            report['error'] = 'No valid jump distances calculated'
            return report
        
        report['jump_data'] = jump_data
        
        # Analyze jump distribution
        jumps = jump_data['jump_distance'].values
        distribution_analysis = analyze_jump_distribution(jumps)
        report['distribution_analysis'] = distribution_analysis
        
        # Motion classification
        motion_classification = classify_motion_from_jumps(jump_data)
        report['motion_classification'] = motion_classification
        
        # Autocorrelation analysis
        autocorr_analysis = analyze_jump_autocorrelation(jump_data)
        report['autocorrelation_analysis'] = autocorr_analysis
        
        # Summary statistics
        report['summary'] = {
            'total_jumps': len(jumps),
            'unique_tracks': jump_data['track_id'].nunique(),
            'mean_jump_distance': np.mean(jumps),
            'median_jump_distance': np.median(jumps),
            'std_jump_distance': np.std(jumps),
            'range_jump_distance': np.ptp(jumps)
        }
        
    except Exception as e:
        report['error'] = f'Analysis failed: {str(e)}'
    
    return report