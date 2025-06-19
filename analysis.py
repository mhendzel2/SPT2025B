"""
Analysis modules for the SPT Analysis application.
This file contains all analytical methods for processing tracking data:
- Diffusion analysis (standard, anomalous, confined)
- Motion analysis (velocity, directional persistence, motion types)
- Clustering analysis (spatial and feature-based)
- Dwell time analysis (binding/unbinding events)
- Gel structure analysis (mesh characteristics)
- Diffusion population analysis (subpopulation extraction)
- Crowding analysis (local density estimation)
- Active transport analysis (directed motion detection)
- Boundary crossing analysis (detection and statistics)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import scipy.optimize as optimize
from scipy.stats import norm, linregress, pearsonr
from scipy import spatial
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.cluster.hierarchy import fcluster, linkage


# --- Diffusion Analysis ---

def calculate_msd(tracks_df: pd.DataFrame, max_lag: int = 20, pixel_size: float = 1.0, 
                 frame_interval: float = 1.0, min_track_length: int = 5) -> pd.DataFrame:
    """
    Calculate mean squared displacement (MSD) for all tracks.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format with track_id, frame, x, y columns
    max_lag : int
        Maximum lag time (in frames) for MSD calculation
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    min_track_length : int
        Minimum track length to include in analysis
        
    Returns
    -------
    pd.DataFrame
        DataFrame with MSD values for each track at different lag times (columns: track_id, lag_time, msd, n_points)
    """    
    # Parameter validation
    if tracks_df.empty:
        raise ValueError("tracks_df cannot be empty")
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    
    if min_track_length < 2:
        raise ValueError("min_track_length must be at least 2")

    # Group by track_id
    grouped = tracks_df.groupby('track_id')
    
    # Initialize results dictionary
    msd_results = {'track_id': [], 'lag_time': [], 'msd': [], 'n_points': []}
    
    for track_id, track_data in grouped:
        # Skip short tracks
        if len(track_data) < min_track_length:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions
        x = track_data['x'].values.astype(float) * pixel_size
        y = track_data['y'].values.astype(float) * pixel_size
        frames = track_data['frame'].values.astype(float)
        
        # Calculate MSD for each lag time
        for lag in range(1, min(max_lag + 1, len(track_data))):
            # Calculate squared displacements
            sd_list = []
            lag_time_list = []
            
            n_points = len(track_data) - lag
            if n_points > 0:
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                sd = dx**2 + dy**2
                dt = (frames[lag:] - frames[:-lag]) * frame_interval
                
                sd_list.extend(sd)
                lag_time_list.extend(dt)
            
            if sd_list:
                # Store results
                msd_results['track_id'].append(track_id)
                # Use actual time difference in seconds
                mean_lag_time = np.mean(lag_time_list)
                msd_results['lag_time'].append(mean_lag_time)
                msd_results['msd'].append(np.mean(sd_list))
                msd_results['n_points'].append(len(sd_list))
    
    # Convert to DataFrame
    msd_df = pd.DataFrame(msd_results)
    
    return msd_df

def analyze_diffusion(tracks_df: pd.DataFrame, max_lag: int = 20, pixel_size: float = 1.0, 
                     frame_interval: float = 1.0, min_track_length: int = 5, 
                     fit_method: str = 'linear', analyze_anomalous: bool = True, 
                     check_confinement: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive diffusion analysis on track data.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    max_lag : int
        Maximum lag time for MSD calculation
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    min_track_length : int
        Minimum track length to include in analysis
    fit_method : str
        Method for fitting MSD curves ('linear', 'weighted', 'nonlinear')
    analyze_anomalous : bool
        Whether to analyze anomalous diffusion
    check_confinement : bool
        Whether to check for confined diffusion
        
    Returns
    -------
    dict
        Dictionary containing diffusion analysis results
    """
    # Calculate MSD
    msd_df = calculate_msd(
        tracks_df, 
        max_lag=max_lag, 
        pixel_size=pixel_size, 
        frame_interval=frame_interval, 
        min_track_length=min_track_length
    )
    
    if msd_df.empty:
        return {
            'success': False,
            'error': 'No tracks of sufficient length for analysis',
            'msd_data': msd_df
        }
    
    # Initialize results dict
    results = {
        'success': True,
        'msd_data': msd_df,
        'track_results': [],
        'ensemble_results': {}
    }
    
    # Analyze each track individually
    track_results = []
    
    for track_id, track_msd in msd_df.groupby('track_id'):
        # Sort by lag time
        track_msd = track_msd.sort_values('lag_time')
        
        # Extract lag times and MSD values
        lag_times = track_msd['lag_time'].values
        msd_values = track_msd['msd'].values
        
        # Initialize track result dict
        track_result = {'track_id': track_id}
        
        # Standard diffusion coefficient (short time)
        # Use only the first few points for initial diffusion coefficient
        short_lag_cutoff = min(4, len(lag_times))
        
        # Linear fit: MSD = 4*D*t (2D) or MSD = 6*D*t (3D)
        # Here we use 2D (4*D*t)
        if fit_method == 'linear':
            # Linear fit
            if short_lag_cutoff >= 2:
                slope, intercept, r_value, p_value, std_err = linregress(
                    lag_times[:short_lag_cutoff], 
                    msd_values[:short_lag_cutoff]
                )
                D_short = slope / 4  # µm²/s
                D_err = std_err / 4
            else:
                D_short = np.nan
                D_err = np.nan
                
        elif fit_method == 'weighted':
            if short_lag_cutoff >= 2:
                def linear_func_with_offset(t, D, offset):
                    return 4 * D * t + offset
                try:
                    # Weights inversely proportional to lag time (variance ~ t)
                    # So, sigma ~ sqrt(t) for curve_fit
                    sigma_vals = np.sqrt(lag_times[:short_lag_cutoff])
                    # Ensure sigma_vals are not zero
                    sigma_vals[sigma_vals == 0] = 1e-9
                    
                    popt, pcov = optimize.curve_fit(
                        linear_func_with_offset, 
                        lag_times[:short_lag_cutoff], 
                        msd_values[:short_lag_cutoff],
                        sigma=sigma_vals,
                        absolute_sigma=True
                    )
                    D_short = popt[0]  # µm²/s
                    D_err = np.sqrt(pcov[0, 0])
                except (RuntimeError, ValueError):
                    D_short = np.nan
                    D_err = np.nan
            else:
                D_short = np.nan
                D_err = np.nan
                
        elif fit_method == 'nonlinear':
            # Nonlinear fit for MSD = 4*D*t + offset
            if short_lag_cutoff >= 3:
                def msd_func(t, D, offset):
                    return 4 * D * t + offset
                
                try:
                    popt, pcov = optimize.curve_fit(
                        msd_func, 
                        lag_times[:short_lag_cutoff], 
                        msd_values[:short_lag_cutoff]
                    )
                    D_short = popt[0]  # µm²/s
                    D_err = np.sqrt(pcov[0, 0])
                except:
                    D_short = np.nan
                    D_err = np.nan
            else:
                D_short = np.nan
                D_err = np.nan
        else:
            D_short = np.nan
            D_err = np.nan
        
        # Store diffusion coefficient results
        track_result['diffusion_coefficient'] = D_short
        track_result['diffusion_coefficient_error'] = D_err
        
        # Analyze anomalous diffusion
        if analyze_anomalous and len(lag_times) >= 5:
            # Fit MSD = c * t^alpha using log-log
            # Ensure lag_times and msd_values are positive for log
            valid_indices = (lag_times > 0) & (msd_values > 0)
            log_lag_valid = np.log(lag_times[valid_indices])
            log_msd_valid = np.log(msd_values[valid_indices])
            
            if len(log_lag_valid) >= 2:  # Need at least 2 points for linregress
                try:
                    slope, intercept, r_value, p_value, std_err_slope = linregress(log_lag_valid, log_msd_valid)
                    
                    # Alpha is the slope in log-log space
                    alpha = slope
                    alpha_err = std_err_slope
                    
                    # Categorize diffusion type
                    diffusion_type = 'normal'
                    if alpha < 0.9:
                        diffusion_type = 'subdiffusive'
                    elif alpha > 1.1:
                        diffusion_type = 'superdiffusive'
                        
                    track_result['alpha'] = alpha
                    track_result['alpha_error'] = alpha_err
                    track_result['diffusion_type'] = diffusion_type
                except ValueError:
                    track_result['alpha'] = np.nan
                    track_result['alpha_error'] = np.nan
                    track_result['diffusion_type'] = 'unknown'
            else:
                track_result['alpha'] = np.nan
                track_result['alpha_error'] = np.nan
                track_result['diffusion_type'] = 'unknown'
        
        # Check for confined diffusion
        if check_confinement and len(lag_times) >= 8:
            # Look for plateau in MSD curve
            # Simple approach: check if MSD stops increasing with lag time
            
            # Calculate slope at different regions of the curve
            early_region = min(3, len(lag_times)-1)
            late_region = min(8, len(lag_times))
            
            early_slope, _, _, _, _ = linregress(
                lag_times[:early_region], 
                msd_values[:early_region]
            )
            
            late_slope, _, _, _, _ = linregress(
                lag_times[early_region:late_region], 
                msd_values[early_region:late_region]
            )
            
            # Check for significant decrease in slope
            if late_slope < 0.3 * early_slope:
                confined = True
                # Estimate confinement radius
                plateau_value = np.mean(msd_values[early_region:late_region])
                confinement_radius = np.sqrt(plateau_value / 4)  # Radius = sqrt(MSD/4) for 2D
            else:
                confined = False
                confinement_radius = np.nan
                
            track_result['confined'] = confined
            track_result['confinement_radius'] = confinement_radius
            
        track_results.append(track_result)
    
    # Combine track results
    results['track_results'] = pd.DataFrame(track_results)
    
    # Ensemble statistics
    if not results['track_results'].empty:
        # Ensemble averages
        results['ensemble_results'] = {
            'mean_diffusion_coefficient': results['track_results']['diffusion_coefficient'].mean(),
            'median_diffusion_coefficient': results['track_results']['diffusion_coefficient'].median(),
            'std_diffusion_coefficient': results['track_results']['diffusion_coefficient'].std(),
            'n_tracks': len(results['track_results'])
        }
        
        if analyze_anomalous:
            results['ensemble_results']['mean_alpha'] = results['track_results']['alpha'].mean()
            results['ensemble_results']['median_alpha'] = results['track_results']['alpha'].median()
            results['ensemble_results']['std_alpha'] = results['track_results']['alpha'].std()
            
            # Count diffusion types
            type_counts = results['track_results']['diffusion_type'].value_counts()
            for diff_type in ['normal', 'subdiffusive', 'superdiffusive']:
                if diff_type in type_counts:
                    results['ensemble_results'][f'{diff_type}_count'] = type_counts[diff_type]
                    results['ensemble_results'][f'{diff_type}_fraction'] = type_counts[diff_type] / type_counts.sum()
                else:
                    results['ensemble_results'][f'{diff_type}_count'] = 0
                    results['ensemble_results'][f'{diff_type}_fraction'] = 0.0
        
        if check_confinement:
            confined_tracks = results['track_results'][results['track_results']['confined'] == True]
            results['ensemble_results']['confined_count'] = len(confined_tracks)
            results['ensemble_results']['confined_fraction'] = len(confined_tracks) / len(results['track_results'])
            
            if len(confined_tracks) > 0:
                results['ensemble_results']['mean_confinement_radius'] = confined_tracks['confinement_radius'].mean()
                results['ensemble_results']['median_confinement_radius'] = confined_tracks['confinement_radius'].median()
    
    return results


# --- Motion Analysis ---

def analyze_motion(tracks_df: pd.DataFrame, window_size: int = 5, 
                  analyze_velocity_autocorr: bool = True, analyze_persistence: bool = True,
                  motion_classification: str = 'basic', min_track_length: int = 10, 
                  pixel_size: float = 1.0, frame_interval: float = 1.0) -> Dict[str, Any]:
    """
    Analyze motion characteristics of tracks.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    window_size : int
        Window size for calculating local properties
    analyze_velocity_autocorr : bool
        Whether to calculate velocity autocorrelation
    analyze_persistence : bool
        Whether to analyze directional persistence
    motion_classification : str
        Method for classifying motion ('none', 'basic', 'advanced')
    min_track_length : int
        Minimum track length to include in analysis
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
        
    Returns
    -------
    dict
        Dictionary containing motion analysis results
    """
    # Group by track_id
    grouped = tracks_df.groupby('track_id')
    
    # Initialize results
    track_results = []
    
    for track_id, track_data in grouped:
        # Skip short tracks
        if len(track_data) < min_track_length:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions
        x = track_data['x'].values.astype(float) * pixel_size
        y = track_data['y'].values.astype(float) * pixel_size
        frames = track_data['frame'].values.astype(float)
        
        # Calculate displacements
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(frames) * frame_interval
        
        # Calculate velocities (protect against division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            vx = np.where(dt != 0, dx / dt, 0)
            vy = np.where(dt != 0, dy / dt, 0)
            speeds = np.sqrt(vx**2 + vy**2)
        
        # Calculate angles between consecutive steps
        angles = np.arctan2(vy, vx)
        angle_changes = np.diff(angles)
        # Normalize to [-pi, pi]
        angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
        
        # Initialize track result
        track_result = {
            'track_id': track_id,
            'mean_speed': np.mean(speeds),
            'median_speed': np.median(speeds),
            'max_speed': np.max(speeds),
            'speed_std': np.std(speeds),
            'mean_abs_angle_change': np.mean(np.abs(angle_changes)) if len(angle_changes) > 0 else np.nan,
            'track_length': len(track_data),
            'duration': (frames[-1] - frames[0]) * frame_interval
        }
        
        # Calculate local properties using rolling window
        if len(track_data) >= window_size:
            local_speeds = []
            local_angle_changes = []
            
            for i in range(len(track_data) - window_size + 1):
                window_x = x[i:i+window_size]
                window_y = y[i:i+window_size]
                window_frames = frames[i:i+window_size]
                
                # Calculate local displacements
                local_dx = np.diff(window_x)
                local_dy = np.diff(window_y)
                local_dt = np.diff(window_frames) * frame_interval
                
                # Calculate local velocities
                local_vx = local_dx / local_dt
                local_vy = local_dy / local_dt
                local_speeds.append(np.mean(np.sqrt(local_vx**2 + local_vy**2)))
                
                # Calculate local angles
                local_angles = np.arctan2(local_vy, local_vx)
                local_angle_diff = np.diff(local_angles)
                local_angle_diff = (local_angle_diff + np.pi) % (2 * np.pi) - np.pi
                
                if len(local_angle_diff) > 0:
                    local_angle_changes.append(np.mean(np.abs(local_angle_diff)))
            
            track_result['max_local_speed'] = np.max(local_speeds) if local_speeds else np.nan
            track_result['min_local_speed'] = np.min(local_speeds) if local_speeds else np.nan
            track_result['local_speed_variation'] = np.std(local_speeds) / np.mean(local_speeds) if local_speeds and np.mean(local_speeds) > 0 else np.nan
        
        # Analyze directional persistence
        if analyze_persistence and len(track_data) >= 3:
            # Calculate net displacement
            net_displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            total_path_length = np.sum(np.sqrt(dx**2 + dy**2))
            
            # Straightness ratio (0-1)
            straightness = net_displacement / total_path_length if total_path_length > 0 else 0
            
            # Directional correlation
            # Calculate autocorrelation of direction changes
            if len(angle_changes) >= 3:
                # Use absolute angle changes
                abs_angle_changes = np.abs(angle_changes)
                
                # Calculate correlation coefficient
                direction_corr = np.corrcoef(abs_angle_changes[:-1], abs_angle_changes[1:])[0, 1]
            else:
                direction_corr = np.nan
                
            track_result['straightness'] = straightness
            track_result['direction_correlation'] = direction_corr
            track_result['net_displacement'] = net_displacement
            track_result['total_path_length'] = total_path_length
        
        # Calculate velocity autocorrelation
        if analyze_velocity_autocorr and len(vx) >= 5:
            # Calculate velocity autocorrelation
            max_lag = min(10, len(vx) - 1)
            vel_autocorr = []
            
            for lag in range(max_lag + 1):
                if lag == 0:
                    vel_autocorr.append(1.0)  # Normalized autocorrelation at lag 0
                else:
                    # Calculate correlation between v(t) and v(t+lag)
                    corr_x = np.corrcoef(vx[:-lag], vx[lag:])[0, 1]
                    corr_y = np.corrcoef(vy[:-lag], vy[lag:])[0, 1]
                    # Average of x and y components
                    avg_corr = (corr_x + corr_y) / 2
                    vel_autocorr.append(avg_corr)
            
            # Calculate correlation time (lag where autocorr drops below 1/e)
            corr_threshold = 1/np.e
            for lag, corr in enumerate(vel_autocorr):
                if corr < corr_threshold:
                    corr_time = lag * frame_interval
                    break
            else:
                corr_time = max_lag * frame_interval
                
            track_result['velocity_autocorr'] = vel_autocorr
            track_result['correlation_time'] = corr_time
        
        # Motion classification
        if motion_classification != 'none' and len(track_data) >= min_track_length:
            if motion_classification == 'basic':
                # Basic classification based on straightness and speed variation
                if 'straightness' in track_result:
                    if track_result['straightness'] > 0.8:
                        motion_type = 'directed'
                    elif track_result['straightness'] < 0.3:
                        motion_type = 'confined'
                    else:
                        motion_type = 'diffusive'
                        
                    track_result['motion_type'] = motion_type
            
            elif motion_classification == 'advanced':
                # Advanced classification using multiple metrics
                # 1. Calculate MSD
                msd_values = []
                max_lag = min(10, len(track_data) - 1)
                
                for lag in range(1, max_lag + 1):
                    # Calculate squared displacements
                    sd_list = []
                    
                    for i in range(len(track_data) - lag):
                        dx = x[i + lag] - x[i]
                        dy = y[i + lag] - y[i]
                        sd = dx**2 + dy**2
                        sd_list.append(sd)
                    
                    if sd_list:
                        msd_values.append(np.mean(sd_list))
                
                # Fit MSD curve
                if len(msd_values) >= 3:
                    # Log-log fit to get alpha
                    lag_times = np.arange(1, len(msd_values) + 1) * frame_interval
                    log_lag = np.log(lag_times)
                    log_msd = np.log(msd_values)
                    
                    try:
                        slope, _, r_value, _, _ = linregress(log_lag, log_msd)
                        
                        # Determine motion type based on combined metrics
                        alpha = slope
                        straightness = track_result.get('straightness', 0)
                        
                        if alpha > 1.7 and straightness > 0.6:
                            motion_type = 'directed'
                        elif alpha < 0.9 and straightness < 0.4:
                            motion_type = 'confined'
                        elif 0.9 <= alpha <= 1.1:
                            motion_type = 'diffusive'
                        elif alpha > 1.1 and alpha <= 1.7:
                            motion_type = 'superdiffusive'
                        elif alpha < 0.9:
                            motion_type = 'subdiffusive'
                        else:
                            motion_type = 'complex'
                            
                        track_result['alpha'] = alpha
                        track_result['motion_type'] = motion_type
                    except:
                        track_result['motion_type'] = 'unknown'
        
        track_results.append(track_result)
    
    # Combine results
    results = {
        'success': True if track_results else False,
        'track_results': pd.DataFrame(track_results) if track_results else pd.DataFrame(),
        'ensemble_results': {}
    }
    
    # Ensemble statistics
    if track_results:
        results_df = pd.DataFrame(track_results)
        
        # Calculate ensemble statistics
        results['ensemble_results'] = {
            'mean_speed': results_df['mean_speed'].mean(),
            'median_speed': results_df['mean_speed'].median(),
            'speed_std': results_df['mean_speed'].std(),
            'n_tracks': len(results_df)
        }
        
        if 'straightness' in results_df.columns:
            results['ensemble_results']['mean_straightness'] = results_df['straightness'].mean()
            results['ensemble_results']['median_straightness'] = results_df['straightness'].median()
        
        if 'motion_type' in results_df.columns:
            type_counts = results_df['motion_type'].value_counts()
            for motion_type in type_counts.index:
                results['ensemble_results'][f'{motion_type}_count'] = type_counts[motion_type]
                results['ensemble_results'][f'{motion_type}_fraction'] = type_counts[motion_type] / type_counts.sum()
    
    return results


# --- Clustering Analysis ---

def analyze_clustering(tracks_df: pd.DataFrame, 
                      method: str = 'DBSCAN', 
                      epsilon: float = 0.5, 
                      min_samples: int = 3,
                      track_clusters: bool = True, 
                      analyze_dynamics: bool = True,
                      pixel_size: float = 1.0) -> Dict[str, Any]:
    """
    Perform spatial clustering analysis on particles.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    method : str
        Clustering method ('DBSCAN', 'OPTICS', 'Hierarchical', 'Density-based')
    epsilon : float
        Maximum distance between points in a cluster (in µm)
    min_samples : int
        Minimum number of points to form a cluster
    track_clusters : bool
        Whether to track clusters over time
    analyze_dynamics : bool
        Whether to analyze cluster dynamics
    pixel_size : float
        Pixel size in micrometers
        
    Returns
    -------
    dict
        Dictionary containing clustering analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # Get unique frames
    frames = sorted(tracks_df_um['frame'].unique())
    
    # Initialize results
    frame_results = []
    all_clusters = []
    
    # Analyze each frame
    for frame in frames:
        # Get particles in this frame
        frame_data = tracks_df_um[tracks_df_um['frame'] == frame]
        
        # Create default empty structures
        cluster_stats = []
        
        if len(frame_data) < min_samples:
            # Not enough points for clustering, but still record frame result with zero clusters
            frame_result = {
                'frame': frame,
                'n_points': len(frame_data),
                'n_clusters': 0,
                'clustered_fraction': 0,
                'mean_cluster_size': 0,
                'cluster_stats': pd.DataFrame(cluster_stats)
            }
            frame_results.append(frame_result)
            continue
            
        # Get coordinates
        coords = frame_data[['x', 'y']].values
        
        # Apply clustering based on method
        if method == 'DBSCAN':
            clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
            labels = clustering.fit_predict(coords)
            
        elif method == 'OPTICS':
            clustering = OPTICS(min_samples=min_samples, max_eps=epsilon)
            labels = clustering.fit_predict(coords)
            
        elif method == 'Hierarchical':
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist
            
            # Perform hierarchical clustering
            Z = linkage(coords, method='ward')
            
            # Use distance threshold for noise detection
            distances = pdist(coords)
            if len(distances) > 0:
                # Use 75th percentile of distances as threshold
                threshold = np.percentile(distances, 75)
                labels = fcluster(Z, threshold, criterion='distance') - 1
            else:
                labels = np.array([0] * len(coords))
            
        elif method == 'Density-based':
            # Custom density-based clustering
            # Use KD-tree to find neighbors within epsilon
            tree = KDTree(coords)
            neighbors = tree.query_radius(coords, r=epsilon)
            
            # Initialize labels
            labels = np.zeros(len(coords), dtype=int) - 1
            current_label = 0
            
            # Assign labels
            for i, point_neighbors in enumerate(neighbors):
                if len(point_neighbors) >= min_samples:
                    # This point is a core point
                    if labels[i] == -1:
                        # Assign new label
                        labels[i] = current_label
                        
                        # Expand cluster
                        to_expand = list(point_neighbors)
                        while to_expand:
                            j = to_expand.pop(0)
                            if labels[j] == -1:
                                labels[j] = current_label
                                if len(neighbors[j]) >= min_samples:
                                    # Add new neighbors to expand
                                    for nbr in neighbors[j]:
                                        if labels[nbr] == -1 and nbr not in to_expand:
                                            to_expand.append(nbr)
                        
                        current_label += 1
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Count clusters (exclude noise points with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Process clustering results
        if n_clusters > 0:
            # Add cluster labels to the data
            frame_data = frame_data.copy()
            frame_data['cluster_label'] = labels
            
            # Get cluster statistics
            cluster_stats = []
            
            for label in range(n_clusters):
                cluster_points = frame_data[frame_data['cluster_label'] == label]
                
                if len(cluster_points) > 0:
                    # Calculate cluster centroid
                    centroid_x = cluster_points['x'].mean()
                    centroid_y = cluster_points['y'].mean()
                    
                    # Calculate cluster radius (standard deviation of distances from centroid)
                    dx = cluster_points['x'] - centroid_x
                    dy = cluster_points['y'] - centroid_y
                    distances = np.sqrt(dx**2 + dy**2)
                    radius = distances.std()
                    
                    # Get track IDs in this cluster
                    track_ids = cluster_points['track_id'].unique()
                    
                    cluster_stats.append({
                        'frame': frame,
                        'cluster_id': label,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'n_points': len(cluster_points),
                        'n_tracks': len(track_ids),
                        'radius': radius,
                        'density': len(cluster_points) / (np.pi * radius**2) if radius > 0 else np.nan,
                        'track_ids': track_ids
                    })
                    
                    # Store all points in the cluster
                    all_clusters.append({
                        'frame': frame,
                        'cluster_id': label,
                        'points': cluster_points
                    })
            
            # Calculate frame statistics
            noise_points = frame_data[frame_data['cluster_label'] == -1]
            clustered_points = frame_data[frame_data['cluster_label'] != -1]
            
            frame_result = {
                'frame': frame,
                'n_points': len(frame_data),
                'n_clusters': n_clusters,
                'clustered_fraction': len(clustered_points) / len(frame_data) if len(frame_data) > 0 else 0,
                'mean_cluster_size': clustered_points['cluster_label'].value_counts().mean() if len(clustered_points) > 0 else 0,
                'cluster_stats': pd.DataFrame(cluster_stats) if cluster_stats else pd.DataFrame()
            }
            
            frame_results.append(frame_result)
    
    # Track clusters over time
    cluster_tracks = []
    
    if track_clusters and len(frame_results) > 1:
        # Use a simple linking approach based on proximity of cluster centroids
        prev_frame_clusters = None
        next_cluster_id = 0
        
        for frame_result in frame_results:
            frame = frame_result['frame']
            cluster_stats = frame_result['cluster_stats']
            
            if cluster_stats.empty:
                continue
                
            if prev_frame_clusters is None:
                # First frame with clusters
                # Assign initial cluster track IDs
                cluster_stats['cluster_track_id'] = range(next_cluster_id, next_cluster_id + len(cluster_stats))
                next_cluster_id += len(cluster_stats)
                
                prev_frame_clusters = cluster_stats
            else:
                # Link clusters between frames
                prev_centroids = prev_frame_clusters[['centroid_x', 'centroid_y']].values
                curr_centroids = cluster_stats[['centroid_x', 'centroid_y']].values
                
                # Calculate distances between all centroids
                distances = distance.cdist(prev_centroids, curr_centroids)
                
                # Link clusters based on proximity (simple nearest neighbor)
                cluster_track_ids = np.full(len(cluster_stats), -1)
                used_prev = set()
                
                # Sort distances from smallest to largest
                dist_indices = np.dstack(np.unravel_index(np.argsort(distances.ravel()), distances.shape))[0]
                
                for idx_pair in dist_indices:
                    i, j = idx_pair[0], idx_pair[1]
                    if (j < len(prev_frame_clusters) and i < len(cluster_stats) and 
                        j not in used_prev and cluster_track_ids[i] == -1):
                        if distances[i, j] <= 2 * epsilon:  # Use 2*epsilon as max linking distance
                            # Link this cluster to the track
                            cluster_track_ids[i] = prev_frame_clusters.iloc[j]['cluster_track_id']
                            used_prev.add(j)
                
                # Assign new track IDs to unlinked clusters
                for i in range(len(cluster_stats)):
                    if cluster_track_ids[i] == -1:
                        cluster_track_ids[i] = next_cluster_id
                        next_cluster_id += 1
                
                cluster_stats['cluster_track_id'] = cluster_track_ids
                prev_frame_clusters = cluster_stats
            
            # Store cluster tracks for this frame
            for _, cluster in cluster_stats.iterrows():
                cluster_tracks.append({
                    'frame': frame,
                    'cluster_id': cluster['cluster_id'],
                    'cluster_track_id': cluster['cluster_track_id'],
                    'centroid_x': cluster['centroid_x'],
                    'centroid_y': cluster['centroid_y'],
                    'n_points': cluster['n_points'],
                    'radius': cluster['radius']
                })
    
    # Analyze cluster dynamics
    cluster_dynamics = []
    
    if analyze_dynamics and cluster_tracks:
        # Group by cluster track ID
        cluster_tracks_df = pd.DataFrame(cluster_tracks)
        grouped = cluster_tracks_df.groupby('cluster_track_id')
        
        for track_id, track_data in grouped:
            # Skip short tracks
            if len(track_data) < 2:
                continue
                
            # Sort by frame
            track_data = track_data.sort_values('frame')
            
            # Calculate changes over time
            frames = track_data['frame'].values.astype(float)
            centroids_x = track_data['centroid_x'].values
            centroids_y = track_data['centroid_y'].values
            n_points = track_data['n_points'].values
            radii = track_data['radius'].values
            
            # Calculate displacements
            dx = np.diff(centroids_x)
            dy = np.diff(centroids_y)
            displacements = np.sqrt(dx**2 + dy**2)
            
            # Calculate size changes
            size_changes = np.diff(n_points)
            radius_changes = np.diff(radii)
            
            # Calculate metrics
            mean_displacement = np.mean(displacements)
            mean_size_change = np.mean(size_changes)
            mean_radius_change = np.mean(radius_changes)
            
            dynamics = {
                'cluster_track_id': track_id,
                'duration': len(track_data),
                'mean_displacement': mean_displacement,
                'mean_size_change': mean_size_change,
                'mean_radius_change': mean_radius_change,
                'max_displacement': np.max(displacements) if len(displacements) > 0 else 0,
                'stable': np.all(np.abs(size_changes) <= 2) and np.all(displacements <= epsilon)
            }
            
            cluster_dynamics.append(dynamics)
    
    # Compile final results
    results = {
        'success': True,  # Always return success if analysis completed
        'frames_analyzed': len(frame_results),
        'frame_results': frame_results,
        'cluster_tracks': pd.DataFrame(cluster_tracks) if cluster_tracks else pd.DataFrame(columns=['frame', 'cluster_id', 'cluster_track_id', 'centroid_x', 'centroid_y', 'n_points', 'radius']),
        'cluster_dynamics': pd.DataFrame(cluster_dynamics) if cluster_dynamics else pd.DataFrame(columns=['cluster_track_id', 'start_frame', 'end_frame', 'duration', 'mean_displacement', 'mean_size_change', 'mean_radius_change', 'max_displacement', 'stable'])
    }
    
    # Calculate ensemble statistics
    # Always add ensemble_results, even if no clusters found
    n_clusters = [fr['n_clusters'] for fr in frame_results] if frame_results else [0]
    clustered_fractions = [fr['clustered_fraction'] for fr in frame_results] if frame_results else [0]
    mean_cluster_sizes = [fr['mean_cluster_size'] for fr in frame_results] if frame_results else [0]
    
    results['ensemble_results'] = {
        'mean_n_clusters': np.mean(n_clusters),
        'max_n_clusters': np.max(n_clusters),
        'mean_clustered_fraction': np.mean(clustered_fractions),
        'mean_cluster_size': np.mean(mean_cluster_sizes),
        'n_frames_with_clusters': sum(1 for fr in frame_results if fr['n_clusters'] > 0) if frame_results else 0,
        'n_frames_analyzed': len(frames),
        'n_tracks_analyzed': len(tracks_df['track_id'].unique()),
        'clustering_threshold': epsilon,
        'min_points_per_cluster': min_samples,
        'clustering_method': method
    }
    
    if cluster_dynamics:
        # Extract cluster track statistics
        cluster_dynamics_df = pd.DataFrame(cluster_dynamics)
        
        if not cluster_dynamics_df.empty:
            results['ensemble_results']['n_cluster_tracks'] = len(cluster_dynamics_df)
            results['ensemble_results']['mean_cluster_duration'] = cluster_dynamics_df['duration'].mean()
            results['ensemble_results']['n_stable_clusters'] = sum(cluster_dynamics_df['stable']) 
            results['ensemble_results']['stable_cluster_fraction'] = sum(cluster_dynamics_df['stable']) / len(cluster_dynamics_df)
        else:
            results['ensemble_results']['n_cluster_tracks'] = 0
            results['ensemble_results']['mean_cluster_duration'] = 0
            results['ensemble_results']['n_stable_clusters'] = 0
            results['ensemble_results']['stable_cluster_fraction'] = 0
    
    return results


# --- Dwell Time Analysis ---

def analyze_dwell_time(tracks_df: pd.DataFrame, 
                      regions=None, 
                      threshold_distance: float = 0.5,
                      min_dwell_frames: int = 3,
                      pixel_size: float = 1.0,
                      frame_interval: float = 1.0) -> Dict[str, Any]:
    """
    Analyze dwell times (binding/unbinding events).
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    regions : list of dict, optional
        List of regions to analyze (each with 'x', 'y', 'radius')
        If None, analysis is based on track motion
    threshold_distance : float
        Maximum distance to consider a particle as dwelling (µm)
    min_dwell_frames : int
        Minimum number of frames to consider a dwell event
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
        
    Returns
    -------
    dict
        Dictionary containing dwell time analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # Initialize results
    dwell_events = []
    track_results = []
    
    # Group by track_id
    grouped = tracks_df_um.groupby('track_id')
    
    for track_id, track_data in grouped:
        # Skip very short tracks
        if len(track_data) < min_dwell_frames:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions and frames
        x = track_data['x'].values
        y = track_data['y'].values
        frames = track_data['frame'].values.astype(float)
        
        # Initialize track results
        track_result = {
            'track_id': track_id,
            'track_length': len(track_data),
            'duration': (frames[-1] - frames[0]) * frame_interval,
            'n_dwell_events': 0,
            'total_dwell_time': 0,
            'mean_dwell_time': 0,
            'dwell_fraction': 0
        }
        
        if regions is not None:
            # Region-based dwell time analysis
            # Check for dwelling in each region
            for region_idx, region in enumerate(regions):
                region_x = region['x'] * pixel_size
                region_y = region['y'] * pixel_size
                region_radius = region['radius'] * pixel_size
                
                # Calculate distances to region center
                distances = np.sqrt((x - region_x)**2 + (y - region_y)**2)
                
                # Find frames where the particle is within the region
                in_region = distances <= region_radius
                
                # Find continuous segments of dwelling
                dwell_segments = []
                current_segment = []
                
                for i, in_reg in enumerate(in_region):
                    if in_reg:
                        current_segment.append(i)
                    elif current_segment:
                        if len(current_segment) >= min_dwell_frames:
                            dwell_segments.append(current_segment)
                        current_segment = []
                
                # Check last segment
                if current_segment and len(current_segment) >= min_dwell_frames:
                    dwell_segments.append(current_segment)
                
                # Process dwell segments
                for segment in dwell_segments:
                    start_idx = segment[0]
                    end_idx = segment[-1]
                    
                    dwell_start_frame = frames[start_idx]
                    dwell_end_frame = frames[end_idx]
                    dwell_frames = dwell_end_frame - dwell_start_frame + 1
                    dwell_time = dwell_frames * frame_interval
                    
                    dwell_events.append({
                        'track_id': track_id,
                        'region_id': region_idx,
                        'start_frame': dwell_start_frame,
                        'end_frame': dwell_end_frame,
                        'dwell_frames': dwell_frames,
                        'dwell_time': dwell_time,
                        'start_x': x[start_idx],
                        'start_y': y[start_idx],
                        'end_x': x[end_idx],
                        'end_y': y[end_idx],
                        'mean_x': np.mean(x[start_idx:end_idx+1]),
                        'mean_y': np.mean(y[start_idx:end_idx+1])
                    })
                    
                    track_result['n_dwell_events'] += 1
                    track_result['total_dwell_time'] += dwell_time
            
            if track_result['n_dwell_events'] > 0:
                track_result['mean_dwell_time'] = track_result['total_dwell_time'] / track_result['n_dwell_events']
                track_result['dwell_fraction'] = track_result['total_dwell_time'] / track_result['duration']
        
        else:
            # Motion-based dwell time analysis
            # Detect periods of little movement
            
            # Calculate displacements between consecutive frames
            dx = np.diff(x)
            dy = np.diff(y)
            displacements = np.sqrt(dx**2 + dy**2)
            
            # Find frames with displacement below threshold
            # Use a more sensitive approach - consider both absolute threshold and relative to track scale
            track_median_displacement = np.median(displacements) if len(displacements) > 0 else 0
            adaptive_threshold = min(threshold_distance, track_median_displacement * 0.5) if track_median_displacement > 0 else threshold_distance
            
            is_dwelling = displacements <= adaptive_threshold
            # Pad to match original length for indexing
            is_dwelling = np.append(is_dwelling, False)
            
            # Find continuous segments of dwelling
            dwell_segments = []
            current_segment = []
            
            for i, dwelling in enumerate(is_dwelling):
                if dwelling:
                    current_segment.append(i)
                elif current_segment:
                    if len(current_segment) >= min_dwell_frames:
                        dwell_segments.append(current_segment)
                    current_segment = []
            
            # Check last segment
            if current_segment and len(current_segment) >= min_dwell_frames:
                dwell_segments.append(current_segment)
            
            # Process dwell segments
            for segment in dwell_segments:
                start_idx = segment[0]
                end_idx = segment[-1]
                
                dwell_start_frame = frames[start_idx]
                dwell_end_frame = frames[end_idx]
                dwell_frames = dwell_end_frame - dwell_start_frame + 1
                dwell_time = dwell_frames * frame_interval
                
                dwell_events.append({
                    'track_id': track_id,
                    'region_id': -1,  # No specific region
                    'start_frame': dwell_start_frame,
                    'end_frame': dwell_end_frame,
                    'dwell_frames': dwell_frames,
                    'dwell_time': dwell_time,
                    'start_x': x[start_idx],
                    'start_y': y[start_idx],
                    'end_x': x[end_idx],
                    'end_y': y[end_idx],
                    'mean_x': np.mean(x[start_idx:end_idx+1]),
                    'mean_y': np.mean(y[start_idx:end_idx+1])
                })
                
                track_result['n_dwell_events'] += 1
                track_result['total_dwell_time'] += dwell_time
            
            if track_result['n_dwell_events'] > 0:
                track_result['mean_dwell_time'] = track_result['total_dwell_time'] / track_result['n_dwell_events']
                track_result['dwell_fraction'] = track_result['total_dwell_time'] / track_result['duration']
        
        track_results.append(track_result)
    
    # Create default dataframes with proper columns
    if not dwell_events:
        dwell_events_df = pd.DataFrame(columns=['track_id', 'region_id', 'start_frame', 'end_frame', 
                                              'dwell_frames', 'dwell_time', 'start_x', 'start_y', 
                                              'end_x', 'end_y', 'mean_x', 'mean_y'])
    else:
        dwell_events_df = pd.DataFrame(dwell_events)
    
    if not track_results:
        track_results_df = pd.DataFrame(columns=['track_id', 'track_length', 'duration', 
                                               'n_dwell_events', 'total_dwell_time', 
                                               'mean_dwell_time', 'dwell_fraction'])
    else:
        track_results_df = pd.DataFrame(track_results)
    
    # Combine results
    results = {
        'success': True,  # Always return success even if no dwell events found
        'dwell_events': dwell_events_df,
        'track_results': track_results_df
    }
    
    # Calculate ensemble statistics (even when no dwell events are found)
    # This ensures we always have statistics to display
    results['ensemble_results'] = {
        'n_tracks_analyzed': len(tracks_df['track_id'].unique()),
        'n_tracks_with_dwell': len(track_results_df[track_results_df['n_dwell_events'] > 0]) if 'n_dwell_events' in track_results_df.columns and not track_results_df.empty else 0,
        'n_dwell_events': len(dwell_events_df),
        'mean_dwell_time': dwell_events_df['dwell_time'].mean() if 'dwell_time' in dwell_events_df.columns and not dwell_events_df.empty else 0,
        'median_dwell_time': dwell_events_df['dwell_time'].median() if 'dwell_time' in dwell_events_df.columns and not dwell_events_df.empty else 0,
        'max_dwell_time': dwell_events_df['dwell_time'].max() if 'dwell_time' in dwell_events_df.columns and not dwell_events_df.empty else 0,
        'mean_dwells_per_track': track_results_df['n_dwell_events'].mean() if 'n_dwell_events' in track_results_df.columns and not track_results_df.empty else 0,
        'mean_dwell_fraction': track_results_df['dwell_fraction'].mean() if 'dwell_fraction' in track_results_df.columns and not track_results_df.empty else 0,
        'dwell_threshold_frames': min_dwell_frames,
        'threshold_distance': threshold_distance,
        'analysis_type': 'motion-based' if regions is None else 'region-based',
        'total_regions_analyzed': len(regions) if regions is not None else 0
    }
    
    # If regions were provided, calculate region-specific statistics
    if regions is not None:
        for region_idx in range(len(regions)):
            region_events = dwell_events_df[dwell_events_df['region_id'] == region_idx]
            
            if len(region_events) > 0:
                results['ensemble_results'][f'region{region_idx}_n_events'] = len(region_events)
                results['ensemble_results'][f'region{region_idx}_mean_dwell_time'] = region_events['dwell_time'].mean()
                results['ensemble_results'][f'region{region_idx}_n_tracks'] = len(region_events['track_id'].unique())
    
    return results


# --- Gel Structure Analysis ---

def analyze_gel_structure(tracks_df: pd.DataFrame, 
                         min_confinement_time: int = 5, 
                         pixel_size: float = 1.0,
                         frame_interval: float = 1.0,
                         diffusion_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Analyze gel structure based on particle trajectories.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    min_confinement_time : int
        Minimum frames for detecting confinement
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    diffusion_threshold : float
        Threshold for identifying confined diffusion (µm²/s)
        
    Returns
    -------
    dict
        Dictionary containing gel structure analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # Analyze each track for confinement
    confined_regions = []
    track_results = []
    
    # Group by track_id
    grouped = tracks_df_um.groupby('track_id')
    
    for track_id, track_data in grouped:
        # Skip very short tracks
        if len(track_data) < min_confinement_time + 5:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions and frames
        x = track_data['x'].values
        y = track_data['y'].values
        frames = track_data['frame'].values.astype(float)
        
        # Initialize track result
        track_result = {
            'track_id': track_id,
            'track_length': len(track_data),
            'duration': (frames[-1] - frames[0]) * frame_interval,
            'n_confined_regions': 0
        }
        
        # Analyze windows of the trajectory
        window_size = min(20, len(track_data) // 2)
        
        for i in range(len(track_data) - window_size + 1):
            window_x = x[i:i+window_size]
            window_y = y[i:i+window_size]
            window_frames = frames[i:i+window_size]
            
            # Calculate MSD for this window
            msd_values = []
            max_lag = min(5, window_size - 1)
            
            for lag in range(1, max_lag + 1):
                # Calculate squared displacements
                sd_list = []
                
                for j in range(window_size - lag):
                    dx = window_x[j + lag] - window_x[j]
                    dy = window_y[j + lag] - window_y[j]
                    sd = dx**2 + dy**2
                    sd_list.append(sd)
                
                if sd_list:
                    msd_values.append(np.mean(sd_list))
            
            # Calculate diffusion coefficient
            if len(msd_values) >= 3:
                # Linear fit to first few points
                lag_times = np.arange(1, len(msd_values) + 1) * frame_interval
                
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        lag_times[:3], 
                        msd_values[:3]
                    )
                    
                    D = slope / 4  # µm²/s
                    
                    # Check if diffusion is confined
                    if D < diffusion_threshold and r_value > 0.7:
                        # Calculate confinement statistics
                        x_center = np.mean(window_x)
                        y_center = np.mean(window_y)
                        
                        # Calculate distances from center
                        distances = np.sqrt((window_x - x_center)**2 + (window_y - y_center)**2)
                        radius = np.std(distances)
                        
                        confined_regions.append({
                            'track_id': track_id,
                            'start_frame': window_frames[0],
                            'end_frame': window_frames[-1],
                            'center_x': x_center,
                            'center_y': y_center,
                            'radius': radius,
                            'diffusion_coeff': D,
                            'msd_r_value': r_value
                        })
                        
                        track_result['n_confined_regions'] += 1
                except:
                    pass
        
        track_results.append(track_result)
    
    # Calculate mesh statistics
    mesh_size = None
    mesh_heterogeneity = None
    pore_distribution = None
    
    if confined_regions:
        confined_df = pd.DataFrame(confined_regions)
        
        # Estimate mesh size from confinement radii
        radii = confined_df['radius'].values
        mesh_size = 2 * np.median(radii)  # Estimate mesh size as twice the median confinement radius
        mesh_heterogeneity = np.std(radii) / np.mean(radii)  # Coefficient of variation
        
        # Get pore locations
        pore_centers = confined_df[['center_x', 'center_y']].values
        
        # Detect pore clustering
        if len(pore_centers) >= 5:
            # Use DBSCAN to find clusters of pores
            clustering = DBSCAN(eps=2*mesh_size, min_samples=3)
            labels = clustering.fit_predict(pore_centers)
            
            # Count clusters (exclude noise points with label -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Calculate pore distribution
            if n_clusters > 0:
                # Calculate cluster statistics
                pore_distribution = {
                    'n_clusters': n_clusters,
                    'clustered_fraction': np.sum(labels != -1) / len(labels),
                    'mean_cluster_size': np.mean([np.sum(labels == i) for i in range(n_clusters)])
                }
            else:
                # No clusters found
                pore_distribution = {
                    'n_clusters': 0,
                    'clustered_fraction': 0,
                    'mean_cluster_size': 0
                }
                
                # Calculate nearest neighbor distances for pores
                if len(pore_centers) >= 2:
                    nn = NearestNeighbors(n_neighbors=2)
                    nn.fit(pore_centers)
                    distances, _ = nn.kneighbors(pore_centers)
                    
                    # First column is distance to self (0), second is nearest neighbor
                    nn_distances = distances[:, 1]
                    
                    pore_distribution['mean_nn_distance'] = np.mean(nn_distances)
                    pore_distribution['std_nn_distance'] = np.std(nn_distances)
                    pore_distribution['regularity'] = np.mean(nn_distances) / np.std(nn_distances) if np.std(nn_distances) > 0 else np.inf
    
    # Create default dataframes with proper columns
    if not confined_regions:
        confined_regions_df = pd.DataFrame(columns=['track_id', 'region_id', 'start_frame', 'end_frame', 'duration', 
                                                  'center_x', 'center_y', 'radius', 'mean_diffusion'])
    else:
        confined_regions_df = pd.DataFrame(confined_regions)
        
    if not track_results:
        track_results_df = pd.DataFrame(columns=['track_id', 'track_length', 'duration', 'n_confined_regions', 
                                               'total_confined_time', 'confined_fraction', 'mean_confinement_radius'])
    else:
        track_results_df = pd.DataFrame(track_results)
        
    # Add mesh properties
    mesh_properties = {
        'mean_mesh_size': mesh_size,
        'mesh_heterogeneity': mesh_heterogeneity,
        'total_tracks_analyzed': len(tracks_df['track_id'].unique()),
        'diffusion_threshold': diffusion_threshold
    }
    
    if pore_distribution:
        mesh_properties.update(pore_distribution)
    
    # Combine results
    results = {
        'success': True,  # Always return success even if no confined regions found
        'confined_regions': confined_regions_df,
        'track_results': track_results_df,
        'mesh_properties': mesh_properties
    }
    
    # Fix mesh properties to handle None values
    if mesh_properties['mean_mesh_size'] is None:
        mesh_properties['mean_mesh_size'] = 0.0
    if mesh_properties['mesh_heterogeneity'] is None:
        mesh_properties['mesh_heterogeneity'] = 0.0
    
    # Calculate ensemble statistics (even if no confined regions found)
    results['ensemble_results'] = {
        'n_tracks_analyzed': len(tracks_df['track_id'].unique()),
        'n_tracks_with_confinement': len(track_results_df[track_results_df['n_confined_regions'] > 0]) if not track_results_df.empty else 0,
        'n_confined_regions': len(confined_regions_df),
        'mean_confinement_radius': confined_regions_df['radius'].mean() if 'radius' in confined_regions_df.columns and not confined_regions_df.empty else 0,
        'median_confinement_radius': confined_regions_df['radius'].median() if 'radius' in confined_regions_df.columns and not confined_regions_df.empty else 0,
        'std_confinement_radius': confined_regions_df['radius'].std() if 'radius' in confined_regions_df.columns and not confined_regions_df.empty else 0,
        'mean_confined_regions_per_track': track_results_df['n_confined_regions'].mean() if 'n_confined_regions' in track_results_df.columns and not track_results_df.empty else 0,
        'mesh_size': mesh_size,
        'mesh_heterogeneity': mesh_heterogeneity,
        'confinement_threshold': min_confinement_time,
        'diffusion_coefficient_threshold': diffusion_threshold
    }
    
    if pore_distribution:
        results['ensemble_results'].update(pore_distribution)
    
    return results


# --- Diffusion Population Analysis ---

def analyze_diffusion_population(tracks_df: pd.DataFrame, 
                               max_lag: int = 20, 
                               pixel_size: float = 1.0,
                               frame_interval: float = 1.0, 
                               min_track_length: int = 10,
                               n_populations: int = 2) -> Dict[str, Any]:
    """
    Analyze diffusion populations using mixture models.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    max_lag : int
        Maximum lag time for MSD calculation
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    min_track_length : int
        Minimum track length to include in analysis
    n_populations : int
        Number of diffusion populations to identify
        
    Returns
    -------
    dict
        Dictionary containing diffusion population analysis results
    """
    # Calculate diffusion coefficients for all tracks
    # Use the MSD at lag 1 for initial estimation
    track_diffusion = []
    
    # Group by track_id
    grouped = tracks_df.groupby('track_id')
    
    for track_id, track_data in grouped:
        # Skip short tracks
        if len(track_data) < min_track_length:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions
        x = track_data['x'].values.astype(float) * pixel_size
        y = track_data['y'].values.astype(float) * pixel_size
        frames = track_data['frame'].values.astype(float)
        
        # Calculate MSD for different lag times
        msd_values = []
        
        for lag in range(1, min(max_lag + 1, len(track_data))):
            # Calculate squared displacements
            sd_list = []
            
            for i in range(len(track_data) - lag):
                dx = x[i + lag] - x[i]
                dy = y[i + lag] - y[i]
                sd = dx**2 + dy**2
                sd_list.append(sd)
            
            if sd_list:
                msd_values.append(np.mean(sd_list))
        
        if len(msd_values) >= 3:
            # Calculate diffusion coefficient from MSD curve
            lag_times = np.arange(1, len(msd_values) + 1) * frame_interval
            
            try:
                # Linear fit to first few points
                slope, intercept, r_value, p_value, std_err = linregress(
                    lag_times[:3], 
                    msd_values[:3]
                )
                
                D = slope / 4  # µm²/s
                
                # Check if fit is reasonable
                if D > 0 and r_value > 0.7:
                    # Calculate anomalous exponent (alpha)
                    log_lag = np.log(lag_times)
                    log_msd = np.log(msd_values)
                    
                    alpha_slope, alpha_intercept, alpha_r, alpha_p, alpha_err = linregress(log_lag, log_msd)
                    
                    track_diffusion.append({
                        'track_id': track_id,
                        'diffusion_coeff': D,
                        'r_value': r_value,
                        'alpha': alpha_slope,
                        'track_length': len(track_data)
                    })
            except:
                pass
    
    # Create empty default results if no valid tracks found
    if not track_diffusion:
        empty_populations = [{
            'population_id': i,
            'weight': 1.0 if i == 0 else 0.0,
            'mean_diffusion_coeff': 0.0,
            'std_diffusion_coeff': 0.0,
            'log_mean': 0.0,
            'log_std': 0.0
        } for i in range(n_populations)]
        
        return {
            'success': True,  # Return success even when no valid tracks
            'populations': pd.DataFrame(empty_populations),
            'track_assignments': pd.DataFrame(columns=['track_id', 'population_id', 'diffusion_coeff']),
            'n_tracks_analyzed': len(tracks_df['track_id'].unique()),
            'n_tracks_with_valid_diffusion': 0,
            'n_populations': n_populations,
            'notes': 'No tracks suitable for diffusion population analysis'
        }
    
    # Convert to DataFrame
    diffusion_df = pd.DataFrame(track_diffusion)
    
    # Apply log transform to diffusion coefficients for better fitting
    log_D = np.log10(diffusion_df['diffusion_coeff'].values)
    
    # Handle outliers and invalid values
    valid_indices = ~np.isnan(log_D) & ~np.isinf(log_D)
    log_D_valid = log_D[valid_indices]
    
    # Check if we have enough data points for the requested number of populations
    if len(log_D_valid) < n_populations * 2:
        # Adjust n_populations to a reasonable value based on data available
        adjusted_n_populations = max(1, int(len(log_D_valid) / 2))
        
        # Create simple default results with adjusted number of populations
        if adjusted_n_populations == 1:
            # Just one population - use mean and std directly
            mean_D = np.mean(10 ** log_D_valid)
            std_D = np.std(10 ** log_D_valid)
            
            populations = [{
                'population_id': 0,
                'weight': 1.0,
                'mean_diffusion_coeff': mean_D,
                'std_diffusion_coeff': std_D,
                'log_mean': np.mean(log_D_valid),
                'log_std': np.std(log_D_valid)
            }]
            
            # Assign all tracks to this population
            track_assignments = []
            for i, (idx, row) in enumerate(diffusion_df.iterrows()):
                if valid_indices[i]:
                    track_assignments.append({
                        'track_id': row['track_id'],
                        'population_id': 0,
                        'diffusion_coeff': row['diffusion_coeff']
                    })
            
            return {
                'success': True,
                'populations': pd.DataFrame(populations),
                'track_assignments': pd.DataFrame(track_assignments) if track_assignments else pd.DataFrame(columns=['track_id', 'population_id', 'diffusion_coeff']),
                'n_tracks_analyzed': len(tracks_df['track_id'].unique()),
                'n_tracks_with_valid_diffusion': len(track_assignments),
                'n_populations': 1,
                'notes': f'Requested {n_populations} populations, but only enough data for 1 population'
            }
            
        # Otherwise, continue with adjusted_n_populations
        n_populations = adjusted_n_populations
    
    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=n_populations, random_state=42)
    gmm.fit(log_D_valid.reshape(-1, 1))
    
    # Get population parameters
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    # Sort populations by mean diffusion coefficient (ascending)
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    covariances = covariances[sorted_indices]
    weights = weights[sorted_indices]
    
    # Create DataFrame with population parameters
    populations = []
    
    for i in range(n_populations):
        # Convert log means back to original scale
        D_mean = 10 ** means[i]
        D_std = D_mean * np.log(10) * np.sqrt(covariances[i])
        
        populations.append({
            'population_id': i,
            'weight': weights[i],
            'mean_diffusion_coeff': D_mean,
            'std_diffusion_coeff': D_std,
            'log_mean': means[i],
            'log_std': np.sqrt(covariances[i])
        })
    
    # Assign each track to a population
    # Predict population for each valid track
    populations_df = pd.DataFrame(populations)
    
    if valid_indices.sum() > 0:
        valid_track_indices = np.where(valid_indices)[0]
        predictions = gmm.predict(log_D_valid.reshape(-1, 1))
        
        # Reindex predictions according to sorted populations
        prediction_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        predictions = np.array([prediction_map[p] for p in predictions])
        
        # Add population assignments to tracks
        track_assignments = []
        
        for i, pred in enumerate(predictions):
            track_idx = valid_track_indices[i]
            track_id = diffusion_df.iloc[track_idx]['track_id']
            
            track_assignments.append({
                'track_id': track_id,
                'diffusion_coeff': diffusion_df.iloc[track_idx]['diffusion_coeff'],
                'log_diffusion_coeff': log_D[track_idx],
                'population_id': pred,
                'alpha': diffusion_df.iloc[track_idx]['alpha'] if 'alpha' in diffusion_df.columns else np.nan
            })
        
        track_assignments_df = pd.DataFrame(track_assignments)
    else:
        track_assignments_df = pd.DataFrame(columns=['track_id', 'diffusion_coeff', 'log_diffusion_coeff', 'population_id', 'alpha'])
    
    # Combine results
    results = {
        'success': True,
        'populations': populations_df,
        'track_assignments': track_assignments_df,
        'raw_diffusion_data': diffusion_df,
        'n_tracks_analyzed': len(diffusion_df),
        'n_valid_tracks': valid_indices.sum()
    }
    
    # Calculate additional statistics for each population
    if not track_assignments_df.empty:
        for i in range(n_populations):
            pop_tracks = track_assignments_df[track_assignments_df['population_id'] == i]
            
            if len(pop_tracks) > 0:
                populations_df.at[i, 'n_tracks'] = len(pop_tracks)
                populations_df.at[i, 'track_fraction'] = len(pop_tracks) / len(track_assignments_df)
                
                if 'alpha' in pop_tracks.columns:
                    populations_df.at[i, 'mean_alpha'] = pop_tracks['alpha'].mean()
                    populations_df.at[i, 'median_alpha'] = pop_tracks['alpha'].median()
    
    return results


# --- Crowding Analysis ---

def analyze_crowding(tracks_df: pd.DataFrame, 
                    radius_of_influence: float = 2.0,
                    pixel_size: float = 1.0,
                    min_track_length: int = 5) -> Dict[str, Any]:
    """
    Analyze effects of molecular crowding on particle dynamics.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    radius_of_influence : float
        Radius to consider for density calculation (µm)
    pixel_size : float
        Pixel size in micrometers
    min_track_length : int
        Minimum track length to include in analysis
        
    Returns
    -------
    dict
        Dictionary containing crowding analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # Get unique frames
    frames = sorted(tracks_df_um['frame'].unique())
    
    # Initialize results
    crowding_data = []
    track_results = []
    
    # Calculate local densities for each frame
    for frame in frames:
        # Get particles in this frame
        frame_data = tracks_df_um[tracks_df_um['frame'] == frame]
        
        if len(frame_data) < 2:
            # Not enough points for density analysis
            continue
            
        # Get coordinates
        coords = frame_data[['x', 'y']].values
        track_ids = frame_data['track_id'].values
        
        # Calculate pairwise distances
        distances = distance.pdist(coords)
        distance_matrix = distance.squareform(distances)
        
        # Calculate local density for each particle
        for i in range(len(coords)):
            # Count neighbors within radius_of_influence
            neighbors = np.sum(distance_matrix[i] <= radius_of_influence) - 1  # Exclude self
            
            # Calculate density (particles per square µm)
            density = neighbors / (np.pi * radius_of_influence**2)
            
            crowding_data.append({
                'frame': frame,
                'track_id': track_ids[i],
                'x': coords[i, 0],
                'y': coords[i, 1],
                'local_density': density,
                'n_neighbors': neighbors
            })
    
    if not crowding_data:
        return {
            'success': False,
            'error': 'Not enough data for crowding analysis'
        }
    
    # Convert to DataFrame
    crowding_df = pd.DataFrame(crowding_data)
    
    # Analyze effect of crowding on each track
    # Group by track_id
    grouped = tracks_df_um.groupby('track_id')
    
    for track_id, track_data in grouped:
        # Skip short tracks
        if len(track_data) < min_track_length:
            continue
            
        # Get crowding data for this track
        track_crowding = crowding_df[crowding_df['track_id'] == track_id]
        
        if len(track_crowding) < min_track_length:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        track_crowding = track_crowding.sort_values('frame')
        
        # Match frames
        common_frames = set(track_data['frame']) & set(track_crowding['frame'])
        
        if len(common_frames) < min_track_length:
            continue
            
        # Filter data to common frames
        track_data = track_data[track_data['frame'].isin(common_frames)]
        track_crowding = track_crowding[track_crowding['frame'].isin(common_frames)]
        
        # Sort by frame
        track_data = track_data.sort_values('frame')
        track_crowding = track_crowding.sort_values('frame')
        
        # Extract positions and densities
        x = track_data['x'].values
        y = track_data['y'].values
        frames = track_data['frame'].values.astype(float)
        densities = track_crowding['local_density'].values
        
        # Calculate displacements
        dx = np.diff(x)
        dy = np.diff(y)
        displacements = np.sqrt(dx**2 + dy**2)
        
        # Match displacements with densities (use density at starting point)
        disp_densities = densities[:-1]
        
        # Analyze correlation between displacement and density
        if len(displacements) >= 3:
            corr_coef, p_value = np.corrcoef(displacements, disp_densities)[0, 1], 0
            
            try:
                from scipy.stats import pearsonr
                corr_coef, p_value = pearsonr(displacements, disp_densities)
            except:
                pass
            
            # Calculate median displacement at different density levels
            # Divide densities into tertiles (low, medium, high)
            density_tertiles = np.percentile(disp_densities, [33, 66])
            
            low_density_disp = displacements[disp_densities <= density_tertiles[0]]
            med_density_disp = displacements[(disp_densities > density_tertiles[0]) & 
                                          (disp_densities <= density_tertiles[1])]
            high_density_disp = displacements[disp_densities > density_tertiles[1]]
            
            # Calculate median displacement for each group
            low_density_median = np.median(low_density_disp) if len(low_density_disp) > 0 else np.nan
            med_density_median = np.median(med_density_disp) if len(med_density_disp) > 0 else np.nan
            high_density_median = np.median(high_density_disp) if len(high_density_disp) > 0 else np.nan
            
            # Calculate mean displacement for each group
            low_density_mean = np.mean(low_density_disp) if len(low_density_disp) > 0 else np.nan
            med_density_mean = np.mean(med_density_disp) if len(med_density_disp) > 0 else np.nan
            high_density_mean = np.mean(high_density_disp) if len(high_density_disp) > 0 else np.nan
            
            # Calculate crowding effect ratio
            crowding_effect = high_density_median / low_density_median if (
                not np.isnan(high_density_median) and 
                not np.isnan(low_density_median) and 
                low_density_median > 0
            ) else np.nan
            
            track_results.append({
                'track_id': track_id,
                'n_points': len(track_data),
                'mean_density': np.mean(densities),
                'median_density': np.median(densities),
                'density_std': np.std(densities),
                'mean_displacement': np.mean(displacements),
                'median_displacement': np.median(displacements),
                'displacement_std': np.std(displacements),
                'density_displacement_correlation': corr_coef,
                'correlation_p_value': p_value,
                'low_density_median_disp': low_density_median,
                'med_density_median_disp': med_density_median,
                'high_density_median_disp': high_density_median,
                'low_density_mean_disp': low_density_mean,
                'med_density_mean_disp': med_density_mean,
                'high_density_mean_disp': high_density_mean,
                'crowding_effect_ratio': crowding_effect
            })
    
    # Combine results
    results = {
        'success': True if track_results else False,
        'crowding_data': crowding_df,
        'track_results': pd.DataFrame(track_results) if track_results else pd.DataFrame()
    }
    
    # Calculate ensemble statistics
    if track_results:
        track_results_df = pd.DataFrame(track_results)
        
        # Overall statistics
        results['ensemble_results'] = {
            'n_tracks_analyzed': len(track_results_df),
            'mean_density': crowding_df['local_density'].mean(),
            'median_density': crowding_df['local_density'].median(),
            'max_density': crowding_df['local_density'].max(),
            'density_std': crowding_df['local_density'].std(),
            'density_cv': crowding_df['local_density'].std() / crowding_df['local_density'].mean(),
            'mean_correlation': track_results_df['density_displacement_correlation'].mean(),
            'significant_correlations': np.sum(track_results_df['correlation_p_value'] < 0.05),
            'mean_crowding_effect': track_results_df['crowding_effect_ratio'].mean(),
            'median_crowding_effect': track_results_df['crowding_effect_ratio'].median()
        }
        
        # Count tracks with different correlation types
        neg_corr = np.sum((track_results_df['density_displacement_correlation'] < -0.3) & 
                          (track_results_df['correlation_p_value'] < 0.05))
        pos_corr = np.sum((track_results_df['density_displacement_correlation'] > 0.3) & 
                         (track_results_df['correlation_p_value'] < 0.05))
        no_corr = len(track_results_df) - neg_corr - pos_corr
        
        results['ensemble_results']['negative_correlation_count'] = neg_corr
        results['ensemble_results']['positive_correlation_count'] = pos_corr
        results['ensemble_results']['no_correlation_count'] = no_corr
        
        results['ensemble_results']['negative_correlation_fraction'] = neg_corr / len(track_results_df)
        results['ensemble_results']['positive_correlation_fraction'] = pos_corr / len(track_results_df)
        results['ensemble_results']['no_correlation_fraction'] = no_corr / len(track_results_df)
    
    return results


# --- Active Transport Analysis ---

def analyze_active_transport(tracks_df: pd.DataFrame, 
                            window_size: int = 5,
                            min_track_length: int = 10,
                            pixel_size: float = 1.0,
                            frame_interval: float = 1.0,
                            straightness_threshold: float = 0.8,
                            min_segment_length: int = 5) -> Dict[str, Any]:
    """
    Analyze directed motion and active transport.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    window_size : int
        Window size for detecting directed segments
    min_track_length : int
        Minimum track length to include in analysis
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    straightness_threshold : float
        Threshold for straightness to identify directed motion
    min_segment_length : int
        Minimum length of directed motion segment
        
    Returns
    -------
    dict
        Dictionary containing active transport analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # Initialize results
    directed_segments = []
    track_results = []
    
    # Group by track_id
    grouped = tracks_df_um.groupby('track_id')
    
    for track_id, track_data in grouped:
        # Skip short tracks
        if len(track_data) < min_track_length:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions and frames
        x = track_data['x'].values
        y = track_data['y'].values
        frames = track_data['frame'].values.astype(float)
        
        # Calculate displacements
        dx = np.diff(x)
        dy = np.diff(y)
        displacements = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        
        # Initialize track result
        track_result = {
            'track_id': track_id,
            'track_length': len(track_data),
            'duration': (frames[-1] - frames[0]) * frame_interval,
            'mean_speed': np.mean(displacements / frame_interval),
            'max_speed': np.max(displacements / frame_interval),
            'n_directed_segments': 0,
            'directed_fraction': 0.0
        }
        
        # Detect directed motion segments using sliding window
        if len(track_data) >= window_size + min_segment_length:
            segment_start = None
            current_segment = []
            
            for i in range(len(track_data) - window_size + 1):
                # Analyze window from i to i+window_size
                window_x = x[i:i+window_size]
                window_y = y[i:i+window_size]
                
                # Calculate straightness within window
                # Net displacement / total path length
                net_disp = np.sqrt((window_x[-1] - window_x[0])**2 + (window_y[-1] - window_y[0])**2)
                
                # Calculate total path length
                path_length = 0
                for j in range(1, window_size):
                    path_length += np.sqrt((window_x[j] - window_x[j-1])**2 + (window_y[j] - window_y[j-1])**2)
                
                if path_length > 0:
                    straightness = net_disp / path_length
                else:
                    straightness = 0
                
                # Check if this window shows directed motion
                if straightness >= straightness_threshold:
                    # This window is part of a directed segment
                    if segment_start is None:
                        segment_start = i
                        current_segment = list(range(i, i+window_size))
                    else:
                        # Extend current segment
                        for j in range(i+1, i+window_size):
                            if j not in current_segment:
                                current_segment.append(j)
                elif segment_start is not None:
                    # End of a directed segment
                    if len(current_segment) >= min_segment_length:
                        # This is a valid segment
                        segment_frames = frames[current_segment]
                        segment_x = x[current_segment]
                        segment_y = y[current_segment]
                        
                        # Calculate segment properties
                        segment_dx = segment_x[-1] - segment_x[0]
                        segment_dy = segment_y[-1] - segment_y[0]
                        segment_displacement = np.sqrt(segment_dx**2 + segment_dy**2)
                        segment_duration = (segment_frames[-1] - segment_frames[0]) * frame_interval
                        segment_speed = segment_displacement / segment_duration if segment_duration > 0 else 0
                        segment_angle = np.arctan2(segment_dy, segment_dx)
                        
                        # Calculate straightness of the entire segment
                        segment_path_length = 0
                        for j in range(1, len(current_segment)):
                            idx1 = current_segment[j-1]
                            idx2 = current_segment[j]
                            segment_path_length += np.sqrt((x[idx2] - x[idx1])**2 + (y[idx2] - y[idx1])**2)
                        
                        segment_straightness = segment_displacement / segment_path_length if segment_path_length > 0 else 0
                        
                        directed_segments.append({
                            'track_id': track_id,
                            'start_frame': segment_frames[0],
                            'end_frame': segment_frames[-1],
                            'duration': segment_duration,
                            'n_frames': len(current_segment),
                            'start_x': segment_x[0],
                            'start_y': segment_y[0],
                            'end_x': segment_x[-1],
                            'end_y': segment_y[-1],
                            'displacement': segment_displacement,
                            'path_length': segment_path_length,
                            'straightness': segment_straightness,
                            'speed': segment_speed,
                            'angle': segment_angle
                        })
                        
                        track_result['n_directed_segments'] += 1
                    
                    # Reset segment
                    segment_start = None
                    current_segment = []
            
            # Check if there's an ongoing segment at the end
            if segment_start is not None and len(current_segment) >= min_segment_length:
                # This is a valid segment
                segment_frames = frames[current_segment]
                segment_x = x[current_segment]
                segment_y = y[current_segment]
                
                # Calculate segment properties
                segment_dx = segment_x[-1] - segment_x[0]
                segment_dy = segment_y[-1] - segment_y[0]
                segment_displacement = np.sqrt(segment_dx**2 + segment_dy**2)
                segment_duration = (segment_frames[-1] - segment_frames[0]) * frame_interval
                segment_speed = segment_displacement / segment_duration if segment_duration > 0 else 0
                segment_angle = np.arctan2(segment_dy, segment_dx)
                
                # Calculate straightness of the entire segment
                segment_path_length = 0
                for j in range(1, len(current_segment)):
                    idx1 = current_segment[j-1]
                    idx2 = current_segment[j]
                    segment_path_length += np.sqrt((x[idx2] - x[idx1])**2 + (y[idx2] - y[idx1])**2)
                
                segment_straightness = segment_displacement / segment_path_length if segment_path_length > 0 else 0
                
                directed_segments.append({
                    'track_id': track_id,
                    'start_frame': segment_frames[0],
                    'end_frame': segment_frames[-1],
                    'duration': segment_duration,
                    'n_frames': len(current_segment),
                    'start_x': segment_x[0],
                    'start_y': segment_y[0],
                    'end_x': segment_x[-1],
                    'end_y': segment_y[-1],
                    'displacement': segment_displacement,
                    'path_length': segment_path_length,
                    'straightness': segment_straightness,
                    'speed': segment_speed,
                    'angle': segment_angle
                })
                
                track_result['n_directed_segments'] += 1
            
            # Calculate directed fraction for the track
            if track_result['n_directed_segments'] > 0:
                # Count frames in directed segments
                directed_frames = set()
                
                for segment in directed_segments:
                    if segment['track_id'] == track_id:
                        for frame in range(int(segment['start_frame']), int(segment['end_frame']) + 1):
                            directed_frames.add(frame)
                
                directed_count = len(directed_frames)
                track_result['directed_fraction'] = directed_count / len(track_data)
        
        track_results.append(track_result)
    
    # Combine results
    results = {
        'success': True if directed_segments else False,
        'directed_segments': pd.DataFrame(directed_segments) if directed_segments else pd.DataFrame(),
        'track_results': pd.DataFrame(track_results) if track_results else pd.DataFrame()
    }
    
    # Calculate ensemble statistics
    if directed_segments:
        segments_df = pd.DataFrame(directed_segments)
        track_results_df = pd.DataFrame(track_results)
        
        results['ensemble_results'] = {
            'n_tracks_analyzed': len(track_results_df),
            'n_tracks_with_directed_motion': len(track_results_df[track_results_df['n_directed_segments'] > 0]),
            'directed_motion_fraction': len(track_results_df[track_results_df['n_directed_segments'] > 0]) / len(track_results_df),
            'n_directed_segments': len(segments_df),
            'mean_segments_per_track': track_results_df['n_directed_segments'].mean(),
            'mean_segment_duration': segments_df['duration'].mean(),
            'mean_segment_speed': segments_df['speed'].mean(),
            'mean_segment_straightness': segments_df['straightness'].mean(),
            'mean_directed_fraction': track_results_df['directed_fraction'].mean()
        }
        
        # Calculate mean transport velocity from all directed segments
        results['ensemble_results']['mean_transport_velocity'] = segments_df['speed'].mean()
        results['ensemble_results']['median_transport_velocity'] = segments_df['speed'].median()
        results['ensemble_results']['max_transport_velocity'] = segments_df['speed'].max()
    
    return results


# --- Boundary Crossing Analysis ---

def analyze_boundary_crossing(tracks_df: pd.DataFrame, 
                             boundaries=None,
                             pixel_size: float = 1.0,
                             frame_interval: float = 1.0,
                             min_track_length: int = 5) -> Dict[str, Any]:
    """
    Analyze boundary crossing events and statistics.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    boundaries : list of dict
        List of boundaries to analyze (each with parameters defining the boundary)
        If None, tries to identify boundaries from data
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    min_track_length : int
        Minimum track length to include in analysis
        
    Returns
    -------
    dict
        Dictionary containing boundary crossing analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # Initialize results
    crossing_events = []
    track_results = []
    
    # If no boundaries provided, try to detect them
    detected_boundaries = []
    
    if boundaries is None:
        # Get overall spatial distribution
        x_min, x_max = tracks_df_um['x'].min(), tracks_df_um['x'].max()
        y_min, y_max = tracks_df_um['y'].min(), tracks_df_um['y'].max()
        
        # Calculate binned densities to find areas of high gradient
        try:
            from scipy.stats import gaussian_kde
            
            # Sample points for KDE
            points = tracks_df_um[['x', 'y']].values
            
            # Create grid for density estimation
            n_grid = 20
            x_grid = np.linspace(x_min, x_max, n_grid)
            y_grid = np.linspace(y_min, y_max, n_grid)
            X, Y = np.meshgrid(x_grid, y_grid)
            xy_grid = np.vstack([X.ravel(), Y.ravel()]).T
            
            # Estimate density
            if len(points) > 10:
                kde = gaussian_kde(points.T)
                Z = kde(np.vstack([xy_grid[:, 0], xy_grid[:, 1]]))
                Z = Z.reshape(X.shape)
                
                # Find gradient of density
                from scipy.ndimage import gaussian_filter
                
                # Smooth density
                Z_smooth = gaussian_filter(Z, sigma=1)
                
                # Calculate gradient
                gy, gx = np.gradient(Z_smooth)
                gradient_magnitude = np.sqrt(gx**2 + gy**2)
                
                # Find high gradient regions (potential boundaries)
                threshold = np.percentile(gradient_magnitude, 75)
                high_gradient = gradient_magnitude > threshold
                
                # Extract boundary coordinates
                for i in range(n_grid):
                    for j in range(n_grid):
                        if high_gradient[i, j]:
                            # This is a potential boundary point
                            detected_boundaries.append({
                                'type': 'point',
                                'x': X[i, j],
                                'y': Y[i, j],
                                'gradient': gradient_magnitude[i, j]
                            })
                
                # If enough boundary points, try to fit lines or curves
                if len(detected_boundaries) >= 5:
                    # Cluster boundary points
                    boundary_points = np.array([[b['x'], b['y']] for b in detected_boundaries])
                    clustering = DBSCAN(eps=min((x_max-x_min), (y_max-y_min))/10, min_samples=3)
                    labels = clustering.fit_predict(boundary_points)
                    
                    # For each cluster, fit a line
                    for label in set(labels):
                        if label == -1:
                            continue  # Skip noise
                            
                        cluster_points = boundary_points[labels == label]
                        
                        if len(cluster_points) < 5:
                            continue
                            
                        # Fit line: y = mx + b
                        try:
                            # Check if points are more aligned with x or y axis
                            x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
                            y_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
                            
                            if x_range > y_range:
                                # Fit y = mx + b
                                m, b, r, p, se = linregress(cluster_points[:, 0], cluster_points[:, 1])
                                
                                if abs(r) > 0.7:
                                    # Good linear fit
                                    detected_boundaries = []  # Clear point boundaries
                                    boundaries = [{
                                        'type': 'line',
                                        'slope': m,
                                        'intercept': b,
                                        'orientation': 'y=mx+b',
                                        'x_min': np.min(cluster_points[:, 0]),
                                        'x_max': np.max(cluster_points[:, 0]),
                                        'gradient': np.mean([d['gradient'] for i, d in enumerate(detected_boundaries) 
                                                          if labels[i] == label])
                                    }]
                                    break
                            else:
                                # Fit x = my + b
                                m, b, r, p, se = linregress(cluster_points[:, 1], cluster_points[:, 0])
                                
                                if abs(r) > 0.7:
                                    # Good linear fit
                                    detected_boundaries = []  # Clear point boundaries
                                    boundaries = [{
                                        'type': 'line',
                                        'slope': m,
                                        'intercept': b,
                                        'orientation': 'x=my+b',
                                        'y_min': np.min(cluster_points[:, 1]),
                                        'y_max': np.max(cluster_points[:, 1]),
                                        'gradient': np.mean([d['gradient'] for i, d in enumerate(detected_boundaries) 
                                                          if labels[i] == label])
                                    }]
                                    break
                        except:
                            pass
        except:
            pass
        
        # If no boundaries detected, use simple division
        if not boundaries and not detected_boundaries:
            # Divide the space into quadrants or use midpoints
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            
            # Add horizontal and vertical dividers
            boundaries = [
                {
                    'type': 'line',
                    'orientation': 'horizontal',
                    'y': y_mid,
                    'x_min': x_min,
                    'x_max': x_max
                },
                {
                    'type': 'line',
                    'orientation': 'vertical',
                    'x': x_mid,
                    'y_min': y_min,
                    'y_max': y_max
                }
            ]
        
        # If we still have point boundaries, add them
        if detected_boundaries and not boundaries:
            boundaries = detected_boundaries
    
    # Analyze tracks for boundary crossings
    # Group by track_id
    grouped = tracks_df_um.groupby('track_id')
    
    for track_id, track_data in grouped:
        # Skip short tracks
        if len(track_data) < min_track_length:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Extract positions and frames
        x = track_data['x'].values
        y = track_data['y'].values
        frames = track_data['frame'].values.astype(float)
        
        # Initialize track result
        track_result = {
            'track_id': track_id,
            'track_length': len(track_data),
            'duration': (frames[-1] - frames[0]) * frame_interval,
            'n_crossings': 0
        }
        
        # Check for boundary crossings
        for boundary in boundaries:
            # Determine boundary type and check for crossings
            if boundary['type'] == 'line':
                if boundary['orientation'] == 'horizontal' or boundary['orientation'] == 'y=c':
                    # Horizontal line: y = c
                    boundary_y = boundary.get('y', boundary.get('intercept', 0))
                    
                    # Check each pair of consecutive points
                    for i in range(len(track_data) - 1):
                        y1, y2 = y[i], y[i+1]
                        
                        # Check if the boundary is crossed
                        if (y1 <= boundary_y and y2 > boundary_y) or (y1 > boundary_y and y2 <= boundary_y):
                            # Calculate crossing point
                            x1, x2 = x[i], x[i+1]
                            t = (boundary_y - y1) / (y2 - y1) if y2 != y1 else 0
                            crossing_x = x1 + t * (x2 - x1)
                            
                            # Check if crossing is within the boundary limits
                            x_min_bound = boundary.get('x_min', x_min)
                            x_max_bound = boundary.get('x_max', x_max)
                            
                            if x_min_bound <= crossing_x <= x_max_bound:
                                # This is a valid crossing
                                crossing_frame = frames[i] + t * (frames[i+1] - frames[i])
                                crossing_time = crossing_frame * frame_interval
                                
                                # Determine crossing direction
                                direction = 'up' if y2 > y1 else 'down'
                                
                                crossing_events.append({
                                    'track_id': track_id,
                                    'boundary_id': boundaries.index(boundary),
                                    'boundary_type': 'horizontal',
                                    'crossing_frame': crossing_frame,
                                    'crossing_time': crossing_time,
                                    'crossing_x': crossing_x,
                                    'crossing_y': boundary_y,
                                    'direction': direction,
                                    'pre_x': x1,
                                    'pre_y': y1,
                                    'post_x': x2,
                                    'post_y': y2
                                })
                                
                                track_result['n_crossings'] += 1
                                
                elif boundary['orientation'] == 'vertical' or boundary['orientation'] == 'x=c':
                    # Vertical line: x = c
                    boundary_x = boundary.get('x', boundary.get('intercept', 0))
                    
                    # Check each pair of consecutive points
                    for i in range(len(track_data) - 1):
                        x1, x2 = x[i], x[i+1]
                        
                        # Check if the boundary is crossed
                        if (x1 <= boundary_x and x2 > boundary_x) or (x1 > boundary_x and x2 <= boundary_x):
                            # Calculate crossing point
                            y1, y2 = y[i], y[i+1]
                            t = (boundary_x - x1) / (x2 - x1) if x2 != x1 else 0
                            crossing_y = y1 + t * (y2 - y1)
                            
                            # Check if crossing is within the boundary limits
                            y_min_bound = boundary.get('y_min', y_min)
                            y_max_bound = boundary.get('y_max', y_max)
                            
                            if y_min_bound <= crossing_y <= y_max_bound:
                                # This is a valid crossing
                                crossing_frame = frames[i] + t * (frames[i+1] - frames[i])
                                crossing_time = crossing_frame * frame_interval
                                
                                # Determine crossing direction
                                direction = 'right' if x2 > x1 else 'left'
                                
                                crossing_events.append({
                                    'track_id': track_id,
                                    'boundary_id': boundaries.index(boundary),
                                    'boundary_type': 'vertical',
                                    'crossing_frame': crossing_frame,
                                    'crossing_time': crossing_time,
                                    'crossing_x': boundary_x,
                                    'crossing_y': crossing_y,
                                    'direction': direction,
                                    'pre_x': x1,
                                    'pre_y': y1,
                                    'post_x': x2,
                                    'post_y': y2
                                })
                                
                                track_result['n_crossings'] += 1
                                
                elif boundary['orientation'] == 'y=mx+b':
                    # Sloped line: y = mx + b
                    m = boundary['slope']
                    b = boundary['intercept']
                    
                    # Check each pair of consecutive points
                    for i in range(len(track_data) - 1):
                        x1, y1 = x[i], y[i]
                        x2, y2 = x[i+1], y[i+1]
                        
                        # Calculate y values on the line
                        y1_line = m * x1 + b
                        y2_line = m * x2 + b
                        
                        # Check if the line is crossed
                        if (y1 <= y1_line and y2 > y2_line) or (y1 > y1_line and y2 <= y2_line):
                            # Calculate crossing point
                            # Solve for intersection of two lines:
                            # y = y1 + (y2-y1)/(x2-x1) * (x - x1)
                            # y = m*x + b
                            if x2 != x1:
                                track_slope = (y2 - y1) / (x2 - x1)
                                track_intercept = y1 - track_slope * x1
                                
                                # x-coordinate of intersection
                                crossing_x = (track_intercept - b) / (m - track_slope) if m != track_slope else x1
                                crossing_y = m * crossing_x + b
                            else:
                                # Vertical track segment
                                crossing_x = x1
                                crossing_y = m * crossing_x + b
                            
                            # Check if crossing is within the boundary limits
                            x_min_bound = boundary.get('x_min', x_min)
                            x_max_bound = boundary.get('x_max', x_max)
                            
                            if x_min_bound <= crossing_x <= x_max_bound:
                                # Interpolate crossing frame/time
                                dist1 = np.sqrt((crossing_x - x1)**2 + (crossing_y - y1)**2)
                                dist_total = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                t = dist1 / dist_total if dist_total > 0 else 0
                                
                                crossing_frame = frames[i] + t * (frames[i+1] - frames[i])
                                crossing_time = crossing_frame * frame_interval
                                
                                # Determine crossing direction (above/below line)
                                # Check if point is above or below the line y = mx + b
                                # If y > mx + b, point is above the line
                                pre_side = 'above' if y1 > y1_line else 'below'
                                post_side = 'above' if y2 > y2_line else 'below'
                                direction = f"{pre_side}_to_{post_side}"
                                
                                crossing_events.append({
                                    'track_id': track_id,
                                    'boundary_id': boundaries.index(boundary),
                                    'boundary_type': 'sloped',
                                    'crossing_frame': crossing_frame,
                                    'crossing_time': crossing_time,
                                    'crossing_x': crossing_x,
                                    'crossing_y': crossing_y,
                                    'direction': direction,
                                    'pre_x': x1,
                                    'pre_y': y1,
                                    'post_x': x2,
                                    'post_y': y2
                                })
                                
                                track_result['n_crossings'] += 1
                                
                elif boundary['orientation'] == 'x=my+b':
                    # Sloped line: x = my + b
                    m = boundary['slope']
                    b = boundary['intercept']
                    
                    # Check each pair of consecutive points
                    for i in range(len(track_data) - 1):
                        x1, y1 = x[i], y[i]
                        x2, y2 = x[i+1], y[i+1]
                        
                        # Calculate x values on the line
                        x1_line = m * y1 + b
                        x2_line = m * y2 + b
                        
                        # Check if the line is crossed
                        if (x1 <= x1_line and x2 > x2_line) or (x1 > x1_line and x2 <= x2_line):
                            # Calculate crossing point
                            # Solve for intersection of two lines:
                            # x = x1 + (x2-x1)/(y2-y1) * (y - y1)
                            # x = m*y + b
                            if y2 != y1:
                                track_slope = (x2 - x1) / (y2 - y1)
                                track_intercept = x1 - track_slope * y1
                                
                                # y-coordinate of intersection
                                crossing_y = (track_intercept - b) / (m - track_slope) if m != track_slope else y1
                                crossing_x = m * crossing_y + b
                            else:
                                # Horizontal track segment
                                crossing_y = y1
                                crossing_x = m * crossing_y + b
                            
                            # Check if crossing is within the boundary limits
                            y_min_bound = boundary.get('y_min', y_min)
                            y_max_bound = boundary.get('y_max', y_max)
                            
                            if y_min_bound <= crossing_y <= y_max_bound:
                                # Interpolate crossing frame/time
                                dist1 = np.sqrt((crossing_x - x1)**2 + (crossing_y - y1)**2)
                                dist_total = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                t = dist1 / dist_total if dist_total > 0 else 0
                                
                                crossing_frame = frames[i] + t * (frames[i+1] - frames[i])
                                crossing_time = crossing_frame * frame_interval
                                
                                # Determine crossing direction (left/right of line)
                                # Check if point is left or right of the line x = my + b
                                # If x > my + b, point is right of the line
                                pre_side = 'right' if x1 > x1_line else 'left'
                                post_side = 'right' if x2 > x2_line else 'left'
                                direction = f"{pre_side}_to_{post_side}"
                                
                                crossing_events.append({
                                    'track_id': track_id,
                                    'boundary_id': boundaries.index(boundary),
                                    'boundary_type': 'sloped',
                                    'crossing_frame': crossing_frame,
                                    'crossing_time': crossing_time,
                                    'crossing_x': crossing_x,
                                    'crossing_y': crossing_y,
                                    'direction': direction,
                                    'pre_x': x1,
                                    'pre_y': y1,
                                    'post_x': x2,
                                    'post_y': y2
                                })
                                
                                track_result['n_crossings'] += 1
                    
            elif boundary['type'] == 'circle':
                # Circular boundary
                center_x = boundary['center_x']
                center_y = boundary['center_y']
                radius = boundary['radius']
                
                # Check each pair of consecutive points
                for i in range(len(track_data) - 1):
                    x1, y1 = x[i], y[i]
                    x2, y2 = x[i+1], y[i+1]
                    
                    # Calculate distances from center
                    dist1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
                    dist2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
                    
                    # Check if the boundary is crossed
                    if (dist1 <= radius and dist2 > radius) or (dist1 > radius and dist2 <= radius):
                        # Calculate crossing point
                        # Solve for intersection of line segment and circle
                        dx = x2 - x1
                        dy = y2 - y1
                        
                        # Quadratic formula coefficients
                        a = dx**2 + dy**2
                        b = 2 * ((x1 - center_x) * dx + (y1 - center_y) * dy)
                        c = (x1 - center_x)**2 + (y1 - center_y)**2 - radius**2
                        
                        # Discriminant
                        discriminant = b**2 - 4 * a * c
                        
                        if discriminant >= 0 and a != 0:
                            # There is an intersection
                            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
                            
                            # Choose the t that is in [0, 1]
                            if 0 <= t1 <= 1:
                                t = t1
                            elif 0 <= t2 <= 1:
                                t = t2
                            else:
                                continue  # No intersection in this segment
                                
                            # Calculate crossing point
                            crossing_x = x1 + t * dx
                            crossing_y = y1 + t * dy
                            
                            # Interpolate crossing frame/time
                            crossing_frame = frames[i] + t * (frames[i+1] - frames[i])
                            crossing_time = crossing_frame * frame_interval
                            
                            # Determine crossing direction
                            direction = 'out' if dist1 <= radius and dist2 > radius else 'in'
                            
                            crossing_events.append({
                                'track_id': track_id,
                                'boundary_id': boundaries.index(boundary),
                                'boundary_type': 'circle',
                                'crossing_frame': crossing_frame,
                                'crossing_time': crossing_time,
                                'crossing_x': crossing_x,
                                'crossing_y': crossing_y,
                                'direction': direction,
                                'pre_x': x1,
                                'pre_y': y1,
                                'post_x': x2,
                                'post_y': y2
                            })
                            
                            track_result['n_crossings'] += 1
                
            elif boundary['type'] == 'point':
                # Point-like boundary, check proximity
                boundary_x = boundary['x']
                boundary_y = boundary['y']
                threshold = 1.0  # 1 µm threshold for proximity
                
                # Check each point in track
                for i in range(len(track_data)):
                    x_i, y_i = x[i], y[i]
                    dist = np.sqrt((x_i - boundary_x)**2 + (y_i - boundary_y)**2)
                    
                    if dist <= threshold:
                        # This point is near the boundary
                        crossing_events.append({
                            'track_id': track_id,
                            'boundary_id': boundaries.index(boundary),
                            'boundary_type': 'point',
                            'crossing_frame': frames[i],
                            'crossing_time': frames[i] * frame_interval,
                            'crossing_x': x_i,
                            'crossing_y': y_i,
                            'direction': 'proximity',
                            'pre_x': x_i,
                            'pre_y': y_i,
                            'post_x': x_i,
                            'post_y': y_i,
                            'distance': dist
                        })
                        
                        track_result['n_crossings'] += 1
        
        track_results.append(track_result)
    
    # Calculate residence times
    residence_times = []
    
    if crossing_events:
        crossing_df = pd.DataFrame(crossing_events)
        
        # Process only for linear or circular boundaries
        boundary_types = set([b['type'] for b in boundaries if 'type' in b])
        
        if 'line' in boundary_types or 'circle' in boundary_types:
            # Group by track_id
            for track_id, track_data in grouped:
                # Skip short tracks
                if len(track_data) < min_track_length:
                    continue
                    
                # Get crossings for this track
                track_crossings = crossing_df[crossing_df['track_id'] == track_id]
                
                if len(track_crossings) < 2:
                    continue
                    
                # Sort by crossing time
                track_crossings = track_crossings.sort_values('crossing_time')
                
                # Calculate residence times for each region
                for boundary_id, boundary in enumerate(boundaries):
                    if boundary['type'] != 'line' and boundary['type'] != 'circle':
                        continue
                        
                    # Get crossings for this boundary
                    boundary_crossings = track_crossings[track_crossings['boundary_id'] == boundary_id]
                    
                    if len(boundary_crossings) < 2:
                        continue
                        
                    # Calculate residence times
                    for i in range(len(boundary_crossings) - 1):
                        crossing1 = boundary_crossings.iloc[i]
                        crossing2 = boundary_crossings.iloc[i+1]
                        
                        # Skip if directions are the same (should be alternating in/out or up/down)
                        if crossing1['direction'] == crossing2['direction']:
                            continue
                            
                        # Calculate residence time
                        residence_time = crossing2['crossing_time'] - crossing1['crossing_time']
                        
                        if residence_time > 0:
                            # Determine region based on boundary type and direction
                            if boundary['type'] == 'line':
                                if boundary['orientation'] == 'horizontal' or boundary['orientation'] == 'y=c':
                                    region = 'above' if crossing1['direction'] == 'up' else 'below'
                                elif boundary['orientation'] == 'vertical' or boundary['orientation'] == 'x=c':
                                    region = 'right' if crossing1['direction'] == 'right' else 'left'
                                else:
                                    # Sloped line
                                    dirs = crossing1['direction'].split('_to_')
                                    region = dirs[1] if len(dirs) > 1 else 'unknown'
                            elif boundary['type'] == 'circle':
                                region = 'inside' if crossing1['direction'] == 'in' else 'outside'
                            else:
                                region = 'unknown'
                                
                            residence_times.append({
                                'track_id': track_id,
                                'boundary_id': boundary_id,
                                'region': region,
                                'start_time': crossing1['crossing_time'],
                                'end_time': crossing2['crossing_time'],
                                'residence_time': residence_time,
                                'start_frame': crossing1['crossing_frame'],
                                'end_frame': crossing2['crossing_frame']
                            })
    
    # Combine results
    results = {
        'success': True if crossing_events else False,
        'crossing_events': pd.DataFrame(crossing_events) if crossing_events else pd.DataFrame(),
        'track_results': pd.DataFrame(track_results) if track_results else pd.DataFrame(),
        'residence_times': pd.DataFrame(residence_times) if residence_times else pd.DataFrame(),
        'boundaries': boundaries
    }
    
    # Calculate ensemble statistics
    if crossing_events:
        crossing_df = pd.DataFrame(crossing_events)
        track_results_df = pd.DataFrame(track_results)
        
        results['ensemble_results'] = {
            'n_tracks_analyzed': len(track_results_df),
            'n_tracks_with_crossings': len(track_results_df[track_results_df['n_crossings'] > 0]),
            'crossing_fraction': len(track_results_df[track_results_df['n_crossings'] > 0]) / len(track_results_df),
            'n_crossing_events': len(crossing_df),
            'mean_crossings_per_track': track_results_df['n_crossings'].mean(),
            'max_crossings_per_track': track_results_df['n_crossings'].max()
        }
        
        # Boundary-specific statistics
        for boundary_id in range(len(boundaries)):
            boundary_crossings = crossing_df[crossing_df['boundary_id'] == boundary_id]
            
            if len(boundary_crossings) > 0:
                results['ensemble_results'][f'boundary{boundary_id}_n_crossings'] = len(boundary_crossings)
                results['ensemble_results'][f'boundary{boundary_id}_fraction'] = len(boundary_crossings) / len(crossing_df)
                
                # Direction statistics
                if 'direction' in boundary_crossings.columns:
                    direction_counts = boundary_crossings['direction'].value_counts()
                    for direction, count in direction_counts.items():
                        results['ensemble_results'][f'boundary{boundary_id}_{direction}_count'] = count
                        results['ensemble_results'][f'boundary{boundary_id}_{direction}_fraction'] = count / len(boundary_crossings)
        
        # Residence time statistics
        if residence_times:
            residence_df = pd.DataFrame(residence_times)
            
            results['ensemble_results']['n_residence_periods'] = len(residence_df)
            results['ensemble_results']['mean_residence_time'] = residence_df['residence_time'].mean()
            results['ensemble_results']['median_residence_time'] = residence_df['residence_time'].median()
            results['ensemble_results']['max_residence_time'] = residence_df['residence_time'].max()
            
            # Region-specific statistics
            for region in residence_df['region'].unique():
                region_data = residence_df[residence_df['region'] == region]
                
                if len(region_data) > 0:
                    results['ensemble_results'][f'region_{region}_n_periods'] = len(region_data)
                    results['ensemble_results'][f'region_{region}_mean_residence'] = region_data['residence_time'].mean()
                    results['ensemble_results'][f'region_{region}_median_residence'] = region_data['residence_time'].median()
    
    # Add missing keys that the UI expects
    results['class_transitions'] = {}
    results['dwell_times'] = {}
    results['crossing_tracks'] = []
    results['total_tracks'] = len(tracks_df['track_id'].unique()) if 'track_id' in tracks_df.columns else 0
    
    return results
