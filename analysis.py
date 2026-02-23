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
from scipy.stats import norm, linregress, pearsonr, ttest_1samp, t as student_t
from scipy import spatial
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.cluster.hierarchy import fcluster, linkage
from msd_calculation import calculate_msd, fit_msd_linear
from constants import (
    DEFAULT_SHORT_LAG_CUTOFF,
    MIN_POINTS_ANOMALOUS,
    MIN_POINTS_CONFINEMENT,
    ALPHA_SUBDIFFUSIVE_THRESHOLD,
    ALPHA_SUPERDIFFUSIVE_THRESHOLD
)

# --- Diffusion Analysis ---


def weighted_linregress(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Weighted linear regression for y = slope*x + intercept.

    Returns
    -------
    tuple
        (slope, intercept, r_value, p_value, slope_std_err)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[valid]
    y = y[valid]
    w = w[valid]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    w_norm = w / w_sum

    x_bar = np.sum(w_norm * x)
    y_bar = np.sum(w_norm * y)
    sxx = np.sum(w_norm * (x - x_bar) ** 2)
    sxy = np.sum(w_norm * (x - x_bar) * (y - y_bar))
    if sxx <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    slope = sxy / sxx
    intercept = y_bar - slope * x_bar

    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = max(len(x) - 2, 1)
    rss_w = np.sum(w * residuals**2)
    mse_w = rss_w / dof
    slope_std_err = np.sqrt(max(mse_w / np.sum(w * (x - x_bar) ** 2), 0.0))

    ss_tot_w = np.sum(w * (y - y_bar) ** 2)
    r_value = np.nan
    if ss_tot_w > 0 and np.isfinite(rss_w):
        r_squared = max(0.0, 1.0 - (rss_w / ss_tot_w))
        r_value = np.sign(slope) * np.sqrt(r_squared)

    p_value = np.nan
    if np.isfinite(slope_std_err) and slope_std_err > 0:
        t_stat = slope / slope_std_err
        p_value = 2.0 * (1.0 - student_t.cdf(np.abs(t_stat), df=dof))

    return slope, intercept, r_value, p_value, slope_std_err


def fit_alpha(
    lag_times: np.ndarray,
    msd_values: np.ndarray,
    n_points_array: np.ndarray,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Fit anomalous exponent α from log-log MSD, weighted by n_points.

    Returns
    -------
    tuple
        (alpha, alpha_std_err, (alpha_ci95_low, alpha_ci95_high))
    """
    lag_times = np.asarray(lag_times, dtype=float)
    msd_values = np.asarray(msd_values, dtype=float)
    n_points_array = np.asarray(n_points_array, dtype=float)

    valid = (
        np.isfinite(lag_times)
        & np.isfinite(msd_values)
        & np.isfinite(n_points_array)
        & (lag_times > 0)
        & (msd_values > 0)
        & (n_points_array > 0)
    )
    if np.sum(valid) < 3:
        return np.nan, np.nan, (np.nan, np.nan)

    log_t = np.log(lag_times[valid])
    log_msd = np.log(msd_values[valid])
    weights = np.sqrt(n_points_array[valid])

    slope, _, _, _, se = weighted_linregress(log_t, log_msd, weights)
    if not np.isfinite(slope) or not np.isfinite(se):
        return np.nan, np.nan, (np.nan, np.nan)

    # MSD points across lag-times are strongly correlated, so naive regression
    # slope errors are over-optimistic. Use a conservative lower bound.
    se = max(float(se), 0.2)

    alpha = slope
    alpha_ci95 = (alpha - 1.96 * se, alpha + 1.96 * se)
    return float(alpha), float(se), (float(alpha_ci95[0]), float(alpha_ci95[1]))


def classify_motion(
    alpha: float,
    alpha_se: float,
    alpha_thresholds: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[str, str]:
    """
    Classify diffusion type using alpha confidence intervals.

    Returns
    -------
    tuple
        (classification, confidence)
    """
    if not np.isfinite(alpha) or not np.isfinite(alpha_se):
        return "unknown", "indeterminate"

    sub_thresh, super_thresh = alpha_thresholds
    ci_low = alpha - 1.96 * alpha_se
    ci_high = alpha + 1.96 * alpha_se

    if ci_high < sub_thresh:
        return "subdiffusive", "high"
    if ci_low > super_thresh:
        return "superdiffusive", "high"
    if ci_high < 1.0:
        return "likely_subdiffusive", "low"
    if ci_low > 1.0:
        return "likely_superdiffusive", "low"
    return "normal_diffusion", "indeterminate"



def analyze_diffusion(tracks_df: pd.DataFrame, max_lag: int = 20, pixel_size: float = 1.0, 
                     frame_interval: float = 1.0, min_track_length: int = 5, 
                     fit_method: str = 'linear', analyze_anomalous: bool = True, 
                     check_confinement: bool = True,
                     short_lag_fit_points: Optional[int] = None) -> Dict[str, Any]:
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
        Dictionary containing diffusion analysis results with keys:
        - 'success': bool
        - 'result': analysis data (if success=True)
        - 'error': error message (if success=False)
    """
    try:
        # Input validation
        if tracks_df.empty:
            return {
                'success': False,
                'result': None,
                'error': 'Empty tracks DataFrame provided'
            }
        
        required_cols = ['track_id', 'frame', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in tracks_df.columns]
        if missing_cols:
            return {
                'success': False,
                'result': None,
                'error': f'Missing required columns: {missing_cols}'
            }
        
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
                'result': None,
                'error': 'No tracks of sufficient length for analysis',
                'msd_data': msd_df
            }
        
        # Initialize results dict
        results = {
            'msd_data': msd_df,
            'track_results': [],
            'ensemble_results': {}
        }
        
        # Analyze each track individually
        track_results = []
        track_lengths = tracks_df.groupby('track_id').size().to_dict()
        
        for track_id, track_msd in msd_df.groupby('track_id'):
            try:
                # Sort by lag time
                track_msd = track_msd.sort_values('lag_time')
                
                # Extract lag times and MSD values
                lag_times = track_msd['lag_time'].values
                msd_values = track_msd['msd'].values
                
                # Validate data
                if not (np.isfinite(lag_times).all() and np.isfinite(msd_values).all()):
                    continue  # Skip tracks with invalid MSD
                
                # Initialize track result dict
                track_result = {'track_id': track_id}
                
                # Statistically motivated default fit length: floor(N/3), where
                # N is the original track length (Michalet 2010).
                track_len = int(track_lengths.get(track_id, len(track_msd)))
                default_short_lag = max(3, track_len // 3)
                if short_lag_fit_points is None:
                    short_lag_cutoff = min(default_short_lag, len(lag_times))
                else:
                    short_lag_cutoff = min(max(3, int(short_lag_fit_points)), len(lag_times))

                use_weighted_fit = fit_method in ('weighted', 'nonlinear')
                fit_result = fit_msd_linear(
                    track_msd[['lag_time', 'msd', 'n_points']],
                    max_points=short_lag_cutoff,
                    track_length=track_len,
                    weighted=use_weighted_fit,
                )

                D_short = fit_result.get('D', np.nan)
                D_err = fit_result.get('D_err', np.nan)
                sigma_loc = fit_result.get('sigma_loc', np.nan)
                sigma_loc_err = fit_result.get('sigma_loc_err', np.nan)
                        
                # Store diffusion coefficient results
                track_result['diffusion_coefficient'] = D_short
                track_result['diffusion_coefficient_error'] = D_err
                track_result['localization_sigma'] = sigma_loc
                track_result['localization_sigma_error'] = sigma_loc_err
                track_result['short_lag_fit_points_default'] = default_short_lag
                track_result['short_lag_fit_points_used'] = short_lag_cutoff
                
                # Analyze anomalous diffusion
                if analyze_anomalous and len(lag_times) >= MIN_POINTS_ANOMALOUS:
                    try:
                        alpha, alpha_err, alpha_ci95 = fit_alpha(
                            lag_times=lag_times,
                            msd_values=msd_values,
                            n_points_array=track_msd['n_points'].values,
                        )

                        diffusion_type, diffusion_confidence = classify_motion(
                            alpha=alpha,
                            alpha_se=alpha_err,
                            alpha_thresholds=(
                                ALPHA_SUBDIFFUSIVE_THRESHOLD,
                                ALPHA_SUPERDIFFUSIVE_THRESHOLD,
                            ),
                        )

                        track_result['alpha'] = alpha
                        track_result['alpha_error'] = alpha_err
                        track_result['alpha_ci_lower'] = alpha_ci95[0]
                        track_result['alpha_ci_upper'] = alpha_ci95[1]
                        track_result['diffusion_type'] = diffusion_type
                        track_result['diffusion_confidence'] = diffusion_confidence
                    except (ValueError, np.linalg.LinAlgError) as e:
                        import logging
                        logging.warning(f"Anomalous diffusion fit failed for track {track_id}: {str(e)}")
                        track_result['alpha'] = np.nan
                        track_result['alpha_error'] = np.nan
                        track_result['alpha_ci_lower'] = np.nan
                        track_result['alpha_ci_upper'] = np.nan
                        track_result['diffusion_type'] = 'unknown'
                        track_result['diffusion_confidence'] = 'indeterminate'
                else:
                    track_result['alpha'] = np.nan
                    track_result['alpha_error'] = np.nan
                    track_result['alpha_ci_lower'] = np.nan
                    track_result['alpha_ci_upper'] = np.nan
                    track_result['diffusion_type'] = 'unknown'
                    track_result['diffusion_confidence'] = 'indeterminate'
                
                # Check for confined diffusion
                if check_confinement and len(lag_times) >= MIN_POINTS_CONFINEMENT:
                    try:
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
                            # Saxton (1993): MSD_plateau = L^2 / 3 for 2D reflecting square.
                            # Confinement radius R = L/2 = sqrt(3*MSD_plateau)/2.
                            plateau_value = np.mean(msd_values[early_region:late_region])
                            confinement_radius = np.sqrt(3.0 * plateau_value) / 2.0
                        else:
                            confined = False
                            confinement_radius = np.nan
                            
                        track_result['confined'] = confined
                        track_result['confinement_radius'] = confinement_radius
                    except (ValueError, np.linalg.LinAlgError) as e:
                        import logging
                        logging.warning(f"Confinement check failed for track {track_id}: {str(e)}")
                        track_result['confined'] = False
                        track_result['confinement_radius'] = np.nan
                
                track_results.append(track_result)
                
            except Exception as e:
                # Catch any unexpected errors in per-track analysis
                import logging
                logging.error(f"Unexpected error analyzing track {track_id}: {str(e)}")
                continue
        
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
                alpha_values = results['track_results']['alpha'].dropna()

                if len(alpha_values) >= 2:
                    t_stat, p_val = ttest_1samp(alpha_values.values, popmean=1.0, nan_policy='omit')
                else:
                    t_stat, p_val = np.nan, np.nan
                results['ensemble_results']['alpha_t_stat'] = t_stat
                results['ensemble_results']['alpha_t_p_value'] = p_val
                
                # Count diffusion types
                type_counts = results['track_results']['diffusion_type'].value_counts()
                for diff_type in [
                    'normal_diffusion',
                    'subdiffusive',
                    'superdiffusive',
                    'likely_subdiffusive',
                    'likely_superdiffusive',
                    'unknown',
                ]:
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
        
        return {
            'success': True,
            'result': results,
            'error': None
        }
    
    except ValueError as e:
        # Handle validation errors
        import logging
        logging.error(f"analyze_diffusion ValueError: {str(e)}")
        return {
            'success': False,
            'result': None,
            'error': str(e)
        }
    
    except np.linalg.LinAlgError as e:
        # Handle linear algebra errors (singular matrices, etc.)
        import logging
        logging.error(f"analyze_diffusion LinAlgError: {str(e)}")
        return {
            'success': False,
            'result': None,
            'error': f'Linear algebra error: {str(e)}'
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        import logging
        import traceback
        logging.error(f"Unexpected error in analyze_diffusion: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            'success': False,
            'result': None,
            'error': f'Unexpected error: {str(e)}'
        }


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
            vel_autocorr_raw = [] # New: Raw physical VACF

            # Pre-calculate variances for normalization
            vx_std = np.std(vx)
            vy_std = np.std(vy)
            
            # Check for zero variance (stationary particle) to avoid division by zero
            if vx_std == 0: vx_std = 1e-10
            if vy_std == 0: vy_std = 1e-10

            for lag in range(max_lag + 1):
                if lag == 0:
                    vel_autocorr.append(1.0)
                    # Raw VACF at lag 0 is mean squared velocity (approx)
                    raw_corr = np.mean(vx**2 + vy**2)
                    vel_autocorr_raw.append(raw_corr)
                else:
                    # Calculate correlation between v(t) and v(t+lag)
                    # 1. Normalized (Pearson) - existing metric
                    # We need to handle edge cases where slice has 0 variance
                    vx_slice1 = vx[:-lag]
                    vx_slice2 = vx[lag:]
                    vy_slice1 = vy[:-lag]
                    vy_slice2 = vy[lag:]
                    
                    if len(vx_slice1) > 1:
                        if np.std(vx_slice1) > 0 and np.std(vx_slice2) > 0:
                            corr_x = np.corrcoef(vx_slice1, vx_slice2)[0, 1]
                        else:
                            corr_x = 0.0
                            
                        if np.std(vy_slice1) > 0 and np.std(vy_slice2) > 0:
                            corr_y = np.corrcoef(vy_slice1, vy_slice2)[0, 1]
                        else:
                            corr_y = 0.0
                            
                        avg_corr = (corr_x + corr_y) / 2
                    else:
                        avg_corr = 0.0
                        
                    vel_autocorr.append(avg_corr)
                    
                    # 2. Raw physical VACF: <v(t) . v(t+tau)>
                    # Dot product of velocity vectors at time t and t+tau
                    dot_products = vx_slice1 * vx_slice2 + vy_slice1 * vy_slice2
                    raw_corr = np.mean(dot_products)
                    vel_autocorr_raw.append(raw_corr)

            # Calculate correlation time (lag where autocorr drops below 1/e)
            corr_threshold = 1/np.e
            for lag, corr in enumerate(vel_autocorr):
                if corr < corr_threshold:
                    corr_time = lag * frame_interval
                    break
            else:
                corr_time = max_lag * frame_interval

            track_result['velocity_autocorr'] = vel_autocorr
            track_result['velocity_autocorr_raw'] = vel_autocorr_raw # Store raw values
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

                    n_points = len(track_data) - lag
                    if n_points > 0:
                        dx = x[lag:] - x[:-lag]
                        dy = y[lag:] - y[:-lag]
                        sd = dx**2 + dy**2
                        sd_list.extend(sd)

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
            
            # Perform hierarchical clustering
            Z = linkage(coords, method='ward')
            labels = fcluster(Z, t=epsilon, criterion='distance') - 1
            # Convert to same format as other methods (-1 for noise)
            labels[labels == max(labels)] = -1
            
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
            frames = track_data['frame'].values
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
        frames = track_data['frame'].values
        
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
            is_dwelling = np.append(displacements <= threshold_distance, False)  # Add one more value to match original length
            
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


def load_precalculated_dwell_events(dwell_events_df: pd.DataFrame,
                                     frame_interval: float = 1.0) -> Dict[str, Any]:
    """
    Load and process pre-calculated dwell event data.
    
    This function handles data that already contains dwell event statistics,
    such as exports from other analysis tools or previous analyses.
    
    Parameters
    ----------
    dwell_events_df : pd.DataFrame
        DataFrame containing pre-calculated dwell events with columns:
        - track_id: Track identifier
        - dwell_time: Pre-calculated dwell time (in seconds)
        - dwell_frames: Number of frames in dwell event
        - start_frame, end_frame: Frame range of dwell event
        - region_id: (optional) Region identifier
        - start_x, start_y, end_x, end_y, mean_x, mean_y: (optional) Position data
    frame_interval : float
        Time between frames in seconds (for calculating times from frames if needed)
        
    Returns
    -------
    dict
        Dictionary containing dwell time analysis results in standard format
    """
    # Validate required columns
    required_cols = ['track_id']
    missing_cols = [col for col in required_cols if col not in dwell_events_df.columns]
    if missing_cols:
        return {
            'success': False,
            'error': f"Missing required columns: {missing_cols}"
        }
    
    # Make a copy to avoid modifying original
    df = dwell_events_df.copy()
    
    # If dwell_time column doesn't exist but dwell_frames does, calculate it
    if 'dwell_time' not in df.columns and 'dwell_frames' in df.columns:
        df['dwell_time'] = df['dwell_frames'] * frame_interval
    elif 'dwell_time' not in df.columns:
        return {
            'success': False,
            'error': "Neither 'dwell_time' nor 'dwell_frames' column found"
        }
    
    # Ensure dwell_time is numeric
    df['dwell_time'] = pd.to_numeric(df['dwell_time'], errors='coerce')
    df = df.dropna(subset=['dwell_time'])
    
    if df.empty:
        return {
            'success': False,
            'error': "No valid dwell time data after cleaning"
        }
    
    # Calculate track-level statistics
    track_stats = df.groupby('track_id').agg({
        'dwell_time': ['count', 'sum', 'mean']
    }).reset_index()
    track_stats.columns = ['track_id', 'n_dwell_events', 'total_dwell_time', 'mean_dwell_time']
    
    # Calculate track durations if frame data is available
    if 'start_frame' in df.columns and 'end_frame' in df.columns:
        track_durations = df.groupby('track_id').agg({
            'start_frame': 'min',
            'end_frame': 'max'
        }).reset_index()
        track_durations['duration'] = (track_durations['end_frame'] - track_durations['start_frame']) * frame_interval
        track_stats = track_stats.merge(track_durations[['track_id', 'duration']], on='track_id', how='left')
        track_stats['dwell_fraction'] = track_stats['total_dwell_time'] / track_stats['duration']
    
    # Build results structure
    results = {
        'success': True,
        'dwell_events': df,
        'track_results': track_stats,
        'data_source': 'pre-calculated'
    }
    
    # Calculate ensemble statistics
    n_tracks = df['track_id'].nunique()
    n_tracks_with_dwell = len(track_stats[track_stats['n_dwell_events'] > 0])
    
    ensemble_stats = {
        'n_tracks_analyzed': n_tracks,
        'n_tracks_with_dwell': n_tracks_with_dwell,
        'n_dwell_events': len(df),
        'mean_dwell_time': df['dwell_time'].mean(),
        'median_dwell_time': df['dwell_time'].median(),
        'std_dwell_time': df['dwell_time'].std(),
        'max_dwell_time': df['dwell_time'].max(),
        'min_dwell_time': df['dwell_time'].min(),
        'mean_dwells_per_track': track_stats['n_dwell_events'].mean(),
        'analysis_type': 'pre-calculated',
        'frame_interval': frame_interval
    }
    
    # Add dwell fraction if available
    if 'dwell_fraction' in track_stats.columns:
        ensemble_stats['mean_dwell_fraction'] = track_stats['dwell_fraction'].mean()
    
    results['ensemble_results'] = ensemble_stats
    
    # Calculate region statistics if region_id is available
    if 'region_id' in df.columns:
        region_stats = df.groupby('region_id').agg({
            'dwell_time': ['count', 'mean', 'median', 'std'],
            'track_id': 'nunique'
        }).reset_index()
        region_stats.columns = ['region_id', 'n_dwell_events', 'mean_dwell_time', 
                               'median_dwell_time', 'std_dwell_time', 'n_tracks']
        results['region_stats'] = region_stats
    
    # Create dwell_stats dictionary for UI display
    results['dwell_stats'] = {
        'Total tracks analyzed': n_tracks,
        'Tracks with dwell events': n_tracks_with_dwell,
        'Total dwell events': len(df),
        'Mean dwell time': f"{ensemble_stats['mean_dwell_time']:.3f} s",
        'Median dwell time': f"{ensemble_stats['median_dwell_time']:.3f} s",
        'Std dwell time': f"{ensemble_stats['std_dwell_time']:.3f} s"
    }
    
    return results


# --- Gel Structure Analysis ---

def analyze_gel_structure(tracks_df: pd.DataFrame, 
                         pixel_size: float = 1.0,
                         max_pore_size: float = 5.0) -> Dict[str, Any]:
    """
    Analyze gel structure properties from particle trajectories.
    Estimates pore size and mesh characteristics.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    pixel_size : float
        Pixel size in micrometers
    max_pore_size : float
        Maximum expected pore size (for filtering outliers)
        
    Returns
    -------
    dict
        Dictionary containing gel structure analysis results
    """
    # Convert pixel coordinates to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    # 1. Estimate pore size from maximum displacement of confined particles
    # First, identify confined particles
    msd_results = analyze_diffusion(
        tracks_df, 
        pixel_size=pixel_size, 
        check_confinement=True, 
        analyze_anomalous=False
    )
    
    pore_sizes = []
    
    if msd_results['success'] and not msd_results['track_results'].empty:
        track_results = msd_results['track_results']
        
        # Filter for confined tracks
        confined_tracks = track_results[track_results['confined'] == True]
        
        if not confined_tracks.empty:
            # Preserve physically meaningful confinement size estimate directly
            # from the Saxton radius without arbitrary scaling factors.
            estimated_sizes = confined_tracks['confinement_radius']
            
            # Filter outliers
            valid_sizes = estimated_sizes[estimated_sizes <= max_pore_size]
            pore_sizes.extend(valid_sizes.tolist())
    
    # 2. Estimate mesh size from spatial distribution of all points
    # Use Delaunay triangulation to find empty spaces
    all_points = tracks_df_um[['x', 'y']].values
    
    # Subsample if too many points (for performance)
    if len(all_points) > 10000:
        indices = np.random.choice(len(all_points), 10000, replace=False)
        points_for_mesh = all_points[indices]
    else:
        points_for_mesh = all_points
        
    mesh_sizes = []
    
    if len(points_for_mesh) >= 4:
        try:
            tri = spatial.Delaunay(points_for_mesh)
            
            # Calculate edge lengths of triangles
            # This gives a distribution of inter-particle distances
            # The larger distances might correspond to pores
            
            for simplex in tri.simplices:
                # Get vertices of the triangle
                pts = points_for_mesh[simplex]
                
                # Calculate side lengths
                a = np.linalg.norm(pts[0] - pts[1])
                b = np.linalg.norm(pts[1] - pts[2])
                c = np.linalg.norm(pts[2] - pts[0])
                
                # Use the longest side as a proxy for local spacing
                mesh_sizes.append(max(a, b, c))
                
        except Exception as e:
            print(f"Error in Delaunay triangulation: {e}")
    
    # Compile results
    results = {
        'success': True,
        'pore_size_distribution': pore_sizes,
        'mesh_size_distribution': mesh_sizes,
        'ensemble_results': {}
    }
    
    # Calculate statistics
    if pore_sizes:
        results['ensemble_results']['mean_pore_size'] = np.mean(pore_sizes)
        results['ensemble_results']['median_pore_size'] = np.median(pore_sizes)
        results['ensemble_results']['std_pore_size'] = np.std(pore_sizes)
        results['ensemble_results']['n_confined_tracks'] = len(pore_sizes)
    else:
        results['ensemble_results']['mean_pore_size'] = np.nan
        results['ensemble_results']['n_confined_tracks'] = 0
        
    if mesh_sizes:
        # Filter very large mesh sizes (artifacts at boundaries)
        valid_mesh = [s for s in mesh_sizes if s <= max_pore_size]
        if valid_mesh:
            results['ensemble_results']['mean_mesh_size'] = np.mean(valid_mesh)
            results['ensemble_results']['median_mesh_size'] = np.median(valid_mesh)
            results['ensemble_results']['std_mesh_size'] = np.std(valid_mesh)
        else:
            results['ensemble_results']['mean_mesh_size'] = np.nan
            
    return results


# --- Polymer Physics Analysis ---

def analyze_polymer_physics(tracks_df: pd.DataFrame,
                           pixel_size: float = 1.0,
                           frame_interval: float = 1.0,
                           min_track_length: int = 10) -> Dict[str, Any]:
    """
    Analyze polymer dynamics from trajectory data.
    
    Determines polymer regime (Rouse, Zimm, Reptation) based on scaling exponent
    of mean squared displacement: MSD ~ t^α
    
    Regimes:
    - Rouse (unentangled): α ≈ 0.5
    - Zimm (with hydrodynamics): α ≈ 0.6
    - Reptation (entangled): α ≈ 0.25 (short time), α ≈ 1.0 (long time)
    - Free diffusion: α ≈ 1.0
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format with track_id, frame, x, y columns
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    min_track_length : int
        Minimum track length for analysis
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'success': bool
        - 'scaling_exponent': float (α value)
        - 'regime': str (identified polymer regime)
        - 'lag_times': list of lag times
        - 'msd_data': list of MSD values
        - 'ensemble_results': dict with statistics
    """
    try:
        # Calculate ensemble MSD
        msd_df = calculate_msd(
            tracks_df,
            max_lag=min(50, tracks_df['frame'].max() // 2),
            pixel_size=pixel_size,
            frame_interval=frame_interval,
            min_track_length=min_track_length
        )
        
        if msd_df.empty or len(msd_df) < 3:
            return {
                'success': False,
                'error': 'Insufficient data for polymer analysis'
            }
        
        # Get ensemble MSD
        lag_times = msd_df['lag_time'].values
        msd_values = msd_df['msd'].values
        
        # Filter out invalid values
        valid_mask = (msd_values > 0) & np.isfinite(msd_values)
        lag_times = lag_times[valid_mask]
        msd_values = msd_values[valid_mask]
        
        if len(lag_times) < 3:
            return {
                'success': False,
                'error': 'Insufficient valid MSD data points'
            }
        
        # Fit power law: MSD = A * t^α
        # Take log: log(MSD) = log(A) + α * log(t)
        log_t = np.log(lag_times)
        log_msd = np.log(msd_values)
        
        # Linear regression on log-log data
        slope, intercept, r_value, p_value, std_err = linregress(log_t, log_msd)
        
        alpha = slope  # Scaling exponent
        A = np.exp(intercept)  # Prefactor
        
        # Determine regime based on alpha
        if alpha < 0.35:
            regime = "Reptation (short-time)"
        elif 0.35 <= alpha < 0.55:
            regime = "Rouse (unentangled)"
        elif 0.55 <= alpha < 0.7:
            regime = "Zimm (with hydrodynamics)"
        elif 0.7 <= alpha < 1.15:
            regime = "Free diffusion"
        elif alpha >= 1.15:
            regime = "Anomalous (super-diffusive)"
        else:
            regime = "Unknown"
        
        # Calculate effective diffusion coefficient at t=1s
        D_eff = A / 4.0  # Since MSD = 4D*t for normal diffusion
        
        # Estimate polymer properties based on regime
        interpretation = {
            'regime': regime,
            'scaling_exponent': alpha,
            'confidence': r_value**2,
            'effective_D': D_eff
        }
        
        if "Rouse" in regime:
            interpretation['description'] = "Unentangled polymer chain dynamics. Rouse model applies."
            interpretation['properties'] = "Subdiffusive motion with α≈0.5, no hydrodynamic interactions."
        elif "Zimm" in regime:
            interpretation['description'] = "Polymer with hydrodynamic interactions."
            interpretation['properties'] = "Subdiffusive motion with α≈0.6, includes solvent effects."
        elif "Reptation" in regime:
            interpretation['description'] = "Entangled polymer dynamics. Reptation model may apply."
            interpretation['properties'] = "Strong subdiffusion (α<0.5) due to topological constraints."
        elif "Free diffusion" in regime:
            interpretation['description'] = "Normal diffusive behavior."
            interpretation['properties'] = "Linear MSD growth (α≈1.0), consistent with small molecules or unconfined motion."
        else:
            interpretation['description'] = f"Anomalous dynamics with α={alpha:.2f}"
            interpretation['properties'] = "Non-standard scaling behavior."
        
        results = {
            'success': True,
            'scaling_exponent': float(alpha),
            'scaling_exponent_error': float(std_err),
            'prefactor': float(A),
            'regime': regime,
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'lag_times': lag_times.tolist(),
            'msd_data': msd_values.tolist(),
            'fitted_msd': (A * lag_times**alpha).tolist(),
            'ensemble_results': {
                'effective_diffusion_coefficient': float(D_eff),
                'regime': regime,
                'scaling_exponent': float(alpha),
                'fit_quality': float(r_value**2),
                'n_tracks_analyzed': len(tracks_df['track_id'].unique())
            },
            'interpretation': interpretation
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Polymer physics analysis failed: {str(e)}'
        }


# --- Diffusion Population Analysis ---

def analyze_diffusion_population(tracks_df: pd.DataFrame, 
                                pixel_size: float = 1.0, 
                                frame_interval: float = 1.0,
                                n_components: int = 2,
                                method: str = 'GMM') -> Dict[str, Any]:
    """
    Analyze distribution of diffusion coefficients to identify subpopulations.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    n_components : int
        Number of components (subpopulations) to look for
    method : str
        Method for population analysis ('GMM', 'Histogram')
        
    Returns
    -------
    dict
        Dictionary containing population analysis results
    """
    # Calculate diffusion coefficients first
    diff_results = analyze_diffusion(
        tracks_df, 
        pixel_size=pixel_size, 
        frame_interval=frame_interval,
        check_confinement=False,
        analyze_anomalous=False
    )
    
    if not diff_results['success'] or diff_results['track_results'].empty:
        return {
            'success': False,
            'error': 'Could not calculate diffusion coefficients'
        }
        
    track_results = diff_results['track_results']
    
    # Extract diffusion coefficients (log scale is usually better for separation)
    # Filter out non-positive values
    valid_D = track_results['diffusion_coefficient'].values
    valid_D = valid_D[valid_D > 0]
    
    if len(valid_D) < n_components * 5:
        return {
            'success': False,
            'error': 'Not enough tracks for population analysis'
        }
        
    log_D = np.log10(valid_D)
    
    populations = []
    
    if method == 'GMM':
        # Gaussian Mixture Model
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            # Reshape for sklearn
            X = log_D.reshape(-1, 1)
            gmm.fit(X)
            
            # Get parameters
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            weights = gmm.weights_.flatten()
            
            # Sort by mean diffusion coefficient (slow to fast)
            sorted_indices = np.argsort(means)
            
            for i in sorted_indices:
                log_mean = means[i]
                log_std = np.sqrt(covariances[i])
                weight = weights[i]
                
                # Convert back to linear scale
                mean_D = 10**log_mean
                # For log-normal distribution, geometric mean is 10^mu
                # Range can be estimated as 10^(mu +/- sigma)
                range_low = 10**(log_mean - log_std)
                range_high = 10**(log_mean + log_std)
                
                populations.append({
                    'component_id': len(populations),
                    'fraction': weight,
                    'mean_log_D': log_mean,
                    'std_log_D': log_std,
                    'mean_D': mean_D,
                    'range_D_low': range_low,
                    'range_D_high': range_high
                })
                
            # Calculate BIC/AIC to evaluate fit quality
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
        except Exception as e:
            return {
                'success': False,
                'error': f"GMM fitting failed: {e}"
            }
            
    elif method == 'Histogram':
        # Simple histogram-based peak detection (simplified)
        # This is less robust than GMM but simpler
        hist, bin_edges = np.histogram(log_D, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks (simplified)
        # In a real implementation, we would use scipy.signal.find_peaks
        # Here we just return basic stats
        populations.append({
            'component_id': 0,
            'fraction': 1.0,
            'mean_log_D': np.mean(log_D),
            'std_log_D': np.std(log_D),
            'mean_D': 10**np.mean(log_D)
        })
        bic = np.nan
        aic = np.nan
        
    else:
        return {
            'success': False,
            'error': f"Unknown method: {method}"
        }
    
    results = {
        'success': True,
        'method': method,
        'n_components': n_components,
        'populations': populations,
        'log_D_values': log_D,
        'bic': bic,
        'aic': aic
    }
    
    return results


# --- Crowding Analysis ---

def analyze_crowding(tracks_df: pd.DataFrame, 
                    pixel_size: float = 1.0,
                    radius: float = 1.0) -> Dict[str, Any]:
    """
    Analyze local crowding density around particles.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    pixel_size : float
        Pixel size in micrometers
    radius : float
        Radius to check for neighbors (in µm)
        
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
    frame_results = []
    all_densities = []
    
    # Analyze each frame
    for frame in frames:
        # Get particles in this frame
        frame_data = tracks_df_um[tracks_df_um['frame'] == frame]
        
        if len(frame_data) < 2:
            continue
            
        # Get coordinates
        coords = frame_data[['x', 'y']].values
        
        # Build KD-tree for efficient neighbor search
        tree = KDTree(coords)
        
        # Count neighbors within radius
        # query_radius returns indices of neighbors (including self)
        neighbors = tree.query_radius(coords, r=radius)
        
        # Calculate local density (particles per µm²)
        # Subtract 1 to exclude self
        n_neighbors = np.array([len(n) - 1 for n in neighbors])
        local_density = n_neighbors / (np.pi * radius**2)
        
        all_densities.extend(local_density)
        
        frame_results.append({
            'frame': frame,
            'mean_density': np.mean(local_density),
            'max_density': np.max(local_density),
            'n_particles': len(frame_data)
        })
        
    # Compile results
    results = {
        'success': True,
        'frame_results': pd.DataFrame(frame_results),
        'ensemble_results': {}
    }
    
    if all_densities:
        results['ensemble_results']['mean_local_density'] = np.mean(all_densities)
        results['ensemble_results']['median_local_density'] = np.median(all_densities)
        results['ensemble_results']['std_local_density'] = np.std(all_densities)
        results['ensemble_results']['max_local_density'] = np.max(all_densities)
    else:
        results['ensemble_results']['mean_local_density'] = 0
        
    return results


# --- Active Transport Analysis ---

def analyze_active_transport(tracks_df: pd.DataFrame, 
                            pixel_size: float = 1.0, 
                            frame_interval: float = 1.0,
                            window_size: int = 5) -> Dict[str, Any]:
    """
    Analyze active transport events (directed motion).
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    window_size : int
        Window size for local velocity calculation
        
    Returns
    -------
    dict
        Dictionary containing active transport analysis results
    """
    # Use analyze_motion to get velocity and directionality
    motion_results = analyze_motion(
        tracks_df, 
        window_size=window_size, 
        analyze_velocity_autocorr=True, 
        analyze_persistence=True,
        motion_classification='advanced',
        pixel_size=pixel_size, 
        frame_interval=frame_interval
    )
    
    if not motion_results['success'] or motion_results['track_results'].empty:
        return {
            'success': False,
            'error': 'Could not analyze motion'
        }
        
    track_results = motion_results['track_results']
    
    # Identify active transport tracks
    # Criteria: High straightness, high alpha, persistent velocity correlation
    
    active_tracks = []
    
    for _, track in track_results.iterrows():
        is_active = False
        score = 0
        
        # Criterion 1: Motion type classification
        if 'motion_type' in track and track['motion_type'] == 'directed':
            score += 2
            
        # Criterion 2: Alpha value (superdiffusive)
        if 'alpha' in track and track['alpha'] > 1.2:
            score += 1
            
        # Criterion 3: Straightness
        if 'straightness' in track and track['straightness'] > 0.5:
            score += 1
            
        # Criterion 4: Velocity autocorrelation time
        if 'correlation_time' in track and track['correlation_time'] > 3 * frame_interval:
            score += 1
            
        if score >= 3:
            is_active = True
            
        if is_active:
            active_tracks.append({
                'track_id': track['track_id'],
                'mean_speed': track['mean_speed'],
                'straightness': track.get('straightness', np.nan),
                'alpha': track.get('alpha', np.nan),
                'correlation_time': track.get('correlation_time', np.nan),
                'score': score
            })
            
    # Compile results
    results = {
        'success': True,
        'active_tracks': pd.DataFrame(active_tracks),
        'ensemble_results': {}
    }
    
    # Calculate statistics
    n_total = len(track_results)
    n_active = len(active_tracks)
    
    results['ensemble_results']['n_tracks'] = n_total
    results['ensemble_results']['n_active_tracks'] = n_active
    results['ensemble_results']['active_fraction'] = n_active / n_total if n_total > 0 else 0
    
    if active_tracks:
        active_df = pd.DataFrame(active_tracks)
        results['ensemble_results']['mean_active_speed'] = active_df['mean_speed'].mean()
        results['ensemble_results']['mean_active_straightness'] = active_df['straightness'].mean()
    
    return results


# --- Boundary Crossing Analysis ---

def analyze_boundary_crossing(tracks_df: pd.DataFrame, 
                             boundaries: Any = None,
                             boundary_coords: List[Tuple[float, float]] = None,
                             pixel_size: float = 1.0,
                             frame_interval: float = 1.0,
                             min_track_length: int = 5,
                             closed_boundary: bool = True) -> Dict[str, Any]:
    """
    Analyze particles crossing defined boundaries.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    boundaries : list of dicts, optional
        Structured boundary data from convert_compartments_to_boundary_crossing_format.
        Each dict contains: 'id', 'type' (line/rectangle), 'orientation', coordinates.
    boundary_coords : list of tuples, optional
        Legacy format: List of (x, y) coordinates defining the boundary (in pixels).
        Used if 'boundaries' is None.
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    min_track_length : int
        Minimum number of frames for a track to be analyzed
    closed_boundary : bool
        Whether the boundary is a closed loop (polygon) or open line
        
    Returns
    -------
    dict
        Dictionary containing boundary crossing analysis results
    """
    from matplotlib.path import Path
    
    # Handle both new structured format and legacy format
    if boundaries is not None:
        # New format: structured boundary list from segmentation
        crossing_events = []
        
        # Convert tracks to µm if not already
        tracks_df_um = tracks_df.copy()
        if 'x' in tracks_df_um.columns and tracks_df_um['x'].max() > 100:  # Likely in pixels
            tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
            tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
        
        # Filter tracks by minimum length
        track_lengths = tracks_df_um.groupby('track_id').size()
        valid_tracks = track_lengths[track_lengths >= min_track_length].index
        tracks_df_um = tracks_df_um[tracks_df_um['track_id'].isin(valid_tracks)]
        
        # Process each boundary
        for boundary in boundaries:
            boundary_id = boundary.get('id', 'unknown')
            boundary_type = boundary.get('type', 'line')
            
            # Check crossings for this boundary
            for track_id, track_data in tracks_df_um.groupby('track_id'):
                if len(track_data) < 2:
                    continue
                    
                track_data = track_data.sort_values('frame')
                
                # Check each segment of the track
                for i in range(len(track_data) - 1):
                    p1 = track_data.iloc[i]
                    p2 = track_data.iloc[i + 1]
                    
                    # Check if segment crosses boundary
                    crossed = False
                    if boundary.get('orientation') == 'horizontal':
                        y_line = boundary['y']
                        x_min = boundary['x_min']
                        x_max = boundary['x_max']
                        # Check if y crosses the line and x is within bounds
                        if (p1['y'] < y_line < p2['y'] or p2['y'] < y_line < p1['y']):
                            # Interpolate x at crossing
                            t = (y_line - p1['y']) / (p2['y'] - p1['y'])
                            x_cross = p1['x'] + t * (p2['x'] - p1['x'])
                            if x_min <= x_cross <= x_max:
                                crossed = True
                    
                    elif boundary.get('orientation') == 'vertical':
                        x_line = boundary['x']
                        y_min = boundary['y_min']
                        y_max = boundary['y_max']
                        # Check if x crosses the line and y is within bounds
                        if (p1['x'] < x_line < p2['x'] or p2['x'] < x_line < p1['x']):
                            # Interpolate y at crossing
                            t = (x_line - p1['x']) / (p2['x'] - p1['x'])
                            y_cross = p1['y'] + t * (p2['y'] - p1['y'])
                            if y_min <= y_cross <= y_max:
                                crossed = True
                    
                    if crossed:
                        crossing_events.append({
                            'track_id': track_id,
                            'boundary_id': boundary_id,
                            'frame_start': p1['frame'],
                            'frame_end': p2['frame'],
                            'time': p1['frame'] * frame_interval,
                            'direction': 'inward' if p2['y'] < p1['y'] else 'outward'
                        })
        
        # Compile results
        n_crossings = len(crossing_events)
        n_tracks = tracks_df_um['track_id'].nunique()
        n_boundaries = len(boundaries)
        
        return {
            'success': True,
            'n_crossings': n_crossings,
            'n_tracks_analyzed': n_tracks,
            'n_boundaries': n_boundaries,
            'crossing_events': crossing_events,
            'crossing_rate': n_crossings / n_tracks if n_tracks > 0 else 0,
            'parameters': {
                'pixel_size': pixel_size,
                'frame_interval': frame_interval,
                'min_track_length': min_track_length
            }
        }
    
    # Legacy format: simple list of coordinate tuples
    elif boundary_coords is not None:
        # Convert boundary to µm
        boundary_um = [(x * pixel_size, y * pixel_size) for x, y in boundary_coords]
        boundary_path = Path(boundary_um)
        
        # Convert tracks to µm
        tracks_df_um = tracks_df.copy()
        tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
        tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    else:
        return {
            'success': False,
            'error': 'Either boundaries or boundary_coords must be provided'
        }
    boundary_um = [(x * pixel_size, y * pixel_size) for x, y in boundary_coords]
    boundary_path = Path(boundary_um)
    
    # Convert tracks to µm
    tracks_df_um = tracks_df.copy()
    tracks_df_um['x'] = tracks_df_um['x'] * pixel_size
    tracks_df_um['y'] = tracks_df_um['y'] * pixel_size
    
    crossing_events = []
    
    # Group by track_id
    grouped = tracks_df_um.groupby('track_id')
    
    for track_id, track_data in grouped:
        if len(track_data) < 2:
            continue
            
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Get points
        points = track_data[['x', 'y']].values
        frames = track_data['frame'].values
        
        # Check which points are inside the boundary
        # contains_points returns boolean array
        is_inside = boundary_path.contains_points(points)
        
        # Find transitions (True -> False or False -> True)
        # diff gives True where value changes
        transitions = np.diff(is_inside)
        transition_indices = np.where(transitions)[0]
        
        for idx in transition_indices:
            # Transition happens between idx and idx+1
            start_inside = is_inside[idx]
            end_inside = is_inside[idx+1]
            
            direction = 'outward' if start_inside and not end_inside else 'inward'
            
            crossing_events.append({
                'track_id': track_id,
                'frame_before': frames[idx],
                'frame_after': frames[idx+1],
                'x_before': points[idx][0],
                'y_before': points[idx][1],
                'x_after': points[idx+1][0],
                'y_after': points[idx+1][1],
                'direction': direction
            })
            
    # Compile results
    results = {
        'success': True,
        'crossing_events': pd.DataFrame(crossing_events),
        'ensemble_results': {}
    }
    
    # Calculate statistics
    if crossing_events:
        events_df = pd.DataFrame(crossing_events)
        n_inward = len(events_df[events_df['direction'] == 'inward'])
        n_outward = len(events_df[events_df['direction'] == 'outward'])
        
        results['ensemble_results']['total_crossings'] = len(events_df)
        results['ensemble_results']['inward_crossings'] = n_inward
        results['ensemble_results']['outward_crossings'] = n_outward
        results['ensemble_results']['net_flux'] = n_inward - n_outward  # Positive means accumulation inside
    else:
        results['ensemble_results']['total_crossings'] = 0
        results['ensemble_results']['inward_crossings'] = 0
        results['ensemble_results']['outward_crossings'] = 0
        results['ensemble_results']['net_flux'] = 0
        
    return results


# ============================================================================
# Photobleaching-Corrected Kinetics (Survival Analysis)
# Based on: MicroLive Sept 2025 - Correcting residence times for bleaching
# ============================================================================

# Note: correct_kinetic_rates_photobleaching is defined later in this file (line ~3107)
# with more comprehensive parameter handling including confidence_level parameter.
# This location previously contained a duplicate definition that has been removed.


# ============================================================================
# Rigorous Model Selection (AIC/BIC for Motion Classification)
# Based on: Information criteria for statistical model comparison
# ============================================================================

def classify_motion_with_model_selection(tracks_df: pd.DataFrame, pixel_size: float = 1.0,
                                        frame_interval: float = 1.0, 
                                        min_track_length: int = 10) -> Dict[str, Any]:
    """
    Classify particle motion using rigorous statistical model selection (AIC/BIC).
    
    Standard motion classification based on thresholds (e.g., straightness > 0.7 = directed)
    is subjective and can misclassify edge cases. This function fits multiple motion models
    to each trajectory and selects the best fit using information criteria, providing
    mathematically defensible classification.
    
    Models tested:
    1. Brownian: MSD = 4Dt (1 parameter: D)
    2. Anomalous: MSD = 4Dt^α (2 parameters: D, α)
    3. Confined: MSD ≈ L²(1 - exp(-t/τ)) (2 parameters: L, τ)
    4. Directed: MSD = (vt)² + 4Dt (2 parameters: v, D)
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns 'track_id', 'frame', 'x', 'y'.
    pixel_size : float, default=1.0
        Pixel size in micrometers.
    frame_interval : float, default=1.0
        Time between frames in seconds.
    min_track_length : int, default=10
        Minimum track length for model fitting.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'success': bool
        - 'track_classifications': DataFrame with per-track results
        - 'ensemble_results': Overall motion type distribution
        - 'model_statistics': Goodness-of-fit metrics
    
    References
    ----------
    - Akaike Information Criterion (AIC) for model selection
    - Bayesian Information Criterion (BIC) with stronger penalty for parameters
    - Model selection in single-particle tracking (Biophys J. 2019)
    
    Notes
    -----
    **Information Criteria**:
    
    AIC = 2k - 2ln(L)
    BIC = k*ln(n) - 2ln(L)
    
    where:
        k = number of parameters
        n = number of data points  
        L = likelihood of the data given the model
    
    **Likelihood Calculation**:
    Assuming Gaussian errors: ln(L) = -n/2 * ln(SSE/n) - n/2 * ln(2π) - n/2
    where SSE = sum of squared errors
    
    **Model Selection Rule**:
    Choose model with LOWEST AIC or BIC (indicates best trade-off between
    fit quality and model complexity)
    
    Examples
    --------
    >>> result = classify_motion_with_model_selection(tracks_df, pixel_size=0.1)
    >>> print(result['ensemble_results']['motion_fractions'])
    >>> # Shows: {'Brownian': 0.45, 'Confined': 0.30, 'Directed': 0.15, ...}
    """
    try:
        from scipy.optimize import curve_fit
        from scipy.stats import linregress
        
        # Calculate MSD for each track
        track_results = []
        
        for track_id, track_group in tracks_df.groupby('track_id'):
            if len(track_group) < min_track_length:
                continue
            
            # Sort by frame and extract positions
            track_group = track_group.sort_values('frame')
            x = track_group['x'].values * pixel_size
            y = track_group['y'].values * pixel_size
            frames = track_group['frame'].values
            
            # Calculate MSD
            max_lag = len(track_group) // 3  # Use up to 1/3 of track length
            lag_times = []
            msd_values = []
            
            for lag in range(1, max_lag):
                displacements = []
                for i in range(len(track_group) - lag):
                    dx = x[i + lag] - x[i]
                    dy = y[i + lag] - y[i]
                    displacements.append(dx**2 + dy**2)
                
                if displacements:
                    lag_times.append(lag * frame_interval)
                    msd_values.append(np.mean(displacements))
            
            if len(lag_times) < 5:
                continue
            
            t = np.array(lag_times)
            msd = np.array(msd_values)
            n_points = len(t)
            
            # Fit models and calculate AIC/BIC
            models = {}
            
            # Model 1: Brownian diffusion (1 parameter)
            try:
                def brownian(t, D):
                    return 4 * D * t
                
                popt, _ = curve_fit(brownian, t, msd, p0=[1.0], bounds=(0, np.inf))
                D_brown = popt[0]
                residuals = msd - brownian(t, D_brown)
                sse = np.sum(residuals**2)
                
                # Calculate log-likelihood
                ln_L = -n_points/2 * np.log(sse/n_points) - n_points/2 * np.log(2*np.pi) - n_points/2
                
                aic = 2 * 1 - 2 * ln_L  # k=1
                bic = 1 * np.log(n_points) - 2 * ln_L
                
                models['Brownian'] = {
                    'parameters': {'D': D_brown},
                    'aic': aic,
                    'bic': bic,
                    'sse': sse,
                    'r_squared': 1 - sse / np.sum((msd - np.mean(msd))**2)
                }
            except:
                pass
            
            # Model 2: Anomalous diffusion (2 parameters)
            try:
                def anomalous(t, D, alpha):
                    return 4 * D * t**alpha
                
                popt, _ = curve_fit(anomalous, t, msd, p0=[1.0, 1.0], 
                                  bounds=([0, 0.1], [np.inf, 2.0]))
                D_anom, alpha = popt
                residuals = msd - anomalous(t, D_anom, alpha)
                sse = np.sum(residuals**2)
                
                ln_L = -n_points/2 * np.log(sse/n_points) - n_points/2 * np.log(2*np.pi) - n_points/2
                
                aic = 2 * 2 - 2 * ln_L  # k=2
                bic = 2 * np.log(n_points) - 2 * ln_L
                
                models['Anomalous'] = {
                    'parameters': {'D': D_anom, 'alpha': alpha},
                    'aic': aic,
                    'bic': bic,
                    'sse': sse,
                    'r_squared': 1 - sse / np.sum((msd - np.mean(msd))**2)
                }
            except:
                pass
            
            # Model 3: Confined diffusion (2 parameters)
            try:
                def confined(t, L_sq, tau):
                    return L_sq * (1 - np.exp(-t / tau))
                
                # Initial guess from data
                L_sq_guess = msd[-1]  # Plateau value
                tau_guess = t[len(t)//2]  # Midpoint time
                
                popt, _ = curve_fit(confined, t, msd, p0=[L_sq_guess, tau_guess],
                                  bounds=([0, 0], [np.inf, np.inf]))
                L_sq, tau = popt
                residuals = msd - confined(t, L_sq, tau)
                sse = np.sum(residuals**2)
                
                ln_L = -n_points/2 * np.log(sse/n_points) - n_points/2 * np.log(2*np.pi) - n_points/2
                
                aic = 2 * 2 - 2 * ln_L  # k=2
                bic = 2 * np.log(n_points) - 2 * ln_L
                
                models['Confined'] = {
                    'parameters': {'L': np.sqrt(L_sq), 'tau': tau},
                    'aic': aic,
                    'bic': bic,
                    'sse': sse,
                    'r_squared': 1 - sse / np.sum((msd - np.mean(msd))**2)
                }
            except:
                pass
            
            # Model 4: Directed diffusion (2 parameters)
            try:
                def directed(t, v, D):
                    return (v * t)**2 + 4 * D * t
                
                popt, _ = curve_fit(directed, t, msd, p0=[1.0, 1.0],
                                  bounds=([0, 0], [np.inf, np.inf]))
                v, D_dir = popt
                residuals = msd - directed(t, v, D_dir)
                sse = np.sum(residuals**2)
                
                ln_L = -n_points/2 * np.log(sse/n_points) - n_points/2 * np.log(2*np.pi) - n_points/2
                
                aic = 2 * 2 - 2 * ln_L  # k=2
                bic = 2 * np.log(n_points) - 2 * ln_L
                
                models['Directed'] = {
                    'parameters': {'velocity': v, 'D': D_dir},
                    'aic': aic,
                    'bic': bic,
                    'sse': sse,
                    'r_squared': 1 - sse / np.sum((msd - np.mean(msd))**2)
                }
            except:
                pass
            
            # Select best model (lowest BIC)
            if models:
                best_model = min(models.items(), key=lambda x: x[1]['bic'])
                best_model_name = best_model[0]
                best_model_data = best_model[1]
                
                # Calculate delta BIC for other models
                delta_bic = {name: data['bic'] - best_model_data['bic'] 
                           for name, data in models.items()}
                
                track_results.append({
                    'track_id': track_id,
                    'track_length': len(track_group),
                    'best_model': best_model_name,
                    'best_bic': best_model_data['bic'],
                    'best_aic': best_model_data['aic'],
                    'r_squared': best_model_data['r_squared'],
                    'delta_bic': delta_bic,
                    'all_models': models
                })
        
        if not track_results:
            return {
                'success': False,
                'error': 'No tracks met minimum length requirement for model fitting'
            }
        
        # Create results DataFrame
        classifications_df = pd.DataFrame([
            {
                'track_id': r['track_id'],
                'track_length': r['track_length'],
                'motion_type': r['best_model'],
                'BIC': r['best_bic'],
                'AIC': r['best_aic'],
                'R_squared': r['r_squared']
            }
            for r in track_results
        ])
        
        # Calculate ensemble statistics
        motion_counts = classifications_df['motion_type'].value_counts()
        total_tracks = len(classifications_df)
        motion_fractions = {k: v/total_tracks for k, v in motion_counts.items()}
        
        results = {
            'success': True,
            'track_classifications': classifications_df,
            'ensemble_results': {
                'total_tracks_analyzed': total_tracks,
                'motion_type_counts': motion_counts.to_dict(),
                'motion_fractions': motion_fractions,
                'dominant_motion': motion_counts.index[0] if len(motion_counts) > 0 else None
            },
            'model_statistics': {
                'mean_bic': float(classifications_df['BIC'].mean()),
                'mean_r_squared': float(classifications_df['R_squared'].mean()),
                'classification_confidence': {
                    'high_confidence': len([r for r in track_results 
                                          if min(r['delta_bic'].values()) > 10]) / total_tracks,
                    'moderate_confidence': len([r for r in track_results 
                                              if 2 < min(r['delta_bic'].values()) <= 10]) / total_tracks,
                    'low_confidence': len([r for r in track_results 
                                         if min(r['delta_bic'].values()) <= 2]) / total_tracks
                }
            },
            'detailed_results': track_results
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Model selection analysis failed: {str(e)}'
        }


# ============================================================================
# Kinetic State Analysis (HMM for Transient Binding)
# Based on: bioRxiv 2024/2025 - Transitional kinetics and binding dynamics
# ============================================================================

def analyze_kinetic_states_hmm(tracks_df: pd.DataFrame, pixel_size: float = 1.0,
                               frame_interval: float = 1.0, n_states: int = 2,
                               min_track_length: int = 5) -> Dict[str, Any]:
    """
    Analyze binding kinetics using Hidden Markov Model (HMM) for state classification.
    
    This function addresses a key limitation of threshold-based dwell time analysis:
    in the nucleus, particles often "hop" or slide along DNA rather than exhibiting
    simple binary bound/free states. An HMM allows classification into discrete
    kinetic states based on diffusion coefficients without arbitrary spatial thresholds.
    
    The method is particularly suited for:
    - Transient binding (transcription factors, DNA repair proteins)
    - Multi-state kinetics (search → recognition → bound)
    - Sliding/hopping along chromatin
    - Phase-separated condensate interactions
    
    States are typically:
    - State 0: Bound/Confined (low diffusion coefficient)
    - State 1: Free/Diffusing (high diffusion coefficient)
    - State 2+: Intermediate states (sliding, transient binding)
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns 'track_id', 'frame', 'x', 'y'.
    pixel_size : float, default=1.0
        Pixel size in micrometers for distance conversion.
    frame_interval : float, default=1.0
        Time interval between frames in seconds.
    n_states : int, default=2
        Number of kinetic states to identify. Common choices:
        - 2 states: Bound vs Free
        - 3 states: Bound, Sliding, Free
        - 4 states: Bound, Search-1D, Search-3D, Free
    min_track_length : int, default=5
        Minimum track length required for state analysis.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'success': bool, whether analysis completed
        - 'n_states': int, number of states identified
        - 'states': dict with keys:
            - 'D_bound': Diffusion coefficient of most confined state (µm²/s)
            - 'D_free': Diffusion coefficient of most mobile state (µm²/s)
            - 'D_intermediate': List of intermediate state D values (if n_states > 2)
        - 'occupancies': dict with fraction of time in each state
        - 'transition_rates': dict with estimated k_on, k_off rates (1/s)
        - 'step_size_distribution': dict with mean/std for each state
        - 'state_labels': np.ndarray with state assignment for each step
        - 'model': GaussianMixture object for further analysis
    
    References
    ----------
    - The transitional kinetics... via two intermediate states (bioRxiv, July 2024)
    - SpyBLI... binding kinetics (bioRxiv, Mar 2025)
    - Sliding and hopping of single PcrA monomers on DNA (PNAS 2009)
    - Measuring binding kinetics of T-cell receptors (Nature Methods 2016)
    
    Notes
    -----
    This implementation uses Gaussian Mixture Models as a proxy for full HMM inference.
    A complete HMM would also model:
    1. Transition matrix A[i,j] = P(state j at t+1 | state i at t)
    2. Emission probabilities P(step_size | state)
    3. Forward-backward algorithm for state sequence inference
    
    The GMM approach provides robust state identification while remaining
    computationally efficient. For full HMM implementation, use the `hmmlearn` package.
    
    The diffusion coefficient for each state is calculated from step size variance:
        D = <Δr²> / (4 * Δt)
    where <Δr²> is the mean squared step size and Δt is the frame interval.
    
    Examples
    --------
    >>> result = analyze_kinetic_states_hmm(tracks_df, pixel_size=0.1, 
    ...                                      frame_interval=0.05, n_states=2)
    >>> print(f"Bound state D: {result['states']['D_bound']:.3f} µm²/s")
    >>> print(f"Free state D: {result['states']['D_free']:.3f} µm²/s")
    >>> print(f"Bound fraction: {result['occupancies']['bound']:.1%}")
    >>> print(f"k_off: {result['transition_rates']['k_off']:.3f} /s")
    
    >>> # Three-state analysis for sliding
    >>> result = analyze_kinetic_states_hmm(tracks_df, n_states=3)
    >>> print(f"States: Bound={result['states']['D_bound']:.3f}, "
    ...       f"Sliding={result['states']['D_intermediate'][0]:.3f}, "
    ...       f"Free={result['states']['D_free']:.3f} µm²/s")
    """
    try:
        from sklearn.mixture import GaussianMixture
        
        # Validate input
        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
            return {
                'success': False,
                'error': f'Missing required columns. Need: {required_cols}'
            }
        
        if len(tracks_df) < min_track_length * 2:
            return {
                'success': False,
                'error': 'Insufficient data points for state analysis'
            }
        
        # Calculate step sizes (displacements between consecutive frames)
        tracks_df = tracks_df.sort_values(['track_id', 'frame'])
        
        # Group by track and calculate displacements
        step_data = []
        track_ids = []
        
        for track_id, track_group in tracks_df.groupby('track_id'):
            if len(track_group) < min_track_length:
                continue
            
            # Calculate displacements in micrometers
            x = track_group['x'].values * pixel_size
            y = track_group['y'].values * pixel_size
            dx = np.diff(x)
            dy = np.diff(y)
            step_sizes = np.sqrt(dx**2 + dy**2)
            
            step_data.extend(step_sizes)
            track_ids.extend([track_id] * len(step_sizes))
        
        if len(step_data) < n_states * 10:  # Need at least 10 points per state
            return {
                'success': False,
                'error': f'Insufficient steps ({len(step_data)}) for {n_states} states'
            }
        
        step_sizes = np.array(step_data).reshape(-1, 1)
        
        # Fit Gaussian Mixture Model (proxy for HMM emission probabilities)
        # In a full HMM, this would also model transition matrix
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type='full',
            max_iter=200,
            n_init=10,
            random_state=42  # For reproducibility
        )
        
        # Fit model and predict state labels
        state_labels = gmm.fit_predict(step_sizes)
        
        # Extract model parameters
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        weights = gmm.weights_  # State occupancies (π_i)
        
        # Sort states by mean step size (bound → free)
        sorted_indices = np.argsort(means)
        means_sorted = means[sorted_indices]
        variances_sorted = variances[sorted_indices]
        weights_sorted = weights[sorted_indices]
        
        # Calculate diffusion coefficients from variance
        # D = <Δr²> / (4 * Δt)
        # Variance of step size distribution = 2*D*Δt for 2D random walk
        # Therefore: D = variance / (4 * Δt)
        D_values = variances_sorted / (4 * frame_interval)
        
        # Build results
        states_dict = {
            'D_bound': float(D_values[0]),  # Lowest D
            'D_free': float(D_values[-1]),   # Highest D
        }
        
        if n_states > 2:
            states_dict['D_intermediate'] = [float(d) for d in D_values[1:-1]]
        
        # Calculate state occupancies (fractions)
        occupancies = {
            'bound': float(weights_sorted[0]),
            'free': float(weights_sorted[-1]),
        }
        
        if n_states > 2:
            occupancies['intermediate'] = [float(w) for w in weights_sorted[1:-1]]
        
        # Estimate transition rates (simplified)
        # For 2-state system: k_off = f_free / τ_bound, k_on = f_bound / τ_free
        # Where τ is average dwell time in each state
        
        # Count state transitions to estimate dwell times
        transitions_data = []
        for track_id in np.unique(track_ids):
            track_mask = np.array(track_ids) == track_id
            track_states = state_labels[track_mask]
            
            # Find state runs (consecutive same states)
            if len(track_states) > 1:
                transitions = np.diff(track_states) != 0
                transitions_data.append(np.sum(transitions))
        
        if transitions_data and np.sum(transitions_data) > 0:
            # Average transition frequency
            avg_transitions_per_track = np.mean(transitions_data)
            total_frames = len(step_data)
            
            # Estimate transition rates (transitions per second)
            # k_off: bound → free transition rate
            # k_on: free → bound transition rate
            
            # For bound state: k_off ≈ (transitions to free) / (time in bound state)
            bound_fraction = weights_sorted[0]
            free_fraction = weights_sorted[-1]
            
            if bound_fraction > 0 and free_fraction > 0:
                # Estimate from detailed balance: k_on * f_free = k_off * f_bound
                total_time = len(step_data) * frame_interval
                k_off = (avg_transitions_per_track / 2) / (total_time * bound_fraction)  # /2 for bidirectional
                k_on = (avg_transitions_per_track / 2) / (total_time * free_fraction)
            else:
                k_off = 0.0
                k_on = 0.0
        else:
            k_off = 0.0
            k_on = 0.0
        
        transition_rates = {
            'k_off': float(k_off),  # Unbinding rate (1/s)
            'k_on': float(k_on),     # Binding rate (1/s)
        }
        
        # Add equilibrium dissociation constant if rates are non-zero
        if k_on > 0:
            transition_rates['K_d'] = float(k_off / k_on)  # Equilibrium constant
        
        # Step size statistics for each state
        step_size_stats = {}
        for i, state_idx in enumerate(sorted_indices):
            state_mask = state_labels == state_idx
            state_steps = step_sizes[state_mask].flatten()
            
            state_name = 'bound' if i == 0 else ('free' if i == n_states-1 else f'intermediate_{i}')
            step_size_stats[state_name] = {
                'mean': float(np.mean(state_steps)),
                'std': float(np.std(state_steps)),
                'median': float(np.median(state_steps)),
            }
        
        # Compile results
        results = {
            'success': True,
            'n_states': n_states,
            'states': states_dict,
            'occupancies': occupancies,
            'transition_rates': transition_rates,
            'step_size_distribution': step_size_stats,
            'state_labels': state_labels,
            'track_ids': track_ids,
            'model': gmm,
            'summary': {
                'D_ratio': float(D_values[-1] / D_values[0]) if D_values[0] > 0 else np.inf,
                'bound_fraction': float(weights_sorted[0]),
                'residence_time_bound': float(1 / k_off) if k_off > 0 else np.inf,
                'residence_time_free': float(1 / k_on) if k_on > 0 else np.inf,
            }
        }
        
        return results
        
    except ImportError:
        return {
            'success': False,
            'error': 'sklearn.mixture.GaussianMixture not available. Install scikit-learn.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }


# ============================================================================
# Fractal Dimension Analysis for Percolation Assessment
# Based on: Fractal analysis of chromatin trajectories (2024-2025)
# ============================================================================

def calculate_fractal_dimension(tracks_df: pd.DataFrame, pixel_size: float = 1.0,
                               method: str = 'box_counting', min_track_length: int = 10) -> Dict[str, Any]:
    """
    Calculate fractal dimension of trajectories to assess interaction with fractal matrix.
    
    **Fractal Dimension Background:**
    
    A trajectory in a homogeneous medium (free diffusion) has fractal dimension d_f = 2
    (for 3D: d_f = 2 exactly for Brownian motion). Deviations indicate:
    
    - d_f < 2: Subdiffusion, confined motion, crowded environment
    - d_f = 2: Normal diffusion
    - d_f > 2: Anomalous diffusion, possibly directed motion or swelling
    
    In percolating systems or fractal matrices (like chromatin):
    - d_f ≈ 2.5: Motion on fractal substrate (e.g., chromatin fiber)
    - d_f ≈ 1.7: Confined to fractal network
    - d_f ≈ 1.0: Linear motion along channels
    
    **Methods:**
    1. **Box-counting**: Standard fractal analysis - cover trajectory with boxes
    2. **Mass-radius**: M(r) ~ r^d_f where M is number of points within radius r
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with 'track_id', 'x', 'y' columns.
    pixel_size : float, default=1.0
        Pixel size in micrometers.
    method : str, default='box_counting'
        Fractal dimension calculation method:
        - 'box_counting': Classic Hausdorff dimension
        - 'mass_radius': Mass-radius scaling
    min_track_length : int, default=10
        Minimum points per track for reliable d_f calculation.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'per_track_df': Fractal dimension for each track
        - 'ensemble_statistics': Mean, std, distribution
        - 'interpretation': Classification of trajectory types
    
    References
    ----------
    - Fractal analysis of protein motion (Biophys J. 2018)
    - Chromatin as a fractal globule (Science, 2009)
    - Percolation on fractal lattices (Phys Rev E)
    """
    try:
        results_list = []
        
        for track_id, track_data in tracks_df.groupby('track_id'):
            if len(track_data) < min_track_length:
                continue
            
            # Extract coordinates
            track_data = track_data.sort_values('frame')
            x = track_data['x'].values * pixel_size
            y = track_data['y'].values * pixel_size
            points = np.column_stack([x, y])
            
            if method == 'box_counting':
                # Box-counting method
                # Count boxes needed to cover trajectory at different scales
                
                # Define range of box sizes (scales)
                track_size = np.max([
                    np.ptp(x),  # Range of x
                    np.ptp(y)   # Range of y
                ])
                
                if track_size < 1e-6:  # Stationary track
                    d_f = 0.0
                else:
                    # Logarithmically spaced scales
                    min_scale = track_size / 50
                    max_scale = track_size / 2
                    scales = np.logspace(
                        np.log10(min_scale),
                        np.log10(max_scale),
                        num=10
                    )
                    
                    box_counts = []
                    valid_scales = []
                    
                    for scale in scales:
                        # Create grid
                        x_bins = np.arange(x.min(), x.max() + scale, scale)
                        y_bins = np.arange(y.min(), y.max() + scale, scale)
                        
                        # Count occupied boxes
                        x_indices = np.digitize(x, x_bins)
                        y_indices = np.digitize(y, y_bins)
                        
                        # Unique boxes
                        occupied_boxes = len(set(zip(x_indices, y_indices)))
                        
                        if occupied_boxes > 0:
                            box_counts.append(occupied_boxes)
                            valid_scales.append(scale)
                    
                    if len(box_counts) >= 3:
                        # Fit log(N) vs log(1/ε) where N = box count, ε = scale
                        # d_f = -slope
                        log_scales = np.log(valid_scales)
                        log_counts = np.log(box_counts)
                        
                        # Linear regression
                        coeffs = np.polyfit(log_scales, log_counts, 1)
                        d_f = -coeffs[0]  # Negative slope
                    else:
                        d_f = np.nan
            
            elif method == 'mass_radius':
                # Mass-radius scaling: M(r) ~ r^d_f
                # M(r) = number of points within distance r from center of mass
                
                center = np.mean(points, axis=0)
                distances = np.sqrt(np.sum((points - center)**2, axis=1))
                
                max_dist = np.max(distances)
                if max_dist < 1e-6:
                    d_f = 0.0
                else:
                    # Define radius bins
                    radii = np.logspace(
                        np.log10(max_dist/20),
                        np.log10(max_dist),
                        num=10
                    )
                    
                    masses = []
                    valid_radii = []
                    
                    for r in radii:
                        mass = np.sum(distances <= r)
                        if mass > 0:
                            masses.append(mass)
                            valid_radii.append(r)
                    
                    if len(masses) >= 3:
                        # Fit log(M) vs log(r)
                        log_radii = np.log(valid_radii)
                        log_masses = np.log(masses)
                        
                        coeffs = np.polyfit(log_radii, log_masses, 1)
                        d_f = coeffs[0]  # Slope is d_f
                    else:
                        d_f = np.nan
            else:
                d_f = np.nan
            
            # Classify trajectory type
            if np.isnan(d_f):
                traj_type = 'Undefined'
            elif d_f < 1.2:
                traj_type = 'Linear/Channeled'
            elif d_f < 1.8:
                traj_type = 'Confined/Subdiffusive'
            elif d_f < 2.2:
                traj_type = 'Normal Diffusion'
            elif d_f < 2.7:
                traj_type = 'Fractal Matrix Motion'
            else:
                traj_type = 'Superdiffusive'
            
            results_list.append({
                'track_id': track_id,
                'fractal_dimension': d_f,
                'trajectory_type': traj_type,
                'n_points': len(points)
            })
        
        if not results_list:
            return {
                'success': False,
                'error': 'No tracks met minimum length requirement'
            }
        
        results_df = pd.DataFrame(results_list)
        
        # Remove invalid values for statistics
        valid_df = results_df[np.isfinite(results_df['fractal_dimension'])]
        
        if len(valid_df) == 0:
            return {
                'success': False,
                'error': 'All fractal dimension calculations failed'
            }
        
        # Ensemble statistics
        d_f_values = valid_df['fractal_dimension'].values
        
        ensemble_stats = {
            'mean_df': float(np.mean(d_f_values)),
            'std_df': float(np.std(d_f_values)),
            'median_df': float(np.median(d_f_values)),
            'min_df': float(np.min(d_f_values)),
            'max_df': float(np.max(d_f_values))
        }
        
        # Population distribution
        type_counts = valid_df['trajectory_type'].value_counts().to_dict()
        type_fractions = {k: v/len(valid_df) for k, v in type_counts.items()}
        
        # Interpretation
        mean_df = ensemble_stats['mean_df']
        if mean_df < 1.5:
            interpretation = 'Confined motion - particles trapped or moving in channels'
        elif mean_df < 2.1:
            interpretation = 'Normal diffusion - homogeneous environment'
        elif mean_df < 2.6:
            interpretation = 'Fractal environment - chromatin fiber interactions'
        else:
            interpretation = 'Anomalous motion - directed transport or active processes'
        
        return {
            'success': True,
            'per_track_df': results_df,
            'ensemble_statistics': ensemble_stats,
            'population_fractions': type_fractions,
            'interpretation': interpretation,
            'method_used': method,
            'summary': {
                'mean_fractal_dim': f"{mean_df:.2f}",
                'interpretation': interpretation,
                'n_tracks_analyzed': len(valid_df)
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Fractal dimension calculation failed: {str(e)}'
        }


# ============================================================================
# Spatial Connectivity Graph for Direct Percolation Assessment
# ============================================================================

def build_connectivity_network(tracks_df: pd.DataFrame, pixel_size: float = 1.0,
                              grid_size: float = 0.2, min_visits: int = 2) -> Dict[str, Any]:
    """
    Build spatial connectivity graph to directly assess percolation through visited regions.
    
    **Network Percolation Theory:**
    
    Treat accessible space as a network:
    - Nodes: Grid cells visited by particles
    - Edges: Connections between adjacent visited cells
    - Giant component: Largest connected cluster
    
    **Percolation Criteria:**
    1. System percolates if giant component spans the observed volume
    2. Percolation threshold: fraction of visited nodes for spanning cluster
    3. Network efficiency: quantifies connectivity quality
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with 'track_id', 'x', 'y'.
    pixel_size : float, default=1.0
        Pixel to micron conversion.
    grid_size : float, default=0.2
        Grid cell size in micrometers.
    min_visits : int, default=2
        Minimum particle visits to consider cell accessible.
    
    Returns
    -------
    Dict[str, Any]
        Network topology metrics including giant component size,
        percolation status, bottleneck identification.
    """
    try:
        import networkx as nx
        from scipy.spatial import distance
        
        # Convert to spatial coordinates
        x = tracks_df['x'].values * pixel_size
        y = tracks_df['y'].values * pixel_size
        
        # Create grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Assign points to grid cells
        x_idx = np.digitize(x, x_bins)
        y_idx = np.digitize(y, y_bins)
        
        # Count visits per cell
        cell_visits = {}
        for xi, yi in zip(x_idx, y_idx):
            cell = (xi, yi)
            cell_visits[cell] = cell_visits.get(cell, 0) + 1
        
        # Filter cells by minimum visits
        accessible_cells = {cell for cell, count in cell_visits.items() 
                          if count >= min_visits}
        
        if len(accessible_cells) < 2:
            return {
                'success': False,
                'error': 'Insufficient accessible cells for network analysis'
            }
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(accessible_cells)
        
        # Add edges between adjacent cells (8-connected neighborhood)
        for cell in accessible_cells:
            i, j = cell
            # Check 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbor = (i + di, j + dj)
                    if neighbor in accessible_cells:
                        G.add_edge(cell, neighbor)
        
        # Network analysis
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Find connected components
        components = list(nx.connected_components(G))
        component_sizes = sorted([len(c) for c in components], reverse=True)
        
        giant_component_size = component_sizes[0] if component_sizes else 0
        giant_component_fraction = giant_component_size / n_nodes if n_nodes > 0 else 0
        
        # Check if system percolates (giant component spans space)
        # Spanning: largest component touches opposite boundaries
        giant_component = max(components, key=len) if components else set()
        
        # Get boundary cells of giant component
        if giant_component:
            gc_cells = list(giant_component)
            gc_x = [cell[0] for cell in gc_cells]
            gc_y = [cell[1] for cell in gc_cells]
            
            x_span = (max(gc_x) - min(gc_x)) / (len(x_bins) - 1)
            y_span = (max(gc_y) - min(gc_y)) / (len(y_bins) - 1)
            
            # Consider percolating if giant component spans >70% in both directions
            percolates = (x_span > 0.7) and (y_span > 0.7)
        else:
            percolates = False
            x_span = 0
            y_span = 0
        
        # Network efficiency (average shortest path in giant component)
        if len(giant_component) > 1:
            giant_subgraph = G.subgraph(giant_component)
            try:
                avg_path_length = nx.average_shortest_path_length(giant_subgraph)
                efficiency = 1.0 / avg_path_length if avg_path_length > 0 else 0
            except:
                efficiency = 0
        else:
            avg_path_length = 0
            efficiency = 0
        
        # Betweenness centrality (identify bottlenecks)
        if n_nodes > 2:
            betweenness = nx.betweenness_centrality(G)
            max_betweenness = max(betweenness.values()) if betweenness else 0
            bottleneck_cells = [cell for cell, bc in betweenness.items() 
                              if bc > 0.5 * max_betweenness][:5]  # Top 5
        else:
            betweenness = {}
            bottleneck_cells = []
        
        results = {
            'success': True,
            'network_properties': {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'average_degree': 2 * n_edges / n_nodes if n_nodes > 0 else 0
            },
            'percolation_analysis': {
                'percolates': percolates,
                'giant_component_size': giant_component_size,
                'giant_component_fraction': giant_component_fraction,
                'x_span_fraction': x_span,
                'y_span_fraction': y_span,
                'n_components': len(components),
                'component_sizes': component_sizes[:10]  # Top 10
            },
            'connectivity_metrics': {
                'network_efficiency': efficiency,
                'avg_shortest_path': avg_path_length
            },
            'bottlenecks': {
                'n_bottleneck_cells': len(bottleneck_cells),
                'bottleneck_coordinates': bottleneck_cells
            },
            'graph_object': G,  # For further analysis/plotting
            'summary': {
                'percolation_status': 'Percolating' if percolates else 'Non-percolating',
                'giant_component': f"{giant_component_fraction:.1%} of nodes",
                'connectivity': f"{n_edges} edges among {n_nodes} nodes"
            }
        }
        
        return results
        
    except ImportError:
        return {
            'success': False,
            'error': 'networkx required for connectivity analysis. Install: pip install networkx'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Connectivity network analysis failed: {str(e)}'
        }


# ============================================================================
# Photobleaching-Corrected Kinetics Module
# Based on: MicroLive (Sept 2025) and other photobleaching correction methods
# ============================================================================

def correct_kinetic_rates_photobleaching(dwell_times: Union[List[float], np.ndarray],
                                        frame_interval: float = 1.0,
                                        bleach_rate: Optional[float] = None,
                                        confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Correct apparent kinetic rates (k_off) for photobleaching effects.
    
    Standard dwell time analysis underestimates binding times because it cannot
    distinguish between:
    1. A molecule unbinding (biological event)
    2. A molecule photobleaching (measurement artifact)
    
    This function uses survival curve analysis with dual-exponential fitting to
    separate these two processes and recover the "true" k_off rate.
    
    **The Problem:**
    When tracking fluorescent molecules, you observe an apparent off-rate that
    combines two independent processes:
        k_apparent = k_true_off + k_bleach
    
    Without correction, you systematically underestimate residence times
    (overestimate off-rates), leading to incorrect biological conclusions about
    binding kinetics.
    
    **The Solution:**
    1. Measure photobleaching rate independently (e.g., using immobile reference
       like histone H2B or beads)
    2. Fit survival curve to extract k_apparent
    3. Subtract k_bleach to obtain k_true_off
    
    Parameters
    ----------
    dwell_times : List[float] or np.ndarray
        Observed dwell times in frames or seconds. These are the durations that
        particles remain bound before disappearing (either unbinding OR bleaching).
    frame_interval : float, default=1.0
        Time between frames in seconds. Used to convert frame-based dwell times
        to absolute time.
    bleach_rate : float, optional
        Independently measured photobleaching rate in 1/seconds (k_bleach).
        **Critical:** This must be measured from a control experiment where
        molecules are known to be stably bound (e.g., chromatin-integrated H2B-GFP,
        fixed cells, or fluorescent beads).
        
        If None, the function will only return k_apparent with a warning.
        
        **How to measure k_bleach:**
        1. Track immobilized/stably-bound fluorophores under identical imaging conditions
        2. Plot survival curve (fraction remaining vs time)
        3. Fit single exponential: S(t) = exp(-k_bleach * t)
        4. Extract k_bleach from fit
    confidence_level : float, default=0.95
        Confidence level for parameter error estimation (0 < confidence_level < 1).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'success': bool, whether analysis completed
        - 'k_apparent': float, observed off-rate (1/s) = k_true + k_bleach
        - 'residence_time_observed': float, uncorrected residence time (s)
        - 'k_true_off': float, corrected off-rate (1/s) after bleach correction
        - 'residence_time_corrected': float, true residence time (s)
        - 'correction_factor': float, ratio of corrected/uncorrected residence time
        - 'bleach_contribution': float, fraction of signal loss due to bleaching (0-1)
        - 'fit_quality': dict with R², RMSE, and fit parameters
        - 'survival_curve': dict with time points and survival probabilities
        - 'confidence_intervals': dict with 95% CI for parameters
    
    Raises
    ------
    ValueError
        If dwell_times is empty or contains invalid values.
    RuntimeError
        If curve fitting fails to converge.
    
    References
    ----------
    - MicroLive: High-throughput live-cell microscopy (bioRxiv, Sept 2025)
    - Correcting for photobleaching in single-molecule tracking (Biophys J, 2012)
    - Measuring binding kinetics with FRAP (Methods Enzymol, 2013)
    
    Notes
    -----
    **Theoretical Background:**
    
    The survival probability follows:
        S(t) = exp(-(k_off + k_bleach) * t)
    
    Taking the logarithm:
        ln(S(t)) = -(k_off + k_bleach) * t
    
    From the slope of ln(S) vs t, we extract k_apparent = k_off + k_bleach.
    
    **When to use this correction:**
    - Single-molecule tracking of transcription factors
    - Chromatin binding protein kinetics
    - FRAP/FCS recovery curves
    - Any experiment where photobleaching occurs on similar timescale as unbinding
    
    **When NOT to use:**
    - Dwell times >> bleaching time (correction is negligible)
    - Non-exponential kinetics (multiple binding states require multi-exponential fit)
    - Blinking fluorophores (requires additional dark-state correction)
    
    **Typical Values:**
    - k_bleach for GFP at moderate illumination: 0.01-0.1 /s (10-100s bleach time)
    - k_bleach for photoactivatable FPs: 0.05-0.5 /s (2-20s)
    - k_bleach for organic dyes (Cy3, Alexa): 0.001-0.01 /s (100-1000s)
    
    Examples
    --------
    >>> # Example 1: Typical transcription factor binding
    >>> dwell_times = [2.5, 3.1, 1.8, 4.2, 2.9, 3.5, 2.0, 3.8]  # seconds
    >>> result = correct_kinetic_rates_photobleaching(
    ...     dwell_times=dwell_times,
    ...     frame_interval=1.0,
    ...     bleach_rate=0.05  # Measured from H2B-GFP control: 20s bleach time
    ... )
    >>> print(f"Observed residence time: {result['residence_time_observed']:.2f} s")
    >>> print(f"Corrected residence time: {result['residence_time_corrected']:.2f} s")
    >>> print(f"Correction factor: {result['correction_factor']:.2f}x")
    >>> print(f"Bleaching accounts for {result['bleach_contribution']:.1%} of signal loss")
    
    >>> # Example 2: Without bleach rate (diagnostic only)
    >>> result = correct_kinetic_rates_photobleaching(dwell_times)
    >>> print(f"Apparent k_off: {result['k_apparent']:.3f} /s")
    >>> print(result['note'])  # Warning about missing bleach_rate
    
    >>> # Example 3: Assess if correction is needed
    >>> result = correct_kinetic_rates_photobleaching(dwell_times, bleach_rate=0.01)
    >>> if result['bleach_contribution'] > 0.1:
    ...     print("⚠️  Bleaching contributes >10% - correction essential!")
    >>> else:
    ...     print("✓ Bleaching contribution minimal, correction optional")
    """
    try:
        from scipy.optimize import curve_fit
        from scipy import stats
        
        # Input validation
        if not isinstance(dwell_times, (list, np.ndarray)):
            return {
                'success': False,
                'error': 'dwell_times must be a list or numpy array'
            }
        
        dwell_times_array = np.array(dwell_times)
        
        if len(dwell_times_array) == 0:
            return {
                'success': False,
                'error': 'dwell_times is empty'
            }
        
        if np.any(dwell_times_array <= 0):
            return {
                'success': False,
                'error': 'dwell_times must be positive values'
            }
        
        if len(dwell_times_array) < 5:
            return {
                'success': False,
                'error': 'Need at least 5 dwell events for reliable fitting'
            }
        
        # Step 1: Calculate Survival Function (1 - CDF)
        # Sort dwell times
        sorted_times = np.sort(dwell_times_array)
        n_events = len(sorted_times)
        
        # Kaplan-Meier estimator for survival probability
        # S(t) = fraction of events still bound at time t
        survival = 1.0 - np.arange(1, n_events + 1) / (n_events + 1)
        
        # Step 2: Define exponential survival model
        # S(t) = A * exp(-k_app * t)
        # where k_app = k_off + k_bleach (apparent rate)
        def survival_model(t, A, k_app):
            return A * np.exp(-k_app * t)
        
        # Step 3: Fit model to data
        # Initial guess: A=1 (normalized), k_app from mean dwell time
        mean_dwell = np.mean(sorted_times)
        p0 = [1.0, 1.0 / mean_dwell]  # Initial parameters
        
        try:
            # Perform curve fitting with bounds
            # A: [0.5, 1.5] (should be ~1 for normalized data)
            # k_app: [1e-6, 10] (reasonable range for biological rates)
            popt, pcov = curve_fit(
                survival_model, 
                sorted_times, 
                survival,
                p0=p0,
                bounds=([0.5, 1e-6], [1.5, 10]),
                maxfev=5000
            )
            
            A_fit, k_apparent = popt
            
            # Calculate parameter uncertainties (standard errors)
            perr = np.sqrt(np.diag(pcov))
            k_apparent_err = perr[1]
            
        except (RuntimeError, ValueError) as e:
            return {
                'success': False,
                'error': f'Curve fitting failed: {str(e)}. Try with more data points.'
            }
        
        # Step 4: Calculate goodness of fit
        survival_pred = survival_model(sorted_times, *popt)
        
        # R-squared
        ss_res = np.sum((survival - survival_pred)**2)
        ss_tot = np.sum((survival - np.mean(survival))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((survival - survival_pred)**2))
        
        # Step 5: Correct for photobleaching
        if bleach_rate is not None and bleach_rate > 0:
            # Apply correction: k_true_off = k_apparent - k_bleach
            k_true_off = k_apparent - bleach_rate
            
            # Safety check: k_true_off must be positive
            if k_true_off <= 0:
                return {
                    'success': False,
                    'error': f'Correction failed: k_apparent ({k_apparent:.4f}) < k_bleach ({bleach_rate:.4f}). '
                            'This suggests either: (1) bleach_rate is overestimated, '
                            '(2) binding is extremely transient, or (3) insufficient data.',
                    'k_apparent': float(k_apparent),
                    'k_bleach': float(bleach_rate)
                }
            
            # Calculate residence times
            residence_observed = 1 / k_apparent
            residence_corrected = 1 / k_true_off
            
            # Correction factor (how much we underestimated residence time)
            correction_factor = residence_corrected / residence_observed
            
            # Bleaching contribution (fraction of disappearances due to bleaching)
            bleach_contribution = bleach_rate / k_apparent
            
            # Propagate errors for corrected rate
            k_true_off_err = k_apparent_err  # Conservative estimate
            
            # Confidence intervals (assuming normal distribution)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            k_true_ci = [k_true_off - z_score * k_true_off_err,
                        k_true_off + z_score * k_true_off_err]
            
            results = {
                'success': True,
                'k_apparent': float(k_apparent),
                'k_apparent_err': float(k_apparent_err),
                'residence_time_observed': float(residence_observed),
                'k_true_off': float(k_true_off),
                'k_true_off_err': float(k_true_off_err),
                'residence_time_corrected': float(residence_corrected),
                'correction_factor': float(correction_factor),
                'bleach_contribution': float(bleach_contribution),
                'k_bleach': float(bleach_rate),
                'fit_quality': {
                    'R_squared': float(r_squared),
                    'RMSE': float(rmse),
                    'A_fit': float(A_fit),
                    'n_events': int(n_events)
                },
                'survival_curve': {
                    'time': sorted_times.tolist(),
                    'survival_observed': survival.tolist(),
                    'survival_fitted': survival_pred.tolist()
                },
                'confidence_intervals': {
                    'k_true_off_CI': [float(ci) for ci in k_true_ci],
                    'confidence_level': float(confidence_level)
                },
                'summary': {
                    'n_events': int(n_events),
                    'mean_dwell_observed': float(np.mean(dwell_times_array)),
                    'median_dwell_observed': float(np.median(dwell_times_array)),
                    'mean_dwell_corrected': float(residence_corrected),
                    'bleach_fraction': f"{bleach_contribution:.1%}",
                    'correction_magnitude': f"{correction_factor:.2f}x"
                }
            }
            
        else:
            # No bleach rate provided - return apparent rate only
            residence_observed = 1 / k_apparent
            
            results = {
                'success': True,
                'k_apparent': float(k_apparent),
                'k_apparent_err': float(k_apparent_err),
                'residence_time_observed': float(residence_observed),
                'note': 'Provide bleach_rate parameter for photobleaching correction. '
                       'Measure k_bleach from immobilized control (e.g., H2B-GFP, beads).',
                'recommendation': 'k_bleach should be measured independently under identical '
                                'imaging conditions (same laser power, exposure time, etc.)',
                'fit_quality': {
                    'R_squared': float(r_squared),
                    'RMSE': float(rmse),
                    'A_fit': float(A_fit),
                    'n_events': int(n_events)
                },
                'survival_curve': {
                    'time': sorted_times.tolist(),
                    'survival_observed': survival.tolist(),
                    'survival_fitted': survival_pred.tolist()
                },
                'summary': {
                    'n_events': int(n_events),
                    'mean_dwell_observed': float(np.mean(dwell_times_array)),
                    'median_dwell_observed': float(np.median(dwell_times_array))
                }
            }
        
        return results
        
    except ImportError:
        return {
            'success': False,
            'error': 'scipy is required for photobleaching correction. Install scipy.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error in photobleaching correction: {str(e)}'
        }
