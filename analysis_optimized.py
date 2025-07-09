"""
Optimized Analysis Functions for SPT Analysis Application.
Provides high-performance implementations using vectorized operations and parallel processing.
"""

import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def calculate_msd_numba(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MSD using Numba for speed optimization.
    
    Parameters
    ----------
    x, y : np.ndarray
        Position arrays
    max_lag : int
        Maximum lag time
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (lag_times, msd_values)
    """
    n_points = len(x)
    max_lag = min(max_lag, n_points - 1)
    
    msd_values = np.zeros(max_lag)
    lag_times = np.arange(1, max_lag + 1)
    
    for lag in range(1, max_lag + 1):
        sum_sq_disp = 0.0
        count = 0
        
        for i in range(n_points - lag):
            dx = x[i + lag] - x[i]
            dy = y[i + lag] - y[i]
            sum_sq_disp += dx * dx + dy * dy
            count += 1
        
        if count > 0:
            msd_values[lag - 1] = sum_sq_disp / count
    
    return lag_times, msd_values

@jit(nopython=True)
def calculate_velocities_numba(x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
    Calculate instantaneous velocities using Numba.
    
    Parameters
    ----------
    x, y : np.ndarray
        Position arrays
    dt : np.ndarray
        Time intervals
        
    Returns
    -------
    np.ndarray
        Velocity array
    """
    n_points = len(x)
    velocities = np.zeros(n_points - 1)
    
    for i in range(n_points - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        time_interval = dt[i] if dt[i] > 0 else 1.0
        velocities[i] = np.sqrt(dx * dx + dy * dy) / time_interval
    
    return velocities

@jit(nopython=True, parallel=True)
def calculate_track_features_parallel(positions: np.ndarray) -> np.ndarray:
    """
    Calculate track features in parallel using Numba.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n_tracks, max_length, 2) with track positions
        
    Returns
    -------
    np.ndarray
        Feature array of shape (n_tracks, n_features)
    """
    n_tracks = positions.shape[0]
    n_features = 8  # Total, mean_velocity, std_velocity, net_displacement, etc.
    features = np.zeros((n_tracks, n_features))
    
    for i in prange(n_tracks):
        track = positions[i]
        
        # Find valid points (non-NaN)
        valid_mask = ~(np.isnan(track[:, 0]) | np.isnan(track[:, 1]))
        valid_points = np.sum(valid_mask)
        
        if valid_points < 2:
            continue
        
        # Extract valid coordinates
        x = track[valid_mask, 0]
        y = track[valid_mask, 1]
        
        # Feature 0: Track length
        features[i, 0] = valid_points
        
        # Calculate velocities
        if valid_points > 1:
            velocities = np.zeros(valid_points - 1)
            for j in range(valid_points - 1):
                dx = x[j + 1] - x[j]
                dy = y[j + 1] - y[j]
                velocities[j] = np.sqrt(dx * dx + dy * dy)
            
            # Feature 1: Mean velocity
            features[i, 1] = np.mean(velocities)
            
            # Feature 2: Velocity standard deviation
            features[i, 2] = np.std(velocities)
            
            # Feature 3: Maximum velocity
            features[i, 3] = np.max(velocities)
        
        # Feature 4: Net displacement
        if valid_points >= 2:
            net_disp = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            features[i, 4] = net_disp
        
        # Feature 5: Total path length
        if valid_points > 1:
            path_length = 0.0
            for j in range(valid_points - 1):
                dx = x[j + 1] - x[j]
                dy = y[j + 1] - y[j]
                path_length += np.sqrt(dx * dx + dy * dy)
            features[i, 5] = path_length
            
            # Feature 6: Straightness (net displacement / path length)
            if path_length > 0:
                features[i, 6] = features[i, 4] / path_length
        
        # Feature 7: Radius of gyration
        if valid_points >= 2:
            center_x = np.mean(x)
            center_y = np.mean(y)
            rg_sq = 0.0
            for j in range(valid_points):
                rg_sq += (x[j] - center_x)**2 + (y[j] - center_y)**2
            features[i, 7] = np.sqrt(rg_sq / valid_points)
    
    return features

class OptimizedAnalyzer:
    """
    High-performance analyzer using vectorized operations and parallel processing.
    """
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or min(4, multiprocessing.cpu_count())
    
    def calculate_ensemble_msd_optimized(self, tracks_df: pd.DataFrame,
                                       max_lag: int = 50,
                                       pixel_size: float = 1.0,
                                       frame_interval: float = 1.0) -> Dict[str, Any]:
        """
        Calculate ensemble MSD using optimized algorithms.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        max_lag : int
            Maximum lag time
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
            
        Returns
        -------
        Dict[str, Any]
            Optimized MSD results
        """
        if tracks_df.empty:
            return {'success': False, 'error': 'No track data provided'}
        
        # Group tracks and prepare for parallel processing
        track_groups = tracks_df.groupby('track_id')
        
        # Use parallel processing for individual track MSDs
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for track_id, track_data in track_groups:
                if len(track_data) >= 10:  # Minimum points for MSD
                    future = executor.submit(
                        self._calculate_single_track_msd,
                        track_data, max_lag, pixel_size, frame_interval
                    )
                    futures.append((track_id, future))
            
            # Collect results
            individual_msds = {}
            all_msd_values = []
            
            for track_id, future in futures:
                try:
                    result = future.result()
                    if result['success']:
                        individual_msds[track_id] = result
                        all_msd_values.append(result['msd_values'])
                except Exception as e:
                    continue
        
        if not all_msd_values:
            return {'success': False, 'error': 'No valid tracks for MSD calculation'}
        
        # Calculate ensemble average using vectorized operations
        min_length = min(len(msd) for msd in all_msd_values)
        msd_matrix = np.array([msd[:min_length] for msd in all_msd_values])
        
        ensemble_msd = np.mean(msd_matrix, axis=0)
        ensemble_std = np.std(msd_matrix, axis=0)
        
        # Create lag times
        lag_times = np.arange(1, min_length + 1) * frame_interval
        
        return {
            'success': True,
            'ensemble_msd': ensemble_msd,
            'ensemble_std': ensemble_std,
            'lag_times': lag_times,
            'individual_msds': individual_msds,
            'n_tracks': len(individual_msds)
        }
    
    def _calculate_single_track_msd(self, track_data: pd.DataFrame,
                                  max_lag: int, pixel_size: float,
                                  frame_interval: float) -> Dict[str, Any]:
        """Calculate MSD for a single track using Numba optimization."""
        try:
            # Sort by frame and extract coordinates
            track_sorted = track_data.sort_values('frame')
            x = track_sorted['x'].values * pixel_size
            y = track_sorted['y'].values * pixel_size
            
            # Calculate MSD using Numba
            lag_times, msd_values = calculate_msd_numba(x, y, max_lag)
            
            return {
                'success': True,
                'lag_times': lag_times * frame_interval,
                'msd_values': msd_values,
                'n_points': len(x)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_diffusion_coefficients_batch(self, tracks_df: pd.DataFrame,
                                             pixel_size: float = 1.0,
                                             frame_interval: float = 1.0,
                                             n_points_fit: int = 4) -> Dict[str, Any]:
        """
        Calculate diffusion coefficients for all tracks in batch mode.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
        n_points_fit : int
            Number of points to use for linear fit
            
        Returns
        -------
        Dict[str, Any]
            Batch diffusion coefficient results
        """
        track_groups = tracks_df.groupby('track_id')
        
        # Prepare data for vectorized processing
        track_ids = []
        diffusion_coeffs = []
        r_squared_values = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for track_id, track_data in track_groups:
                if len(track_data) >= 10:
                    future = executor.submit(
                        self._calculate_single_diffusion_coeff,
                        track_data, pixel_size, frame_interval, n_points_fit
                    )
                    futures.append((track_id, future))
            
            for track_id, future in futures:
                try:
                    result = future.result()
                    if result['success']:
                        track_ids.append(track_id)
                        diffusion_coeffs.append(result['diffusion_coefficient'])
                        r_squared_values.append(result['r_squared'])
                except Exception:
                    continue
        
        if not diffusion_coeffs:
            return {'success': False, 'error': 'No valid diffusion coefficients calculated'}
        
        # Convert to arrays for vectorized statistics
        diffusion_array = np.array(diffusion_coeffs)
        r_squared_array = np.array(r_squared_values)
        
        # Calculate statistics using vectorized operations
        stats = {
            'mean': float(np.mean(diffusion_array)),
            'median': float(np.median(diffusion_array)),
            'std': float(np.std(diffusion_array)),
            'min': float(np.min(diffusion_array)),
            'max': float(np.max(diffusion_array)),
            'mean_r_squared': float(np.mean(r_squared_array))
        }
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'track_id': track_ids,
            'diffusion_coefficient': diffusion_coeffs,
            'r_squared': r_squared_values
        })
        
        return {
            'success': True,
            'diffusion_coefficients': results_df,
            'statistics': stats,
            'n_tracks': len(track_ids)
        }
    
    def _calculate_single_diffusion_coeff(self, track_data: pd.DataFrame,
                                        pixel_size: float, frame_interval: float,
                                        n_points_fit: int) -> Dict[str, Any]:
        """Calculate diffusion coefficient for a single track."""
        try:
            # Calculate MSD
            track_sorted = track_data.sort_values('frame')
            x = track_sorted['x'].values * pixel_size
            y = track_sorted['y'].values * pixel_size
            
            max_lag = min(n_points_fit + 5, len(x) - 1)
            lag_times, msd_values = calculate_msd_numba(x, y, max_lag)
            
            if len(msd_values) < n_points_fit:
                return {'success': False, 'error': 'Insufficient data for fitting'}
            
            # Linear fit to first n_points_fit
            lag_times_fit = lag_times[:n_points_fit] * frame_interval
            msd_fit = msd_values[:n_points_fit]
            
            # Vectorized linear regression
            X = lag_times_fit
            y = msd_fit
            
            # Calculate slope (diffusion coefficient = slope / 4 for 2D)
            slope = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X))**2)
            diffusion_coeff = slope / 4.0
            
            # Calculate R-squared
            y_pred = slope * X
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'success': True,
                'diffusion_coefficient': float(diffusion_coeff),
                'r_squared': float(r_squared),
                'slope': float(slope)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_track_features_vectorized(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze track features using vectorized operations.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
            
        Returns
        -------
        Dict[str, Any]
            Vectorized feature analysis results
        """
        # Prepare position arrays
        track_ids = tracks_df['track_id'].unique()
        max_length = tracks_df.groupby('track_id').size().max()
        
        # Create 3D array: (n_tracks, max_length, 2)
        positions = np.full((len(track_ids), max_length, 2), np.nan)
        
        for i, track_id in enumerate(track_ids):
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            track_length = len(track_data)
            positions[i, :track_length, 0] = track_data['x'].values
            positions[i, :track_length, 1] = track_data['y'].values
        
        # Calculate features using parallel Numba function
        features = calculate_track_features_parallel(positions)
        
        # Create feature DataFrame
        feature_names = [
            'track_length', 'mean_velocity', 'velocity_std', 'max_velocity',
            'net_displacement', 'path_length', 'straightness', 'radius_of_gyration'
        ]
        
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['track_id'] = track_ids
        
        # Calculate ensemble statistics using vectorized operations
        ensemble_stats = {}
        for feature in feature_names:
            values = features_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                ensemble_stats[feature] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return {
            'success': True,
            'track_features': features_df,
            'ensemble_statistics': ensemble_stats,
            'n_tracks': len(track_ids)
        }
    
    def spatial_analysis_vectorized(self, tracks_df: pd.DataFrame,
                                  grid_size: float = 10.0) -> Dict[str, Any]:
        """
        Perform spatial analysis using vectorized operations.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        grid_size : float
            Grid size for spatial binning
            
        Returns
        -------
        Dict[str, Any]
            Spatial analysis results
        """
        # Extract all positions
        positions = tracks_df[['x', 'y']].values
        
        if len(positions) == 0:
            return {'success': False, 'error': 'No position data available'}
        
        # Create spatial grid using vectorized operations
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Define grid boundaries
        x_edges = np.arange(x_min, x_max + grid_size, grid_size)
        y_edges = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Bin positions using numpy histogram2d (vectorized)
        density_map, _, _ = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=[x_edges, y_edges]
        )
        
        # Calculate spatial statistics
        total_detections = len(positions)
        occupied_bins = np.sum(density_map > 0)
        max_density = np.max(density_map)
        mean_density = np.mean(density_map[density_map > 0]) if occupied_bins > 0 else 0
        
        # Calculate center of mass using vectorized operations
        center_of_mass = np.mean(positions, axis=0)
        
        # Calculate spatial spread (radius of gyration)
        distances_from_center = np.sqrt(np.sum((positions - center_of_mass)**2, axis=1))
        spatial_spread = np.sqrt(np.mean(distances_from_center**2))
        
        return {
            'success': True,
            'density_map': density_map,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'spatial_statistics': {
                'total_detections': int(total_detections),
                'occupied_bins': int(occupied_bins),
                'max_density': float(max_density),
                'mean_density': float(mean_density),
                'center_of_mass': center_of_mass.tolist(),
                'spatial_spread': float(spatial_spread)
            }
        }
