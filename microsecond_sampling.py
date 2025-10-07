"""
Microsecond Sampling Support Module

Enables high-frequency SPT analysis with irregular time intervals.
Critical for:
- Fast processes (<1 ms timescale)
- Camera frame synchronization issues  
- Variable framerate acquisition
- Combining datasets with different Δt

Handles:
1. Irregular Δt detection and validation
2. MSD calculation with variable lags
3. GSER rheology with non-uniform sampling
4. Data quality metrics for irregular sampling

Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
import warnings


class IrregularSamplingHandler:
    """
    Handles particle tracking data with irregular time intervals.
    
    Automatically detects regular vs irregular sampling and applies
    appropriate algorithms.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize handler.
        
        Parameters
        ----------
        tolerance : float
            Relative tolerance for considering sampling "regular".
            Default 0.01 = 1% variation allowed.
        """
        self.tolerance = tolerance
    
    
    def detect_sampling_type(self, track_df: pd.DataFrame) -> Dict:
        """
        Detect if track has regular or irregular time sampling.
        
        Parameters
        ----------
        track_df : pd.DataFrame
            Track with 'frame' or 'time' column
        
        Returns
        -------
        Dict
            {
                'is_regular': bool,
                'mean_dt': float (seconds),
                'dt_std': float (seconds),
                'dt_cv': float (coefficient of variation),
                'time_column': str ('frame' or 'time'),
                'actual_times': ndarray (seconds)
            }
        """
        # Check for time column
        if 'time' in track_df.columns:
            times = track_df['time'].values
            time_col = 'time'
        elif 'frame' in track_df.columns:
            # Assume uniform frame intervals if no time column
            # Will use default frame_interval from elsewhere
            frames = track_df['frame'].values
            times = frames.astype(float)  # Placeholder
            time_col = 'frame'
        else:
            return {
                'success': False,
                'error': 'Track must have "frame" or "time" column'
            }
        
        if len(times) < 2:
            return {
                'success': False,
                'error': 'Track too short for sampling analysis'
            }
        
        # Compute time intervals
        dt_values = np.diff(times)
        
        if np.any(dt_values <= 0):
            return {
                'success': False,
                'error': 'Non-monotonic time values detected'
            }
        
        mean_dt = np.mean(dt_values)
        std_dt = np.std(dt_values)
        cv_dt = std_dt / mean_dt if mean_dt > 0 else np.inf
        
        # Check regularity
        is_regular = cv_dt < self.tolerance
        
        return {
            'success': True,
            'is_regular': is_regular,
            'mean_dt': mean_dt,
            'dt_std': std_dt,
            'dt_cv': cv_dt,
            'dt_min': np.min(dt_values),
            'dt_max': np.max(dt_values),
            'time_column': time_col,
            'actual_times': times,
            'dt_values': dt_values
        }
    
    
    def calculate_msd_irregular(self, track_df: pd.DataFrame,
                                pixel_size: float = 1.0,
                                max_lag_s: Optional[float] = None,
                                n_lag_bins: int = 30) -> Dict:
        """
        Calculate MSD for irregularly sampled trajectory.
        
        Uses binned lag times rather than discrete frame lags.
        
        Parameters
        ----------
        track_df : pd.DataFrame
            Track with x, y (z optional) and time columns
        pixel_size : float
            Pixel size in μm
        max_lag_s : float, optional
            Maximum lag time in seconds. Default: 25% of track duration
        n_lag_bins : int
            Number of lag time bins
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'lag_times_s': ndarray,
                'msd_um2': ndarray,
                'n_observations': ndarray (points per lag),
                'msd_std': ndarray (standard deviation)
            }
        """
        # Detect sampling
        sampling_info = self.detect_sampling_type(track_df)
        if not sampling_info['success']:
            return sampling_info
        
        # Get positions and times
        pos_cols = ['x', 'y'] if 'z' not in track_df.columns else ['x', 'y', 'z']
        positions = track_df[pos_cols].values * pixel_size
        times = sampling_info['actual_times']
        
        # If using frame column, need actual time
        if sampling_info['time_column'] == 'frame':
            # Check if we have frame_interval stored
            if 'frame_interval' in track_df.columns:
                frame_interval = track_df['frame_interval'].iloc[0]
            else:
                # Assume default
                frame_interval = 0.1
                warnings.warn(
                    'No time column or frame_interval found, assuming dt=0.1s',
                    UserWarning
                )
            times = times * frame_interval
        
        # Determine lag range
        total_duration = times[-1] - times[0]
        if max_lag_s is None:
            max_lag_s = total_duration * 0.25
        
        # Create log-spaced lag bins
        min_lag_s = np.min(sampling_info['dt_values']) if sampling_info['time_column'] == 'time' \
                   else frame_interval
        lag_bins = np.logspace(np.log10(min_lag_s), np.log10(max_lag_s), n_lag_bins + 1)
        lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
        
        # Compute MSD using Welford's online algorithm for mean and variance
        # Single-pass computation instead of two-pass
        # Welford's method: M_n = M_{n-1} + (x_n - M_{n-1})/n
        #                   S_n = S_{n-1} + (x_n - M_{n-1})(x_n - M_n)
        # Variance = S_n / (n-1)
        
        welford_M = np.zeros(n_lag_bins)  # Running mean
        welford_S = np.zeros(n_lag_bins)  # Running sum of squared differences
        n_observations = np.zeros(n_lag_bins, dtype=int)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                lag = times[j] - times[i]
                
                # Find appropriate bin
                bin_idx = np.searchsorted(lag_bins[:-1], lag, side='right') - 1
                if 0 <= bin_idx < n_lag_bins:
                    displacement = positions[j] - positions[i]
                    squared_disp = np.sum(displacement**2)
                    
                    # Welford update
                    n_observations[bin_idx] += 1
                    n = n_observations[bin_idx]
                    delta = squared_disp - welford_M[bin_idx]
                    welford_M[bin_idx] += delta / n
                    delta2 = squared_disp - welford_M[bin_idx]
                    welford_S[bin_idx] += delta * delta2
        
        # Extract results
        msd_values = welford_M
        valid = n_observations > 1  # Need > 1 for variance
        msd_std_values = np.zeros(n_lag_bins)
        msd_std_values[valid] = np.sqrt(welford_S[valid] / (n_observations[valid] - 1))
        
        return {
            'success': True,
            'lag_times_s': lag_centers,
            'msd_um2': msd_values,
            'msd_std': msd_std_values,
            'n_observations': n_observations,
            'is_regular_sampling': sampling_info['is_regular'],
            'sampling_info': sampling_info
        }
    
    
    def convert_to_uniform_time_grid(self, track_df: pd.DataFrame,
                                     target_dt: Optional[float] = None,
                                     method: str = 'linear') -> Dict:
        """
        Interpolate irregular trajectory onto uniform time grid.
        
        Useful for algorithms requiring regular sampling.
        
        Parameters
        ----------
        track_df : pd.DataFrame
            Irregular track
        target_dt : float, optional
            Target time interval. Default: median of actual Δt
        method : str
            Interpolation method: 'linear', 'cubic', 'nearest'
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'track_df': DataFrame (uniformly sampled),
                'target_dt': float,
                'n_points_original': int,
                'n_points_interpolated': int
            }
        """
        sampling_info = self.detect_sampling_type(track_df)
        if not sampling_info['success']:
            return sampling_info
        
        times = sampling_info['actual_times']
        
        # Determine target dt
        if target_dt is None:
            target_dt = np.median(sampling_info['dt_values'])
        
        # Create uniform time grid
        time_min = times[0]
        time_max = times[-1]
        n_points_new = int((time_max - time_min) / target_dt) + 1
        uniform_times = np.linspace(time_min, time_max, n_points_new)
        
        # Interpolate positions
        pos_cols = ['x', 'y'] if 'z' not in track_df.columns else ['x', 'y', 'z']
        new_positions = {}
        
        for col in pos_cols:
            values = track_df[col].values
            interpolator = interp1d(times, values, kind=method, 
                                   bounds_error=False, fill_value='extrapolate')
            new_positions[col] = interpolator(uniform_times)
        
        # Create new dataframe
        new_track = pd.DataFrame({
            'frame': np.arange(len(uniform_times)),
            'time': uniform_times,
            **new_positions
        })
        
        # Preserve track_id if present
        if 'track_id' in track_df.columns:
            new_track['track_id'] = track_df['track_id'].iloc[0]
        
        return {
            'success': True,
            'track_df': new_track,
            'target_dt': target_dt,
            'n_points_original': len(track_df),
            'n_points_interpolated': len(new_track),
            'interpolation_method': method
        }
    
    
    def validate_microsecond_data(self, tracks_df: pd.DataFrame) -> Dict:
        """
        Quality checks for high-frequency (μs-ms) tracking data.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            High-frequency tracking data
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'warnings': List[str],
                'time_resolution_s': float,
                'nyquist_frequency_hz': float,
                'recommended_max_freq_hz': float
            }
        """
        warnings_list = []
        
        # Detect sampling
        if 'track_id' in tracks_df.columns:
            # Analyze first track as representative
            first_track = tracks_df[tracks_df['track_id'] == tracks_df['track_id'].iloc[0]]
        else:
            first_track = tracks_df
        
        sampling_info = self.detect_sampling_type(first_track)
        if not sampling_info['success']:
            return sampling_info
        
        mean_dt = sampling_info['mean_dt']
        
        # Check 1: Localization noise vs motion
        if 'x' in first_track.columns and 'y' in first_track.columns:
            positions = first_track[['x', 'y']].values
            displacements = np.diff(positions, axis=0)
            rms_displacement = np.sqrt(np.mean(np.sum(displacements**2, axis=1)))
            
            # Estimate localization uncertainty (from stationary segments)
            # Use first 10 points as proxy
            if len(first_track) >= 10:
                local_displacements = np.diff(positions[:10], axis=0)
                local_rms = np.sqrt(np.mean(np.sum(local_displacements**2, axis=1)))
                
                snr = rms_displacement / (local_rms + 1e-12)
                if snr < 3:
                    warnings_list.append(
                        f'Low SNR ({snr:.1f}) - localization noise dominates motion. '
                        'Consider longer exposure or slower framerate.'
                    )
        
        # Check 2: Temporal resolution
        time_resolution = mean_dt
        nyquist_hz = 1 / (2 * time_resolution)
        recommended_max_hz = nyquist_hz / 3  # Factor of 3 safety margin
        
        if time_resolution < 1e-6:
            warnings_list.append(
                'Sub-microsecond sampling detected. Verify camera specifications '
                'and check for timestamp errors.'
            )
        
        if time_resolution < 1e-3:
            warnings_list.append(
                f'High-frequency sampling ({1/time_resolution:.0f} Hz). '
                'Ensure motion blur is minimal (exposure << frame interval).'
            )
        
        # Check 3: Track length
        n_points = len(first_track)
        duration_s = sampling_info['actual_times'][-1] - sampling_info['actual_times'][0]
        
        if n_points < 100:
            warnings_list.append(
                f'Short track ({n_points} points, {duration_s*1e3:.1f} ms). '
                'Statistical power limited - use ensemble averaging.'
            )
        
        # Check 4: Irregular sampling variability
        if not sampling_info['is_regular']:
            cv = sampling_info['dt_cv']
            warnings_list.append(
                f'Irregular sampling detected (CV = {cv:.1%}). '
                'Using binned lag-time MSD calculation.'
            )
            
            if cv > 0.5:
                warnings_list.append(
                    'Highly irregular sampling. Consider resampling to uniform grid '
                    'or checking for acquisition issues.'
                )
        
        return {
            'success': True,
            'warnings': warnings_list if warnings_list else ['No issues detected'],
            'time_resolution_s': time_resolution,
            'nyquist_frequency_hz': nyquist_hz,
            'recommended_max_freq_hz': recommended_max_hz,
            'sampling_type': 'regular' if sampling_info['is_regular'] else 'irregular',
            'track_duration_s': duration_s,
            'n_points': n_points,
            'effective_framerate_hz': 1 / mean_dt
        }


def add_time_column_from_frame(tracks_df: pd.DataFrame, 
                               frame_interval: float) -> pd.DataFrame:
    """
    Add explicit time column from frame numbers.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks with 'frame' column
    frame_interval : float
        Time between frames in seconds
    
    Returns
    -------
    pd.DataFrame
        Tracks with added 'time' column
    """
    if 'time' in tracks_df.columns:
        warnings.warn('Time column already exists, not overwriting', UserWarning)
        return tracks_df
    
    if 'frame' not in tracks_df.columns:
        raise ValueError('tracks_df must have "frame" column')
    
    tracks_df = tracks_df.copy()
    tracks_df['time'] = tracks_df['frame'] * frame_interval
    tracks_df['frame_interval'] = frame_interval
    
    return tracks_df


def combine_multi_framerate_data(track_datasets: List[pd.DataFrame],
                                 frame_intervals: List[float],
                                 resample_method: str = 'keep_original') -> Dict:
    """
    Combine tracking datasets with different framerates.
    
    Parameters
    ----------
    track_datasets : List[pd.DataFrame]
        List of tracking datasets
    frame_intervals : List[float]
        Frame interval for each dataset (seconds)
    resample_method : str
        'keep_original': Keep as-is with time column
        'resample_uniform': Interpolate all to common grid
        'bin_lag_times': Use binned MSD (recommended)
    
    Returns
    -------
    Dict
        {
            'success': bool,
            'combined_tracks': DataFrame or List[DataFrame],
            'method_used': str,
            'metadata': Dict
        }
    """
    if len(track_datasets) != len(frame_intervals):
        return {
            'success': False,
            'error': 'Number of datasets must match number of frame_intervals'
        }
    
    if resample_method == 'keep_original':
        # Add time columns and combine
        processed_datasets = []
        for tracks_df, dt in zip(track_datasets, frame_intervals):
            tracks_with_time = add_time_column_from_frame(tracks_df, dt)
            processed_datasets.append(tracks_with_time)
        
        # Concatenate with unique track IDs
        offset = 0
        combined_list = []
        for i, df in enumerate(processed_datasets):
            df_copy = df.copy()
            if 'track_id' in df_copy.columns:
                df_copy['track_id'] = df_copy['track_id'] + offset
                offset = df_copy['track_id'].max() + 1
            df_copy['dataset_id'] = i
            df_copy['source_framerate_hz'] = 1 / frame_intervals[i]
            combined_list.append(df_copy)
        
        combined_tracks = pd.concat(combined_list, ignore_index=True)
        
        return {
            'success': True,
            'combined_tracks': combined_tracks,
            'method_used': 'keep_original',
            'metadata': {
                'n_datasets': len(track_datasets),
                'framerates_hz': [1/dt for dt in frame_intervals],
                'total_tracks': len(combined_tracks['track_id'].unique())
            }
        }
    
    elif resample_method == 'resample_uniform':
        # Find common time resolution (finest)
        target_dt = min(frame_intervals)
        
        handler = IrregularSamplingHandler()
        resampled_datasets = []
        
        for tracks_df, dt in zip(track_datasets, frame_intervals):
            tracks_with_time = add_time_column_from_frame(tracks_df, dt)
            
            # Resample each track
            if 'track_id' in tracks_with_time.columns:
                track_ids = tracks_with_time['track_id'].unique()
                resampled_tracks = []
                
                for tid in track_ids:
                    track = tracks_with_time[tracks_with_time['track_id'] == tid]
                    result = handler.convert_to_uniform_time_grid(track, target_dt=target_dt)
                    if result['success']:
                        resampled_tracks.append(result['track_df'])
                
                if resampled_tracks:
                    resampled_datasets.append(pd.concat(resampled_tracks, ignore_index=True))
        
        combined_tracks = pd.concat(resampled_datasets, ignore_index=True)
        
        return {
            'success': True,
            'combined_tracks': combined_tracks,
            'method_used': 'resample_uniform',
            'metadata': {
                'target_dt_s': target_dt,
                'target_framerate_hz': 1/target_dt,
                'n_datasets': len(track_datasets),
                'total_tracks': len(combined_tracks['track_id'].unique())
            }
        }
    
    elif resample_method == 'bin_lag_times':
        # Keep separate, will use binned MSD
        processed_datasets = []
        for tracks_df, dt in zip(track_datasets, frame_intervals):
            tracks_with_time = add_time_column_from_frame(tracks_df, dt)
            processed_datasets.append(tracks_with_time)
        
        return {
            'success': True,
            'combined_tracks': processed_datasets,  # Keep as list
            'method_used': 'bin_lag_times',
            'metadata': {
                'n_datasets': len(track_datasets),
                'framerates_hz': [1/dt for dt in frame_intervals],
                'note': 'Use IrregularSamplingHandler.calculate_msd_irregular() for each dataset'
            }
        }
    
    else:
        return {
            'success': False,
            'error': f'Unknown resample_method: {resample_method}'
        }


def quick_microsecond_check(tracks_df: pd.DataFrame) -> str:
    """
    Quick check for microsecond data issues.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracking data
    
    Returns
    -------
    str
        Human-readable summary
    """
    handler = IrregularSamplingHandler()
    validation = handler.validate_microsecond_data(tracks_df)
    
    if not validation['success']:
        return f"Error: {validation.get('error', 'Unknown error')}"
    
    summary = f"Sampling: {validation['sampling_type']}\n"
    summary += f"Time resolution: {validation['time_resolution_s']*1e3:.3f} ms\n"
    summary += f"Effective framerate: {validation['effective_framerate_hz']:.1f} Hz\n"
    summary += f"Recommended max analysis freq: {validation['recommended_max_freq_hz']:.1f} Hz\n"
    summary += f"\nWarnings:\n"
    for warning in validation['warnings']:
        summary += f"  - {warning}\n"
    
    return summary
