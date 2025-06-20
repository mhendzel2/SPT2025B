"""
Advanced Correlative Analysis Module for SPT Analysis Application.
Provides tools for multi-parameter correlation, temporal cross-correlation, 
and intensity-motion coupling analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns


class CorrelativeAnalyzer:
    """
    Comprehensive correlative analysis for single particle tracking data.
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_intensity_motion_coupling(self, tracks_df: pd.DataFrame, 
                                        intensity_columns: List[str] = None,
                                        lag_range: int = 5) -> Dict[str, Any]:
        """
        Analyze coupling between particle motion and intensity fluctuations.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with intensity information
        intensity_columns : list
            List of intensity column names to analyze
        lag_range : int
            Range of time lags to analyze for correlation
            
        Returns
        -------
        dict
            Coupling analysis results
        """
        if intensity_columns is None:
            # Use enhanced channel detection with intelligent labeling
            try:
                from intensity_analysis import extract_intensity_channels, get_channel_labels_with_particle_default
                channels_dict = extract_intensity_channels(tracks_df)
                channel_labels = get_channel_labels_with_particle_default(tracks_df)
                intensity_columns = []
                for ch_key, col_list in channels_dict.items():
                    intensity_columns.extend(col_list)
            except ImportError:
                # Fallback to basic detection
                intensity_columns = [col for col in tracks_df.columns if 'intensity' in col.lower() or 'ch' in col.lower()]
        
        coupling_results = {
            'track_coupling': [],
            'ensemble_correlations': {},
            'lag_correlations': {}
        }
        
        # Analyze each track individually
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 10:  # Skip short tracks
                continue
                
            # Calculate motion parameters
            x = track_data['x'].values
            y = track_data['y'].values
            frames = track_data['frame'].values
            
            # Calculate instantaneous velocity
            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(frames)
            dt[dt == 0] = 1  # Avoid division by zero
            
            velocity = np.sqrt(dx**2 + dy**2) / dt
            
            # Calculate mean squared displacement for each time point
            msd_inst = dx**2 + dy**2
            
            # Analyze coupling with each intensity channel
            track_coupling = {'track_id': track_id}
            
            for int_col in intensity_columns:
                if int_col in track_data.columns:
                    intensity = track_data[int_col].values
                    
                    # Remove any NaN values
                    valid_mask = ~np.isnan(intensity)
                    if np.sum(valid_mask) < 5:
                        continue
                        
                    intensity_clean = intensity[valid_mask]
                    
                    # Correlate intensity with motion parameters
                    if len(intensity_clean) > len(velocity):
                        intensity_for_vel = intensity_clean[:-1]
                    else:
                        intensity_for_vel = intensity_clean
                        velocity_truncated = velocity[:len(intensity_clean)]
                    
                    if len(intensity_for_vel) == len(velocity):
                        # Velocity-intensity correlation
                        if np.std(intensity_for_vel) > 0 and np.std(velocity) > 0:
                            vel_int_corr, vel_int_p = stats.pearsonr(velocity, intensity_for_vel)
                            track_coupling[f'{int_col}_velocity_corr'] = vel_int_corr
                            track_coupling[f'{int_col}_velocity_p'] = vel_int_p
                    
                    # MSD-intensity correlation
                    if len(intensity_clean) > len(msd_inst):
                        intensity_for_msd = intensity_clean[:-1]
                    else:
                        intensity_for_msd = intensity_clean
                        msd_truncated = msd_inst[:len(intensity_clean)]
                    
                    if len(intensity_for_msd) == len(msd_inst):
                        if np.std(intensity_for_msd) > 0 and np.std(msd_inst) > 0:
                            msd_int_corr, msd_int_p = stats.pearsonr(msd_inst, intensity_for_msd)
                            track_coupling[f'{int_col}_msd_corr'] = msd_int_corr
                            track_coupling[f'{int_col}_msd_p'] = msd_int_p
                    
                    # Intensity autocorrelation
                    if len(intensity_clean) > lag_range * 2:
                        autocorr = []
                        for lag in range(1, lag_range + 1):
                            if lag < len(intensity_clean):
                                corr_val, _ = stats.pearsonr(intensity_clean[:-lag], intensity_clean[lag:])
                                autocorr.append(corr_val)
                        
                        track_coupling[f'{int_col}_autocorr_decay'] = np.mean(autocorr) if autocorr else 0
            
            coupling_results['track_coupling'].append(track_coupling)
        
        # Calculate ensemble statistics
        if coupling_results['track_coupling']:
            coupling_df = pd.DataFrame(coupling_results['track_coupling'])
            
            for col in coupling_df.columns:
                if col != 'track_id' and coupling_df[col].dtype in [np.float64, np.int64]:
                    valid_data = coupling_df[col].dropna()
                    if len(valid_data) > 0:
                        coupling_results['ensemble_correlations'][col] = {
                            'mean': valid_data.mean(),
                            'std': valid_data.std(),
                            'median': valid_data.median(),
                            'significant_fraction': np.sum(valid_data.abs() > 0.3) / len(valid_data)
                        }
        
        return coupling_results
    
    def temporal_cross_correlation(self, tracks_df: pd.DataFrame, 
                                 param1: str, param2: str,
                                 max_lag: int = 10) -> Dict[str, Any]:
        """
        Calculate temporal cross-correlation between two parameters.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        param1, param2 : str
            Column names for parameters to cross-correlate
        max_lag : int
            Maximum lag time for cross-correlation
            
        Returns
        -------
        dict
            Cross-correlation results
        """
        xcorr_results = {
            'track_xcorr': [],
            'ensemble_xcorr': np.zeros(2 * max_lag + 1),
            'lags': np.arange(-max_lag, max_lag + 1),
            'track_count': 0
        }
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < max_lag * 3:  # Need sufficient data
                continue
                
            if param1 not in track_data.columns or param2 not in track_data.columns:
                continue
                
            # Extract parameters
            p1 = track_data[param1].values
            p2 = track_data[param2].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(p1) | np.isnan(p2))
            if np.sum(valid_mask) < max_lag * 2:
                continue
                
            p1_clean = p1[valid_mask]
            p2_clean = p2[valid_mask]
            
            # Normalize
            if np.std(p1_clean) > 0:
                p1_norm = (p1_clean - np.mean(p1_clean)) / np.std(p1_clean)
            else:
                continue
                
            if np.std(p2_clean) > 0:
                p2_norm = (p2_clean - np.mean(p2_clean)) / np.std(p2_clean)
            else:
                continue
            
            # Calculate cross-correlation
            xcorr = signal.correlate(p2_norm, p1_norm, mode='full')
            
            # Extract relevant lags
            center = len(xcorr) // 2
            start_idx = max(0, center - max_lag)
            end_idx = min(len(xcorr), center + max_lag + 1)
            
            xcorr_segment = xcorr[start_idx:end_idx]
            
            # Normalize by length
            xcorr_segment = xcorr_segment / len(p1_norm)
            
            # Store track result
            track_result = {
                'track_id': track_id,
                'max_xcorr': np.max(np.abs(xcorr_segment)),
                'lag_at_max': xcorr_results['lags'][np.argmax(np.abs(xcorr_segment))],
                'zero_lag_corr': xcorr_segment[max_lag] if len(xcorr_segment) > max_lag else 0
            }
            
            xcorr_results['track_xcorr'].append(track_result)
            
            # Add to ensemble
            if len(xcorr_segment) == len(xcorr_results['ensemble_xcorr']):
                xcorr_results['ensemble_xcorr'] += xcorr_segment
                xcorr_results['track_count'] += 1
        
        # Normalize ensemble correlation
        if xcorr_results['track_count'] > 0:
            xcorr_results['ensemble_xcorr'] /= xcorr_results['track_count']
        
        return xcorr_results
    
    def multi_parameter_correlation_matrix(self, tracks_df: pd.DataFrame,
                                         parameters: List[str] = None) -> Dict[str, Any]:
        """
        Calculate correlation matrix between multiple track parameters.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        parameters : list
            List of parameters to include in correlation matrix
            
        Returns
        -------
        dict
            Correlation matrix and related statistics
        """
        if parameters is None:
            # Auto-detect numeric parameters
            numeric_cols = tracks_df.select_dtypes(include=[np.number]).columns
            parameters = [col for col in numeric_cols if col not in ['track_id', 'frame']]
        
        # Calculate track-averaged parameters
        track_params = []
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id]
            
            if len(track_data) < 5:
                continue
                
            track_summary = {'track_id': track_id}
            
            for param in parameters:
                if param in track_data.columns:
                    values = track_data[param].dropna()
                    if len(values) > 0:
                        track_summary[f'{param}_mean'] = values.mean()
                        track_summary[f'{param}_std'] = values.std()
                        track_summary[f'{param}_median'] = values.median()
                        
                        # Calculate motion-specific parameters
                        if param in ['x', 'y']:
                            track_summary[f'{param}_range'] = values.max() - values.min()
                            
            # Calculate derived motion parameters
            x = track_data['x'].values
            y = track_data['y'].values
            
            if len(x) > 2:
                # Track length
                distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                track_summary['total_distance'] = np.sum(distances)
                track_summary['mean_step_size'] = np.mean(distances)
                track_summary['step_size_std'] = np.std(distances)
                
                # Net displacement
                track_summary['net_displacement'] = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
                
                # Confinement ratio
                if track_summary['total_distance'] > 0:
                    track_summary['confinement_ratio'] = track_summary['net_displacement'] / track_summary['total_distance']
                
            track_params.append(track_summary)
        
        if not track_params:
            return {'success': False, 'error': 'No valid tracks for correlation analysis'}
        
        # Create DataFrame and correlation matrix
        param_df = pd.DataFrame(track_params)
        numeric_cols = param_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'track_id']
        
        if len(numeric_cols) < 2:
            return {'success': False, 'error': 'Insufficient numeric parameters'}
        
        # Calculate correlations
        corr_matrix = param_df[numeric_cols].corr()
        
        # Calculate p-values
        n_params = len(numeric_cols)
        p_matrix = np.ones((n_params, n_params))
        
        for i in range(n_params):
            for j in range(i+1, n_params):
                col_i = numeric_cols[i]
                col_j = numeric_cols[j]
                
                data_i = param_df[col_i].dropna()
                data_j = param_df[col_j].dropna()
                
                # Find common indices
                common_idx = param_df[[col_i, col_j]].dropna().index
                if len(common_idx) > 3:
                    _, p_val = stats.pearsonr(param_df.loc[common_idx, col_i], 
                                            param_df.loc[common_idx, col_j])
                    p_matrix[i, j] = p_val
                    p_matrix[j, i] = p_val
        
        # Find strongest correlations
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, 0)  # Ignore diagonal
        
        # Get indices of strongest correlations
        strongest_indices = np.unravel_index(np.argmax(np.abs(corr_values)), corr_values.shape)
        strongest_corr = {
            'parameters': (numeric_cols[strongest_indices[0]], numeric_cols[strongest_indices[1]]),
            'correlation': corr_values[strongest_indices],
            'p_value': p_matrix[strongest_indices]
        }
        
        return {
            'success': True,
            'correlation_matrix': corr_matrix,
            'p_value_matrix': pd.DataFrame(p_matrix, index=numeric_cols, columns=numeric_cols),
            'track_parameters': param_df,
            'strongest_correlation': strongest_corr,
            'significant_correlations': np.sum((np.abs(corr_values) > 0.5) & (p_matrix < 0.05)) // 2
        }


class MultiChannelAnalyzer:
    """
    Specialized analyzer for multi-channel particle tracking data.
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_channel_colocalization(self, primary_tracks: pd.DataFrame,
                                     secondary_tracks: pd.DataFrame,
                                     distance_threshold: float = 2.0,
                                     time_tolerance: int = 1) -> Dict[str, Any]:
        """
        Analyze colocalization between particles in different channels.
        
        Parameters
        ----------
        primary_tracks : pd.DataFrame
            Primary channel track data
        secondary_tracks : pd.DataFrame
            Secondary channel track data
        distance_threshold : float
            Maximum distance for colocalization (pixels)
        time_tolerance : int
            Time tolerance for colocalization (frames)
            
        Returns
        -------
        dict
            Colocalization analysis results
        """
        coloc_events = []
        
        # Get common frames
        primary_frames = set(primary_tracks['frame'].unique())
        secondary_frames = set(secondary_tracks['frame'].unique())
        common_frames = primary_frames & secondary_frames
        
        for frame in common_frames:
            primary_frame = primary_tracks[primary_tracks['frame'] == frame]
            
            # Check frames within time tolerance
            frame_range = range(max(0, frame - time_tolerance), frame + time_tolerance + 1)
            secondary_nearby = secondary_tracks[secondary_tracks['frame'].isin(frame_range)]
            
            if len(primary_frame) == 0 or len(secondary_nearby) == 0:
                continue
            
            # Calculate distances between all pairs using vectorized operations
            if len(primary_frame) > 0 and len(secondary_nearby) > 0:
                p_coords = primary_frame[['x', 'y']].values
                s_coords = secondary_nearby[['x', 'y']].values
                
                # Calculate all pairwise distances using broadcasting
                p_x = p_coords[:, 0][:, np.newaxis]
                p_y = p_coords[:, 1][:, np.newaxis]
                s_x = s_coords[:, 0]
                s_y = s_coords[:, 1]
                
                distances = np.sqrt((p_x - s_x)**2 + (p_y - s_y)**2)
                
                p_indices, s_indices = np.where(distances <= distance_threshold)
                
                for p_idx, s_idx in zip(p_indices, s_indices):
                    p_particle = primary_frame.iloc[p_idx]
                    s_particle = secondary_nearby.iloc[s_idx]
                    distance = distances[p_idx, s_idx]
                    
                    coloc_events.append({
                        'frame': frame,
                        'primary_track_id': p_particle['track_id'],
                        'secondary_track_id': s_particle['track_id'],
                        'secondary_frame': s_particle['frame'],
                        'distance': distance,
                        'primary_x': p_particle['x'],
                        'primary_y': p_particle['y'],
                        'secondary_x': s_particle['x'],
                        'secondary_y': s_particle['y']
                    })
        
        # Calculate statistics
        stats = {}
        if coloc_events:
            coloc_df = pd.DataFrame(coloc_events)
            
            stats = {
                'total_colocalization_events': len(coloc_events),
                'unique_primary_tracks_colocalized': len(coloc_df['primary_track_id'].unique()),
                'unique_secondary_tracks_colocalized': len(coloc_df['secondary_track_id'].unique()),
                'mean_colocalization_distance': coloc_df['distance'].mean(),
                'median_colocalization_distance': coloc_df['distance'].median(),
                'colocalization_efficiency_primary': len(coloc_df['primary_track_id'].unique()) / len(primary_tracks['track_id'].unique()),
                'colocalization_efficiency_secondary': len(coloc_df['secondary_track_id'].unique()) / len(secondary_tracks['track_id'].unique()),
                'frames_with_colocalization': len(coloc_df['frame'].unique()),
                'total_frames_analyzed': len(common_frames)
            }
        
        return {
            'success': True,
            'colocalization_events': pd.DataFrame(coloc_events) if coloc_events else pd.DataFrame(),
            'statistics': stats
        }


class TemporalCrossCorrelator:
    """
    Advanced temporal cross-correlation analysis.
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_temporal_patterns(self, tracks_df: pd.DataFrame,
                                signal_column: str,
                                reference_events: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze temporal patterns in particle behavior relative to reference events.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with temporal signal
        signal_column : str
            Column name containing the signal to analyze
        reference_events : pd.DataFrame
            Reference events with 'frame' column
            
        Returns
        -------
        dict
            Temporal pattern analysis results
        """
        if reference_events is None:
            # Use intensity peaks as reference events
            reference_events = self._detect_intensity_peaks(tracks_df, signal_column)
        
        # Analyze signal around reference events
        window_size = 20  # frames before and after event
        triggered_averages = []
        
        # Vectorized processing of reference events
        event_frames = reference_events['frame'].values
        
        for event_frame in event_frames:
            # Get signal in window around event
            window_start = event_frame - window_size
            window_end = event_frame + window_size
            
            window_data = tracks_df[
                (tracks_df['frame'] >= window_start) & 
                (tracks_df['frame'] <= window_end)
            ]
            
            if len(window_data) > window_size and signal_column in window_data.columns:
                frame_range = np.arange(window_start, window_end + 1)
                window_data_grouped = window_data.groupby('frame')[signal_column].mean()
                
                for frame in frame_range:
                    if frame in window_data_grouped.index:
                        triggered_averages.append({
                            'relative_frame': frame - event_frame,
                            'signal': window_data_grouped[frame]
                        })
        
        # Calculate ensemble triggered average
        if triggered_averages:
            triggered_df = pd.DataFrame(triggered_averages)
            ensemble_avg = triggered_df.groupby('relative_frame')['signal'].mean()
            ensemble_std = triggered_df.groupby('relative_frame')['signal'].std()
            
            return {
                'success': True,
                'triggered_average': ensemble_avg,
                'triggered_std': ensemble_std,
                'reference_events': reference_events,
                'n_events_analyzed': len(reference_events)
            }
        
        return {'success': False, 'error': 'No valid temporal patterns found'}
    
    def _detect_intensity_peaks(self, tracks_df: pd.DataFrame, 
                              signal_column: str,
                              prominence: float = 0.5) -> pd.DataFrame:
        """
        Detect peaks in intensity signal to use as reference events.
        """
        peaks_data = []
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if signal_column not in track_data.columns:
                continue
                
            signal = track_data[signal_column].dropna()
            frames = track_data.loc[signal.index, 'frame']
            
            if len(signal) < 10:
                continue
            
            # Find peaks
            peaks, properties = signal.find_peaks(signal.values, prominence=prominence)
            
            for peak_idx in peaks:
                peaks_data.append({
                    'track_id': track_id,
                    'frame': frames.iloc[peak_idx],
                    'signal_value': signal.iloc[peak_idx],
                    'prominence': properties['prominences'][list(peaks).index(peak_idx)]
                })
        
        return pd.DataFrame(peaks_data)
