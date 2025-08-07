"""
Changepoint Detection Module for SPT Analysis Application.
Provides tools for detecting behavioral switches, regime changes, 
and transition states in particle trajectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ChangePointDetector:
    """
    Comprehensive changepoint detection for particle tracking data.
    """
    
    def __init__(self):
        self.results = {}
    
    def detect_motion_regime_changes(self, tracks_df: pd.DataFrame,
                                   window_size: int = 10,
                                   min_segment_length: int = 5,
                                   significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Detect changes in motion regimes using statistical tests.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data in standard format
        window_size : int
            Window size for calculating local motion statistics
        min_segment_length : int
            Minimum length of motion segments
        significance_level : float
            Statistical significance level for changepoint detection
            
        Returns
        -------
        dict
            Changepoint detection results
        """
        # Validate input DataFrame
        required_columns = ['track_id', 'x', 'y']
        missing_columns = [col for col in required_columns if col not in tracks_df.columns]
        
        if tracks_df.empty or missing_columns:
            return {
                'success': False,
                'changepoints': pd.DataFrame(),
                'motion_segments': pd.DataFrame(),
                'regime_classification': {'success': False, 'error': f'Invalid input data - missing columns: {missing_columns}'},
                'analysis_parameters': {
                    'window_size': window_size,
                    'min_segment_length': min_segment_length,
                    'significance_level': significance_level
                }
            }
        
        changepoints_data = []
        track_segments = []
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id]
            
            # Ensure track_data is a DataFrame and not empty
            if not isinstance(track_data, pd.DataFrame) or track_data.empty:
                continue
                
            # Sort by frame if column exists
            if 'frame' in track_data.columns:
                track_data = track_data.sort_values('frame')
            
            if len(track_data) < window_size * 3:
                continue
            
            # Extract position data with error handling
            try:
                x = track_data['x'].values
                y = track_data['y'].values
                frames = track_data['frame'].values if 'frame' in track_data.columns else np.arange(len(track_data))
            except (KeyError, AttributeError):
                continue
            
            # Calculate motion parameters
            dx = np.diff(x)
            dy = np.diff(y)
            displacements = np.sqrt(dx**2 + dy**2)
            
            # Handle case where all displacements are zero
            if np.all(displacements == 0):
                continue
                
            angles = np.arctan2(dy, dx)
            
            # Calculate sliding window statistics
            motion_features = self._calculate_sliding_window_features(
                displacements, angles, window_size
            )
            
            # Detect changepoints using multiple methods
            changepoints = []
            
            # Method 1: Variance changepoint detection
            variance_changes = self._detect_variance_changepoints(
                displacements, window_size, significance_level
            )
            changepoints.extend([{'type': 'variance', 'position': cp} for cp in variance_changes])
            
            # Method 2: Mean displacement changepoint detection
            mean_changes = self._detect_mean_changepoints(
                displacements, window_size, significance_level
            )
            changepoints.extend([{'type': 'mean_displacement', 'position': cp} for cp in mean_changes])
            
            # Method 3: Directional changepoint detection
            direction_changes = self._detect_directional_changepoints(
                angles, window_size, significance_level
            )
            changepoints.extend([{'type': 'direction', 'position': cp} for cp in direction_changes])
            
            # Merge nearby changepoints
            if changepoints:
                merged_changepoints = self._merge_nearby_changepoints(
                    changepoints, min_distance=window_size//2
                )
                
                # Create segments between changepoints
                segments = self._create_motion_segments(
                    merged_changepoints, len(track_data), min_segment_length
                )
                
                # Analyze each segment
                for seg_idx, segment in enumerate(segments):
                    start_idx, end_idx = segment['start'], segment['end']
                    
                    if end_idx - start_idx >= min_segment_length:
                        seg_displacements = displacements[start_idx:min(end_idx, len(displacements))]
                        seg_angles = angles[start_idx:min(end_idx, len(angles))] if start_idx < len(angles) else []
                        
                        # Calculate segment statistics
                        segment_stats = {
                            'track_id': track_id,
                            'segment_id': seg_idx,
                            'start_frame': frames[start_idx],
                            'end_frame': frames[min(end_idx, len(frames)-1)],
                            'length': end_idx - start_idx,
                            'mean_displacement': np.mean(seg_displacements) if len(seg_displacements) > 0 else 0,
                            'std_displacement': np.std(seg_displacements) if len(seg_displacements) > 0 else 0,
                            'mean_speed': np.mean(seg_displacements) if len(seg_displacements) > 0 else 0,
                            'displacement_cv': np.std(seg_displacements) / np.mean(seg_displacements) if len(seg_displacements) > 0 and np.mean(seg_displacements) > 0 else 0
                        }
                        
                        if len(seg_angles) > 0:
                            # Calculate directional persistence
                            segment_stats['directional_persistence'] = self._calculate_directional_persistence(seg_angles)
                            segment_stats['mean_angle'] = np.mean(seg_angles)
                            segment_stats['angle_std'] = np.std(seg_angles)
                        else:
                            segment_stats['directional_persistence'] = 0
                            segment_stats['mean_angle'] = 0
                            segment_stats['angle_std'] = 0
                        
                        track_segments.append(segment_stats)
                
                # Store changepoints
                for cp in merged_changepoints:
                    if cp['position'] < len(frames):
                        changepoints_data.append({
                            'track_id': track_id,
                            'changepoint_frame': frames[cp['position']],
                            'changepoint_type': cp['type'],
                            'position_in_track': cp['position']
                        })
        
        # Convert track_segments list to DataFrame
        motion_segments_df = pd.DataFrame(track_segments) if track_segments else pd.DataFrame()
        
        # Classify motion regimes using the DataFrame
        regime_classification = self._classify_motion_regimes(motion_segments_df)
        
        return {
            'success': True if changepoints_data or track_segments else False,
            'changepoints': pd.DataFrame(changepoints_data) if changepoints_data else pd.DataFrame(),
            'motion_segments': motion_segments_df,
            'regime_classification': regime_classification,
            'analysis_parameters': {
                'window_size': window_size,
                'min_segment_length': min_segment_length,
                'significance_level': significance_level
            }
        }
    
    def detect_intensity_changepoints(self, tracks_df: pd.DataFrame,
                                    intensity_column: str,
                                    method: str = 'cusum') -> Dict[str, Any]:
        """
        Detect changepoints in intensity signals.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with intensity information
        intensity_column : str
            Name of intensity column to analyze
        method : str
            Changepoint detection method ('cusum', 'pelt', or 'variance')
            
        Returns
        -------
        dict
            Intensity changepoint results
        """
        # Validate input DataFrame
        if tracks_df.empty or 'track_id' not in tracks_df.columns:
            return {
                'success': False,
                'error': 'Invalid input data: empty DataFrame or missing track_id column'
            }
        
        if intensity_column not in tracks_df.columns:
            return {
                'success': False,
                'error': f'Column {intensity_column} not found in data'
            }
        
        intensity_changepoints = []
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 20:  # Need sufficient data
                continue
            
            intensity = track_data[intensity_column].dropna()
            if len(intensity) < 10:
                continue
            
            frames = track_data.loc[intensity.index, 'frame'].values
            intensity_values = intensity.values
            
            # Apply different changepoint detection methods
            changepoints = []
            
            if method == 'cusum':
                changepoints = self._cusum_changepoint_detection(intensity_values)
            elif method == 'variance':
                changepoints = self._variance_changepoint_detection(intensity_values)
            else:  # Default to sliding window method
                changepoints = self._sliding_window_changepoint_detection(intensity_values)
            
            # Convert to frame numbers and store
            for cp in changepoints:
                if 0 <= cp < len(frames):
                    intensity_changepoints.append({
                        'track_id': track_id,
                        'frame': frames[cp],
                        'position_in_track': cp,
                        'intensity_before': intensity_values[max(0, cp-5):cp].mean() if cp > 5 else intensity_values[0],
                        'intensity_after': intensity_values[cp:cp+5].mean() if cp+5 < len(intensity_values) else intensity_values[-1],
                        'intensity_change': (intensity_values[cp:cp+5].mean() if cp+5 < len(intensity_values) else intensity_values[-1]) - 
                                          (intensity_values[max(0, cp-5):cp].mean() if cp > 5 else intensity_values[0])
                    })
        
        return {
            'success': True if intensity_changepoints else False,
            'intensity_changepoints': pd.DataFrame(intensity_changepoints) if intensity_changepoints else pd.DataFrame(),
            'method_used': method
        }
    
    def analyze_transition_kinetics(self, changepoints_df: pd.DataFrame,
                                  segments_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze kinetics of transitions between motion regimes.
        
        Parameters
        ----------
        changepoints_df : pd.DataFrame
            Changepoint data
        segments_df : pd.DataFrame
            Motion segment data
            
        Returns
        -------
        dict
            Transition kinetics analysis
        """
        if changepoints_df.empty or segments_df.empty:
            return {
                'success': False,
                'error': 'No changepoints or segments data provided'
            }
        
        transition_data = []
        
        # Analyze transitions between consecutive segments
        for track_id in segments_df['track_id'].unique():
            track_segments = segments_df[segments_df['track_id'] == track_id].sort_values('start_frame')
            
            for i in range(len(track_segments) - 1):
                current_seg = track_segments.iloc[i]
                next_seg = track_segments.iloc[i + 1]
                
                # Calculate transition properties
                transition = {
                    'track_id': track_id,
                    'from_segment': current_seg['segment_id'],
                    'to_segment': next_seg['segment_id'],
                    'transition_frame': current_seg['end_frame'],
                    'from_regime': current_seg.get('regime_class', 'unknown'),
                    'to_regime': next_seg.get('regime_class', 'unknown'),
                    'from_mean_displacement': current_seg['mean_displacement'],
                    'to_mean_displacement': next_seg['mean_displacement'],
                    'displacement_change_ratio': next_seg['mean_displacement'] / current_seg['mean_displacement'] if current_seg['mean_displacement'] > 0 else np.inf,
                    'from_directional_persistence': current_seg.get('directional_persistence', 0),
                    'to_directional_persistence': next_seg.get('directional_persistence', 0)
                }
                
                transition_data.append(transition)
        
        # Calculate ensemble statistics
        ensemble_stats = {}
        if transition_data:
            trans_df = pd.DataFrame(transition_data)
            
            # Count different types of transitions
            transition_types = trans_df.groupby(['from_regime', 'to_regime']).size()
            
            ensemble_stats = {
                'total_transitions': len(trans_df),
                'transition_types': transition_types.to_dict(),
                'mean_displacement_change_ratio': trans_df['displacement_change_ratio'].replace([np.inf, -np.inf], np.nan).mean(),
                'median_displacement_change_ratio': trans_df['displacement_change_ratio'].replace([np.inf, -np.inf], np.nan).median()
            }
            
            # Calculate transition probabilities
            if len(transition_types) > 0:
                total_transitions = transition_types.sum()
                transition_probabilities = (transition_types / total_transitions).to_dict()
                ensemble_stats['transition_probabilities'] = transition_probabilities
        
        return {
            'success': True if transition_data else False,
            'transitions': pd.DataFrame(transition_data) if transition_data else pd.DataFrame(),
            'ensemble_statistics': ensemble_stats
        }
    
    def _calculate_sliding_window_features(self, displacements: np.ndarray,
                                         angles: np.ndarray, 
                                         window_size: int) -> np.ndarray:
        """Calculate motion features in sliding windows."""
        n_windows = len(displacements) - window_size + 1
        features = np.zeros((n_windows, 4))  # mean, std, directional_persistence, angle_variance
        
        for i in range(n_windows):
            window_disp = displacements[i:i+window_size]
            window_angles = angles[i:i+window_size]
            
            features[i, 0] = np.mean(window_disp)
            features[i, 1] = np.std(window_disp)
            features[i, 2] = self._calculate_directional_persistence(window_angles)
            features[i, 3] = np.var(window_angles)
        
        return features
    
    def _detect_variance_changepoints(self, data: np.ndarray, 
                                    window_size: int, 
                                    significance_level: float) -> List[int]:
        """Detect changepoints based on variance changes."""
        changepoints = []
        
        for i in range(window_size, len(data) - window_size):
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # F-test for variance equality
            if len(before) > 1 and len(after) > 1:
                var_before = np.var(before, ddof=1)
                var_after = np.var(after, ddof=1)
                
                if var_before > 0 and var_after > 0:
                    f_stat = var_after / var_before
                    
                    # Calculate proper F-test critical values
                    dfn = len(after) - 1
                    dfd = len(before) - 1
                    
                    # Two-sided F-test critical values
                    critical_upper = stats.f.ppf(1 - significance_level / 2, dfn, dfd)
                    critical_lower = stats.f.ppf(significance_level / 2, dfn, dfd)
                    
                    if f_stat > critical_upper or f_stat < critical_lower:
                        changepoints.append(i)
        
        return changepoints
    
    def _detect_mean_changepoints(self, data: np.ndarray,
                                window_size: int,
                                significance_level: float) -> List[int]:
        """Detect changepoints based on mean changes."""
        changepoints = []
        
        for i in range(window_size, len(data) - window_size):
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # T-test for mean equality
            if len(before) > 1 and len(after) > 1:
                t_stat, p_value = stats.ttest_ind(before, after)
                
                if p_value < significance_level:
                    changepoints.append(i)
        
        return changepoints
    
    def _detect_directional_changepoints(self, angles: np.ndarray,
                                       window_size: int,
                                       significance_level: float) -> List[int]:
        """Detect changepoints in directional motion."""
        changepoints = []
        
        for i in range(window_size, len(angles) - window_size):
            before = angles[i-window_size:i]
            after = angles[i:i+window_size]
            
            # Calculate directional persistence for both windows
            persist_before = self._calculate_directional_persistence(before)
            persist_after = self._calculate_directional_persistence(after)
            
            # Threshold-based detection (simplified)
            if abs(persist_before - persist_after) > 0.5:
                changepoints.append(i)
        
        return changepoints
    
    def _calculate_directional_persistence(self, angles: np.ndarray) -> float:
        """Calculate directional persistence from angle sequence."""
        if len(angles) < 2:
            return 0
        
        # Calculate angle differences
        angle_diffs = np.diff(angles)
        
        # Wrap angles to [-pi, pi]
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        
        # Calculate persistence as inverse of angular variance
        if len(angle_diffs) > 0:
            angular_var = np.var(angle_diffs)
            persistence = np.exp(-angular_var)  # Exponential decay with variance
        else:
            persistence = 0
        
        return persistence
    
    def _merge_nearby_changepoints(self, changepoints: List[Dict], 
                                 min_distance: int) -> List[Dict]:
        """Merge changepoints that are close together."""
        if not changepoints:
            return []
        
        # Sort by position
        sorted_cps = sorted(changepoints, key=lambda x: x['position'])
        
        merged = [sorted_cps[0]]
        
        for cp in sorted_cps[1:]:
            if cp['position'] - merged[-1]['position'] > min_distance:
                merged.append(cp)
            else:
                # Keep the one with higher significance (simplified: keep the first)
                pass
        
        return merged
    
    def _create_motion_segments(self, changepoints: List[Dict], 
                              track_length: int, 
                              min_segment_length: int) -> List[Dict]:
        """Create motion segments between changepoints."""
        segments = []
        
        if not changepoints:
            # Single segment for entire track
            if track_length >= min_segment_length:
                segments.append({'start': 0, 'end': track_length})
        else:
            # Sort changepoints
            sorted_cps = sorted(changepoints, key=lambda x: x['position'])
            
            # First segment
            if sorted_cps[0]['position'] >= min_segment_length:
                segments.append({'start': 0, 'end': sorted_cps[0]['position']})
            
            # Middle segments
            for i in range(len(sorted_cps) - 1):
                start = sorted_cps[i]['position']
                end = sorted_cps[i + 1]['position']
                
                if end - start >= min_segment_length:
                    segments.append({'start': start, 'end': end})
            
            # Last segment
            last_start = sorted_cps[-1]['position']
            if track_length - last_start >= min_segment_length:
                segments.append({'start': last_start, 'end': track_length})
        
        return segments
    
    def _classify_motion_regimes(self, segments_df: pd.DataFrame) -> Dict[str, Any]:
        """Classify motion segments into regimes using clustering."""
        if segments_df.empty:
            return {'success': False, 'error': 'No segments to classify'}
        
        # Extract features for classification
        features = []
        feature_names = ['mean_displacement', 'std_displacement', 'displacement_cv', 'directional_persistence']
        
        for feat in feature_names:
            if feat not in segments_df.columns:
                segments_df[feat] = 0
        
        features = segments_df[feature_names].fillna(0).values
        
        features = np.array(features)
        
        # Handle NaN values
        features = np.nan_to_num(features)
        
        if features.shape[0] < 2:
            return {'success': False, 'error': 'Insufficient segments for classification'}
        
        # K-means clustering to identify regimes
        n_clusters = min(4, features.shape[0])  # Max 4 regimes
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(features)
            
            # Assign regime names based on characteristics
            regime_names = []
            for cluster_id in range(n_clusters):
                cluster_mask = regime_labels == cluster_id
                cluster_features = features[cluster_mask]
                
                mean_displacement = np.mean(cluster_features[:, 0])
                mean_persistence = np.mean(cluster_features[:, 3])
                
                if mean_displacement < 1.0:
                    if mean_persistence > 0.5:
                        regime_name = 'confined_directed'
                    else:
                        regime_name = 'confined_random'
                else:
                    if mean_persistence > 0.5:
                        regime_name = 'directed_transport'
                    else:
                        regime_name = 'free_diffusion'
                
                regime_names.append(regime_name)
            
            # Add regime classification to segments
            segments_with_regimes = segments_df.copy()
            segments_with_regimes['regime_class'] = [regime_names[label] for label in regime_labels]
            segments_with_regimes['regime_id'] = regime_labels
            
            return {
                'success': True,
                'classified_segments': segments_with_regimes,
                'regime_centers': kmeans.cluster_centers_,
                'regime_names': regime_names,
                'n_regimes': n_clusters
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Classification failed: {str(e)}'
            }
    
    def _cusum_changepoint_detection(self, data: np.ndarray) -> List[int]:
        """CUSUM-based changepoint detection."""
        changepoints = []
        
        # Parameters for CUSUM
        threshold = 5 * np.std(data)
        drift = 0.5 * np.std(data)
        
        # Initialize
        s_pos = s_neg = 0
        
        for i in range(1, len(data)):
            diff = data[i] - data[i-1]
            
            s_pos = max(0, s_pos + diff - drift)
            s_neg = min(0, s_neg + diff + drift)
            
            if s_pos > threshold or s_neg < -threshold:
                changepoints.append(i)
                s_pos = s_neg = 0  # Reset
        
        return changepoints
    
    def _variance_changepoint_detection(self, data: np.ndarray) -> List[int]:
        """Variance-based changepoint detection using sliding windows."""
        changepoints = []
        window_size = min(10, len(data) // 4)
        
        if window_size < 3:
            return changepoints
        
        for i in range(window_size, len(data) - window_size):
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            var_before = np.var(before)
            var_after = np.var(after)
            
            # Variance ratio test
            if var_before > 0 and var_after > 0:
                ratio = max(var_after, var_before) / min(var_after, var_before)
                if ratio > 3:  # Threshold for significant variance change
                    changepoints.append(i)
        
        return changepoints
    
    def _sliding_window_changepoint_detection(self, data: np.ndarray) -> List[int]:
        """Simple sliding window changepoint detection."""
        changepoints = []
        window_size = min(5, len(data) // 5)
        
        if window_size < 2:
            return changepoints
        
        for i in range(window_size, len(data) - window_size):
            before = np.mean(data[i-window_size:i])
            after = np.mean(data[i:i+window_size])
            
            # Threshold-based detection
            threshold = 2 * np.std(data)
            if abs(after - before) > threshold:
                changepoints.append(i)
        
        return changepoints
