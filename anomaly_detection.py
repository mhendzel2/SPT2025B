"""
AI-Powered Anomaly Detection for Single Particle Tracking

This module implements advanced machine learning algorithms to detect anomalous
particle behavior patterns in SPT data, including:
- Sudden velocity changes
- Confinement violations 
- Directional reversals
- Energy state transitions
- Spatial clustering anomalies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Advanced anomaly detection for particle tracking data using multiple ML approaches.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in the data
        random_state : int
            Random state for reproducible results
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.anomaly_scores = {}
        self.anomaly_types = {}
        
        # Emoji mapping for different anomaly types
        self.anomaly_emojis = {
            'velocity_spike': '⚡',      # High velocity anomalies
            'confinement_break': '💥',   # Particles breaking confinement
            'direction_reversal': '🔄',  # Sudden directional changes
            'energy_transition': '🌊',   # Energy state changes
            'spatial_outlier': '🎯',     # Spatial clustering anomalies
            'ml_anomaly': '🤖',         # Machine learning detected
            'statistical_outlier': '📊', # Statistical outliers
            'track_fragmentation': '💔', # Track interruptions
            'drift_anomaly': '🌀',      # Systematic drift
            'binding_event': '🔗',      # Potential binding/unbinding
            'normal': '✅'              # Normal behavior
        }
        
    def extract_features(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for anomaly detection.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
            
        Returns
        -------
        pd.DataFrame
            Feature matrix for anomaly detection
        """
        features_list = []
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 3:
                continue
                
            # Basic trajectory features
            x = track_data['x'].values
            y = track_data['y'].values
            frames = track_data['frame'].values
            
            # Calculate displacements and velocities
            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(frames)
            dt[dt == 0] = 1  # Avoid division by zero
            
            velocities = np.sqrt(dx**2 + dy**2) / dt
            accelerations = np.diff(velocities) if len(velocities) > 1 else np.array([0])
            
            # Directional features
            angles = np.arctan2(dy, dx)
            angle_changes = np.diff(angles) if len(angles) > 1 else np.array([0])
            # Handle angle wrapping
            angle_changes = np.where(angle_changes > np.pi, angle_changes - 2*np.pi, angle_changes)
            angle_changes = np.where(angle_changes < -np.pi, angle_changes + 2*np.pi, angle_changes)
            
            # Confinement features
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            bounding_area = x_range * y_range
            
            # MSD-based features (simplified)
            max_lag = min(10, len(x) // 2)
            msd_values = []
            for lag in range(1, max_lag + 1):
                if lag < len(x):
                    msd = np.mean((x[lag:] - x[:-lag])**2 + (y[lag:] - y[:-lag])**2)
                    msd_values.append(msd)
            
            msd_slope = 0
            if len(msd_values) > 1:
                try:
                    msd_slope = np.polyfit(range(len(msd_values)), msd_values, 1)[0]
                except:
                    msd_slope = 0
            
            # Calculate displacement ratio safely
            total_displacement = np.sum(np.sqrt(dx**2 + dy**2))
            net_displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            displacement_ratio = net_displacement / total_displacement if total_displacement > 0 else 0
            
            # Calculate confinement ratio safely
            mean_velocity = np.mean(velocities) if len(velocities) > 0 else 0
            confinement_ratio = np.sqrt(bounding_area) / mean_velocity if mean_velocity > 0 else 0
            
            # Compile features
            features = {
                'track_id': track_id,
                'track_length': len(track_data),
                'velocity_mean': np.mean(velocities) if len(velocities) > 0 else 0,
                'velocity_std': np.std(velocities) if len(velocities) > 0 else 0,
                'velocity_max': np.max(velocities) if len(velocities) > 0 else 0,
                'acceleration_mean': np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0,
                'acceleration_std': np.std(accelerations) if len(accelerations) > 0 else 0,
                'angle_change_mean': np.mean(np.abs(angle_changes)) if len(angle_changes) > 0 else 0,
                'angle_change_std': np.std(angle_changes) if len(angle_changes) > 0 else 0,
                'directional_reversals': np.sum(np.abs(angle_changes) > np.pi/2) if len(angle_changes) > 0 else 0,
                'x_range': x_range,
                'y_range': y_range,
                'bounding_area': bounding_area,
                'displacement_ratio': displacement_ratio,
                'msd_slope': msd_slope,
                'confinement_ratio': confinement_ratio
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def detect_velocity_anomalies(self, tracks_df: pd.DataFrame, z_threshold: float = 3.0) -> Dict[int, List[int]]:
        """
        Detect sudden velocity changes using statistical thresholds.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        z_threshold : float
            Z-score threshold for anomaly detection
            
        Returns
        -------
        Dict[int, List[int]]
            Dictionary mapping track_id to list of anomalous frame indices
        """
        velocity_anomalies = {}
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 4:
                continue
                
            x = track_data['x'].values
            y = track_data['y'].values
            frames = track_data['frame'].values
            
            # Calculate velocities
            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(frames)
            dt[dt == 0] = 1
            
            velocities = np.sqrt(dx**2 + dy**2) / dt
            
            # Detect outliers using z-score
            if len(velocities) > 2:
                try:
                    z_scores = np.abs(stats.zscore(velocities))
                    anomalous_indices = np.where(z_scores > z_threshold)[0]
                    
                    if len(anomalous_indices) > 0:
                        # Map back to frame numbers
                        anomalous_frames = frames[anomalous_indices + 1].tolist()  # +1 because velocities are calculated between frames
                        velocity_anomalies[track_id] = anomalous_frames
                except:
                    # Handle case where z-score calculation fails
                    continue
        
        return velocity_anomalies
    
    def detect_confinement_violations(self, tracks_df: pd.DataFrame, expansion_threshold: float = 2.0) -> Dict[int, List[int]]:
        """
        Detect sudden expansions beyond expected confinement regions.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        expansion_threshold : float
            Threshold for detecting expansion beyond normal range
            
        Returns
        -------
        Dict[int, List[int]]
            Dictionary mapping track_id to list of frames with confinement violations
        """
        confinement_anomalies = {}
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 10:  # Need sufficient data to establish baseline
                continue
                
            x = track_data['x'].values
            y = track_data['y'].values
            frames = track_data['frame'].values
            
            # Calculate rolling statistics to establish baseline confinement
            window_size = min(10, len(x) // 3)
            if window_size < 3:
                continue
                
            # Calculate distances from track center
            center_x = np.mean(x[:window_size])
            center_y = np.mean(y[:window_size])
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Establish baseline confinement radius
            baseline_radius = np.percentile(distances[:window_size], 90)
            
            # Detect violations
            violations = distances > baseline_radius * expansion_threshold
            violation_frames = frames[violations].tolist()
            
            if len(violation_frames) > 0:
                confinement_anomalies[track_id] = violation_frames
        
        return confinement_anomalies
    
    def detect_directional_anomalies(self, tracks_df: pd.DataFrame, reversal_threshold: float = 2.5) -> Dict[int, List[int]]:
        """
        Detect unusual directional changes and reversals.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        reversal_threshold : float
            Threshold in radians for detecting significant directional changes
            
        Returns
        -------
        Dict[int, List[int]]
            Dictionary mapping track_id to list of frames with directional anomalies
        """
        directional_anomalies = {}
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 4:
                continue
                
            x = track_data['x'].values
            y = track_data['y'].values
            frames = track_data['frame'].values
            
            # Calculate directional angles
            dx = np.diff(x)
            dy = np.diff(y)
            angles = np.arctan2(dy, dx)
            
            # Calculate angle changes
            angle_changes = np.diff(angles) if len(angles) > 1 else np.array([])
            if len(angle_changes) == 0:
                continue
                
            # Handle angle wrapping
            angle_changes = np.where(angle_changes > np.pi, angle_changes - 2*np.pi, angle_changes)
            angle_changes = np.where(angle_changes < -np.pi, angle_changes + 2*np.pi, angle_changes)
            
            # Detect significant reversals
            significant_changes = np.abs(angle_changes) > reversal_threshold
            anomalous_indices = np.where(significant_changes)[0]
            
            if len(anomalous_indices) > 0:
                # Map back to frame numbers
                anomalous_frames = frames[anomalous_indices + 2].tolist()  # +2 because angle_changes is diff of diff
                directional_anomalies[track_id] = anomalous_frames
        
        return directional_anomalies
    
    def detect_ml_anomalies(self, tracks_df: pd.DataFrame) -> Dict[int, float]:
        """
        Use machine learning (Isolation Forest) to detect anomalous tracks.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping track_id to anomaly score (negative = anomalous)
        """
        try:
            # Extract features
            features_df = self.extract_features(tracks_df)
            
            if len(features_df) < 2:
                return {}
            
            # Prepare feature matrix
            feature_cols = [col for col in features_df.columns if col != 'track_id']
            X = features_df[feature_cols].values
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100
            )
            
            anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
            anomaly_scores = self.isolation_forest.score_samples(X_scaled)
            
            # Create results dictionary
            results = {}
            for i, track_id in enumerate(features_df['track_id']):
                results[track_id] = anomaly_scores[i]
            
            return results
            
        except Exception as e:
            print(f"Error in ML anomaly detection: {e}")
            return {}
    
    def detect_spatial_clustering_anomalies(self, tracks_df: pd.DataFrame, eps: float = 5.0) -> Dict[int, str]:
        """
        Detect tracks that don't follow expected spatial clustering patterns.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        eps : float
            DBSCAN epsilon parameter for clustering
            
        Returns
        -------
        Dict[int, str]
            Dictionary mapping track_id to cluster label ('outlier' for anomalies)
        """
        try:
            # Calculate track centroids
            track_centers = []
            track_ids = []
            
            for track_id in tracks_df['track_id'].unique():
                track_data = tracks_df[tracks_df['track_id'] == track_id]
                center_x = track_data['x'].mean()
                center_y = track_data['y'].mean()
                track_centers.append([center_x, center_y])
                track_ids.append(track_id)
            
            if len(track_centers) < 2:
                return {}
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=2)
            cluster_labels = clustering.fit_predict(track_centers)
            
            # Identify outliers (label = -1)
            results = {}
            for i, track_id in enumerate(track_ids):
                if cluster_labels[i] == -1:
                    results[track_id] = 'outlier'
                else:
                    results[track_id] = f'cluster_{cluster_labels[i]}'
            
            return results
            
        except Exception as e:
            print(f"Error in spatial clustering: {e}")
            return {}
    
    def classify_anomaly_type(self, track_id: int, velocity_anomalies: Dict, confinement_anomalies: Dict, 
                             directional_anomalies: Dict, ml_scores: Dict, spatial_anomalies: Dict) -> Tuple[str, str, str]:
        """
        Classify the primary anomaly type for a track and return emoji and description.
        
        Parameters
        ----------
        track_id : int
            Track identifier
        velocity_anomalies, confinement_anomalies, directional_anomalies : Dict
            Anomaly detection results from various methods
        ml_scores : Dict
            Machine learning anomaly scores
        spatial_anomalies : Dict
            Spatial clustering anomaly results
            
        Returns
        -------
        Tuple[str, str, str]
            (anomaly_type, emoji, description)
        """
        anomaly_types = []
        
        # Check for velocity anomalies
        if track_id in velocity_anomalies and len(velocity_anomalies[track_id]) > 0:
            anomaly_types.append('velocity_spike')
        
        # Check for confinement breaks
        if track_id in confinement_anomalies and len(confinement_anomalies[track_id]) > 0:
            anomaly_types.append('confinement_break')
        
        # Check for directional reversals
        if track_id in directional_anomalies and len(directional_anomalies[track_id]) > 0:
            anomaly_types.append('direction_reversal')
        
        # Check for ML-detected anomalies
        if track_id in ml_scores and ml_scores[track_id] < -0.1:
            anomaly_types.append('ml_anomaly')
        
        # Check for spatial outliers
        if track_id in spatial_anomalies and spatial_anomalies[track_id] == 'outlier':
            anomaly_types.append('spatial_outlier')
        
        # Determine primary anomaly type
        if not anomaly_types:
            return 'normal', self.anomaly_emojis['normal'], "Normal particle behavior"
        
        # Priority order for multiple anomalies
        priority_order = ['velocity_spike', 'confinement_break', 'direction_reversal', 'ml_anomaly', 'spatial_outlier']
        
        primary_type = None
        for ptype in priority_order:
            if ptype in anomaly_types:
                primary_type = ptype
                break
        
        if primary_type is None:
            primary_type = anomaly_types[0]
        
        # Generate descriptions
        descriptions = {
            'velocity_spike': "Sudden velocity increase detected - potential motor protein activation or external force",
            'confinement_break': "Particle escaped confinement region - possible membrane permeation or structure change",
            'direction_reversal': "Sharp directional change detected - potential obstacle encounter or binding event",
            'ml_anomaly': "Machine learning detected unusual behavior pattern - complex anomaly requiring investigation",
            'spatial_outlier': "Track location deviates from typical spatial distribution - possible experimental artifact",
            'normal': "Normal particle behavior within expected parameters"
        }
        
        return primary_type, self.anomaly_emojis[primary_type], descriptions[primary_type]
    
    def generate_detailed_tooltip(self, track_id: int, track_data: pd.DataFrame, 
                                 anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive tooltip information for a specific track.
        
        Parameters
        ----------
        track_id : int
            Track identifier
        track_data : pd.DataFrame
            Data for the specific track
        anomaly_results : Dict
            Results from comprehensive anomaly detection
            
        Returns
        -------
        Dict[str, Any]
            Detailed tooltip information with emojis and descriptions
        """
        # Basic track statistics
        track_stats = {
            'track_id': track_id,
            'length': len(track_data),
            'duration': track_data['frame'].max() - track_data['frame'].min() + 1,
            'mean_velocity': 0,
            'max_displacement': 0
        }
        
        # Calculate velocities if possible
        if len(track_data) > 1:
            dx = np.diff(track_data['x'].values)
            dy = np.diff(track_data['y'].values)
            velocities = np.sqrt(dx**2 + dy**2)
            track_stats['mean_velocity'] = np.mean(velocities)
            track_stats['max_displacement'] = np.max(velocities)
        
        # Get anomaly classification
        anomaly_type, emoji, description = self.classify_anomaly_type(
            track_id,
            anomaly_results.get('velocity_anomalies', {}),
            anomaly_results.get('confinement_violations', {}),  # Fixed key name
            anomaly_results.get('directional_anomalies', {}),
            anomaly_results.get('ml_anomaly_scores', {}),
            anomaly_results.get('spatial_clustering', {})  # Fixed key name
        )
        
        # Build tooltip data
        tooltip_data = {
            'primary_emoji': emoji,
            'anomaly_type': anomaly_type,
            'description': description,
            'track_stats': track_stats,
            'anomaly_details': []
        }
        
        # Add specific anomaly details
        velocity_frames = anomaly_results.get('velocity_anomalies', {}).get(track_id, [])
        if velocity_frames:
            tooltip_data['anomaly_details'].append({
                'type': 'Velocity Spike',
                'emoji': '⚡',
                'frames': velocity_frames,
                'count': len(velocity_frames),
                'description': f"High velocity detected at {len(velocity_frames)} time points"
            })
        
        confinement_frames = anomaly_results.get('confinement_violations', {}).get(track_id, [])
        if confinement_frames:
            tooltip_data['anomaly_details'].append({
                'type': 'Confinement Break',
                'emoji': '💥',
                'frames': confinement_frames,
                'count': len(confinement_frames),
                'description': f"Escaped confinement at {len(confinement_frames)} time points"
            })
        
        directional_frames = anomaly_results.get('directional_anomalies', {}).get(track_id, [])
        if directional_frames:
            tooltip_data['anomaly_details'].append({
                'type': 'Direction Change',
                'emoji': '🔄',
                'frames': directional_frames,
                'count': len(directional_frames),
                'description': f"Sharp directional changes at {len(directional_frames)} time points"
            })
        
        ml_score = anomaly_results.get('ml_anomaly_scores', {}).get(track_id, 0)
        if ml_score < -0.1:
            tooltip_data['anomaly_details'].append({
                'type': 'ML Anomaly',
                'emoji': '🤖',
                'score': ml_score,
                'description': f"ML anomaly score: {ml_score:.3f} (more negative = more anomalous)"
            })
        
        spatial_label = anomaly_results.get('spatial_clustering', {}).get(track_id, 'normal')
        if spatial_label == 'outlier':
            tooltip_data['anomaly_details'].append({
                'type': 'Spatial Outlier',
                'emoji': '🎯',
                'description': "Track location deviates from typical spatial distribution"
            })
        
        return tooltip_data
    
    def create_emoji_summary_table(self, anomaly_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary table with emoji indicators for all tracks.
        
        Parameters
        ----------
        anomaly_results : Dict
            Results from comprehensive anomaly detection
            
        Returns
        -------
        pd.DataFrame
            Summary table with emoji indicators and basic statistics
        """
        summary_data = []
        
        # Get all unique track IDs from all anomaly detection results
        all_track_ids = set()
        for result_dict in anomaly_results.values():
            if isinstance(result_dict, dict):
                all_track_ids.update(result_dict.keys())
        
        for track_id in sorted(all_track_ids):
            # Classify anomaly type
            anomaly_type, emoji, description = self.classify_anomaly_type(
                track_id,
                anomaly_results.get('velocity_anomalies', {}),
                anomaly_results.get('confinement_violations', {}),
                anomaly_results.get('directional_anomalies', {}),
                anomaly_results.get('ml_anomaly_scores', {}),
                anomaly_results.get('spatial_clustering', {})
            )
            
            # Count total anomalies
            total_anomalies = 0
            anomaly_types_present = []
            
            if track_id in anomaly_results.get('velocity_anomalies', {}):
                count = len(anomaly_results['velocity_anomalies'][track_id])
                total_anomalies += count
                if count > 0:
                    anomaly_types_present.append(f"⚡×{count}")
            
            if track_id in anomaly_results.get('confinement_violations', {}):
                count = len(anomaly_results['confinement_violations'][track_id])
                total_anomalies += count
                if count > 0:
                    anomaly_types_present.append(f"💥×{count}")
            
            if track_id in anomaly_results.get('directional_anomalies', {}):
                count = len(anomaly_results['directional_anomalies'][track_id])
                total_anomalies += count
                if count > 0:
                    anomaly_types_present.append(f"🔄×{count}")
            
            if track_id in anomaly_results.get('ml_anomaly_scores', {}):
                score = anomaly_results['ml_anomaly_scores'][track_id]
                if score < -0.1:
                    anomaly_types_present.append(f"🤖({score:.2f})")
            
            if track_id in anomaly_results.get('spatial_clustering', {}):
                if anomaly_results['spatial_clustering'][track_id] == 'outlier':
                    anomaly_types_present.append("🎯")
            
            summary_data.append({
                'Track ID': track_id,
                'Status': emoji,
                'Anomaly Type': anomaly_type.replace('_', ' ').title(),
                'Total Anomalies': total_anomalies,
                'Details': ' '.join(anomaly_types_present) if anomaly_types_present else '✅',
                'Description': description[:50] + "..." if len(description) > 50 else description
            })
        
        return pd.DataFrame(summary_data)
    
    def comprehensive_anomaly_detection(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive anomaly detection using all available methods.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive anomaly detection results
        """
        results = {
            'velocity_anomalies': self.detect_velocity_anomalies(tracks_df),
            'confinement_violations': self.detect_confinement_violations(tracks_df),
            'directional_anomalies': self.detect_directional_anomalies(tracks_df),
            'ml_anomaly_scores': self.detect_ml_anomalies(tracks_df),
            'spatial_clustering': self.detect_spatial_clustering_anomalies(tracks_df)
        }
        
        # Store results for visualization
        self.anomaly_scores = results['ml_anomaly_scores']
        self.anomaly_types = self._categorize_anomalies(results)
        
        return results
    
    def _categorize_anomalies(self, results: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Categorize each track by types of anomalies detected.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Results from comprehensive_anomaly_detection
            
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping track_id to list of anomaly types
        """
        track_anomalies = {}
        
        # Get all track IDs
        all_track_ids = set()
        for category_results in results.values():
            if isinstance(category_results, dict):
                all_track_ids.update(category_results.keys())
        
        # Categorize each track
        for track_id in all_track_ids:
            anomaly_types = []
            
            # Check each anomaly type
            if track_id in results.get('velocity_anomalies', {}):
                anomaly_types.append('velocity')
            
            if track_id in results.get('confinement_violations', {}):
                anomaly_types.append('confinement')
            
            if track_id in results.get('directional_anomalies', {}):
                anomaly_types.append('directional')
            
            if track_id in results.get('ml_anomaly_scores', {}) and results['ml_anomaly_scores'][track_id] < 0:
                anomaly_types.append('ml_detected')
            
            if track_id in results.get('spatial_clustering', {}) and results['spatial_clustering'][track_id] == 'outlier':
                anomaly_types.append('spatial_outlier')
            
            if anomaly_types:
                track_anomalies[track_id] = anomaly_types
        
        return track_anomalies
    
    def analyze_anomalies(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method to analyze anomalies in tracking data.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Tracking data with columns: track_id, frame, x, y
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive anomaly analysis results
        """
        return self.comprehensive_anomaly_detection(tracks_df)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get a summary of detected anomalies.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics of anomaly detection results
        """
        if not self.anomaly_types:
            return {}
        
        total_anomalous_tracks = len(self.anomaly_types)
        total_tracks = len(self.anomaly_scores) if self.anomaly_scores else total_anomalous_tracks
        anomaly_counts = {}
        
        # Count anomalies by type
        for track_id, anomaly_list in self.anomaly_types.items():
            for anomaly_type in anomaly_list:
                anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
        
        return {
            'total_anomalous_tracks': total_anomalous_tracks,
            'total_tracks': total_tracks,
            'anomaly_type_counts': anomaly_counts,
            'anomaly_percentage': (total_anomalous_tracks / total_tracks) * 100 if total_tracks > 0 else 0
        }


def detect_anomalies_iforest(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in tracks using Isolation Forest.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns: track_id, frame, x, y
        
    Returns
    -------
    pd.DataFrame
        Anomalous tracks data
    """
    print("Detecting anomalies with Isolation Forest...")
    
    if tracks_df.empty:
        return pd.DataFrame()
        
    try:
        # Feature extraction (e.g., displacement between frames)
        tracks_df_copy = tracks_df.copy()
        tracks_df_copy['dx'] = tracks_df_copy.groupby('track_id')['x'].diff().fillna(0)
        tracks_df_copy['dy'] = tracks_df_copy.groupby('track_id')['y'].diff().fillna(0)
        tracks_df_copy['displacement'] = np.sqrt(tracks_df_copy['dx']**2 + tracks_df_copy['dy']**2)
        
        # Calculate speed (displacement per frame)
        tracks_df_copy['speed'] = tracks_df_copy['displacement']
        
        # Calculate directional change
        tracks_df_copy['direction'] = np.arctan2(tracks_df_copy['dy'], tracks_df_copy['dx'])
        tracks_df_copy['direction_change'] = tracks_df_copy.groupby('track_id')['direction'].diff().fillna(0)
        
        # Select features for anomaly detection
        features = tracks_df_copy[['displacement', 'speed', 'direction_change']].fillna(0)
        
        # Apply Isolation Forest
        model = IsolationForest(contamination='auto', random_state=42)
        tracks_df_copy['anomaly'] = model.fit_predict(features)
        
        # Anomalies are marked as -1 by the model
        anomalous_tracks = tracks_df_copy[tracks_df_copy['anomaly'] == -1].copy()
        
        return anomalous_tracks
        
    except ImportError:
        print("scikit-learn not available. Isolation Forest requires scikit-learn.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return pd.DataFrame()
