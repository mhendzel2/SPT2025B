"""
Advanced tracking methods for the SPT Analysis application.
Implements Bayesian tracking (Particle Filter), Kalman Filter, and CNN-based detection capabilities.
Enhanced with Numba JIT compilation for performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Performance optimization imports
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    
    prange = range

class ParticleFilter:
    """
    Advanced Bayesian particle filter for robust single particle tracking.
    Handles non-linear dynamics and non-Gaussian noise effectively.
    """
    
    def __init__(self, n_particles: int = 100, motion_std: float = 5.0, 
                 measurement_std: float = 2.0):
        """
        Initialize the particle filter.
        
        Parameters
        ----------
        n_particles : int
            Number of particles to maintain
        motion_std : float
            Standard deviation for motion model (pixels)
        measurement_std : float
            Standard deviation for measurement noise (pixels)
        """
        self.n_particles = n_particles
        self.motion_std = motion_std
        self.measurement_std = measurement_std
        self.particles = None
        self.weights = None
        self.state_estimate = None
        
    def initialize(self, initial_position: np.ndarray):
        """Initialize particles around initial position."""
        self.particles = np.random.normal(
            initial_position, 
            self.motion_std, 
            (self.n_particles, 2)
        )
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.state_estimate = initial_position.copy()
        
    def predict(self):
        """Prediction step - propagate particles according to motion model."""
        if self.particles is None:
            return
            
        # Simple Brownian motion model
        motion_noise = np.random.normal(0, self.motion_std, self.particles.shape)
        self.particles += motion_noise
        
    def update(self, measurement: np.ndarray):
        """Update step - reweight particles based on measurement."""
        if self.particles is None:
            return
            
        # Calculate likelihood for each particle
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        likelihoods = np.exp(-0.5 * (distances / self.measurement_std) ** 2)
        
        # Update weights
        self.weights *= likelihoods
        self.weights += 1e-12  # Avoid zero weights
        self.weights /= np.sum(self.weights)
        
        # Update state estimate
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Check if resampling is needed
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        if effective_sample_size < self.n_particles / 2:
            self.resample()
            
    def resample(self):
        """Resample particles if effective sample size is too low."""
        indices = np.random.choice(
            self.n_particles, 
            self.n_particles, 
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def get_state_estimate(self) -> np.ndarray:
        """Get weighted mean position estimate."""
        return self.state_estimate if self.state_estimate is not None else np.array([0, 0])


class KalmanFilter:
    """
    Kalman Filter implementation for single particle tracking.
    Highly effective for particles exhibiting near-Brownian motion.
    Faster than Particle Filter for linear systems with Gaussian noise.
    """
    
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        Initialize Kalman Filter.
        
        Parameters
        ----------
        process_noise : float
            Process noise variance (motion uncertainty)
        measurement_noise : float
            Measurement noise variance (detection uncertainty)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State vector: [x, y, vx, vy]
        self.state = None
        self.covariance = None
        
        # System matrices
        self.F = np.array([  # State transition matrix
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.H = np.array([  # Observation matrix
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
    
    def initialize(self, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None):
        """Initialize filter with position and optional velocity."""
        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0])
        
        self.state = np.array([
            initial_position[0], initial_position[1],
            initial_velocity[0], initial_velocity[1]
        ], dtype=np.float32)
        
        # Initial covariance
        self.covariance = np.eye(4, dtype=np.float32) * 10.0
    
    def predict(self):
        """Prediction step."""
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """Update step with new measurement."""
        # Innovation
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.covariance = (I - K @ self.H) @ self.covariance
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:2]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.state[2:]
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance)."""
        return np.trace(self.covariance[:2, :2])


# JIT-optimized functions for performance-critical operations
@jit(nopython=True, parallel=True)
def calculate_distance_matrix_jit(positions1: np.ndarray, positions2: np.ndarray) -> np.ndarray:
    """
    JIT-optimized distance matrix calculation.
    
    Parameters
    ----------
    positions1 : np.ndarray
        First set of positions (N x 2)
    positions2 : np.ndarray
        Second set of positions (M x 2)
        
    Returns
    -------
    np.ndarray
        Distance matrix (N x M)
    """
    n1, n2 = positions1.shape[0], positions2.shape[0]
    distances = np.zeros((n1, n2), dtype=np.float32)
    
    for i in prange(n1):
        for j in range(n2):
            dx = positions1[i, 0] - positions2[j, 0]
            dy = positions1[i, 1] - positions2[j, 1]
            distances[i, j] = np.sqrt(dx * dx + dy * dy)
    
    return distances


@jit(nopython=True)
def calculate_msd_jit(positions: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-optimized Mean Squared Displacement calculation.
    
    Parameters
    ----------
    positions : np.ndarray
        Track positions (N x 2)
    max_lag : int
        Maximum lag time
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lag times and MSD values
    """
    n_points = positions.shape[0]
    max_lag = min(max_lag, n_points - 1)
    
    lags = np.arange(1, max_lag + 1)
    msd_values = np.zeros(max_lag, dtype=np.float32)
    
    for lag in range(1, max_lag + 1):
        squared_displacements = np.zeros(n_points - lag, dtype=np.float32)
        
        for i in range(n_points - lag):
            dx = positions[i + lag, 0] - positions[i, 0]
            dy = positions[i + lag, 1] - positions[i, 1]
            squared_displacements[i] = dx * dx + dy * dy
        
        msd_values[lag - 1] = np.mean(squared_displacements)
    
    return lags.astype(np.float32), msd_values


@jit(nopython=True)
def calculate_velocities_jit(positions: np.ndarray) -> np.ndarray:
    """
    JIT-optimized velocity calculation.
    
    Parameters
    ----------
    positions : np.ndarray
        Track positions (N x 2)
        
    Returns
    -------
    np.ndarray
        Velocities (N-1,)
    """
    n_points = positions.shape[0]
    if n_points < 2:
        return np.zeros(0, dtype=np.float32)
    
    velocities = np.zeros(n_points - 1, dtype=np.float32)
    
    for i in range(n_points - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        velocities[i] = np.sqrt(dx * dx + dy * dy)
    
    return velocities


def kalman_filter_tracking(detections: Dict[int, pd.DataFrame], 
                          max_search_radius: float = 20.0,
                          min_track_length: int = 3,
                          process_noise: float = 1.0,
                          measurement_noise: float = 1.0) -> pd.DataFrame:
    """
    Kalman Filter-based tracking for near-Brownian motion particles.
    
    Parameters
    ----------
    detections : dict
        Dictionary mapping frame numbers to detection DataFrames
    max_search_radius : float
        Maximum search radius for associations
    min_track_length : int
        Minimum track length to keep
    process_noise : float
        Process noise variance
    measurement_noise : float
        Measurement noise variance
        
    Returns
    -------
    pd.DataFrame
        Linked tracks with Kalman filter estimates
    """
    
    # Sort frames
    frames = sorted(detections.keys())
    
    # Active tracks
    active_tracks = []
    finished_tracks = []
    next_track_id = 1
    
    for frame_idx, frame in enumerate(frames):
        frame_detections = detections[frame]
        
        if frame_detections.empty:
            # Age all tracks
            for track in active_tracks:
                track['missed_frames'] += 1
            continue
        
        detection_positions = frame_detections[['x', 'y']].values
        
        if frame_idx == 0:
            # Initialize tracks from first frame
            for _, detection in frame_detections.iterrows():
                kf = KalmanFilter(process_noise, measurement_noise)
                kf.initialize(np.array([detection['x'], detection['y']]))
                
                track = {
                    'track_id': next_track_id,
                    'kalman_filter': kf,
                    'positions': [np.array([detection['x'], detection['y']])],
                    'frames': [frame],
                    'missed_frames': 0
                }
                active_tracks.append(track)
                next_track_id += 1
        
        else:
            # Predict all active tracks
            for track in active_tracks:
                track['kalman_filter'].predict()
            
            # Get predicted positions
            predicted_positions = np.array([
                track['kalman_filter'].get_position() 
                for track in active_tracks
            ])
            
            # Calculate cost matrix using JIT
            if len(predicted_positions) > 0:
                cost_matrix = calculate_distance_matrix_jit(
                    predicted_positions.astype(np.float32),
                    detection_positions.astype(np.float32)
                )
                
                # Set high cost for distances beyond search radius
                cost_matrix[cost_matrix > max_search_radius] = 1e6
                
                # Solve assignment problem
                if cost_matrix.size > 0:
                    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
                    
                    # Update matched tracks
                    matched_tracks = set()
                    matched_detections = set()
                    
                    for track_idx, det_idx in zip(track_indices, detection_indices):
                        if cost_matrix[track_idx, det_idx] < max_search_radius:
                            # Update Kalman filter
                            active_tracks[track_idx]['kalman_filter'].update(
                                detection_positions[det_idx]
                            )
                            
                            # Store position and frame
                            active_tracks[track_idx]['positions'].append(
                                detection_positions[det_idx]
                            )
                            active_tracks[track_idx]['frames'].append(frame)
                            active_tracks[track_idx]['missed_frames'] = 0
                            
                            matched_tracks.add(track_idx)
                            matched_detections.add(det_idx)
                    
                    # Handle unmatched detections (new tracks)
                    for det_idx in range(len(detection_positions)):
                        if det_idx not in matched_detections:
                            kf = KalmanFilter(process_noise, measurement_noise)
                            kf.initialize(detection_positions[det_idx])
                            
                            track = {
                                'track_id': next_track_id,
                                'kalman_filter': kf,
                                'positions': [detection_positions[det_idx]],
                                'frames': [frame],
                                'missed_frames': 0
                            }
                            active_tracks.append(track)
                            next_track_id += 1
                    
                    # Age unmatched tracks
                    for track_idx in range(len(active_tracks)):
                        if track_idx not in matched_tracks:
                            active_tracks[track_idx]['missed_frames'] += 1
        
        # Remove tracks that have been missed too long
        tracks_to_remove = []
        for i, track in enumerate(active_tracks):
            if track['missed_frames'] > 5:  # Max missed frames
                if len(track['positions']) >= min_track_length:
                    finished_tracks.append(track)
                tracks_to_remove.append(i)
        
        # Remove tracks in reverse order to maintain indices
        for i in reversed(tracks_to_remove):
            active_tracks.pop(i)
    
    # Add remaining active tracks
    for track in active_tracks:
        if len(track['positions']) >= min_track_length:
            finished_tracks.append(track)
    
    # Convert to DataFrame
    track_data = []
    for track in finished_tracks:
        for i, (pos, frame) in enumerate(zip(track['positions'], track['frames'])):
            track_data.append({
                'track_id': track['track_id'],
                'frame': frame,
                'x': pos[0],
                'y': pos[1],
                'point_id': i
            })
    
    return pd.DataFrame(track_data)

def cnn_detect_particles(image: np.ndarray, model=None, threshold: float = 0.5,
                        method: str = "cellpose", model_type: str = "cyto") -> pd.DataFrame:
    """
    CNN-based particle detection using CellSAM or Cellpose segmentation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image for particle detection
    model : object
        Pre-trained model instance (CellSAMSegmentation or CellposeSegmentation)
    threshold : float
        Confidence threshold for detection
    method : str
        Detection method ('cellsam' or 'cellpose')
    model_type : str
        Model type for initialization if model not provided
        
    Returns
    -------
    pd.DataFrame
        Detected particle coordinates and properties
    """
    try:
        from advanced_segmentation import CellSAMSegmentation, CellposeSegmentation
        
        # Initialize model if not provided
        if model is None:
            if method.lower() == "cellsam":
                model = CellSAMSegmentation(model_type=model_type)
                if not model.load_model():
                    return _fallback_blob_detection(image, threshold)
            else:  # cellpose
                model = CellposeSegmentation(model_type=model_type)
                if not model.load_model():
                    return _fallback_blob_detection(image, threshold)
        
        # Run detection
        detections = model.detect_particles(
            image, 
            confidence_threshold=threshold,
            size_filter=(10, 1000)
        )
        
        # Standardize output format
        if not detections.empty:
            # Ensure required columns exist
            required_cols = ['x', 'y', 'confidence']
            for col in required_cols:
                if col not in detections.columns:
                    detections[col] = 0.0
            
            # Add sigma column for compatibility
            if 'area' in detections.columns:
                detections['sigma'] = np.sqrt(detections['area'] / np.pi)
            else:
                detections['sigma'] = 2.0
                
            # Add intensity column if not present
            if 'intensity' not in detections.columns:
                detections['intensity'] = detections.get('mean_intensity', detections['confidence'] * 255)
        
        return detections
        
    except ImportError:
        # Fall back to blob detection if advanced segmentation not available
        return _fallback_blob_detection(image, threshold)
    except Exception as e:
        print(f"CNN detection failed: {e}")
        return _fallback_blob_detection(image, threshold)


def _fallback_blob_detection(image: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Fallback blob detection when advanced methods are unavailable."""
    from skimage.feature import blob_log
    from skimage.filters import gaussian
    
    # Preprocess image
    image_smooth = gaussian(image, sigma=1.0)
    
    # Detect blobs with Laplacian of Gaussian
    blobs = blob_log(
        image_smooth, 
        min_sigma=0.5, 
        max_sigma=3.0, 
        num_sigma=10, 
        threshold=threshold,
        overlap=0.7
    )
    
    if len(blobs) == 0:
        return pd.DataFrame(columns=['x', 'y', 'sigma', 'intensity', 'confidence'])
    
    # Extract properties
    y_coords, x_coords, sigmas = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    
    # Calculate intensities at detected positions
    intensities = []
    for y, x in zip(y_coords, x_coords):
        y_int, x_int = int(round(y)), int(round(x))
        if 0 <= y_int < image.shape[0] and 0 <= x_int < image.shape[1]:
            intensities.append(image[y_int, x_int])
        else:
            intensities.append(0)
    
    # Confidence based on intensity and sigma
    max_intensity = np.max(intensities) if intensities else 1
    confidences = np.array(intensities) / max_intensity
    
    detections = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'sigma': sigmas,
        'intensity': intensities,
        'confidence': confidences
    })
    
    return detections

def particle_filter_tracking(detections: Dict[int, pd.DataFrame], 
                            max_search_radius: float = 20.0,
                            min_track_length: int = 3,
                            n_particles: int = 100,
                            motion_std: float = 5.0,
                            measurement_std: float = 2.0,
                            min_likelihood: float = 1e-6) -> pd.DataFrame:
    """
    Advanced Bayesian tracking using particle filters for robust performance.
    
    Parameters
    ----------
    detections : dict
        Dictionary mapping frame numbers to detection DataFrames
    max_search_radius : float
        Maximum search radius for associations
    min_track_length : int
        Minimum track length to keep
    n_particles : int
        Number of particles per filter
    motion_std : float
        Motion model standard deviation
    measurement_std : float
        Measurement noise standard deviation
    min_likelihood : float
        Minimum likelihood to accept association
        
    Returns
    -------
    pd.DataFrame
        Linked tracks with particle filter estimates
    """
    if not detections:
        return pd.DataFrame()
    
    # Initialize tracking structures
    active_filters = {}  # track_id -> ParticleFilter
    tracks = []
    next_track_id = 0
    
    frames = sorted(detections.keys())
    
    for frame_idx, frame in enumerate(frames):
        current_detections = detections[frame]
        
        if current_detections.empty:
            continue
            
        positions = current_detections[['x', 'y']].values
        
        if frame_idx == 0:
            # Initialize tracks with first frame detections
            for i, pos in enumerate(positions):
                pf = ParticleFilter(n_particles, motion_std, measurement_std)
                pf.initialize(pos)
                active_filters[next_track_id] = pf
                
                # Add to tracks
                tracks.append({
                    'track_id': next_track_id,
                    'frame': frame,
                    'x': pos[0],
                    'y': pos[1],
                    'x_filtered': pos[0],
                    'y_filtered': pos[1]
                })
                next_track_id += 1
        else:
            # Predict step for all active filters
            for pf in active_filters.values():
                pf.predict()
            
            # Get predicted positions
            predicted_positions = []
            track_ids = list(active_filters.keys())
            
            for track_id in track_ids:
                pred_pos = active_filters[track_id].get_state_estimate()
                predicted_positions.append(pred_pos)
            
            if len(predicted_positions) == 0:
                continue
                
            predicted_positions = np.array(predicted_positions)
            
            # Calculate cost matrix
            if len(positions) > 0 and len(predicted_positions) > 0:
                cost_matrix = cdist(predicted_positions, positions)
                
                # Apply distance threshold
                cost_matrix[cost_matrix > max_search_radius] = 1e6
                
                # Solve assignment problem
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Update existing tracks
                used_detections = set()
                updated_tracks = set()
                
                for row_idx, col_idx in zip(row_indices, col_indices):
                    if cost_matrix[row_idx, col_idx] < max_search_radius:
                        track_id = track_ids[row_idx]
                        detection_pos = positions[col_idx]
                        
                        # Update particle filter
                        active_filters[track_id].update(detection_pos)
                        filtered_pos = active_filters[track_id].get_state_estimate()
                        
                        # Add to tracks
                        tracks.append({
                            'track_id': track_id,
                            'frame': frame,
                            'x': detection_pos[0],
                            'y': detection_pos[1],
                            'x_filtered': filtered_pos[0],
                            'y_filtered': filtered_pos[1]
                        })
                        
                        used_detections.add(col_idx)
                        updated_tracks.add(track_id)
                
                # Remove tracks that weren't updated
                tracks_to_remove = []
                for track_id in track_ids:
                    if track_id not in updated_tracks:
                        tracks_to_remove.append(track_id)
                
                for track_id in tracks_to_remove:
                    del active_filters[track_id]
                
                # Start new tracks for unassigned detections
                for det_idx, pos in enumerate(positions):
                    if det_idx not in used_detections:
                        pf = ParticleFilter(n_particles, motion_std, measurement_std)
                        pf.initialize(pos)
                        active_filters[next_track_id] = pf
                        
                        tracks.append({
                            'track_id': next_track_id,
                            'frame': frame,
                            'x': pos[0],
                            'y': pos[1],
                            'x_filtered': pos[0],
                            'y_filtered': pos[1]
                        })
                        next_track_id += 1
    
    if not tracks:
        return pd.DataFrame()
    
    # Convert to DataFrame
    tracks_df = pd.DataFrame(tracks)
    
    # Filter by minimum track length
    track_lengths = tracks_df.groupby('track_id').size()
    valid_tracks = track_lengths[track_lengths >= min_track_length].index
    tracks_df = tracks_df[tracks_df['track_id'].isin(valid_tracks)].copy()
    
    return tracks_df

def bayesian_detection_refinement(detections: pd.DataFrame, 
                                 prior_intensity: float = 0.5,
                                 prior_sigma: float = 2.0) -> pd.DataFrame:
    """
    Refine particle detections using Bayesian inference.
    
    Parameters
    ----------
    detections : pd.DataFrame
        Initial detections to refine
    prior_intensity : float
        Prior belief about particle intensity
    prior_sigma : float
        Prior belief about particle size
        
    Returns
    -------
    pd.DataFrame
        Refined detections with Bayesian updates
    """
    if detections.empty:
        return detections
    
    refined = detections.copy()
    
    # Bayesian update for intensity
    if 'intensity' in detections.columns:
        intensities = detections['intensity'].values
        
        # Simple Bayesian update assuming Gaussian likelihood
        posterior_intensities = (intensities + prior_intensity) / 2
        refined['intensity'] = posterior_intensities
        
        # Update confidence based on posterior
        max_intensity = np.max(posterior_intensities)
        refined['confidence'] = posterior_intensities / max_intensity
    
    # Bayesian update for size (sigma)
    if 'sigma' in detections.columns:
        sigmas = detections['sigma'].values
        
        # Update with prior
        posterior_sigmas = (sigmas + prior_sigma) / 2
        refined['sigma'] = posterior_sigmas
    
    return refined

def machine_learning_cost_function(pos1: np.ndarray, pos2: np.ndarray, 
                                  features1: Dict, features2: Dict,
                                  learned_weights: Optional[np.ndarray] = None) -> float:
    """
    Machine learning-based cost function for particle linking.
    
    Parameters
    ----------
    pos1, pos2 : np.ndarray
        Particle positions
    features1, features2 : dict
        Feature dictionaries for particles
    learned_weights : np.ndarray, optional
        Learned feature weights
        
    Returns
    -------
    float
        Linking cost
    """
    # Distance component
    distance = np.linalg.norm(pos2 - pos1)
    
    # Feature differences
    feature_costs = []
    
    # Intensity difference
    if 'intensity' in features1 and 'intensity' in features2:
        intensity_diff = abs(features1['intensity'] - features2['intensity'])
        feature_costs.append(intensity_diff)
    
    # Size difference
    if 'sigma' in features1 and 'sigma' in features2:
        size_diff = abs(features1['sigma'] - features2['sigma'])
        feature_costs.append(size_diff)
    
    # Default weights if not learned
    if learned_weights is None:
        learned_weights = np.array([1.0] + [0.5] * len(feature_costs))
    
    # Combine costs
    all_costs = [distance] + feature_costs
    total_cost = np.dot(learned_weights[:len(all_costs)], all_costs)
    
    return total_cost

def adaptive_tracking_parameters(tracks_df: pd.DataFrame) -> Dict[str, float]:
    """
    Adaptively estimate optimal tracking parameters from existing tracks.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Existing track data
        
    Returns
    -------
    Dict[str, float]
        Optimized tracking parameters
    """
    if tracks_df.empty:
        return {
            'max_search_radius': 20.0,
            'motion_std': 5.0,
            'measurement_std': 2.0
        }
    
    # Calculate typical displacement between frames
    displacements = []
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        if len(track_data) > 1:
            dx = np.diff(track_data['x'].values)
            dy = np.diff(track_data['y'].values)
            track_displacements = np.sqrt(dx**2 + dy**2)
            displacements.extend(track_displacements)
    
    if displacements:
        mean_displacement = np.mean(displacements)
        std_displacement = np.std(displacements)
        
        # Adaptive parameters
        max_search_radius = mean_displacement + 3 * std_displacement
        motion_std = std_displacement
        measurement_std = std_displacement * 0.5
        
        return {
            'max_search_radius': max(max_search_radius, 5.0),  # Minimum threshold
            'motion_std': max(motion_std, 1.0),
            'measurement_std': max(measurement_std, 0.5)
        }
    
    # Default values
    return {
        'max_search_radius': 20.0,
        'motion_std': 5.0,
        'measurement_std': 2.0
    }