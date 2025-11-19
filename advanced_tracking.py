"""
Advanced Tracking Module for SPT Analysis Application.
Provides sophisticated particle detection and linking algorithms.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import feature, measure, filters
from sklearn.neighbors import KDTree
from typing import Dict, List, Tuple, Optional, Any
import warnings
from track import run_btrack
from filter import filter_tracks_by_length
warnings.filterwarnings('ignore')

class BtrackTracker:
    """
    Wrapper for the btrack tracker.
    """
    def __init__(self, config_path: str = "models/cell_config.json"):
        self.config_path = config_path

    def track(self, detections: pd.DataFrame, min_track_length: int = 5) -> pd.DataFrame:
        """
        Track particles using btrack.

        Args:
            detections: DataFrame with columns 'x', 'y', 'z', 't', and 'label'.
            min_track_length: The minimum length of a track to be kept.

        Returns:
            A DataFrame with the tracked particles.
        """
        tracks, properties, graph = run_btrack(detections, self.config_path)

        # Convert tracks to a DataFrame
        track_data = []
        for tracklet in tracks:
            for particle in tracklet:
                track_data.append({
                    'track_id': tracklet.ID,
                    'frame': particle.t,
                    'x': particle.x,
                    'y': particle.y,
                    'z': particle.z,
                })

        if not track_data:
            return pd.DataFrame()

        tracks_df = pd.DataFrame(track_data)

        # Filter tracks by length
        tracks_df = filter_tracks_by_length(tracks_df, min_track_length)

        return tracks_df

class AdvancedParticleDetector:
    """
    Advanced particle detection using multiple algorithms.
    """
    
    def __init__(self):
        self.detection_results = {}
    
    def detect_particles_log(self, image: np.ndarray, sigma: float = 1.0, 
                            threshold: float = 0.01, min_distance: int = 5) -> List[Dict[str, Any]]:
        """
        Detect particles using Laplacian of Gaussian (LoG) method.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        sigma : float
            Standard deviation for Gaussian kernel
        threshold : float
            Detection threshold
        min_distance : int
            Minimum distance between detections
            
        Returns
        -------
        List[Dict[str, Any]]
            List of detected particles with coordinates and properties
        """
        # Apply LoG filter
        log_filtered = -filters.laplacian(filters.gaussian(image, sigma))
        
        # Find local maxima
        coordinates = feature.peak_local_max(
            log_filtered, 
            min_distance=min_distance,
            threshold_abs=threshold
        )
        
        particles = []
        for i, (y, x) in enumerate(coordinates):
            # Calculate intensity and other properties
            intensity = float(image[y, x])
            log_response = float(log_filtered[y, x])
            
            # Estimate particle size using second moment
            roi_size = max(3, int(3 * sigma))
            y1, y2 = max(0, y - roi_size), min(image.shape[0], y + roi_size + 1)
            x1, x2 = max(0, x - roi_size), min(image.shape[1], x + roi_size + 1)
            
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                estimated_size = self._estimate_particle_size(roi)
            else:
                estimated_size = sigma
            
            particles.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'intensity': intensity,
                'log_response': log_response,
                'estimated_size': estimated_size,
                'detection_method': 'LoG'
            })
        
        return particles
    
    def detect_particles_dog(self, image: np.ndarray, sigma1: float = 1.0, 
                            sigma2: float = 1.6, threshold: float = 0.01,
                            min_distance: int = 5) -> List[Dict[str, Any]]:
        """
        Detect particles using Difference of Gaussians (DoG) method.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        sigma1, sigma2 : float
            Standard deviations for Gaussian kernels
        threshold : float
            Detection threshold
        min_distance : int
            Minimum distance between detections
            
        Returns
        -------
        List[Dict[str, Any]]
            List of detected particles
        """
        # Apply DoG filter
        gaussian1 = filters.gaussian(image, sigma1)
        gaussian2 = filters.gaussian(image, sigma2)
        dog_filtered = gaussian1 - gaussian2
        
        # Find local maxima
        coordinates = feature.peak_local_maxima(
            dog_filtered,
            min_distance=min_distance,
            threshold_abs=threshold
        )
        
        particles = []
        for i, (y, x) in enumerate(zip(coordinates[0], coordinates[1])):
            intensity = float(image[y, x])
            dog_response = float(dog_filtered[y, x])
            
            # Estimate particle size
            roi_size = max(3, int(3 * max(sigma1, sigma2)))
            y1, y2 = max(0, y - roi_size), min(image.shape[0], y + roi_size + 1)
            x1, x2 = max(0, x - roi_size), min(image.shape[1], x + roi_size + 1)
            
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                estimated_size = self._estimate_particle_size(roi)
            else:
                estimated_size = (sigma1 + sigma2) / 2
            
            particles.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'intensity': intensity,
                'dog_response': dog_response,
                'estimated_size': estimated_size,
                'detection_method': 'DoG'
            })
        
        return particles
    
    def detect_particles_enhanced(self, image: np.ndarray, 
                                 adaptive_threshold: bool = True,
                                 multi_scale: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced particle detection using multiple techniques.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        adaptive_threshold : bool
            Whether to use adaptive thresholding
        multi_scale : bool
            Whether to use multi-scale detection
            
        Returns
        -------
        List[Dict[str, Any]]
            List of detected particles
        """
        particles = []
        
        if multi_scale:
            # Multi-scale detection
            scales = [0.8, 1.0, 1.2, 1.5]
            all_detections = []
            
            for scale in scales:
                log_particles = self.detect_particles_log(image, sigma=scale)
                for particle in log_particles:
                    particle['detection_scale'] = scale
                all_detections.extend(log_particles)
            
            # Merge nearby detections
            particles = self._merge_detections(all_detections)
        else:
            particles = self.detect_particles_log(image)
        
        # Apply adaptive thresholding if requested
        if adaptive_threshold:
            particles = self._adaptive_threshold_filter(particles, image)
        
        return particles
    
    def _estimate_particle_size(self, roi: np.ndarray) -> float:
        """Estimate particle size from ROI using second moments."""
        if roi.size == 0:
            return 1.0
        
        # Calculate center of mass
        y_indices, x_indices = np.indices(roi.shape)
        total_intensity = np.sum(roi)
        
        if total_intensity == 0:
            return 1.0
        
        y_center = np.sum(y_indices * roi) / total_intensity
        x_center = np.sum(x_indices * roi) / total_intensity
        
        # Calculate second moments
        sigma_y = np.sqrt(np.sum((y_indices - y_center)**2 * roi) / total_intensity)
        sigma_x = np.sqrt(np.sum((x_indices - x_center)**2 * roi) / total_intensity)
        
        return float(np.mean([sigma_x, sigma_y]))
    
    def _merge_detections(self, detections: List[Dict], merge_distance: float = 3.0) -> List[Dict]:
        """Merge nearby detections from multi-scale analysis."""
        if not detections:
            return []
        
        # Convert to array for efficient processing
        coords = np.array([[d['x'], d['y']] for d in detections])
        
        # Build KD-tree for efficient neighbor search
        tree = KDTree(coords)
        
        merged = []
        used_indices = set()
        
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
            
            # Find nearby detections
            neighbors = tree.query_radius([[detection['x'], detection['y']]], r=merge_distance)[0]
            
            # Merge detections
            nearby_detections = [detections[j] for j in neighbors if j not in used_indices]
            
            if nearby_detections:
                # Weight by intensity for merging
                total_weight = sum(d['intensity'] for d in nearby_detections)
                
                if total_weight > 0:
                    merged_x = sum(d['x'] * d['intensity'] for d in nearby_detections) / total_weight
                    merged_y = sum(d['y'] * d['intensity'] for d in nearby_detections) / total_weight
                    max_intensity = max(d['intensity'] for d in nearby_detections)
                    avg_size = np.mean([d['estimated_size'] for d in nearby_detections])
                    
                    merged.append({
                        'id': len(merged),
                        'x': float(merged_x),
                        'y': float(merged_y),
                        'intensity': float(max_intensity),
                        'estimated_size': float(avg_size),
                        'detection_method': 'enhanced_merged',
                        'num_merged': len(nearby_detections)
                    })
                
                # Mark as used
                for j in neighbors:
                    used_indices.add(j)
        
        return merged
    
    def _adaptive_threshold_filter(self, particles: List[Dict], image: np.ndarray) -> List[Dict]:
        """Filter particles using adaptive thresholding."""
        if not particles:
            return particles
        
        # Calculate local background for each particle
        filtered_particles = []
        
        for particle in particles:
            x, y = int(particle['x']), int(particle['y'])
            
            # Define local region
            roi_size = 10
            y1, y2 = max(0, y - roi_size), min(image.shape[0], y + roi_size + 1)
            x1, x2 = max(0, x - roi_size), min(image.shape[1], x + roi_size + 1)
            
            local_region = image[y1:y2, x1:x2]
            
            if local_region.size > 0:
                local_bg = np.percentile(local_region, 10)  # 10th percentile as background
                signal_to_background = particle['intensity'] / local_bg if local_bg > 0 else 0
                
                # Keep particles with good signal-to-background ratio
                if signal_to_background > 1.5:
                    particle['signal_to_background'] = float(signal_to_background)
                    particle['local_background'] = float(local_bg)
                    filtered_particles.append(particle)
        
        return filtered_particles


class AdvancedParticleLinker:
    """
    Advanced particle linking using global optimization.
    """
    
    def __init__(self, max_distance: float = 10.0, max_gap_frames: int = 2):
        self.max_distance = max_distance
        self.max_gap_frames = max_gap_frames
        self.linking_results = {}
    
    def link_particles_global(self, detections_by_frame: Dict[int, List[Dict]]) -> pd.DataFrame:
        """
        Link particles across frames using global optimization.
        
        Parameters
        ----------
        detections_by_frame : Dict[int, List[Dict]]
            Dictionary mapping frame numbers to lists of particle detections
            
        Returns
        -------
        pd.DataFrame
            Track data with linked particles
        """
        if not detections_by_frame:
            return pd.DataFrame()
        
        # Sort frames
        frames = sorted(detections_by_frame.keys())
        
        if len(frames) < 2:
            # Single frame - convert to track format
            tracks = []
            frame = frames[0]
            for i, particle in enumerate(detections_by_frame[frame]):
                tracks.append({
                    'track_id': i,
                    'frame': frame,
                    'x': particle['x'],
                    'y': particle['y'],
                    'intensity': particle.get('intensity', 0),
                    'particle_id': particle.get('id', i)
                })
            return pd.DataFrame(tracks)
        
        # Initialize tracks with first frame
        tracks = {}
        track_id_counter = 0
        
        for particle in detections_by_frame[frames[0]]:
            tracks[track_id_counter] = {
                'track_id': track_id_counter,
                'particles': [particle],
                'frames': [frames[0]],
                'last_frame': frames[0],
                'active': True
            }
            track_id_counter += 1
        
        # Link subsequent frames
        for frame_idx in range(1, len(frames)):
            current_frame = frames[frame_idx]
            current_particles = detections_by_frame[current_frame]
            
            if not current_particles:
                continue
            
            # Find best assignments using Hungarian algorithm approximation
            assignments = self._find_best_assignments(tracks, current_particles, current_frame)
            
            # Update tracks based on assignments
            self._update_tracks(tracks, current_particles, assignments, current_frame)
            
            # Create new tracks for unassigned particles
            for i, particle in enumerate(current_particles):
                if i not in assignments.values():
                    tracks[track_id_counter] = {
                        'track_id': track_id_counter,
                        'particles': [particle],
                        'frames': [current_frame],
                        'last_frame': current_frame,
                        'active': True
                    }
                    track_id_counter += 1
        
        # Convert to DataFrame
        track_data = []
        for track in tracks.values():
            for i, (particle, frame) in enumerate(zip(track['particles'], track['frames'])):
                track_data.append({
                    'track_id': track['track_id'],
                    'frame': frame,
                    'x': particle['x'],
                    'y': particle['y'],
                    'intensity': particle.get('intensity', 0),
                    'particle_id': particle.get('id', i),
                    'detection_method': particle.get('detection_method', 'unknown')
                })
        
        return pd.DataFrame(track_data)
    
    def _find_best_assignments(self, tracks: Dict, particles: List[Dict], frame: int) -> Dict[int, int]:
        """Find best track-particle assignments using distance-based cost."""
        active_tracks = {tid: track for tid, track in tracks.items() 
                        if track['active'] and (frame - track['last_frame']) <= self.max_gap_frames}
        
        if not active_tracks or not particles:
            return {}
        
        # Calculate cost matrix
        track_ids = list(active_tracks.keys())
        costs = np.full((len(track_ids), len(particles)), np.inf)
        
        for i, track_id in enumerate(track_ids):
            track = active_tracks[track_id]
            last_particle = track['particles'][-1]
            
            for j, particle in enumerate(particles):
                distance = np.sqrt((particle['x'] - last_particle['x'])**2 + 
                                 (particle['y'] - last_particle['y'])**2)
                
                if distance <= self.max_distance:
                    # Add penalty for frame gaps
                    gap_penalty = (frame - track['last_frame'] - 1) * 0.5
                    costs[i, j] = distance + gap_penalty
        
        # Simple greedy assignment (for efficiency)
        assignments = {}
        used_particles = set()
        
        # Sort by cost
        cost_indices = []
        for i in range(costs.shape[0]):
            for j in range(costs.shape[1]):
                if costs[i, j] < np.inf:
                    cost_indices.append((costs[i, j], i, j))
        
        cost_indices.sort()
        
        for cost, i, j in cost_indices:
            track_id = track_ids[i]
            if track_id not in assignments and j not in used_particles:
                assignments[track_id] = j
                used_particles.add(j)
        
        return assignments
    
    def _update_tracks(self, tracks: Dict, particles: List[Dict], 
                      assignments: Dict[int, int], frame: int):
        """Update tracks with new assignments."""
        # Deactivate tracks that weren't assigned
        for track_id, track in tracks.items():
            if track['active'] and track_id not in assignments:
                gap = frame - track['last_frame']
                if gap > self.max_gap_frames:
                    track['active'] = False
        
        # Update assigned tracks
        for track_id, particle_idx in assignments.items():
            if track_id in tracks:
                track = tracks[track_id]
                track['particles'].append(particles[particle_idx])
                track['frames'].append(frame)
                track['last_frame'] = frame
                track['active'] = True


class TrackingQualityAssessment:
    """
    Assess and improve tracking quality.
    """
    
    def __init__(self):
        self.quality_metrics = {}
    
    def assess_track_quality(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall tracking quality and provide improvement suggestions.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
            
        Returns
        -------
        Dict[str, Any]
            Quality assessment results
        """
        if tracks_df.empty:
            return {'success': False, 'error': 'No track data provided'}
        
        quality_metrics = {}
        
        # Track length distribution
        track_lengths = tracks_df.groupby('track_id').size()
        quality_metrics['track_lengths'] = {
            'mean': float(track_lengths.mean()),
            'median': float(track_lengths.median()),
            'min': int(track_lengths.min()),
            'max': int(track_lengths.max()),
            'std': float(track_lengths.std())
        }
        
        # Temporal continuity
        gaps = self._assess_temporal_gaps(tracks_df)
        quality_metrics['temporal_continuity'] = gaps
        
        # Spatial consistency
        spatial_quality = self._assess_spatial_consistency(tracks_df)
        quality_metrics['spatial_consistency'] = spatial_quality
        
        # Detection density
        frames = tracks_df['frame'].unique()
        particles_per_frame = tracks_df.groupby('frame').size()
        quality_metrics['detection_density'] = {
            'mean_particles_per_frame': float(particles_per_frame.mean()),
            'frames_analyzed': len(frames),
            'total_detections': len(tracks_df)
        }
        
        # Overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        quality_metrics['overall_quality_score'] = quality_score
        
        # Improvement suggestions
        suggestions = self._generate_improvement_suggestions(quality_metrics)
        quality_metrics['improvement_suggestions'] = suggestions
        
        return {
            'success': True,
            'quality_metrics': quality_metrics
        }
    
    def _assess_temporal_gaps(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess temporal gaps in tracks."""
        gap_info = {'tracks_with_gaps': 0, 'total_gaps': 0, 'gap_sizes': []}
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            frames = track_data['frame'].values
            
            if len(frames) > 1:
                frame_diffs = np.diff(frames)
                gaps = frame_diffs[frame_diffs > 1]
                
                if len(gaps) > 0:
                    gap_info['tracks_with_gaps'] += 1
                    gap_info['total_gaps'] += len(gaps)
                    gap_info['gap_sizes'].extend((gaps - 1).tolist())
        
        if gap_info['gap_sizes']:
            gap_info['mean_gap_size'] = float(np.mean(gap_info['gap_sizes']))
            gap_info['max_gap_size'] = int(np.max(gap_info['gap_sizes']))
        else:
            gap_info['mean_gap_size'] = 0
            gap_info['max_gap_size'] = 0
        
        return gap_info
    
    def _assess_spatial_consistency(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess spatial consistency of tracks."""
        spatial_metrics = {'erratic_tracks': 0, 'mean_step_size_variation': 0}
        
        step_size_variations = []
        
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) < 3:
                continue
            
            x = track_data['x'].values
            y = track_data['y'].values
            
            # Calculate step sizes
            step_sizes = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            
            if len(step_sizes) > 1:
                # Coefficient of variation for step sizes
                cv = np.std(step_sizes) / np.mean(step_sizes) if np.mean(step_sizes) > 0 else 0
                step_size_variations.append(cv)
                
                # Flag erratic tracks (high variation in step sizes)
                if cv > 2.0:
                    spatial_metrics['erratic_tracks'] += 1
        
        if step_size_variations:
            spatial_metrics['mean_step_size_variation'] = float(np.mean(step_size_variations))
        
        return spatial_metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Penalize short tracks
        mean_length = metrics['track_lengths']['mean']
        if mean_length < 10:
            score -= (10 - mean_length) * 3
        
        # Penalize temporal gaps
        gap_fraction = metrics['temporal_continuity']['tracks_with_gaps'] / max(1, len(metrics.get('track_lengths', {}).get('mean', 1)))
        score -= gap_fraction * 20
        
        # Penalize spatial inconsistency
        erratic_fraction = metrics['spatial_consistency']['erratic_tracks'] / max(1, len(metrics.get('track_lengths', {}).get('mean', 1)))
        score -= erratic_fraction * 15
        
        return max(0.0, min(100.0, score))
    
    def _generate_improvement_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving tracking quality."""
        suggestions = []
        
        mean_length = metrics['track_lengths']['mean']
        if mean_length < 10:
            suggestions.append("Consider increasing maximum linking distance or gap frames to improve track continuity")
        
        gap_fraction = metrics['temporal_continuity']['tracks_with_gaps'] / max(1, metrics['temporal_continuity']['total_gaps'])
        if gap_fraction > 0.3:
            suggestions.append("High number of temporal gaps detected - check detection parameters")
        
        if metrics['spatial_consistency']['erratic_tracks'] > 0:
            suggestions.append("Some erratic tracks detected - consider improving detection quality or adjusting linking parameters")
        
        if metrics['detection_density']['mean_particles_per_frame'] < 5:
            suggestions.append("Low detection density - consider adjusting detection thresholds")
        
        quality_score = metrics['overall_quality_score']
        if quality_score < 60:
            suggestions.append("Overall tracking quality is low - consider reviewing detection and linking parameters")
        elif quality_score > 85:
            suggestions.append("Good tracking quality achieved!")
        
        return suggestions

# Add the missing ParticleFilter class needed by the functions below

class ParticleFilter:
    """
    Particle filter for tracking with state estimation.
    """
    
    def __init__(self, n_particles: int = 100, motion_std: float = 5.0, measurement_std: float = 2.0):
        self.n_particles = n_particles
        self.motion_std = motion_std  # Standard deviation of motion model
        self.measurement_std = measurement_std  # Standard deviation of measurement model
        self.particles = None
        self.weights = None
        
    def initialize(self, initial_position: np.ndarray):
        """Initialize filter with particles around the initial position."""
        self.particles = initial_position + self.motion_std * np.random.randn(self.n_particles, 2)
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def predict(self):
        """Predict step - apply motion model to particles."""
        self.particles += self.motion_std * np.random.randn(self.n_particles, 2)
        
    def update(self, measurement: np.ndarray):
        """Update step - weight particles based on measurement."""
        # Calculate likelihood for each particle
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        likelihoods = np.exp(-0.5 * (distances**2) / (self.measurement_std**2))
        
        # Update weights
        self.weights *= likelihoods
        
        # Normalize weights
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            # All weights are zero, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
            
        # Resample particles
        self._resample()
        
    def get_state_estimate(self) -> np.ndarray:
        """Get estimated state (position)."""
        if self.particles is None:
            return np.zeros(2)
        
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def _resample(self):
        """Resample particles based on weights."""
        # Systematic resampling
        cumsum = np.cumsum(self.weights)
        step = 1.0 / self.n_particles
        i = 0
        u = np.random.uniform(0, step)
        
        new_particles = np.zeros_like(self.particles)
        
        for j in range(self.n_particles):
            while u > cumsum[i]:
                i += 1
                if i >= len(cumsum):
                    i = len(cumsum) - 1
                    break
            new_particles[j] = self.particles[i]
            u += step
            
        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

class AdvancedTracking:
    """
    Advanced tracking methods for linking particles across frames.
    """
    
    def __init__(self):
        pass
    def track_particles_btrack(self, detections: pd.DataFrame,
                                 min_track_length: int = 5) -> pd.DataFrame:
        """
        Track particles using btrack.

        Parameters
        ----------
        detections : pd.DataFrame
            DataFrame with columns 'x', 'y', 'z', 't', and 'label'.
        min_track_length : int
            Minimum track length to keep

        Returns
        -------
        pd.DataFrame
            Track data with filtered trajectories
        """
        btrack_tracker = BtrackTracker()
        return btrack_tracker.track(detections, min_track_length)
        
    def track_particles(self, detections: Dict[int, pd.DataFrame], 
                        max_search_radius: float = 20.0,
                        motion_std: float = 5.0, 
                        measurement_std: float = 2.0,
                        min_track_length: int = 5,
                        n_particles: int = 100) -> pd.DataFrame:
        """
        Track particles using particle filters.
        
        Parameters
        ----------
        detections : Dict[int, pd.DataFrame]
            Dictionary mapping frame numbers to particle detections
        max_search_radius : float
            Maximum search radius for particle linking
        motion_std : float
            Standard deviation for motion model
        measurement_std : float
            Standard deviation for measurement model
        min_track_length : int
            Minimum track length to keep
        n_particles : int
            Number of particles for particle filter
            
        Returns
        -------
        pd.DataFrame
            Track data with filtered trajectories
        """
        if not detections:
            return pd.DataFrame()
            
        # Initialize tracking
        active_filters = {}  # track_id -> particle filter
        tracks = []
        next_track_id = 0
        
        # Process frames in order
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
                    from scipy.spatial.distance import cdist
                    from scipy.optimize import linear_sum_assignment
                    
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


# ============================================================================
# AI-Enhanced Trajectory Linking (SPTnet / Transformer Approach)
# Based on: bioRxiv 2025 - Deep learning for end-to-end SPT
# ============================================================================

class DeepTrackLinker:
    """
    Transformer-based trajectory linking for dense nuclear environments.
    
    This class implements a momentum-based trajectory prediction method inspired
    by recent Transformer architectures (SPTnet, bioRxiv Feb 2025). Unlike 
    traditional frame-to-frame linking (Hungarian/Nearest Neighbor), this 
    approach analyzes the *entire* trajectory context to predict future positions.
    
    Standard algorithms often fail in dense chromatin environments where particles
    create "cluttered" motion patterns. This context-aware approach is particularly
    effective for:
    - High particle density (chromatin crowding)
    - Transient binding events (transcription factors)
    - Heterogeneous nuclear compartments (condensates, nucleoli)
    
    Parameters
    ----------
    history_len : int, default=5
        Number of historical frames to use for trajectory prediction.
        Longer histories capture persistent motion but may be less adaptive.
    damping_factor : float, default=0.8
        Velocity damping for chromatin density effects. Lower values (0.6-0.8)
        account for increased viscosity in crowded nuclear environments.
    use_acceleration : bool, default=True
        Whether to include acceleration terms in prediction (second derivative).
    
    References
    ----------
    - SPTnet: a deep learning framework for end-to-end single-particle tracking
      (bioRxiv, Feb 2025)
    - Transformer architectures for temporal sequence prediction
    
    Notes
    -----
    This is a heuristic implementation using momentum vectors as a proxy for
    full Transformer inference. A complete deployment would require:
    1. Pre-trained model weights (ONNX/PyTorch format)
    2. Feature extraction layer (position embeddings)
    3. Multi-head attention mechanism
    4. Output decoder for position prediction
    
    The current implementation provides robust linking performance while
    maintaining computational efficiency for real-time analysis.
    
    Examples
    --------
    >>> linker = DeepTrackLinker(history_len=5, damping_factor=0.75)
    >>> current_tracks = [
    ...     [(0, 0), (1, 1), (2, 2), (3, 3)],  # Linear motion
    ...     [(0, 5), (0.5, 4), (1, 3.5)]       # Curved trajectory
    ... ]
    >>> predictions = linker.predict_next_positions(current_tracks)
    >>> print(predictions)  # Predicted (x, y) for each track
    """
    
    def __init__(self, history_len: int = 5, damping_factor: float = 0.8, 
                 use_acceleration: bool = True):
        self.history_len = history_len
        self.damping_factor = damping_factor
        self.use_acceleration = use_acceleration
        
    def extract_track_features(self, tracks: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Convert track history into vector embeddings for prediction.
        
        In a full Transformer implementation, this would:
        1. Embed positions into high-dimensional space
        2. Add positional encoding (time information)
        3. Feed through multi-head attention layers
        4. Generate trajectory embeddings
        
        This heuristic version calculates momentum vectors (velocity + acceleration)
        as a computationally efficient proxy.
        
        Parameters
        ----------
        tracks : List[List[Tuple[float, float]]]
            List of tracks, where each track is a list of (x, y) coordinates.
        
        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_tracks, 2) for velocity or (n_tracks, 4)
            if acceleration is included. Features are [dx, dy] or [dx, dy, ddx, ddy].
        """
        vectors = []
        
        for track in tracks:
            if len(track) < 2:
                # Insufficient history - predict no motion
                if self.use_acceleration:
                    vectors.append([0, 0, 0, 0])
                else:
                    vectors.append([0, 0])
                continue
            
            # Extract recent trajectory segment
            recent = track[-min(len(track), self.history_len):]
            recent_array = np.array(recent)
            
            # Calculate velocity (first derivative)
            dx = recent_array[-1, 0] - recent_array[0, 0]
            dy = recent_array[-1, 1] - recent_array[0, 1]
            dt = len(recent) - 1
            vx = dx / dt if dt > 0 else 0
            vy = dy / dt if dt > 0 else 0
            
            if self.use_acceleration and len(recent) >= 3:
                # Calculate acceleration (second derivative)
                # Compare recent velocity to earlier velocity
                mid_idx = len(recent) // 2
                dx_early = recent_array[mid_idx, 0] - recent_array[0, 0]
                dy_early = recent_array[mid_idx, 1] - recent_array[0, 1]
                dt_early = mid_idx
                vx_early = dx_early / dt_early if dt_early > 0 else 0
                vy_early = dy_early / dt_early if dt_early > 0 else 0
                
                # Acceleration = change in velocity
                ax = (vx - vx_early) / dt_early if dt_early > 0 else 0
                ay = (vy - vy_early) / dt_early if dt_early > 0 else 0
                
                vectors.append([vx, vy, ax, ay])
            else:
                vectors.append([vx, vy])
        
        return np.array(vectors)
    
    def predict_next_positions(self, current_tracks: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Predict next positions using trajectory context (Transformer proxy).
        
        This method implements a simplified version of attention-based prediction:
        1. Extract features from track history (velocity + acceleration)
        2. Apply damping to account for chromatin viscosity
        3. Extrapolate future position from current position + predicted displacement
        
        The damping factor is crucial for nuclear environments where particles
        experience increased drag due to chromatin crowding and molecular interactions.
        
        Parameters
        ----------
        current_tracks : List[List[Tuple[float, float]]]
            List of active tracks with their position histories.
        
        Returns
        -------
        np.ndarray
            Predicted (x, y) positions for each track at the next time step.
            Shape: (n_tracks, 2)
        
        Notes
        -----
        Prediction equation:
            x_pred = x_current + vx * damping + 0.5 * ax * damping²  (if acceleration enabled)
            x_pred = x_current + vx * damping                         (velocity only)
        
        The quadratic acceleration term follows the kinematic equation:
            Δx = v*t + 0.5*a*t²
        where t=1 frame and damping acts as a viscosity correction.
        """
        # Extract current positions (last point of each track)
        current_pos = np.array([track[-1] for track in current_tracks])
        
        # Extract trajectory features (velocity + optional acceleration)
        features = self.extract_track_features(current_tracks)
        
        if features.shape[1] == 4:  # Velocity + acceleration
            vx, vy, ax, ay = features[:, 0], features[:, 1], features[:, 2], features[:, 3]
            # Kinematic prediction: x + v*t + 0.5*a*t²
            predicted_dx = vx * self.damping_factor + 0.5 * ax * self.damping_factor**2
            predicted_dy = vy * self.damping_factor + 0.5 * ay * self.damping_factor**2
        else:  # Velocity only
            vx, vy = features[:, 0], features[:, 1]
            predicted_dx = vx * self.damping_factor
            predicted_dy = vy * self.damping_factor
        
        predicted_pos = current_pos + np.column_stack([predicted_dx, predicted_dy])
        
        return predicted_pos
    
    def link_frame_to_frame(self, prev_frame_positions: np.ndarray, 
                           current_frame_positions: np.ndarray,
                           active_tracks: List[List[Tuple[float, float]]],
                           max_distance: float = 5.0) -> List[Tuple[int, int]]:
        """
        Link detections between consecutive frames using predicted positions.
        
        This method replaces standard Hungarian algorithm linking with a
        context-aware approach:
        1. Predict where each active track should be in the current frame
        2. Match current detections to predictions (not just previous positions)
        3. Use predicted positions as priors in cost matrix calculation
        
        Parameters
        ----------
        prev_frame_positions : np.ndarray
            Positions from previous frame (n_prev, 2).
        current_frame_positions : np.ndarray
            Detected positions in current frame (n_current, 2).
        active_tracks : List[List[Tuple[float, float]]]
            Full trajectory history for each active track.
        max_distance : float, default=5.0
            Maximum linking distance (pixels). Pairs beyond this are rejected.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (prev_idx, current_idx) pairs representing links.
            Unlinked detections can start new tracks.
        
        Notes
        -----
        This method provides better linking in crowded environments by considering
        trajectory momentum rather than just proximity. Particles moving in
        consistent directions are correctly linked even when closer particles exist.
        """
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        
        if len(prev_frame_positions) == 0 or len(current_frame_positions) == 0:
            return []
        
        # Predict next positions using trajectory context
        predicted_positions = self.predict_next_positions(active_tracks)
        
        # Calculate cost matrix: distance from predictions to current detections
        # This differs from standard linking which uses prev_frame_positions
        cost_matrix = cdist(predicted_positions, current_frame_positions)
        
        # Apply distance threshold - set costs beyond max_distance to infinity
        cost_matrix[cost_matrix > max_distance] = 1e10
        
        # Solve assignment problem (Hungarian algorithm on predicted positions)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments that exceed max_distance
        valid_links = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e10:
                valid_links.append((r, c))
        
        return valid_links
