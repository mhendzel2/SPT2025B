"""
Tracking module - provides particle detection and linking functions.
This is a compatibility wrapper for the advanced tracking functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

def detect_particles(image: np.ndarray, 
                    method: str = 'log',
                    particle_size: float = 3.0,
                    threshold_factor: float = 1.5,
                    min_distance: int = 5,
                    **kwargs) -> List[Tuple[float, float]]:
    """
    Detect particles in an image using various methods.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D array)
    method : str
        Detection method: 'log', 'dog', 'intensity', 'enhanced'
    particle_size : float
        Expected particle size in pixels
    threshold_factor : float
        Threshold multiplier for detection sensitivity
    min_distance : int
        Minimum distance between detected particles
    **kwargs : dict
        Additional method-specific parameters
        
    Returns
    -------
    List[Tuple[float, float]]
        List of (x, y) coordinates of detected particles
    """
    try:
        from advanced_tracking import AdvancedParticleDetector
        detector = AdvancedParticleDetector()
        
        # Use appropriate detection method
        if method == 'log':
            sigma = particle_size / 2.0
            threshold = threshold_factor * np.std(image)
            particles_dict = detector.detect_particles_log(image, sigma=sigma, 
                                                          threshold=threshold, 
                                                          min_distance=min_distance)
        elif method == 'dog':
            sigma1 = particle_size / 2.0
            sigma2 = sigma1 * 1.6
            threshold = threshold_factor * np.std(image)
            particles_dict = detector.detect_particles_dog(image, sigma1=sigma1, 
                                                          sigma2=sigma2,
                                                          threshold=threshold,
                                                          min_distance=min_distance)
        elif method == 'enhanced':
            particles_dict = detector.detect_particles_enhanced(image,
                                                               particle_size=particle_size,
                                                               threshold_factor=threshold_factor,
                                                               min_distance=min_distance,
                                                               **kwargs)
        else:
            # Default intensity-based detection
            from skimage import filters, measure
            
            # Threshold image
            threshold = filters.threshold_otsu(image) * threshold_factor
            binary = image > threshold
            
            # Label regions
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled, intensity_image=image)
            
            particles_dict = []
            for region in regions:
                if region.area >= particle_size:
                    y, x = region.centroid
                    particles_dict.append({
                        'x': x,
                        'y': y,
                        'intensity': region.mean_intensity,
                        'area': region.area
                    })
        
        # Convert to list of tuples (x, y)
        if isinstance(particles_dict, list) and len(particles_dict) > 0:
            if isinstance(particles_dict[0], dict):
                particles = [(p['x'], p['y']) for p in particles_dict]
            else:
                particles = particles_dict
        else:
            particles = []
            
        return particles
        
    except Exception as e:
        # Fallback to simple detection if advanced tracking fails
        from skimage import filters, measure
        
        try:
            threshold = filters.threshold_otsu(image) * threshold_factor
        except:
            threshold = np.mean(image) + threshold_factor * np.std(image)
        
        binary = image > threshold
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled, intensity_image=image)
        
        particles = []
        for region in regions:
            if region.area >= particle_size:
                y, x = region.centroid
                particles.append((x, y))
        
        return particles


def link_particles(detections_by_frame: Dict[int, np.ndarray],
                   max_distance: float = 15.0,
                   memory: int = 3,
                   min_track_length: int = 5) -> pd.DataFrame:
    """
    Link detected particles across frames to create tracks.
    
    Parameters
    ----------
    detections_by_frame : Dict[int, np.ndarray]
        Dictionary mapping frame numbers to arrays of particle coordinates
        Each array should be shape (N, 2) or (N, 4) for (x, y) or (x, y, intensity, SNR)
    max_distance : float
        Maximum distance for linking particles between frames
    memory : int
        Number of frames a particle can disappear and still be linked
    min_track_length : int
        Minimum length of tracks to keep
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: track_id, frame, x, y, [intensity, SNR if available]
    """
    try:
        # Try to use trackpy if available
        import trackpy as tp
        
        # Convert detections to DataFrame format expected by trackpy
        all_detections = []
        for frame, coords in detections_by_frame.items():
            if len(coords) == 0:
                continue
                
            if coords.ndim == 1:
                # Single detection
                coords = coords.reshape(1, -1)
            
            for i, coord in enumerate(coords):
                detection = {'frame': frame, 'x': coord[0], 'y': coord[1]}
                if len(coord) > 2:
                    detection['intensity'] = coord[2]
                if len(coord) > 3:
                    detection['SNR'] = coord[3]
                all_detections.append(detection)
        
        if not all_detections:
            return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y'])
        
        detections_df = pd.DataFrame(all_detections)
        
        # Link particles using trackpy
        tracks = tp.link(detections_df, search_range=max_distance, memory=memory)
        tracks.rename(columns={'particle': 'track_id'}, inplace=True)
        
        # Filter by minimum track length
        track_lengths = tracks.groupby('track_id').size()
        valid_tracks = track_lengths[track_lengths >= min_track_length].index
        tracks = tracks[tracks['track_id'].isin(valid_tracks)]
        
        # Reset track IDs to be sequential
        unique_tracks = tracks['track_id'].unique()
        track_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_tracks)}
        tracks['track_id'] = tracks['track_id'].map(track_id_map)
        
        return tracks
        
    except ImportError:
        # Fallback to simple nearest-neighbor linking if trackpy not available
        tracks = []
        track_id = 0
        unlinked_particles = {}  # frame -> list of (x, y, ...) coordinates
        
        # Sort frames
        frames = sorted(detections_by_frame.keys())
        
        # Initialize with first frame
        if len(frames) > 0:
            first_frame = frames[0]
            coords = detections_by_frame[first_frame]
            if len(coords) > 0:
                if coords.ndim == 1:
                    coords = coords.reshape(1, -1)
                for coord in coords:
                    track_data = {
                        'track_id': track_id,
                        'frame': first_frame,
                        'x': coord[0],
                        'y': coord[1]
                    }
                    if len(coord) > 2:
                        track_data['intensity'] = coord[2]
                    if len(coord) > 3:
                        track_data['SNR'] = coord[3]
                    tracks.append(track_data)
                    track_id += 1
        
        # Link subsequent frames
        for i in range(1, len(frames)):
            current_frame = frames[i]
            prev_frame = frames[i-1]
            
            current_coords = detections_by_frame[current_frame]
            if len(current_coords) == 0:
                continue
            if current_coords.ndim == 1:
                current_coords = current_coords.reshape(1, -1)
            
            # Get previous frame tracks
            prev_tracks = [t for t in tracks if t['frame'] == prev_frame]
            prev_coords = np.array([[t['x'], t['y']] for t in prev_tracks])
            
            if len(prev_coords) == 0:
                # Start new tracks
                for coord in current_coords:
                    track_data = {
                        'track_id': track_id,
                        'frame': current_frame,
                        'x': coord[0],
                        'y': coord[1]
                    }
                    if len(coord) > 2:
                        track_data['intensity'] = coord[2]
                    if len(coord) > 3:
                        track_data['SNR'] = coord[3]
                    tracks.append(track_data)
                    track_id += 1
            else:
                # Link to nearest previous particle
                linked = set()
                for j, coord in enumerate(current_coords):
                    distances = np.sqrt(np.sum((prev_coords - coord[:2])**2, axis=1))
                    min_idx = np.argmin(distances)
                    
                    if distances[min_idx] <= max_distance and min_idx not in linked:
                        # Link to existing track
                        track_data = {
                            'track_id': prev_tracks[min_idx]['track_id'],
                            'frame': current_frame,
                            'x': coord[0],
                            'y': coord[1]
                        }
                        if len(coord) > 2:
                            track_data['intensity'] = coord[2]
                        if len(coord) > 3:
                            track_data['SNR'] = coord[3]
                        tracks.append(track_data)
                        linked.add(min_idx)
                    else:
                        # Start new track
                        track_data = {
                            'track_id': track_id,
                            'frame': current_frame,
                            'x': coord[0],
                            'y': coord[1]
                        }
                        if len(coord) > 2:
                            track_data['intensity'] = coord[2]
                        if len(coord) > 3:
                            track_data['SNR'] = coord[3]
                        tracks.append(track_data)
                        track_id += 1
        
        if not tracks:
            return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y'])
        
        tracks_df = pd.DataFrame(tracks)
        
        # Filter by minimum track length
        track_lengths = tracks_df.groupby('track_id').size()
        valid_tracks = track_lengths[track_lengths >= min_track_length].index
        tracks_df = tracks_df[tracks_df['track_id'].isin(valid_tracks)]
        
        return tracks_df
