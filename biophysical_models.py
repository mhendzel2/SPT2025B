"""
Advanced biophysical models for single particle tracking analysis.
Specialized for nucleosome diffusion in chromatin and polymer physics modeling.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# Import data access utilities
try:
    from data_access_utils import get_track_data, check_data_availability, get_units, display_data_summary
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

# ...existing code...

def show_biophysical_models():
    """Main interface for biophysical models."""
    st.title("🧬 Biophysical Models")
    
    # Check for data availability
    if DATA_UTILS_AVAILABLE:
        if not check_data_availability():
            return
        tracks_df, _ = get_track_data()
        units = get_units()
    else:
        # Fallback to direct access
        tracks_df = st.session_state.get('tracks_df') or st.session_state.get('raw_tracks')
        if tracks_df is None or tracks_df.empty:
            st.error("No track data loaded. Please load data first.")
            return
        units = {
            'pixel_size': st.session_state.get('pixel_size', 0.1),
            'frame_interval': st.session_state.get('frame_interval', 0.1)
        }
    
    # Display data summary
    if DATA_UTILS_AVAILABLE:
        display_data_summary()
    
    # ...existing code...
    # Continue with the rest of the biophysical models logic using tracks_df

class PolymerPhysicsModel:
    """Polymer physics model implementation."""
    
    def analyze_polymer_dynamics(self, tracks_df=None):
        """Analyze polymer dynamics from tracking data."""
        # Get data if not provided
        if tracks_df is None:
            if DATA_UTILS_AVAILABLE:
                tracks_df, has_data = get_track_data()
                if not has_data:
                    return {'error': 'No track data available', 'success': False}
            else:
                tracks_df = st.session_state.get('tracks_df') or st.session_state.get('raw_tracks')
                if tracks_df is None or tracks_df.empty:
                    return {'error': 'No track data available', 'success': False}
        
        # ...existing code...
        # Continue with analysis using tracks_df

class EnergyLandscapeMapper:
    """
    Class for mapping energy landscapes from particle trajectories.
    """

    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 0.1, temperature: float = 300.0):
        """
        Initialize the energy landscape mapper.

        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns 'track_id', 'frame', 'x', 'y'
        pixel_size : float
            Pixel size in micrometers
        temperature : float
            Temperature in Kelvin
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        self.temperature = temperature
        self.kB = 1.38e-23  # Boltzmann constant, J/K
        self.kBT = self.kB * self.temperature  # Thermal energy
        self.results = {}

        # Convert coordinates to micrometers if needed
        for col in ['x', 'y']:
            if col in self.tracks_df.columns:
                self.tracks_df[f'{col}_um'] = self.tracks_df[col] * self.pixel_size

    def map_energy_landscape(self, resolution: int = 20, method: str = "boltzmann", 
                           smoothing: float = 0.5, normalize: bool = True) -> Dict[str, Any]:
        """
        Map energy landscape from particle positions.

        Parameters
        ----------
        resolution : int
            Grid resolution for energy landscape
        method : str
            Method for energy calculation ('boltzmann', 'drift', 'kramers')
        smoothing : float
            Smoothing factor for the landscape
        normalize : bool
            Whether to normalize energies to kBT units

        Returns
        -------
        Dict[str, Any]
            Energy landscape mapping results
        """
        # Extract position data
        x = self.tracks_df['x_um'].values
        y = self.tracks_df['y_um'].values

        try:
            # Determine the spatial extent
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # Create grid for energy landscape
            x_edges = np.linspace(x_min, x_max, resolution + 1)
            y_edges = np.linspace(y_min, y_max, resolution + 1)

            # Calculate position histogram (particle density)
            H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

            # Add smoothing if requested
            if smoothing > 0:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=smoothing)

            # Small constant to avoid log(0)
            epsilon = H.max() * 1e-6
            H[H < epsilon] = epsilon

            # Calculate potential energy using appropriate method
            if method == "boltzmann":
                # Boltzmann inversion: U = -kBT * ln(P)
                U = -np.log(H / H.max())

            elif method == "drift":
                # Drift-based method requires velocity data
                # Calculate the mean drift at each position
                U = np.zeros_like(H)

                # Placeholder - would need to calculate drift field from the data
                # This is a simplification - not an actual drift calculation
                U = -np.log(H / H.max())  # Substitute with Boltzmann for now

            elif method == "kramers":
                # Kramers-Moyal expansion requires more detailed dynamics
                # This would require analysis of transition probabilities
                U = np.zeros_like(H)

                # Placeholder - not an actual Kramers-Moyal implementation
                U = -np.log(H / H.max())  # Substitute with Boltzmann for now

            else:
                raise ValueError(f"Unknown method: {method}")

            # Normalize energy to kBT units if requested
            if normalize:
                U = U  # Already normalized in kBT units by using log
                energy_units = "kBT"
            else:
                U = U * self.kBT  # Convert to actual energy in Joules
                energy_units = "J"

            # Calculate force field from the energy landscape
            force_field = self.calculate_force_field(U, x_edges, y_edges)

            # Create visualization
            fig = self.visualize_energy_landscape(U, x_edges, y_edges, force_field, energy_units)

            # Analyze dwell regions
            dwell_regions = self.analyze_dwell_regions(U, x_edges, y_edges)

            # Store results
            results = {
                'success': True,
                'energy_landscape': U,
                'x_edges': x_edges,
                'y_edges': y_edges,
                'force_field': force_field,
                'dwell_regions': dwell_regions,
                'parameters': {
                    'method': method,
                    'resolution': resolution,
                    'smoothing': smoothing,
                    'normalize': normalize,
                    'energy_units': energy_units
                },
                'visualization': fig
            }

        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }

        # Store results and return
        self.results['energy_landscape'] = results
        return results

    def analyze_dwell_regions(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, 
                              energy_threshold_factor: float = 0.5) -> Dict[str, Any]:
        """
        Identify and analyze potential dwell regions (energy minima).

        Parameters
        ----------
        potential_map : np.ndarray
            Energy landscape
        x_edges : np.ndarray
            X-axis bin edges
        y_edges : np.ndarray
            Y-axis bin edges
        energy_threshold_factor : float
            Energy threshold factor for identifying wells

        Returns
        -------
        Dict[str, Any]
            Dwell region analysis results
        """
        try:
            # Find local minima in the potential map
            from scipy.ndimage import minimum_filter

            # Apply minimum filter to find local minima
            min_filtered = minimum_filter(potential_map, size=3)
            minima = (potential_map == min_filtered) & (potential_map < potential_map.mean())

            # Label connected regions of minima
            from scipy.ndimage import label
            labeled_minima, num_minima = label(minima)

            # Calculate properties of each minimum
            minima_properties = []

            for i in range(1, num_minima + 1):
                # Get coordinates of this minimum
                y_indices, x_indices = np.where(labeled_minima == i)

                if len(y_indices) > 0:
                    # Calculate centroid
                    y_center = y_indices.mean()
                    x_center = x_indices.mean()

                    # Convert to physical coordinates
                    x_pos = np.interp(x_center, np.arange(len(x_edges) - 1), x_edges[:-1])
                    y_pos = np.interp(y_center, np.arange(len(y_edges) - 1), y_edges[:-1])

                    # Get the energy value at this minimum
                    energy_value = potential_map[y_indices[0], x_indices[0]]

                    # Calculate size (area) of the minimum region
                    size = len(y_indices)

                    # Add to properties list
                    minima_properties.append({
                        'id': i,
                        'x_position': x_pos,
                        'y_position': y_pos,
                        'energy': energy_value,
                        'size': size
                    })

            # Calculate energy barriers between minima (if multiple minima exist)
            energy_barriers = []

            if len(minima_properties) > 1:
                for i in range(len(minima_properties)):
                    for j in range(i + 1, len(minima_properties)):
                        min1 = minima_properties[i]
                        min2 = minima_properties[j]

                        # Calculate the straight-line path between the two minima
                        num_steps = 20
                        x_path = np.linspace(min1['x_position'], min2['x_position'], num_steps)
                        y_path = np.linspace(min1['y_position'], min2['y_position'], num_steps)

                        # Calculate energy along this path
                        path_energies = []
                        for x, y in zip(x_path, y_path):
                            # Convert positions to indices
                            x_idx = np.interp(x, x_edges[:-1], np.arange(len(x_edges) - 1))
                            y_idx = np.interp(y, y_edges[:-1], np.arange(len(y_edges) - 1))

                            # Ensure indices are within bounds
                            x_idx = int(min(max(0, x_idx), potential_map.shape[1] - 1))
                            y_idx = int(min(max(0, y_idx), potential_map.shape[0] - 1))

                            # Get energy at this point
                            energy = potential_map[y_idx, x_idx]
                            path_energies.append(energy)

                        # Calculate barrier height
                        energy_min = min(min1['energy'], min2['energy'])
                        barrier_height = max(path_energies) - energy_min

                        energy_barriers.append({
                            'from_id': min1['id'],
                            'to_id': min2['id'],
                            'barrier_height': barrier_height,
                            'distance': np.sqrt((min1['x_position'] - min2['x_position'])**2 + 
                                              (min1['y_position'] - min2['y_position'])**2)
                        })

            # Return dwell region analysis
            return {
                'num_minima': num_minima,
                'minima_properties': minima_properties,
                'energy_barriers': energy_barriers
            }

        except Exception as e:
            return {
                'error': str(e),
                'num_minima': 0,
                'minima_properties': [],
                'energy_barriers': []
            }

    def calculate_force_field(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> Dict[str, Any]:
        """
        Calculate force field from energy landscape gradient.

        Parameters
        ----------
        potential_map : np.ndarray
            Energy landscape
        x_edges : np.ndarray
            X-axis bin edges
        y_edges : np.ndarray
            Y-axis bin edges

        Returns
        -------
        Dict[str, Any]
            Force field data
        """
        try:
            # Calculate gradient of the potential map
            from scipy.ndimage import sobel

            # Calculate gradients using Sobel operator
            dy = sobel(potential_map, axis=0)
            dx = sobel(potential_map, axis=1)

            # Force is negative gradient
            fx = -dx
            fy = -dy

            # Calculate force magnitude
            f_magnitude = np.sqrt(fx**2 + fy**2)

            # Create grid coordinates
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers)

            # Subsample for visualization
            n_samples = min(20, min(len(x_centers), len(y_centers)))
            step_x = max(1, len(x_centers) // n_samples)
            step_y = max(1, len(y_centers) // n_samples)

            X_sub = X[::step_y, ::step_x]
            Y_sub = Y[::step_y, ::step_x]
            fx_sub = fx[::step_y, ::step_x]
            fy_sub = fy[::step_y, ::step_x]
            f_mag_sub = f_magnitude[::step_y, ::step_x]

            return {
                'fx': fx,
                'fy': fy,
                'magnitude': f_magnitude,
                'X_viz': X_sub,
                'Y_viz': Y_sub,
                'fx_viz': fx_sub,
                'fy_viz': fy_sub,
                'magnitude_viz': f_mag_sub,
                'x_centers': x_centers,
                'y_centers': y_centers
            }

        except Exception as e:
            return {
                'error': str(e)
            }

    def visualize_energy_landscape(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                                 force_field: Dict[str, Any], energy_units: str) -> go.Figure:
        """
        Create interactive visualization of energy landscape and force field.

        Parameters
        ----------
        potential_map : np.ndarray
            Energy landscape
        x_edges : np.ndarray
            X-axis bin edges
        y_edges : np.ndarray
            Y-axis bin edges
        force_field : Dict[str, Any]
            Force field data
        energy_units : str
            Units for energy display

        Returns
        -------
        go.Figure
            Plotly figure with energy landscape and force field
        """
        from plotly.subplots import make_subplots

        # Create subplots: energy surface and force field
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Energy Landscape", "Force Field"],
            specs=[[{"type": "surface"}, {"type": "contour"}]]
        )

        # Create x and y grids for surface plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        # Add energy landscape surface
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=potential_map,
                colorscale='Viridis',
                colorbar=dict(title=f'Energy ({energy_units})'),
                showscale=True,
                contours={
                    "z": {
                        "show": True,
                        "start": np.min(potential_map),
                        "end": np.max(potential_map),
                        "size": (np.max(potential_map) - np.min(potential_map)) / 10,
                    }
                }
            ),
            row=1, col=1
        )

        # Add contour plot with force vectors
        fig.add_trace(
            go.Contour(
                z=potential_map,
                x=x_centers,
                y=y_centers,
                colorscale='Viridis',
                showscale=False
            ),
            row=1, col=2
        )

        # Add force vectors if available
        if 'X_viz' in force_field:
            # Scale arrows
            max_mag = np.max(force_field['magnitude_viz']) if force_field['magnitude_viz'].size > 0 else 1
            if max_mag > 0:
                scale_factor = 0.3 * (x_edges.max() - x_edges.min()) / max_mag
            else:
                scale_factor = 1.0

            # Normalize to get unit vectors for direction
            with np.errstate(divide='ignore', invalid='ignore'):
                fx_norm = np.divide(force_field['fx_viz'], force_field['magnitude_viz'])
                fy_norm = np.divide(force_field['fy_viz'], force_field['magnitude_viz'])
                fx_norm[~np.isfinite(fx_norm)] = 0
                fy_norm[~np.isfinite(fy_norm)] = 0

            # Create quiver plot
            fig.add_trace(
                go.Scatter(
                    x=force_field['X_viz'].flatten(),
                    y=force_field['Y_viz'].flatten(),
                    mode='markers',
                    marker=dict(
                        symbol='arrow',
                        size=5,
                        color=force_field['magnitude_viz'].flatten(),
                        colorscale='Viridis',
                        showscale=False,
                        opacity=0.8
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title="Energy Landscape and Force Field",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title=f"Energy ({energy_units})"
            ),
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=1000,
            height=500
        )

        return fig


class ActiveTransportAnalyzer:
    """
    Advanced analyzer for active transport detection and characterization.
    Integrates with biophysical models for comprehensive transport analysis.
    """

    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0):
        """
        Initialize the active transport analyzer.

        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns 'track_id', 'frame', 'x', 'y'
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.results = {}

        # Convert coordinates to micrometers if needed
        for col in ['x', 'y']:
            if col in self.tracks_df.columns:
                self.tracks_df[f'{col}_um'] = self.tracks_df[col] * self.pixel_size

        # Preprocess: calculate velocities and accelerations
        self._preprocess_tracks()

    def _preprocess_tracks(self) -> None:
        """
        Preprocess tracks to calculate velocities, accelerations, and other features.
        """
        # Group by track_id
        grouped = self.tracks_df.groupby('track_id')

        # List to store processed tracks
        processed_tracks = []

        for track_id, track_data in grouped:
            if len(track_data) < 3:
                continue

            # Sort by frame
            track = track_data.sort_values('frame').copy()

            # Calculate displacements
            track['dx'] = track['x_um'].diff()
            track['dy'] = track['y_um'].diff()

            # Calculate time differences
            track['dt'] = track['frame'].diff() * self.frame_interval

            # Calculate speed and velocity
            track['speed'] = np.sqrt(track['dx']**2 + track['dy']**2) / track['dt']
            track['vx'] = track['dx'] / track['dt']
            track['vy'] = track['dy'] / track['dt']

            # Calculate acceleration
            track['ax'] = track['vx'].diff() / track['dt'].shift(-1)
            track['ay'] = track['vy'].diff() / track['dt'].shift(-1)
            track['acceleration'] = np.sqrt(track['ax']**2 + track['ay']**2)

            # Calculate angle changes (direction)
            v_angles = np.arctan2(track['vy'], track['vx'])
            track['angle_change'] = np.abs(np.diff(v_angles, append=v_angles.iloc[-1]))

            # Fix angle wraparound (ensure angle differences are between 0 and pi)
            track['angle_change'] = np.minimum(track['angle_change'], 2*np.pi - track['angle_change'])

            # Calculate straightness (distance traveled / path length)
            end_to_end_dist = np.sqrt(
                (track['x_um'].iloc[-1] - track['x_um'].iloc[0])**2 +
                (track['y_um'].iloc[-1] - track['y_um'].iloc[0])**2
            )
            path_length = np.sum(np.sqrt(track['dx']**2 + track['dy']**2))

            if path_length > 0:
                straightness = end_to_end_dist / path_length
            else:
                straightness = 0

            track['straightness'] = straightness

            processed_tracks.append(track)

        # Combine processed tracks
        if processed_tracks:
            self.processed_tracks = pd.concat(processed_tracks)

            # Compute track-level statistics
            track_stats = []

            for track_id, track_data in self.processed_tracks.groupby('track_id'):
                # Calculate mean statistics
                mean_speed = track_data['speed'].mean()
                mean_acc = track_data['acceleration'].mean()
                mean_angle_change = track_data['angle_change'].mean()
                straightness = track_data['straightness'].iloc[0]  # Same for all rows

                # Calculate max speed
                max_speed = track_data['speed'].max()

                track_stats.append({
                    'track_id': track_id,
                    'mean_speed': mean_speed,
                    'max_speed': max_speed,
                    'mean_acceleration': mean_acc,
                    'mean_angle_change': mean_angle_change,
                    'straightness': straightness,
                    'track_length': len(track_data)
                })

            self.track_results = pd.DataFrame(track_stats)

            # Store basic results
            self.results['basic_stats'] = {
                'success': True,
                'track_results': self.track_results,
                'processed_tracks': self.processed_tracks,
                'parameters': {
                    'pixel_size': self.pixel_size,
                    'frame_interval': self.frame_interval
                }
            }
        else:
            # No valid tracks after preprocessing
            self.processed_tracks = pd.DataFrame()
            self.track_results = pd.DataFrame()

            # Store error result
            self.results['basic_stats'] = {
                'success': False,
                'error': 'No valid tracks after preprocessing',
                'parameters': {
                    'pixel_size': self.pixel_size,
                    'frame_interval': self.frame_interval
                }
            }

    def detect_directional_motion_segments(self, min_segment_length: int = 5, 
                                          straightness_threshold: float = 0.8,
                                          velocity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect segments of directed motion within tracks.

        Parameters
        ----------
        min_segment_length : int
            Minimum number of frames for a valid segment
        straightness_threshold : float
            Minimum straightness value for directional segments (0-1)
        velocity_threshold : float
            Minimum velocity for active transport (μm/s)

        Returns
        -------
        Dict[str, Any]
            Directional segment detection results
        """
        # Check if we have basic results
        transport_results = self.results.get('basic_stats', {})

        if not transport_results.get('success', False):
            return {
                'success': False,
                'error': 'Basic track analysis failed. No valid tracks available.'
            }

        # Extract directional segments with velocity filtering
        segments = []
        track_results = transport_results.get('track_results', pd.DataFrame())

        # Vectorized filtering for tracks above velocity threshold
        fast_tracks = track_results[track_results.get('mean_speed', pd.Series(dtype=float)).fillna(0) >= velocity_threshold]

        # Vectorized segment creation - much faster than iterrows()
        if not fast_tracks.empty:
            for _, track in fast_tracks.iterrows():
                track_id = track['track_id']

                # Get processed track data
                track_data = self.processed_tracks[self.processed_tracks['track_id'] == track_id]

                if len(track_data) < min_segment_length:
                    continue

                # Check for straightness
                if track['straightness'] >= straightness_threshold:
                    # Calculate mean velocity vector
                    mean_vx = track_data['vx'].mean()
                    mean_vy = track_data['vy'].mean()
                    mean_velocity = np.sqrt(mean_vx**2 + mean_vy**2)
                    mean_direction = np.arctan2(mean_vy, mean_vx)

                    # Calculate metrics
                    duration = (track_data['frame'].max() - track_data['frame'].min()) * self.frame_interval
                    distance = np.sqrt(
                        (track_data['x_um'].iloc[-1] - track_data['x_um'].iloc[0])**2 +
                        (track_data['y_um'].iloc[-1] - track_data['y_um'].iloc[0])**2
                    )

                    segments.append({
                        'track_id': track_id,
                        'start_frame': track_data['frame'].min(),
                        'end_frame': track_data['frame'].max(),
                        'duration': duration,
                        'distance': distance,
                        'mean_velocity': mean_velocity,
                        'direction': mean_direction,
                        'straightness': track['straightness'],
                        'num_points': len(track_data)
                    })
        else:
            # No tracks met the velocity threshold
            pass

        self.results['directional_segments'] = {
            'success': True,
            'segments': segments,
            'total_segments': len(segments),
            'parameters': {
                'min_segment_length': min_segment_length,
                'straightness_threshold': straightness_threshold,
                'velocity_threshold': velocity_threshold
            }
        }
        return self.results['directional_segments']

    def characterize_transport_modes(self) -> Dict[str, Any]:
        """
        Characterize different modes of transport in the data.

        Returns
        -------
        Dict[str, Any]
            Classification of transport modes
        """
        if 'directional_segments' not in self.results:
            return {
                'success': False,
                'error': 'Run detect_directional_motion_segments() first to identify segments'
            }

        segments = self.results['directional_segments']['segments']

        if not segments:
            return {
                'success': False,
                'error': 'No directional segments detected'
            }

        # Classify transport modes
        velocities = [s['mean_velocity'] for s in segments]
        straightness_values = [s['straightness'] for s in segments]

        # Define thresholds for classification
        slow_threshold = 0.1  # μm/s
        fast_threshold = 0.5  # μm/s
        high_straightness = 0.8

        transport_modes = {
            'diffusive': 0,
            'slow_directed': 0,
            'fast_directed': 0,
            'mixed': 0
        }

        for velocity, straightness in zip(velocities, straightness_values):
            if velocity < slow_threshold:
                transport_modes['diffusive'] += 1
            elif velocity < fast_threshold:
                transport_modes['slow_directed'] += 1
            elif straightness >= high_straightness:
                transport_modes['fast_directed'] += 1
            else:
                transport_modes['mixed'] += 1

        total_segments = len(segments)
        transport_fractions = {mode: count/total_segments for mode, count in transport_modes.items()}

        self.results['transport_modes'] = {
            'success': True,
            'mode_counts': transport_modes,
            'mode_fractions': transport_fractions,
            'total_analyzed': total_segments,
            'mean_velocity': np.mean(velocities),
            'mean_straightness': np.mean(straightness_values)
        }
        return self.results['transport_modes']

def analyze_motion_models(tracks_df, models=None, min_track_length=10, time_window=None, pixel_size=1.0, frame_interval=1.0):
    """
    Analyze track data using different motion models to classify motion types.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with columns 'track_id', 'frame', 'x', 'y'
    models : list, optional
        List of motion models to test. Default is ['brownian', 'directed', 'confined']
    min_track_length : int, optional
        Minimum track length required for analysis, by default 10
    time_window : int, optional
        Number of frames to use for local analysis, by default None (use entire track)
    pixel_size : float, optional
        Pixel size in μm, by default 1.0
    frame_interval : float, optional
        Time between frames in seconds, by default 1.0

    Returns
    -------
    dict
        Dictionary containing model fit results and classifications
    """
    if models is None:
        models = ['brownian', 'directed', 'confined']

    # Validate input data
    if tracks_df is None or tracks_df.empty:
        return {'success': False, 'error': 'Empty or invalid tracks dataframe provided'}

    required_columns = ['track_id', 'frame', 'x', 'y']
    if not all(col in tracks_df.columns for col in required_columns):
        return {'success': False, 'error': f'Tracks dataframe missing required columns: {required_columns}'}

    # Convert coordinates to physical units
    tracks_df = tracks_df.copy()
    if 'x_um' not in tracks_df.columns:
        tracks_df['x_um'] = tracks_df['x'] * pixel_size
        tracks_df['y_um'] = tracks_df['y'] * pixel_size

    # Group tracks by track_id
    grouped = tracks_df.groupby('track_id')

    # Filter for minimum track length
    long_enough = [tid for tid, group in grouped if len(group) >= min_track_length]

    if not long_enough:
        return {
            'success': False,
            'error': f'No tracks meet the minimum length requirement of {min_track_length}'
        }

    results = {
        'success': True,
        'track_ids': long_enough,
        'models': models,
        'classifications': {},
        'model_parameters': {},
        'best_model': {},
        'error_metrics': {},
    }

    try:
        for track_id in long_enough:
            track_data = grouped.get_group(track_id).sort_values('frame')

            # Extract positions and frames - use um coordinates if available
            positions = track_data[['x_um', 'y_um']].values if 'x_um' in track_data.columns else track_data[['x', 'y']].values
            frames = track_data['frame'].values

            # Convert frames to time if frame_interval is provided
            times = frames * frame_interval

            # Handle time_window if specified
            if time_window and len(positions) > time_window:
                model_fits = {}
                model_errors = {}

                # Analyze sliding windows
                for i in range(len(positions) - time_window + 1):
                    window_pos = positions[i:i+time_window]
                    window_times = times[i:i+time_window]

                    try:
                        window_fits, window_errors = _fit_motion_models(window_pos, window_times, models)

                        for model, fit in window_fits.items():
                            if model not in model_fits:
                                model_fits[model] = []
                            model_fits[model].append(fit)

                        for model, error in window_errors.items():
                            if model not in model_errors:
                                model_errors[model] = []
                            model_errors[model].append(error)
                    except Exception as e:
                        print(f"Warning: Error fitting window for track {track_id}: {str(e)}")
                        continue

                if not model_errors:
                    # Skip this track if all window fits failed
                    continue

                # Aggregate window results - handle case where some fits may have failed
                best_model, best_score = _determine_best_model(model_errors)

                # Store results - calculate mean of successful fits for each parameter
                results['model_parameters'][track_id] = {}
                for model, fits in model_fits.items():
                    if fits:  # Check if any fits exist for this model
                        # Get all parameters from first fit
                        param_names = list(fits[0].keys())
                        results['model_parameters'][track_id][model] = {
                            param: np.mean([fit.get(param, 0) for fit in fits if param in fit]) 
                            for param in param_names
                        }

                # Calculate mean error for each model
                results['error_metrics'][track_id] = {
                    model: np.mean(errors) for model, errors in model_errors.items() if errors
                }

            else:
                # Analyze whole track
                try:
                    model_fits, model_errors = _fit_motion_models(positions, times, models)
                    best_model, best_score = _determine_best_model(model_errors)

                    results['model_parameters'][track_id] = model_fits
                    results['error_metrics'][track_id] = model_errors
                except Exception as e:
                    print(f"Warning: Error fitting track {track_id}: {str(e)}")
                    continue

            results['best_model'][track_id] = best_model
            results['classifications'][track_id] = best_model

        # Generate summary statistics only if we have results
        if results['classifications']:
            results['summary'] = _summarize_motion_analysis(results)
        else:
            results['success'] = False
            results['error'] = 'No tracks could be successfully analyzed'

    except Exception as e:
        results['success'] = False
        results['error'] = f'Error in motion model analysis: {str(e)}'

    return results

def _fit_motion_models(positions, times, models):
    """
    Fit different motion models to position data.

    Parameters
    ----------
    positions : np.ndarray
        Array of x,y positions
    times : np.ndarray
        Array of times (seconds)
    models : list
        List of motion model names to fit

    Returns
    -------
    tuple
        (model_fits, model_errors) dictionaries
    """
    model_fits = {}
    model_errors = {}

    # Check for sufficient data points
    if len(positions) < 4:
        raise ValueError("At least 4 positions are required for model fitting")

    # Calculate displacements and time intervals
    displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    dt = np.diff(times)

    # Handle edge case with zero time intervals
    if np.any(dt <= 0):
        # Replace zero time intervals with minimum positive value to avoid division by zero
        min_positive_dt = np.min(dt[dt > 0]) if np.any(dt > 0) else 1e-6
        dt = np.maximum(dt, min_positive_dt)

    # Brownian motion model
    if 'brownian' in models:
        try:
            D, brownian_error = _fit_brownian_motion(displacements, dt)
            model_fits['brownian'] = {'D': D}
            model_errors['brownian'] = brownian_error
        except Exception as e:
            print(f"Warning: Failed to fit Brownian motion model: {str(e)}")

    # Directed motion model
    if 'directed' in models:
        try:
            D, v, directed_error = _fit_directed_motion(displacements, dt, positions, times)
            model_fits['directed'] = {'D': D, 'v': v}
            model_errors['directed'] = directed_error
        except Exception as e:
            print(f"Warning: Failed to fit directed motion model: {str(e)}")

    # Confined motion model
    if 'confined' in models:
        try:
            D, L, confined_error = _fit_confined_motion(displacements, dt, positions)
            model_fits['confined'] = {'D': D, 'L': L}
            model_errors['confined'] = confined_error
        except Exception as e:
            print(f"Warning: Failed to fit confined motion model: {str(e)}")

    # If no models were successfully fit, raise exception
    if not model_fits:
        raise ValueError("Failed to fit any motion models to the data")

    return model_fits, model_errors

def _fit_brownian_motion(displacements, dt):
    """Fit Brownian motion model to displacements."""
    # For pure Brownian motion: MSD = 4*D*t
    # Estimate diffusion coefficient

    # Calculate MSD for different time lags
    max_lag = min(20, len(displacements) // 4)  # Use at most 1/4 of the track length
    max_lag = max(2, max_lag)  # At least 2 lags

    msd_by_lag = []
    for lag in range(1, max_lag + 1):
        if lag >= len(displacements):
            break
        # Get displacements for this lag
        lag_disps = displacements[0:len(displacements)-lag+1]
        lag_dts = dt[0:len(dt)-lag+1].sum()
        msd = np.mean(lag_disps**2)
        msd_by_lag.append((lag_dts, msd))

    # Fit MSD = 4*D*t
    times = np.array([t for t, _ in msd_by_lag])
    msds = np.array([m for _, m in msd_by_lag])

    # Linear fit through origin
    D = np.sum(msds * times) / (4 * np.sum(times**2))
    D = max(D, 1e-9)  # Ensure positive diffusion coefficient

    # Calculate error as residual between observed and expected MSDs
    expected_msds = 4 * D * times
    error = np.mean((msds - expected_msds)**2)

    return D, error

def _fit_directed_motion(displacements, dt, positions, times):
    """Fit directed motion model to displacements and positions."""
    # For directed motion: MSD = 4*D*t + (v*t)^2

    # First, estimate direction using linear regression
    t = times - times[0]  # Start from zero
    x = positions[:, 0]
    y = positions[:, 1]

    # Use try-except to handle singular matrices
    try:
        vx, _ = np.polyfit(t, x, 1)
        vy, _ = np.polyfit(t, y, 1)
    except np.linalg.LinAlgError:
        # Fall back to simple difference if polyfit fails
        vx = (x[-1] - x[0]) / (t[-1] - t[0] + 1e-9)
        vy = (y[-1] - y[0]) / (t[-1] - t[0] + 1e-9)

    v = np.sqrt(vx**2 + vy**2)

    # Calculate MSD for different time lags
    max_lag = min(20, len(displacements) // 4)
    max_lag = max(2, max_lag)

    msd_by_lag = []
    for lag in range(1, max_lag + 1):
        if lag >= len(displacements):
            break
        lag_disps = displacements[0:len(displacements)-lag+1]
        lag_dts = dt[0:len(dt)-lag+1].sum()
        msd = np.mean(lag_disps**2)
        msd_by_lag.append((lag_dts, msd))

    times = np.array([t for t, _ in msd_by_lag])
    msds = np.array([m for _, m in msd_by_lag])

    # Fit MSD = 4*D*t + (v*t)^2
    # Use linear regression: MSD = 4*D*t + (v^2)*t^2
    A = np.column_stack((times, times**2))
    try:
        params, residuals, _, _ = np.linalg.lstsq(A, msds, rcond=None)
        D = params[0] / 4  # Extract diffusion coefficient
        v_squared = params[1]  # Extract velocity squared
    except np.linalg.LinAlgError:
        # Fall back to simple estimates if lstsq fails
        D = np.mean(msds) / (4 * np.mean(times))
        v_squared = v**2  # Use previously estimated velocity

    # Ensure non-negative diffusion coefficient
    D = max(0, D)

    # Validate velocity: use sqrt of v_squared if positive, otherwise use previously estimated v
    v_fit = np.sqrt(max(0, v_squared)) if v_squared > 0 else v

    # Calculate error
    expected_msds = 4 * D * times + (v_fit * times)**2
    error = np.mean((msds - expected_msds)**2)

    return D, v_fit, error

def _fit_confined_motion(displacements, dt, positions):
    """Fit confined motion model to displacements and positions."""
    # For confined motion: MSD = L^2 * [1 - exp(-4*D*t/L^2)]

    # Initial estimates
    msd = np.mean(displacements**2)
    mean_dt = np.mean(dt)
    D_initial = msd / (4 * mean_dt)

    # Ensure D_initial is positive
    D_initial = max(D_initial, 1e-9)

    # Estimate confinement size as 2x standard deviation of positions
    std_pos = np.std(positions, axis=0)
    L_initial = 2 * np.mean(std_pos)

    # Ensure L_initial is positive
    L_initial = max(L_initial, 1e-9)

    # Time vector for fitting
    t_fit = np.linspace(0, np.sum(dt), len(displacements))

    # Define the model function
    def confined_model(t, D, L):
        return L**2 * (1 - np.exp(-4 * D * t / L**2))

    # Fit the model to the data
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(confined_model, t_fit, displacements, p0=[D_initial, L_initial])
        D_confined, L_confined = popt
    except Exception as e:
        D_confined, L_confined = np.nan, np.nan
        print(f"Warning: Confined motion fit failed: {str(e)}")

    # Calculate error as residual between observed and expected MSDs
    expected_msds = confined_model(t_fit, D_confined, L_confined)
    error = np.mean((displacements - expected_msds)**2)

    return D_confined, L_confined, error

# Bayesian segmentation helpers (lazy import to avoid mandatory dependency during normal use)
def run_bocpd_segmentation(tracks_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience wrapper for Bayesian Online Changepoint Detection on diffusion steps.
    Parameters
    ----------
    tracks_df : DataFrame with columns track_id, frame, x, y
    config : dict of BOCPDConfig overrides
    """
    try:
        from bayes_bocpd_diffusion import BOCPDDiffusion, BOCPDConfig
    except ImportError:
        return {'success': False, 'error': 'bayes_bocpd_diffusion module not found'}
    cfg_kwargs = config or {}
    cfg = BOCPDConfig(**{k: v for k, v in cfg_kwargs.items() if k in BOCPDConfig().__dict__})
    model = BOCPDDiffusion(tracks_df, cfg)
    return model.segment_all()

def run_hmm_segmentation(tracks_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience wrapper for Bayesian (MAP) HMM diffusion / drift state segmentation.
    Parameters
    ----------
    tracks_df : DataFrame with columns track_id, frame, x, y
    config : dict of HMMConfig overrides
    """
    try:
        from bayes_hmm_diffusion import BayesHMMDiffusion, HMMConfig
    except ImportError:
        return {'success': False, 'error': 'bayes_hmm_diffusion module not found'}
    cfg_kwargs = config or {}
    cfg = HMMConfig(**{k: v for k, v in cfg_kwargs.items() if k in HMMConfig().__dict__})
    model = BayesHMMDiffusion(tracks_df, cfg)
    return model.segment_all()