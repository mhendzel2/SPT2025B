"""
Interactive Particle Trajectory Heatmap Visualization for SPT Analysis
Provides comprehensive heatmap visualizations for particle trajectory analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class TrajectoryHeatmapVisualizer:
    """
    Advanced heatmap visualization system for particle trajectories.
    """
    
    def __init__(self):
        self.pixel_size = 0.1  # Default pixel size in micrometers
        self.frame_interval = 0.1  # Default frame interval in seconds
        
    def update_parameters(self, pixel_size: float, frame_interval: float):
        """Update visualization parameters."""
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
    
    def create_density_heatmap(self, track_data: pd.DataFrame, 
                              grid_resolution: int = 100,
                              bandwidth: float = 1.0,
                              normalize: bool = True) -> Dict[str, Any]:
        """
        Create a 2D density heatmap of particle positions.
        
        Parameters
        ----------
        track_data : pd.DataFrame
            Track data with columns ['x', 'y', 'track_id']
        grid_resolution : int
            Resolution of the heatmap grid
        bandwidth : float
            Bandwidth for kernel density estimation
        normalize : bool
            Whether to normalize the density values
            
        Returns
        -------
        Dict[str, Any]
            Heatmap data and visualization
        """
        if track_data.empty:
            return {"error": "No track data provided"}
        
        # Extract position data
        x_coords = track_data['x'].values * self.pixel_size
        y_coords = track_data['y'].values * self.pixel_size
        
        # Define grid boundaries
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Perform kernel density estimation
        positions = np.column_stack([x_coords, y_coords])
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(positions)
            density = np.exp(kde.score_samples(grid_points))
            density = density.reshape(X.shape)
            
            if normalize:
                density = density / np.max(density)
                
        except Exception as e:
            # Fallback to histogram-based approach
            density, _, _ = np.histogram2d(x_coords, y_coords, 
                                        bins=[x_grid, y_grid], 
                                        density=True)
            density = density.T
            
            if normalize:
                density = density / np.max(density) if np.max(density) > 0 else density
        
        return {
            "density": density,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "stats": {
                "total_points": len(x_coords),
                "unique_tracks": track_data['track_id'].nunique() if 'track_id' in track_data.columns else 1,
                "density_range": [np.min(density), np.max(density)],
                "spatial_extent": [x_max - x_min, y_max - y_min]
            }
        }
    
    def create_velocity_heatmap(self, track_data: pd.DataFrame,
                               grid_resolution: int = 100,
                               velocity_metric: str = 'magnitude') -> Dict[str, Any]:
        """
        Create a heatmap of velocity distributions.
        
        Parameters
        ----------
        track_data : pd.DataFrame
            Track data with columns ['x', 'y', 'frame', 'track_id']
        grid_resolution : int
            Resolution of the heatmap grid
        velocity_metric : str
            Velocity metric to visualize ('magnitude', 'x_component', 'y_component')
            
        Returns
        -------
        Dict[str, Any]
            Velocity heatmap data and visualization
        """
        if track_data.empty:
            return {"error": "No track data provided"}
        
        velocity_data = []
        
        for track_id in track_data['track_id'].unique():
            track = track_data[track_data['track_id'] == track_id].sort_values('frame')
            
            if len(track) < 2:
                continue
            
            # Calculate velocities
            x_positions = track['x'].values * self.pixel_size
            y_positions = track['y'].values * self.pixel_size
            
            vx = np.diff(x_positions) / self.frame_interval
            vy = np.diff(y_positions) / self.frame_interval
            
            # Calculate velocity magnitude
            v_mag = np.sqrt(vx**2 + vy**2)
            
            # Position at midpoints
            x_mid = (x_positions[:-1] + x_positions[1:]) / 2
            y_mid = (y_positions[:-1] + y_positions[1:]) / 2
            
            for i in range(len(vx)):
                velocity_data.append({
                    'x': x_mid[i],
                    'y': y_mid[i],
                    'vx': vx[i],
                    'vy': vy[i],
                    'v_magnitude': v_mag[i],
                    'track_id': track_id
                })
        
        if not velocity_data:
            return {"error": "No velocity data could be calculated"}
        
        velocity_df = pd.DataFrame(velocity_data)
        
        # Select velocity component
        if velocity_metric == 'magnitude':
            velocity_values = velocity_df['v_magnitude'].values
        elif velocity_metric == 'x_component':
            velocity_values = velocity_df['vx'].values
        elif velocity_metric == 'y_component':
            velocity_values = velocity_df['vy'].values
        else:
            velocity_values = velocity_df['v_magnitude'].values
        
        # Create grid
        x_coords = velocity_df['x'].values
        y_coords = velocity_df['y'].values
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate velocity data onto grid
        try:
            velocity_grid = griddata(
                (x_coords, y_coords), 
                velocity_values,
                (X, Y), 
                method='cubic',
                fill_value=0
            )
        except:
            # Fallback to linear interpolation
            velocity_grid = griddata(
                (x_coords, y_coords), 
                velocity_values,
                (X, Y), 
                method='linear',
                fill_value=0
            )
        
        return {
            "velocity_grid": velocity_grid,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "velocity_data": velocity_df,
            "metric": velocity_metric,
            "stats": {
                "mean_velocity": np.mean(velocity_values),
                "max_velocity": np.max(velocity_values),
                "velocity_std": np.std(velocity_values),
                "total_measurements": len(velocity_values)
            }
        }
    
    def create_dwell_time_heatmap(self, track_data: pd.DataFrame,
                                 grid_resolution: int = 100,
                                 radius_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Create a heatmap showing dwell times in different regions.
        
        Parameters
        ----------
        track_data : pd.DataFrame
            Track data with columns ['x', 'y', 'frame', 'track_id']
        grid_resolution : int
            Resolution of the heatmap grid
        radius_threshold : float
            Radius threshold for defining dwell regions (in micrometers)
            
        Returns
        -------
        Dict[str, Any]
            Dwell time heatmap data
        """
        if track_data.empty:
            return {"error": "No track data provided"}
        
        # Convert to physical units
        x_coords = track_data['x'].values * self.pixel_size
        y_coords = track_data['y'].values * self.pixel_size
        
        # Define grid boundaries
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        
        # Initialize dwell time grid
        dwell_grid = np.zeros((grid_resolution, grid_resolution))
        
        # Calculate dwell times for each track
        for track_id in track_data['track_id'].unique():
            track = track_data[track_data['track_id'] == track_id].sort_values('frame')
            
            if len(track) < 2:
                continue
            
            track_x = track['x'].values * self.pixel_size
            track_y = track['y'].values * self.pixel_size
            track_frames = track['frame'].values
            
            # Find dwell regions
            for i in range(len(track) - 1):
                current_pos = np.array([track_x[i], track_y[i]])
                next_pos = np.array([track_x[i+1], track_y[i+1]])
                
                # Check if particle stayed in same region
                distance = np.linalg.norm(next_pos - current_pos)
                
                if distance <= radius_threshold:
                    # Find grid indices
                    x_idx = np.searchsorted(x_grid, current_pos[0])
                    y_idx = np.searchsorted(y_grid, current_pos[1])
                    
                    # Ensure indices are within bounds
                    x_idx = max(0, min(x_idx, grid_resolution - 1))
                    y_idx = max(0, min(y_idx, grid_resolution - 1))
                    
                    # Add dwell time (frame interval)
                    dwell_grid[y_idx, x_idx] += self.frame_interval
        
        return {
            "dwell_grid": dwell_grid,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "radius_threshold": radius_threshold,
            "stats": {
                "max_dwell_time": np.max(dwell_grid),
                "mean_dwell_time": np.mean(dwell_grid[dwell_grid > 0]) if np.any(dwell_grid > 0) else 0,
                "total_dwell_regions": np.sum(dwell_grid > 0)
            }
        }
    
    def create_temporal_heatmap(self, track_data: pd.DataFrame,
                               time_bins: int = 20,
                               grid_resolution: int = 50) -> Dict[str, Any]:
        """
        Create temporal heatmaps showing particle density evolution over time.
        
        Parameters
        ----------
        track_data : pd.DataFrame
            Track data with columns ['x', 'y', 'frame', 'track_id']
        time_bins : int
            Number of temporal bins
        grid_resolution : int
            Spatial resolution of each time bin
            
        Returns
        -------
        Dict[str, Any]
            Temporal heatmap data
        """
        if track_data.empty:
            return {"error": "No track data provided"}
        
        # Convert to physical units
        x_coords = track_data['x'].values * self.pixel_size
        y_coords = track_data['y'].values * self.pixel_size
        frames = track_data['frame'].values
        
        # Define spatial grid
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        
        # Define temporal bins
        frame_min, frame_max = np.min(frames), np.max(frames)
        frame_bins = np.linspace(frame_min, frame_max, time_bins + 1)
        
        temporal_heatmaps = []
        time_labels = []
        
        for i in range(time_bins):
            # Select data in current time bin
            time_mask = (frames >= frame_bins[i]) & (frames < frame_bins[i+1])
            
            if np.sum(time_mask) == 0:
                # No data in this time bin
                temporal_heatmaps.append(np.zeros((grid_resolution, grid_resolution)))
            else:
                bin_x = x_coords[time_mask]
                bin_y = y_coords[time_mask]
                
                # Create 2D histogram for this time bin
                density, _, _ = np.histogram2d(bin_x, bin_y, 
                                            bins=[x_grid, y_grid])
                density = density.T
                temporal_heatmaps.append(density)
            
            # Create time label
            start_time = frame_bins[i] * self.frame_interval
            end_time = frame_bins[i+1] * self.frame_interval
            time_labels.append(f"{start_time:.2f}-{end_time:.2f}s")
        
        return {
            "temporal_heatmaps": temporal_heatmaps,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "time_labels": time_labels,
            "frame_bins": frame_bins,
            "stats": {
                "time_bins": time_bins,
                "total_timespan": (frame_max - frame_min) * self.frame_interval,
                "frames_per_bin": len(frames) / time_bins
            }
        }
    
    def plot_density_heatmap(self, heatmap_data: Dict[str, Any],
                           title: str = "Particle Density Heatmap") -> go.Figure:
        """Create interactive density heatmap plot."""
        if "error" in heatmap_data:
            fig = go.Figure()
            fig.add_annotation(text=heatmap_data["error"], 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=heatmap_data["density"],
            x=heatmap_data["x_grid"],
            y=heatmap_data["y_grid"],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Density")
        ))
        
        # Add scatter overlay of actual positions
        fig.add_trace(go.Scatter(
            x=heatmap_data["x_coords"],
            y=heatmap_data["y_coords"],
            mode='markers',
            marker=dict(size=2, color='white', opacity=0.3),
            name='Particle Positions',
            showlegend=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_velocity_heatmap(self, velocity_data: Dict[str, Any],
                            title: str = "Velocity Heatmap") -> go.Figure:
        """Create interactive velocity heatmap plot."""
        if "error" in velocity_data:
            fig = go.Figure()
            fig.add_annotation(text=velocity_data["error"], 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = go.Figure()
        
        # Add velocity heatmap
        fig.add_trace(go.Heatmap(
            z=velocity_data["velocity_grid"],
            x=velocity_data["x_grid"],
            y=velocity_data["y_grid"],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title=f"Velocity {velocity_data['metric']} (μm/s)")
        ))
        
        fig.update_layout(
            title=f"{title} - {velocity_data['metric'].replace('_', ' ').title()}",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_dwell_time_heatmap(self, dwell_data: Dict[str, Any],
                               title: str = "Dwell Time Heatmap") -> go.Figure:
        """Create interactive dwell time heatmap plot."""
        if "error" in dwell_data:
            fig = go.Figure()
            fig.add_annotation(text=dwell_data["error"], 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = go.Figure()
        
        # Add dwell time heatmap
        fig.add_trace(go.Heatmap(
            z=dwell_data["dwell_grid"],
            x=dwell_data["x_grid"],
            y=dwell_data["y_grid"],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Dwell Time (s)")
        ))
        
        fig.update_layout(
            title=f"{title} (Radius threshold: {dwell_data['radius_threshold']:.2f} μm)",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_temporal_heatmap(self, temporal_data: Dict[str, Any],
                            title: str = "Temporal Evolution Heatmap") -> go.Figure:
        """Create animated temporal heatmap plot."""
        if "error" in temporal_data:
            fig = go.Figure()
            fig.add_annotation(text=temporal_data["error"], 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create frames for animation
        frames = []
        
        for i, (heatmap, time_label) in enumerate(zip(temporal_data["temporal_heatmaps"], 
                                                     temporal_data["time_labels"])):
            frame = go.Frame(
                data=[go.Heatmap(
                    z=heatmap,
                    x=temporal_data["x_grid"],
                    y=temporal_data["y_grid"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Particle Count")
                )],
                name=time_label
            )
            frames.append(frame)
        
        # Initial frame
        fig = go.Figure(
            data=[go.Heatmap(
                z=temporal_data["temporal_heatmaps"][0],
                x=temporal_data["x_grid"],
                y=temporal_data["y_grid"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Particle Count")
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=title,
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=800,
            height=600,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 100}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "steps": [
                    {
                        "args": [[time_label], 
                                {"frame": {"duration": 100, "redraw": True},
                                 "mode": "immediate", "transition": {"duration": 100}}],
                        "label": time_label,
                        "method": "animate"
                    }
                    for time_label in temporal_data["time_labels"]
                ],
                "x": 0.1,
                "len": 0.9,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        return fig
    
    def create_combined_analysis(self, track_data: pd.DataFrame,
                               grid_resolution: int = 100) -> Dict[str, Any]:
        """Create comprehensive heatmap analysis combining multiple metrics."""
        results = {}
        
        # Density heatmap
        results["density"] = self.create_density_heatmap(track_data, grid_resolution)
        
        # Velocity heatmap
        results["velocity"] = self.create_velocity_heatmap(track_data, grid_resolution)
        
        # Dwell time heatmap
        results["dwell_time"] = self.create_dwell_time_heatmap(track_data, grid_resolution)
        
        # Temporal heatmap
        results["temporal"] = self.create_temporal_heatmap(track_data, 
                                                         time_bins=10, 
                                                         grid_resolution=grid_resolution//2)
        
        return results


def create_streamlit_heatmap_interface():
    """Create Streamlit interface for trajectory heatmap visualization."""
    
    st.subheader("🔥 Interactive Particle Trajectory Heatmap Visualization")
    st.markdown("Generate comprehensive heatmap visualizations to analyze particle movement patterns and spatial distributions.")
    
    # Check for track data
    if 'track_data' not in st.session_state or st.session_state.track_data is None:
        st.warning("⚠️ No track data loaded. Please load track data in the Data Loading section first.")
        return
    
    track_data = st.session_state.track_data
    
    # Initialize visualizer
    visualizer = TrajectoryHeatmapVisualizer()
    
    # Update parameters from session state
    pixel_size = st.session_state.get('pixel_size', 0.1)
    frame_interval = st.session_state.get('frame_interval', 0.1)
    visualizer.update_parameters(pixel_size, frame_interval)
    
    # Sidebar controls
    st.sidebar.markdown("### Heatmap Parameters")
    
    grid_resolution = st.sidebar.slider(
        "Grid Resolution",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        help="Higher resolution provides more detail but slower processing"
    )
    
    bandwidth = st.sidebar.slider(
        "Density Bandwidth",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Smoothing parameter for density estimation"
    )
    
    # Heatmap type selection
    heatmap_type = st.selectbox(
        "Heatmap Type",
        ["Density", "Velocity", "Dwell Time", "Temporal Evolution", "Combined Analysis"],
        help="Select the type of heatmap visualization"
    )
    
    if heatmap_type == "Density":
        st.markdown("### Particle Density Heatmap")
        st.markdown("Shows the spatial distribution of particle positions using kernel density estimation.")
        
        if st.button("Generate Density Heatmap", type="primary"):
            with st.spinner("Generating density heatmap..."):
                heatmap_data = visualizer.create_density_heatmap(
                    track_data, 
                    grid_resolution=grid_resolution,
                    bandwidth=bandwidth
                )
                
                if "error" not in heatmap_data:
                    fig = visualizer.plot_density_heatmap(heatmap_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Points", heatmap_data["stats"]["total_points"])
                    with col2:
                        st.metric("Unique Tracks", heatmap_data["stats"]["unique_tracks"])
                    with col3:
                        st.metric("Max Density", f"{heatmap_data['stats']['density_range'][1]:.3f}")
                    with col4:
                        st.metric("Spatial Extent", f"{heatmap_data['stats']['spatial_extent'][0]:.2f} × {heatmap_data['stats']['spatial_extent'][1]:.2f} μm")
                else:
                    st.error(heatmap_data["error"])
    
    elif heatmap_type == "Velocity":
        st.markdown("### Velocity Heatmap")
        st.markdown("Shows the spatial distribution of particle velocities.")
        
        velocity_metric = st.selectbox(
            "Velocity Component",
            ["magnitude", "x_component", "y_component"],
            help="Select which velocity component to visualize"
        )
        
        if st.button("Generate Velocity Heatmap", type="primary"):
            with st.spinner("Generating velocity heatmap..."):
                velocity_data = visualizer.create_velocity_heatmap(
                    track_data,
                    grid_resolution=grid_resolution,
                    velocity_metric=velocity_metric
                )
                
                if "error" not in velocity_data:
                    fig = visualizer.plot_velocity_heatmap(velocity_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Velocity", f"{velocity_data['stats']['mean_velocity']:.3f} μm/s")
                    with col2:
                        st.metric("Max Velocity", f"{velocity_data['stats']['max_velocity']:.3f} μm/s")
                    with col3:
                        st.metric("Velocity Std", f"{velocity_data['stats']['velocity_std']:.3f} μm/s")
                    with col4:
                        st.metric("Measurements", velocity_data['stats']['total_measurements'])
                else:
                    st.error(velocity_data["error"])
    
    elif heatmap_type == "Dwell Time":
        st.markdown("### Dwell Time Heatmap")
        st.markdown("Shows regions where particles spend more time (slow or confined motion).")
        
        radius_threshold = st.slider(
            "Dwell Radius Threshold (μm)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Maximum distance to consider as dwelling in the same region"
        )
        
        if st.button("Generate Dwell Time Heatmap", type="primary"):
            with st.spinner("Generating dwell time heatmap..."):
                dwell_data = visualizer.create_dwell_time_heatmap(
                    track_data,
                    grid_resolution=grid_resolution,
                    radius_threshold=radius_threshold
                )
                
                if "error" not in dwell_data:
                    fig = visualizer.plot_dwell_time_heatmap(dwell_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Dwell Time", f"{dwell_data['stats']['max_dwell_time']:.3f} s")
                    with col2:
                        st.metric("Mean Dwell Time", f"{dwell_data['stats']['mean_dwell_time']:.3f} s")
                    with col3:
                        st.metric("Dwell Regions", dwell_data['stats']['total_dwell_regions'])
                else:
                    st.error(dwell_data["error"])
    
    elif heatmap_type == "Temporal Evolution":
        st.markdown("### Temporal Evolution Heatmap")
        st.markdown("Shows how particle density changes over time with animation controls.")
        
        time_bins = st.slider(
            "Number of Time Bins",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of time segments for temporal analysis"
        )
        
        if st.button("Generate Temporal Heatmap", type="primary"):
            with st.spinner("Generating temporal heatmap..."):
                temporal_data = visualizer.create_temporal_heatmap(
                    track_data,
                    time_bins=time_bins,
                    grid_resolution=grid_resolution//2
                )
                
                if "error" not in temporal_data:
                    fig = visualizer.plot_temporal_heatmap(temporal_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Use the play button and slider controls below the plot to navigate through time.**")
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Time Bins", temporal_data['stats']['time_bins'])
                    with col2:
                        st.metric("Total Timespan", f"{temporal_data['stats']['total_timespan']:.2f} s")
                    with col3:
                        st.metric("Avg Frames/Bin", f"{temporal_data['stats']['frames_per_bin']:.1f}")
                else:
                    st.error(temporal_data["error"])
    
    elif heatmap_type == "Combined Analysis":
        st.markdown("### Combined Heatmap Analysis")
        st.markdown("Comprehensive analysis showing multiple heatmap types for complete trajectory characterization.")
        
        if st.button("Generate Combined Analysis", type="primary"):
            with st.spinner("Generating comprehensive heatmap analysis..."):
                combined_results = visualizer.create_combined_analysis(
                    track_data,
                    grid_resolution=grid_resolution
                )
                
                # Display all heatmaps
                tabs = st.tabs(["Density", "Velocity", "Dwell Time", "Temporal"])
                
                with tabs[0]:
                    if "error" not in combined_results["density"]:
                        fig = visualizer.plot_density_heatmap(combined_results["density"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(combined_results["density"]["error"])
                
                with tabs[1]:
                    if "error" not in combined_results["velocity"]:
                        fig = visualizer.plot_velocity_heatmap(combined_results["velocity"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(combined_results["velocity"]["error"])
                
                with tabs[2]:
                    if "error" not in combined_results["dwell_time"]:
                        fig = visualizer.plot_dwell_time_heatmap(combined_results["dwell_time"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(combined_results["dwell_time"]["error"])
                
                with tabs[3]:
                    if "error" not in combined_results["temporal"]:
                        fig = visualizer.plot_temporal_heatmap(combined_results["temporal"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(combined_results["temporal"]["error"])
    
    # Export options
    st.markdown("---")
    st.markdown("### Export Options")
    
    export_format = st.selectbox(
        "Export Format",
        ["PNG", "SVG", "PDF", "HTML"],
        help="Select format for exporting heatmap visualizations"
    )
    
    if st.button("Export Current Heatmap"):
        st.info(f"Heatmap export functionality for {export_format} format will be implemented in the next update.")