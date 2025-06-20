"""
Visualization functions for the SPT Analysis application.
This module provides various visualization tools for track data and analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional, Tuple, Union
import io
import base64
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# --- Track Visualization ---

def plot_tracks(tracks_df: pd.DataFrame, frame_range: Optional[Tuple[int, int]] = None, 
                color_by: str = 'track_id', colormap: str = 'viridis', 
                plot_type: str = 'plotly') -> Union[go.Figure, plt.Figure]:
    """
    # Validate input data
    if tracks_df.empty:
        st.warning("No track data available for visualization.")
        return None
    
    required_columns = ['x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    if missing_columns:
        st.error(f"Missing required columns for plotting: {missing_columns}")
        return None
    Plot particle tracks as 2D trajectories.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks data in standard format
    frame_range : tuple, optional
        Range of frames to include (min, max)
    color_by : str
        Column to use for track coloring
    colormap : str
        Colormap name
    plot_type : str
        Type of plot: 'plotly' or 'matplotlib'
        
    Returns
    -------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        Plot figure object
    """
    if tracks_df.empty:
        if plot_type == 'plotly':
            fig = go.Figure()
            fig.update_layout(
                title="No track data available",
                xaxis_title="X",
                yaxis_title="Y"
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No track data available", 
                    ha='center', va='center', fontsize=14)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            return fig
    
    # Filter by frame range if provided
    if frame_range is not None:
        tracks_df = tracks_df[(tracks_df['frame'] >= frame_range[0]) & 
                              (tracks_df['frame'] <= frame_range[1])]
    
    # Get unique tracks
    unique_tracks = tracks_df['track_id'].unique()
    
    if plot_type == 'plotly':
        # Create Plotly figure
        fig = go.Figure()
        
        # Color scale
        if color_by == 'track_id':
            # Use track_id for coloring
            colors = px.colors.sample_colorscale(
                colormap, np.linspace(0, 1, len(unique_tracks))
            )
            
            for i, track_id in enumerate(unique_tracks):
                track_data = tracks_df[tracks_df['track_id'] == track_id]
                track_data = track_data.sort_values('frame')
                
                fig.add_trace(go.Scatter(
                    x=track_data['x'],
                    y=track_data['y'],
                    mode='lines+markers',
                    name=f'Track {track_id}',
                    marker=dict(color=colors[i]),
                    line=dict(color=colors[i]),
                    showlegend=len(unique_tracks) <= 20  # Only show legend if few tracks
                ))
                
                # Add starting point marker
                fig.add_trace(go.Scatter(
                    x=[track_data['x'].iloc[0]],
                    y=[track_data['y'].iloc[0]],
                    mode='markers',
                    marker=dict(
                        color=colors[i],
                        symbol='circle-open',
                        size=10,
                        line=dict(width=2)
                    ),
                    showlegend=False
                ))
        else:
            # Use another column for coloring
            if color_by in tracks_df.columns and pd.api.types.is_numeric_dtype(tracks_df[color_by]):
                # Create a continuous color scale
                for track_id in unique_tracks:
                    track_data = tracks_df[tracks_df['track_id'] == track_id]
                    track_data = track_data.sort_values('frame')
                    
                    fig.add_trace(go.Scatter(
                        x=track_data['x'],
                        y=track_data['y'],
                        mode='lines+markers',
                        name=f'Track {track_id}',
                        marker=dict(
                            color=track_data[color_by],
                            colorscale=colormap,
                            showscale=True,
                            colorbar=dict(title=color_by)
                        ),
                        line=dict(color=px.colors.sample_colorscale(
                            colormap, [track_data[color_by].mean()])[0]),
                        showlegend=len(unique_tracks) <= 20
                    ))
            else:
                # Use track_id as fallback
                colors = px.colors.sample_colorscale(
                    colormap, np.linspace(0, 1, len(unique_tracks))
                )
                
                for i, track_id in enumerate(unique_tracks):
                    track_data = tracks_df[tracks_df['track_id'] == track_id]
                    track_data = track_data.sort_values('frame')
                    
                    fig.add_trace(go.Scatter(
                        x=track_data['x'],
                        y=track_data['y'],
                        mode='lines+markers',
                        name=f'Track {track_id}',
                        marker=dict(color=colors[i]),
                        line=dict(color=colors[i]),
                        showlegend=len(unique_tracks) <= 20
                    ))
        
        # Customize layout
        fig.update_layout(
            title=f"Particle Tracks ({len(unique_tracks)} tracks)",
            xaxis_title="X position",
            yaxis_title="Y position",
            hovermode="closest",
            template="plotly_white"
        )
        
        # Make x and y scales equal
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        return fig
    
    else:
        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Track coloring
        if color_by == 'track_id':
            cmap = plt.get_cmap(colormap)
            colors = [cmap(i / len(unique_tracks)) for i in range(len(unique_tracks))]
            
            for i, track_id in enumerate(unique_tracks):
                track_data = tracks_df[tracks_df['track_id'] == track_id]
                track_data = track_data.sort_values('frame')
                
                ax.plot(track_data['x'], track_data['y'], '-o', 
                        color=colors[i], label=f'Track {track_id}', 
                        markersize=3, linewidth=1)
                
                # Mark starting point
                ax.plot(track_data['x'].iloc[0], track_data['y'].iloc[0], 'o', 
                        markeredgecolor=colors[i], markerfacecolor='none', 
                        markersize=8, markeredgewidth=2)
                
        else:
            # Use another column for coloring if possible
            if color_by in tracks_df.columns and pd.api.types.is_numeric_dtype(tracks_df[color_by]):
                for track_id in unique_tracks:
                    track_data = tracks_df[tracks_df['track_id'] == track_id]
                    track_data = track_data.sort_values('frame')
                    
                    scatter = ax.scatter(track_data['x'], track_data['y'], c=track_data[color_by], 
                                        cmap=colormap, s=15)
                    ax.plot(track_data['x'], track_data['y'], '-', color='gray', 
                            alpha=0.7, linewidth=1)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label=color_by)
            else:
                # Fallback to track_id
                cmap = plt.get_cmap(colormap)
                colors = [cmap(i / len(unique_tracks)) for i in range(len(unique_tracks))]
                
                for i, track_id in enumerate(unique_tracks):
                    track_data = tracks_df[tracks_df['track_id'] == track_id]
                    track_data = track_data.sort_values('frame')
                    
                    ax.plot(track_data['x'], track_data['y'], '-o', 
                            color=colors[i], label=f'Track {track_id}', 
                            markersize=3, linewidth=1)
        
        # Add legend if there aren't too many tracks
        if len(unique_tracks) <= 20:
            ax.legend(fontsize=8, loc='upper right')
            
        ax.set_title(f"Particle Tracks ({len(unique_tracks)} tracks)")
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return fig

def plot_tracks_3d(tracks_df: pd.DataFrame, frame_range: Optional[Tuple[int, int]] = None,
                  color_by: str = 'track_id', colormap: str = 'viridis', 
                  max_tracks: int = 20, pixel_size: float = 0.1,
                  use_true_3d: bool = False) -> go.Figure:
    """
    Create immersive 3D trajectory visualization with interactive controls.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks data with columns: track_id, frame, x, y, [z]
    frame_range : tuple, optional
        Range of frames to include (min, max)
    color_by : str
        Column to use for track coloring ('track_id', 'speed', 'displacement')
    colormap : str
        Plotly colormap name
    max_tracks : int
        Maximum number of tracks to display for performance
    pixel_size : float
        Pixel size in micrometers for unit conversion
    use_true_3d : bool
        Whether to use Z coordinates (True) or time as Z-axis (False)
        
    Returns
    -------
    go.Figure
        Interactive 3D Plotly figure
    """
    # Validate input data
    if tracks_df.empty:
        st.warning("No track data available for 3D visualization.")
        return go.Figure()
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    if missing_columns:
        st.error(f"Missing required columns for 3D plotting: {missing_columns}")
        return go.Figure()
    
    # Filter by frame range if specified
    if frame_range is not None:
        tracks_df = tracks_df[
            (tracks_df['frame'] >= frame_range[0]) & 
            (tracks_df['frame'] <= frame_range[1])
        ]
    
    # Check if we have Z coordinates for true 3D
    has_z = 'z' in tracks_df.columns and use_true_3d
    
    # Limit number of tracks for performance
    unique_tracks = tracks_df['track_id'].unique()
    if len(unique_tracks) > max_tracks:
        selected_tracks = np.random.choice(unique_tracks, max_tracks, replace=False)
        tracks_df = tracks_df[tracks_df['track_id'].isin(selected_tracks)]
        unique_tracks = selected_tracks
    
    # Create 3D figure
    fig = go.Figure()
    
    # Color scheme setup
    if color_by == 'track_id':
        colors = px.colors.qualitative.Set1 * (len(unique_tracks) // len(px.colors.qualitative.Set1) + 1)
    
    # Process each track
    for i, track_id in enumerate(unique_tracks):
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 2:
            continue
        
        # Convert coordinates to micrometers
        x_coords = track_data['x'].values * pixel_size
        y_coords = track_data['y'].values * pixel_size
        
        # Determine Z coordinate
        if has_z:
            z_coords = track_data['z'].values * pixel_size
            z_label = 'Z (μm)'
        else:
            z_coords = track_data['frame'].values
            z_label = 'Time (frames)'
        
        # Calculate color values based on color_by parameter
        if color_by == 'track_id':
            line_color = colors[i % len(colors)]
            marker_color = line_color
        elif color_by == 'speed':
            # Calculate instantaneous speed
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            dz = np.diff(z_coords) if has_z else np.zeros_like(dx)
            dt = np.diff(track_data['frame'].values) if has_z else np.ones_like(dx)
            dt[dt == 0] = 1  # Avoid division by zero
            
            speeds = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            speeds = np.concatenate([[speeds[0]], speeds])  # Prepend first speed
            line_color = speeds
            marker_color = speeds
        elif color_by == 'displacement':
            # Calculate displacement from starting point
            x_start, y_start = x_coords[0], y_coords[0]
            z_start = z_coords[0] if has_z else 0
            
            displacements = np.sqrt(
                (x_coords - x_start)**2 + 
                (y_coords - y_start)**2 + 
                (z_coords - z_start)**2 if has_z else (x_coords - x_start)**2 + (y_coords - y_start)**2
            )
            line_color = displacements
            marker_color = displacements
        else:
            line_color = colors[i % len(colors)]
            marker_color = line_color
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            name=f'Track {track_id}',
            line=dict(
                color=line_color if isinstance(line_color, str) else None,
                colorscale=colormap if not isinstance(line_color, str) else None,
                width=3
            ),
            marker=dict(
                size=3,
                color=marker_color,
                colorscale=colormap if not isinstance(marker_color, str) else None,
                opacity=0.8
            ),
            showlegend=True if i < 10 else False,  # Limit legend entries
            hovertemplate=f'Track {track_id}<br>' +
                         'X: %{x:.2f} μm<br>' +
                         'Y: %{y:.2f} μm<br>' +
                         f'{z_label}: %{{z:.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        # Mark start point
        fig.add_trace(go.Scatter3d(
            x=[x_coords[0]],
            y=[y_coords[0]],
            z=[z_coords[0]],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='diamond',
                opacity=0.9
            ),
            name=f'Start {track_id}',
            showlegend=False,
            hovertemplate=f'Track {track_id} Start<br>' +
                         'X: %{x:.2f} μm<br>' +
                         'Y: %{y:.2f} μm<br>' +
                         f'{z_label}: %{{z:.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        # Mark end point
        fig.add_trace(go.Scatter3d(
            x=[x_coords[-1]],
            y=[y_coords[-1]],
            z=[z_coords[-1]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='square',
                opacity=0.9
            ),
            name=f'End {track_id}',
            showlegend=False,
            hovertemplate=f'Track {track_id} End<br>' +
                         'X: %{x:.2f} μm<br>' +
                         'Y: %{y:.2f} μm<br>' +
                         f'{z_label}: %{{z:.2f}}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout for immersive 3D experience
    fig.update_layout(
        title=f'Immersive 3D Trajectory Visualization ({len(unique_tracks)} tracks)',
        scene=dict(
            xaxis_title='X Position (μm)',
            yaxis_title='Y Position (μm)',
            zaxis_title=z_label,
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='white',
            xaxis=dict(
                backgroundcolor='rgb(240, 240, 240)',
                gridcolor='white',
                showbackground=True,
                zerolinecolor='white'
            ),
            yaxis=dict(
                backgroundcolor='rgb(240, 240, 240)',
                gridcolor='white',
                showbackground=True,
                zerolinecolor='white'
            ),
            zaxis=dict(
                backgroundcolor='rgb(240, 240, 240)',
                gridcolor='white',
                showbackground=True,
                zerolinecolor='white'
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Add annotations for interaction instructions
    fig.add_annotation(
        text="🖱️ Drag to rotate • 🔍 Scroll to zoom • ⌨️ Shift+drag to pan",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        xanchor='center', yanchor='bottom',
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig

def plot_tracks_time_series(tracks_df: pd.DataFrame, time_variable: str = 'frame',
                           y_variables: List[str] = None) -> go.Figure:
    """
    Plot track variables over time as line series.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks data in standard format
    time_variable : str
        Column name for time axis (default: 'frame')
    y_variables : List[str], optional
        List of variables to plot on y-axis
        
    Returns
    -------
    go.Figure
        Time series plot figure
    """
    if tracks_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No track data available",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Time (frame)"
            )
        )
        return fig
    
    # Filter by frame range if provided
    if frame_range is not None:
        tracks_df = tracks_df[(tracks_df['frame'] >= frame_range[0]) & 
                              (tracks_df['frame'] <= frame_range[1])]
    
    # Get unique tracks
    unique_tracks = tracks_df['track_id'].unique()
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Color scale
    if color_by == 'track_id':
        # Use track_id for coloring
        colors = px.colors.sample_colorscale(
            colormap, np.linspace(0, 1, len(unique_tracks))
        )
        
        for i, track_id in enumerate(unique_tracks):
            track_data = tracks_df[tracks_df['track_id'] == track_id]
            track_data = track_data.sort_values('frame')
            
            fig.add_trace(go.Scatter3d(
                x=track_data['x'],
                y=track_data['y'],
                z=track_data['frame'],
                mode='lines+markers',
                name=f'Track {track_id}',
                marker=dict(
                    color=colors[i],
                    size=3
                ),
                line=dict(color=colors[i], width=3),
                showlegend=len(unique_tracks) <= 20  # Only show legend if few tracks
            ))
    else:
        # Use another column for coloring
        if color_by in tracks_df.columns and pd.api.types.is_numeric_dtype(tracks_df[color_by]):
            # Create a continuous color scale
            for track_id in unique_tracks:
                track_data = tracks_df[tracks_df['track_id'] == track_id]
                track_data = track_data.sort_values('frame')
                
                fig.add_trace(go.Scatter3d(
                    x=track_data['x'],
                    y=track_data['y'],
                    z=track_data['frame'],
                    mode='lines+markers',
                    name=f'Track {track_id}',
                    marker=dict(
                        color=track_data[color_by],
                        colorscale=colormap,
                        size=3,
                        showscale=True,
                        colorbar=dict(title=color_by)
                    ),
                    line=dict(color=px.colors.sample_colorscale(
                        colormap, [track_data[color_by].mean()])[0], width=3),
                    showlegend=len(unique_tracks) <= 20
                ))
        else:
            # Use track_id as fallback
            colors = px.colors.sample_colorscale(
                colormap, np.linspace(0, 1, len(unique_tracks))
            )
            
            for i, track_id in enumerate(unique_tracks):
                track_data = tracks_df[tracks_df['track_id'] == track_id]
                track_data = track_data.sort_values('frame')
                
                fig.add_trace(go.Scatter3d(
                    x=track_data['x'],
                    y=track_data['y'],
                    z=track_data['frame'],
                    mode='lines+markers',
                    name=f'Track {track_id}',
                    marker=dict(color=colors[i], size=3),
                    line=dict(color=colors[i], width=3),
                    showlegend=len(unique_tracks) <= 20
                ))
    
    # Customize layout
    fig.update_layout(
        title=f"3D Particle Tracks (Time as Z-axis, {len(unique_tracks)} tracks)",
        scene=dict(
            xaxis_title="X position",
            yaxis_title="Y position",
            zaxis_title="Time (frame)",
            aspectmode='auto'
        ),
        template="plotly_white"
    )
    
    return fig

def plot_track_statistics(track_stats: pd.DataFrame, plot_type: str = 'bar') -> Dict[str, go.Figure]:
    """
    Plot various statistics for tracks.
    
    Parameters
    ----------
    track_stats : pd.DataFrame
        DataFrame with track statistics
    plot_type : str
        Type of plot: 'bar', 'histogram', or 'box'
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if track_stats.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No track statistics available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Track length distribution
    if 'track_length' in track_stats.columns:
        if plot_type == 'histogram':
            fig = px.histogram(
                track_stats, x='track_length', 
                title='Track Length Distribution',
                labels={'track_length': 'Track Length (frames)', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#1f77b4']
            )
        elif plot_type == 'bar':
            value_counts = track_stats['track_length'].value_counts().sort_index()
            fig = px.bar(
                x=value_counts.index, y=value_counts.values,
                title='Track Length Distribution',
                labels={'x': 'Track Length (frames)', 'y': 'Number of Tracks'},
                color_discrete_sequence=['#1f77b4']
            )
        else:  # box plot
            fig = px.box(
                track_stats, y='track_length',
                title='Track Length Distribution',
                labels={'track_length': 'Track Length (frames)'},
                color_discrete_sequence=['#1f77b4']
            )
        
        fig.update_layout(template="plotly_white")
        figures['track_length'] = fig
    
    # Duration distribution
    if 'duration' in track_stats.columns:
        if plot_type == 'histogram':
            fig = px.histogram(
                track_stats, x='duration', 
                title='Track Duration Distribution',
                labels={'duration': 'Duration', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#ff7f0e']
            )
        elif plot_type == 'bar':
            # Group durations into bins
            bins = np.linspace(track_stats['duration'].min(), track_stats['duration'].max(), 15)
            track_stats['duration_bin'] = pd.cut(track_stats['duration'], bins)
            value_counts = track_stats['duration_bin'].value_counts().sort_index()
            
            # Convert bins to strings for plotting
            bin_labels = [f"{b.left:.1f}-{b.right:.1f}" for b in value_counts.index]
            
            fig = px.bar(
                x=bin_labels, y=value_counts.values,
                title='Track Duration Distribution',
                labels={'x': 'Duration Bins', 'y': 'Number of Tracks'},
                color_discrete_sequence=['#ff7f0e']
            )
        else:  # box plot
            fig = px.box(
                track_stats, y='duration',
                title='Track Duration Distribution',
                labels={'duration': 'Duration'},
                color_discrete_sequence=['#ff7f0e']
            )
        
        fig.update_layout(template="plotly_white")
        figures['duration'] = fig
    
    # Displacement distribution
    if 'net_displacement' in track_stats.columns:
        if plot_type == 'histogram':
            fig = px.histogram(
                track_stats, x='net_displacement', 
                title='Net Displacement Distribution',
                labels={'net_displacement': 'Net Displacement', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#2ca02c']
            )
        elif plot_type == 'bar':
            # Group displacements into bins
            bins = np.linspace(0, track_stats['net_displacement'].max(), 15)
            track_stats['displacement_bin'] = pd.cut(track_stats['net_displacement'], bins)
            value_counts = track_stats['displacement_bin'].value_counts().sort_index()
            
            # Convert bins to strings for plotting
            bin_labels = [f"{b.left:.1f}-{b.right:.1f}" for b in value_counts.index]
            
            fig = px.bar(
                x=bin_labels, y=value_counts.values,
                title='Net Displacement Distribution',
                labels={'x': 'Net Displacement Bins', 'y': 'Number of Tracks'},
                color_discrete_sequence=['#2ca02c']
            )
        else:  # box plot
            fig = px.box(
                track_stats, y='net_displacement',
                title='Net Displacement Distribution',
                labels={'net_displacement': 'Net Displacement'},
                color_discrete_sequence=['#2ca02c']
            )
        
        fig.update_layout(template="plotly_white")
        figures['net_displacement'] = fig
    
    # Speed distribution
    if 'mean_speed' in track_stats.columns:
        if plot_type == 'histogram':
            fig = px.histogram(
                track_stats, x='mean_speed', 
                title='Mean Speed Distribution',
                labels={'mean_speed': 'Mean Speed', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#d62728']
            )
        elif plot_type == 'bar':
            # Group speeds into bins
            bins = np.linspace(0, track_stats['mean_speed'].max(), 15)
            track_stats['speed_bin'] = pd.cut(track_stats['mean_speed'], bins)
            value_counts = track_stats['speed_bin'].value_counts().sort_index()
            
            # Convert bins to strings for plotting
            bin_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in value_counts.index]
            
            fig = px.bar(
                x=bin_labels, y=value_counts.values,
                title='Mean Speed Distribution',
                labels={'x': 'Mean Speed Bins', 'y': 'Number of Tracks'},
                color_discrete_sequence=['#d62728']
            )
        else:  # box plot
            fig = px.box(
                track_stats, y='mean_speed',
                title='Mean Speed Distribution',
                labels={'mean_speed': 'Mean Speed'},
                color_discrete_sequence=['#d62728']
            )
        
        fig.update_layout(template="plotly_white")
        figures['mean_speed'] = fig
    
    # Straightness distribution
    if 'straightness' in track_stats.columns:
        if plot_type == 'histogram':
            fig = px.histogram(
                track_stats, x='straightness', 
                title='Straightness Distribution',
                labels={'straightness': 'Straightness (0-1)', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#9467bd']
            )
        elif plot_type == 'bar':
            # Group straightness into bins
            bins = np.linspace(0, 1, 10)
            track_stats['straightness_bin'] = pd.cut(track_stats['straightness'], bins)
            value_counts = track_stats['straightness_bin'].value_counts().sort_index()
            
            # Convert bins to strings for plotting
            bin_labels = [f"{b.left:.1f}-{b.right:.1f}" for b in value_counts.index]
            
            fig = px.bar(
                x=bin_labels, y=value_counts.values,
                title='Straightness Distribution',
                labels={'x': 'Straightness Bins (0-1)', 'y': 'Number of Tracks'},
                color_discrete_sequence=['#9467bd']
            )
        else:  # box plot
            fig = px.box(
                track_stats, y='straightness',
                title='Straightness Distribution',
                labels={'straightness': 'Straightness (0-1)'},
                color_discrete_sequence=['#9467bd']
            )
        
        fig.update_layout(template="plotly_white")
        figures['straightness'] = fig
    
    return figures

# --- MSD and Diffusion Analysis Visualization ---

def plot_msd_curves(msd_data: pd.DataFrame, fit_data: Optional[pd.DataFrame] = None, 
                   individual_tracks: bool = True, average_curve: bool = True,
                   log_scale: bool = False) -> go.Figure:
    """
    Plot MSD curves for tracks.
    
    Parameters
    ----------
    msd_data : pd.DataFrame
        DataFrame with MSD data (track_id, lag_time, msd)
    fit_data : pd.DataFrame, optional
        DataFrame with fit data for MSD curves
    individual_tracks : bool
        Whether to plot individual track curves
    average_curve : bool
        Whether to plot average MSD curve
    log_scale : bool
        Whether to use log-log scale
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot figure object
    """
    if msd_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No MSD data available",
            xaxis_title="Lag Time",
            yaxis_title="MSD"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Get unique tracks
    unique_tracks = msd_data['track_id'].unique()
    colors = px.colors.qualitative.Plotly
    
    # Plot individual curves
    if individual_tracks:
        for i, track_id in enumerate(unique_tracks):
            track_msd = msd_data[msd_data['track_id'] == track_id]
            color_idx = i % len(colors)
            
            fig.add_trace(go.Scatter(
                x=track_msd['lag_time'],
                y=track_msd['msd'],
                mode='markers+lines',
                name=f'Track {track_id}',
                line=dict(color=colors[color_idx], width=1),
                marker=dict(color=colors[color_idx], size=4),
                opacity=0.7,
                showlegend=len(unique_tracks) <= 20  # Only show legend if few tracks
            ))
    
    # Plot average curve
    if average_curve and len(msd_data) > 0:
        # Group by lag_time and calculate mean and std
        grouped = msd_data.groupby('lag_time')
        mean_msd = grouped['msd'].mean()
        std_msd = grouped['msd'].std()
        sem_msd = std_msd / np.sqrt(grouped['msd'].count())
        
        # Add average curve
        fig.add_trace(go.Scatter(
            x=mean_msd.index,
            y=mean_msd.values,
            mode='markers+lines',
            name='Average MSD',
            line=dict(color='black', width=3),
            marker=dict(color='black', size=8)
        ))
        
        # Add error bands
        fig.add_trace(go.Scatter(
            x=np.concatenate([mean_msd.index, mean_msd.index[::-1]]),
            y=np.concatenate([mean_msd + sem_msd, (mean_msd - sem_msd)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,0,0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            name='SEM'
        ))
        
        # Add fit curve if provided
        if fit_data is not None and not fit_data.empty:
            fig.add_trace(go.Scatter(
                x=fit_data['lag_time'],
                y=fit_data['msd_fit'],
                mode='lines',
                name='Fit',
                line=dict(color='red', width=2, dash='dash')
            ))
    
    # Set log scale if requested
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
        title_suffix = " (Log-Log Scale)"
    else:
        title_suffix = ""
    
    # Customize layout
    fig.update_layout(
        title=f"MSD vs. Lag Time{title_suffix}",
        xaxis_title="Lag Time",
        yaxis_title="Mean Squared Displacement",
        template="plotly_white"
    )
    
    return fig

def plot_diffusion_coefficients(diffusion_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot diffusion analysis results.
    
    Parameters
    ----------
    diffusion_results : dict
        Dictionary with diffusion analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not diffusion_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No diffusion analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Get track results
    track_results = diffusion_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Diffusion coefficient distribution
        fig = px.histogram(
            track_results, x='diffusion_coefficient', 
            title='Diffusion Coefficient Distribution',
            labels={
                'diffusion_coefficient': 'Diffusion Coefficient (µm²/s)', 
                'count': 'Number of Tracks'
            },
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['diffusion_coefficients'] = fig
        
        # Add log scale version
        fig_log = px.histogram(
            track_results, x='diffusion_coefficient', 
            title='Diffusion Coefficient Distribution (Log Scale)',
            labels={
                'diffusion_coefficient': 'Diffusion Coefficient (µm²/s)', 
                'count': 'Number of Tracks'
            },
            log_x=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig_log.update_layout(template="plotly_white")
        figures['diffusion_coefficients_log'] = fig_log
        
        # Alpha exponent distribution (if available)
        if 'alpha' in track_results.columns:
            fig = px.histogram(
                track_results, x='alpha', 
                title='Anomalous Diffusion Exponent Distribution',
                labels={'alpha': 'Alpha Exponent', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#ff7f0e']
            )
            # Add reference lines for different diffusion regimes
            fig.add_shape(
                type='line',
                x0=1, y0=0, x1=1, y1=1,
                yref='paper',
                line=dict(color='black', width=2, dash='dash')
            )
            fig.add_annotation(
                x=1, y=0.95, yref='paper',
                text="α=1: Normal Diffusion",
                showarrow=False,
                font=dict(color='black')
            )
            fig.update_layout(template="plotly_white")
            figures['alpha_exponents'] = fig
        
        # Diffusion type pie chart (if available)
        if 'diffusion_type' in track_results.columns:
            type_counts = track_results['diffusion_type'].value_counts()
            fig = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title='Diffusion Types',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(template="plotly_white")
            figures['diffusion_types'] = fig
        
        # Confinement scatter plot (if available)
        if 'confined' in track_results.columns and 'confinement_radius' in track_results.columns:
            confined_tracks = track_results[track_results['confined'] == True]
            unconfined_tracks = track_results[track_results['confined'] == False]
            
            fig = go.Figure()
            
            # Add scatter points for unconfined tracks
            if not unconfined_tracks.empty:
                fig.add_trace(go.Scatter(
                    x=unconfined_tracks['diffusion_coefficient'],
                    y=unconfined_tracks['track_id'],
                    mode='markers',
                    name='Unconfined',
                    marker=dict(
                        color='blue',
                        size=10,
                        symbol='circle'
                    )
                ))
            
            # Add scatter points for confined tracks
            if not confined_tracks.empty:
                fig.add_trace(go.Scatter(
                    x=confined_tracks['diffusion_coefficient'],
                    y=confined_tracks['track_id'],
                    mode='markers',
                    name='Confined',
                    marker=dict(
                        color='red',
                        size=confined_tracks['confinement_radius'] * 10,  # Scale size by confinement radius
                        symbol='circle',
                        line=dict(color='black', width=1)
                    )
                ))
            
            # Customize layout
            fig.update_layout(
                title='Confinement Analysis',
                xaxis_title='Diffusion Coefficient (µm²/s)',
                yaxis_title='Track ID',
                template="plotly_white"
            )
            figures['confinement'] = fig
    
    # Ensemble statistics summary
    ensemble_results = diffusion_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add mean diffusion coefficient
        if 'mean_diffusion_coefficient' in ensemble_results:
            labels.append('Mean Diffusion Coefficient')
            values.append(ensemble_results['mean_diffusion_coefficient'])
        
        # Add median diffusion coefficient
        if 'median_diffusion_coefficient' in ensemble_results:
            labels.append('Median Diffusion Coefficient')
            values.append(ensemble_results['median_diffusion_coefficient'])
        
        # Add standard deviation
        if 'std_diffusion_coefficient' in ensemble_results:
            labels.append('Std Dev of Diffusion Coefficient')
            values.append(ensemble_results['std_diffusion_coefficient'])
        
        # Add mean alpha if available
        if 'mean_alpha' in ensemble_results:
            labels.append('Mean Alpha Exponent')
            values.append(ensemble_results['mean_alpha'])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Diffusion Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Motion Analysis Visualization ---

def plot_motion_analysis(motion_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot motion analysis results.
    
    Parameters
    ----------
    motion_results : dict
        Dictionary with motion analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not motion_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No motion analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Get track results
    track_results = motion_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Speed distribution
        fig = px.histogram(
            track_results, x='mean_speed', 
            title='Mean Speed Distribution',
            labels={'mean_speed': 'Mean Speed (µm/s)', 'count': 'Number of Tracks'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['speed_distribution'] = fig
        
        # Straightness distribution (if available)
        if 'straightness' in track_results.columns:
            fig = px.histogram(
                track_results, x='straightness', 
                title='Straightness Distribution',
                labels={'straightness': 'Straightness (0-1)', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#ff7f0e']
            )
            # Add reference line for perfect straightness
            fig.add_shape(
                type='line',
                x0=1, y0=0, x1=1, y1=1,
                yref='paper',
                line=dict(color='black', width=2, dash='dash')
            )
            fig.update_layout(template="plotly_white")
            figures['straightness'] = fig
        
        # Direction correlation (if available)
        if 'direction_correlation' in track_results.columns:
            fig = px.histogram(
                track_results, x='direction_correlation', 
                title='Direction Correlation Distribution',
                labels={'direction_correlation': 'Direction Correlation', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(template="plotly_white")
            figures['direction_correlation'] = fig
        
        # Motion type pie chart (if available)
        if 'motion_type' in track_results.columns:
            type_counts = track_results['motion_type'].value_counts()
            fig = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title='Motion Types',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(template="plotly_white")
            figures['motion_types'] = fig
        
        # Speed vs. straightness scatter plot
        if 'straightness' in track_results.columns:
            fig = px.scatter(
                track_results, 
                x='mean_speed', 
                y='straightness',
                title='Speed vs. Straightness',
                labels={
                    'mean_speed': 'Mean Speed (µm/s)', 
                    'straightness': 'Straightness (0-1)'
                },
                color='motion_type' if 'motion_type' in track_results.columns else None,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(template="plotly_white")
            figures['speed_vs_straightness'] = fig
    
    # Velocity autocorrelation (if available)
    if track_results is not None and not track_results.empty:
        if 'velocity_autocorr' in track_results.columns and not track_results['velocity_autocorr'].isna().all():
            fig_autocorr = go.Figure()
            has_autocorr_data = False

            # Vectorized processing of track results
            valid_tracks = track_results.dropna(subset=['velocity_autocorr'])
            
            for idx in valid_tracks.index:
                track = valid_tracks.loc[idx]
                autocorr_val = track.get('velocity_autocorr')
                
                skip_plotting = False
                if isinstance(autocorr_val, float) and pd.isna(autocorr_val):
                    skip_plotting = True
                elif isinstance(autocorr_val, (list, np.ndarray)):
                    if len(autocorr_val) == 0:
                        skip_plotting = True
                    else:
                        try:
                            # Ensure it's a numpy array of float for np.isnan and .all()
                            autocorr_np = np.asarray(autocorr_val, dtype=float)
                            if np.all(np.isnan(autocorr_np)):
                                skip_plotting = True
                        except ValueError:
                            skip_plotting = True 
                elif autocorr_val is None:
                    skip_plotting = True
                
                if skip_plotting:
                    continue

                autocorr = autocorr_val 
                if isinstance(autocorr, str):
                    try:
                        # Use ast.literal_eval for safety
                        import ast
                        autocorr = ast.literal_eval(autocorr)
                    except (ValueError, SyntaxError):
                        continue 

                if not isinstance(autocorr, (list, np.ndarray)) or len(autocorr) == 0:
                    continue
                
                # Further check: ensure autocorr contains numeric data
                try:
                    autocorr_numeric = np.asarray(autocorr, dtype=float)
                    if np.all(np.isnan(autocorr_numeric)):
                        continue
                except ValueError:
                    continue

                lag_times = np.arange(len(autocorr_numeric))
                
                fig_autocorr.add_trace(go.Scatter(
                    x=lag_times,
                    y=autocorr_numeric,
                    mode='lines',
                    name=f"Track {track.get('track_id', 'Unknown')}",
                    opacity=0.5,
                    showlegend=False
                ))
                has_autocorr_data = True
            
            if has_autocorr_data:
                max_lag_overall = 0
                # Determine max_lag based on plotted data for the 1/e line
                if fig_autocorr.data:
                    for trace in fig_autocorr.data:
                        if trace.x is not None and len(trace.x) > 0:
                             max_lag_overall = max(max_lag_overall, max(trace.x))
                if max_lag_overall == 0: 
                    max_lag_overall = 10

                fig_autocorr.add_shape(
                    type='line',
                    x0=0, y0=1/np.e, x1=max_lag_overall,
                    y1=1/np.e,
                    line=dict(color='red', width=2, dash='dash')
                )
                
                fig_autocorr.add_annotation(
                    x=max_lag_overall/2,
                    y=1/np.e,
                    text="1/e",
                    showarrow=False,
                    yshift=10,
                    font=dict(color='red')
                )
                
                fig_autocorr.update_layout(
                    title='Velocity Autocorrelation',
                    xaxis_title='Lag Time',
                    yaxis_title='Autocorrelation',
                    template="plotly_white"
                )
                figures['velocity_autocorrelation'] = fig_autocorr
    
    # Ensemble statistics summary
    ensemble_results = motion_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add mean speed
        if 'mean_speed' in ensemble_results:
            labels.append('Mean Speed')
            values.append(ensemble_results['mean_speed'])
        
        # Add median speed
        if 'median_speed' in ensemble_results:
            labels.append('Median Speed')
            values.append(ensemble_results['median_speed'])
        
        # Add mean straightness if available
        if 'mean_straightness' in ensemble_results:
            labels.append('Mean Straightness')
            values.append(ensemble_results['mean_straightness'])
        
        # Add motion type fractions if available
        for key in ensemble_results:
            if key.endswith('_fraction') and not key.startswith('no_correlation'):
                labels.append(key.replace('_fraction', '').replace('_', ' ').title())
                values.append(ensemble_results[key])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Motion Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Clustering Analysis Visualization ---

def plot_spatial_clustering(clustering_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot clustering analysis results.
    
    Parameters
    ----------
    clustering_results : dict
        Dictionary with clustering analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not clustering_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No clustering analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Frame results (cluster information for each frame)
    frame_results = clustering_results.get('frame_results', [])
    
    if frame_results:
        # Number of clusters over time
        frames = [fr['frame'] for fr in frame_results]
        n_clusters = [fr['n_clusters'] for fr in frame_results]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames,
            y=n_clusters,
            mode='markers+lines',
            name='Number of Clusters',
            marker=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Number of Clusters Over Time',
            xaxis_title='Frame',
            yaxis_title='Number of Clusters',
            template="plotly_white"
        )
        figures['clusters_over_time'] = fig
        
        # Clustered fraction over time
        clustered_fractions = [fr['clustered_fraction'] for fr in frame_results]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames,
            y=clustered_fractions,
            mode='markers+lines',
            name='Clustered Fraction',
            marker=dict(color='orange')
        ))
        
        fig.update_layout(
            title='Clustered Fraction Over Time',
            xaxis_title='Frame',
            yaxis_title='Fraction of Points in Clusters',
            template="plotly_white"
        )
        figures['clustered_fraction'] = fig
        
        # Cluster size over time
        mean_cluster_sizes = [fr['mean_cluster_size'] for fr in frame_results]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames,
            y=mean_cluster_sizes,
            mode='markers+lines',
            name='Mean Cluster Size',
            marker=dict(color='green')
        ))
        
        fig.update_layout(
            title='Mean Cluster Size Over Time',
            xaxis_title='Frame',
            yaxis_title='Mean Cluster Size (points)',
            template="plotly_white"
        )
        figures['cluster_size'] = fig
        
        # Visualize clusters for a specific frame
        # Choose the frame with the most clusters
        if frame_results:
            max_cluster_frame = max(frame_results, key=lambda x: x['n_clusters'])
            max_cluster_idx = frame_results.index(max_cluster_frame)
            
            if 'cluster_stats' in max_cluster_frame and not max_cluster_frame['cluster_stats'].empty:
                cluster_stats = max_cluster_frame['cluster_stats']
                
                fig = go.Figure()
                
                # Plot each cluster as a circle using vectorized approach
                if not cluster_stats.empty:
                    fig.add_trace(go.Scatter(
                        x=cluster_stats['centroid_x'],
                        y=cluster_stats['centroid_y'],
                        mode='markers',
                        marker=dict(
                            size=cluster_stats.get('size', 10),
                            color=cluster_stats.index,
                            colorscale='viridis'
                        ),
                        text=[f"Cluster {i}" for i in cluster_stats.index],
                        name="Clusters"
                    ))
                
                fig.update_layout(
                    title=f"Clusters at Frame {max_cluster_frame['frame']}",
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    template="plotly_white"
                )
                
                # Make axes equal scale
                fig.update_layout(
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1
                    )
                )
                
                figures['frame_clusters'] = fig
    
    # Cluster tracks
    cluster_tracks = clustering_results.get('cluster_tracks')
    
    if cluster_tracks is not None and not cluster_tracks.empty:
        # Track clusters over time
        fig = go.Figure()
        
        # Get unique cluster track IDs
        unique_cluster_tracks = cluster_tracks['cluster_track_id'].unique()
        colors = px.colors.qualitative.Plotly
        
        for i, cluster_id in enumerate(unique_cluster_tracks):
            track_data = cluster_tracks[cluster_tracks['cluster_track_id'] == cluster_id]
            color_idx = i % len(colors)
            
            fig.add_trace(go.Scatter(
                x=track_data['frame'],
                y=track_data['n_points'],
                mode='markers+lines',
                name=f'Cluster {cluster_id}',
                marker=dict(color=colors[color_idx]),
                line=dict(color=colors[color_idx])
            ))
        
        fig.update_layout(
            title='Cluster Size Over Time',
            xaxis_title='Frame',
            yaxis_title='Number of Points in Cluster',
            template="plotly_white"
        )
        figures['cluster_tracks'] = fig
        
        # Cluster movement visualization
        fig = go.Figure()
        
        for i, cluster_id in enumerate(unique_cluster_tracks):
            track_data = cluster_tracks[cluster_tracks['cluster_track_id'] == cluster_id]
            
            if len(track_data) < 2:
                continue
                
            color_idx = i % len(colors)
            
            fig.add_trace(go.Scatter(
                x=track_data['centroid_x'],
                y=track_data['centroid_y'],
                mode='markers+lines',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    color=colors[color_idx],
                    size=track_data['n_points'] / track_data['n_points'].max() * 20,  # Size by number of points
                    line=dict(width=1, color='black')
                ),
                line=dict(color=colors[color_idx])
            ))
        
        fig.update_layout(
            title='Cluster Movement',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            template="plotly_white"
        )
        
        # Make axes equal scale
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        figures['cluster_movement'] = fig
    
    # Cluster dynamics
    cluster_dynamics = clustering_results.get('cluster_dynamics')
    
    if cluster_dynamics is not None and not cluster_dynamics.empty:
        # Histogram of cluster durations
        fig = px.histogram(
            cluster_dynamics, x='duration',
            title='Cluster Duration Distribution',
            labels={'duration': 'Duration (frames)', 'count': 'Number of Clusters'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['cluster_duration'] = fig
        
        # Histogram of mean displacements
        fig = px.histogram(
            cluster_dynamics, x='mean_displacement',
            title='Cluster Displacement Distribution',
            labels={'mean_displacement': 'Mean Displacement (µm)', 'count': 'Number of Clusters'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(template="plotly_white")
        figures['cluster_displacement'] = fig
        
        # Stability pie chart
        stable_counts = cluster_dynamics['stable'].value_counts()
        fig = px.pie(
            values=stable_counts.values, 
            names=['Unstable', 'Stable'] if False in stable_counts.index else ['Stable'],
            title='Cluster Stability',
            color_discrete_sequence=['#d62728', '#2ca02c']
        )
        fig.update_layout(template="plotly_white")
        figures['cluster_stability'] = fig
    
    # Ensemble statistics summary
    ensemble_results = clustering_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add key metrics
        if 'mean_n_clusters' in ensemble_results:
            labels.append('Mean # of Clusters')
            values.append(ensemble_results['mean_n_clusters'])
        
        if 'max_n_clusters' in ensemble_results:
            labels.append('Max # of Clusters')
            values.append(ensemble_results['max_n_clusters'])
        
        if 'mean_clustered_fraction' in ensemble_results:
            labels.append('Mean Clustered Fraction')
            values.append(ensemble_results['mean_clustered_fraction'])
        
        if 'mean_cluster_size' in ensemble_results:
            labels.append('Mean Cluster Size')
            values.append(ensemble_results['mean_cluster_size'])
        
        if 'n_cluster_tracks' in ensemble_results:
            labels.append('# of Cluster Tracks')
            values.append(ensemble_results['n_cluster_tracks'])
        
        if 'mean_cluster_duration' in ensemble_results:
            labels.append('Mean Cluster Duration')
            values.append(ensemble_results['mean_cluster_duration'])
        
        if 'stable_cluster_fraction' in ensemble_results:
            labels.append('Stable Cluster Fraction')
            values.append(ensemble_results['stable_cluster_fraction'])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Clustering Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Dwell Time Analysis Visualization ---

def plot_dwell_time_analysis(dwell_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot dwell time analysis results.
    
    Parameters
    ----------
    dwell_results : dict
        Dictionary with dwell time analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not dwell_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No dwell time analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Dwell events
    dwell_events = dwell_results.get('dwell_events')
    
    if dwell_events is not None and not dwell_events.empty:
        # Histogram of dwell times
        fig = px.histogram(
            dwell_events, x='dwell_time',
            title='Dwell Time Distribution',
            labels={'dwell_time': 'Dwell Time', 'count': 'Number of Events'},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Add exponential fit if there are enough data points
        if len(dwell_events) > 10:
            from scipy.stats import expon
            
            # Fit exponential distribution
            dwell_times = dwell_events['dwell_time'].values
            loc, scale = expon.fit(dwell_times)
            
            # Create x values for fit line
            x = np.linspace(min(dwell_times), max(dwell_times), 100)
            y = expon.pdf(x, loc=loc, scale=scale) * len(dwell_times) * (max(dwell_times) - min(dwell_times)) / 20
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'Exp Fit (τ={scale:.2f})',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(template="plotly_white")
        figures['dwell_times'] = fig
        
        # Histogram of dwell times (log scale)
        fig = px.histogram(
            dwell_events, x='dwell_time',
            title='Dwell Time Distribution (Log Scale)',
            labels={'dwell_time': 'Dwell Time', 'count': 'Number of Events'},
            log_x=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['dwell_times_log'] = fig
        
        # Spatial distribution of dwell events
        fig = go.Figure()
        
        # Get unique regions
        regions = dwell_events['region_id'].unique()
        colors = px.colors.qualitative.Plotly
        
        for i, region_id in enumerate(regions):
            region_events = dwell_events[dwell_events['region_id'] == region_id]
            color_idx = i % len(colors)
            
            fig.add_trace(go.Scatter(
                x=region_events['mean_x'],
                y=region_events['mean_y'],
                mode='markers',
                name=f'Region {region_id}',
                marker=dict(
                    color=colors[color_idx],
                    size=region_events['dwell_time'] / region_events['dwell_time'].max() * 20,  # Size by dwell time
                    line=dict(width=1, color='black')
                )
            ))
        
        fig.update_layout(
            title='Spatial Distribution of Dwell Events',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            template="plotly_white"
        )
        
        # Make axes equal scale
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        figures['dwell_spatial'] = fig
        
        # Dwell events over time
        fig = px.scatter(
            dwell_events, x='start_frame', y='track_id',
            color='dwell_time', color_continuous_scale='Viridis',
            size='dwell_time', size_max=20,
            title='Dwell Events Over Time',
            labels={
                'start_frame': 'Start Frame',
                'track_id': 'Track ID',
                'dwell_time': 'Dwell Time'
            }
        )
        fig.update_layout(template="plotly_white")
        figures['dwell_timeline'] = fig
    
    # Track results
    track_results = dwell_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Filter to tracks with at least one dwell event
        tracks_with_dwell = track_results[track_results['n_dwell_events'] > 0]
        
        if not tracks_with_dwell.empty:
            # Histogram of dwell events per track
            fig = px.histogram(
                tracks_with_dwell, x='n_dwell_events',
                title='Dwell Events per Track',
                labels={'n_dwell_events': 'Number of Dwell Events', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(template="plotly_white")
            figures['events_per_track'] = fig
            
            # Histogram of dwell fractions
            fig = px.histogram(
                tracks_with_dwell, x='dwell_fraction',
                title='Dwell Fraction Distribution',
                labels={'dwell_fraction': 'Dwell Fraction (0-1)', 'count': 'Number of Tracks'},
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(template="plotly_white")
            figures['dwell_fraction'] = fig
    
    # Ensemble statistics summary
    ensemble_results = dwell_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add key metrics
        if 'mean_dwell_time' in ensemble_results:
            labels.append('Mean Dwell Time')
            values.append(ensemble_results['mean_dwell_time'])
        
        if 'median_dwell_time' in ensemble_results:
            labels.append('Median Dwell Time')
            values.append(ensemble_results['median_dwell_time'])
        
        if 'mean_dwells_per_track' in ensemble_results:
            labels.append('Mean Dwells per Track')
            values.append(ensemble_results['mean_dwells_per_track'])
        
        if 'mean_dwell_fraction' in ensemble_results:
            labels.append('Mean Dwell Fraction')
            values.append(ensemble_results['mean_dwell_fraction'])
        
        # Add region-specific stats
        for key in ensemble_results:
            if key.startswith('region') and key.endswith('mean_dwell_time'):
                region_id = key.split('_')[0].replace('region', '')
                labels.append(f'Region {region_id} Mean Dwell')
                values.append(ensemble_results[key])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Dwell Time Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Gel Structure Analysis Visualization ---

def plot_gel_structure_analysis(gel_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot gel structure analysis results.
    
    Parameters
    ----------
    gel_results : dict
        Dictionary with gel structure analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not gel_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No gel structure analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Confined regions
    confined_regions = gel_results.get('confined_regions')
    
    if confined_regions is not None and not confined_regions.empty:
        # Histogram of confinement radii
        fig = px.histogram(
            confined_regions, x='radius',
            title='Confinement Radius Distribution',
            labels={'radius': 'Confinement Radius (µm)', 'count': 'Number of Regions'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['confinement_radii'] = fig
        
        # Spatial distribution of confined regions
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=confined_regions['center_x'],
            y=confined_regions['center_y'],
            mode='markers',
            marker=dict(
                color=confined_regions['diffusion_coeff'],
                colorscale='Viridis',
                colorbar=dict(title='Diffusion Coefficient (µm²/s)'),
                size=confined_regions['radius'] * 20,  # Size by radius
                line=dict(width=1, color='black')
            )
        ))
        
        fig.update_layout(
            title='Spatial Distribution of Confined Regions',
            xaxis_title='X Position (µm)',
            yaxis_title='Y Position (µm)',
            template="plotly_white"
        )
        
        # Make axes equal scale
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        figures['confined_spatial'] = fig
        
        # Scatter plot: Diffusion coefficient vs. confinement radius
        fig = px.scatter(
            confined_regions, x='diffusion_coeff', y='radius',
            title='Diffusion Coefficient vs. Confinement Radius',
            labels={
                'diffusion_coeff': 'Diffusion Coefficient (µm²/s)',
                'radius': 'Confinement Radius (µm)'
            },
            color='track_id',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(template="plotly_white")
        figures['diffusion_vs_radius'] = fig
    
    # Track results
    track_results = gel_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Histogram of confined regions per track
        fig = px.histogram(
            track_results, x='n_confined_regions',
            title='Confined Regions per Track',
            labels={'n_confined_regions': 'Number of Confined Regions', 'count': 'Number of Tracks'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(template="plotly_white")
        figures['regions_per_track'] = fig
    
    # Mesh statistics visualization
    mesh_size = gel_results.get('mesh_size')
    mesh_heterogeneity = gel_results.get('mesh_heterogeneity')
    pore_distribution = gel_results.get('pore_distribution')
    
    if mesh_size is not None and mesh_heterogeneity is not None:
        # Create summary bar chart
        labels = ['Mesh Size', 'Mesh Heterogeneity']
        values = [mesh_size, mesh_heterogeneity]
        
        if pore_distribution is not None:
            for key, value in pore_distribution.items():
                if key != 'n_clusters':  # Skip number of clusters for this chart
                    labels.append(key.replace('_', ' ').title())
                    values.append(value)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Gel Structure Statistics',
            xaxis_title='Statistic',
            yaxis_title='Value',
            template="plotly_white"
        )
        figures['mesh_stats'] = fig
        
        # Create mesh visualization
        if confined_regions is not None and not confined_regions.empty:
            fig = go.Figure()
            
            # Plot mesh as a network of pores
            for idx in confined_regions.index:
                region = confined_regions.loc[idx]
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=region['center_x'] - region['radius'],
                    y0=region['center_y'] - region['radius'],
                    x1=region['center_x'] + region['radius'],
                    y1=region['center_y'] + region['radius'],
                    line_color="rgba(100, 100, 100, 0.2)",
                    fillcolor="rgba(100, 100, 250, 0.1)"
                )
            
            # Then add all pore centers as points
            fig.add_trace(go.Scatter(
                x=confined_regions['center_x'],
                y=confined_regions['center_y'],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=6
                ),
                name='Pore Centers'
            ))
            
            fig.update_layout(
                title='Gel Mesh Visualization',
                xaxis_title='X Position (µm)',
                yaxis_title='Y Position (µm)',
                template="plotly_white",
                showlegend=False
            )
            
            # Make axes equal scale
            fig.update_layout(
                yaxis=dict(
                    scaleanchor="x",
                    scaleratio=1
                )
            )
            
            figures['mesh_visualization'] = fig
    
    # Ensemble statistics summary
    ensemble_results = gel_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add key metrics
        if 'mean_confinement_radius' in ensemble_results:
            labels.append('Mean Confinement Radius')
            values.append(ensemble_results['mean_confinement_radius'])
        
        if 'median_confinement_radius' in ensemble_results:
            labels.append('Median Confinement Radius')
            values.append(ensemble_results['median_confinement_radius'])
        
        if 'std_confinement_radius' in ensemble_results:
            labels.append('Std Dev of Confinement Radius')
            values.append(ensemble_results['std_confinement_radius'])
        
        if 'mesh_size' in ensemble_results:
            labels.append('Mesh Size')
            values.append(ensemble_results['mesh_size'])
        
        if 'mesh_heterogeneity' in ensemble_results:
            labels.append('Mesh Heterogeneity')
            values.append(ensemble_results['mesh_heterogeneity'])
        
        if 'mean_nn_distance' in ensemble_results:
            labels.append('Mean Nearest Neighbor Distance')
            values.append(ensemble_results['mean_nn_distance'])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Gel Structure Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Diffusion Population Analysis Visualization ---

def plot_diffusion_population_analysis(pop_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot diffusion population analysis results.
    
    Parameters
    ----------
    pop_results : dict
        Dictionary with diffusion population analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not pop_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No diffusion population analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Raw diffusion data
    raw_data = pop_results.get('raw_diffusion_data')
    
    if raw_data is not None and not raw_data.empty:
        # Histogram of diffusion coefficients (log scale)
        fig = px.histogram(
            raw_data, x='diffusion_coeff',
            title='Diffusion Coefficient Distribution (Log Scale)',
            labels={'diffusion_coeff': 'Diffusion Coefficient (µm²/s)', 'count': 'Number of Tracks'},
            log_x=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['diffusion_coeff_log'] = fig
    
    # Population results
    populations = pop_results.get('populations')
    track_assignments = pop_results.get('track_assignments')
    
    if populations is not None and not populations.empty and track_assignments is not None and not track_assignments.empty:
        # Histogram of diffusion coefficients with population components
        fig = go.Figure()
        
        # Add histogram of all tracks
        log_D = np.log10(track_assignments['diffusion_coeff'])
        fig.add_trace(go.Histogram(
            x=log_D,
            name='All Tracks',
            opacity=0.5,
            marker_color='gray'
        ))
        
        # Add normal distributions for each population
        x = np.linspace(min(log_D), max(log_D), 100)
        
        for idx in populations.index:
            pop = populations.loc[idx]
            # Calculate normal distribution based on population parameters
            mean = pop['log_mean']
            std = pop['log_std']
            weight = pop['weight']
            
            # Calculate PDF
            from scipy.stats import norm
            pdf = norm.pdf(x, mean, std)
            
            # Scale PDF to match histogram
            scale_factor = len(track_assignments) * (max(log_D) - min(log_D)) / 10 * weight
            y = pdf * scale_factor
            
            # Add distribution curve
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'Population {pop["population_id"]} (D={10**mean:.2e} µm²/s)',
                line=dict(width=2)
            ))
        
        # Add marginal rug plot showing individual tracks
        fig.add_trace(go.Scatter(
            x=log_D,
            y=np.zeros(len(log_D)),
            mode='markers',
            marker=dict(
                color='black',
                symbol='line-ns',
                size=8,
                opacity=0.5
            ),
            name='Individual Tracks'
        ))
        
        fig.update_layout(
            title='Diffusion Coefficient Distribution with Population Components',
            xaxis_title='Log10(Diffusion Coefficient)',
            yaxis_title='Number of Tracks',
            template="plotly_white",
            bargap=0.1
        )
        
        figures['population_distribution'] = fig
        
        # Scatter plot of diffusion coefficient vs. alpha (if available)
        if 'alpha' in track_assignments.columns:
            fig = px.scatter(
                track_assignments, x='diffusion_coeff', y='alpha',
                color='population_id', color_discrete_sequence=px.colors.qualitative.Plotly,
                log_x=True,
                title='Diffusion Coefficient vs. Alpha by Population',
                labels={
                    'diffusion_coeff': 'Diffusion Coefficient (µm²/s)',
                    'alpha': 'Alpha Exponent',
                    'population_id': 'Population'
                }
            )
            
            # Add reference line for normal diffusion
            fig.add_shape(
                type='line',
                x0=min(track_assignments['diffusion_coeff']),
                y0=1,
                x1=max(track_assignments['diffusion_coeff']),
                y1=1,
                line=dict(color='black', width=2, dash='dash')
            )
            
            fig.update_layout(template="plotly_white")
            figures['diffusion_vs_alpha'] = fig
        
        # Population summary chart
        fig = px.bar(
            populations, x='population_id', y='weight',
            title='Population Weights',
            labels={'population_id': 'Population', 'weight': 'Weight (Fraction)'},
            color='population_id',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Add text labels showing diffusion coefficients using vectorized approach
        for idx in populations.index:
            pop = populations.loc[idx]
            fig.add_annotation(
                x=pop['population_id'],
                y=pop['weight'],
                text=f"D={10**pop['log_mean']:.2e}",
                showarrow=False,
                yshift=10,
                font=dict(color='black')
            )
        
        fig.update_layout(template="plotly_white")
        figures['population_weights'] = fig
    
    return figures

# --- Crowding Analysis Visualization ---

def plot_crowding_analysis(crowding_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot crowding analysis results.
    
    Parameters
    ----------
    crowding_results : dict
        Dictionary with crowding analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not crowding_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No crowding analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Crowding data
    crowding_data = crowding_results.get('crowding_data')
    
    if crowding_data is not None and not crowding_data.empty:
        # Histogram of local densities
        fig = px.histogram(
            crowding_data, x='local_density',
            title='Local Density Distribution',
            labels={'local_density': 'Local Density (particles/µm²)', 'count': 'Number of Points'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['density_distribution'] = fig
        
        # Spatial distribution of densities (heatmap)
        fig = px.density_heatmap(
            crowding_data, x='x', y='y', z='local_density',
            title='Spatial Distribution of Local Densities',
            labels={
                'x': 'X Position (µm)',
                'y': 'Y Position (µm)',
                'local_density': 'Local Density (particles/µm²)'
            },
            color_continuous_scale='Viridis'
        )
        
        # Make axes equal scale
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            ),
            template="plotly_white"
        )
        
        figures['density_heatmap'] = fig
    
    # Track results
    track_results = crowding_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Histogram of density-displacement correlations
        fig = px.histogram(
            track_results, x='density_displacement_correlation',
            title='Density-Displacement Correlation Distribution',
            labels={
                'density_displacement_correlation': 'Correlation Coefficient',
                'count': 'Number of Tracks'
            },
            color_discrete_sequence=['#ff7f0e']
        )
        
        # Add reference line at zero
        fig.add_shape(
            type='line',
            x0=0, y0=0, x1=0, y1=1,
            yref='paper',
            line=dict(color='black', width=2, dash='dash')
        )
        
        fig.update_layout(template="plotly_white")
        figures['density_correlation'] = fig
        
        # Scatter plot: density vs displacement
        fig = px.scatter(
            track_results, x='mean_density', y='mean_displacement',
            title='Mean Density vs. Mean Displacement',
            labels={
                'mean_density': 'Mean Local Density (particles/µm²)',
                'mean_displacement': 'Mean Displacement (µm)'
            },
            color='density_displacement_correlation',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            hover_data=['track_id', 'density_displacement_correlation']
        )
        fig.update_layout(template="plotly_white")
        figures['density_vs_displacement'] = fig
        
        # Grouped displacement comparison
        # Create dataframe for grouped plot
        group_data = []
        
        # Vectorized processing of track results
        valid_tracks = track_results.dropna(subset=['low_density_median_disp'])
        
        for idx in valid_tracks.index:
            track = valid_tracks.loc[idx]
            group_data.append({
                'track_id': track['track_id'],
                    'density_group': 'Low Density',
                    'displacement': track['low_density_median_disp']
                })
            
            if not np.isnan(track['med_density_median_disp']):
                group_data.append({
                    'track_id': track['track_id'],
                    'density_group': 'Medium Density',
                    'displacement': track['med_density_median_disp']
                })
            
            if not np.isnan(track['high_density_median_disp']):
                group_data.append({
                    'track_id': track['track_id'],
                    'density_group': 'High Density',
                    'displacement': track['high_density_median_disp']
                })
        
        if group_data:
            group_df = pd.DataFrame(group_data)
            
            fig = px.box(
                group_df, x='density_group', y='displacement',
                title='Displacement by Density Group',
                labels={
                    'density_group': 'Density Group',
                    'displacement': 'Displacement (µm)',
                },
                category_orders={'density_group': ['Low Density', 'Medium Density', 'High Density']},
                color='density_group',
                color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728']
            )
            fig.update_layout(template="plotly_white")
            figures['density_group_comparison'] = fig
    
    # Ensemble statistics summary
    ensemble_results = crowding_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add key metrics
        if 'mean_density' in ensemble_results:
            labels.append('Mean Density')
            values.append(ensemble_results['mean_density'])
        
        if 'median_density' in ensemble_results:
            labels.append('Median Density')
            values.append(ensemble_results['median_density'])
        
        if 'density_cv' in ensemble_results:
            labels.append('Density CV')
            values.append(ensemble_results['density_cv'])
        
        if 'mean_correlation' in ensemble_results:
            labels.append('Mean Correlation')
            values.append(ensemble_results['mean_correlation'])
        
        if 'mean_crowding_effect' in ensemble_results:
            labels.append('Mean Crowding Effect')
            values.append(ensemble_results['mean_crowding_effect'])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Crowding Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
        
        # Correlation type distribution
        corr_types = []
        corr_counts = []
        
        if 'negative_correlation_count' in ensemble_results:
            corr_types.append('Negative')
            corr_counts.append(ensemble_results['negative_correlation_count'])
        
        if 'no_correlation_count' in ensemble_results:
            corr_types.append('No Correlation')
            corr_counts.append(ensemble_results['no_correlation_count'])
        
        if 'positive_correlation_count' in ensemble_results:
            corr_types.append('Positive')
            corr_counts.append(ensemble_results['positive_correlation_count'])
        
        if corr_types and corr_counts:
            fig = go.Figure(data=[go.Pie(
                labels=corr_types,
                values=corr_counts,
                hole=0.4,
                marker_colors=['#d62728', '#7f7f7f', '#2ca02c']
            )])
            
            fig.update_layout(
                title='Correlation Type Distribution',
                template="plotly_white"
            )
            
            figures['correlation_types'] = fig
    
    return figures

# --- Active Transport Analysis Visualization ---

def plot_active_transport_analysis(transport_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot active transport analysis results.
    
    Parameters
    ----------
    transport_results : dict
        Dictionary with active transport analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not transport_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No active transport analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Directed segments
    directed_segments = transport_results.get('directed_segments')
    
    if directed_segments is not None and not directed_segments.empty:
        # Histogram of segment speeds
        fig = px.histogram(
            directed_segments, x='speed',
            title='Directed Segment Speed Distribution',
            labels={'speed': 'Speed (µm/s)', 'count': 'Number of Segments'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['segment_speeds'] = fig
        
        # Histogram of segment durations
        fig = px.histogram(
            directed_segments, x='duration',
            title='Directed Segment Duration Distribution',
            labels={'duration': 'Duration (s)', 'count': 'Number of Segments'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(template="plotly_white")
        figures['segment_durations'] = fig
        
        # Polar histogram of segment angles
        fig = px.histogram_polar(
            directed_segments, r='speed', theta='angle',
            range_theta=[-np.pi, np.pi],
            title='Directed Motion Directionality',
            color_discrete_sequence=['#2ca02c']
        )
        fig.update_layout(template="plotly_white")
        figures['segment_directions'] = fig
        
        # Map of directed segments
        fig = go.Figure()
        
        # Vectorized processing of directed segments
        for idx in directed_segments.index:
            segment = directed_segments.loc[idx]
            # Draw arrow for each segment
            fig.add_trace(go.Scatter(
                x=[segment['start_x'], segment['end_x']],
                y=[segment['start_y'], segment['end_y']],
                mode='lines',
                line=dict(
                    color='blue',
                    width=2
                ),
                showlegend=False
            ))
            
            # Add arrow head (simple triangle)
            # Calculate direction vector
            dx = segment['end_x'] - segment['start_x']
            dy = segment['end_y'] - segment['start_y']
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize
                dx /= length
                dy /= length
                
                # Calculate perpendicular vector
                px, py = -dy, dx
                
                # Create arrowhead
                ax = segment['end_x']
                ay = segment['end_y']
                
                # Arrow head points
                arrow_size = 0.3  # Size of arrow head
                p1 = (ax - arrow_size * dx + arrow_size * 0.5 * px, ay - arrow_size * dy + arrow_size * 0.5 * py)
                p2 = (ax - arrow_size * dx - arrow_size * 0.5 * px, ay - arrow_size * dy - arrow_size * 0.5 * py)
                
                fig.add_trace(go.Scatter(
                    x=[ax, p1[0], p2[0], ax],
                    y=[ay, p1[1], p2[1], ay],
                    fill="toself",
                    mode='lines',
                    line=dict(color='blue'),
                    showlegend=False
                ))
        
        fig.update_layout(
            title='Directed Motion Segments',
            xaxis_title='X Position (µm)',
            yaxis_title='Y Position (µm)',
            template="plotly_white"
        )
        
        # Make axes equal scale
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        figures['segment_map'] = fig
        
        # Speed vs duration scatter plot
        fig = px.scatter(
            directed_segments, x='duration', y='speed',
            title='Segment Speed vs. Duration',
            labels={
                'duration': 'Duration (s)',
                'speed': 'Speed (µm/s)'
            },
            color='straightness',
            color_continuous_scale='Viridis',
            hover_data=['track_id', 'displacement']
        )
        fig.update_layout(template="plotly_white")
        figures['speed_vs_duration'] = fig
    
    # Track results
    track_results = transport_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Histogram of directed fraction
        fig = px.histogram(
            track_results, x='directed_fraction',
            title='Directed Motion Fraction Distribution',
            labels={'directed_fraction': 'Directed Fraction (0-1)', 'count': 'Number of Tracks'},
            color_discrete_sequence=['#9467bd']
        )
        fig.update_layout(template="plotly_white")
        figures['directed_fraction'] = fig
        
        # Histogram of directed segments per track
        fig = px.histogram(
            track_results, x='n_directed_segments',
            title='Directed Segments per Track',
            labels={'n_directed_segments': 'Number of Directed Segments', 'count': 'Number of Tracks'},
            color_discrete_sequence=['#8c564b']
        )
        fig.update_layout(template="plotly_white")
        figures['segments_per_track'] = fig
    
    # Ensemble statistics summary
    ensemble_results = transport_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add key metrics
        if 'mean_segment_speed' in ensemble_results:
            labels.append('Mean Segment Speed')
            values.append(ensemble_results['mean_segment_speed'])
        
        if 'mean_segment_duration' in ensemble_results:
            labels.append('Mean Segment Duration')
            values.append(ensemble_results['mean_segment_duration'])
        
        if 'mean_segment_straightness' in ensemble_results:
            labels.append('Mean Segment Straightness')
            values.append(ensemble_results['mean_segment_straightness'])
        
        if 'mean_directed_fraction' in ensemble_results:
            labels.append('Mean Directed Fraction')
            values.append(ensemble_results['mean_directed_fraction'])
        
        if 'mean_transport_velocity' in ensemble_results:
            labels.append('Mean Transport Velocity')
            values.append(ensemble_results['mean_transport_velocity'])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Active Transport Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Boundary Crossing Analysis Visualization ---

def plot_boundary_crossing_analysis(boundary_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot boundary crossing analysis results.
    
    Parameters
    ----------
    boundary_results : dict
        Dictionary with boundary crossing analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not boundary_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No boundary crossing analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Crossing events
    crossing_events = boundary_results.get('crossing_events')
    
    if crossing_events is not None and not crossing_events.empty:
        # Histogram of crossing times
        fig = px.histogram(
            crossing_events, x='crossing_time',
            title='Boundary Crossing Time Distribution',
            labels={'crossing_time': 'Crossing Time (s)', 'count': 'Number of Events'},
            color='boundary_id',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(template="plotly_white")
        figures['crossing_times'] = fig
        
        # Direction counts
        direction_counts = crossing_events['direction'].value_counts().reset_index()
        direction_counts.columns = ['direction', 'count']
        
        fig = px.bar(
            direction_counts, x='direction', y='count',
            title='Crossing Direction Distribution',
            labels={'direction': 'Direction', 'count': 'Number of Events'},
            color='direction',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(template="plotly_white")
        figures['crossing_directions'] = fig
        
        # Map of boundaries and crossings
        fig = go.Figure()
        
        # Draw boundaries
        boundaries = boundary_results.get('boundaries', [])
        
        for i, boundary in enumerate(boundaries):
            if boundary.get('type') == 'line':
                if boundary.get('orientation') == 'horizontal' or boundary.get('orientation') == 'y=c':
                    # Horizontal line
                    y = boundary.get('y', boundary.get('intercept', 0))
                    x_min = boundary.get('x_min', crossing_events['pre_x'].min())
                    x_max = boundary.get('x_max', crossing_events['pre_x'].max())
                    
                    fig.add_shape(
                        type="line",
                        x0=x_min, y0=y, x1=x_max, y1=y,
                        line=dict(color='black', width=2, dash='solid')
                    )
                    
                    # Add label
                    fig.add_annotation(
                        x=(x_min + x_max) / 2,
                        y=y,
                        text=f"Boundary {i}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color='black')
                    )
                    
                elif boundary.get('orientation') == 'vertical' or boundary.get('orientation') == 'x=c':
                    # Vertical line
                    x = boundary.get('x', boundary.get('intercept', 0))
                    y_min = boundary.get('y_min', crossing_events['pre_y'].min())
                    y_max = boundary.get('y_max', crossing_events['pre_y'].max())
                    
                    fig.add_shape(
                        type="line",
                        x0=x, y0=y_min, x1=x, y1=y_max,
                        line=dict(color='black', width=2, dash='solid')
                    )
                    
                    # Add label
                    fig.add_annotation(
                        x=x,
                        y=(y_min + y_max) / 2,
                        text=f"Boundary {i}",
                        showarrow=False,
                        xshift=10,
                        font=dict(color='black')
                    )
                    
                elif boundary.get('orientation') == 'y=mx+b':
                    # Sloped line y = mx + b
                    m = boundary.get('slope', 0)
                    b = boundary.get('intercept', 0)
                    x_min = boundary.get('x_min', crossing_events['pre_x'].min())
                    x_max = boundary.get('x_max', crossing_events['pre_x'].max())
                    
                    y_min = m * x_min + b
                    y_max = m * x_max + b
                    
                    fig.add_shape(
                        type="line",
                        x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                        line=dict(color='black', width=2, dash='solid')
                    )
                    
                    # Add label at midpoint
                    mid_x = (x_min + x_max) / 2
                    mid_y = m * mid_x + b
                    
                    fig.add_annotation(
                        x=mid_x,
                        y=mid_y,
                        text=f"Boundary {i}",
                        showarrow=False,
                        font=dict(color='black')
                    )
                
                elif boundary.get('orientation') == 'x=my+b':
                    # Sloped line x = my + b
                    m = boundary.get('slope', 0)
                    b = boundary.get('intercept', 0)
                    y_min = boundary.get('y_min', crossing_events['pre_y'].min())
                    y_max = boundary.get('y_max', crossing_events['pre_y'].max())
                    
                    x_min = m * y_min + b
                    x_max = m * y_max + b
                    
                    fig.add_shape(
                        type="line",
                        x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                        line=dict(color='black', width=2, dash='solid')
                    )
                    
                    # Add label at midpoint
                    mid_y = (y_min + y_max) / 2
                    mid_x = m * mid_y + b
                    
                    fig.add_annotation(
                        x=mid_x,
                        y=mid_y,
                        text=f"Boundary {i}",
                        showarrow=False,
                        font=dict(color='black')
                    )
            
            elif boundary.get('type') == 'circle':
                # Circular boundary
                center_x = boundary.get('center_x', 0)
                center_y = boundary.get('center_y', 0)
                radius = boundary.get('radius', 1)
                
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=center_x - radius, y0=center_y - radius,
                    x1=center_x + radius, y1=center_y + radius,
                    line=dict(color='black', width=2)
                )
                
                # Add label
                fig.add_annotation(
                    x=center_x,
                    y=center_y,
                    text=f"Boundary {i}",
                    showarrow=False,
                    font=dict(color='black')
                )
            
            elif boundary.get('type') == 'point':
                # Point-like boundary
                x = boundary.get('x', 0)
                y = boundary.get('y', 0)
                
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=10,
                        symbol='circle'
                    ),
                    name=f'Boundary {i}'
                ))
        
        # Add crossing points
        fig.add_trace(go.Scatter(
            x=crossing_events['crossing_x'],
            y=crossing_events['crossing_y'],
            mode='markers',
            marker=dict(
                color=crossing_events['boundary_id'],
                colorscale='Viridis',
                size=8,
                line=dict(width=1, color='black')
            ),
            text=[f"Crossing: Track {track_id}, Direction: {direction}" 
                  for track_id, direction in zip(crossing_events['track_id'], crossing_events['direction'])],
            name='Crossing Points'
        ))
        
        fig.update_layout(
            title='Boundary Map with Crossing Points',
            xaxis_title='X Position (µm)',
            yaxis_title='Y Position (µm)',
            template="plotly_white"
        )
        
        # Make axes equal scale
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        figures['boundary_map'] = fig
        
        # Crossing timeline
        fig = px.scatter(
            crossing_events, x='crossing_time', y='track_id',
            color='boundary_id',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            symbol='direction',
            title='Boundary Crossings Timeline',
            labels={
                'crossing_time': 'Crossing Time (s)',
                'track_id': 'Track ID',
                'boundary_id': 'Boundary',
                'direction': 'Direction'
            }
        )
        fig.update_layout(template="plotly_white")
        figures['crossing_timeline'] = fig
    
    # Residence times
    residence_times = boundary_results.get('residence_times')
    
    if residence_times is not None and not residence_times.empty:
        # Histogram of residence times
        fig = px.histogram(
            residence_times, x='residence_time',
            title='Residence Time Distribution',
            labels={'residence_time': 'Residence Time (s)', 'count': 'Number of Events'},
            color='region',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Add exponential fit if there are enough data points
        if len(residence_times) > 10:
            from scipy.stats import expon
            
            # Fit exponential distribution
            res_times = residence_times['residence_time'].values
            loc, scale = expon.fit(res_times)
            
            # Create x values for fit line
            x = np.linspace(min(res_times), max(res_times), 100)
            y = expon.pdf(x, loc=loc, scale=scale) * len(res_times) * (max(res_times) - min(res_times)) / 20
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'Exp Fit (τ={scale:.2f})',
                line=dict(color='black', width=2, dash='dash')
            ))
        
        fig.update_layout(template="plotly_white")
        figures['residence_times'] = fig
        
        # Box plot of residence times by region
        fig = px.box(
            residence_times, x='region', y='residence_time',
            title='Residence Times by Region',
            labels={
                'region': 'Region',
                'residence_time': 'Residence Time (s)'
            },
            color='region',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(template="plotly_white")
        figures['residence_by_region'] = fig
    
    # Track results
    track_results = boundary_results.get('track_results')
    
    if track_results is not None and not track_results.empty:
        # Histogram of crossings per track
        fig = px.histogram(
            track_results, x='n_crossings',
            title='Boundary Crossings per Track',
            labels={'n_crossings': 'Number of Crossings', 'count': 'Number of Tracks'},
            color_discrete_sequence=['#e377c2']
        )
        fig.update_layout(template="plotly_white")
        figures['crossings_per_track'] = fig
    
    # Ensemble statistics summary
    ensemble_results = boundary_results.get('ensemble_results', {})
    
    if ensemble_results:
        # Create bar chart for key statistics
        labels = []
        values = []
        
        # Add key metrics
        if 'mean_crossings_per_track' in ensemble_results:
            labels.append('Mean Crossings per Track')
            values.append(ensemble_results['mean_crossings_per_track'])
        
        if 'crossing_fraction' in ensemble_results:
            labels.append('Crossing Fraction')
            values.append(ensemble_results['crossing_fraction'])
        
        # Add residence time statistics
        if 'mean_residence_time' in ensemble_results:
            labels.append('Mean Residence Time')
            values.append(ensemble_results['mean_residence_time'])
        
        if 'median_residence_time' in ensemble_results:
            labels.append('Median Residence Time')
            values.append(ensemble_results['median_residence_time'])
        
        # Add region-specific stats
        for key in ensemble_results:
            if key.startswith('region_') and key.endswith('_mean_residence'):
                region = key.split('_mean_residence')[0].replace('region_', '')
                labels.append(f'Region {region} Mean Residence')
                values.append(ensemble_results[key])
        
        if labels and values:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Boundary Crossing Ensemble Statistics',
                xaxis_title='Statistic',
                yaxis_title='Value',
                template="plotly_white"
            )
            figures['ensemble_stats'] = fig
    
    return figures

# --- Comparative Visualization ---

def plot_comparative_bar(data: pd.DataFrame, x_col: str, y_col: str, color_col: str,
                      title: str = '', x_label: str = '', y_label: str = '') -> go.Figure:
    """
    Create a comparative bar chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data for plotting
    x_col : str
        Column to use for x-axis
    y_col : str
        Column to use for y-axis values
    color_col : str
        Column to use for bar color
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Bar chart figure
    """
    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            xaxis_title=x_label if x_label else x_col,
            yaxis_title=y_label if y_label else y_col
        )
        return fig
    
    fig = px.bar(
        data, x=x_col, y=y_col, color=color_col,
        title=title,
        labels={
            x_col: x_label if x_label else x_col,
            y_col: y_label if y_label else y_col
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(template="plotly_white")
    return fig

def plot_comparative_box(data: pd.DataFrame, x_col: str, y_col: str, color_col: str,
                       title: str = '', x_label: str = '', y_label: str = '') -> go.Figure:
    """
    Create a comparative box plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data for plotting
    x_col : str
        Column to use for x-axis grouping
    y_col : str
        Column to use for y-axis values
    color_col : str
        Column to use for box color
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Box plot figure
    """
    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            xaxis_title=x_label if x_label else x_col,
            yaxis_title=y_label if y_label else y_col
        )
        return fig
    
    fig = px.box(
        data, x=x_col, y=y_col, color=color_col,
        title=title,
        labels={
            x_col: x_label if x_label else x_col,
            y_col: y_label if y_label else y_col
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(template="plotly_white")
    return fig

def plot_comparative_histogram(data_dict: Dict[str, pd.DataFrame], x_col: str,
                             bins: int = 15, log_scale: bool = False,
                             title: str = '', x_label: str = '') -> go.Figure:
    """
    Create a comparative histogram.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping dataset names to DataFrames
    x_col : str
        Column to use for histogram values
    bins : int
        Number of bins
    log_scale : bool
        Whether to use log scale for x-axis
    title : str
        Plot title
    x_label : str
        X-axis label
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Histogram figure
    """
    if not data_dict:
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            xaxis_title=x_label if x_label else x_col,
            yaxis_title="Count"
        )
        return fig
    
    fig = go.Figure()
    
    for dataset_name, df in data_dict.items():
        if df.empty or x_col not in df.columns:
            continue
            
        fig.add_trace(go.Histogram(
            x=df[x_col],
            name=dataset_name,
            nbinsx=bins,
            opacity=0.7
        ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label if x_label else x_col,
        yaxis_title="Count",
        barmode='overlay',
        template="plotly_white"
    )
    
    if log_scale:
        fig.update_xaxes(type='log')
    
    return fig

def plot_comparative_scatter(data_dict: Dict[str, pd.DataFrame], x_col: str, y_col: str,
                           title: str = '', x_label: str = '', y_label: str = '') -> go.Figure:
    """
    Create a comparative scatter plot.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping dataset names to DataFrames
    x_col : str
        Column to use for x-axis values
    y_col : str
        Column to use for y-axis values
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Scatter plot figure
    """
    if not data_dict:
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            xaxis_title=x_label if x_label else x_col,
            yaxis_title=y_label if y_label else y_col
        )
        return fig
    
    fig = go.Figure()
    
    for dataset_name, df in data_dict.items():
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
            
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name=dataset_name,
            marker=dict(size=8, opacity=0.7)
        ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label if x_label else x_col,
        yaxis_title=y_label if y_label else y_col,
        template="plotly_white"
    )
    
    return fig

def plot_comparative_line(data_dict: Dict[str, pd.DataFrame], x_col: str, y_col: str,
                        title: str = '', x_label: str = '', y_label: str = '') -> go.Figure:
    """
    Create a comparative line plot.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping dataset names to DataFrames
    x_col : str
        Column to use for x-axis values
    y_col : str
        Column to use for y-axis values
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Line plot figure
    """
    if not data_dict:
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            xaxis_title=x_label if x_label else x_col,
            yaxis_title=y_label if y_label else y_col
        )
        return fig
    
    fig = go.Figure()
    
    for dataset_name, df in data_dict.items():
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
            
        # Sort by x column to ensure proper line plot
        sorted_df = df.sort_values(x_col)
        
        fig.add_trace(go.Scatter(
            x=sorted_df[x_col],
            y=sorted_df[y_col],
            mode='lines+markers',
            name=dataset_name
        ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label if x_label else x_col,
        yaxis_title=y_label if y_label else y_col,
        template="plotly_white"
    )
    
    return fig

def plot_comparative_msd(msd_data_dict: Dict[str, pd.DataFrame], 
                        log_scale: bool = False) -> go.Figure:
    """
    Create a comparative MSD plot.
    
    Parameters
    ----------
    msd_data_dict : dict
        Dictionary mapping dataset names to MSD DataFrames
    log_scale : bool
        Whether to use log-log scale
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        MSD plot figure
    """
    if not msd_data_dict:
        fig = go.Figure()
        fig.update_layout(
            title="No MSD data available",
            xaxis_title="Lag Time",
            yaxis_title="MSD"
        )
        return fig
    
    fig = go.Figure()
    
    for dataset_name, msd_df in msd_data_dict.items():
        if msd_df.empty or 'lag_time' not in msd_df.columns or 'msd' not in msd_df.columns:
            continue
            
        # Group by lag_time and calculate mean and std
        grouped = msd_df.groupby('lag_time')
        mean_msd = grouped['msd'].mean()
        std_msd = grouped['msd'].std()
        sem_msd = std_msd / np.sqrt(grouped['msd'].count())
        
        # Add average curve
        fig.add_trace(go.Scatter(
            x=mean_msd.index,
            y=mean_msd.values,
            mode='markers+lines',
            name=dataset_name,
            line=dict(width=2)
        ))
        
        # Add error bands
        fig.add_trace(go.Scatter(
            x=np.concatenate([mean_msd.index, mean_msd.index[::-1]]),
            y=np.concatenate([mean_msd + sem_msd, (mean_msd - sem_msd)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,0,0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            name=f'{dataset_name} SEM'
        ))
    
    # Set log scale if requested
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
        title_suffix = " (Log-Log Scale)"
    else:
        title_suffix = ""
    
    # Customize layout
    fig.update_layout(
        title=f"Comparative MSD vs. Lag Time{title_suffix}",
        xaxis_title="Lag Time",
        yaxis_title="Mean Squared Displacement",
        template="plotly_white"
    )
    
    return fig

def plot_comparative_violin(data: pd.DataFrame, x_col: str, y_col: str, title: str = '',
                         x_label: str = '', y_label: str = '') -> go.Figure:
    """
    Create a comparative violin plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data for plotting
    x_col : str
        Column to use for x-axis grouping
    y_col : str
        Column to use for y-axis values
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Violin plot figure
    """
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            xaxis_title=x_label if x_label else x_col,
            yaxis_title=y_label if y_label else y_col
        )
        return fig
    
    fig = px.violin(
        data, x=x_col, y=y_col, color=x_col,
        box=True, # include box plot inside the violin
        points='all', # show all points
        title=title,
        labels={
            x_col: x_label if x_label else x_col,
            y_col: y_label if y_label else y_col
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(template="plotly_white")
    return fig

# --- Figure Export Utilities ---

def fig_to_base64(fig: Union[go.Figure, plt.Figure]) -> str:
    """
    Convert a figure to base64 string.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        Figure to convert
        
    Returns
    -------
    str
        Base64 encoded string
    """
    if isinstance(fig, go.Figure):
        # Plotly figure
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
    else:
        # Matplotlib figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        base64_str = base64.b64encode(buf.read()).decode("utf-8")
    
    return base64_str

def download_figure(fig: Union[go.Figure, plt.Figure], format: str = "png"):
    """
    Create a download link for a figure.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        Figure to download
    format : str
        File format: "png", "jpg", "svg", "pdf"
        
    Returns
    -------
    bytes
        Figure in the specified format
    """
    if isinstance(fig, go.Figure):
        # Plotly figure
        if format == "png":
            img_bytes = fig.to_image(format="png", width=1200, height=800)
        elif format == "jpg":
            img_bytes = fig.to_image(format="jpeg", width=1200, height=800)
        elif format == "svg":
            img_bytes = fig.to_image(format="svg", width=1200, height=800)
        elif format == "pdf":
            img_bytes = fig.to_image(format="pdf", width=1200, height=800)
        else:
            raise ValueError(f"Unsupported format: {format}")
    else:
        # Matplotlib figure
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=300, bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.read()
    
    return img_bytes

# --- Enhanced Analysis Visualizations ---

def plot_active_transport_analysis(transport_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Plot active transport analysis results with enhanced visualizations.
    
    Parameters
    ----------
    transport_results : dict
        Dictionary with active transport analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not transport_results.get('success', False):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No active transport analysis results available",
            xaxis_title="Value",
            yaxis_title="Count"
        )
        figures['empty'] = empty_fig
        return figures
    
    # Directed segments
    directed_segments = transport_results.get('directed_segments')
    
    if directed_segments is not None and not directed_segments.empty:
        # Histogram of segment speeds
        fig = px.histogram(
            directed_segments, x='speed',
            title='Directed Segment Speed Distribution',
            labels={'speed': 'Speed (µm/s)', 'count': 'Number of Segments'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template="plotly_white")
        figures['segment_speeds'] = fig
        
        # Polar histogram of segment angles
        if 'angle' in directed_segments.columns:
            angles_deg = np.degrees(directed_segments['angle'])
            fig = go.Figure()
            
            # Create polar histogram manually
            n_bins = 16
            bin_edges = np.linspace(-180, 180, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist, _ = np.histogram(angles_deg, bins=bin_edges)
            
            fig.add_trace(go.Scatterpolar(
                r=hist,
                theta=bin_centers,
                mode='lines',
                fill='toself',
                name='Direction Distribution'
            ))
            
            fig.update_layout(
                title='Directed Motion Directionality',
                polar=dict(
                    radialaxis=dict(visible=True),
                    angularaxis=dict(direction='counterclockwise', period=360)
                ),
                template="plotly_white"
            )
            figures['segment_directions'] = fig
    
    return figures

def plot_enhanced_msd_analysis(msd_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Enhanced MSD plotting with additional analysis features.
    
    Parameters
    ----------
    msd_results : dict
        Dictionary with MSD analysis results
        
    Returns
    -------
    figures : dict
        Dictionary of plot figures
    """
    figures = {}
    
    if not msd_results.get('success', False):
        return figures
    
    msd_data = msd_results.get('msd_data')
    ensemble_msd = msd_results.get('ensemble_msd')
    
    if msd_data is not None and not msd_data.empty:
        # Individual MSD curves with ensemble average
        fig = go.Figure()
        
        # Plot individual tracks (sample to avoid overcrowding)
        unique_tracks = msd_data['track_id'].unique()
        sample_tracks = np.random.choice(unique_tracks, min(20, len(unique_tracks)), replace=False)
        
        for track_id in sample_tracks:
            track_msd = msd_data[msd_data['track_id'] == track_id]
            fig.add_trace(go.Scatter(
                x=track_msd['lag_time'],
                y=track_msd['msd'],
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.3,
                showlegend=False,
                name=f'Track {track_id}'
            ))
        
        # Add ensemble average
        if ensemble_msd is not None and not ensemble_msd.empty:
            fig.add_trace(go.Scatter(
                x=ensemble_msd['lag_time'],
                y=ensemble_msd['msd_mean'],
                mode='lines+markers',
                line=dict(color='red', width=3),
                name='Ensemble Average',
                error_y=dict(
                    type='data',
                    array=ensemble_msd['msd_std'] if 'msd_std' in ensemble_msd.columns else None,
                    visible=True
                )
            ))
        
        fig.update_layout(
            title='Enhanced Mean Squared Displacement Analysis',
            xaxis_title='Lag Time (s)',
            yaxis_title='MSD (µm²)',
            template="plotly_white"
        )
        figures['enhanced_msd'] = fig
        
        # MSD/t vs t plot for diffusion regime identification
        if ensemble_msd is not None and not ensemble_msd.empty:
            ensemble_msd_copy = ensemble_msd.copy()
            ensemble_msd_copy['msd_over_t'] = ensemble_msd_copy['msd_mean'] / ensemble_msd_copy['lag_time']
            
            fig = px.scatter(
                ensemble_msd_copy, x='lag_time', y='msd_over_t',
                title='MSD/t vs. Time (Diffusion Regime Analysis)',
                labels={'lag_time': 'Lag Time (s)', 'msd_over_t': 'MSD/t (µm²/s)'}
            )
            fig.update_layout(template="plotly_white")
            figures['msd_regime'] = fig
    
    return figures

def plot_velocity_correlation_analysis(tracks_df: pd.DataFrame, max_lag: int = 10) -> go.Figure:
    """
    Plot velocity autocorrelation analysis.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format
    max_lag : int
        Maximum lag for correlation analysis
        
    Returns
    -------
    fig : go.Figure
        Plotly figure with velocity correlation analysis
    """
    fig = go.Figure()
    
    if tracks_df.empty:
        fig.update_layout(title="No track data available for velocity correlation analysis")
        return fig
    
    # Calculate velocity for each track
    correlation_data = []
    
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 3:
            continue
        
        # Calculate velocity components
        vx = np.diff(track_data['x'].values)
        vy = np.diff(track_data['y'].values)
        
        # Calculate velocity autocorrelation
        for lag in range(1, min(max_lag + 1, len(vx))):
            if len(vx) > lag:
                # Calculate correlation for this lag
                vx_corr = np.corrcoef(vx[:-lag], vx[lag:])[0, 1] if len(vx) > lag else 0
                vy_corr = np.corrcoef(vy[:-lag], vy[lag:])[0, 1] if len(vy) > lag else 0
                
                if not np.isnan(vx_corr) and not np.isnan(vy_corr):
                    correlation_data.append({
                        'lag': lag,
                        'vx_correlation': vx_corr,
                        'vy_correlation': vy_corr,
                        'track_id': track_id
                    })
    
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)
        
        # Calculate ensemble average
        ensemble_corr = corr_df.groupby('lag').agg({
            'vx_correlation': ['mean', 'std'],
            'vy_correlation': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        ensemble_corr.columns = ['lag', 'vx_mean', 'vx_std', 'vy_mean', 'vy_std']
        
        # Plot ensemble averages
        fig.add_trace(go.Scatter(
            x=ensemble_corr['lag'],
            y=ensemble_corr['vx_mean'],
            mode='lines+markers',
            name='Vx Correlation',
            error_y=dict(type='data', array=ensemble_corr['vx_std']),
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=ensemble_corr['lag'],
            y=ensemble_corr['vy_mean'],
            mode='lines+markers',
            name='Vy Correlation',
            error_y=dict(type='data', array=ensemble_corr['vy_std']),
            line=dict(color='red')
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='Velocity Autocorrelation Analysis',
        xaxis_title='Lag (frames)',
        yaxis_title='Correlation Coefficient',
        template="plotly_white"
    )
    
    return fig

def plot_diffusion_coefficients(track_results) -> go.Figure:
    """
    Plot diffusion coefficients for individual tracks.
    
    Parameters
    ----------
    track_results : pd.DataFrame or dict
        Track results containing diffusion coefficients
        
    Returns
    -------
    go.Figure
        Plotly figure showing diffusion coefficient distribution
    """
    # Handle both dict and DataFrame inputs
    if isinstance(track_results, dict):
        # Extract track_results from nested dict structure if needed
        if 'track_results' in track_results:
            track_data = track_results['track_results']
        else:
            track_data = track_results
        
        # Convert to DataFrame if it's a dict
        if isinstance(track_data, dict):
            if 'diffusion_coefficient' in track_data:
                track_df = pd.DataFrame({'diffusion_coefficient': track_data['diffusion_coefficient']})
            else:
                track_df = pd.DataFrame()
        else:
            track_df = track_data
    else:
        track_df = track_results
    
    # Check if we have valid data
    if track_df is None or (hasattr(track_df, 'empty') and track_df.empty):
        fig = go.Figure()
        fig.add_annotation(
            text="No track results available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Check if diffusion_coefficient column exists
    if not hasattr(track_df, 'columns') or 'diffusion_coefficient' not in track_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No diffusion coefficient data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Remove invalid values - handle both DataFrame and dict cases
    if hasattr(track_df, 'dropna'):
        valid_data = track_df.dropna(subset=['diffusion_coefficient'])
        valid_data = valid_data[valid_data['diffusion_coefficient'] > 0]
        data_empty = valid_data.empty
        diff_coeffs = valid_data['diffusion_coefficient']
    else:
        # Handle dict case
        if 'diffusion_coefficient' in track_df:
            diff_coeffs = pd.Series(track_df['diffusion_coefficient'])
            diff_coeffs = diff_coeffs.dropna()
            diff_coeffs = diff_coeffs[diff_coeffs > 0]
            data_empty = len(diff_coeffs) == 0
        else:
            data_empty = True
            diff_coeffs = pd.Series([])
    
    if data_empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid diffusion coefficient data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create histogram of diffusion coefficients
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=diff_coeffs,
        nbinsx=30,
        name='Diffusion Coefficients',
        marker_color='skyblue',
        opacity=0.7
    ))
    
    # Add mean line
    mean_val = valid_data['diffusion_coefficient'].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.3f} µm²/s"
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Diffusion Coefficients",
        xaxis_title="Diffusion Coefficient (µm²/s)",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False
    )
    
    return fig
