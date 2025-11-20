"""
Enhanced Trajectory Visualizations for SPT Reports
Includes temporal color coding ("dragon tails") and 3D space-time cubes
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_trajectories_temporal_color(tracks_df: pd.DataFrame,
                                     pixel_size: float = 0.1,
                                     max_tracks: Optional[int] = 20,
                                     colormap: str = 'viridis') -> go.Figure:
    """
    Plot trajectories with temporal color coding ("dragon tails").
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns: track_id, frame, x, y
    pixel_size : float
        Pixel size in μm
    max_tracks : int, optional
        Maximum number of tracks to display
    colormap : str
        Matplotlib colormap name (e.g., 'viridis', 'plasma', 'coolwarm')
        
    Returns
    -------
    go.Figure
        Plotly figure with temporal color coding
    """
    # Convert to μm
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    
    # Limit number of tracks
    track_ids = plot_df['track_id'].unique()
    if max_tracks and len(track_ids) > max_tracks:
        selected_tracks = np.random.choice(track_ids, max_tracks, replace=False)
        plot_df = plot_df[plot_df['track_id'].isin(selected_tracks)]
    
    fig = go.Figure()
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Plot each track with temporal color gradient
    for track_id in plot_df['track_id'].unique():
        track = plot_df[plot_df['track_id'] == track_id].sort_values('frame')
        
        if len(track) < 2:
            continue
        
        # Normalize time to [0, 1]
        frames = track['frame'].values
        t_norm = (frames - frames.min()) / max(frames.max() - frames.min(), 1)
        
        # Create color array
        colors = [mcolors.rgb2hex(cmap(t)[:3]) for t in t_norm]
        
        # Plot trajectory as line segments with varying colors
        for i in range(len(track) - 1):
            fig.add_trace(
                go.Scatter(
                    x=track['x_um'].iloc[i:i+2],
                    y=track['y_um'].iloc[i:i+2],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4, color=colors[i]),
                    showlegend=False,
                    hovertemplate=f'Track {track_id}<br>Frame: {frames[i]}<br>X: %{{x:.2f}} μm<br>Y: %{{y:.2f}} μm<extra></extra>'
                )
            )
        
        # Mark start point
        fig.add_trace(
            go.Scatter(
                x=[track['x_um'].iloc[0]],
                y=[track['y_um'].iloc[0]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='circle-open', line=dict(width=2)),
                name=f'Start (Track {track_id})' if track_id == track_ids[0] else None,
                showlegend=(track_id == track_ids[0]),
                hovertext=f'Track {track_id} Start'
            )
        )
        
        # Mark end point
        fig.add_trace(
            go.Scatter(
                x=[track['x_um'].iloc[-1]],
                y=[track['y_um'].iloc[-1]],
                mode='markers',
                marker=dict(size=10, color='red', symbol='x', line=dict(width=2)),
                name=f'End (Track {track_id})' if track_id == track_ids[0] else None,
                showlegend=(track_id == track_ids[0]),
                hovertext=f'Track {track_id} End'
            )
        )
    
    # Add colorbar to show time mapping
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=colormap,
                showscale=True,
                cmin=0,
                cmax=plot_df['frame'].max(),
                colorbar=dict(
                    title="Frame",
                    x=1.02,
                    thickness=20
                )
            ),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.update_layout(
        title="Trajectory Visualization with Temporal Color Coding (Dragon Tails)",
        xaxis_title="X Position (μm)",
        yaxis_title="Y Position (μm)",
        template='plotly_white',
        height=700,
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
    )
    
    return fig


def plot_spacetime_cube(tracks_df: pd.DataFrame,
                        pixel_size: float = 0.1,
                        frame_interval: float = 0.1,
                        max_tracks: Optional[int] = 10,
                        colormap: str = 'rainbow') -> go.Figure:
    """
    Create 3D space-time cube visualization.
    X and Y on horizontal plane, Time on vertical Z-axis.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns: track_id, frame, x, y
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Time between frames in seconds
    max_tracks : int, optional
        Maximum number of tracks to display
    colormap : str
        Matplotlib colormap name
        
    Returns
    -------
    go.Figure
        3D plotly figure
    """
    # Convert to physical units
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    plot_df['time_s'] = plot_df['frame'] * frame_interval
    
    # Limit number of tracks
    track_ids = plot_df['track_id'].unique()
    if max_tracks and len(track_ids) > max_tracks:
        selected_tracks = np.random.choice(track_ids, max_tracks, replace=False)
        plot_df = plot_df[plot_df['track_id'].isin(selected_tracks)]
    
    fig = go.Figure()
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Plot each track in 3D
    for i, track_id in enumerate(plot_df['track_id'].unique()):
        track = plot_df[plot_df['track_id'] == track_id].sort_values('frame')
        
        if len(track) < 2:
            continue
        
        # Assign color
        color = mcolors.rgb2hex(cmap(i / max(len(plot_df['track_id'].unique()) - 1, 1))[:3])
        
        # Plot trajectory as 3D line
        fig.add_trace(
            go.Scatter3d(
                x=track['x_um'],
                y=track['y_um'],
                z=track['time_s'],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=3, color=color),
                name=f'Track {track_id}',
                hovertemplate=f'Track {track_id}<br>X: %{{x:.2f}} μm<br>Y: %{{y:.2f}} μm<br>Time: %{{z:.2f}} s<extra></extra>'
            )
        )
        
        # Highlight dwelling events (vertical lines indicate pauses)
        # Calculate instantaneous speed
        if len(track) > 2:
            dx = np.diff(track['x_um'])
            dy = np.diff(track['y_um'])
            dt = np.diff(track['time_s'])
            speed = np.sqrt(dx**2 + dy**2) / dt
            
            # Identify dwelling (low speed)
            threshold = np.percentile(speed, 10)  # Bottom 10% of speeds
            dwelling_indices = np.where(speed < threshold)[0]
            
            # Mark dwelling regions
            for idx in dwelling_indices:
                if idx < len(track) - 1:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[track['x_um'].iloc[idx], track['x_um'].iloc[idx]],
                            y=[track['y_um'].iloc[idx], track['y_um'].iloc[idx]],
                            z=[track['time_s'].iloc[idx], track['time_s'].iloc[idx+1]],
                            mode='lines',
                            line=dict(color='red', width=6, dash='dot'),
                            showlegend=False,
                            hovertext=f'Dwelling event (Track {track_id})',
                            hoverinfo='text'
                        )
                    )
    
    fig.update_layout(
        title="3D Space-Time Cube Visualization",
        scene=dict(
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            zaxis_title="Time (s)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        template='plotly_white',
        height=800
    )
    
    return fig


def plot_combined_trajectory_views(tracks_df: pd.DataFrame,
                                   pixel_size: float = 0.1,
                                   frame_interval: float = 0.1,
                                   max_tracks: int = 10) -> go.Figure:
    """
    Create combined figure with 2D temporal color and 3D space-time views.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    max_tracks : int
        Maximum tracks to display
        
    Returns
    -------
    go.Figure
        Combined subplot figure
    """
    # Create subplot with 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Temporal Color Coding", "3D Space-Time Cube"),
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    
    # Generate individual plots
    fig_2d = plot_trajectories_temporal_color(tracks_df, pixel_size, max_tracks)
    fig_3d = plot_spacetime_cube(tracks_df, pixel_size, frame_interval, max_tracks)
    
    # Add traces from 2D plot
    for trace in fig_2d.data:
        if trace.type == 'scatter':
            fig.add_trace(trace, row=1, col=1)
    
    # Add traces from 3D plot
    for trace in fig_3d.data:
        if trace.type == 'scatter3d':
            fig.add_trace(trace, row=1, col=2)
    
    fig.update_layout(
        title_text="Enhanced Trajectory Visualizations",
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    return fig
