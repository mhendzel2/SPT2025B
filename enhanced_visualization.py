"""
Enhanced Visualization Module
Provides advanced, publication-ready visualizations for SPT data.

Features:
- Interactive trajectory plots with controls
- Animated trajectories
- Spatial heatmaps and density maps
- Publication-ready figure templates
- Multi-panel figures
- Customizable styling
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Ellipse
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Animation features will be limited.")


# ==================== INTERACTIVE TRAJECTORY PLOTS ====================

def plot_interactive_trajectories(tracks_df: pd.DataFrame,
                                  pixel_size: float = 0.1,
                                  color_by: str = 'track_id',
                                  show_points: bool = True,
                                  show_lines: bool = True,
                                  max_tracks: Optional[int] = None,
                                  highlight_tracks: Optional[List[int]] = None,
                                  title: str = 'Interactive Trajectory Visualization') -> go.Figure:
    """
    Create interactive trajectory plot with hover information and controls.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns: track_id, frame, x, y
    pixel_size : float
        Pixel size in μm
    color_by : str
        Column to color by: 'track_id', 'frame', 'compartment', etc.
    show_points : bool
        Show individual points
    show_lines : bool
        Show connecting lines
    max_tracks : int, optional
        Maximum number of tracks to display
    highlight_tracks : list, optional
        Track IDs to highlight
    title : str
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure
    """
    # Convert to μm
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    
    # Limit number of tracks if requested
    track_ids = plot_df['track_id'].unique()
    if max_tracks and len(track_ids) > max_tracks:
        selected_tracks = np.random.choice(track_ids, max_tracks, replace=False)
        plot_df = plot_df[plot_df['track_id'].isin(selected_tracks)]
    
    fig = go.Figure()
    
    # Determine coloring
    if color_by not in plot_df.columns:
        color_by = 'track_id'
    
    # Add traces for each track
    for track_id in plot_df['track_id'].unique():
        track = plot_df[plot_df['track_id'] == track_id].sort_values('frame')
        
        # Determine if this track should be highlighted
        is_highlighted = highlight_tracks and track_id in highlight_tracks
        
        # Color
        if color_by == 'track_id':
            color = f'hsl({(track_id * 137.5) % 360}, 70%, 50%)'
        else:
            color = 'blue'
        
        # Line width
        line_width = 3 if is_highlighted else 1
        opacity = 1.0 if is_highlighted else 0.6
        
        # Add line trace
        if show_lines:
            fig.add_trace(go.Scatter(
                x=track['x_um'],
                y=track['y_um'],
                mode='lines',
                name=f'Track {track_id}',
                line=dict(color=color, width=line_width),
                opacity=opacity,
                hovertemplate='Track %{customdata[0]}<br>Frame: %{customdata[1]}<br>x: %{x:.3f} μm<br>y: %{y:.3f} μm<extra></extra>',
                customdata=np.column_stack([track['track_id'], track['frame']]),
                showlegend=is_highlighted
            ))
        
        # Add point trace
        if show_points:
            fig.add_trace(go.Scatter(
                x=track['x_um'],
                y=track['y_um'],
                mode='markers',
                name=f'Track {track_id} (points)',
                marker=dict(
                    color=color,
                    size=5 if is_highlighted else 3,
                    symbol='circle',
                    line=dict(width=1, color='white') if is_highlighted else dict(width=0)
                ),
                opacity=opacity,
                hovertemplate='Track %{customdata[0]}<br>Frame: %{customdata[1]}<br>x: %{x:.3f} μm<br>y: %{y:.3f} μm<extra></extra>',
                customdata=np.column_stack([track['track_id'], track['frame']]),
                showlegend=False
            ))
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis_title='X (μm)',
        yaxis_title='Y (μm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),  # Equal aspect ratio
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=800
    )
    
    # Add range slider and buttons
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


def plot_trajectory_time_evolution(tracks_df: pd.DataFrame,
                                   track_ids: Optional[List[int]] = None,
                                   pixel_size: float = 0.1,
                                   frame_interval: float = 0.1) -> go.Figure:
    """
    Plot trajectory evolution over time with time slider.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    track_ids : list, optional
        Specific tracks to plot
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Time between frames in seconds
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure with time slider
    """
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    plot_df['time'] = plot_df['frame'] * frame_interval
    
    if track_ids:
        plot_df = plot_df[plot_df['track_id'].isin(track_ids)]
    
    # Create frames for animation
    frames_list = []
    frame_numbers = sorted(plot_df['frame'].unique())
    
    for frame_num in frame_numbers:
        frame_data = plot_df[plot_df['frame'] <= frame_num]
        
        # Create traces for this frame
        traces = []
        for track_id in plot_df['track_id'].unique():
            track_frame = frame_data[frame_data['track_id'] == track_id]
            
            if len(track_frame) > 0:
                color = f'hsl({(track_id * 137.5) % 360}, 70%, 50%)'
                
                # Trajectory line
                traces.append(go.Scatter(
                    x=track_frame['x_um'],
                    y=track_frame['y_um'],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    name=f'Track {track_id}',
                    showlegend=False
                ))
                
                # Current position (larger marker)
                current = track_frame.iloc[-1]
                traces.append(go.Scatter(
                    x=[current['x_um']],
                    y=[current['y_um']],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='circle', line=dict(width=2, color='white')),
                    name=f'Track {track_id}',
                    showlegend=False
                ))
        
        frames_list.append(go.Frame(data=traces, name=str(frame_num)))
    
    # Initial frame
    fig = go.Figure(data=frames_list[0].data if frames_list else [], frames=frames_list)
    
    # Add slider
    sliders = [{
        'active': 0,
        'steps': [
            {
                'args': [[f.name], {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                'label': f'Frame {f.name}',
                'method': 'animate'
            }
            for f in frames_list
        ],
        'x': 0.1,
        'y': 0,
        'xanchor': 'left',
        'yanchor': 'top',
        'len': 0.8
    }]
    
    # Add play/pause buttons
    updatemenus = [{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
    
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        title='Trajectory Evolution Over Time',
        xaxis_title='X (μm)',
        yaxis_title='Y (μm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        template='plotly_white',
        width=900,
        height=800
    )
    
    return fig


# ==================== SPATIAL ANALYSIS PLOTS ====================

def plot_density_heatmap(tracks_df: pd.DataFrame,
                        pixel_size: float = 0.1,
                        bin_size: float = 0.5,
                        colorscale: str = 'Hot',
                        title: str = 'Spatial Density Heatmap') -> go.Figure:
    """
    Create 2D density heatmap of particle positions.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    bin_size : float
        Bin size for heatmap in μm
    colorscale : str
        Plotly colorscale name
    title : str
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap figure
    """
    # Convert to μm
    x_um = tracks_df['x'].values * pixel_size
    y_um = tracks_df['y'].values * pixel_size
    
    # Create bins
    x_bins = np.arange(x_um.min(), x_um.max() + bin_size, bin_size)
    y_bins = np.arange(y_um.min(), y_um.max() + bin_size, bin_size)
    
    # Calculate 2D histogram
    H, xedges, yedges = np.histogram2d(x_um, y_um, bins=[x_bins, y_bins])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=H.T,
        x=xedges,
        y=yedges,
        colorscale=colorscale,
        colorbar=dict(title='Count'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='X (μm)',
        yaxis_title='Y (μm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        template='plotly_white',
        width=800,
        height=800
    )
    
    return fig


def plot_spatial_compartments(tracks_df: pd.DataFrame,
                              pixel_size: float = 0.1,
                              compartment_column: str = 'compartment') -> go.Figure:
    """
    Visualize trajectories colored by compartment/region.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with compartment labels
    pixel_size : float
        Pixel size in μm
    compartment_column : str
        Column containing compartment labels
        
    Returns
    -------
    plotly.graph_objects.Figure
        Compartment visualization
    """
    if compartment_column not in tracks_df.columns:
        warnings.warn(f"Column '{compartment_column}' not found. Using default colors.")
        return plot_interactive_trajectories(tracks_df, pixel_size)
    
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    
    fig = go.Figure()
    
    # Get unique compartments
    compartments = plot_df[compartment_column].unique()
    
    # Color scheme
    colors = px.colors.qualitative.Set2
    
    for idx, compartment in enumerate(compartments):
        comp_data = plot_df[plot_df[compartment_column] == compartment]
        color = colors[idx % len(colors)]
        
        # Plot all tracks in this compartment
        for track_id in comp_data['track_id'].unique():
            track = comp_data[comp_data['track_id'] == track_id].sort_values('frame')
            
            fig.add_trace(go.Scatter(
                x=track['x_um'],
                y=track['y_um'],
                mode='lines+markers',
                name=str(compartment),
                line=dict(color=color, width=2),
                marker=dict(size=3, color=color),
                showlegend=(track_id == comp_data['track_id'].iloc[0]),  # Only show legend once per compartment
                legendgroup=str(compartment)
            ))
    
    fig.update_layout(
        title='Trajectories by Compartment',
        xaxis_title='X (μm)',
        yaxis_title='Y (μm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        template='plotly_white',
        width=800,
        height=800
    )
    
    return fig


# ==================== PUBLICATION-READY FIGURES ====================

def create_multi_panel_figure(tracks_df: pd.DataFrame,
                              pixel_size: float = 0.1,
                              frame_interval: float = 0.1,
                              include_panels: List[str] = ['trajectories', 'msd', 'displacement', 'velocity']) -> go.Figure:
    """
    Create publication-ready multi-panel figure.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    include_panels : list
        Panels to include: 'trajectories', 'msd', 'displacement', 'velocity', 'heatmap'
        
    Returns
    -------
    plotly.graph_objects.Figure
        Multi-panel figure
    """
    n_panels = len(include_panels)
    
    # Determine subplot layout
    if n_panels == 2:
        rows, cols = 1, 2
    elif n_panels == 3:
        rows, cols = 1, 3
    elif n_panels == 4:
        rows, cols = 2, 2
    elif n_panels <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    # Create subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[p.capitalize() for p in include_panels],
        specs=[[{'type': 'xy'} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Convert coordinates
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    plot_df['time'] = plot_df['frame'] * frame_interval
    
    panel_idx = 0
    
    for panel in include_panels:
        row = panel_idx // cols + 1
        col = panel_idx % cols + 1
        
        if panel == 'trajectories':
            # Sample tracks for clarity
            sample_tracks = np.random.choice(plot_df['track_id'].unique(), 
                                           min(20, len(plot_df['track_id'].unique())), 
                                           replace=False)
            
            for track_id in sample_tracks:
                track = plot_df[plot_df['track_id'] == track_id].sort_values('frame')
                color = f'hsl({(track_id * 137.5) % 360}, 70%, 50%)'
                
                fig.add_trace(
                    go.Scatter(x=track['x_um'], y=track['y_um'], 
                             mode='lines', line=dict(color=color, width=1),
                             showlegend=False),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text='X (μm)', row=row, col=col)
            fig.update_yaxes(title_text='Y (μm)', row=row, col=col, scaleanchor='x', scaleratio=1)
        
        elif panel == 'msd':
            # Calculate ensemble MSD
            from msd_calculation import calculate_msd_ensemble
            msd_data = calculate_msd_ensemble(tracks_df, pixel_size, frame_interval)
            
            if msd_data['success']:
                lag_times = msd_data['lag_times']
                msd = msd_data['msd']
                
                fig.add_trace(
                    go.Scatter(x=lag_times, y=msd, mode='lines+markers',
                             marker=dict(size=6, color='blue'),
                             line=dict(color='blue', width=2),
                             showlegend=False),
                    row=row, col=col
                )
                
                fig.update_xaxes(title_text='Lag Time (s)', type='log', row=row, col=col)
                fig.update_yaxes(title_text='MSD (μm²)', type='log', row=row, col=col)
        
        elif panel == 'displacement':
            # Calculate displacement distribution
            displacements = []
            for track_id in plot_df['track_id'].unique():
                track = plot_df[plot_df['track_id'] == track_id].sort_values('frame')
                if len(track) > 1:
                    coords = track[['x_um', 'y_um']].values
                    disp = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                    displacements.extend(disp)
            
            fig.add_trace(
                go.Histogram(x=displacements, nbinsx=50, 
                           marker=dict(color='green'),
                           showlegend=False),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text='Displacement (μm)', row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)
        
        elif panel == 'velocity':
            # Calculate velocity distribution
            velocities = []
            for track_id in plot_df['track_id'].unique():
                track = plot_df[plot_df['track_id'] == track_id].sort_values('frame')
                if len(track) > 1:
                    coords = track[['x_um', 'y_um']].values
                    disp = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                    vel = disp / frame_interval
                    velocities.extend(vel)
            
            fig.add_trace(
                go.Histogram(x=velocities, nbinsx=50,
                           marker=dict(color='red'),
                           showlegend=False),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text='Velocity (μm/s)', row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)
        
        elif panel == 'heatmap':
            # 2D histogram
            x_bins = np.linspace(plot_df['x_um'].min(), plot_df['x_um'].max(), 50)
            y_bins = np.linspace(plot_df['y_um'].min(), plot_df['y_um'].max(), 50)
            H, xedges, yedges = np.histogram2d(plot_df['x_um'], plot_df['y_um'], 
                                              bins=[x_bins, y_bins])
            
            fig.add_trace(
                go.Heatmap(z=H.T, x=xedges, y=yedges, colorscale='Hot',
                          showscale=False),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text='X (μm)', row=row, col=col)
            fig.update_yaxes(title_text='Y (μm)', row=row, col=col)
        
        panel_idx += 1
    
    # Update layout
    fig.update_layout(
        title_text='Comprehensive SPT Analysis',
        showlegend=False,
        template='plotly_white',
        height=300 * rows,
        width=400 * cols
    )
    
    return fig


def export_publication_figure(fig: go.Figure,
                             filename: str,
                             format: str = 'png',
                             width: int = 1200,
                             height: int = 1000,
                             scale: float = 3.0) -> str:
    """
    Export figure in publication-ready format.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to export
    filename : str
        Output filename (without extension)
    format : str
        Format: 'png', 'pdf', 'svg', 'eps'
    width : int
        Width in pixels
    height : int
        Height in pixels
    scale : float
        Scale factor for higher resolution
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        output_path = f"{filename}.{format}"
        
        fig.write_image(
            output_path,
            format=format,
            width=width,
            height=height,
            scale=scale
        )
        
        return output_path
    except Exception as e:
        warnings.warn(f"Export failed: {str(e)}. Install kaleido: pip install kaleido")
        return None


# ==================== ANIMATION ====================

def create_trajectory_animation(tracks_df: pd.DataFrame,
                               pixel_size: float = 0.1,
                               frame_interval: float = 0.1,
                               output_file: str = 'trajectory_animation.mp4',
                               fps: int = 10,
                               trail_length: int = 10) -> str:
    """
    Create animated visualization of trajectories.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    output_file : str
        Output filename
    fps : int
        Frames per second
    trail_length : int
        Number of previous positions to show
        
    Returns
    -------
    str
        Path to saved animation file
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib not available. Animation cannot be created.")
        return None
    
    # Convert to μm
    plot_df = tracks_df.copy()
    plot_df['x_um'] = plot_df['x'] * pixel_size
    plot_df['y_um'] = plot_df['y'] * pixel_size
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get bounds
    x_min, x_max = plot_df['x_um'].min(), plot_df['x_um'].max()
    y_min, y_max = plot_df['y_um'].min(), plot_df['y_um'].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Get all frames
    frames = sorted(plot_df['frame'].unique())
    
    # Initialize plot elements
    lines = {}
    points = {}
    
    for track_id in plot_df['track_id'].unique():
        color = plt.cm.tab20(track_id % 20)
        lines[track_id], = ax.plot([], [], '-', color=color, linewidth=1, alpha=0.6)
        points[track_id], = ax.plot([], [], 'o', color=color, markersize=8)
    
    title = ax.set_title(f'Frame 0 / Time 0.00 s', fontsize=14)
    
    def animate(frame_idx):
        current_frame = frames[frame_idx]
        current_data = plot_df[plot_df['frame'] <= current_frame]
        
        for track_id in plot_df['track_id'].unique():
            track_data = current_data[current_data['track_id'] == track_id]
            
            if len(track_data) > 0:
                # Show trail
                if len(track_data) > trail_length:
                    track_data = track_data.iloc[-trail_length:]
                
                lines[track_id].set_data(track_data['x_um'], track_data['y_um'])
                
                # Current position
                last_point = track_data.iloc[-1]
                points[track_id].set_data([last_point['x_um']], [last_point['y_um']])
            else:
                lines[track_id].set_data([], [])
                points[track_id].set_data([], [])
        
        title.set_text(f'Frame {current_frame} / Time {current_frame * frame_interval:.2f} s')
        
        return list(lines.values()) + list(points.values()) + [title]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames),
        interval=1000/fps, blit=True
    )
    
    # Save
    try:
        anim.save(output_file, fps=fps, dpi=150)
        plt.close(fig)
        return output_file
    except Exception as e:
        warnings.warn(f"Animation save failed: {str(e)}")
        plt.close(fig)
        return None


# ==================== HIGH-LEVEL API ====================

def create_comprehensive_visualization(tracks_df: pd.DataFrame,
                                      pixel_size: float = 0.1,
                                      frame_interval: float = 0.1,
                                      max_tracks_display: int = 50) -> Dict[str, go.Figure]:
    """
    Generate comprehensive set of visualizations.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    max_tracks_display : int
        Maximum tracks to display in trajectory plots
        
    Returns
    -------
    dict
        Dictionary of figure objects
    """
    figures = {}
    
    try:
        # Interactive trajectories
        figures['interactive_trajectories'] = plot_interactive_trajectories(
            tracks_df, pixel_size, max_tracks=max_tracks_display
        )
        
        # Density heatmap
        figures['density_heatmap'] = plot_density_heatmap(
            tracks_df, pixel_size
        )
        
        # Multi-panel figure
        figures['multi_panel'] = create_multi_panel_figure(
            tracks_df, pixel_size, frame_interval,
            include_panels=['trajectories', 'msd', 'displacement', 'velocity']
        )
        
        # Time evolution (for subset of tracks)
        sample_tracks = np.random.choice(
            tracks_df['track_id'].unique(),
            min(10, len(tracks_df['track_id'].unique())),
            replace=False
        )
        figures['time_evolution'] = plot_trajectory_time_evolution(
            tracks_df, list(sample_tracks), pixel_size, frame_interval
        )
        
    except Exception as e:
        warnings.warn(f"Visualization error: {str(e)}")
    
    return figures
