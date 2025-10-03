"""
Example: Optimized Plot Functions Using Visualization Optimization Utilities

This file demonstrates how to integrate the visualization optimization utilities
into existing plotting functions for better performance with large datasets.
"""

import pandas as pd
import plotly.graph_objects as go
from visualization_optimization import (
    downsample_tracks,
    cached_plot,
    apply_colorblind_palette,
    optimize_figure_size,
    add_plot_metadata,
    create_data_size_warning,
    COLORBLIND_SAFE_PALETTES
)


def plot_tracks_optimized(
    tracks_df: pd.DataFrame,
    max_tracks: int = 50,
    max_points_per_track: int = 1000,
    downsample_method: str = 'uniform',
    colormap: str = 'viridis',
    colorblind_mode: bool = False,
    include_markers: bool = True,
    marker_size: int = 5,
    line_width: int = 1,
    title: str = "Particle Tracks",
    use_cache: bool = True
) -> go.Figure:
    """
    Plot particle tracks with optimization and caching.
    
    This is an optimized version that:
    - Downsamples large datasets for faster rendering
    - Caches generated plots for repeated views
    - Supports colorblind-safe palettes
    - Adds metadata about data size and optimization
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with 'track_id', 'frame', 'x', 'y' columns
    max_tracks : int, default 50
        Maximum number of tracks to display
    max_points_per_track : int, default 1000
        Maximum points per track (0 = no limit)
    downsample_method : str, default 'uniform'
        Downsampling method: 'uniform', 'random', or 'temporal'
    colormap : str, default 'viridis'
        Color scheme or colorblind-safe palette name
    colorblind_mode : bool, default False
        Use colorblind-safe colors
    include_markers : bool, default True
        Show markers at each point
    marker_size : int, default 5
        Size of markers in pixels
    line_width : int, default 1
        Width of track lines
    title : str, default "Particle Tracks"
        Plot title
    use_cache : bool, default True
        Use plot caching
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if tracks_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No track data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font_size=14
        )
        return fig
    
    # Store original size for metadata
    original_tracks = tracks_df['track_id'].nunique()
    original_points = len(tracks_df)
    
    # Downsample if needed
    downsampled = False
    if max_tracks or max_points_per_track:
        tracks_df = downsample_tracks(
            tracks_df,
            max_tracks=max_tracks,
            max_points_per_track=max_points_per_track,
            method=downsample_method
        )
        downsampled = (len(tracks_df) < original_points)
    
    # Show warning for large datasets
    warning = create_data_size_warning(
        original_tracks,
        original_points,
        max_tracks=max_tracks
    )
    
    # Use caching if enabled
    if use_cache:
        params = {
            'max_tracks': max_tracks,
            'max_points_per_track': max_points_per_track,
            'colormap': colormap,
            'include_markers': include_markers,
            'marker_size': marker_size,
            'line_width': line_width,
            'colorblind_mode': colorblind_mode
        }
        
        # Define inner function for caching
        def _create_plot(df, **kwargs):
            return _plot_tracks_internal(df, **kwargs)
        
        fig = cached_plot(_create_plot, tracks_df, 'tracks', **params)
    else:
        fig = _plot_tracks_internal(
            tracks_df,
            colormap=colormap,
            colorblind_mode=colorblind_mode,
            include_markers=include_markers,
            marker_size=marker_size,
            line_width=line_width
        )
    
    # Add title with warning if applicable
    full_title = title
    if warning:
        full_title += f"<br><sub>{warning}</sub>"
    
    fig.update_layout(
        title=full_title,
        xaxis_title="X Position",
        yaxis_title="Y Position",
        template="plotly_white",
        hovermode='closest'
    )
    
    # Apply colorblind palette if requested
    if colorblind_mode and colormap in COLORBLIND_SAFE_PALETTES:
        fig = apply_colorblind_palette(fig, colormap)
    
    # Optimize rendering based on data size
    n_tracks = tracks_df['track_id'].nunique()
    n_points = len(tracks_df)
    fig = optimize_figure_size(fig, n_points, n_tracks)
    
    # Add metadata annotation
    fig = add_plot_metadata(
        fig,
        n_tracks=n_tracks,
        n_points=n_points,
        downsampled=downsampled,
        cached=use_cache
    )
    
    return fig


def _plot_tracks_internal(
    tracks_df: pd.DataFrame,
    colormap: str = 'viridis',
    colorblind_mode: bool = False,
    include_markers: bool = True,
    marker_size: int = 5,
    line_width: int = 1
) -> go.Figure:
    """
    Internal function to create the actual plot.
    Separated for caching purposes.
    """
    fig = go.Figure()
    
    track_ids = tracks_df['track_id'].unique()
    
    # Use colorblind palette if requested
    if colorblind_mode and colormap in COLORBLIND_SAFE_PALETTES:
        colors = COLORBLIND_SAFE_PALETTES[colormap]
        if isinstance(colors, list):
            color_func = lambda i: colors[i % len(colors)]
        else:
            import plotly.express as px
            color_scale = px.colors.get_colorscale(colors)
            n_colors = len(track_ids)
            color_func = lambda i: color_scale[int(i / n_colors * (len(color_scale) - 1))][1]
    else:
        import plotly.express as px
        color_scale = px.colors.get_colorscale(colormap)
        n_colors = len(track_ids)
        color_func = lambda i: color_scale[int(i / n_colors * (len(color_scale) - 1))][1]
    
    # Add traces - use scatter with line mode for efficiency
    for i, track_id in enumerate(track_ids):
        track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        mode = 'lines+markers' if include_markers else 'lines'
        
        fig.add_trace(go.Scatter(
            x=track['x'],
            y=track['y'],
            mode=mode,
            name=f'Track {track_id}',
            line=dict(width=line_width, color=color_func(i)),
            marker=dict(size=marker_size, color=color_func(i)) if include_markers else None,
            hovertemplate='Track %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
            text=[track_id] * len(track)
        ))
    
    return fig


def plot_msd_optimized(
    msd_data: pd.DataFrame,
    colorblind_mode: bool = False,
    use_cache: bool = True,
    title: str = "Mean Squared Displacement"
) -> go.Figure:
    """
    Plot MSD curves with optimization.
    
    Parameters
    ----------
    msd_data : pd.DataFrame
        MSD data with 'lag_time' and 'msd' columns
    colorblind_mode : bool, default False
        Use colorblind-safe colors
    use_cache : bool, default True
        Use plot caching
    title : str, default "Mean Squared Displacement"
        Plot title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if msd_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No MSD data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font_size=14
        )
        return fig
    
    # Create plot
    fig = go.Figure()
    
    if 'track_id' in msd_data.columns:
        # Individual track MSDs
        for track_id in msd_data['track_id'].unique():
            track_msd = msd_data[msd_data['track_id'] == track_id]
            fig.add_trace(go.Scatter(
                x=track_msd['lag_time'],
                y=track_msd['msd'],
                mode='lines+markers',
                name=f'Track {track_id}',
                marker=dict(size=4),
                line=dict(width=1)
            ))
    else:
        # Ensemble MSD
        fig.add_trace(go.Scatter(
            x=msd_data['lag_time'],
            y=msd_data['msd'],
            mode='lines+markers',
            name='Ensemble MSD',
            marker=dict(size=6),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Lag Time (s)",
        yaxis_title="MSD (μm²)",
        template="plotly_white",
        hovermode='closest'
    )
    
    # Apply colorblind palette if requested
    if colorblind_mode:
        fig = apply_colorblind_palette(fig, 'default')
    
    return fig


# Example usage:
if __name__ == "__main__":
    # Create sample data
    import numpy as np
    
    # Generate random walks
    n_tracks = 100
    n_points = 200
    
    data = []
    for track_id in range(n_tracks):
        x = np.cumsum(np.random.randn(n_points)) * 0.1
        y = np.cumsum(np.random.randn(n_points)) * 0.1
        frames = np.arange(n_points)
        
        for i in range(n_points):
            data.append({
                'track_id': track_id,
                'frame': frames[i],
                'x': x[i],
                'y': y[i]
            })
    
    tracks_df = pd.DataFrame(data)
    
    # Test optimized plotting
    print("Creating optimized plot...")
    fig = plot_tracks_optimized(
        tracks_df,
        max_tracks=50,
        max_points_per_track=100,
        colorblind_mode=True,
        use_cache=True
    )
    
    print(f"Plot created with {len(fig.data)} traces")
    
    # Test caching
    print("\nTesting cache (should be instant)...")
    fig2 = plot_tracks_optimized(
        tracks_df,
        max_tracks=50,
        max_points_per_track=100,
        colorblind_mode=True,
        use_cache=True
    )
    
    print("Cache test complete")
