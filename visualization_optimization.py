"""
Visualization Optimization Utilities for SPT2025B.
Provides caching, downsampling, and performance optimization for plots.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple, List
import hashlib
import json

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# Colorblind-safe palettes
COLORBLIND_SAFE_PALETTES = {
    'default': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', 
                '#CA9161', '#949494', '#ECE133', '#56B4E9'],
    'contrast': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
    'viridis': 'viridis',  # Built-in colorblind-friendly
    'cividis': 'cividis',  # Built-in colorblind-friendly
    'paul_tol': ['#332288', '#88CCEE', '#44AA99', '#117733',
                 '#999933', '#DDCC77', '#CC6677', '#882255']
}


def downsample_track(
    track_df: pd.DataFrame,
    max_points: int,
    method: str = 'uniform'
) -> pd.DataFrame:
    """
    Downsample a single track to reduce plotting overhead.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track data sorted by frame
    max_points : int
        Maximum number of points to keep
    method : str, default 'uniform'
        Downsampling method: 'uniform', 'random', or 'temporal'
    
    Returns
    -------
    pd.DataFrame
        Downsampled track
    """
    if len(track_df) <= max_points:
        return track_df
    
    if method == 'uniform':
        # Uniform sampling across track length
        indices = np.linspace(0, len(track_df)-1, max_points, dtype=int)
        return track_df.iloc[indices]
    
    elif method == 'random':
        # Random sampling
        return track_df.sample(n=max_points, random_state=42)
    
    elif method == 'temporal':
        # Keep every nth point
        n = len(track_df) // max_points
        return track_df.iloc[::max(1, n)]
    
    else:
        raise ValueError(f"Unknown downsampling method: {method}")


def downsample_tracks(
    tracks_df: pd.DataFrame,
    max_tracks: Optional[int] = None,
    max_points_per_track: Optional[int] = None,
    method: str = 'uniform',
    seed: int = 42
) -> pd.DataFrame:
    """
    Downsample tracks for efficient visualization.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with 'track_id', 'frame', 'x', 'y' columns
    max_tracks : int, optional
        Maximum number of tracks to keep
    max_points_per_track : int, optional
        Maximum points per individual track
    method : str, default 'uniform'
        Downsampling method for points
    seed : int, default 42
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Downsampled tracks
    """
    if tracks_df.empty:
        return tracks_df
    
    # Filter to max_tracks first
    if max_tracks is not None:
        track_ids = tracks_df['track_id'].unique()
        if len(track_ids) > max_tracks:
            np.random.seed(seed)
            selected_tracks = np.random.choice(track_ids, max_tracks, replace=False)
            tracks_df = tracks_df[tracks_df['track_id'].isin(selected_tracks)]
    
    # Downsample individual tracks
    if max_points_per_track is not None and max_points_per_track > 0:
        downsampled_tracks = []
        
        for track_id in tracks_df['track_id'].unique():
            track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track) > max_points_per_track:
                track = downsample_track(track, max_points_per_track, method)
            
            downsampled_tracks.append(track)
        
        tracks_df = pd.concat(downsampled_tracks, ignore_index=True)
    
    return tracks_df


def get_plot_cache_key(
    tracks_df: pd.DataFrame,
    plot_type: str,
    params: Dict[str, Any]
) -> str:
    """
    Generate a cache key for a plot.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    plot_type : str
        Type of plot (e.g., 'tracks', 'msd', 'histogram')
    params : Dict
        Plot parameters
    
    Returns
    -------
    str
        Cache key
    """
    # Hash the track data
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(tracks_df).values
    ).hexdigest()[:16]
    
    # Hash the parameters (only hashable types)
    cacheable_params = {
        k: v for k, v in params.items()
        if isinstance(v, (int, float, str, bool, tuple))
    }
    param_str = json.dumps(cacheable_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    return f"{plot_type}_{data_hash}_{param_hash}"


class PlotCache:
    """Cache for generated plots."""
    
    def __init__(self, max_size: int = 20):
        """
        Initialize plot cache.
        
        Parameters
        ----------
        max_size : int, default 20
            Maximum number of plots to cache
        """
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    def get(self, cache_key: str) -> Optional[go.Figure]:
        """
        Retrieve a cached plot.
        
        Parameters
        ----------
        cache_key : str
            Cache key
        
        Returns
        -------
        Optional[go.Figure]
            Cached figure or None
        """
        if cache_key in self._cache:
            # Update access order (move to end)
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]
        return None
    
    def set(self, cache_key: str, figure: go.Figure):
        """
        Store a plot in cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key
        figure : go.Figure
            Figure to cache
        """
        # Add to cache
        self._cache[cache_key] = figure
        self._access_order.append(cache_key)
        
        # Remove oldest if over limit
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
    
    def clear(self):
        """Clear the cache."""
        self._cache = {}
        self._access_order = []
    
    def size(self) -> int:
        """Get number of cached plots."""
        return len(self._cache)


# Global plot cache
_plot_cache = PlotCache()


def get_plot_cache() -> PlotCache:
    """Get the global plot cache."""
    return _plot_cache


def cached_plot(
    plot_func,
    tracks_df: pd.DataFrame,
    plot_type: str,
    **kwargs
) -> go.Figure:
    """
    Wrapper to add caching to plot functions.
    
    Parameters
    ----------
    plot_func : callable
        Plot function to cache
    tracks_df : pd.DataFrame
        Track data
    plot_type : str
        Type of plot
    **kwargs : dict
        Plot parameters
    
    Returns
    -------
    go.Figure
        Plot figure (cached or newly generated)
    """
    cache = get_plot_cache()
    cache_key = get_plot_cache_key(tracks_df, plot_type, kwargs)
    
    # Try to get from cache
    cached_fig = cache.get(cache_key)
    if cached_fig is not None:
        # Add cache indicator to title
        if hasattr(cached_fig, 'layout') and hasattr(cached_fig.layout, 'title'):
            return cached_fig
    
    # Generate plot
    fig = plot_func(tracks_df, **kwargs)
    
    # Store in cache
    cache.set(cache_key, fig)
    
    return fig


def apply_colorblind_palette(
    fig: go.Figure,
    palette: str = 'default'
) -> go.Figure:
    """
    Apply a colorblind-safe color palette to a figure.
    
    Parameters
    ----------
    fig : go.Figure
        Figure to modify
    palette : str, default 'default'
        Palette name
    
    Returns
    -------
    go.Figure
        Modified figure
    """
    if palette not in COLORBLIND_SAFE_PALETTES:
        return fig
    
    colors = COLORBLIND_SAFE_PALETTES[palette]
    
    # Apply to traces
    for i, trace in enumerate(fig.data):
        if isinstance(colors, list):
            color = colors[i % len(colors)]
            if hasattr(trace, 'marker'):
                trace.marker.color = color
            if hasattr(trace, 'line'):
                trace.line.color = color
    
    return fig


def optimize_figure_size(
    fig: go.Figure,
    n_points: int,
    n_traces: int
) -> go.Figure:
    """
    Optimize figure rendering based on data size.
    
    Parameters
    ----------
    fig : go.Figure
        Figure to optimize
    n_points : int
        Total number of data points
    n_traces : int
        Number of traces
    
    Returns
    -------
    go.Figure
        Optimized figure
    """
    # Reduce marker size for large datasets
    if n_points > 10000:
        for trace in fig.data:
            if hasattr(trace, 'marker') and trace.marker.size:
                trace.marker.size = max(1, trace.marker.size // 2)
    
    # Simplify line rendering for many traces
    if n_traces > 50:
        for trace in fig.data:
            if hasattr(trace, 'line'):
                trace.line.simplify = True
    
    # Disable hover for very large datasets
    if n_points > 50000:
        fig.update_traces(hoverinfo='skip')
    
    return fig


def create_data_size_warning(
    n_tracks: int,
    n_points: int,
    max_tracks: int = 100,
    max_points: int = 50000
) -> Optional[str]:
    """
    Generate a warning message for large datasets.
    
    Parameters
    ----------
    n_tracks : int
        Number of tracks
    n_points : int
        Total number of points
    max_tracks : int, default 100
        Threshold for track warning
    max_points : int, default 50000
        Threshold for points warning
    
    Returns
    -------
    Optional[str]
        Warning message or None
    """
    warnings = []
    
    if n_tracks > max_tracks:
        warnings.append(f"{n_tracks} tracks (showing subset)")
    
    if n_points > max_points:
        warnings.append(f"{n_points:,} points (downsampled)")
    
    if warnings:
        return "⚠️ Large dataset: " + ", ".join(warnings)
    
    return None


def add_plot_metadata(
    fig: go.Figure,
    n_tracks: int,
    n_points: int,
    downsampled: bool = False,
    cached: bool = False
) -> go.Figure:
    """
    Add metadata annotations to a plot.
    
    Parameters
    ----------
    fig : go.Figure
        Figure to annotate
    n_tracks : int
        Number of tracks displayed
    n_points : int
        Number of points displayed
    downsampled : bool, default False
        Whether data was downsampled
    cached : bool, default False
        Whether plot was retrieved from cache
    
    Returns
    -------
    go.Figure
        Annotated figure
    """
    metadata_text = f"Tracks: {n_tracks} | Points: {n_points:,}"
    
    if downsampled:
        metadata_text += " (downsampled)"
    
    if cached:
        metadata_text += " (cached)"
    
    fig.add_annotation(
        text=metadata_text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        xanchor='right', yanchor='bottom',
        showarrow=False,
        font=dict(size=10, color='gray'),
        bgcolor='rgba(255, 255, 255, 0.8)'
    )
    
    return fig


def estimate_plot_memory(n_points: int, n_traces: int) -> str:
    """
    Estimate memory usage of a plot.
    
    Parameters
    ----------
    n_points : int
        Number of data points
    n_traces : int
        Number of traces
    
    Returns
    -------
    str
        Estimated memory usage
    """
    # Rough estimate: ~100 bytes per point + overhead per trace
    bytes_per_point = 100
    bytes_per_trace = 10000
    
    total_bytes = n_points * bytes_per_point + n_traces * bytes_per_trace
    
    if total_bytes < 1024:
        return f"{total_bytes} bytes"
    elif total_bytes < 1024**2:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024**3:
        return f"{total_bytes / (1024**2):.1f} MB"
    else:
        return f"{total_bytes / (1024**3):.1f} GB"
