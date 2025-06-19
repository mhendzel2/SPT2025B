"""
Utility functions for the SPT2025B particle tracking analysis application.
This module provides essential functions for session state management, 
track data processing, and global parameter handling.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import plotly.colors as pc
from sklearn.cluster import DBSCAN

try:
    from constants import DEFAULT_PIXEL_SIZE, DEFAULT_FRAME_INTERVAL
except ImportError:
    DEFAULT_PIXEL_SIZE = 0.1
    DEFAULT_FRAME_INTERVAL = 0.1

def initialize_session_state():
    """
    Initialize all required session state variables for the SPT2025B application.
    This function sets up default values for data storage, analysis results, and UI state.
    """
    if 'tracks_data' not in st.session_state:
        st.session_state.tracks_data = None
    if 'track_statistics' not in st.session_state:
        st.session_state.track_statistics = None
    if 'image_data' not in st.session_state:
        st.session_state.image_data = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'recent_analyses' not in st.session_state:
        st.session_state.recent_analyses = []
    
    if 'global_pixel_size' not in st.session_state:
        st.session_state.global_pixel_size = DEFAULT_PIXEL_SIZE
    if 'global_frame_interval' not in st.session_state:
        st.session_state.global_frame_interval = DEFAULT_FRAME_INTERVAL
    if 'current_pixel_size' not in st.session_state:
        st.session_state.current_pixel_size = DEFAULT_PIXEL_SIZE
    if 'current_frame_interval' not in st.session_state:
        st.session_state.current_frame_interval = DEFAULT_FRAME_INTERVAL
    if 'pixel_size' not in st.session_state:
        st.session_state.pixel_size = DEFAULT_PIXEL_SIZE
    if 'frame_interval' not in st.session_state:
        st.session_state.frame_interval = DEFAULT_FRAME_INTERVAL
    
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Home"
    if 'available_masks' not in st.session_state:
        st.session_state.available_masks = {}
    if 'mask_metadata' not in st.session_state:
        st.session_state.mask_metadata = {}
    
    if 'md_simulation' not in st.session_state:
        st.session_state.md_simulation = None
    if 'md_tracks' not in st.session_state:
        st.session_state.md_tracks = None
    if 'channel2_data' not in st.session_state:
        st.session_state.channel2_data = None
    if 'loaded_datasets' not in st.session_state:
        st.session_state.loaded_datasets = {}

def validate_tracks_dataframe(tracks_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that a tracks DataFrame has the required structure and data quality.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing particle tracks
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message) - validation result and descriptive message
    """
    if tracks_df is None:
        return False, "DataFrame is None"
    
    if tracks_df.empty:
        return False, "DataFrame is empty"
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if tracks_df[['x', 'y']].isnull().any().any():
        return False, "Contains null values in position columns"
    
    if len(tracks_df['track_id'].unique()) == 0:
        return False, "No tracks found"
    
    min_track_length = tracks_df.groupby('track_id').size().min()
    if min_track_length < 2:
        return False, f"Tracks too short (minimum length: {min_track_length})"
    
    return True, f"Valid tracks DataFrame with {len(tracks_df['track_id'].unique())} tracks"

def convert_coordinates_to_microns(tracks_df: pd.DataFrame, pixel_size: float = 0.1) -> pd.DataFrame:
    """
    Convert pixel coordinates to micrometers.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with x, y coordinates in pixels
    pixel_size : float
        Size of one pixel in micrometers
    
    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates converted to micrometers
    """
    if tracks_df is None or tracks_df.empty:
        return tracks_df
    
    converted_df = tracks_df.copy()
    
    if 'x' in converted_df.columns:
        converted_df['x'] = pd.to_numeric(converted_df['x'], errors='coerce') * pixel_size
    if 'y' in converted_df.columns:
        converted_df['y'] = pd.to_numeric(converted_df['y'], errors='coerce') * pixel_size
    
    converted_df = converted_df.dropna(subset=['x', 'y'])
    
    return converted_df

def calculate_basic_statistics(tracks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for a tracks dataset.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing particle tracks
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing basic statistics
    """
    if tracks_df is None or tracks_df.empty:
        return {}
    
    stats = {
        'total_tracks': len(tracks_df['track_id'].unique()) if 'track_id' in tracks_df.columns else 0,
        'total_points': len(tracks_df),
        'mean_track_length': tracks_df.groupby('track_id').size().mean() if 'track_id' in tracks_df.columns else 0,
        'median_track_length': tracks_df.groupby('track_id').size().median() if 'track_id' in tracks_df.columns else 0,
        'min_track_length': tracks_df.groupby('track_id').size().min() if 'track_id' in tracks_df.columns else 0,
        'max_track_length': tracks_df.groupby('track_id').size().max() if 'track_id' in tracks_df.columns else 0,
    }
    
    if 'x' in tracks_df.columns and 'y' in tracks_df.columns:
        stats.update({
            'x_range': tracks_df['x'].max() - tracks_df['x'].min(),
            'y_range': tracks_df['y'].max() - tracks_df['y'].min(),
            'mean_x': tracks_df['x'].mean(),
            'mean_y': tracks_df['y'].mean(),
        })
    
    if 'frame' in tracks_df.columns:
        stats.update({
            'total_frames': tracks_df['frame'].nunique(),
            'frame_range': tracks_df['frame'].max() - tracks_df['frame'].min() + 1,
        })
    
    return stats

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number for display with specified decimal places.
    
    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Number of decimal places
    
    Returns
    -------
    str
        Formatted number string
    """
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    
    if abs(value) >= 1000:
        return f"{value:.{decimals}e}"
    else:
        return f"{value:.{decimals}f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if denominator is zero.
    
    Parameters
    ----------
    numerator : float
        Numerator value
    denominator : float
        Denominator value
    default : float
        Default value to return if division by zero
    
    Returns
    -------
    float
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator) or np.isnan(denominator):
        return default
    return numerator / denominator

def create_download_button(data: Any, filename: str, label: str = "Download") -> None:
    """
    Create a Streamlit download button for data.
    
    Parameters
    ----------
    data : Any
        Data to download (DataFrame, dict, etc.)
    filename : str
        Name of the file to download
    label : str
        Label for the download button
    """
    if isinstance(data, pd.DataFrame):
        csv_data = data.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv_data,
            file_name=filename,
            mime='text/csv'
        )
    elif isinstance(data, dict):
        import json
        json_data = json.dumps(data, indent=2)
        st.download_button(
            label=label,
            data=json_data,
            file_name=filename,
            mime='application/json'
        )

def get_color_palette(n_colors: int = 10) -> List[str]:
    """
    Get a color palette for plotting.
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed
    
    Returns
    -------
    List[str]
        List of color hex codes
    """
    if n_colors <= 10:
        return pc.qualitative.Plotly[:n_colors]
    else:
        return pc.sample_colorscale('viridis', [i/(n_colors-1) for i in range(n_colors)])

def filter_tracks_by_length(tracks_df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    """
    Filter tracks by minimum length.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing particle tracks
    min_length : int
        Minimum track length to keep
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if tracks_df is None or tracks_df.empty:
        return tracks_df
    
    if 'track_id' not in tracks_df.columns:
        return tracks_df
    
    track_lengths = tracks_df.groupby('track_id').size()
    valid_tracks = track_lengths[track_lengths >= min_length].index
    
    return tracks_df[tracks_df['track_id'].isin(valid_tracks)].reset_index(drop=True)

def merge_close_detections(tracks_df: pd.DataFrame, distance_threshold: float = 2.0) -> pd.DataFrame:
    """
    Merge detections that are very close to each other using clustering.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing particle tracks
    distance_threshold : float
        Maximum distance for merging detections
    
    Returns
    -------
    pd.DataFrame
        DataFrame with merged detections
    """
    if tracks_df is None or tracks_df.empty:
        return tracks_df
    
    if 'x' not in tracks_df.columns or 'y' not in tracks_df.columns:
        return tracks_df
    
    merged_df = tracks_df.copy()
    
    for frame in tracks_df['frame'].unique():
        frame_data = tracks_df[tracks_df['frame'] == frame]
        
        if len(frame_data) < 2:
            continue
        
        coordinates = frame_data[['x', 'y']].values
        
        clustering = DBSCAN(eps=distance_threshold, min_samples=1)
        clusters = clustering.fit_predict(coordinates)
        
        for cluster_id in np.unique(clusters):
            cluster_points = frame_data.iloc[clusters == cluster_id]
            
            if len(cluster_points) > 1:
                mean_x = cluster_points['x'].mean()
                mean_y = cluster_points['y'].mean()
                
                first_idx = cluster_points.index[0]
                merged_df.loc[first_idx, 'x'] = mean_x
                merged_df.loc[first_idx, 'y'] = mean_y
                
                merged_df = merged_df.drop(cluster_points.index[1:])
    
    return merged_df.reset_index(drop=True)

def format_track_data(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize track data format to ensure compatibility with analysis functions.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Raw track data DataFrame
    
    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with required columns: track_id, frame, x, y
    """
    if tracks_df is None or tracks_df.empty:
        return tracks_df
    
    formatted_df = tracks_df.copy()
    
    if 'TRACK_ID' in formatted_df.columns:
        formatted_df = formatted_df[~formatted_df['TRACK_ID'].astype(str).str.contains('Track|ID|track', case=False, na=False)]
    
    column_mapping = {
        'TRACK_ID': 'track_id',
        'Track_ID': 'track_id',
        'TrackID': 'track_id',
        'track_ID': 'track_id',
        'Track': 'track_id',
        'FRAME': 'frame',
        'Frame': 'frame',
        'Time': 'frame',
        'T': 'frame',
        'POSITION_X': 'x',
        'POSITION_Y': 'y',
        'X': 'x',
        'Y': 'y',
        'Position_X': 'x',
        'Position_Y': 'y',
        'Pos_X': 'x',
        'Pos_Y': 'y'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in formatted_df.columns and new_name not in formatted_df.columns:
            formatted_df = formatted_df.rename(columns={old_name: new_name})
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in formatted_df.columns]
    
    if missing_columns:
        raise ValueError(f"Cannot format track data: missing required columns {missing_columns}")
    
    formatted_df['track_id'] = pd.to_numeric(formatted_df['track_id'], errors='coerce')
    formatted_df['frame'] = pd.to_numeric(formatted_df['frame'], errors='coerce')
    formatted_df['x'] = pd.to_numeric(formatted_df['x'], errors='coerce')
    formatted_df['y'] = pd.to_numeric(formatted_df['y'], errors='coerce')
    
    formatted_df = formatted_df.dropna(subset=['track_id', 'frame', 'x', 'y'])
    
    formatted_df['track_id'] = formatted_df['track_id'].astype(int)
    formatted_df['frame'] = formatted_df['frame'].astype(int)
    
    formatted_df = formatted_df.sort_values(['track_id', 'frame']).reset_index(drop=True)
    
    return formatted_df
