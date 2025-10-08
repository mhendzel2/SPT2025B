"""
Utility functions for the SPT2025B particle tracking analysis application.
This module provides essential functions for session state management,
track data processing, and global parameter handling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
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
    if 'mask_images' not in st.session_state:
        st.session_state.mask_images = None

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

def validate_tracks_dataframe(tracks_df: pd.DataFrame, 
                             check_duplicates: bool = True,
                             check_continuity: bool = False,
                             max_frame_gap: int = 10) -> Tuple[bool, str]:
    """
    Enhanced validation for tracks DataFrame with comprehensive data quality checks.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing particle tracks
    check_duplicates : bool, default True
        Check for duplicate track/frame combinations
    check_continuity : bool, default False
        Check for large gaps in frame sequences (slower)
    max_frame_gap : int, default 10
        Maximum allowed gap in frames for continuity check

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

    # Check for null values
    null_counts = tracks_df[['x', 'y']].isnull().sum()
    if null_counts.any():
        total_nulls = null_counts.sum()
        return False, f"Contains {total_nulls} null values in position columns"

    if len(tracks_df['track_id'].unique()) == 0:
        return False, "No tracks found"

    # Check for negative frames
    if (tracks_df['frame'] < 0).any():
        return False, "Contains negative frame numbers"
    
    # Check for duplicate positions in same track
    if check_duplicates:
        duplicates = tracks_df.groupby(['track_id', 'frame']).size()
        if (duplicates > 1).any():
            n_dups = (duplicates > 1).sum()
            return False, f"Contains {n_dups} duplicate track/frame combinations"
    
    # Check frame continuity if requested
    if check_continuity:
        for track_id in tracks_df['track_id'].unique()[:10]:  # Check first 10 tracks
            track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            frames = track['frame'].values
            if len(frames) > 1:
                gaps = np.diff(frames)
                if (gaps > max_frame_gap).any():
                    max_gap = gaps.max()
                    return False, f"Track {track_id} has large frame gap ({max_gap} frames)"

    min_track_length = tracks_df.groupby('track_id').size().min()
    if min_track_length < 2:
        return False, f"Tracks too short (minimum length: {min_track_length})"

    n_tracks = len(tracks_df['track_id'].unique())
    return True, f"Valid tracks DataFrame with {n_tracks} tracks"

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
    This function combines the logic of the two previous implementations.
    """
    if tracks_df is None or tracks_df.empty:
        return pd.DataFrame()

    formatted_df = tracks_df.copy()

    # Define a comprehensive column mapping
    column_mapping = {
        'TRACK_ID': 'track_id', 'Track_ID': 'track_id', 'TrackID': 'track_id',
        'Track ID': 'track_id', 'track_ID': 'track_id', 'Track': 'track_id', 
        'particle': 'track_id', 'trajectory': 'track_id', 'id': 'track_id',
        'FRAME': 'frame', 'Frame': 'frame', 'Time': 'frame', 'time': 'frame',
        't': 'frame', 'T': 'frame', 'timepoint': 'frame',
        'POSITION_X': 'x', 'Position_X': 'x', 'Pos_X': 'x', 'X': 'x',
        'POSITION_Y': 'y', 'Position_Y': 'y', 'Pos_Y': 'y', 'Y': 'y',
        'POSITION_Z': 'z', 'Position_Z': 'z', 'Pos_Z': 'z', 'Z': 'z',
        'POSITION_T': 'frame', 'Position_T': 'frame', 'Pos_T': 'frame',
        'Quality': 'quality', 'QUALITY': 'quality',
        'SNR': 'snr', 'snr': 'snr',
        'Intensity': 'intensity', 'intensity': 'intensity',
    }

    # Rename columns based on the mapping
    # Use a loop to handle cases where multiple old names map to the same new name
    for old_name, new_name in column_mapping.items():
        if old_name in formatted_df.columns and new_name not in formatted_df.columns:
            formatted_df.rename(columns={old_name: new_name}, inplace=True)

    # Define required columns
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in formatted_df.columns]

    if missing_columns:
        raise ValueError(f"Cannot format track data: missing required columns {missing_columns}")

    # Convert columns to numeric types, coercing errors
    for col in required_columns + ['z']:
        if col in formatted_df.columns:
            formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')

    # Drop rows where essential columns are NaN
    formatted_df.dropna(subset=required_columns, inplace=True)

    # Convert integer columns to int type
    for col in ['track_id', 'frame']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].astype(int)

    # Sort the DataFrame and reset the index
    formatted_df = formatted_df.sort_values(['track_id', 'frame']).reset_index(drop=True)

    return formatted_df

def calculate_track_statistics(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic statistics for each track.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data in standard format

    Returns
    -------
    pd.DataFrame
        DataFrame with track statistics
    """
    if tracks_df.empty:
        return pd.DataFrame()

    # Group by track_id and calculate statistics
    stats = []

    for track_id, track_data in tracks_df.groupby('track_id'):
        # Sort by frame to ensure correct calculations
        track_data = track_data.sort_values('frame')

        # Basic info
        track_length = len(track_data)
        duration = track_data['frame'].max() - track_data['frame'].min() + 1

        # Single point tracks need special handling
        if track_length == 1:
            stat = {
                'track_id': track_id,
                'track_length': track_length,
                'duration': duration,
                'start_frame': track_data['frame'].iloc[0],
                'end_frame': track_data['frame'].iloc[0],
                'total_distance': 0.0,
                'net_displacement': 0.0,
                'straightness': 0.0,
                'mean_speed': 0.0,
                'x_min': track_data['x'].min(),
                'x_max': track_data['x'].max(),
                'y_min': track_data['y'].min(),
                'y_max': track_data['y'].max(),
                'x_std': 0.0,
                'y_std': 0.0
            }
        else:
            # Calculate distances between consecutive points
            dx = track_data['x'].diff()
            dy = track_data['y'].diff()
            step_distances = np.sqrt(dx**2 + dy**2)
            total_distance = step_distances.sum()

            # Calculate net displacement (start to end)
            x_start, y_start = track_data[['x', 'y']].iloc[0]
            x_end, y_end = track_data[['x', 'y']].iloc[-1]
            net_displacement = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)

            # Calculate straightness (0-1, with 1 being a perfectly straight line)
            straightness = net_displacement / total_distance if total_distance > 0 else 0

            # Calculate mean speed (distance per frame)
            mean_speed = total_distance / (duration - 1) if duration > 1 else 0

            stat = {
                'track_id': track_id,
                'track_length': track_length,
                'duration': duration,
                'start_frame': track_data['frame'].min(),
                'end_frame': track_data['frame'].max(),
                'total_distance': total_distance,
                'net_displacement': net_displacement,
                'straightness': straightness,
                'mean_speed': mean_speed,
                'x_min': track_data['x'].min(),
                'x_max': track_data['x'].max(),
                'y_min': track_data['y'].min(),
                'y_max': track_data['y'].max(),
                'x_std': track_data['x'].std(),
                'y_std': track_data['y'].std()
            }

            # Add MSD at different lag times if track is long enough
            if track_length >= 4:
                msd_values = calculate_msd_single_track(track_data, max_lag=min(10, track_length-1))
                for lag, msd in enumerate(msd_values):
                    if lag > 0:  # Skip lag 0
                        stat[f'msd_lag{lag}'] = msd

        stats.append(stat)

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)

    return stats_df

def calculate_msd_single_track(track_data: pd.DataFrame, max_lag: int = 10) -> List[float]:
    """
    DEPRECATED: Use msd_calculation.calculate_msd_single_track() instead.
    
    Calculate mean squared displacement for a single track.
    This function is maintained for backwards compatibility.
    """
    from msd_calculation import calculate_msd_single_track as msd_calc
    
    # Convert to DataFrame format expected by new function
    result_df = msd_calc(track_data, max_lag=max_lag)
    
    # Convert back to list format for compatibility
    if result_df.empty:
        return [np.nan] * (max_lag + 1)
    
    return result_df['msd'].tolist()

def convert_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to Excel bytes for download.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert

    Returns
    -------
    bytes
        Excel file as bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = output.getvalue()
    return excel_data

def convert_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for download.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert

    Returns
    -------
    bytes
        CSV file as bytes
    """
    return df.to_csv(index=False).encode('utf-8')

def sync_global_parameters():
    """Synchronize global parameters between widgets and session state"""
    if 'global_pixel_size' in st.session_state:
        st.session_state.current_pixel_size = st.session_state.global_pixel_size

    if 'global_frame_interval' in st.session_state:
        st.session_state.current_frame_interval = st.session_state.global_frame_interval

def get_global_pixel_size():
    """Get the current global pixel size with proper fallback"""
    return st.session_state.get('current_pixel_size',
                               st.session_state.get('global_pixel_size', 0.1))

def get_global_frame_interval():
    """Get the current global frame interval with proper fallback"""
    return st.session_state.get('current_frame_interval',
                               st.session_state.get('global_frame_interval', 0.1))

def create_analysis_record(name: str, analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a record of an analysis for storage in session state.

    Parameters
    ----------
    name : str
        Name of the analysis
    analysis_type : str
        Type of analysis performed
    parameters : dict
        Parameters used for the analysis

    Returns
    -------
    dict
        Analysis record with ID, timestamp, and input values
    """
    import uuid
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now()

    return {
        'id': analysis_id,
        'name': name,
        'type': analysis_type,
        'parameters': parameters,
        'date': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'datetime': timestamp
    }

def validate_track_data(tracks_df):
    """Validate track data format and return status with potential error message."""
    if tracks_df is None or len(tracks_df) == 0:
        return False, "Track data is empty"

    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]

    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    # Make sure there's at least one track
    if tracks_df['track_id'].nunique() == 0:
        return False, "No tracks found in data (track_id column is empty)"

    # Check for numeric data in coordinate columns
    for col in ['x', 'y']:
        if not pd.api.types.is_numeric_dtype(tracks_df[col]):
            return False, f"Column '{col}' must contain numeric values"

    return True, "Track data is valid"

def ensure_session_state_consistency():
    """Ensure consistency of session state variables for track data."""
    import streamlit as st

    # Initialize track loading flag if not present
    if 'tracks_loaded' not in st.session_state:
        st.session_state.tracks_loaded = False

    # Check if tracks are actually available when tracks_loaded is True
    if st.session_state.get('tracks_loaded', False):
        if 'tracks_df' not in st.session_state or st.session_state.tracks_df is None or st.session_state.tracks_df.empty:
            # Inconsistent state detected - fix it
            st.session_state.tracks_loaded = False

    # Ensure track_metadata exists if tracks are loaded
    if st.session_state.get('tracks_loaded', False) and 'track_metadata' not in st.session_state:
        # Create minimal metadata
        if 'tracks_df' in st.session_state and st.session_state.tracks_df is not None:
            tracks_df = st.session_state.tracks_df
            st.session_state.track_metadata = {
                'file_name': 'Unknown file',
                'num_tracks': tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 0,
                'num_frames': tracks_df['frame'].nunique() if 'frame' in tracks_df.columns else 0,
                'total_localizations': len(tracks_df),
                'pixel_size': st.session_state.get('pixel_size', 0.16),
                'frame_interval': st.session_state.get('frame_interval', 0.1)
            }

    # Ensure pixel_size and frame_interval are in session state if tracks are loaded
    if st.session_state.get('tracks_loaded', False):
        if 'pixel_size' not in st.session_state and 'track_metadata' in st.session_state:
            st.session_state.pixel_size = st.session_state.track_metadata.get('pixel_size', 0.16)

        if 'frame_interval' not in st.session_state and 'track_metadata' in st.session_state:
            st.session_state.frame_interval = st.session_state.track_metadata.get('frame_interval', 0.1)
