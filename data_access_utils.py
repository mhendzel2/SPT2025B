"""
Data Access Utilities for SPT Analysis Application
Provides consistent data access methods across all modules.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, Tuple

# Try to import state manager
try:
    from state_manager import StateManager
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False


def get_track_data() -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Get track data from session state using multiple fallback methods.
    
    Returns
    -------
    tuple
        (tracks_df, has_data) - DataFrame and boolean indicating if data exists
    """
    # Method 1: Try state manager first
    if STATE_MANAGER_AVAILABLE:
        try:
            state_manager = StateManager()
            if state_manager.has_data():
                return state_manager.get_tracks(), True
        except Exception:
            pass
    
    # Method 2: Check primary location
    if 'tracks_df' in st.session_state:
        df = st.session_state.tracks_df
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df, True
    
    # Method 3: Check legacy locations
    for key in ['raw_tracks', 'raw_tracks_df', 'track_data']:
        if key in st.session_state:
            df = st.session_state[key]
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, True
    
    # No data found
    return None, False


def get_analysis_results() -> Dict[str, Any]:
    """
    Get analysis results from session state.
    
    Returns
    -------
    dict
        Dictionary of analysis results
    """
    if STATE_MANAGER_AVAILABLE:
        try:
            state_manager = StateManager()
            return state_manager.get_analysis_results()
        except Exception:
            pass
    
    return st.session_state.get('analysis_results', {})


def get_units() -> Dict[str, float]:
    """
    Get unit settings (pixel size and frame interval).
    
    Returns
    -------
    dict
        Dictionary with 'pixel_size' and 'frame_interval' keys
    """
    if STATE_MANAGER_AVAILABLE:
        try:
            state_manager = StateManager()
            return {
                'pixel_size': state_manager.get_pixel_size(),
                'frame_interval': state_manager.get_frame_interval()
            }
        except Exception:
            pass
    
    return {
        'pixel_size': st.session_state.get('pixel_size', 0.1),
        'frame_interval': st.session_state.get('frame_interval', 0.1)
    }


def check_data_availability(show_error: bool = True) -> bool:
    """
    Check if track data is available and optionally show error message.
    
    Parameters
    ----------
    show_error : bool
        Whether to show error message in Streamlit UI
    
    Returns
    -------
    bool
        True if data is available, False otherwise
    """
    _, has_data = get_track_data()
    
    if not has_data and show_error:
        st.error("❌ No track data loaded. Please load data first.")
        st.info("💡 Go to the 'Data Loading' tab to upload track data.")
        
        # Debug information
        if st.checkbox("Show debug information"):
            st.write("**Session state keys:**", list(st.session_state.keys()))
            
            # Check each possible location
            debug_info = {
                'tracks_df': 'tracks_df' in st.session_state,
                'raw_tracks': 'raw_tracks' in st.session_state,
                'raw_tracks_df': 'raw_tracks_df' in st.session_state,
                'track_data': 'track_data' in st.session_state
            }
            st.write("**Data locations:**", debug_info)
            
            # Show data summary if state manager available
            if STATE_MANAGER_AVAILABLE:
                try:
                    state_manager = StateManager()
                    st.write("**State manager summary:**", state_manager.get_data_summary())
                    st.write("**Debug state:**", state_manager.debug_data_state())
                except Exception as e:
                    st.write(f"State manager error: {e}")
    
    return has_data


def display_data_summary():
    """Display a summary of loaded data."""
    tracks_df, has_data = get_track_data()
    
    if has_data:
        st.success(f"✅ Track data loaded: {len(tracks_df)} points")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'track_id' in tracks_df.columns:
                st.metric("Number of Tracks", tracks_df['track_id'].nunique())
            else:
                st.metric("Number of Points", len(tracks_df))
        
        with col2:
            if 'frame' in tracks_df.columns:
                st.metric("Time Points", tracks_df['frame'].nunique())
        
        with col3:
            units = get_units()
            st.metric("Pixel Size (μm)", f"{units['pixel_size']:.3f}")
    else:
        st.warning("No track data loaded")
