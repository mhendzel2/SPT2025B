"""
State Manager for SPT Analysis Application
Centralized management of session state with type safety and error checking.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List

class StateManager:
    """
    Centralized state management for the SPT Analysis application.
    Provides type-safe access to session state with fallbacks.
    """
    
    def __init__(self):
        """Initialize the state manager."""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables with defaults."""
        defaults = {
            'raw_tracks': None,  # Standardized key for main DataFrame
            'current_file': None,
            'coordinates_in_microns': False,
            'frame_interval': 0.01,
            'pixel_size': 0.03,
            'analysis_results': {},
            'current_project_id': None,
            'track_statistics': None,
            'image_data': None,
            'segmentation_results': None,
            'parameter_optimization_results': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def has_data(self) -> bool:
        """Check if tracking data is available."""
        return (st.session_state.get('raw_tracks') is not None and 
                not st.session_state.get('raw_tracks', pd.DataFrame()).empty)
    
    def get_tracks(self) -> Optional[pd.DataFrame]:
        """Get the current tracking data (alias for get_raw_tracks)."""
        return st.session_state.get('raw_tracks')
    
    def get_raw_tracks(self) -> Optional[pd.DataFrame]:
        """Get the current raw tracking data."""
        return st.session_state.get('raw_tracks')
    
    def set_tracks(self, df: pd.DataFrame, filename: str = None):
        """Set tracking data with validation (alias for set_raw_tracks)."""
        if df is None or df.empty:
            st.warning("Attempted to set empty or None DataFrame to raw_tracks.")
            st.session_state.raw_tracks = None
            st.session_state.current_file = None
            return
        
        # Basic validation
        required_cols = ['x', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")
        
        st.session_state.raw_tracks = df
        if filename:
            st.session_state.current_file = filename

    def set_raw_tracks(self, df: pd.DataFrame, filename: str = None):
        """Set raw tracking data with validation."""
        if df is None or df.empty:
            st.warning("Attempted to set empty or None DataFrame to raw_tracks.")
            st.session_state.raw_tracks = None
            st.session_state.current_file = None
            return
        
        # Basic validation
        required_cols = ['x', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")
        
        st.session_state.raw_tracks = df
        if filename:
            st.session_state.current_file = filename
    
    def get_current_file(self) -> str:
        """Get the current file name."""
        return st.session_state.current_file or "Unknown"
    
    def get_coordinates_in_microns(self) -> bool:
        """Check if coordinates are in microns."""
        return st.session_state.coordinates_in_microns
    
    def set_coordinates_in_microns(self, value: bool):
        """Set coordinate units."""
        st.session_state.coordinates_in_microns = value
    
    def get_frame_interval(self) -> float:
        """Get frame interval in seconds."""
        return st.session_state.frame_interval
    
    def set_frame_interval(self, value: float):
        """Set frame interval."""
        if value <= 0:
            raise ValueError("Frame interval must be positive")
        st.session_state.frame_interval = value
    
    def get_pixel_size(self) -> float:
        """Get pixel size in microns."""
        return st.session_state.pixel_size
    
    def set_pixel_size(self, value: float):
        """Set pixel size."""
        if value <= 0:
            raise ValueError("Pixel size must be positive")
        st.session_state.pixel_size = value

    def load_tracks(self, df: pd.DataFrame, source: str = None):
        """Load track data and reset related state."""
        self.set_tracks(df, filename=source)
        st.session_state.track_statistics = None
        st.session_state.analysis_results = {}

    def load_image_data(self, image_data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Load image data with optional metadata."""
        self.set_image_data(image_data)
        if metadata is not None:
            if 'image_metadata' not in st.session_state:
                st.session_state.image_metadata = {}
            if isinstance(metadata, dict):
                st.session_state.image_metadata.update(metadata)
    

    
    def set_loaded_image(self, image_data):
        """Set loaded image data."""
        st.session_state.image_data = image_data
    
    def get_loaded_images(self):
        """Get loaded image data."""
        return st.session_state.get('image_data', {})
    
    def are_coordinates_in_microns(self) -> bool:
        """Check if coordinates are in microns."""
        return st.session_state.coordinates_in_microns
    
    def set_detections(self, detections):
        """Set particle detections."""
        st.session_state.detections = detections
    
    def get_detections(self):
        """Get particle detections."""
        return st.session_state.get('detections', None)
    
    def get_analysis_results(self, analysis_type: str = None) -> Dict[str, Any]:
        """Get analysis results."""
        if analysis_type:
            return st.session_state.analysis_results.get(analysis_type, {})
        return st.session_state.analysis_results
    
    def set_analysis_results(self, analysis_type: str, results: Dict[str, Any]):
        """Store analysis results."""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        st.session_state.analysis_results[analysis_type] = results
    
    def get_project_id(self) -> Optional[str]:
        """Get current project ID."""
        return st.session_state.current_project_id
    
    def set_project_id(self, project_id: str):
        """Set current project ID."""
        st.session_state.current_project_id = project_id
    
    def get_track_statistics(self) -> Optional[Dict[str, Any]]:
        """Get track statistics."""
        return st.session_state.track_statistics
    
    def set_track_statistics(self, stats: Dict[str, Any]):
        """Set track statistics."""
        st.session_state.track_statistics = stats
    
    def has_image_data(self) -> bool:
        """Check if image data is available."""
        return st.session_state.image_data is not None
    
    def get_image_data(self) -> Optional[Any]:
        """Get image data."""
        return st.session_state.image_data
    
    def set_image_data(self, image_data: Any):
        """Set image data."""
        st.session_state.image_data = image_data
    
    def get_segmentation_results(self) -> Optional[Dict[str, Any]]:
        """Get segmentation results."""
        return st.session_state.segmentation_results
    
    def set_segmentation_results(self, results: Dict[str, Any]):
        """Set segmentation results."""
        st.session_state.segmentation_results = results
    
    def get_secondary_channel_data(self) -> Optional[pd.DataFrame]:
        """Get secondary channel data."""
        return st.session_state.get('channel2_data', None) or st.session_state.get('secondary_channel_data', None)
    
    def set_secondary_channel_data(self, df: pd.DataFrame, channel_name: str = None):
        """Set secondary channel data with validation."""
        if df is None or df.empty:
            st.warning("Attempted to set empty or None DataFrame for secondary channel.")
            st.session_state.channel2_data = None
            st.session_state.secondary_channel_data = None
            return
        
        # Basic validation
        required_cols = ['x', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in secondary channel DataFrame: {missing_cols}")
        
        # Store in both possible keys for compatibility
        st.session_state.channel2_data = df
        st.session_state.secondary_channel_data = df
        
        if channel_name:
            st.session_state.secondary_channel_name = channel_name
    
    def has_secondary_channel_data(self) -> bool:
        """Check if secondary channel data is available."""
        secondary_data = self.get_secondary_channel_data()
        return secondary_data is not None and not secondary_data.empty
    
    def get_channel_labels(self) -> Dict[str, str]:
        """Get channel labels for intensity columns."""
        return st.session_state.get('channel_labels', {})
    
    def set_channel_labels(self, labels: Dict[str, str]):
        """Set channel labels for intensity columns."""
        st.session_state.channel_labels = labels
    
    def clear_data(self):
        """Clear all tracking data."""
        st.session_state.raw_tracks = None
        st.session_state.current_file = None
        st.session_state.track_statistics = None
        st.session_state.analysis_results = {}
    
    def clear_analysis_results(self):
        """Clear analysis results."""
        st.session_state.analysis_results = {}
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of current data state."""
        summary = {
            'has_tracks': self.has_data(),
            'has_image': self.has_image_data(),
            'current_file': self.get_current_file(),
            'coordinates_in_microns': self.get_coordinates_in_microns(),
            'frame_interval': self.get_frame_interval(),
            'pixel_size': self.get_pixel_size(),
            'project_id': self.get_project_id(),
            'analysis_types': list(st.session_state.analysis_results.keys())
        }
        
        if self.has_data():
            tracks_df = self.get_tracks()
            summary.update({
                'n_tracks': tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 0,
                'n_points': len(tracks_df),
                'columns': list(tracks_df.columns)
            })
        
        return summary
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for backup/restore."""
        exportable_state = {}
        
        # Export basic settings
        exportable_state['coordinates_in_microns'] = self.get_coordinates_in_microns()
        exportable_state['frame_interval'] = self.get_frame_interval()
        exportable_state['pixel_size'] = self.get_pixel_size()
        exportable_state['current_file'] = self.get_current_file()
        exportable_state['project_id'] = self.get_project_id()
        
        # Export data if available
        if self.has_data():
            exportable_state['tracks_data'] = self.get_tracks().to_dict('records')
        
        # Export analysis results
        exportable_state['analysis_results'] = st.session_state.analysis_results
        
        return exportable_state
    
    def import_state(self, state_data: Dict[str, Any]):
        """Import state from backup."""
        # Import basic settings
        if 'coordinates_in_microns' in state_data:
            self.set_coordinates_in_microns(state_data['coordinates_in_microns'])
        
        if 'frame_interval' in state_data:
            self.set_frame_interval(state_data['frame_interval'])
        
        if 'pixel_size' in state_data:
            self.set_pixel_size(state_data['pixel_size'])
        
        if 'current_file' in state_data:
            st.session_state.current_file = state_data['current_file']
        
        if 'project_id' in state_data:
            self.set_project_id(state_data['project_id'])
        
        # Import tracking data
        if 'tracks_data' in state_data:
            tracks_df = pd.DataFrame(state_data['tracks_data'])
            self.set_tracks(tracks_df)
        
        # Import analysis results
        if 'analysis_results' in state_data:
            st.session_state.analysis_results = state_data['analysis_results']

# Global state manager instance
_state_manager = None

def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
