"""
State Manager for SPT Analysis Application
Centralized management of session state with type safety and error checking.
"""

from typing import Any, Dict, Optional
import pandas as pd
import datetime

# Graceful fallback if Streamlit not installed (e.g., during tests)
try:
    import streamlit as st
except ImportError:
    class _Shim:
        session_state = {}
    st = _Shim()


class StateManager:
    def __init__(self):
        # Ensure keys exist without indentation errors
        if 'raw_tracks' not in st.session_state:
            st.session_state.raw_tracks = None
        if 'pixel_size' not in st.session_state:
            st.session_state.pixel_size = 0.1
        if 'frame_interval' not in st.session_state:
            st.session_state.frame_interval = 0.1

    # ---- Tracks ----
    def get_raw_tracks(self) -> pd.DataFrame:
        df = st.session_state.get('raw_tracks_df')
        if df is None or not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        return df

    def set_raw_tracks(self, df: pd.DataFrame) -> None:
        if df is None or not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        st.session_state['raw_tracks_df'] = df

    def set_tracks(self, tracks, copy: bool = True, filename: str = None, **kwargs):
        """
        Store trajectory/track dataframe in manager.
        Args:
            tracks: pandas DataFrame or None
            copy: whether to copy the DataFrame (default True)
            filename: optional source filename (stored for reference)
            **kwargs: ignored (forward compatibility so unexpected keywords don't break)
        Side effects:
            Updates internal _tracks plus legacy aliases (tracks, track_df)
            Stores source filename in self._tracks_filename
        Returns:
            self
        """
        import pandas as pd
        if tracks is None:
            self._tracks = None
            self.tracks = None
            self.track_df = None
            self._tracks_filename = filename
            return self
        if not isinstance(tracks, pd.DataFrame):
            raise TypeError(f"tracks must be a pandas DataFrame or None, got {type(tracks)}")
        df = tracks.copy() if copy else tracks
        self._tracks = df
        # Backward compatibility attributes some code might expect
        self.tracks = df
        self.track_df = df
        self._tracks_filename = filename
        # New incremental logic
        self._tracks_loaded_at = datetime.datetime.utcnow()
        self._tracks_persisted = True
        return self

    def get_track_filename(self):
        """Return stored track source filename (or None)."""
        return getattr(self, "_tracks_filename", None)

    def get_tracks(self):
        """Return the currently stored tracks DataFrame (or None)."""
        return getattr(self, "_tracks", None)

    def get_tracks_or_none(self):
        """Alias for code paths expecting a safe getter."""
        return self.get_tracks()

    @property
    def has_tracks(self) -> bool:
        df = self.get_tracks()
        return df is not None and not df.empty if not hasattr(super(), "has_tracks") else super().has_tracks

    def tracks_metadata(self):
        """Lightweight dict for debugging/persistence info."""
        return {
            "filename": getattr(self, "_tracks_filename", None),
            "loaded_at": getattr(self, "_tracks_loaded_at", None),
            "rows": (len(self._tracks) if self.get_tracks() is not None else 0),
        }

    # ---- Units ----
    def get_pixel_size(self) -> float:
        return float(st.session_state.get('pixel_size', 1.0))

    def set_pixel_size(self, px: float) -> None:
        st.session_state['pixel_size'] = float(px)

    def get_frame_interval(self) -> float:
        return float(st.session_state.get('frame_interval', 1.0))

    def set_frame_interval(self, dt: float) -> None:
        st.session_state['frame_interval'] = float(dt)

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
    
    def import_state(self, state_data: Dict[str, Any]) -> None:
        """Import previously exported minimal application state."""
        if not isinstance(state_data, dict):
            raise TypeError("state_data must be a dict")
        
        if 'coordinates_in_microns' in state_data:
            self.set_coordinates_in_microns(state_data['coordinates_in_microns'])
        if 'frame_interval' in state_data:
            self.set_frame_interval(state_data['frame_interval'])
        if 'pixel_size' in state_data:
            self.set_pixel_size(state_data['pixel_size'])
        if 'current_file' in state_data:
            st.session_state['current_file'] = state_data['current_file']
        if 'project_id' in state_data:
            self.set_project_id(state_data['project_id'])
        if 'tracks_data' in state_data:
            tracks_df = pd.DataFrame(state_data['tracks_data'])
            self.set_tracks(tracks_df)
        # Analysis results
        if 'analysis_results' in state_data:
            try:
                ss['analysis_results'] = dict(state_data['analysis_results'])
            except Exception as e:
                st.warning(f"Failed to import analysis_results: {e}")

    def clear_data(self) -> None:
        """Clear all tracking data and dependent analysis state."""
        st.session_state['tracks_df'] = None
        st.session_state['raw_tracks'] = None
        st.session_state['current_file'] = None
        st.session_state['track_statistics'] = None
        st.session_state['analysis_results'] = {}


# Singleton helper
_STATE_MANAGER_SINGLETON: Optional[StateManager] = None

def get_state_manager() -> StateManager:
    global _STATE_MANAGER_SINGLETON
    if _STATE_MANAGER_SINGLETON is None:
        _STATE_MANAGER_SINGLETON = StateManager()
    return _STATE_MANAGER_SINGLETON

__all__ = ["StateManager", "get_state_manager"]
