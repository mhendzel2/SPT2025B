"""
State Manager for SPT Analysis Application
Centralized management of session state with type safety and error checking.
"""

import datetime
from typing import Any, Dict, Optional

import pandas as pd

# Graceful fallback if Streamlit not installed (e.g., during tests)
try:
    import streamlit as st
except ImportError:
    class _Shim:
        session_state = {}
    st = _Shim()


class StateManager:
    def __init__(self):
        # Initialize required session state keys if they don't exist
        keys_to_init = {
            'raw_tracks': None,
            'tracks_data': None,
            'pixel_size': 0.1,
            'frame_interval': 0.1,
            'image_data': None,
            'mask_images': None,
            'track_statistics': None,
            'analysis_results': {},
            'current_file': None,
            'coordinates_in_microns': False,
            'current_project_id': None,
            'channel_labels': {},
        }
        for key, value in keys_to_init.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # ---- Tracks Data ----
    def set_tracks(self, tracks: Optional[pd.DataFrame], copy: bool = True, filename: str = None):
        """
        Store trajectory/track dataframe in the state manager.
        This is the primary method for setting track data.
        """
        if tracks is None:
            st.session_state.tracks_data = None
            self._tracks_filename = filename
            return

        if not isinstance(tracks, pd.DataFrame):
            raise TypeError(f"tracks must be a pandas DataFrame or None, got {type(tracks)}")

        st.session_state.tracks_data = tracks.copy() if copy else tracks
        self._tracks_filename = filename
        self._tracks_loaded_at = datetime.datetime.utcnow()

    def get_tracks(self) -> Optional[pd.DataFrame]:
        """Return the currently stored tracks DataFrame (or None)."""
        return st.session_state.get('tracks_data')

    def has_tracks(self) -> bool:
        """Return True if any track dataframe is available and non-empty."""
        df = self.get_tracks()
        return df is not None and not df.empty

    def load_tracks(self, df: pd.DataFrame, source: str = None):
        """Load track data and reset related analysis state."""
        self.set_tracks(df, filename=source)
        st.session_state.track_statistics = None
        st.session_state.analysis_results = {}

    def get_track_filename(self) -> Optional[str]:
        """Return stored track source filename (or None)."""
        return getattr(self, "_tracks_filename", None)

    # ---- Image Data ----
    def set_image_data(self, image_data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Load image data with optional metadata."""
        st.session_state.image_data = image_data
        if metadata is not None:
            if 'image_metadata' not in st.session_state:
                st.session_state.image_metadata = {}
            if isinstance(metadata, dict):
                st.session_state.image_metadata.update(metadata)

    def get_image_data(self) -> Optional[Any]:
        """Get image data."""
        return st.session_state.get('image_data')

    def has_image_data(self) -> bool:
        """Check if image data is available."""
        return self.get_image_data() is not None

    # ---- Units and Settings ----
    def set_pixel_size(self, value: float):
        """Set pixel size."""
        if value <= 0:
            raise ValueError("Pixel size must be positive")
        st.session_state.pixel_size = value

    def get_pixel_size(self) -> float:
        """Get pixel size in microns."""
        return float(st.session_state.get('pixel_size', 0.1))

    def set_frame_interval(self, value: float):
        """Set frame interval."""
        if value <= 0:
            raise ValueError("Frame interval must be positive")
        st.session_state.frame_interval = value

    def get_frame_interval(self) -> float:
        """Get frame interval in seconds."""
        return float(st.session_state.get('frame_interval', 0.1))

    def set_coordinates_in_microns(self, value: bool):
        """Set coordinate units."""
        st.session_state.coordinates_in_microns = value

    def get_coordinates_in_microns(self) -> bool:
        """Check if coordinates are in microns."""
        return st.session_state.get('coordinates_in_microns', False)

    # ---- Analysis and Results ----
    def set_analysis_results(self, analysis_type: str, results: Dict[str, Any]):
        """Store analysis results."""
        st.session_state.analysis_results[analysis_type] = results

    def get_analysis_results(self, analysis_type: str = None) -> Dict[str, Any]:
        """Get analysis results."""
        if analysis_type:
            return st.session_state.analysis_results.get(analysis_type, {})
        return st.session_state.analysis_results

    def set_track_statistics(self, stats: Optional[pd.DataFrame]):
        """Set track statistics."""
        st.session_state.track_statistics = stats

    def get_track_statistics(self) -> Optional[pd.DataFrame]:
        """Get track statistics."""
        return st.session_state.get('track_statistics')

    # ---- State Management ----
    def clear_all_data(self) -> None:
        """Clear all tracking data, image data, and dependent analysis state."""
        self.set_tracks(None)
        self.set_image_data(None)
        st.session_state.mask_images = None
        st.session_state.track_statistics = None
        st.session_state.analysis_results = {}
        st.session_state.current_file = None
        st.session_state.detections = None
        st.session_state.all_detections = {}

    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of current data state."""
        summary = {
            'has_tracks': self.has_tracks(),
            'has_image': self.has_image_data(),
            'current_file': st.session_state.get('current_file', "Unknown"),
            'coordinates_in_microns': self.get_coordinates_in_microns(),
            'frame_interval': self.get_frame_interval(),
            'pixel_size': self.get_pixel_size(),
            'project_id': st.session_state.get('current_project_id'),
            'analysis_types': list(st.session_state.analysis_results.keys())
        }

        if self.has_tracks():
            tracks_df = self.get_tracks()
            summary.update({
                'n_tracks': tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 0,
                'n_points': len(tracks_df),
                'columns': list(tracks_df.columns)
            })
        return summary


# Singleton helper
_STATE_MANAGER_SINGLETON: Optional[StateManager] = None

def get_state_manager() -> StateManager:
    global _STATE_MANAGER_SINGLETON
    if _STATE_MANAGER_SINGLETON is None:
        _STATE_MANAGER_SINGLETON = StateManager()
    return _STATE_MANAGER_SINGLETON

__all__ = ["StateManager", "get_state_manager"]
