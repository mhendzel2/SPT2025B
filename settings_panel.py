"""
Unified Settings Panel

Centralized settings management to eliminate redundant parameter controls
throughout the application. Single source of truth for all global settings.
"""

import streamlit as st
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import os


@dataclass
class GlobalSettings:
    """Global settings container."""
    # Spatial calibration
    pixel_size: float = 0.1  # micrometers
    
    # Temporal calibration
    frame_interval: float = 0.1  # seconds
    
    # Physical parameters
    temperature: float = 300.0  # Kelvin
    viscosity: float = 0.001  # Pa·s (water at 20°C)
    
    # Analysis defaults
    min_track_length: int = 5
    max_lag_frames: int = 20
    
    # Visualization defaults
    default_colormap: str = "viridis"
    line_width: int = 2
    marker_size: int = 5
    
    # Performance
    use_parallel: bool = True
    num_workers: int = 4
    max_memory_mb: int = 1000
    
    # File handling
    auto_save: bool = True
    save_interval_minutes: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pixel_size': self.pixel_size,
            'frame_interval': self.frame_interval,
            'temperature': self.temperature,
            'viscosity': self.viscosity,
            'min_track_length': self.min_track_length,
            'max_lag_frames': self.max_lag_frames,
            'default_colormap': self.default_colormap,
            'line_width': self.line_width,
            'marker_size': self.marker_size,
            'use_parallel': self.use_parallel,
            'num_workers': self.num_workers,
            'max_memory_mb': self.max_memory_mb,
            'auto_save': self.auto_save,
            'save_interval_minutes': self.save_interval_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlobalSettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def save_to_file(self, filepath: str = ".spt_settings.json"):
        """Save settings to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str = ".spt_settings.json") -> 'GlobalSettings':
        """Load settings from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        return cls()


class SettingsPanel:
    """
    Unified settings panel for display in sidebar.
    
    Features:
    - Single source of truth for all global settings
    - Preset configurations for common microscopy setups
    - Live unit conversion preview
    - Save/load settings
    - Validation and bounds checking
    """
    
    # Microscopy presets
    PRESETS = {
        'Confocal (63x, 1.4 NA)': {
            'pixel_size': 0.065,
            'frame_interval': 0.1,
            'description': 'High-resolution confocal microscopy'
        },
        'Confocal (100x, 1.4 NA)': {
            'pixel_size': 0.041,
            'frame_interval': 0.1,
            'description': 'Ultra-high resolution confocal'
        },
        'TIRF (100x, 1.49 NA)': {
            'pixel_size': 0.107,
            'frame_interval': 0.05,
            'description': 'Total Internal Reflection Fluorescence'
        },
        'Widefield (60x, 1.4 NA)': {
            'pixel_size': 0.108,
            'frame_interval': 0.1,
            'description': 'Standard widefield microscopy'
        },
        'Spinning Disk (60x)': {
            'pixel_size': 0.108,
            'frame_interval': 0.033,
            'description': 'Fast spinning disk confocal (30 fps)'
        },
        'Light Sheet (20x, 1.0 NA)': {
            'pixel_size': 0.325,
            'frame_interval': 0.05,
            'description': 'Light sheet fluorescence microscopy'
        }
    }
    
    def __init__(self):
        """Initialize settings panel."""
        # Get or create global settings
        if 'global_settings' not in st.session_state:
            st.session_state.global_settings = GlobalSettings.load_from_file()
        
        self.settings = st.session_state.global_settings
    
    def show_compact_sidebar(self):
        """Show compact version in sidebar (essential settings only)."""
        st.sidebar.divider()
        st.sidebar.markdown("### ⚙️ Global Settings")
        
        # Quick preset selector
        preset_names = ['Custom'] + list(self.PRESETS.keys())
        selected_preset = st.sidebar.selectbox(
            "Microscopy Setup",
            preset_names,
            help="Quick presets for common microscopy configurations"
        )
        
        if selected_preset != 'Custom':
            preset = self.PRESETS[selected_preset]
            self.settings.pixel_size = preset['pixel_size']
            self.settings.frame_interval = preset['frame_interval']
            st.sidebar.caption(f"ℹ️ {preset['description']}")
        
        # Essential calibration
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            self.settings.pixel_size = st.number_input(
                "Pixel Size (µm)",
                min_value=0.001,
                max_value=10.0,
                value=self.settings.pixel_size,
                step=0.001,
                format="%.3f",
                key="master_pixel_size",
                help="Physical size of one pixel. Used in ALL spatial analyses."
            )
        
        with col2:
            self.settings.frame_interval = st.number_input(
                "Frame Interval (s)",
                min_value=0.001,
                max_value=10.0,
                value=self.settings.frame_interval,
                step=0.001,
                format="%.3f",
                key="master_frame_interval",
                help="Time between frames. Used in ALL temporal analyses."
            )
        
        # Live preview
        with st.sidebar.expander("📊 Unit Preview", expanded=False):
            st.caption(f"""
            **Conversions:**
            • 1 pixel = {self.settings.pixel_size:.3f} µm
            • 1 frame = {self.settings.frame_interval:.3f} s
            • 1 px/frame = {self.settings.pixel_size/self.settings.frame_interval:.3f} µm/s
            • 1 px²/frame = {self.settings.pixel_size**2/self.settings.frame_interval:.3f} µm²/s
            
            **Field of View (typical 512×512 px):**
            • Width: {512 * self.settings.pixel_size:.1f} µm
            • Time/frame: {512 * self.settings.frame_interval:.1f} s
            """)
        
        # Save/Load buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("💾 Save", help="Save current settings", use_container_width=True):
                self.settings.save_to_file()
                st.sidebar.success("✓ Saved")
        with col2:
            if st.button("📁 Load", help="Load saved settings", use_container_width=True):
                st.session_state.global_settings = GlobalSettings.load_from_file()
                st.rerun()
    
    def show_full_settings_page(self):
        """Show comprehensive settings page."""
        st.title("⚙️ Global Settings")
        
        st.markdown("""
        **Centralized Configuration**  
        These settings apply to all analyses throughout the application.  
        Changes here will immediately affect all pages and modules.
        """)
        
        # Calibration section
        st.divider()
        st.header("🔬 Microscopy Calibration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Preset Configurations")
            for name, preset in self.PRESETS.items():
                if st.button(name, use_container_width=True):
                    self.settings.pixel_size = preset['pixel_size']
                    self.settings.frame_interval = preset['frame_interval']
                    st.rerun()
            
            st.caption("Click a preset to apply settings")
        
        with col2:
            st.subheader("Custom Calibration")
            
            self.settings.pixel_size = st.number_input(
                "Pixel Size (µm)",
                min_value=0.001,
                max_value=10.0,
                value=self.settings.pixel_size,
                step=0.001,
                format="%.3f",
                help="Physical size of one pixel in micrometers"
            )
            
            self.settings.frame_interval = st.number_input(
                "Frame Interval (s)",
                min_value=0.001,
                max_value=10.0,
                value=self.settings.frame_interval,
                step=0.001,
                format="%.3f",
                help="Time between consecutive frames in seconds"
            )
            
            self.settings.temperature = st.number_input(
                "Temperature (K)",
                min_value=200.0,
                max_value=400.0,
                value=self.settings.temperature,
                step=1.0,
                help="Temperature for Stokes-Einstein calculations"
            )
            
            self.settings.viscosity = st.number_input(
                "Medium Viscosity (Pa·s)",
                min_value=0.0001,
                max_value=0.1,
                value=self.settings.viscosity,
                step=0.0001,
                format="%.4f",
                help="Viscosity of surrounding medium (water ≈ 0.001 Pa·s)"
            )
        
        # Analysis defaults
        st.divider()
        st.header("📊 Analysis Defaults")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Track Filtering")
            
            self.settings.min_track_length = st.slider(
                "Minimum Track Length (frames)",
                min_value=3,
                max_value=100,
                value=self.settings.min_track_length,
                help="Tracks shorter than this will be filtered out"
            )
            
            self.settings.max_lag_frames = st.slider(
                "Default Max Lag (frames)",
                min_value=5,
                max_value=200,
                value=self.settings.max_lag_frames,
                help="Default maximum lag for MSD calculations"
            )
        
        with col2:
            st.subheader("Visualization")
            
            self.settings.default_colormap = st.selectbox(
                "Default Colormap",
                ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "rainbow"],
                index=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "rainbow"].index(self.settings.default_colormap),
                help="Default color scheme for visualizations"
            )
            
            self.settings.line_width = st.slider(
                "Line Width",
                min_value=1,
                max_value=10,
                value=self.settings.line_width,
                help="Default line width for track plots"
            )
            
            self.settings.marker_size = st.slider(
                "Marker Size",
                min_value=1,
                max_value=20,
                value=self.settings.marker_size,
                help="Default marker size for scatter plots"
            )
        
        # Performance settings
        st.divider()
        st.header("⚡ Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Computation")
            
            self.settings.use_parallel = st.checkbox(
                "Enable Parallel Processing",
                value=self.settings.use_parallel,
                help="Use multiple CPU cores for faster analysis"
            )
            
            if self.settings.use_parallel:
                import psutil
                max_workers = psutil.cpu_count()
                self.settings.num_workers = st.slider(
                    "Worker Threads",
                    min_value=1,
                    max_value=max_workers,
                    value=min(self.settings.num_workers, max_workers),
                    help=f"Number of parallel workers (max: {max_workers})"
                )
        
        with col2:
            st.subheader("Memory Management")
            
            self.settings.max_memory_mb = st.slider(
                "Max Cache Size (MB)",
                min_value=100,
                max_value=5000,
                value=self.settings.max_memory_mb,
                step=100,
                help="Maximum memory for caching analysis results"
            )
        
        # File handling
        st.divider()
        st.header("💾 File Handling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.settings.auto_save = st.checkbox(
                "Auto-save Settings",
                value=self.settings.auto_save,
                help="Automatically save settings on changes"
            )
        
        with col2:
            if self.settings.auto_save:
                self.settings.save_interval_minutes = st.slider(
                    "Save Interval (minutes)",
                    min_value=1,
                    max_value=30,
                    value=self.settings.save_interval_minutes
                )
        
        # Save/Reset buttons
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                self.settings.save_to_file()
                st.success("✓ Settings saved successfully!")
        
        with col2:
            if st.button("📁 Load Settings", use_container_width=True):
                st.session_state.global_settings = GlobalSettings.load_from_file()
                st.info("✓ Settings loaded from file")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                st.session_state.global_settings = GlobalSettings()
                st.warning("✓ Reset to default values")
                st.rerun()
        
        with col4:
            if st.button("📤 Export", use_container_width=True):
                json_str = json.dumps(self.settings.to_dict(), indent=2)
                st.download_button(
                    label="Download Settings JSON",
                    data=json_str,
                    file_name="spt_settings.json",
                    mime="application/json"
                )
    
    def get_units(self) -> Dict[str, float]:
        """Get current unit settings as dictionary (for compatibility)."""
        return {
            'pixel_size': self.settings.pixel_size,
            'frame_interval': self.settings.frame_interval,
            'temperature': self.settings.temperature,
            'viscosity': self.settings.viscosity
        }


# Singleton pattern for easy access
_settings_panel_instance = None

def get_settings_panel() -> SettingsPanel:
    """Get or create global settings panel instance."""
    global _settings_panel_instance
    if _settings_panel_instance is None:
        _settings_panel_instance = SettingsPanel()
    return _settings_panel_instance


def get_global_units() -> Dict[str, float]:
    """Quick access to unit settings."""
    panel = get_settings_panel()
    return panel.get_units()


if __name__ == "__main__":
    # Test the settings panel
    import streamlit as st
    
    st.set_page_config(page_title="Settings Test", layout="wide")
    
    # Show compact sidebar
    panel = get_settings_panel()
    panel.show_compact_sidebar()
    
    # Show full settings page
    panel.show_full_settings_page()
