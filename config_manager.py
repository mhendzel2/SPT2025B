"""
Configuration Management System for SPT Analysis
Handles loading and managing user-configurable parameters from config.toml
"""

import os
import toml
from typing import Dict, Any, Optional
from pathlib import Path
import streamlit as st

class ConfigManager:
    """Manages application configuration from config.toml file."""
    
    def __init__(self, config_file: str = "config.toml"):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_file : str
            Path to the configuration file
        """
        self.config_file = config_file
        self._config = {}
        self._default_config = self._get_default_config()
        self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values as fallback."""
        return {
            'analysis': {
                'max_lag': 20,
                'min_track_length': 5,
                'fit_method': 'linear',
                'analyze_anomalous': True,
                'check_confinement': True,
                'motion_window_size': 5,
                'velocity_autocorr': True,
                'persistence_analysis': True,
                'motion_classification': 'basic',
                'clustering_method': 'DBSCAN',
                'clustering_epsilon': 0.5,
                'clustering_min_samples': 3
            },
            'tracking': {
                'max_search_radius': 10.0,
                'memory': 3,
                'min_track_length_tracking': 5,
                'detection_threshold': 0.01,
                'noise_size': 1,
                'smoothing_size': 1.5
            },
            'microscopy': {
                'pixel_size': 0.16,
                'frame_interval': 0.1,
                'temperature': 298.15
            },
            'visualization': {
                'track_colormap': 'viridis',
                'figure_dpi': 150,
                'figure_size_width': 10,
                'figure_size_height': 8
            },
            'rheology': {
                'particle_radius': 0.5e-6,
                'temperature_k': 298.15,
                'viscosity_medium': 1e-3
            },
            'file_handling': {
                'max_file_size_mb': 1024,
                'chunk_size': 10000,
                'supported_formats': ["csv", "xlsx", "xml", "mvd2", "tiff", "tif"]
            },
            'performance': {
                'max_workers': 4,
                'memory_limit_gb': 10,
                'enable_gpu': False,
                'batch_size': 1000
            },
            'ui': {
                'default_tab': 'Data Loading',
                'show_advanced_options': False,
                'auto_save_results': True,
                'result_cache_size': 100
            }
        }
    
    def load_config(self) -> None:
        """Load configuration from file, falling back to defaults if file doesn't exist."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self._config = toml.load(f)
                    
                # Merge with defaults to ensure all required keys exist
                self._config = self._merge_configs(self._default_config, self._config)
            else:
                # Create default config file if it doesn't exist
                self._config = self._default_config.copy()
                self.save_config()
        except Exception as e:
            st.warning(f"Error loading config file: {e}. Using default configuration.")
            self._config = self._default_config.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults."""
        merged = default.copy()
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                toml.dump(self._config, f)
        except Exception as e:
            st.error(f"Error saving config file: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Parameters
        ----------
        section : str
            Configuration section name
        key : str
            Configuration key name
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        try:
            return self._config.get(section, {}).get(key, default)
        except:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Parameters
        ----------
        section : str
            Configuration section name
        key : str
            Configuration key name
        value : Any
            Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Parameters
        ----------
        section : str
            Section name
            
        Returns
        -------
        Dict[str, Any]
            Configuration section
        """
        return self._config.get(section, {})
    
    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """
        Update multiple values in a configuration section.
        
        Parameters
        ----------
        section : str
            Section name
        values : Dict[str, Any]
            Values to update
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section].update(values)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._default_config.copy()
        self.save_config()
    
    def get_analysis_defaults(self) -> Dict[str, Any]:
        """Get default analysis parameters."""
        return self.get_section('analysis')
    
    def get_tracking_defaults(self) -> Dict[str, Any]:
        """Get default tracking parameters."""
        return self.get_section('tracking')
    
    def get_microscopy_defaults(self) -> Dict[str, Any]:
        """Get default microscopy parameters."""
        return self.get_section('microscopy')
    
    def get_ui_defaults(self) -> Dict[str, Any]:
        """Get default UI parameters."""
        return self.get_section('ui')

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_default_value(section: str, key: str, fallback: Any = None) -> Any:
    """Convenience function to get configuration values."""
    return get_config_manager().get(section, key, fallback)