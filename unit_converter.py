"""
Unit converter utilities for the SPT Analysis application.
Converts between different units for spatial and temporal data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

class UnitConverter:
    """
    Class for handling unit conversions in track data.
    
    Supports conversion between:
    - Spatial units: pixels, micrometers (μm), nanometers (nm)
    - Temporal units: frames, seconds (s), milliseconds (ms)
    """
    
    def __init__(self):
        # Define conversion factors (all relative to base units: μm and seconds)
        self.spatial_conversion = {
            'pixel': None,  # Set dynamically based on pixel size
            'μm': 1.0,      # Base unit
            'nm': 0.001,    # 1 nm = 0.001 μm
        }
        
        self.temporal_conversion = {
            'frame': None,  # Set dynamically based on frame interval
            's': 1.0,       # Base unit
            'ms': 0.001,    # 1 ms = 0.001 s
        }
        
        # Default values
        self.pixel_size = 0.1     # Default: 0.1 μm/pixel
        self.frame_interval = 0.03  # Default: 0.03 s/frame
        
        # Set computed conversion factors
        self.update_conversion_factors()
    
    def update_conversion_factors(self):
        """Update the conversion factors based on current pixel size and frame interval."""
        self.spatial_conversion['pixel'] = self.pixel_size
        self.temporal_conversion['frame'] = self.frame_interval
    
    def set_pixel_size(self, size: float, unit: str = 'μm'):
        """
        Set the pixel size with specified unit.
        
        Parameters
        ----------
        size : float
            Pixel size value
        unit : str
            Unit of the pixel size ('μm' or 'nm')
        """
        if unit == 'μm':
            self.pixel_size = size
        elif unit == 'nm':
            self.pixel_size = size * self.spatial_conversion['nm']
        else:
            raise ValueError(f"Unsupported unit: {unit}. Use 'μm' or 'nm'.")
        
        self.update_conversion_factors()
    
    def set_frame_interval(self, interval: float, unit: str = 's'):
        """
        Set the frame interval with specified unit.
        
        Parameters
        ----------
        interval : float
            Frame interval value
        unit : str
            Unit of the frame interval ('s' or 'ms')
        """
        if unit == 's':
            self.frame_interval = interval
        elif unit == 'ms':
            self.frame_interval = interval * self.temporal_conversion['ms']
        else:
            raise ValueError(f"Unsupported unit: {unit}. Use 's' or 'ms'.")
        
        self.update_conversion_factors()
    
    def convert_distance(self, value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        Convert distance values between units.
        
        Parameters
        ----------
        value : float or numpy.ndarray
            Distance value(s) to convert
        from_unit : str
            Source unit ('pixel', 'μm', or 'nm')
        to_unit : str
            Target unit ('pixel', 'μm', or 'nm')
            
        Returns
        -------
        float or numpy.ndarray
            Converted distance value(s)
        """
        if from_unit not in self.spatial_conversion or to_unit not in self.spatial_conversion:
            raise ValueError(f"Unsupported units. Use 'pixel', 'μm', or 'nm'.")
        
        # Convert to base unit (μm)
        value_in_base = value * self.spatial_conversion[from_unit]
        
        # Convert from base unit to target unit
        return value_in_base / self.spatial_conversion[to_unit]
    
    def convert_time(self, value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        Convert time values between units.
        
        Parameters
        ----------
        value : float or numpy.ndarray
            Time value(s) to convert
        from_unit : str
            Source unit ('frame', 's', or 'ms')
        to_unit : str
            Target unit ('frame', 's', or 'ms')
            
        Returns
        -------
        float or numpy.ndarray
            Converted time value(s)
        """
        if from_unit not in self.temporal_conversion or to_unit not in self.temporal_conversion:
            raise ValueError(f"Unsupported units. Use 'frame', 's', or 'ms'.")
        
        # Convert to base unit (s)
        value_in_base = value * self.temporal_conversion[from_unit]
        
        # Convert from base unit to target unit
        return value_in_base / self.temporal_conversion[to_unit]
    
    def convert_tracks_data(self, tracks_df: pd.DataFrame, 
                           space_from: str = 'pixel', space_to: str = 'μm',
                           time_from: str = 'frame', time_to: str = 's') -> pd.DataFrame:
        """
        Convert track data coordinates and time values to new units.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data in standard format (with 'track_id', 'frame', 'x', 'y', etc.)
        space_from : str
            Source spatial unit
        space_to : str
            Target spatial unit
        time_from : str
            Source temporal unit
        time_to : str
            Target temporal unit
            
        Returns
        -------
        pd.DataFrame
            Converted track data
        """
        # Make a copy to avoid modifying the original data
        converted_df = tracks_df.copy()
        
        # Convert spatial coordinates
        if space_from != space_to:
            if 'x' in converted_df.columns:
                converted_df['x'] = self.convert_distance(converted_df['x'].values, space_from, space_to)
            if 'y' in converted_df.columns:
                converted_df['y'] = self.convert_distance(converted_df['y'].values, space_from, space_to)
            if 'z' in converted_df.columns:
                converted_df['z'] = self.convert_distance(converted_df['z'].values, space_from, space_to)
        
        # Convert temporal values
        if time_from != time_to and 'frame' in converted_df.columns:
            converted_df['time'] = self.convert_time(converted_df['frame'].values, time_from, time_to)
        
        # Add metadata about units
        converted_df.attrs['space_unit'] = space_to
        converted_df.attrs['time_unit'] = time_to
        
        return converted_df
    
    def convert_diffusion_coefficient(self, d_value: float, 
                                     space_from: str = 'μm', time_from: str = 's',
                                     space_to: str = 'μm', time_to: str = 's') -> float:
        """
        Convert diffusion coefficient between units.
        
        Parameters
        ----------
        d_value : float
            Diffusion coefficient value
        space_from : str
            Source spatial unit
        time_from : str
            Source temporal unit
        space_to : str
            Target spatial unit
        time_to : str
            Target temporal unit
            
        Returns
        -------
        float
            Converted diffusion coefficient
        """
        # Diffusion coefficient unit: space^2/time
        # Convert space squared first
        space_conversion = (self.spatial_conversion[space_from] / self.spatial_conversion[space_to])**2
        
        # Then convert time
        time_conversion = self.temporal_conversion[time_to] / self.temporal_conversion[time_from]
        
        # Apply both conversions
        return d_value * space_conversion * time_conversion
    
    def get_unit_text(self, space_unit: str = 'μm', time_unit: str = 's') -> Dict[str, str]:
        """
        Get formatted text for units in various contexts.
        
        Parameters
        ----------
        space_unit : str
            Spatial unit
        time_unit : str
            Temporal unit
            
        Returns
        -------
        dict
            Dictionary with formatted unit text for different quantities
        """
        return {
            'space': space_unit,
            'time': time_unit,
            'velocity': f"{space_unit}/{time_unit}",
            'diffusion': f"{space_unit}²/{time_unit}",
            'msd': f"{space_unit}²"
        }