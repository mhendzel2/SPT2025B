"""
Enhanced Error Handling for SPT Analysis Application
Provides user-friendly error messages and validation functions based on external review recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

class SPTAnalysisError(Exception):
    """Base exception class for SPT analysis errors."""
    pass

class DataValidationError(SPTAnalysisError):
    """Raised when input data fails validation."""
    pass

class AnalysisError(SPTAnalysisError):
    """Raised when analysis calculations fail."""
    pass

class ConfigurationError(SPTAnalysisError):
    """Raised when parameters are incorrectly configured."""
    pass

def validate_track_data(tracks_df: pd.DataFrame, operation: str = "analysis") -> None:
    """
    Validate track data with user-friendly error messages.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data to validate
    operation : str
        Description of the operation being performed for context
    """
    if tracks_df is None:
        raise DataValidationError(
            f"No tracking data available for {operation}. "
            "Please load track data first using the Data Loading page."
        )
    
    if tracks_df.empty:
        raise DataValidationError(
            f"The loaded track data is empty and cannot be used for {operation}. "
            "Please check your data file or try loading a different dataset."
        )
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    
    if missing_columns:
        available_cols = list(tracks_df.columns)
        raise DataValidationError(
            f"Track data is missing required columns for {operation}: {', '.join(missing_columns)}.\n"
            f"Available columns: {', '.join(available_cols)}\n"
            f"Please ensure your data contains:\n"
            f"- 'track_id' or 'particle': unique identifier for each track\n"
            f"- 'frame' or 't': time frame number\n"
            f"- 'x', 'y': particle positions"
        )

def validate_msd_parameters(max_lag: int, pixel_size: float, frame_interval: float, 
                           min_track_length: int) -> None:
    """
    Validate MSD calculation parameters.
    """
    if max_lag <= 0:
        raise ConfigurationError(
            "Maximum lag time must be greater than 0. "
            "This parameter determines how far into the future to look when calculating displacement. "
            "Typical values range from 10-50 frames."
        )
    
    if pixel_size <= 0:
        raise ConfigurationError(
            "Pixel size must be greater than 0 micrometers. "
            "This converts pixel coordinates to physical distances. "
            "Check your microscope settings - typical values range from 0.05-0.5 µm/pixel."
        )
    
    if frame_interval <= 0:
        raise ConfigurationError(
            "Frame interval must be greater than 0 seconds. "
            "This is the time between consecutive images. "
            "Check your acquisition settings - typical values range from 0.01-1.0 seconds."
        )
    
    if min_track_length < 2:
        raise ConfigurationError(
            "Minimum track length must be at least 2 points to calculate displacement. "
            "For reliable diffusion analysis, consider using at least 10-20 points per track."
        )

def validate_diffusion_analysis_data(tracks_df: pd.DataFrame, msd_df: Optional[pd.DataFrame] = None) -> None:
    """
    Validate data specifically for diffusion analysis.
    """
    validate_track_data(tracks_df, "diffusion analysis")
    
    # Check if tracks are long enough for meaningful analysis
    track_lengths = tracks_df.groupby('track_id').size()
    short_tracks = (track_lengths < 5).sum()
    total_tracks = len(track_lengths)
    
    if short_tracks == total_tracks:
        raise AnalysisError(
            "All tracks are too short for reliable diffusion analysis (< 5 points). "
            f"Found {total_tracks} tracks with lengths: {track_lengths.describe()}\n"
            "For meaningful diffusion analysis, tracks should contain at least 10-20 points. "
            "Consider adjusting your tracking parameters or using longer image sequences."
        )
    
    if short_tracks > total_tracks * 0.8:
        raise AnalysisError(
            f"Most tracks are too short for reliable analysis ({short_tracks}/{total_tracks} < 5 points). "
            "Consider adjusting tracking parameters to generate longer tracks."
        )
    
    if msd_df is not None and msd_df.empty:
        raise AnalysisError(
            "MSD calculation produced no results. This usually means:\n"
            "- All tracks are shorter than the minimum required length\n"
            "- The maximum lag time is too large for the available tracks\n"
            "- There's an issue with the track data format"
        )

def validate_statistical_analysis(data: Union[List, np.ndarray, pd.Series], 
                                 analysis_type: str) -> None:
    """
    Validate data for statistical analysis with suggestions.
    """
    if len(data) == 0:
        raise AnalysisError(
            f"No data available for {analysis_type}. "
            "Please ensure your analysis has produced results before running statistical tests."
        )
    
    if len(data) < 3:
        raise AnalysisError(
            f"Insufficient data for {analysis_type} (only {len(data)} values). "
            "Statistical analysis requires at least 3 data points. "
            "Consider combining data from multiple tracks or conditions."
        )
    
    # Check for appropriate sample sizes for different tests
    if analysis_type == "comparative analysis":
        if len(data) < 10:
            raise AnalysisError(
                f"Small sample size for comparative analysis (n={len(data)}). "
                "For reliable statistical comparisons, consider:\n"
                "- Collecting more data (aim for n≥10 per condition)\n"
                "- Using non-parametric tests for small samples\n"
                "- Interpreting results with caution"
            )

def suggest_statistical_test(data1: Union[List, np.ndarray], data2: Union[List, np.ndarray]) -> str:
    """
    Suggest appropriate statistical test based on data characteristics.
    """
    from scipy.stats import shapiro, levene
    
    n1, n2 = len(data1), len(data2)
    
    # Sample size considerations
    if n1 < 5 or n2 < 5:
        return "Fisher's exact test or Mann-Whitney U test (small sample sizes)"
    
    if n1 < 30 or n2 < 30:
        suggested_test = "Mann-Whitney U test (non-parametric, recommended for small samples)"
    else:
        # Test for normality
        try:
            _, p1 = shapiro(data1)
            _, p2 = shapiro(data2)
            
            if p1 > 0.05 and p2 > 0.05:
                # Both normal, test for equal variances
                try:
                    _, p_var = levene(data1, data2)
                    if p_var > 0.05:
                        suggested_test = "Independent t-test (equal variances)"
                    else:
                        suggested_test = "Welch's t-test (unequal variances)"
                except:
                    suggested_test = "Independent t-test"
            else:
                suggested_test = "Mann-Whitney U test (non-normal distribution detected)"
        except:
            suggested_test = "Mann-Whitney U test (unable to test normality)"
    
    return suggested_test

def format_analysis_error(error: Exception, context: str = "") -> str:
    """
    Format analysis errors with helpful context and suggestions.
    """
    error_msg = str(error)
    
    if "tracks_df cannot be empty" in error_msg:
        return (
            "No tracking data loaded. Please load your track data first:\n"
            "1. Go to the 'Data Loading' page\n"
            "2. Upload a CSV file with track data, or\n"
            "3. Select a sample dataset to test the analysis"
        )
    
    if "Missing required columns" in error_msg:
        return (
            f"Data format issue: {error_msg}\n\n"
            "Your data file should contain these columns:\n"
            "- track_id (or particle): unique number for each particle track\n"
            "- frame (or t): time point number\n"
            "- x, y: particle positions (in pixels or micrometers)\n\n"
            "Optional columns: z (for 3D), intensity, etc."
        )
    
    if "too short for" in error_msg:
        return (
            f"Track length issue: {error_msg}\n\n"
            "Suggestions:\n"
            "- Use longer image sequences during acquisition\n"
            "- Adjust tracking parameters to link particles across more frames\n"
            "- Lower the minimum track length requirement (with caution)\n"
            "- Filter your data to include only high-quality tracks"
        )
    
    # Add context if provided
    if context:
        return f"Error in {context}: {error_msg}"
    
    return error_msg

def check_data_quality(tracks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess data quality and provide recommendations.
    """
    quality_report = {
        "total_tracks": 0,
        "total_points": 0,
        "track_lengths": {},
        "quality_issues": [],
        "recommendations": []
    }
    
    if tracks_df.empty:
        quality_report["quality_issues"].append("No data loaded")
        return quality_report
    
    # Basic statistics
    quality_report["total_tracks"] = tracks_df['track_id'].nunique()
    quality_report["total_points"] = len(tracks_df)
    
    # Track length analysis
    track_lengths = tracks_df.groupby('track_id').size()
    quality_report["track_lengths"] = {
        "mean": float(track_lengths.mean()),
        "median": float(track_lengths.median()),
        "min": int(track_lengths.min()),
        "max": int(track_lengths.max()),
        "std": float(track_lengths.std())
    }
    
    # Quality assessment
    short_tracks = (track_lengths < 5).sum()
    medium_tracks = ((track_lengths >= 5) & (track_lengths < 20)).sum()
    long_tracks = (track_lengths >= 20).sum()
    
    if short_tracks > quality_report["total_tracks"] * 0.5:
        quality_report["quality_issues"].append(f"Many short tracks ({short_tracks}/{quality_report['total_tracks']} < 5 points)")
        quality_report["recommendations"].append("Consider adjusting tracking parameters to generate longer tracks")
    
    if long_tracks < quality_report["total_tracks"] * 0.1:
        quality_report["quality_issues"].append("Few long tracks for reliable diffusion analysis")
        quality_report["recommendations"].append("Acquire longer image sequences or optimize tracking settings")
    
    # Check for temporal gaps
    for track_id, track_data in tracks_df.groupby('track_id'):
        frames = sorted(track_data['frame'].values)
        gaps = np.diff(frames)
        if np.any(gaps > 1):
            quality_report["quality_issues"].append(f"Track {track_id} has temporal gaps")
            quality_report["recommendations"].append("Check for missing frames or tracking interruptions")
            break  # Report only first instance
    
    # Overall quality score
    if not quality_report["quality_issues"]:
        quality_report["overall_quality"] = "Good"
    elif len(quality_report["quality_issues"]) <= 2:
        quality_report["overall_quality"] = "Fair"
    else:
        quality_report["overall_quality"] = "Poor"
    
    return quality_report