"""
Report generator for single particle tracking analysis.
"""

import pandas as pd
import numpy as np
import warnings

# Import analysis functions with graceful fallback
try:
    from analysis import analyze_diffusion, analyze_motion, calculate_msd
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    warnings.warn("analysis module not fully available")

# Import biophysical models with graceful fallback  
try:
    from biophysical_models import analyze_motion_models
    BIOPHYSICAL_AVAILABLE = True
except ImportError:
    BIOPHYSICAL_AVAILABLE = False
    analyze_motion_models = None

# Import visualization functions with graceful fallback
try:
    from visualization import (
        plot_msd_curves, plot_diffusion_coefficients, plot_motion_analysis
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import intensity analysis with graceful fallback
try:
    from intensity_analysis import (
        correlate_intensity_movement, 
        analyze_intensity_profiles,
        extract_intensity_channels
    )
    INTENSITY_ANALYSIS_AVAILABLE = True
except ImportError:
    INTENSITY_ANALYSIS_AVAILABLE = False

class SPTReportGenerator:
    """Generates analysis reports for single particle tracking data."""
    
    def generate_report(self, tracks_df, analyses=None, title=None, pixel_size=0.16, frame_interval=0.1):
        """
        Generate a comprehensive report from tracking data.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            DataFrame containing track data with columns 'track_id', 'frame', 'x', 'y'
        analyses : list
            List of analysis types to include
        title : str
            Report title
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
            
        Returns
        -------
        dict
            Report containing analysis results and figures
        """
        # Validate input data
        from utils import validate_track_data
        is_valid, error_msg = validate_track_data(tracks_df)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'title': title or "Failed Report",
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        # Log the data we received
        import logging
        logging.info(f"Generating report with tracks_df shape: {tracks_df.shape}, "
                    f"pixel_size: {pixel_size}, frame_interval: {frame_interval}")
        logging.info(f"First few rows: {tracks_df.head().to_dict()}")
        
        # Set default analyses if none provided
        if analyses is None:
            analyses = ['basic_statistics', 'msd_analysis', 'diffusion_analysis']
            
        # Set default title if none provided
        if title is None:
            title = "Single Particle Tracking Analysis Report"
        
        # Initialize report structure
        report = {
            'title': title,
            'analyses': analyses,
            'success': True,
            'summary': {},
            'analysis_results': {},
            'figures': {},
            'metadata': {
                'num_tracks': len(tracks_df['track_id'].unique()),
                'num_frames': len(tracks_df['frame'].unique()),
                'total_localizations': len(tracks_df),
                'pixel_size': pixel_size,
                'frame_interval': frame_interval,
                'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Make a clean copy of the track data
        tracks = tracks_df.copy()
        
        # Add physical units if not present
        if 'x_um' not in tracks.columns:
            tracks['x_um'] = tracks['x'] * pixel_size
        if 'y_um' not in tracks.columns:
            tracks['y_um'] = tracks['y'] * pixel_size
        if 'time_s' not in tracks.columns:
            tracks['time_s'] = tracks['frame'] * frame_interval
        
        # Perform requested analyses
        try:
            # Basic statistics (compute locally since function may not exist)
            if 'basic_statistics' in analyses:
                stats_result = self._compute_basic_statistics(tracks, pixel_size, frame_interval)
                report['analysis_results']['basic_statistics'] = stats_result
                report['summary']['track_statistics'] = stats_result.get('summary', {})
            
            # MSD analysis
            if 'msd_analysis' in analyses and ANALYSIS_AVAILABLE:
                try:
                    msd_result = calculate_msd(tracks, pixel_size=pixel_size, frame_interval=frame_interval)
                    report['analysis_results']['msd_analysis'] = msd_result
                    if VISUALIZATION_AVAILABLE:
                        report['figures']['msd_curves'] = plot_msd_curves(msd_result)
                    report['summary']['msd_analysis'] = msd_result.get('summary', {})
                except Exception as e:
                    report['analysis_results']['msd_analysis'] = {'error': str(e)}
            
            # Diffusion analysis
            if 'diffusion_analysis' in analyses and ANALYSIS_AVAILABLE:
                try:
                    diff_result = analyze_diffusion(tracks, pixel_size=pixel_size, frame_interval=frame_interval)
                    report['analysis_results']['diffusion_analysis'] = diff_result
                    if VISUALIZATION_AVAILABLE:
                        report['figures']['diffusion_coefficients'] = plot_diffusion_coefficients(diff_result)
                    report['summary']['diffusion_analysis'] = diff_result.get('summary', {})
                except Exception as e:
                    report['analysis_results']['diffusion_analysis'] = {'error': str(e)}
            
            # Motion analysis
            if 'motion_analysis' in analyses and ANALYSIS_AVAILABLE:
                try:
                    motion_result = analyze_motion(tracks, pixel_size=pixel_size, frame_interval=frame_interval)
                    report['analysis_results']['motion_analysis'] = motion_result
                    if VISUALIZATION_AVAILABLE:
                        report['figures']['motion_analysis'] = plot_motion_analysis(motion_result)
                    report['summary']['motion_analysis'] = motion_result.get('summary', {})
                except Exception as e:
                    report['analysis_results']['motion_analysis'] = {'error': str(e)}
            
            # Motion classification (biophysical models)
            if 'motion_classification' in analyses and BIOPHYSICAL_AVAILABLE and analyze_motion_models:
                try:
                    motion_result = analyze_motion_models(tracks, min_track_length=10)
                    report['analysis_results']['motion_classification'] = motion_result
                    if VISUALIZATION_AVAILABLE:
                        report['figures']['motion_classification'] = plot_motion_analysis(motion_result)
                    report['summary']['motion_classification'] = motion_result.get('summary', {})
                except Exception as e:
                    report['analysis_results']['motion_classification'] = {'error': str(e)}
            
            # Intensity analysis
            if 'intensity_analysis' in analyses and INTENSITY_ANALYSIS_AVAILABLE:
                try:
                    # Check for intensity columns
                    intensity_channels = extract_intensity_channels(tracks)
                    if intensity_channels:
                        # Use first available intensity column
                        first_channel = list(intensity_channels.keys())[0]
                        first_col = intensity_channels[first_channel][0] if intensity_channels[first_channel] else None
                        
                        if first_col:
                            intensity_result = analyze_intensity_profiles(tracks, intensity_column=first_col)
                            report['analysis_results']['intensity_analysis'] = intensity_result
                            report['summary']['intensity_analysis'] = intensity_result.get('intensity_statistics', {})
                            
                            # Also run correlation analysis
                            correlation_result = correlate_intensity_movement(tracks, intensity_column=first_col)
                            report['analysis_results']['intensity_correlation'] = correlation_result
                    else:
                        report['analysis_results']['intensity_analysis'] = {
                            'error': 'No intensity columns found in data',
                            'success': False
                        }
                except Exception as e:
                    report['analysis_results']['intensity_analysis'] = {
                        'error': str(e),
                        'success': False
                    }
                
        except Exception as e:
            import traceback
            report['success'] = False
            report['error'] = str(e)
            report['traceback'] = traceback.format_exc()
            
        return report
    
    def _compute_basic_statistics(self, tracks_df, pixel_size, frame_interval):
        """
        Compute basic track statistics locally.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
            
        Returns
        -------
        dict
            Basic statistics dictionary
        """
        stats = {
            'success': True,
            'summary': {},
            'track_lengths': [],
            'track_durations': []
        }
        
        try:
            for track_id in tracks_df['track_id'].unique():
                track = tracks_df[tracks_df['track_id'] == track_id]
                stats['track_lengths'].append(len(track))
                duration = (track['frame'].max() - track['frame'].min()) * frame_interval
                stats['track_durations'].append(duration)
            
            stats['summary'] = {
                'num_tracks': len(stats['track_lengths']),
                'mean_track_length': np.mean(stats['track_lengths']) if stats['track_lengths'] else 0,
                'median_track_length': np.median(stats['track_lengths']) if stats['track_lengths'] else 0,
                'mean_duration_s': np.mean(stats['track_durations']) if stats['track_durations'] else 0,
                'total_localizations': len(tracks_df)
            }
        except Exception as e:
            stats['success'] = False
            stats['error'] = str(e)
        
        return stats