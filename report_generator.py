"""
Report generator for single particle tracking analysis.
"""

import pandas as pd
import numpy as np
from analysis import (
    calculate_msd, analyze_diffusion, analyze_motion
)
from biophysical_models import analyze_motion_models
# Import available visualization functions
try:
    from visualization import (
        plot_tracks, plot_msd_curves, plot_diffusion_coefficients
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

def analyze_track_statistics(tracks_df, pixel_size=1.0, frame_interval=1.0):
    """Basic track statistics analysis."""
    try:
        # Calculate basic statistics
        track_lengths = tracks_df.groupby('track_id').size()
        
        # Calculate velocities if possible
        velocities = []
        for track_id, track_data in tracks_df.groupby('track_id'):
            track_data = track_data.sort_values('frame')
            if len(track_data) > 1:
                positions = track_data[['x', 'y']].values * pixel_size
                dt = frame_interval
                displacements = np.diff(positions, axis=0)
                speeds = np.linalg.norm(displacements, axis=1) / dt
                velocities.extend(speeds)
        
        return {
            'success': True,
            'total_tracks': len(track_lengths),
            'mean_track_length': float(track_lengths.mean()),
            'median_track_length': float(track_lengths.median()),
            'track_length_std': float(track_lengths.std()),
            'mean_velocity': float(np.mean(velocities)) if velocities else 0.0,
            'velocity_std': float(np.std(velocities)) if velocities else 0.0,
            'summary': {
                'tracks_analyzed': len(track_lengths),
                'total_localizations': len(tracks_df),
                'mean_track_length': float(track_lengths.mean())
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_msd(tracks_df, pixel_size=1.0, frame_interval=1.0):
    """MSD analysis wrapper."""
    try:
        msd_data = calculate_msd(tracks_df, pixel_size=pixel_size, frame_interval=frame_interval)
        if msd_data.empty:
            return {'success': False, 'error': 'No MSD data could be calculated'}
        
        # Calculate ensemble average
        ensemble_msd = msd_data.groupby('lag_time')['msd'].mean().reset_index()
        
        return {
            'success': True,
            'msd_data': msd_data,
            'ensemble_msd': ensemble_msd,
            'summary': {
                'tracks_analyzed': len(msd_data['track_id'].unique()),
                'max_lag_time': msd_data['lag_time'].max(),
                'mean_msd_at_1s': ensemble_msd[ensemble_msd['lag_time'] <= 1.0]['msd'].mean() if len(ensemble_msd) > 0 else 0
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_directional_motion(tracks_df, pixel_size=1.0, frame_interval=1.0):
    """Directional motion analysis."""
    try:
        return analyze_motion(tracks_df, pixel_size=pixel_size)
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_displacements(tracks_df, pixel_size=1.0, frame_interval=1.0):
    """Displacement analysis."""
    try:
        # Simple displacement analysis
        displacements = []
        for track_id, track_data in tracks_df.groupby('track_id'):
            track_data = track_data.sort_values('frame')
            if len(track_data) > 1:
                positions = track_data[['x', 'y']].values * pixel_size
                disp = np.diff(positions, axis=0)
                step_sizes = np.linalg.norm(disp, axis=1)
                displacements.extend(step_sizes)
        
        return {
            'success': True,
            'mean_displacement': float(np.mean(displacements)) if displacements else 0.0,
            'std_displacement': float(np.std(displacements)) if displacements else 0.0,
            'total_steps': len(displacements),
            'summary': {
                'mean_step_size': float(np.mean(displacements)) if displacements else 0.0,
                'total_displacements': len(displacements)
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Visualization functions
def plot_track_statistics(stats_result):
    """Generate track statistics plot."""
    if not VISUALIZATION_AVAILABLE:
        return None
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Total Tracks', 'Mean Length', 'Mean Velocity'],
            y=[stats_result.get('total_tracks', 0), 
               stats_result.get('mean_track_length', 0),
               stats_result.get('mean_velocity', 0)],
            name='Statistics'
        ))
        fig.update_layout(title="Track Statistics")
        return fig
    except Exception:
        return None

def plot_directional_analysis(dir_result):
    """Generate directional analysis plot."""
    if not VISUALIZATION_AVAILABLE:
        return None
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text="Directional Analysis Plot", x=0.5, y=0.5)
        return fig
    except Exception:
        return None

def plot_displacement_analysis(disp_result):
    """Generate displacement analysis plot."""
    if not VISUALIZATION_AVAILABLE:
        return None
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Mean Displacement', 'Std Displacement'],
            y=[disp_result.get('mean_displacement', 0),
               disp_result.get('std_displacement', 0)],
            name='Displacements'
        ))
        fig.update_layout(title="Displacement Analysis")
        return fig
    except Exception:
        return None

def plot_motion_analysis(motion_result):
    """Generate motion analysis plot."""
    if not VISUALIZATION_AVAILABLE:
        return None
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text="Motion Analysis Plot", x=0.5, y=0.5)
        return fig
    except Exception:
        return None

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
        
        # Perform requested analyses with individual error handling for graceful degradation
        analysis_errors = []
        
        try:
            # Basic statistics
            if 'basic_statistics' in analyses:
                try:
                    stats_result = analyze_track_statistics(tracks, pixel_size, frame_interval)
                    report['analysis_results']['basic_statistics'] = stats_result
                    report['figures']['track_statistics'] = plot_track_statistics(stats_result)
                    report['summary']['track_statistics'] = stats_result.get('summary', {})
                except Exception as e:
                    analysis_errors.append(f"Basic statistics analysis failed: {str(e)}")
                    report['analysis_results']['basic_statistics'] = {'success': False, 'error': str(e)}
            
            # MSD analysis
            if 'msd_analysis' in analyses:
                try:
                    msd_result = analyze_msd(tracks, pixel_size, frame_interval)
                    report['analysis_results']['msd_analysis'] = msd_result
                    report['figures']['msd_curves'] = plot_msd_curves(msd_result)
                    report['summary']['msd_analysis'] = msd_result.get('summary', {})
                except Exception as e:
                    analysis_errors.append(f"MSD analysis failed: {str(e)}")
                    report['analysis_results']['msd_analysis'] = {'success': False, 'error': str(e)}
            
            # Diffusion analysis
            if 'diffusion_analysis' in analyses:
                try:
                    diff_result = analyze_diffusion(tracks, pixel_size, frame_interval)
                    report['analysis_results']['diffusion_analysis'] = diff_result
                    report['figures']['diffusion_coefficients'] = plot_diffusion_coefficients(diff_result)
                    report['summary']['diffusion_analysis'] = diff_result.get('summary', {})
                except Exception as e:
                    analysis_errors.append(f"Diffusion analysis failed: {str(e)}")
                    report['analysis_results']['diffusion_analysis'] = {'success': False, 'error': str(e)}
                report['summary']['diffusion_analysis'] = diff_result.get('summary', {})
            
            # Directional analysis
            if 'directional_analysis' in analyses:
                try:
                    dir_result = analyze_directional_motion(tracks, pixel_size, frame_interval)
                    report['analysis_results']['directional_analysis'] = dir_result
                    report['figures']['directional_analysis'] = plot_directional_analysis(dir_result)
                    report['summary']['directional_analysis'] = dir_result.get('summary', {})
                except Exception as e:
                    analysis_errors.append(f"Directional analysis failed: {str(e)}")
                    report['analysis_results']['directional_analysis'] = {'success': False, 'error': str(e)}
            
            # Displacement analysis
            if 'displacement_analysis' in analyses:
                try:
                    disp_result = analyze_displacements(tracks, pixel_size, frame_interval)
                    report['analysis_results']['displacement_analysis'] = disp_result
                    report['figures']['displacement_analysis'] = plot_displacement_analysis(disp_result)
                    report['summary']['displacement_analysis'] = disp_result.get('summary', {})
                except Exception as e:
                    analysis_errors.append(f"Displacement analysis failed: {str(e)}")
                    report['analysis_results']['displacement_analysis'] = {'success': False, 'error': str(e)}
            
            # Motion classification
            if 'motion_classification' in analyses:
                try:
                    motion_result = analyze_motion_models(tracks, min_track_length=10)
                    report['analysis_results']['motion_classification'] = motion_result
                    report['figures']['motion_classification'] = plot_motion_analysis(motion_result)
                    report['summary']['motion_classification'] = motion_result.get('summary', {})
                except Exception as e:
                    analysis_errors.append(f"Motion classification failed: {str(e)}")
                    report['analysis_results']['motion_classification'] = {'success': False, 'error': str(e)}
            
            # Add partial success handling
            if analysis_errors:
                report['partial_success'] = True
                report['analysis_errors'] = analysis_errors
                report['successful_analyses'] = [key for key in report['analysis_results'] 
                                               if report['analysis_results'][key].get('success', True)]
                
        except Exception as e:
            import traceback
            import logging
            
            # Log the full error for debugging
            logging.error(f"Report generation failed: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Provide specific error messages based on error type
            if "MSD" in str(e) or "diffusion" in str(e).lower():
                error_message = f"Diffusion analysis failed: {str(e)}. Check track data quality and ensure sufficient track lengths."
            elif "motion" in str(e).lower() or "classification" in str(e).lower():
                error_message = f"Motion classification failed: {str(e)}. This may be due to insufficient track data or short track lengths."
            elif "statistics" in str(e).lower():
                error_message = f"Basic statistics calculation failed: {str(e)}. Verify track data format and completeness."
            elif "visualization" in str(e).lower() or "plot" in str(e).lower():
                error_message = f"Visualization generation failed: {str(e)}. Report data is available but plots could not be generated."
            else:
                error_message = f"Report generation encountered an unexpected error: {str(e)}"
            
            report['success'] = False
            report['error'] = error_message
            report['technical_details'] = {
                'exception_type': type(e).__name__,
                'exception_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            # Add guidance for common issues
            report['troubleshooting'] = {
                'suggestions': [
                    "Verify that track data contains required columns: 'track_id', 'frame', 'x', 'y'",
                    "Ensure tracks have sufficient length (>= 5 points) for analysis",
                    "Check that pixel_size and frame_interval are positive numbers",
                    "Verify track coordinates are within reasonable bounds"
                ],
                'common_fixes': [
                    "Filter tracks by minimum length before analysis",
                    "Check for NaN or infinite values in track coordinates",
                    "Ensure proper unit conversions (pixels to micrometers, frames to seconds)"
                ]
            }
            
        return report