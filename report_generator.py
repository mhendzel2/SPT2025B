"""
Report generator for single particle tracking analysis.
"""

import pandas as pd
from analysis import (
    analyze_track_statistics, analyze_msd, analyze_diffusion, 
    analyze_directional_motion, analyze_displacements
)
from biophysical_models import analyze_motion_models
from visualization import (
    plot_track_statistics, plot_msd_curves, plot_diffusion_coefficients, 
    plot_directional_analysis, plot_displacement_analysis, plot_motion_analysis
)

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
                'error': error_msg
            }
            
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
            # Basic statistics
            if 'basic_statistics' in analyses:
                stats_result = analyze_track_statistics(tracks, pixel_size, frame_interval)
                report['analysis_results']['basic_statistics'] = stats_result
                report['figures']['track_statistics'] = plot_track_statistics(stats_result)
                report['summary']['track_statistics'] = stats_result.get('summary', {})
            
            # MSD analysis
            if 'msd_analysis' in analyses:
                msd_result = analyze_msd(tracks, pixel_size, frame_interval)
                report['analysis_results']['msd_analysis'] = msd_result
                report['figures']['msd_curves'] = plot_msd_curves(msd_result)
                report['summary']['msd_analysis'] = msd_result.get('summary', {})
            
            # Diffusion analysis
            if 'diffusion_analysis' in analyses:
                diff_result = analyze_diffusion(tracks, pixel_size, frame_interval)
                report['analysis_results']['diffusion_analysis'] = diff_result
                report['figures']['diffusion_coefficients'] = plot_diffusion_coefficients(diff_result)
                report['summary']['diffusion_analysis'] = diff_result.get('summary', {})
            
            # Directional analysis
            if 'directional_analysis' in analyses:
                dir_result = analyze_directional_motion(tracks, pixel_size, frame_interval)
                report['analysis_results']['directional_analysis'] = dir_result
                report['figures']['directional_analysis'] = plot_directional_analysis(dir_result)
                report['summary']['directional_analysis'] = dir_result.get('summary', {})
            
            # Displacement analysis
            if 'displacement_analysis' in analyses:
                disp_result = analyze_displacements(tracks, pixel_size, frame_interval)
                report['analysis_results']['displacement_analysis'] = disp_result
                report['figures']['displacement_analysis'] = plot_displacement_analysis(disp_result)
                report['summary']['displacement_analysis'] = disp_result.get('summary', {})
            
            # Motion classification
            if 'motion_classification' in analyses:
                motion_result = analyze_motion_models(tracks, min_track_length=10)
                report['analysis_results']['motion_classification'] = motion_result
                report['figures']['motion_classification'] = plot_motion_analysis(motion_result)
                report['summary']['motion_classification'] = motion_result.get('summary', {})
                
        except Exception as e:
            import traceback
            report['success'] = False
            report['error'] = str(e)
            report['traceback'] = traceback.format_exc()
            
        return report