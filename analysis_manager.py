"""
Analysis Manager for SPT Analysis Application.
Coordinates and manages various analysis workflows.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from analysis_optimized import calculate_msd, analyze_diffusion, analyze_motion, analyze_boundary_crossing
    ANALYSIS_AVAILABLE = True
except ImportError:
    calculate_msd = None
    analyze_diffusion = None
    analyze_motion = None
    analyze_boundary_crossing = None
    ANALYSIS_AVAILABLE = False

try:
    from anomaly_detection import AnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    AnomalyDetector = None
    ANOMALY_DETECTION_AVAILABLE = False

try:
    from changepoint_detection import ChangePointDetector
    CHANGEPOINT_DETECTION_AVAILABLE = True
except ImportError:
    ChangePointDetector = None
    CHANGEPOINT_DETECTION_AVAILABLE = False

try:
    from rheology import MicrorheologyAnalyzer
    MICRORHEOLOGY_AVAILABLE = True
except ImportError:
    MicrorheologyAnalyzer = None
    MICRORHEOLOGY_AVAILABLE = False

try:
    from biophysical_models import PolymerPhysicsModel
    _POLY_OK = True
except Exception:
    PolymerPhysicsModel = None
    _POLY_OK = False

try:
    from correlative_analysis import CorrelativeAnalyzer
    CORRELATIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    CorrelativeAnalyzer = None
    CORRELATIVE_ANALYSIS_AVAILABLE = False

try:
    from multi_channel_analysis import analyze_channel_colocalization, analyze_compartment_occupancy, compare_channel_dynamics
    MULTI_CHANNEL_ANALYSIS_AVAILABLE = True
except ImportError:
    analyze_channel_colocalization = None
    analyze_compartment_occupancy = None
    compare_channel_dynamics = None
    MULTI_CHANNEL_ANALYSIS_AVAILABLE = False

from state_manager import get_state_manager

class AnalysisManager:
    """
    High-level analysis service that manages the execution of analytical workflows.
    This class acts as the interface between the UI and the complex analysis modules.
    """
    
    def __init__(self):
        self.state = get_state_manager()
        self.analysis_cache = {}
        self.debug_mode = False
        self.analysis_results = {}
        self.analysis_history = []
        
        # Available analysis types
        self.available_analyses = {
            'diffusion': {
                'name': 'Diffusion Analysis',
                'description': 'MSD calculation and diffusion coefficient estimation',
                'function': self.run_diffusion_analysis,
                'requirements': ['position_data']
            },
            'motion': {
                'name': 'Motion Classification',
                'description': 'Classify motion types (Brownian, directed, confined)',
                'function': self.run_motion_analysis,
                'requirements': ['position_data']
            },
            'anomaly': {
                'name': 'Anomaly Detection',
                'description': 'Identify outlier trajectories using ML',
                'function': self.run_anomaly_analysis,
                'requirements': ['position_data']
            },
            'changepoint': {
                'name': 'Changepoint Detection',
                'description': 'Detect changes in motion regimes',
                'function': self.run_changepoint_analysis,
                'requirements': ['position_data']
            },
            'microrheology': {
                'name': 'Microrheology',
                'description': 'Extract viscoelastic properties',
                'function': self.run_microrheology_analysis,
                'requirements': ['position_data']
            },
            'biophysical': {
                'name': 'Biophysical Models',
                'description': 'Fit biophysical motion models',
                'function': self.run_biophysical_analysis,
                'requirements': ['position_data']
            },
            'correlative': {
                'name': 'Correlative Analysis',
                'description': 'Analyze intensity-motion coupling and correlations',
                'function': self.run_correlative_analysis,
                'requirements': ['position_data', 'intensity_data']
            },
            'multi_channel': {
                'name': 'Multi-Channel Analysis',
                'description': 'Analyze interactions between multiple channels',
                'function': self.run_multi_channel_analysis,
                'requirements': ['position_data', 'secondary_channel_data']
            },
            'channel_correlation': {
                'name': 'Channel Correlation',
                'description': 'Calculate correlations between particle channels',
                'function': self.run_channel_correlation_analysis,
                'requirements': ['position_data', 'secondary_channel_data']
            }
        }
    
    def log(self, message: str, level: str = 'info'):
        """Log analysis messages with timestamping."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if self.debug_mode:
            print(f"[{timestamp}] Analysis: {message}")
        
        # Store in session state for UI display
        if 'analysis_log' not in st.session_state:
            st.session_state.analysis_log = []
        st.session_state.analysis_log.append(f"[{timestamp}] {message}")
    
    def get_analysis_requirements(self, analysis_type: str) -> List[str]:
        """Get the data requirements for a specific analysis."""
        return self.available_analyses.get(analysis_type, {}).get('requirements', [])
    
    def check_analysis_feasibility(self, analysis_type: str) -> Dict[str, Any]:
        """Check if an analysis can be performed with current data."""
        result = {
            'feasible': False,
            'missing_requirements': [],
            'data_issues': [],
            'warnings': []
        }
        
        if analysis_type not in self.available_analyses:
            result['data_issues'].append(f"Unknown analysis type: {analysis_type}")
            return result
        
        requirements = self.get_analysis_requirements(analysis_type)
        tracks_df = self.state.get_raw_tracks()
        
        # Check basic data availability
        if tracks_df.empty:
            result['missing_requirements'].append('No track data loaded')
            return result
        
        # Check specific requirements
        for req in requirements:
            if req == 'position_data':
                if not all(col in tracks_df.columns for col in ['x', 'y']):
                    result['missing_requirements'].append('Position data (x, y columns)')
            elif req == 'intensity_data':
                # Check for any intensity-related columns
                intensity_cols = [col for col in tracks_df.columns if 'intensity' in col.lower() or 'ch' in col.lower()]
                if not intensity_cols:
                    result['missing_requirements'].append('Intensity data')
            elif req == 'time_data':
                if 'frame' not in tracks_df.columns:
                    result['missing_requirements'].append('Time/frame data')
            elif req == 'secondary_channel_data':
                # Check for secondary channel data using state manager
                if not self.state.has_secondary_channel_data():
                    result['missing_requirements'].append('Secondary channel data (upload second channel)')
                else:
                    # Check that secondary data has required columns
                    secondary_data = self.state.get_secondary_channel_data()
                    if not all(col in secondary_data.columns for col in ['x', 'y']):
                        result['missing_requirements'].append('Secondary channel position data (x, y columns)')
        
        # Check track count
        n_tracks = tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 0
        if n_tracks < 5:
            result['warnings'].append(f'Very few tracks ({n_tracks}) for reliable analysis')
        
        # Determine feasibility
        result['feasible'] = len(result['missing_requirements']) == 0
        
        return result
    
    def execute_analysis_pipeline(self, analysis_types: List[str], 
                                 parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a complete analysis pipeline with multiple analysis types."""
        self.log(f"Starting analysis pipeline with {len(analysis_types)} analyses")
        
        if parameters is None:
            parameters = {}
        
        results = {
            'success': True,
            'analyses': {},
            'summary': {},
            'errors': []
        }
        
        for analysis_type in analysis_types:
            self.log(f"Running {analysis_type} analysis...")
            
            try:
                # Check feasibility
                feasibility = self.check_analysis_feasibility(analysis_type)
                if not feasibility['feasible']:
                    error_msg = f"Cannot run {analysis_type}: {', '.join(feasibility['missing_requirements'])}"
                    results['errors'].append(error_msg)
                    self.log(error_msg, 'error')
                    continue
                
                # Run analysis
                analysis_result = self.run_single_analysis(
                    analysis_type, 
                    parameters.get(analysis_type, {})
                )
                
                results['analyses'][analysis_type] = analysis_result
                
                if analysis_result.get('success', False):
                    self.log(f"✓ {analysis_type} completed successfully")
                else:
                    error_msg = f"✗ {analysis_type} failed: {analysis_result.get('error', 'Unknown error')}"
                    results['errors'].append(error_msg)
                    self.log(error_msg, 'error')
                    
            except Exception as e:
                error_msg = f"✗ {analysis_type} crashed: {str(e)}"
                results['errors'].append(error_msg)
                results['success'] = False
                self.log(error_msg, 'error')
        
        # Generate summary
        successful_analyses = [k for k, v in results['analyses'].items() if v.get('success', False)]
        results['summary'] = {
            'total_requested': len(analysis_types),
            'successful': len(successful_analyses),
            'failed': len(analysis_types) - len(successful_analyses),
            'success_rate': len(successful_analyses) / len(analysis_types) if analysis_types else 0
        }
        
        self.log(f"Pipeline completed: {len(successful_analyses)}/{len(analysis_types)} analyses successful")
        return results
    
    def run_single_analysis(self, analysis_type: str, parameters: Dict = None) -> Dict[str, Any]:
        """Execute a single analysis with error handling and caching."""
        if parameters is None:
            parameters = {}
        
        # Check cache
        cache_key = f"{analysis_type}_{hash(str(parameters))}"
        if cache_key in self.analysis_cache:
            self.log(f"Using cached result for {analysis_type}")
            return self.analysis_cache[cache_key]
        
        try:
            if analysis_type not in self.available_analyses:
                return {'success': False, 'error': f'Unknown analysis type: {analysis_type}'}
            
            analysis_func = self.available_analyses[analysis_type]['function']
            result = analysis_func(parameters)
            
            # Cache successful results
            if result.get('success', False):
                self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Analysis {analysis_type} failed: {str(e)}"
            self.log(error_msg, 'error')
            return {'success': False, 'error': error_msg}
    
    def run_diffusion_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run diffusion analysis with MSD calculation."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            
            if not ANALYSIS_AVAILABLE or analyze_diffusion is None:
                return {'success': False, 'error': 'Diffusion analysis module not available'}
            
            # Get analysis parameters
            max_lag = parameters.get('max_lag', 20) if parameters else 20
            pixel_size = self.state.get_pixel_size()
            frame_interval = self.state.get_frame_interval()
            
            # Run analysis
            result = analyze_diffusion(
                tracks_df, 
                max_lag=max_lag,
                pixel_size=pixel_size,
                frame_interval=frame_interval
            )
            
            # Enhance result with metadata
            result['analysis_type'] = 'diffusion'
            result['parameters'] = {
                'max_lag': max_lag,
                'pixel_size': pixel_size,
                'frame_interval': frame_interval
            }
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.log(f"Diffusion analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_motion_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run motion classification analysis."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            
            if not ANALYSIS_AVAILABLE or analyze_motion is None:
                return {'success': False, 'error': 'Motion analysis module not available'}
            
            result = analyze_motion(tracks_df)
            result['analysis_type'] = 'motion'
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.log(f"Motion analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_anomaly_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run anomaly detection analysis."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            
            if not ANOMALY_DETECTION_AVAILABLE or AnomalyDetector is None:
                return {'success': False, 'error': 'Anomaly detection module not available'}
            
            detector = AnomalyDetector()
            # Updated method call to use analyze_anomalies instead of detect_anomalies
            result = detector.analyze_anomalies(tracks_df)
            result['analysis_type'] = 'anomaly'
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.log(f"Anomaly analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_changepoint_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run changepoint detection analysis."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            
            if not CHANGEPOINT_DETECTION_AVAILABLE or ChangePointDetector is None:
                return {'success': False, 'error': 'Changepoint detection module not available'}
            
            detector = ChangePointDetector()
            result = detector.detect_motion_regime_changes(tracks_df)
            result['analysis_type'] = 'changepoint'
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.log(f"Changepoint analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_microrheology_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run microrheology analysis."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            
            if not MICRORHEOLOGY_AVAILABLE or MicrorheologyAnalyzer is None:
                return {'success': False, 'error': 'Microrheology module not available'}
            
            # Get parameters
            if parameters is None:
                parameters = {}
            
            # Get required parameters
            particle_radius_nm = parameters.get('particle_radius_nm', 50.0)  # Default 50nm
            temperature_K = parameters.get('temperature_K', 300.0)  # Default room temperature
            pixel_size_um = parameters.get('pixel_size_um', self.state.get_pixel_size())
            frame_interval_s = parameters.get('frame_interval_s', self.state.get_frame_interval())
            max_lag = parameters.get('max_lag', 20)
            
            # Convert particle radius to meters
            particle_radius_m = particle_radius_nm * 1e-9
            
            # Initialize analyzer with proper parameters
            analyzer = MicrorheologyAnalyzer(
                particle_radius_m=particle_radius_m,
                temperature_K=temperature_K
            )
            
            # Run analysis
            result = analyzer.analyze_microrheology(
                tracks_df,
                pixel_size_um=pixel_size_um,
                frame_interval_s=frame_interval_s,
                max_lag=max_lag
            )
            
            # Enhance result with metadata
            result['analysis_type'] = 'microrheology'
            result['timestamp'] = datetime.now().isoformat()
            result['parameters'] = {
                'particle_radius_nm': particle_radius_nm,
                'temperature_K': temperature_K,
                'pixel_size_um': pixel_size_um,
                'frame_interval_s': frame_interval_s,
                'max_lag': max_lag
            }
            
            return result
            
        except Exception as e:
            self.log(f"Microrheology analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_biophysical_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run biophysical model analysis."""
        try:
            parameters = parameters or {}
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            if not _POLY_OK or PolymerPhysicsModel is None:
                return {'success': False, 'error': 'Polymer physics module not available'}

            px = self.state.get_pixel_size()
            dt = self.state.get_frame_interval()

            # calculate_msd assumed imported elsewhere in existing code
            msd_df = calculate_msd(
                tracks_df,
                max_lag=parameters.get("max_lag", 20),
                pixel_size=px,
                frame_interval=dt,
                min_track_length=parameters.get("min_track_length", 5)
            )
            if msd_df.empty:
                return {'success': False, 'error': 'MSD empty; not enough tracks'}

            ppm = PolymerPhysicsModel(msd_df, pixel_size=px, frame_interval=dt, lag_units="seconds")

            out = {'success': True, 'analysis_type': 'biophysical'}
            if parameters.get("run_rouse", True):
                out['rouse'] = ppm.fit_rouse_model(
                    fit_alpha=parameters.get("rouse_fit_alpha", False),
                    temperature=float(parameters.get("temperature_K", 300.0)),
                    n_beads=int(parameters.get("n_beads", 100)),
                    friction_coefficient=float(parameters.get("friction_coefficient", 1e-8))
                )
            if parameters.get("run_zimm", False):
                out['zimm'] = ppm.fit_zimm_model(
                    temperature=float(parameters.get("temperature_K", 300.0)),
                    solvent_viscosity=float(parameters.get("solvent_viscosity_Pa_s", 0.001)),
                    hydrodynamic_radius=float(parameters.get("hydrodynamic_radius_m", 5e-9))
                )
            if parameters.get("run_reptation", False):
                out['reptation'] = ppm.fit_reptation_model(
                    temperature=float(parameters.get("temperature_K", 300.0)),
                    tube_diameter=float(parameters.get("tube_diameter_m", 100e-9)),
                    contour_length=float(parameters.get("contour_length_m", 1000e-9))
                )
            return out
        except Exception as e:
            self.log(f"Biophysical analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_correlative_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run correlative analysis including intensity-motion coupling."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No track data available'}
            
            if not CORRELATIVE_ANALYSIS_AVAILABLE or CorrelativeAnalyzer is None:
                return {'success': False, 'error': 'Correlative analysis module not available'}
            
            # Get parameters
            if parameters is None:
                parameters = {}
            
            intensity_columns = parameters.get('intensity_columns', None)
            lag_range = parameters.get('lag_range', 5)
            
            # Initialize analyzer
            analyzer = CorrelativeAnalyzer()
            
            # Run intensity-motion coupling analysis
            coupling_results = analyzer.analyze_intensity_motion_coupling(
                tracks_df, 
                intensity_columns=intensity_columns,
                lag_range=lag_range
            )
            
            # Run additional correlative analyses if track statistics are available
            additional_results = {}
            if hasattr(st, 'session_state') and 'track_statistics' in st.session_state:
                track_stats = st.session_state.track_statistics
                if track_stats is not None:
                    # Add track statistics correlation
                    try:
                        additional_results['track_statistics_correlation'] = analyzer.correlate_track_parameters(track_stats)
                    except AttributeError:
                        # Method might not exist in all versions
                        pass
            
            result = {
                'success': True,
                'analysis_type': 'correlative',
                'timestamp': datetime.now().isoformat(),
                'intensity_motion_coupling': coupling_results,
                'additional_analyses': additional_results,
                'parameters': parameters
            }
            
            return result
            
        except Exception as e:
            self.log(f"Correlative analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_multi_channel_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run multi-channel analysis for colocalization and interactions."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No primary track data available'}
            
            # Check for secondary channel data
            secondary_data = self.state.get_secondary_channel_data()
            
            if secondary_data is None or secondary_data.empty:
                return {'success': False, 'error': 'No secondary channel data available'}
            
            if not MULTI_CHANNEL_ANALYSIS_AVAILABLE or analyze_channel_colocalization is None:
                return {'success': False, 'error': 'Multi-channel analysis module not available'}
            
            # Get parameters
            if parameters is None:
                parameters = {}
            
            distance_threshold = parameters.get('distance_threshold', 2.0)
            frame_tolerance = parameters.get('frame_tolerance', 1)
            
            # Run colocalization analysis
            coloc_results = analyze_channel_colocalization(
                tracks_df,
                secondary_data,
                distance_threshold=distance_threshold,
                frame_tolerance=frame_tolerance
            )
            
            # Run compartment occupancy analysis if compartments are available
            compartment_results = {}
            if (hasattr(st, 'session_state') and 'segmentation_results' in st.session_state and
                analyze_compartment_occupancy is not None):
                segmentation_results = st.session_state.segmentation_results
                if segmentation_results and 'compartments' in segmentation_results:
                    compartments = segmentation_results['compartments']
                    pixel_size = self.state.get_pixel_size()
                    
                    # Analyze compartment occupancy for both channels
                    compartment_results['primary_channel'] = analyze_compartment_occupancy(
                        tracks_df, compartments, pixel_size=pixel_size
                    )
                    compartment_results['secondary_channel'] = analyze_compartment_occupancy(
                        secondary_data, compartments, pixel_size=pixel_size
                    )
            
            result = {
                'success': True,
                'analysis_type': 'multi_channel',
                'timestamp': datetime.now().isoformat(),
                'colocalization': coloc_results,
                'compartment_occupancy': compartment_results,
                'parameters': parameters
            }
            
            return result
            
        except Exception as e:
            self.log(f"Multi-channel analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_channel_correlation_analysis(self, parameters: Dict = None) -> Dict[str, Any]:
        """Run channel correlation analysis."""
        try:
            tracks_df = self.state.get_raw_tracks()
            if tracks_df.empty:
                return {'success': False, 'error': 'No primary track data available'}
            
            # Check for secondary channel data
            secondary_data = self.state.get_secondary_channel_data()
            
            if secondary_data is None or secondary_data.empty:
                return {'success': False, 'error': 'No secondary channel data available'}
            
            if not MULTI_CHANNEL_ANALYSIS_AVAILABLE or compare_channel_dynamics is None:
                return {'success': False, 'error': 'Channel correlation analysis modules not available'}
            
            # Get parameters
            if parameters is None:
                parameters = {}
            
            # Run basic track analysis on both channels
            primary_results = {}
            secondary_results = {}
            
            # Analyze primary channel
            if ANALYSIS_AVAILABLE and analyze_diffusion and analyze_motion:
                primary_results = {
                    'diffusion': analyze_diffusion(tracks_df, 
                                                 max_lag=parameters.get('max_lag', 20),
                                                 pixel_size=self.state.get_pixel_size(),
                                                 frame_interval=self.state.get_frame_interval()),
                    'motion': analyze_motion(tracks_df)
                }
                
                # Analyze secondary channel
                secondary_results = {
                    'diffusion': analyze_diffusion(secondary_data,
                                                 max_lag=parameters.get('max_lag', 20),
                                                 pixel_size=self.state.get_pixel_size(),
                                                 frame_interval=self.state.get_frame_interval()),
                    'motion': analyze_motion(secondary_data)
                }
            
            # Compare dynamics between channels
            dynamics_comparison = {}
            if primary_results and secondary_results:
                dynamics_comparison = compare_channel_dynamics(
                    primary_results,
                    secondary_results,
                    channel1_name=parameters.get('primary_channel_name', 'Primary'),
                    channel2_name=parameters.get('secondary_channel_name', 'Secondary')
                )
            
            # Run intensity correlations if available
            intensity_correlations = {}
            if CORRELATIVE_ANALYSIS_AVAILABLE and CorrelativeAnalyzer:
                try:
                    analyzer = CorrelativeAnalyzer()
                    # Try to find intensity columns in both datasets
                    primary_intensities = [col for col in tracks_df.columns if 'intensity' in col.lower() or 'ch' in col.lower()]
                    secondary_intensities = [col for col in secondary_data.columns if 'intensity' in col.lower() or 'ch' in col.lower()]
                    
                    if primary_intensities:
                        intensity_correlations['primary'] = analyzer.analyze_intensity_motion_coupling(
                            tracks_df, intensity_columns=primary_intensities
                        )
                    
                    if secondary_intensities:
                        intensity_correlations['secondary'] = analyzer.analyze_intensity_motion_coupling(
                            secondary_data, intensity_columns=secondary_intensities
                        )
                except Exception as e:
                    self.log(f"Intensity correlation analysis failed: {str(e)}", 'warning')
                    intensity_correlations['error'] = str(e)
            
            result = {
                'success': True,
                'analysis_type': 'channel_correlation',
                'timestamp': datetime.now().isoformat(),
                'primary_channel_analysis': primary_results,
                'secondary_channel_analysis': secondary_results,
                'dynamics_comparison': dynamics_comparison,
                'intensity_correlations': intensity_correlations,
                'parameters': parameters
            }
            
            return result
            
        except Exception as e:
            self.log(f"Channel correlation analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of all completed analyses."""
        summary = {
            'total_analyses': len(self.analysis_cache),
            'by_type': {},
            'recent_analyses': [],
            'success_rate': 0
        }
        
        successful_count = 0
        for cache_key, result in self.analysis_cache.items():
            analysis_type = result.get('analysis_type', 'unknown')
            
            if analysis_type not in summary['by_type']:
                summary['by_type'][analysis_type] = {'count': 0, 'success': 0}
            
            summary['by_type'][analysis_type]['count'] += 1
            
            if result.get('success', False):
                successful_count += 1
                summary['by_type'][analysis_type]['success'] += 1
            
            # Add to recent analyses
            if 'timestamp' in result:
                summary['recent_analyses'].append({
                    'type': analysis_type,
                    'timestamp': result['timestamp'],
                    'success': result.get('success', False)
                })
        
        # Sort recent analyses by timestamp
        summary['recent_analyses'].sort(key=lambda x: x['timestamp'], reverse=True)
        summary['recent_analyses'] = summary['recent_analyses'][:10]  # Keep last 10
        
        # Calculate success rate
        if len(self.analysis_cache) > 0:
            summary['success_rate'] = successful_count / len(self.analysis_cache)
        
        return summary
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        self.log("Analysis cache cleared")
    
    def export_results(self, format: str = 'json') -> str:
        """Export analysis results in specified format."""
        try:
            if format == 'json':
                import json
                return json.dumps(self.analysis_cache, indent=2, default=str)
            elif format == 'csv':
                # Convert results to flat structure for CSV
                flat_results = []
                for cache_key, result in self.analysis_cache.items():
                    flat_result = {
                        'analysis_type': result.get('analysis_type', 'unknown'),
                        'timestamp': result.get('timestamp', ''),
                        'success': result.get('success', False),
                        'error': result.get('error', ''),
                        'cache_key': cache_key
                    }
                    flat_results.append(flat_result)
                
                df = pd.DataFrame(flat_results)
                return df.to_csv(index=False)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            self.log(f"Export failed: {str(e)}")
            return f"Export failed: {str(e)}"
    
    def run_quality_control(self) -> Dict[str, Any]:
        """Run quality control checks on data and analysis results."""
        self.log("Running quality control checks...")
        
        qc_results = {
            'status': 'passed',
            'data_quality': {},
            'analysis_quality': {},
            'recommendations': []
        }
        
        try:
            # Data quality checks
            tracks_df = self.state.get_raw_tracks()
            
            if not tracks_df.empty:
                qc_results['data_quality'] = {
                    'n_tracks': tracks_df['track_id'].nunique(),
                    'n_points': len(tracks_df),
                    'avg_track_length': tracks_df.groupby('track_id').size().mean(),
                    'position_range_x': tracks_df['x'].max() - tracks_df['x'].min(),
                    'position_range_y': tracks_df['y'].max() - tracks_df['y'].min()
                }
                
                # Check for potential issues
                if qc_results['data_quality']['n_tracks'] < 10:
                    qc_results['recommendations'].append(
                        "Consider collecting more tracks for better statistical power"
                    )
                
                if qc_results['data_quality']['avg_track_length'] < 10:
                    qc_results['recommendations'].append(
                        "Short track lengths may limit diffusion analysis accuracy"
                    )
            
            # Analysis quality checks
            if self.analysis_cache:
                analysis_summary = self.get_analysis_summary()
                qc_results['analysis_quality'] = {
                    'success_rate': analysis_summary['success_rate'],
                    'completed_analyses': len(self.analysis_cache),
                    'analysis_types': list(analysis_summary['by_type'].keys())
                }
                
                if analysis_summary['success_rate'] < 0.8:
                    qc_results['status'] = 'warning'
                    qc_results['recommendations'].append(
                        "High analysis failure rate detected - check data quality and parameters"
                    )
            
        except Exception as e:
            qc_results['status'] = 'failed'
            qc_results['error'] = str(e)
            self.log(f"Quality control failed: {str(e)}")
        
        return qc_results
    
    def run_statistical_validation(self) -> Dict[str, Any]:
        """Run statistical validation on analysis results."""
        try:
            self.log("Running statistical validation...")
            
            validation_results = {
                'status': 'passed',
                'tests': {},
                'warnings': []
            }
            
            tracks_df = self.state.get_raw_tracks()
            
            if tracks_df.empty:
                validation_results['status'] = 'failed'
                validation_results['error'] = 'No data for validation'
                return validation_results
            
            # Basic statistical tests
            n_tracks = tracks_df['track_id'].nunique()
            
            # Test 1: Minimum sample size
            validation_results['tests']['sample_size'] = {
                'n_tracks': n_tracks,
                'minimum_recommended': 30,
                'passed': n_tracks >= 30
            }
            
            if n_tracks < 30:
                validation_results['warnings'].append(
                    f"Small sample size (n={n_tracks}). Results may have high uncertainty."
                )
            
            # Test 2: Track length distribution
            track_lengths = tracks_df.groupby('track_id').size()
            validation_results['tests']['track_lengths'] = {
                'mean_length': float(track_lengths.mean()),
                'min_length': int(track_lengths.min()),
                'max_length': int(track_lengths.max()),
                'std_length': float(track_lengths.std())
            }
            
            if track_lengths.min() < 5:
                validation_results['warnings'].append(
                    "Some tracks are very short (< 5 points). Consider filtering."
                )
            
            # Test 3: Spatial distribution
            x_range = tracks_df['x'].max() - tracks_df['x'].min()
            y_range = tracks_df['y'].max() - tracks_df['y'].min()
            
            validation_results['tests']['spatial_coverage'] = {
                'x_range': float(x_range),
                'y_range': float(y_range),
                'aspect_ratio': float(x_range / y_range) if y_range > 0 else float('inf')
            }
            
            return validation_results
            
        except Exception as e:
            self.log(f"Statistical validation failed: {str(e)}")
            raise

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of loaded data and analysis results."""
        validation_results = {
            'status': 'passed',
            'issues': [],
            'warnings': []
        }
        
        tracks_df = self.state.get_raw_tracks()
        
        # Check if data is loaded
        if tracks_df.empty:
            validation_results['issues'].append("No track data loaded")
            validation_results['status'] = 'failed'
            return validation_results
        
        # Check required columns
        required_cols = ['track_id', 'frame', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in tracks_df.columns]
        if missing_cols:
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
            validation_results['status'] = 'failed'
        
        # Check for reasonable parameter values
        pixel_size = self.state.get_pixel_size()
        frame_interval = self.state.get_frame_interval()
        
        if pixel_size <= 0:
            validation_results['issues'].append(f"Invalid pixel size: {pixel_size}")
            validation_results['status'] = 'failed'
        
        if frame_interval <= 0:
            validation_results['issues'].append(f"Invalid frame interval: {frame_interval}")
            validation_results['status'] = 'failed'
        
        # Check data ranges
        if not tracks_df.empty:
            n_tracks = tracks_df['track_id'].nunique()
            if n_tracks == 0:
                validation_results['issues'].append("No valid tracks found")
                validation_results['status'] = 'failed'
            elif n_tracks < 5:
                validation_results['warnings'].append(f"Very few tracks ({n_tracks}) for statistical analysis")
        
        return validation_results

    def calculate_track_statistics(self) -> None:
        """
        Calculate track statistics for the current tracks data and store in session state.
        """
        import streamlit as st
        from utils import calculate_track_statistics
        
        try:
            if (hasattr(st.session_state, 'tracks_data') and 
                st.session_state.tracks_data is not None and 
                not st.session_state.tracks_data.empty):
                
                # Calculate statistics using the utility function
                track_stats = calculate_track_statistics(st.session_state.tracks_data)
                st.session_state.track_statistics = track_stats
                
                self.log(f"Calculated statistics for {len(track_stats)} tracks", "info")
                
            else:
                st.session_state.track_statistics = None
                self.log("No track data available for statistics calculation", "warning")
                
        except Exception as e:
            self.log(f"Error calculating track statistics: {str(e)}", "error")
            st.session_state.track_statistics = None
    
    def run_comprehensive_analysis(self, tracks_df: pd.DataFrame, 
                                 analysis_types: List[str] = None,
                                 parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run comprehensive analysis suite on track data.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        analysis_types : List[str]
            Types of analysis to run
        parameters : Dict[str, Any]
            Analysis parameters
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results
        """
        if analysis_types is None:
            analysis_types = [
                'basic_statistics',
                'diffusion_analysis', 
                'motion_classification',
                'msd_analysis',
                'active_transport'
            ]
        
        if parameters is None:
            parameters = {}
        
        results = {
            'success': True,
            'analysis_types': analysis_types,
            'results': {},
            'summary': {}
        }
        
        try:
            # Basic track statistics
            if 'basic_statistics' in analysis_types:
                results['results']['basic_statistics'] = self._analyze_basic_statistics(tracks_df)
            
            # Diffusion analysis
            if 'diffusion_analysis' in analysis_types:
                results['results']['diffusion_analysis'] = self._analyze_diffusion(tracks_df, parameters)
            
            # Motion classification
            if 'motion_classification' in analysis_types:
                results['results']['motion_classification'] = self._classify_motion(tracks_df, parameters)
            
            # MSD analysis
            if 'msd_analysis' in analysis_types:
                results['results']['msd_analysis'] = self._analyze_msd(tracks_df, parameters)
            
            # Active transport analysis
            if 'active_transport' in analysis_types:
                results['results']['active_transport'] = self._analyze_active_transport(tracks_df, parameters)
            
            # Anomaly detection
            if 'anomaly_detection' in analysis_types:
                results['results']['anomaly_detection'] = self._detect_anomalies(tracks_df, parameters)
            
            # Generate summary
            results['summary'] = self._generate_analysis_summary(results['results'])
            
            # Store results
            self.analysis_results = results
            self.analysis_history.append({
                'timestamp': pd.Timestamp.now(),
                'analysis_types': analysis_types,
                'n_tracks': len(tracks_df['track_id'].unique()),
                'success': True
            })
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            
            self.analysis_history.append({
                'timestamp': pd.Timestamp.now(),
                'analysis_types': analysis_types,
                'success': False,
                'error': str(e)
            })
        
        return results
    
    def _analyze_basic_statistics(self, tracks_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic track statistics."""
        try:
            from analysis import calculate_track_statistics
            
            # Calculate track-level statistics
            track_stats = []
            for track_id in tracks_df['track_id'].unique():
                track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                stats = calculate_track_statistics(track_data)
                stats['track_id'] = track_id
                track_stats.append(stats)
            
            stats_df = pd.DataFrame(track_stats)
            
            # Calculate ensemble statistics
            ensemble_stats = {}
            for col in stats_df.select_dtypes(include=[np.number]).columns:
                if col != 'track_id':
                    values = stats_df[col].dropna()
                    if len(values) > 0:
                        ensemble_stats[col] = {
                            'mean': float(values.mean()),
                            'median': float(values.median()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
            
            return {
                'success': True,
                'track_statistics': stats_df,
                'ensemble_statistics': ensemble_stats,
                'n_tracks': len(stats_df)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_diffusion(self, tracks_df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze diffusion coefficients."""
        try:
            from analysis import calculate_diffusion_coefficient
            
            pixel_size = parameters.get('pixel_size', 0.1)
            frame_interval = parameters.get('frame_interval', 0.1)
            
            diffusion_results = []
            for track_id in tracks_df['track_id'].unique():
                track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                
                if len(track_data) >= 10:  # Minimum points for reliable diffusion calculation
                    diff_coeff = calculate_diffusion_coefficient(
                        track_data, pixel_size, frame_interval
                    )
                    diffusion_results.append({
                        'track_id': track_id,
                        'diffusion_coefficient': diff_coeff
                    })
            
            if diffusion_results:
                diffusion_df = pd.DataFrame(diffusion_results)
                
                # Calculate statistics
                coeffs = diffusion_df['diffusion_coefficient'].dropna()
                diffusion_stats = {
                    'mean': float(coeffs.mean()),
                    'median': float(coeffs.median()),
                    'std': float(coeffs.std()),
                    'min': float(coeffs.min()),
                    'max': float(coeffs.max())
                }
                
                return {
                    'success': True,
                    'diffusion_coefficients': diffusion_df,
                    'statistics': diffusion_stats,
                    'n_tracks': len(diffusion_df)
                }
            else:
                return {'success': False, 'error': 'No tracks suitable for diffusion analysis'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _classify_motion(self, tracks_df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify motion types."""
        try:
            from motion_analysis import classify_motion_type
            
            motion_results = []
            for track_id in tracks_df['track_id'].unique():
                track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                
                if len(track_data) >= 5:
                    classification = classify_motion_type(track_data)
                    motion_results.append({
                        'track_id': track_id,
                        'motion_type': classification.get('motion_type', 'unknown'),
                        'confidence': classification.get('confidence', 0.0),
                        'features': classification.get('features', {})
                    })
            
            if motion_results:
                motion_df = pd.DataFrame(motion_results)
                
                # Count motion types
                motion_counts = motion_df['motion_type'].value_counts().to_dict()
                
                return {
                    'success': True,
                    'motion_classifications': motion_df,
                    'motion_type_counts': motion_counts,
                    'n_tracks': len(motion_df)
                }
            else:
                return {'success': False, 'error': 'No tracks suitable for motion classification'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_msd(self, tracks_df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mean squared displacement."""
        try:
            from analysis import calculate_ensemble_msd
            
            pixel_size = parameters.get('pixel_size', 0.1)
            frame_interval = parameters.get('frame_interval', 0.1)
            max_lag = parameters.get('max_lag', 20)
            
            msd_result = calculate_ensemble_msd(
                tracks_df, 
                pixel_size=pixel_size,
                frame_interval=frame_interval,
                max_lag=max_lag
            )
            
            return {
                'success': True,
                'ensemble_msd': msd_result,
                'parameters': {
                    'pixel_size': pixel_size,
                    'frame_interval': frame_interval,
                    'max_lag': max_lag
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_active_transport(self, tracks_df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze active transport characteristics."""
        try:
            from analysis import analyze_active_transport
            
            pixel_size = parameters.get('pixel_size', 0.1)
            frame_interval = parameters.get('frame_interval', 0.1)
            
            transport_result = analyze_active_transport(
                tracks_df,
                pixel_size=pixel_size,
                frame_interval=frame_interval
            )
            
            return transport_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_anomalies(self, tracks_df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalous particle behavior."""
        try:
            from anomaly_detection import AnomalyDetector
            
            detector = AnomalyDetector()
            anomaly_results = detector.comprehensive_anomaly_detection(tracks_df)
            
            return {
                'success': True,
                'anomaly_results': anomaly_results,
                'summary': detector.get_anomaly_summary()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        summary = {
            'completed_analyses': [],
            'key_findings': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Track completed analyses
        for analysis_type, result in results.items():
            if result.get('success', False):
                summary['completed_analyses'].append(analysis_type)
        
        # Extract key findings
        if 'basic_statistics' in results and results['basic_statistics']['success']:
            stats = results['basic_statistics']['ensemble_statistics']
            n_tracks = results['basic_statistics']['n_tracks']
            
            summary['key_findings'].append(f"Analyzed {n_tracks} tracks")
            
            if 'track_length' in stats:
                mean_length = stats['track_length']['mean']
                summary['key_findings'].append(f"Average track length: {mean_length:.1f} frames")
        
        if 'diffusion_analysis' in results and results['diffusion_analysis']['success']:
            diff_stats = results['diffusion_analysis']['statistics']
            mean_diff = diff_stats['mean']
            summary['key_findings'].append(f"Mean diffusion coefficient: {mean_diff:.2e} μm²/s")
        
        if 'motion_classification' in results and results['motion_classification']['success']:
            motion_counts = results['motion_classification']['motion_type_counts']
            dominant_motion = max(motion_counts.items(), key=lambda x: x[1])
            summary['key_findings'].append(f"Dominant motion type: {dominant_motion[0]} ({dominant_motion[1]} tracks)")
        
        # Assess data quality
        if 'basic_statistics' in results and results['basic_statistics']['success']:
            stats = results['basic_statistics']['ensemble_statistics']
            
            if 'track_length' in stats:
                mean_length = stats['track_length']['mean']
                if mean_length < 10:
                    summary['data_quality']['track_length'] = 'Short tracks detected'
                    summary['recommendations'].append('Consider optimizing tracking parameters for longer tracks')
                else:
                    summary['data_quality']['track_length'] = 'Good track lengths'
        
        return summary
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of analysis runs."""
        return self.analysis_history.copy()
    
    def export_results(self, format_type: str = 'json') -> Dict[str, Any]:
        """Export analysis results in specified format."""
        if not self.analysis_results:
            return {'success': False, 'error': 'No analysis results to export'}
        
        try:
            if format_type == 'json':
                # Convert DataFrames to dictionaries for JSON serialization
                exportable_results = {}
                for key, value in self.analysis_results.items():
                    if isinstance(value, dict):
                        exportable_results[key] = self._make_json_serializable(value)
                    else:
                        exportable_results[key] = value
                
                return {
                    'success': True,
                    'format': 'json',
                    'data': exportable_results
                }
            else:
                return {'success': False, 'error': f'Unsupported export format: {format_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
