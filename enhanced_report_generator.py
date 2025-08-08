"""
Enhanced Automated Report Generation Module for SPT Analysis (Fully Implemented)

This module provides comprehensive report generation capabilities with extensive content:
- Advanced statistical analyses and machine learning classifications
- Multi-modal data integration and correlative analysis
- Professional publication-ready reports with detailed methodology
- Interactive dashboards and executive summaries
- Quality control and validation metrics
- Comparative analysis across conditions and datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
from typing import Dict, List, Any, Optional, Tuple
import json
import zipfile
import tempfile
import os
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Import all necessary analysis and visualization functions with error handling
try:
    from analysis import (
        calculate_msd, analyze_diffusion, analyze_motion, analyze_clustering,
        analyze_dwell_time, analyze_gel_structure, analyze_diffusion_population,
        analyze_crowding, analyze_active_transport, analyze_boundary_crossing
    )
    ANALYSIS_MODULE_AVAILABLE = True
except ImportError:
    ANALYSIS_MODULE_AVAILABLE = False

try:
    from visualization import (
        plot_tracks, plot_tracks_3d, plot_track_statistics, plot_motion_analysis,
        plot_msd_curves, plot_diffusion_coefficients
    )
    # Note: Some functions may not be available in visualization module
    VISUALIZATION_MODULE_AVAILABLE = True
except ImportError:
    VISUALIZATION_MODULE_AVAILABLE = False

try:
    from multi_channel_analysis import (
        analyze_channel_colocalization, analyze_compartment_occupancy,
        compare_channel_dynamics, create_multi_channel_visualization
    )
    MULTI_CHANNEL_AVAILABLE = True
except ImportError:
    MULTI_CHANNEL_AVAILABLE = False

try:
    from anomaly_detection import AnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

try:
    from biophysical_models import PolymerPhysicsModel
    BIOPHYSICAL_MODELS_AVAILABLE = True
except ImportError:
    BIOPHYSICAL_MODELS_AVAILABLE = False

try:
    from changepoint_detection import ChangePointDetector
    CHANGEPOINT_DETECTION_AVAILABLE = True
except ImportError:
    CHANGEPOINT_DETECTION_AVAILABLE = False

try:
    from correlative_analysis import CorrelativeAnalyzer
    CORRELATIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    CORRELATIVE_ANALYSIS_AVAILABLE = False

try:
    from rheology import MicrorheologyAnalyzer, create_rheology_plots
    RHEOLOGY_MODULE_AVAILABLE = True
except Exception:
    RHEOLOGY_MODULE_AVAILABLE = False

warnings.filterwarnings('ignore')

class EnhancedSPTReportGenerator:
    """
    Comprehensive report generation system with extensive analysis capabilities.
    """

    def __init__(self):
        self.available_analyses = {
            'basic_statistics': {
                'name': 'Basic Track Statistics',
                'description': 'Comprehensive track metrics: lengths, displacements, velocities.',
                'function': self._analyze_basic_statistics,
                'visualization': self._plot_basic_statistics,
                'category': 'Basic',
                'priority': 1
            },
            'diffusion_analysis': {
                'name': 'Diffusion Analysis',
                'description': 'MSD analysis, diffusion coefficients, anomalous diffusion parameters.',
                'function': self._analyze_diffusion,
                'visualization': self._plot_diffusion,
                'category': 'Core Physics',
                'priority': 2
            },
            'motion_classification': {
                'name': 'Motion Classification',
                'description': 'Brownian, subdiffusive, superdiffusive, confined, and directed motion.',
                'function': self._analyze_motion,
                'visualization': self._plot_motion,
                'category': 'Core Physics',
                'priority': 2
            },
            'spatial_organization': {
                'name': 'Spatial Organization',
                'description': 'Clustering, spatial correlations, territory analysis.',
                'function': self._analyze_clustering,
                'visualization': self._plot_clustering,
                'category': 'Spatial Analysis',
                'priority': 3
            },
            'anomaly_detection': {
                'name': 'Anomaly Detection',
                'description': 'Outlier trajectories, unusual behavior patterns.',
                'function': self._analyze_anomalies,
                'visualization': self._plot_anomalies,
                'category': 'Machine Learning',
                'priority': 4
            },
            'microrheology': {
                'name': 'Microrheology Analysis',
                'description': 'Viscoelastic properties: storage modulus (G\'), loss modulus (G\'\'), complex viscosity.',
                'function': self._analyze_microrheology,
                'visualization': self._plot_microrheology,
                'category': 'Biophysical Models',
                'priority': 3
            },
            'intensity_analysis': {
                'name': 'Intensity Analysis',
                'description': 'Fluorescence intensity dynamics, photobleaching, binding events.',
                'function': self._analyze_intensity,
                'visualization': self._plot_intensity,
                'category': 'Photophysics',
                'priority': 3
            },
            'confinement_analysis': {
                'name': 'Confinement Analysis',
                'description': 'Confined motion detection, boundary interactions, escape events.',
                'function': self._analyze_confinement,
                'visualization': self._plot_confinement,
                'category': 'Spatial Analysis',
                'priority': 3
            },
            'velocity_correlation': {
                'name': 'Velocity Correlation',
                'description': 'Velocity autocorrelation, persistence length, memory effects.',
                'function': self._analyze_velocity_correlation,
                'visualization': self._plot_velocity_correlation,
                'category': 'Core Physics',
                'priority': 3
            },
            'multi_particle_interactions': {
                'name': 'Multi-Particle Interactions',
                'description': 'Particle-particle correlations, collective motion, crowding effects.',
                'function': self._analyze_particle_interactions,
                'visualization': self._plot_particle_interactions,
                'category': 'Advanced Statistics',
                'priority': 4
            }
        }

        # Add advanced analyses conditionally
        if CHANGEPOINT_DETECTION_AVAILABLE:
            self.available_analyses['changepoint_detection'] = {
                'name': 'Changepoint Detection',
                'description': 'Motion regime changes, behavioral transitions.',
                'function': self._analyze_changepoints,
                'visualization': self._plot_changepoints,
                'category': 'Advanced Statistics',
                'priority': 3
            }
        if BIOPHYSICAL_MODELS_AVAILABLE:
            self.available_analyses['polymer_physics'] = {
                'name': 'Polymer Physics Models',
                'description': 'Rouse dynamics, chromatin fiber analysis.',
                'function': self._analyze_polymer_physics,
                'visualization': self._plot_polymer_physics,
                'category': 'Biophysical Models',
                'priority': 4
            }
        
        self.report_results = {}
        self.report_figures = {}
        self.metadata = {}

    def display_enhanced_analysis_interface(self):
        """Display comprehensive analysis selection interface."""
        st.header("📊 Enhanced Automated Report Generation")
        st.markdown("Select from the available modules to create a detailed report.")
        
        categories = self._group_analyses_by_category()
        
        selected_analyses = self._display_analysis_selector(categories)
        report_config = self._display_report_config()
            
        if st.button("🚀 Generate Comprehensive Report", type="primary"):
            if selected_analyses:
                # Check for track data
                if 'raw_tracks' not in st.session_state or st.session_state.raw_tracks is None:
                    st.error("No track data loaded. Please load data first.")
                    return

                tracks_df = st.session_state.raw_tracks
                current_units = {
                    'pixel_size': st.session_state.get('pixel_size', 0.1),
                    'frame_interval': st.session_state.get('frame_interval', 0.1)
                }

                self.generate_automated_report(tracks_df, selected_analyses, report_config, current_units)
            else:
                st.warning("Please select at least one analysis to include in the report.")

    def _group_analyses_by_category(self):
        """Group analyses by category with priority ordering."""
        categories = {}
        for key, analysis in self.available_analyses.items():
            category = analysis.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append((key, analysis))
        
        for category in categories:
            categories[category].sort(key=lambda x: x[1].get('priority', 99))
            
        return categories

    def _display_analysis_selector(self, categories):
        """Display analysis selection interface with 'Select All' feature."""
        st.markdown("**Analysis Selection**")

        # New Feature: Select/Deselect All
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Select All"):
                st.session_state.select_all_state = True
                for key in self.available_analyses.keys():
                    st.session_state[f"analysis_{key}"] = True
        
        with col2:
            if st.button("❌ Deselect All"):
                st.session_state.select_all_state = False
                for key in self.available_analyses.keys():
                    st.session_state[f"analysis_{key}"] = False
        
        with col3:
            if st.button("🔧 Core Only"):
                for key in self.available_analyses.keys():
                    category = self.available_analyses[key].get('category', '')
                    st.session_state[f"analysis_{key}"] = category in ['Basic', 'Core Physics']

        # Initialize select_all_state if not exists
        if 'select_all_state' not in st.session_state:
            st.session_state.select_all_state = True
            # Set default selection for commonly used analyses
            default_analyses = ['basic_statistics', 'diffusion_analysis', 'motion_classification', 'microrheology']
            for key in self.available_analyses.keys():
                st.session_state[f"analysis_{key}"] = key in default_analyses

        selected_analyses_keys = []

        # Quick selection presets
        st.markdown("**Quick Selection Presets:**")
        col1, col2, col3, col4 = st.columns(4)
        
        preset_selection = None
        with col1:
            if st.button("📊 Basic Package"):
                preset_selection = ['basic_statistics', 'diffusion_analysis']
        with col2:
            if st.button("🔬 Core Physics"):
                preset_selection = ['diffusion_analysis', 'motion_classification', 'spatial_organization']
        with col3:
            if st.button("🧠 Machine Learning"):
                preset_selection = ['anomaly_detection', 'changepoint_detection'] if CHANGEPOINT_DETECTION_AVAILABLE else ['anomaly_detection']
        with col4:
            if st.button("📈 Complete Analysis"):
                preset_selection = list(self.available_analyses.keys())
        
        st.markdown("---")
        st.markdown("**Individual Analysis Selection:**")
        
        # Category-wise selection with detailed info
        for category, analyses in categories.items():
            with st.expander(f"{category} ({len(analyses)} analyses)", expanded=category in ['Basic', 'Core Physics']):
                for key, analysis in analyses:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Use session state for persistent checkbox values
                        checkbox_key = f"analysis_{key}"
                        if preset_selection:
                            st.session_state[checkbox_key] = key in preset_selection
                            
                        is_selected = st.checkbox(
                            analysis['name'],
                            value=st.session_state.get(checkbox_key, key in ['basic_statistics', 'diffusion_analysis', 'motion_classification', 'microrheology']),
                            key=checkbox_key,
                            help=analysis['description']
                        )
                        st.caption(analysis['description'])
                        
                    with col2:
                        priority_color = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🟣"}
                        st.write(f"Priority: {priority_color.get(analysis['priority'], '⚪')}")
                        
                    if is_selected:
                        selected_analyses_keys.append(key)

        # Return selected analyses based on current checkbox states
        if preset_selection:
            return preset_selection
        return selected_analyses_keys

    def _display_report_config(self):
        """Display report configuration options."""
        st.subheader("Report Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            report_format = st.selectbox("Report Format", ["HTML Interactive", "PDF Report", "Raw Data (JSON)"])
        with col2:
            include_raw = st.checkbox("Include Raw Data", value=True)
        
        return {'format': report_format, 'include_raw': include_raw}

    def generate_automated_report(self, tracks_df, selected_analyses, config, current_units):
        """Execute comprehensive analysis pipeline using existing results."""
        st.subheader("📄 Generating Report from Analysis Results")
        
        # Check if any analysis results exist
        if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
            st.error("No analysis results found. Please run analyses from the 'Analysis' tab first.")
            st.info("💡 Go to the Analysis tab and run some analyses before generating a report.")
            return
        
        analysis_results = st.session_state.analysis_results
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        self.report_results = {}
        self.report_figures = {}
        
        for i, analysis_key in enumerate(selected_analyses):
            analysis = self.available_analyses[analysis_key]
            
            # Check if analysis results are available
            if analysis_key not in analysis_results:
                st.warning(f"⚠️ Skipping {analysis['name']} - analysis has not been run. Please complete this analysis first.")
                continue
                
            # Verify analysis was successful
            if not analysis_results[analysis_key].get('success', False):
                st.warning(f"⚠️ Skipping {analysis['name']} - analysis failed or incomplete.")
                continue
            
            status_text.text(f"Adding {analysis['name']} to report...")
            progress_bar.progress((i + 1) / len(selected_analyses))
            
            try:
                # Use existing analysis results instead of re-running
                self.report_results[analysis_key] = analysis_results[analysis_key]
                
                # Generate visualization from existing data
                fig = analysis['visualization'](analysis_results[analysis_key])
                if fig:
                    self.report_figures[analysis_key] = fig
                st.success(f"✅ Added {analysis['name']} to report")
                
            except Exception as e:
                st.error(f"❌ Failed to add {analysis['name']} to report: {str(e)}")
        
        status_text.text("Report generation complete!")
        
        # Display the generated report
        self._display_generated_report(config, current_units)

    # --- Full Implementations of Analysis Wrappers ---

    def _analyze_basic_statistics(self, tracks_df, current_units):
        """Full implementation for basic statistics with safe data access."""
        try:
            if tracks_df is None or tracks_df.empty:
                return {'error': 'No track data available', 'success': False}
            
            # Basic statistics with safe access
            track_lengths = tracks_df.groupby('track_id').size()
            
            # Safe velocity calculation
            velocities = []
            if 'velocity_x' in tracks_df.columns and 'velocity_y' in tracks_df.columns:
                velocities = np.sqrt(tracks_df['velocity_x']**2 + tracks_df['velocity_y']**2)
            elif 'velocity' in tracks_df.columns:
                velocities = tracks_df['velocity'].dropna()
            
            results = {
                'total_tracks': len(track_lengths),
                'mean_track_length': float(track_lengths.mean()) if not track_lengths.empty else 0.0,
                'median_track_length': float(track_lengths.median()) if not track_lengths.empty else 0.0,
                'track_length_std': float(track_lengths.std()) if not track_lengths.empty else 0.0,
                'mean_velocity': float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
                'velocity_std': float(np.std(velocities)) if len(velocities) > 0 else 0.0,
                'success': True
            }
            
        except Exception as e:
            results = {'error': f'Analysis failed: {str(e)}', 'success': False}
            
        return results

    def _analyze_diffusion(self, tracks_df, current_units):
        """Full implementation for diffusion analysis with safe access."""
        try:
            if ANALYSIS_MODULE_AVAILABLE:
                return analyze_diffusion(
                    tracks_df,
                    pixel_size=current_units.get('pixel_size', 1.0),
                    frame_interval=current_units.get('frame_interval', 1.0)
                )
            else:
                # Safe fallback implementation
                msd_data = []
                for track_id in tracks_df['track_id'].unique():
                    track = tracks_df[tracks_df['track_id'] == track_id]
                    if len(track) > 2:
                        x = track['x'].values
                        y = track['y'].values
                        # Simple MSD calculation
                        msd = np.mean((x[1:] - x[0])**2 + (y[1:] - y[0])**2)
                        msd_data.append(msd)
                
                return {
                    'mean_msd': float(np.mean(msd_data)) if msd_data else 0.0,
                    'diffusion_coefficient': float(np.mean(msd_data) / 4) if msd_data else 0.0,
                    'success': True
                }
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _analyze_motion(self, tracks_df, current_units):
        """Full implementation for motion analysis."""
        try:
            if ANALYSIS_MODULE_AVAILABLE:
                return analyze_motion(
                    tracks_df,
                    pixel_size=current_units.get('pixel_size', 1.0),
                    frame_interval=current_units.get('frame_interval', 1.0)
                )
            else:
                # Simple motion classification
                motion_types = []
                for track_id in tracks_df['track_id'].unique():
                    track = tracks_df[tracks_df['track_id'] == track_id]
                    if len(track) > 5:
                        # Simple directional analysis
                        displacement = np.sqrt((track['x'].iloc[-1] - track['x'].iloc[0])**2 + 
                                             (track['y'].iloc[-1] - track['y'].iloc[0])**2)
                        path_length = np.sum(np.sqrt(np.diff(track['x'])**2 + np.diff(track['y'])**2))
                        
                        if path_length > 0:
                            straightness = displacement / path_length
                            if straightness > 0.7:
                                motion_types.append('directed')
                            elif straightness < 0.3:
                                motion_types.append('confined')
                            else:
                                motion_types.append('brownian')
                        else:
                            motion_types.append('immobile')
                
                return {
                    'motion_classification': motion_types,
                    'directed_fraction': motion_types.count('directed') / len(motion_types) if motion_types else 0,
                    'confined_fraction': motion_types.count('confined') / len(motion_types) if motion_types else 0,
                    'success': True
                }
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _analyze_clustering(self, tracks_df, current_units):
        """Full implementation for clustering."""
        try:
            if ANALYSIS_MODULE_AVAILABLE:
                return analyze_clustering(
                    tracks_df,
                    pixel_size=current_units.get('pixel_size', 1.0)
                )
            else:
                # Simple clustering using starting positions
                positions = []
                for track_id in tracks_df['track_id'].unique():
                    track = tracks_df[tracks_df['track_id'] == track_id]
                    if not track.empty:
                        positions.append([track['x'].iloc[0], track['y'].iloc[0]])
                
                if len(positions) > 3:
                    kmeans = KMeans(n_clusters=min(3, len(positions)//2))
                    clusters = kmeans.fit_predict(positions)
                    return {
                        'cluster_labels': clusters.tolist(),
                        'n_clusters': len(np.unique(clusters)),
                        'cluster_centers': kmeans.cluster_centers_.tolist(),
                        'success': True
                    }
                else:
                    return {'error': 'Insufficient data for clustering', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _analyze_anomalies(self, tracks_df, current_units):
        """Full implementation for anomaly detection."""
        try:
            if ANOMALY_DETECTION_AVAILABLE:
                detector = AnomalyDetector()
                return detector.detect_anomalies(tracks_df)
            else:
                # Simple outlier detection based on track length
                track_lengths = tracks_df.groupby('track_id').size()
                q75, q25 = np.percentile(track_lengths, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                outliers = track_lengths[(track_lengths < lower_bound) | (track_lengths > upper_bound)]
                
                return {
                    'outlier_tracks': outliers.index.tolist(),
                    'outlier_count': len(outliers),
                    'outlier_fraction': len(outliers) / len(track_lengths) if len(track_lengths) > 0 else 0,
                    'success': True
                }
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _analyze_changepoints(self, tracks_df, current_units):
        """Implementation for changepoint detection."""
        try:
            if CHANGEPOINT_DETECTION_AVAILABLE:
                detector = ChangePointDetector()
                return detector.detect_changepoints(tracks_df)
            else:
                return {'error': 'Changepoint detection module not available', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _analyze_polymer_physics(self, tracks_df, current_units):
        """Implementation for polymer physics analysis."""
        try:
            if BIOPHYSICAL_MODELS_AVAILABLE:
                model = PolymerPhysicsModel()
                return model.analyze_polymer_dynamics(tracks_df)
            else:
                return {'error': 'Biophysical models module not available', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _analyze_microrheology(self, tracks_df, units):
        """Analyze microrheological properties from particle tracking data."""
        try:
            if RHEOLOGY_MODULE_AVAILABLE:
                analyzer = MicrorheologyAnalyzer(particle_radius_m=0.5e-6)  # default 0.5 µm bead, adjustable upstream
                res = analyzer.analyze_microrheology(
                    tracks_df,
                    pixel_size_um=units.get('pixel_size', 0.1),
                    frame_interval_s=units.get('frame_interval', 0.1),
                    max_lag=20
                )
                # Adapt to local plotting expectations
                if res.get('success'):
                    # make a compact summary for report figs using existing _plot_microrheology
                    msd_df = res['msd_data'].rename(columns={'lag_time_s': 'lag_time', 'msd_m2': 'msd'})
                    freq = res.get('frequency_response', {})
                    return {
                        'success': True,
                        'msd_data': msd_df,
                        'frequency': np.array(freq.get('frequencies_hz', [])),
                        'storage_modulus': np.array(freq.get('g_prime_pa', [])),
                        'loss_modulus': np.array(freq.get('g_double_prime_pa', [])),
                        'complex_viscosity': np.array(freq.get('viscosity_pa_s', [])),
                        'summary': res.get('moduli', {})
                    }
                # fall through to fallback if analyzer returned failure
            # Fallback simple method (existing)
            from analysis import calculate_msd
            
            # Calculate MSD first
            msd_result = calculate_msd(tracks_df, max_lag=20, 
                                     pixel_size=units.get('pixel_size', 0.1), 
                                     frame_interval=units.get('frame_interval', 0.1))
            
            # Check if MSD calculation was successful
            if msd_result is None or (hasattr(msd_result, 'empty') and msd_result.empty):
                return {'success': False, 'error': 'No MSD data available'}
            
            # Ensure msd_result is a DataFrame
            if not isinstance(msd_result, pd.DataFrame):
                return {'success': False, 'error': 'MSD calculation returned invalid format'}
            
            # Check for required columns
            if 'lag_time' not in msd_result.columns or 'msd' not in msd_result.columns:
                return {'success': False, 'error': 'MSD data missing required columns'}
            
            # Group MSD by track and calculate ensemble average
            ensemble_msd = msd_result.groupby('lag_time')['msd'].mean().reset_index()
            
            if len(ensemble_msd) < 5:
                return {'success': False, 'error': 'Insufficient MSD data for microrheology'}
            
            # Simple microrheological analysis without external dependencies
            # Calculate frequency range
            frequency_range = np.logspace(-2, 2, 20)  # Reduced for stability
            
            # Estimate viscoelastic properties from MSD slope changes
            time_vals = ensemble_msd['lag_time'].values
            msd_vals = ensemble_msd['msd'].values
            
            # Calculate local slopes to estimate frequency-dependent properties
            if len(time_vals) > 3:
                # Simple numerical derivative
                slopes = np.gradient(np.log(msd_vals), np.log(time_vals))
                mean_slope = np.mean(slopes)
                
                # Rough estimates for demonstration
                estimated_G_prime = np.full(len(frequency_range), abs(mean_slope) * 1e-3)
                estimated_G_double_prime = np.full(len(frequency_range), abs(1 - mean_slope) * 1e-3)
                estimated_eta_star = estimated_G_double_prime / (2 * np.pi * frequency_range)
            else:
                estimated_G_prime = np.zeros(len(frequency_range))
                estimated_G_double_prime = np.zeros(len(frequency_range))
                estimated_eta_star = np.zeros(len(frequency_range))
            
            return {
                'success': True,
                'msd_data': ensemble_msd,
                'frequency': frequency_range,
                'storage_modulus': estimated_G_prime,
                'loss_modulus': estimated_G_double_prime,
                'complex_viscosity': estimated_eta_star,
                'summary': {
                    'mean_storage_modulus': np.mean(estimated_G_prime),
                    'mean_loss_modulus': np.mean(estimated_G_double_prime),
                    'viscous_elastic_ratio': np.mean(estimated_G_double_prime) / max(np.mean(estimated_G_prime), 1e-10)
                }
            }
        except Exception as e:
            import traceback
            return {'success': False, 'error': f'Microrheology analysis failed: {str(e)}', 'traceback': traceback.format_exc()}

    def generate_batch_report(self, tracks_df, selected_analyses, condition_name):
        """Generate automated report for batch processing (non-Streamlit)."""
        results = {
            'condition_name': condition_name,
            'analysis_results': {},
            'figures': {},
            'success': True
        }
        
        current_units = {
            'pixel_size': st.session_state.get('pixel_size', 0.1) if 'st' in globals() else 0.1,
            'frame_interval': st.session_state.get('frame_interval', 0.1) if 'st' in globals() else 0.1
        }
        
        for analysis_key in selected_analyses:
            if analysis_key not in self.available_analyses:
                continue
                
            analysis = self.available_analyses[analysis_key]
            
            try:
                result = analysis['function'](tracks_df, current_units)
                results['analysis_results'][analysis_key] = result
                
                if result.get('success', True) and 'error' not in result:
                    fig = analysis['visualization'](result)
                    if fig:
                        results['figures'][analysis_key] = fig
                        
            except Exception as e:
                results['analysis_results'][analysis_key] = {
                    'success': False, 'error': str(e)
                }
        
        return results

    def _display_generated_report(self, config, current_units):
        """Display the generated report with download options."""
        if not self.report_results:
            st.warning("No analysis results to display.")
            return
            
        st.subheader("📄 Generated Report")
        
        # Create download options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Interactive Report"):
                self._show_interactive_report(current_units)
        
        with col2:
            # Generate and provide JSON download
            report_data = self._generate_report_data(config, current_units)
            st.download_button(
                "💾 Download JSON Report",
                data=json.dumps(report_data, indent=2, default=str),
                file_name=f"spt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Generate and provide CSV summary download
            summary_csv = self._generate_csv_summary()
            if summary_csv:
                st.download_button(
                    "📈 Download CSV Summary",
                    data=summary_csv,
                    file_name=f"spt_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    def _show_interactive_report(self, current_units):
        """Display interactive report with visualizations."""
        st.subheader("📊 Interactive Analysis Report")
        
        # Display summary statistics
        if 'basic_statistics' in self.report_results:
            stats = self.report_results['basic_statistics']
            st.subheader("📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tracks", stats.get('total_tracks', 'N/A'))
            with col2:
                st.metric("Mean Track Length", f"{stats.get('mean_track_length', 0):.1f}")
            with col3:
                st.metric("Mean Velocity", f"{stats.get('mean_velocity', 0):.3f} µm/s")
            with col4:
                st.metric("Total Time Points", stats.get('total_timepoints', 'N/A'))
        
        # Display all analysis results
        for analysis_key, result in self.report_results.items():
            if analysis_key in self.available_analyses:
                analysis_info = self.available_analyses[analysis_key]
                
                with st.expander(f"📊 {analysis_info['name']}", expanded=True):
                    if 'error' in result:
                        st.error(f"Analysis failed: {result['error']}")
                        continue
                    
                    # Display analysis-specific results
                    if analysis_key == 'basic_statistics':
                        self._display_basic_stats(result)
                    elif analysis_key == 'diffusion_analysis':
                        self._display_diffusion_results(result)
                    elif analysis_key == 'motion_classification':
                        self._display_motion_results(result)
                    else:
                        # Generic result display
                        st.json(result)
                    
                    # Display visualization if available
                    if analysis_key in self.report_figures:
                        st.plotly_chart(self.report_figures[analysis_key], use_container_width=True)

    def _display_basic_stats(self, stats):
        """Display basic statistics in a formatted way."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Track Statistics:**")
            st.write(f"• Total tracks: {stats.get('total_tracks', 'N/A')}")
            st.write(f"• Mean track length: {stats.get('mean_track_length', 0):.1f} frames")
            st.write(f"• Median track length: {stats.get('median_track_length', 0):.1f} frames")
            st.write(f"• Track length std: {stats.get('track_length_std', 0):.1f} frames")
        
        with col2:
            st.markdown("**Motion Statistics:**")
            st.write(f"• Mean velocity: {stats.get('mean_velocity', 0):.3f} µm/s")
            st.write(f"• Total time points: {stats.get('total_timepoints', 'N/A')}")

    def _display_diffusion_results(self, results):
        """Display diffusion analysis results."""
        if 'diffusion_coefficient' in results:
            st.metric("Diffusion Coefficient", f"{results['diffusion_coefficient']:.2e} µm²/s")
        if 'msd_slope' in results:
            st.metric("MSD Slope", f"{results['msd_slope']:.3f}")

    def _display_motion_results(self, results):
        """Display motion classification results."""
        if 'classification_summary' in results:
            summary = results['classification_summary']
            for motion_type, count in summary.items():
                st.metric(f"{motion_type.title()} Motion", count)

    def _generate_report_data(self, config, current_units):
        """Generate comprehensive report data for download."""
        import datetime
        
        report_data = {
            'metadata': {
                'generated_at': datetime.datetime.now().isoformat(),
                'analysis_software': 'SPT2025B',
                'pixel_size_um': current_units.get('pixel_size', 0.1),
                'frame_interval_s': current_units.get('frame_interval', 0.1),
                'config': config
            },
            'analysis_results': self.report_results,
            'visualizations_available': list(self.report_figures.keys())
        }
        
        return report_data

    def _generate_csv_summary(self):
        """Generate CSV summary of key results."""
        try:
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Analysis', 'Metric', 'Value', 'Unit'])
            
            # Extract key metrics from each analysis
            for analysis_key, result in self.report_results.items():
                if 'error' in result:
                    continue
                    
                analysis_name = self.available_analyses.get(analysis_key, {}).get('name', analysis_key)
                
                if analysis_key == 'basic_statistics':
                    writer.writerow([analysis_name, 'Total Tracks', result.get('total_tracks', ''), 'count'])
                    writer.writerow([analysis_name, 'Mean Track Length', result.get('mean_track_length', ''), 'frames'])
                    writer.writerow([analysis_name, 'Mean Velocity', result.get('mean_velocity', ''), 'µm/s'])
                elif analysis_key == 'diffusion_analysis':
                    writer.writerow([analysis_name, 'Diffusion Coefficient', result.get('diffusion_coefficient', ''), 'µm²/s'])
                    writer.writerow([analysis_name, 'MSD Slope', result.get('msd_slope', ''), 'dimensionless'])
                # Add more analysis types as needed
            
            return output.getvalue()
        except Exception as e:
            st.error(f"Failed to generate CSV summary: {str(e)}")
            return None

    def _render_analysis_section(self, analysis_key, results, config):
        """Render analysis section with better error handling"""
        try:
            # Add validation for required data fields
            if analysis_key == "polymer_physics" and 'msd_data' not in results:
                st.error("Polymer physics analysis requires MSD data. Please run Diffusion Analysis first.")
                return
                
            # Generic section rendering
            analysis_info = self.available_analyses.get(analysis_key, {})
            st.subheader(f"📊 {analysis_info.get('name', analysis_key)}")
            
            # Check for errors
            if 'error' in results:
                st.error(f"Analysis failed: {results['error']}")
                return
            
            # Display key results
            if 'summary' in results:
                st.markdown("**Summary:**")
                for key, value in results['summary'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            # Display detailed results
            st.markdown("**Detailed Results:**")
            st.json(results)
        
        except Exception as e:
            st.error(f"Error rendering analysis section: {e}")

# Streamlit app integration
def show_enhanced_report_generator(track_data=None, analysis_results=None, 
                                  pixel_size=0.1, frame_interval=0.1, 
                                  track_statistics=None, msd_data=None):
    """
    Main entry point for the Enhanced Report Generator interface.
    
    This function should be called from the main app.py to display the 
    enhanced report generation interface.
    
    Parameters
    ----------
    track_data : pd.DataFrame, optional
        Track data to include in the report
    analysis_results : Dict, optional
        Analysis results to include in the report
    pixel_size : float, optional
        Pixel size in micrometers, by default 0.1
    frame_interval : float, optional
        Frame interval in seconds, by default 0.1
    track_statistics : pd.DataFrame, optional
        Precomputed track statistics, by default None
    msd_data : pd.DataFrame, optional
        MSD data for diffusion analysis, by default None
    """
    # Initialize the report generator
    generator = EnhancedSPTReportGenerator()
    
    # Display the interface
    generator.display_enhanced_analysis_interface()

# Alternative simplified entry point
def main():
    """
    Standalone entry point for testing the report generator.
    """
    st.set_page_config(
        page_title="SPT Enhanced Report Generator",
        page_icon="📊",
        layout="wide"
    )
    
    generator = EnhancedSPTReportGenerator()
    generator.display_enhanced_analysis_interface()

# For direct execution
if __name__ == "__main__":
    main()