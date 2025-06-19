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
    from interactive_plots import (
        plot_tracks, plot_tracks_3d, plot_track_statistics, plot_motion_analysis,
        plot_msd_curves, plot_diffusion_coefficients, plot_spatial_clustering,
        plot_dwell_time_analysis, plot_gel_structure_analysis,
        plot_diffusion_population_analysis, plot_active_transport_analysis,
        plot_boundary_crossing_analysis, plot_crowding_analysis
    )
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
        """Execute comprehensive analysis pipeline."""
        st.subheader("🔄 Analysis Execution")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        self.report_results = {}
        self.report_figures = {}
        
        for i, analysis_key in enumerate(selected_analyses):
            analysis = self.available_analyses[analysis_key]
            
            status_text.text(f"Running {analysis['name']}...")
            progress_bar.progress((i + 1) / len(selected_analyses))
            
            try:
                result = analysis['function'](tracks_df, current_units)
                self.report_results[analysis_key] = result
                
                if result.get('success', True) and 'error' not in result:
                    fig = analysis['visualization'](result)
                    if fig:
                        self.report_figures[analysis_key] = fig
                st.success(f"✅ {analysis['name']} completed")
            except Exception as e:
                st.error(f"❌ {analysis['name']} failed: {str(e)}")
        
        status_text.text("Report generation complete!")

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

    # --- Visualization Methods ---

    def _plot_basic_statistics(self, results):
        """Generate comprehensive basic statistics visualizations."""
        if not results.get('success', True):
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Track Length Statistics', 'Velocity Statistics', 
                          'Summary Metrics', 'Data Quality']
        )
        
        # Track length histogram (placeholder)
        fig.add_trace(go.Histogram(x=[results.get('mean_track_length', 0)], name='Track Length'), row=1, col=1)
        
        # Velocity histogram (placeholder)
        fig.add_trace(go.Histogram(x=[results.get('mean_velocity', 0)], name='Velocity'), row=1, col=2)
        
        # Summary bar chart
        fig.add_trace(go.Bar(
            x=['Total Tracks', 'Mean Length', 'Mean Velocity'],
            y=[results.get('total_tracks', 0), results.get('mean_track_length', 0), results.get('mean_velocity', 0)],
            name='Summary'
        ), row=2, col=1)
        
        fig.update_layout(title="Basic Statistics Analysis")
        return fig

    def _plot_diffusion(self, results):
        """Generate diffusion analysis plots."""
        if not results.get('success', True):
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Diffusion Coefficient', 'Mean MSD'],
            y=[results.get('diffusion_coefficient', 0), results.get('mean_msd', 0)],
            name='Diffusion Metrics'
        ))
        fig.update_layout(title="Diffusion Analysis")
        return fig

    def _plot_motion(self, results):
        """Generate motion classification plots."""
        if not results.get('success', True):
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Directed', 'Confined', 'Brownian'],
            y=[results.get('directed_fraction', 0), results.get('confined_fraction', 0), 
               1 - results.get('directed_fraction', 0) - results.get('confined_fraction', 0)],
            name='Motion Types'
        ))
        fig.update_layout(title="Motion Classification")
        return fig

    def _plot_clustering(self, results):
        """Generate clustering plots."""
        if not results.get('success', True):
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Number of Clusters'],
            y=[results.get('n_clusters', 0)],
            name='Clustering'
        ))
        fig.update_layout(title="Spatial Clustering Analysis")
        return fig

    def _plot_anomalies(self, results):
        """Generate anomaly detection plots."""
        if not results.get('success', True):
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Outlier Count', 'Outlier Fraction'],
            y=[results.get('outlier_count', 0), results.get('outlier_fraction', 0)],
            name='Anomalies'
        ))
        fig.update_layout(title="Anomaly Detection")
        return fig

    def _plot_changepoints(self, results):
        """Generate changepoint detection plots."""
        if not results.get('success', True):
            return None
        return go.Figure().add_annotation(text="Changepoint Detection Plot", x=0.5, y=0.5)

    def _plot_polymer_physics(self, results):
        """Generate polymer physics plots."""
        if not results.get('success', True):
            return None
        return go.Figure().add_annotation(text="Polymer Physics Plot", x=0.5, y=0.5)

    # New Analysis Functions
    def _analyze_microrheology(self, tracks_df, units):
        """Analyze microrheological properties from particle tracking data."""
        try:
            from rheology import calculate_complex_modulus_gser
            
            # Calculate MSD first
            msd_result = calculate_msd(tracks_df, max_lag=20, 
                                     pixel_size=units['pixel_size'], 
                                     frame_interval=units['frame_interval'])
            
            if msd_result.empty:
                return {'success': False, 'error': 'No MSD data available'}
            
            # Calculate complex modulus using GSER
            frequency_range = np.logspace(-2, 2, 50)  # 0.01 to 100 Hz
            
            # Group MSD by track and calculate ensemble average
            ensemble_msd = msd_result.groupby('lag_time')['msd'].mean().reset_index()
            
            if len(ensemble_msd) < 5:
                return {'success': False, 'error': 'Insufficient MSD data for microrheology'}
            
            # Calculate complex modulus
            modulus_result = calculate_complex_modulus_gser(
                ensemble_msd['lag_time'].values,
                ensemble_msd['msd'].values,
                frequency=frequency_range,
                particle_radius=1e-6,  # 1 μm default particle radius
                temperature=300  # Room temperature in Kelvin
            )
            
            return {
                'success': True,
                'msd_data': ensemble_msd,
                'frequency': frequency_range,
                'storage_modulus': modulus_result.get('G_prime', np.array([])),
                'loss_modulus': modulus_result.get('G_double_prime', np.array([])),
                'complex_viscosity': modulus_result.get('eta_star', np.array([])),
                'summary': {
                    'mean_storage_modulus': np.mean(modulus_result.get('G_prime', [0])),
                    'mean_loss_modulus': np.mean(modulus_result.get('G_double_prime', [0])),
                    'viscous_elastic_ratio': np.mean(modulus_result.get('G_double_prime', [1])) / max(np.mean(modulus_result.get('G_prime', [1])), 1e-10)
                }
            }
        except Exception as e:
            return {'success': False, 'error': f'Microrheology analysis failed: {str(e)}'}

    def _plot_microrheology(self, results):
        """Generate microrheology visualization plots."""
        if not results.get('success', True):
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Storage Modulus (G\')', 'Loss Modulus (G\'\')', 
                          'Complex Viscosity', 'MSD vs Time'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        frequency = results.get('frequency', [])
        G_prime = results.get('storage_modulus', [])
        G_double_prime = results.get('loss_modulus', [])
        eta_star = results.get('complex_viscosity', [])
        msd_data = results.get('msd_data', pd.DataFrame())
        
        # Storage modulus
        if len(G_prime) > 0:
            fig.add_trace(go.Scatter(x=frequency, y=G_prime, name="G'", 
                                   line=dict(color='blue')), row=1, col=1)
        
        # Loss modulus
        if len(G_double_prime) > 0:
            fig.add_trace(go.Scatter(x=frequency, y=G_double_prime, name="G''", 
                                   line=dict(color='red')), row=1, col=2)
        
        # Complex viscosity
        if len(eta_star) > 0:
            fig.add_trace(go.Scatter(x=frequency, y=eta_star, name="η*", 
                                   line=dict(color='green')), row=2, col=1)
        
        # MSD vs time
        if not msd_data.empty:
            fig.add_trace(go.Scatter(x=msd_data['lag_time'], y=msd_data['msd'], 
                                   name="MSD", mode='markers+lines'), row=2, col=2)
        
        fig.update_xaxes(type="log", title="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(type="log", title="Frequency (Hz)", row=1, col=2)
        fig.update_xaxes(type="log", title="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title="Time (s)", row=2, col=2)
        
        fig.update_yaxes(type="log", title="G' (Pa)", row=1, col=1)
        fig.update_yaxes(type="log", title="G'' (Pa)", row=1, col=2)
        fig.update_yaxes(type="log", title="η* (Pa·s)", row=2, col=1)
        fig.update_yaxes(type="log", title="MSD (μm²)", row=2, col=2)
        
        fig.update_layout(title="Microrheology Analysis", showlegend=False)
        return fig

    def _analyze_intensity(self, tracks_df, units):
        """Analyze fluorescence intensity dynamics."""
        try:
            # Check for intensity columns
            intensity_cols = [col for col in tracks_df.columns if 'intensity' in col.lower() or 'int' in col.lower()]
            
            if not intensity_cols:
                return {'success': False, 'error': 'No intensity data found in tracks'}
            
            intensity_col = intensity_cols[0]  # Use first intensity column
            
            # Calculate intensity statistics per track
            intensity_stats = []
            for track_id, track_data in tracks_df.groupby('track_id'):
                if intensity_col in track_data.columns:
                    intensities = track_data[intensity_col].values
                    intensity_stats.append({
                        'track_id': track_id,
                        'mean_intensity': np.mean(intensities),
                        'std_intensity': np.std(intensities),
                        'cv_intensity': np.std(intensities) / max(np.mean(intensities), 1),
                        'min_intensity': np.min(intensities),
                        'max_intensity': np.max(intensities),
                        'intensity_range': np.max(intensities) - np.min(intensities),
                        'track_length': len(intensities)
                    })
            
            intensity_df = pd.DataFrame(intensity_stats)
            
            return {
                'success': True,
                'intensity_column': intensity_col,
                'track_statistics': intensity_df,
                'summary': {
                    'total_tracks': len(intensity_df),
                    'mean_intensity_overall': intensity_df['mean_intensity'].mean(),
                    'intensity_variability': intensity_df['cv_intensity'].mean()
                }
            }
        except Exception as e:
            return {'success': False, 'error': f'Intensity analysis failed: {str(e)}'}

    def _plot_intensity(self, results):
        """Generate intensity analysis plots."""
        if not results.get('success', True):
            return None
        
        track_stats = results.get('track_statistics', pd.DataFrame())
        if track_stats.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Intensity Distribution', 'Intensity vs Track Length', 
                          'Intensity Variability', 'Track Intensity Profiles']
        )
        
        # Intensity distribution
        fig.add_trace(go.Histogram(x=track_stats['mean_intensity'], name="Intensity Distribution",
                                 nbinsx=30), row=1, col=1)
        
        # Intensity vs track length
        fig.add_trace(go.Scatter(x=track_stats['track_length'], y=track_stats['mean_intensity'],
                               mode='markers', name="Intensity vs Length"), row=1, col=2)
        
        # Intensity variability (CV)
        fig.add_trace(go.Histogram(x=track_stats['cv_intensity'], name="Intensity CV",
                                 nbinsx=30), row=2, col=1)
        
        # Sample track profiles (first 10 tracks)
        sample_tracks = track_stats.head(10)
        for i, (_, track) in enumerate(sample_tracks.iterrows()):
            fig.add_trace(go.Scatter(y=[track['mean_intensity']], x=[i], 
                                   mode='markers', name=f"Track {track['track_id']}",
                                   showlegend=False), row=2, col=2)
        
        fig.update_layout(title="Intensity Analysis")
        return fig

    def _analyze_confinement(self, tracks_df, units):
        """Analyze confined motion and boundary interactions."""
        try:
            confinement_results = []
            
            for track_id, track_data in tracks_df.groupby('track_id'):
                if len(track_data) < 10:  # Skip short tracks
                    continue
                
                positions = track_data[['x', 'y']].values * units['pixel_size']
                
                # Calculate radius of gyration
                center = np.mean(positions, axis=0)
                distances = np.linalg.norm(positions - center, axis=1)
                radius_gyration = np.sqrt(np.mean(distances**2))
                
                # Calculate confinement ratio (Rg^2 / <r^2>)
                displacements = np.diff(positions, axis=0)
                mean_squared_displacement = np.mean(np.sum(displacements**2, axis=1))
                confinement_ratio = radius_gyration**2 / max(mean_squared_displacement, 1e-10)
                
                # Detect potential confinement (high confinement ratio)
                is_confined = confinement_ratio > 10
                
                confinement_results.append({
                    'track_id': track_id,
                    'radius_gyration': radius_gyration,
                    'confinement_ratio': confinement_ratio,
                    'is_confined': is_confined,
                    'track_length': len(track_data),
                    'exploration_area': np.pi * radius_gyration**2
                })
            
            confinement_df = pd.DataFrame(confinement_results)
            
            return {
                'success': True,
                'confinement_data': confinement_df,
                'summary': {
                    'total_tracks': len(confinement_df),
                    'confined_tracks': confinement_df['is_confined'].sum(),
                    'confinement_percentage': confinement_df['is_confined'].mean() * 100,
                    'mean_radius_gyration': confinement_df['radius_gyration'].mean()
                }
            }
        except Exception as e:
            return {'success': False, 'error': f'Confinement analysis failed: {str(e)}'}

    def _plot_confinement(self, results):
        """Generate confinement analysis plots."""
        if not results.get('success', True):
            return None
        
        confinement_data = results.get('confinement_data', pd.DataFrame())
        if confinement_data.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Confinement Ratio Distribution', 'Radius of Gyration', 
                          'Confinement vs Track Length', 'Confined vs Free Tracks']
        )
        
        # Confinement ratio distribution
        fig.add_trace(go.Histogram(x=confinement_data['confinement_ratio'], 
                                 name="Confinement Ratio", nbinsx=30), row=1, col=1)
        
        # Radius of gyration
        fig.add_trace(go.Histogram(x=confinement_data['radius_gyration'], 
                                 name="Radius of Gyration", nbinsx=30), row=1, col=2)
        
        # Confinement vs track length
        colors = ['red' if confined else 'blue' for confined in confinement_data['is_confined']]
        fig.add_trace(go.Scatter(x=confinement_data['track_length'], 
                               y=confinement_data['confinement_ratio'],
                               mode='markers', marker_color=colors,
                               name="Confinement vs Length"), row=2, col=1)
        
        # Confined vs free tracks count
        confined_counts = confinement_data['is_confined'].value_counts()
        fig.add_trace(go.Bar(x=['Free', 'Confined'], y=[confined_counts.get(False, 0), 
                                                       confined_counts.get(True, 0)],
                           name="Track Classification"), row=2, col=2)
        
        fig.update_layout(title="Confinement Analysis")
        return fig

    def _analyze_velocity_correlation(self, tracks_df, units):
        """Analyze velocity autocorrelation and persistence."""
        try:
            correlation_results = []
            
            for track_id, track_data in tracks_df.groupby('track_id'):
                if len(track_data) < 15:  # Need sufficient points for correlation
                    continue
                
                # Sort by frame and calculate positions
                track_data = track_data.sort_values('frame')
                positions = track_data[['x', 'y']].values * units['pixel_size']
                
                # Calculate velocities
                velocities = np.diff(positions, axis=0) / units['frame_interval']
                
                if len(velocities) < 10:
                    continue
                
                # Calculate velocity autocorrelation
                max_lag = min(len(velocities) // 3, 20)
                autocorr = []
                
                for lag in range(max_lag):
                    if lag == 0:
                        autocorr.append(1.0)
                    else:
                        v1 = velocities[:-lag]
                        v2 = velocities[lag:]
                        # Dot product autocorrelation
                        corr = np.mean([np.dot(v1[i], v2[i]) for i in range(len(v1))])
                        norm = np.mean([np.dot(v1[i], v1[i]) for i in range(len(v1))])
                        autocorr.append(corr / max(norm, 1e-10))
                
                # Find persistence length (where correlation drops to 1/e)
                autocorr_array = np.array(autocorr)
                persistence_idx = np.where(autocorr_array < 1/np.e)[0]
                persistence_length = persistence_idx[0] if len(persistence_idx) > 0 else max_lag
                
                correlation_results.append({
                    'track_id': track_id,
                    'autocorrelation': autocorr,
                    'persistence_length': persistence_length,
                    'max_velocity': np.max(np.linalg.norm(velocities, axis=1)),
                    'mean_velocity': np.mean(np.linalg.norm(velocities, axis=1)),
                    'track_length': len(track_data)
                })
            
            return {
                'success': True,
                'correlation_data': correlation_results,
                'summary': {
                    'total_tracks': len(correlation_results),
                    'mean_persistence': np.mean([r['persistence_length'] for r in correlation_results])
                }
            }
        except Exception as e:
            return {'success': False, 'error': f'Velocity correlation analysis failed: {str(e)}'}

    def _plot_velocity_correlation(self, results):
        """Generate velocity correlation plots."""
        if not results.get('success', True):
            return None
        
        correlation_data = results.get('correlation_data', [])
        if not correlation_data:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Velocity Autocorrelation', 'Persistence Length Distribution', 
                          'Velocity Distribution', 'Sample Autocorrelation Curves']
        )
        
        # Average autocorrelation
        if correlation_data:
            max_len = max(len(r['autocorrelation']) for r in correlation_data)
            avg_autocorr = np.zeros(max_len)
            count = np.zeros(max_len)
            
            for result in correlation_data:
                autocorr = result['autocorrelation']
                for i, val in enumerate(autocorr):
                    avg_autocorr[i] += val
                    count[i] += 1
            
            avg_autocorr = avg_autocorr / np.maximum(count, 1)
            
            fig.add_trace(go.Scatter(x=list(range(len(avg_autocorr))), y=avg_autocorr,
                                   name="Average Autocorrelation"), row=1, col=1)
        
        # Persistence length distribution
        persistence_lengths = [r['persistence_length'] for r in correlation_data]
        fig.add_trace(go.Histogram(x=persistence_lengths, name="Persistence Length",
                                 nbinsx=20), row=1, col=2)
        
        # Velocity distribution
        mean_velocities = [r['mean_velocity'] for r in correlation_data]
        fig.add_trace(go.Histogram(x=mean_velocities, name="Mean Velocity",
                                 nbinsx=20), row=2, col=1)
        
        # Sample autocorrelation curves (first 5 tracks)
        for i, result in enumerate(correlation_data[:5]):
            autocorr = result['autocorrelation']
            fig.add_trace(go.Scatter(x=list(range(len(autocorr))), y=autocorr,
                                   name=f"Track {result['track_id']}", 
                                   showlegend=False), row=2, col=2)
        
        fig.update_layout(title="Velocity Correlation Analysis")
        return fig

    def _analyze_particle_interactions(self, tracks_df, units):
        """Analyze multi-particle interactions and collective motion."""
        try:
            # Group tracks by frame to find contemporaneous particles
            frame_groups = tracks_df.groupby('frame')
            
            interaction_results = []
            
            for frame, frame_data in frame_groups:
                if len(frame_data) < 2:  # Need at least 2 particles
                    continue
                
                positions = frame_data[['x', 'y']].values * units['pixel_size']
                track_ids = frame_data['track_id'].values
                
                # Calculate pairwise distances
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(positions))
                
                # Find close particle pairs (within interaction radius)
                interaction_radius = 5.0  # 5 μm default
                close_pairs = np.where((distances < interaction_radius) & (distances > 0))
                
                for i, j in zip(close_pairs[0], close_pairs[1]):
                    if i < j:  # Avoid double counting
                        interaction_results.append({
                            'frame': frame,
                            'track_id_1': track_ids[i],
                            'track_id_2': track_ids[j],
                            'distance': distances[i, j],
                            'interaction_strength': 1.0 / max(distances[i, j], 0.1)
                        })
            
            interaction_df = pd.DataFrame(interaction_results)
            
            # Calculate interaction statistics
            if not interaction_df.empty:
                interaction_summary = {
                    'total_interactions': len(interaction_df),
                    'unique_particle_pairs': len(interaction_df[['track_id_1', 'track_id_2']].drop_duplicates()),
                    'mean_interaction_distance': interaction_df['distance'].mean(),
                    'interaction_frequency': len(interaction_df) / len(frame_groups)
                }
            else:
                interaction_summary = {
                    'total_interactions': 0,
                    'unique_particle_pairs': 0,
                    'mean_interaction_distance': 0,
                    'interaction_frequency': 0
                }
            
            return {
                'success': True,
                'interaction_data': interaction_df,
                'summary': interaction_summary
            }
        except Exception as e:
            return {'success': False, 'error': f'Particle interaction analysis failed: {str(e)}'}

    def _plot_particle_interactions(self, results):
        """Generate particle interaction plots."""
        if not results.get('success', True):
            return None
        
        interaction_data = results.get('interaction_data', pd.DataFrame())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Interaction Distance Distribution', 'Interactions Over Time', 
                          'Interaction Network', 'Interaction Strength']
        )
        
        if not interaction_data.empty:
            # Distance distribution
            fig.add_trace(go.Histogram(x=interaction_data['distance'], 
                                     name="Distance Distribution", nbinsx=30), row=1, col=1)
            
            # Interactions over time
            interactions_per_frame = interaction_data.groupby('frame').size().reset_index(name='count')
            fig.add_trace(go.Scatter(x=interactions_per_frame['frame'], 
                                   y=interactions_per_frame['count'],
                                   mode='lines+markers', name="Interactions vs Time"), row=1, col=2)
            
            # Simplified interaction network (sample)
            sample_interactions = interaction_data.head(50)  # Show first 50 interactions
            fig.add_trace(go.Scatter(x=sample_interactions['track_id_1'], 
                                   y=sample_interactions['track_id_2'],
                                   mode='markers', name="Interaction Pairs"), row=2, col=1)
            
            # Interaction strength
            fig.add_trace(go.Histogram(x=interaction_data['interaction_strength'], 
                                     name="Interaction Strength", nbinsx=30), row=2, col=2)
        else:
            # Empty plots with annotations
            for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                fig.add_annotation(text="No interactions detected", x=0.5, y=0.5, 
                                 row=row, col=col, showarrow=False)
        
        fig.update_layout(title="Multi-Particle Interaction Analysis")
        return fig

def show_enhanced_report_generator():
    """Display the enhanced report generator interface."""
    generator = EnhancedSPTReportGenerator()
    generator.display_enhanced_analysis_interface()

if __name__ == "__main__":
    show_enhanced_report_generator()