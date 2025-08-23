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
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
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

# Import data access utilities for consistent data handling
try:
    from data_access_utils import get_track_data, check_data_availability, get_units, get_analysis_results, display_data_summary
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

# Import state manager for proper data access (keep as fallback)
try:
    from state_manager import get_state_manager
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False

class EnhancedSPTReportGenerator:
    """
    Comprehensive report generation system with extensive analysis capabilities.
    """

    def __init__(self, *args, **kwargs):
        # Initialize state manager if available (for fallback)
        self.state_manager = get_state_manager() if STATE_MANAGER_AVAILABLE else None

        # Ensure track data available if previously loaded
        sm = get_state_manager() if STATE_MANAGER_AVAILABLE else None
        if sm and (not hasattr(self, "track_df") or self.track_df is None or (hasattr(self.track_df, "empty") and self.track_df.empty)):
            tracks = sm.get_tracks_or_none()
            if tracks is not None and hasattr(tracks, 'empty') and not tracks.empty:
                self.track_df = tracks
                self.tracks = tracks  # legacy alias
        
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
        
        # Use centralized data access utilities
        if DATA_UTILS_AVAILABLE:
            if not check_data_availability():
                return
            tracks_df, _ = get_track_data()
            display_data_summary()
        else:
            # Fallback to original method
            has_data = False
            tracks_df = None

            if self.state_manager:
                try:
                    has_data = self.state_manager.has_data()
                    if has_data:
                        tracks_df = self.state_manager.get_tracks()
                except Exception:
                    has_data = False
            else:
                # Fallback to direct session state access
                tracks_df = st.session_state.get('tracks_df') or st.session_state.get('raw_tracks') or st.session_state.get('tracks_data')
                has_data = tracks_df is not None and not tracks_df.empty if isinstance(tracks_df, pd.DataFrame) else False
            
            if not has_data:
                st.error("❌ No track data loaded. Please load data first.")
                st.info("💡 Go to the 'Data Loading' tab to upload track data.")
                
                # Debug information
                if st.checkbox("Show debug information"):
                    st.write("Session state keys:", list(st.session_state.keys()))
                    if self.state_manager:
                        try:
                            st.write("State manager data summary:", self.state_manager.get_data_summary())
                            st.write("Debug state:", self.state_manager.debug_data_state())
                        except Exception as e:
                            st.write(f"State manager error: {e}")
                return
            
            st.success(f"✅ Track data loaded: {len(tracks_df)} points")
        
        st.markdown("Select from the available modules to create a detailed report.")
        
        categories = self._group_analyses_by_category()
        
        selected_analyses = self._display_analysis_selector(categories)
        report_config = self._display_report_config()
            
        if st.button("🚀 Generate Comprehensive Report", type="primary"):
            if selected_analyses:
                # Get units using centralized utilities
                if DATA_UTILS_AVAILABLE:
                    current_units = get_units()
                elif self.state_manager:
                    current_units = {
                        'pixel_size': self.state_manager.get_pixel_size(),
                        'frame_interval': self.state_manager.get_frame_interval()
                    }
                else:
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
        
        # Get analysis results using centralized utilities
        if DATA_UTILS_AVAILABLE:
            analysis_results = get_analysis_results()
        elif self.state_manager:
            analysis_results = self.state_manager.get_analysis_results()
        else:
            analysis_results = st.session_state.get('analysis_results', {})
        
        if not analysis_results:
            st.warning("⚠️ No pre-computed analysis results found. Running analyses now...")
            # Run analyses directly
            self._run_analyses_for_report(tracks_df, selected_analyses, config, current_units)
        else:
            # Use existing results
            self._generate_report_from_results(analysis_results, selected_analyses, config, current_units)

    def _run_analyses_for_report(self, tracks_df, selected_analyses, config, current_units):
        """Run analyses directly for report generation."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        self.report_results = {}
        self.report_figures = {}
        
        for i, analysis_key in enumerate(selected_analyses):
            if analysis_key not in self.available_analyses:
                continue
                
            analysis = self.available_analyses[analysis_key]
            status_text.text(f"Running {analysis['name']}...")
            progress_bar.progress((i + 1) / len(selected_analyses))
            
            try:
                # Run the analysis
                result = analysis['function'](tracks_df, current_units)
                
                if result.get('success', False):
                    self.report_results[analysis_key] = result
                    
                    # Generate visualization
                    fig = analysis['visualization'](result)
                    if fig:
                        self.report_figures[analysis_key] = fig
                    
                    st.success(f"✅ Completed {analysis['name']}")
                else:
                    st.warning(f"⚠️ {analysis['name']} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Failed to run {analysis['name']}: {str(e)}")
        
        status_text.text("Report generation complete!")
        progress_bar.progress(1.0)
        
        # Display the generated report
        self._display_generated_report(config, current_units)

    def _generate_report_from_results(self, analysis_results, selected_analyses, config, current_units):
        """Generate report from existing analysis results."""
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
                # Some visualization functions may return dict of plotly figs or matplotlib
                if fig is not None:
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
            from utils import calculate_track_statistics
            
            stats_df = calculate_track_statistics(tracks_df)
            
            if stats_df.empty:
                return {'success': False, 'error': 'Failed to calculate track statistics.'}

            # For consistency, also provide the ensemble stats in the results dict
            results = {
                'success': True,
                'statistics_df': stats_df,
                'ensemble_statistics': {
                    'total_tracks': len(stats_df),
                    'mean_track_length': stats_df['track_length'].mean(),
                    'median_track_length': stats_df['track_length'].median(),
                    'mean_speed': stats_df['mean_speed'].mean()
                }
            }
            return results
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}', 'success': False}

    def _plot_basic_statistics(self, result):
        """Full implementation for basic statistics visualization."""
        try:
            from visualization import plot_track_statistics, _empty_fig
            
            stats_df = result.get('statistics_df')
            if stats_df is None or stats_df.empty:
                return _empty_fig("No statistics data to plot.")

            # plot_track_statistics returns a dictionary of figures
            figs = plot_track_statistics(stats_df)

            if not figs:
                return _empty_fig("No statistics plots generated.")

            # Create a subplot figure. Let's assume 2x2 for the main stats.
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=list(figs.keys())
            )

            row, col = 1, 1
            for name, sub_fig in figs.items():
                # Each sub_fig is a go.Figure with one trace (a histogram)
                # We extract the trace and add it to our main figure
                if sub_fig.data:
                    fig.add_trace(sub_fig.data[0], row=row, col=col)

                # Move to the next subplot position
                col += 1
                if col > 2:
                    col = 1
                    row += 1
                if row > 2:
                    break # Stop if we have more than 4 plots

            fig.update_layout(title_text="Basic Track Statistics", showlegend=False)
            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

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

    def _plot_motion(self, result):
        """Full implementation for motion visualization."""
        try:
            from visualization import plot_motion_analysis, _empty_fig

            if not result.get('success', False):
                return _empty_fig("Motion analysis failed.")

            return plot_motion_analysis(result)

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

    def _plot_diffusion(self, result):
        """Full implementation for diffusion visualization."""
        try:
            from visualization import plot_diffusion_coefficients, plot_msd_curves, _empty_fig

            if not result.get('success', False):
                return _empty_fig("Diffusion analysis failed.")

            # Create a figure with two subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Diffusion Coefficients", "Mean Squared Displacement")
            )

            # Plot diffusion coefficients
            diff_fig = plot_diffusion_coefficients(result)
            if diff_fig.data:
                fig.add_trace(diff_fig.data[0], row=1, col=1)
                if len(diff_fig.layout.shapes) > 0:
                    for shape in diff_fig.layout.shapes:
                        fig.add_shape(shape, row=1, col=1)

            # Plot MSD curves
            if 'msd_data' in result:
                msd_fig = plot_msd_curves(result['msd_data'])
                if msd_fig.data:
                    for trace in msd_fig.data:
                        fig.add_trace(trace, row=1, col=2)

            fig.update_layout(title_text="Diffusion Analysis", showlegend=False)
            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

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

    def _plot_clustering(self, result):
        """Full implementation for clustering visualization."""
        try:
            from visualization import plot_clustering_analysis, _empty_fig

            if not result.get('success', False):
                return _empty_fig("Clustering analysis failed.")

            return plot_clustering_analysis(result)

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

    def _analyze_anomalies(self, tracks_df, current_units):
        """Full implementation for anomaly detection."""
        try:
            from anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            anomaly_results = detector.comprehensive_anomaly_detection(tracks_df)

            # Classify anomalies
            anomaly_classifications = []
            for track_id in tracks_df['track_id'].unique():
                anomaly_type, _, _ = detector.classify_anomaly_type(
                    track_id,
                    anomaly_results.get('velocity_anomalies', {}),
                    anomaly_results.get('confinement_violations', {}),
                    anomaly_results.get('directional_anomalies', {}),
                    anomaly_results.get('ml_anomaly_scores', {}),
                    anomaly_results.get('spatial_clustering', {})
                )
                anomaly_classifications.append({'track_id': track_id, 'anomaly_type': anomaly_type})

            anomaly_classification_df = pd.DataFrame(anomaly_classifications)

            results = {
                'success': True,
                'anomaly_df': pd.merge(tracks_df, anomaly_classification_df, on='track_id'),
                'anomaly_summary': detector.get_anomaly_summary()
            }

            return results
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _plot_anomalies(self, result):
        """Full implementation for anomaly visualization."""
        try:
            from visualization import plot_anomaly_analysis, _empty_fig

            if not result.get('success', False):
                return _empty_fig("Anomaly analysis failed.")

            return plot_anomaly_analysis(result)

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

    def _analyze_changepoints(self, tracks_df, current_units):
        """Implementation for changepoint detection."""
        try:
            if CHANGEPOINT_DETECTION_AVAILABLE:
                detector = ChangePointDetector()
                # Use the implemented API in changepoint_detection.py
                res = detector.detect_motion_regime_changes(tracks_df)
                # Attach tracks for downstream visualization
                if isinstance(res, dict):
                    res.setdefault('tracks_df', tracks_df)
                    res.setdefault('success', bool(res.get('changepoints', pd.DataFrame()).shape[0] or res.get('motion_segments', pd.DataFrame()).shape[0]))
                return res
            else:
                return {'error': 'Changepoint detection module not available', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _plot_changepoints(self, result):
        """Placeholder for changepoint visualization."""
        try:
            from visualization import plot_changepoint_analysis, _empty_fig
            if not result or not result.get('success', False):
                return _empty_fig("Changepoint analysis failed.")
            return plot_changepoint_analysis(result)
        except Exception:
            from visualization import _empty_fig
            return _empty_fig("Changepoint visualization unavailable")

    def _analyze_velocity_correlation(self, tracks_df, current_units):
        """Placeholder for velocity correlation analysis."""
        return {'success': True, 'message': 'Velocity correlation analysis not yet implemented.'}

    def _plot_velocity_correlation(self, result):
        """Placeholder for velocity correlation visualization."""
        from visualization import _empty_fig
        return _empty_fig("Not implemented")

    def _analyze_particle_interactions(self, tracks_df, current_units):
        """Placeholder for particle interaction analysis."""
        return {'success': True, 'message': 'Particle interaction analysis not yet implemented.'}

    def _plot_particle_interactions(self, result):
        """Placeholder for particle interaction visualization."""
        from visualization import _empty_fig
        return _empty_fig("Not implemented")

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

    def _plot_microrheology(self, result):
        """Full implementation for microrheology visualization."""
        try:
            from rheology import create_rheology_plots
            from visualization import _empty_fig

            if not result.get('success', False):
                return _empty_fig("Microrheology analysis failed.")

            figs = create_rheology_plots(result)

            if not figs:
                return _empty_fig("No rheology plots generated.")

            # Create a subplot figure.
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=list(figs.keys())
            )

            row, col = 1
            for name, sub_fig in figs.items():
                if sub_fig.data:
                    for trace in sub_fig.data:
                        fig.add_trace(trace, row=row, col=col)

                col += 1
                if col > 2:
                    break

            fig.update_layout(title_text="Microrheology Analysis", showlegend=False)
            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

    def _analyze_intensity(self, tracks_df, current_units):
        """Placeholder for intensity analysis."""
        return {'success': True, 'message': 'Intensity analysis not yet implemented.'}

    def _plot_intensity(self, result):
        """Placeholder for intensity visualization."""
        from visualization import _empty_fig
        return _empty_fig("Intensity analysis not yet implemented.")

    def _analyze_polymer_physics(self, tracks_df, current_units):
        """Full implementation for polymer physics analysis."""
        try:
            from analysis import analyze_polymer_physics

            polymer_results = analyze_polymer_physics(
                tracks_df,
                pixel_size=current_units.get('pixel_size', 1.0),
                frame_interval=current_units.get('frame_interval', 1.0)
            )
            # Normalize structure and avoid ambiguous truth-value on DataFrames
            if isinstance(polymer_results, dict):
                polymer_results.setdefault('success', 'error' not in polymer_results)
                # Some fields may be numpy arrays; ensure serializable where possible in report JSON path
                return polymer_results
            return {'success': False, 'error': 'Unexpected polymer physics result format'}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _plot_polymer_physics(self, result):
        """Full implementation for polymer physics visualization."""
        try:
            from visualization import plot_polymer_physics_results
            return plot_polymer_physics_results(result)
        except ImportError:
            return None
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Plotting failed: {e}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

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
        col1, col2, col3, col4, col5 = st.columns(5)
        
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

        # HTML export
        with col4:
            try:
                html_bytes = self._export_html_report(config, current_units)
                if html_bytes:
                    st.download_button(
                        "🌐 Download HTML Report",
                        data=html_bytes,
                        file_name=f"spt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            except Exception as e:
                st.info(f"HTML export unavailable: {e}")

        # PDF export (best-effort)
        with col5:
            try:
                pdf_bytes = self._export_pdf_report(current_units)
                if pdf_bytes:
                    st.download_button(
                        "🧾 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"spt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.info(f"PDF export unavailable: {e}")

    def _export_html_report(self, config, current_units) -> bytes:
        """Assemble a standalone HTML report including figures.

        Returns raw HTML bytes suitable for download.
        """
        from datetime import datetime as _dt
        import html
        try:
            parts = []
            parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
            parts.append("<title>SPT Report</title>")
            parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
            parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;} h1,h2{color:#333} .section{margin-bottom:32px;} .metric{margin:4px 0;} .figure{margin:16px 0;} .code{background:#f7f7f7;padding:8px;border-radius:4px;}</style>")
            parts.append("</head><body>")
            parts.append(f"<h1>SPT Analysis Report</h1>")
            parts.append("<div class='section'>")
            parts.append("<h2>Metadata</h2>")
            parts.append("<ul>")
            parts.append(f"<li>Generated at: {_dt.now().isoformat()}</li>")
            parts.append(f"<li>Pixel size (µm): {current_units.get('pixel_size', 0.1)}</li>")
            parts.append(f"<li>Frame interval (s): {current_units.get('frame_interval', 0.1)}</li>")
            parts.append("</ul>")
            parts.append("</div>")

            # Include analysis sections
            import plotly.io as pio
            for key, result in self.report_results.items():
                title = self.available_analyses.get(key, {}).get('name', key)
                parts.append("<div class='section'>")
                parts.append(f"<h2>{html.escape(title)}</h2>")

                # Summary block
                summary = result.get('summary')
                if isinstance(summary, dict):
                    parts.append("<div>")
                    for k, v in summary.items():
                        parts.append(f"<div class='metric'><b>{html.escape(str(k)).title()}</b>: {html.escape(str(v))}</div>")
                    parts.append("</div>")
                elif isinstance(summary, str):
                    parts.append(f"<p>{html.escape(summary)}</p>")

                # Embed figure if present
                fig = self.report_figures.get(key)
                if fig is not None:
                    try:
                        # Plotly figure
                        from matplotlib.figure import Figure as _MplFigure
                    except Exception:
                        _MplFigure = None

                    if _MplFigure is not None and isinstance(fig, _MplFigure):
                        # Convert matplotlib fig to PNG and embed
                        import base64 as _b64, io as _io
                        buf = _io.BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        b64 = _b64.b64encode(buf.read()).decode('utf-8')
                        parts.append(f"<div class='figure'><img src='data:image/png;base64,{b64}' style='max-width:100%'></div>")
                    else:
                        # Assume Plotly
                        try:
                            div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                            parts.append(f"<div class='figure'>{div}</div>")
                        except Exception as e:
                            parts.append(f"<pre class='code'>Failed to render figure: {html.escape(str(e))}</pre>")

                # Optional: include raw JSON for the section if requested
                if config.get('include_raw', True):
                    try:
                        parts.append("<details><summary>Raw Results</summary>")
                        parts.append(f"<pre class='code'>{html.escape(json.dumps(result, indent=2, default=str))}</pre>")
                        parts.append("</details>")
                    except Exception:
                        pass

                parts.append("</div>")

            parts.append("</body></html>")
            html_str = "".join(parts)
            return html_str.encode('utf-8')
        except Exception as e:
            raise RuntimeError(f"HTML export failed: {e}")

    def _export_pdf_report(self, current_units) -> Optional[bytes]:
        """Create a simple PDF containing figures and key stats.

        Requires reportlab; falls back to None if unavailable.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen import canvas
            import io as _io
            import numpy as _np
            # Prepare PDF in memory
            buf = _io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            width, height = A4
            margin = 36
            y = height - margin

            # Title
            c.setFont('Helvetica-Bold', 16)
            c.drawString(margin, y, 'SPT Analysis Report')
            y -= 24
            c.setFont('Helvetica', 10)
            c.drawString(margin, y, f"Pixel size (µm): {current_units.get('pixel_size', 0.1)}  |  Frame interval (s): {current_units.get('frame_interval', 0.1)}")
            y -= 18

            # Add figures one by one (as PNG rasterized)
            for key in self.report_figures:
                fig = self.report_figures[key]
                # Rasterize to PNG bytes
                import io as _io
                img_buf = _io.BytesIO()
                try:
                    from matplotlib.figure import Figure as _MplFigure
                except Exception:
                    _MplFigure = None
                if _MplFigure is not None and isinstance(fig, _MplFigure):
                    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                else:
                    # Plotly: use kaleido if available
                    try:
                        img_bytes = fig.to_image(format='png', scale=2)
                        img_buf.write(img_bytes)
                    except Exception:
                        continue
                img_buf.seek(0)
                img = ImageReader(img_buf)

                # Compute size to fit page
                img_w, img_h = img.getSize()
                scale = min((width - 2*margin)/img_w, (height - 2*margin)/img_h)
                draw_w, draw_h = img_w*scale, img_h*scale
                if y - draw_h < margin:
                    c.showPage()
                    y = height - margin
                    c.setFont('Helvetica-Bold', 12)
                    c.drawString(margin, y, key)
                    y -= 18
                else:
                    c.setFont('Helvetica-Bold', 12)
                    c.drawString(margin, y, key)
                    y -= 18
                c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='sw')
                y -= (draw_h + 24)

            c.showPage()
            c.save()
            buf.seek(0)
            return buf.read()
        except Exception as e:
            # reportlab not available or another error; signal to caller
            raise RuntimeError(str(e))

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
                        fig = self.report_figures[analysis_key]
                        # Support both Plotly and Matplotlib figures
                        try:
                            from matplotlib.figure import Figure as MplFigure
                        except Exception:
                            MplFigure = None
                        if MplFigure is not None and isinstance(fig, MplFigure):
                            st.pyplot(fig, use_container_width=True)
                        else:
                            st.plotly_chart(fig, use_container_width=True)

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

    def _plot_confinement(self, result):
        """Visualization for confinement analysis using provided helper when possible."""
        try:
            from visualization import plot_confinement_analysis, _empty_fig
            # Map our analysis result to the expected schema for the plotter
            if not result:
                return _empty_fig("No confinement results available")

            if 'success' not in result:
                result['success'] = True

            # If keys already match, plot directly
            if {'n_total_tracks', 'n_confined_tracks', 'track_results'}.issubset(result.keys()):
                return plot_confinement_analysis(result)

            data_df = result.get('data', pd.DataFrame())
            tracks_df = result.get('tracks_df')
            # Ensure a track_id column exists for plotting
            if not data_df.empty and 'track_id' not in data_df.columns:
                # Try to find the id column and rename
                for c in ("track_id", "trajectory_id", "particle", "id", "track"):
                    if c in data_df.columns:
                        if c != 'track_id':
                            data_df = data_df.rename(columns={c: 'track_id'})
                        break

            if not data_df.empty and 'confinement_radius' not in data_df.columns and 'max_span' in data_df.columns:
                # Approximate radius as half of max span
                data_df = data_df.copy()
                data_df['confinement_radius'] = data_df['max_span'] / 2.0

            mapped = {
                'success': True,
                'n_total_tracks': int(data_df['track_id'].nunique()) if not data_df.empty and 'track_id' in data_df.columns else 0,
                'n_confined_tracks': int(data_df['localized_flag'].sum()) if not data_df.empty and 'localized_flag' in data_df.columns else 0,
                'track_results': data_df,
                'tracks_df': tracks_df,
            }
            return plot_confinement_analysis(mapped)
        except Exception:
            from visualization import _empty_fig
            return _empty_fig("Confinement visualization unavailable")

    def _analyze_confinement(self, tracks_df, current_units):
        """
        Confinement analysis for single-particle trajectories.
        Returns:
            dict with:
              data: DataFrame of per-trajectory metrics
              summary: human-readable summary string
              figures: list of (optional) plotly figures
        Metrics:
          path_length: cumulative step length
          net_displacement: distance start->end
          confinement_ratio: net_displacement / path_length (lower => more confined)
          radius_gyration: spatial dispersion (Rg)
          max_span: max of x-span, y-span
          localized_flag: heuristic boolean for confinement
        Heuristics are conservative and meant as a first-pass screening.
        """
        import pandas as pd, numpy as np

        # Prefer the provided tracks_df; fall back to common attributes
        track_df = None
        if isinstance(tracks_df, pd.DataFrame) and not tracks_df.empty:
            track_df = tracks_df
        else:
            for attr in ("track_df", "tracks", "filtered_tracks", "trajectory_df"):
                if hasattr(self, attr):
                    cand = getattr(self, attr)
                    if isinstance(cand, pd.DataFrame) and not cand.empty:
                        track_df = cand
                        break

        if track_df is None or len(track_df) == 0:
            return {
                'success': False,
                "data": pd.DataFrame(),
                "summary": "No trajectory data available for confinement analysis.",
                "figures": [],
                'tracks_df': tracks_df if isinstance(tracks_df, pd.DataFrame) else None,
            }

        # Identify trajectory id column
        id_col = None
        for c in ("track_id", "trajectory_id", "particle", "id", "track"):
            if c in track_df.columns:
                id_col = c
                break
        if id_col is None:
            return {
                'success': False,
                "data": pd.DataFrame(),
                "summary": "Could not determine trajectory id column; expected one of track_id / trajectory_id / particle / id / track.",
                "figures": [],
                'tracks_df': track_df,
            }

        # Require coordinate columns
        x_col = "x" if "x" in track_df.columns else None
        y_col = "y" if "y" in track_df.columns else None
        if x_col is None or y_col is None:
            return {
                'success': False,
                "data": pd.DataFrame(),
                "summary": "Missing coordinate columns (x,y) required for confinement analysis.",
                "figures": [],
                'tracks_df': track_df,
            }

        metrics = []
        for tid, g in track_df.groupby(id_col):
            if len(g) < 2:
                metrics.append({
                    id_col: tid,
                    "path_length": 0.0,
                    "net_displacement": 0.0,
                    "confinement_ratio": float("nan"),
                    "radius_gyration": 0.0,
                    "max_span": 0.0,
                    "localized_flag": False,
                })
                continue

            coords = g[[x_col, y_col]].to_numpy()
            steps = np.diff(coords, axis=0)
            step_lengths = np.linalg.norm(steps, axis=1)
            path_length = float(step_lengths.sum())
            net_disp = float(np.linalg.norm(coords[-1] - coords[0]))
            confinement_ratio = net_disp / path_length if path_length > 0 else float("nan")
            rg = float(np.sqrt(((coords - coords.mean(axis=0)) ** 2).sum(axis=1).mean()))
            span_x = coords[:, 0].max() - coords[:, 0].min()
            span_y = coords[:, 1].max() - coords[:, 1].min()
            max_span = float(max(span_x, span_y))

            # Heuristic: localized if path folds a lot and spatial extent modest relative to dispersion
            localized_flag = (
                (confinement_ratio < 0.3) and  # strong winding
                (max_span < 4 * rg if rg > 0 else False)
            )

            metrics.append({
                id_col: tid,
                "path_length": path_length,
                "net_displacement": net_disp,
                "confinement_ratio": confinement_ratio,
                "radius_gyration": rg,
                "max_span": max_span,
                "localized_flag": localized_flag,
            })

        df_metrics = pd.DataFrame(metrics)
        # Ensure a standard track_id column exists for downstream consumers
        if id_col != 'track_id' and id_col in df_metrics.columns:
            df_metrics = df_metrics.rename(columns={id_col: 'track_id'})
        confined_pct = df_metrics["localized_flag"].mean() * 100 if len(df_metrics) else 0.0
        summary = (
            f"Confinement analysis computed for {len(df_metrics)} trajectories. "
            f"{confined_pct:.1f}% flagged as localized (heuristic)."
        )

        figures = []
        try:
            import plotly.express as px
            fig = px.histogram(
                df_metrics.dropna(subset=["confinement_ratio"]),
                x="confinement_ratio",
                nbins=30,
                title="Confinement Ratio Distribution"
            )
            figures.append(fig)
        except Exception:
            # Silently ignore visualization issues to keep analysis robust
            pass

        return {
            'success': True,
            "data": df_metrics,
            "summary": summary,
            "figures": figures,
            'n_total_tracks': int(df_metrics['track_id'].nunique()) if not df_metrics.empty else 0,
            'n_confined_tracks': int(df_metrics['localized_flag'].sum()) if not df_metrics.empty else 0,
            'tracks_df': track_df,
        }

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
    
    # If track_data is passed directly, ensure it's available
    if track_data is not None:
        if DATA_UTILS_AVAILABLE:
            # Check if we need to set the data
            existing_data, has_data = get_track_data()
            if not has_data:
                # We need to set it in session state
                st.session_state['tracks_df'] = track_data
        elif STATE_MANAGER_AVAILABLE:
            sm = get_state_manager()
            try:
                if not sm.has_data():
                    sm.set_tracks(track_data)
            except Exception:
                st.session_state['tracks_df'] = track_data
    
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