import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_access_utils import get_track_data, get_units
from utils import create_analysis_record, calculate_track_statistics

# Import advanced analysis modules
try:
    from advanced_biophysical import show_advanced_biophysical
    BIOPHYSICAL_AVAILABLE = True
except ImportError:
    BIOPHYSICAL_AVAILABLE = False

try:
    from changepoint_detection import ChangePointDetector
    CHANGEPOINT_AVAILABLE = True
except ImportError:
    CHANGEPOINT_AVAILABLE = False

try:
    from correlative_analysis import CorrelativeAnalyzer
    CORRELATIVE_AVAILABLE = True
except ImportError:
    CORRELATIVE_AVAILABLE = False

try:
    from advanced_tracking import ParticleFilter, AdvancedTracking, bayesian_detection_refinement
    ADVANCED_TRACKING_AVAILABLE = True
except ImportError:
    ADVANCED_TRACKING_AVAILABLE = False

try:
    from intensity_analysis import (
        extract_intensity_channels, correlate_intensity_movement,
        create_intensity_movement_plots, analyze_intensity_profiles,
        classify_intensity_behavior
    )
    INTENSITY_AVAILABLE = True
except ImportError:
    INTENSITY_AVAILABLE = False

try:
    from rheology import MicrorheologyAnalyzer, display_rheology_summary, create_rheology_plots
    MICRORHEOLOGY_AVAILABLE = True
except ImportError:
    MICRORHEOLOGY_AVAILABLE = False

try:
    from advanced_diffusion_models import CTRWAnalyzer, fit_fbm_model
    DIFFUSION_MODELS_AVAILABLE = True
except ImportError:
    DIFFUSION_MODELS_AVAILABLE = False

try:
    from advanced_biophysical_metrics import AdvancedMetricsAnalyzer, MetricConfig
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

try:
    from advanced_statistical_tests import (
        chi_squared_goodness_of_fit, validate_model_fit, bootstrap_confidence_interval
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

try:
    from ornstein_uhlenbeck_analyzer import analyze_ornstein_uhlenbeck
    OU_AVAILABLE = True
except ImportError:
    OU_AVAILABLE = False

try:
    from hmm_analysis import fit_hmm
    HMM_AVAILABLE = True
except (ImportError, RuntimeError):
    HMM_AVAILABLE = False

try:
    from ihmm_analysis import analyze_track_with_ihmm
    IHMM_AVAILABLE = True
except ImportError:
    IHMM_AVAILABLE = False

try:
    from ddm_analyzer import DDMAnalyzer
    DDM_AVAILABLE = True
except ImportError:
    DDM_AVAILABLE = False

def show_advanced_analysis_page():
    st.title("Advanced Analysis")

    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        if st.button("Go to Data Loading"):
            st.session_state.active_page = "Data Loading"
            st.rerun()
    else:
        adv_tabs = st.tabs([
            "Biophysical Models", "Advanced Biophysical", "Changepoint Detection",
            "Correlative Analysis", "Advanced Tracking", "Intensity Analysis",
            "Microrheology", "CTRW Analysis", "FBM Analysis", "Advanced Metrics",
            "Statistical Tests", "Ornstein-Uhlenbeck", "HMM Analysis", "DDM (Tracking-Free)"
        ])

        with adv_tabs[0]:
            st.header("Biophysical Models")
            st.info("Use Advanced Analysis tabs for specific models.")

        with adv_tabs[1]:
            st.header("🔬 Advanced Biophysical Analysis")
            if BIOPHYSICAL_AVAILABLE:
                show_advanced_biophysical()
            else:
                st.error("Module not available.")

        with adv_tabs[2]:
            st.header("Changepoint Detection")
            if CHANGEPOINT_AVAILABLE:
                window_size = st.slider("Window Size", 5, 50, 10)
                min_segment_length = st.slider("Minimum Segment Length", 3, 20, 5)
                if st.button("Run Changepoint Detection"):
                    with st.spinner("Running..."):
                        detector = ChangePointDetector()
                        res = detector.detect_motion_regime_changes(
                            st.session_state.tracks_data, window_size=window_size, min_segment_length=min_segment_length
                        )
                        st.session_state.analysis_results['changepoints'] = res
                        st.success("Done!")
                if 'changepoints' in st.session_state.analysis_results:
                    res = st.session_state.analysis_results['changepoints']
                    if 'changepoints' in res:
                        st.dataframe(res['changepoints'])
            else:
                st.error("Module not available.")

        with adv_tabs[12]:
            st.header("HMM Analysis")
            if HMM_AVAILABLE:
                n_states = st.slider("Number of Hidden States", 2, 10, 3)
                if st.button("Run HMM Analysis"):
                    with st.spinner("Fitting HMM..."):
                        model, predictions = fit_hmm(st.session_state.tracks_data, n_states)
                        st.session_state.analysis_results["hmm"] = {"model": model, "predictions": predictions}
                        st.success("HMM complete!")
                if "hmm" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["hmm"]
                    st.write("Transition Matrix:", res["model"].transmat_)
            else:
                st.warning("HMM module disabled.")

        with adv_tabs[13]:
            st.header("DDM (Tracking-Free)")
            if DDM_AVAILABLE:
                if st.session_state.image_data is None:
                    st.info("Load images for DDM.")
                else:
                    if st.button("Run DDM"):
                        st.success("DDM Placeholder - see ddm_analyzer.py for implementation.")
            else:
                st.warning("DDM module not available.")
