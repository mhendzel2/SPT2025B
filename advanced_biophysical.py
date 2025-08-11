"""
Advanced Biophysical Analysis Module for SPT Data

This module provides advanced biophysical analysis capabilities for SPT data:
- Polymer dynamics analysis
- Viscoelasticity and microrheology
- Active matter and propulsion analysis
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import data access utilities
try:
    from data_access_utils import get_track_data, check_data_availability, get_units, display_data_summary
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

# Import analysis functions with error handling
try:
    from biophysical_analysis import (
        analyze_polymer_dynamics, analyze_viscoelasticity, analyze_active_matter
    )
    ANALYSIS_MODULE_AVAILABLE = True
except ImportError:
    ANALYSIS_MODULE_AVAILABLE = False

def show_advanced_biophysical():
    """Main function for advanced biophysical analysis interface."""
    st.title("🔬 Advanced Biophysical Analysis")
    
    # Check for data availability
    if DATA_UTILS_AVAILABLE:
        if not check_data_availability():
            return
        tracks_df, _ = get_track_data()
        units = get_units()
    else:
        # Fallback to direct access
        tracks_df = st.session_state.get('tracks_df') or st.session_state.get('raw_tracks')
        if tracks_df is None or tracks_df.empty:
            st.error("No track data loaded. Please load data first.")
            st.info("Go to the 'Data Loading' tab to upload track data.")
            return
        units = {
            'pixel_size': st.session_state.get('pixel_size', 0.1),
            'frame_interval': st.session_state.get('frame_interval', 0.1)
        }
    
    # Display data summary
    if DATA_UTILS_AVAILABLE:
        display_data_summary()
    
    # Continue with the rest of the advanced biophysical analysis using tracks_df
    
    # Example: Update any analysis functions that need data
    tab1, tab2, tab3 = st.tabs(["Polymer Dynamics", "Viscoelasticity", "Active Matter"])
    
    with tab1:
        if st.button("Run Polymer Analysis"):
            # Use the tracks_df we got above
            results = analyze_polymer_dynamics(tracks_df, units)
            if results.get('success'):
                st.success("Polymer analysis completed.")
                # Display results (update as needed)
                st.json(results)
            else:
                st.error(f"Polymer analysis failed: {results.get('error', 'Unknown error')}")
    
    with tab2:
        if st.button("Run Viscoelasticity Analysis"):
            results = analyze_viscoelasticity(tracks_df, units)
            if results.get('success'):
                st.success("Viscoelasticity analysis completed.")
                st.json(results)
            else:
                st.error(f"Viscoelasticity analysis failed: {results.get('error', 'Unknown error')}")
    
    with tab3:
        if st.button("Run Active Matter Analysis"):
            results = analyze_active_matter(tracks_df, units)
            if results.get('success'):
                st.success("Active matter analysis completed.")
                st.json(results)
            else:
                st.error(f"Active matter analysis failed: {results.get('error', 'Unknown error')}")

def analyze_polymer_dynamics(tracks_df, units):
    """Analyze polymer dynamics with proper data access."""
    if tracks_df is None or tracks_df.empty:
        return {'error': 'No data available', 'success': False}
    
    try:
        # Example analysis: Calculate mean squared displacement (MSD)
        msd_results = []
        for track_id in tracks_df['track_id'].unique():
            track = tracks_df[tracks_df['track_id'] == track_id]
            if len(track) > 1:
                # Calculate squared displacements
                x_diff = track['x'].diff().fillna(0)
                y_diff = track['y'].diff().fillna(0)
                msd = np.mean(x_diff**2 + y_diff**2)
                msd_results.append(msd)
        
        mean_msd = np.mean(msd_results) if msd_results else 0.0
        
        return {
            'success': True,
            'mean_msd': mean_msd,
            'msd_results': msd_results
        }
    except Exception as e:
        return {'error': str(e), 'success': False}

# For direct execution (optional)
if __name__ == "__main__":
    show_advanced_biophysical()