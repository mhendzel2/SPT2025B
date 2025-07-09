#!/usr/bin/env python3
"""
Test script to verify that the AnalysisManager properly handles channel correlation
and multi-channel analysis with secondary channel data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create test track data for primary and secondary channels."""
    # Create primary channel data
    np.random.seed(42)
    n_tracks = 10
    n_points_per_track = 50
    
    primary_data = []
    for track_id in range(n_tracks):
        x = np.cumsum(np.random.randn(n_points_per_track) * 0.1) + np.random.randn() * 10
        y = np.cumsum(np.random.randn(n_points_per_track) * 0.1) + np.random.randn() * 10
        intensity = np.random.randn(n_points_per_track) * 50 + 200
        
        for i in range(n_points_per_track):
            primary_data.append({
                'track_id': track_id,
                'frame': i,
                'x': x[i],
                'y': y[i],
                'intensity': intensity[i],
                'ch1': intensity[i],
                'Mean_ch1': intensity[i] * 0.8
            })
    
    # Create secondary channel data (partially overlapping)
    secondary_data = []
    for track_id in range(n_tracks // 2):  # Fewer tracks in secondary channel
        x = np.cumsum(np.random.randn(n_points_per_track) * 0.1) + np.random.randn() * 10
        y = np.cumsum(np.random.randn(n_points_per_track) * 0.1) + np.random.randn() * 10
        intensity = np.random.randn(n_points_per_track) * 30 + 150
        
        for i in range(n_points_per_track):
            secondary_data.append({
                'track_id': track_id + 100,  # Different track IDs
                'frame': i,
                'x': x[i] + np.random.randn() * 0.5,  # Slightly offset positions
                'y': y[i] + np.random.randn() * 0.5,
                'intensity': intensity[i],
                'ch2': intensity[i],
                'Mean_ch2': intensity[i] * 0.9
            })
    
    return pd.DataFrame(primary_data), pd.DataFrame(secondary_data)

def test_analysis_manager():
    """Test the AnalysisManager with channel correlation data."""
    print("Testing AnalysisManager with channel correlation...")
    
    # Create test data
    primary_df, secondary_df = create_test_data()
    
    print(f"Primary channel data: {len(primary_df)} points, {primary_df['track_id'].nunique()} tracks")
    print(f"Secondary channel data: {len(secondary_df)} points, {secondary_df['track_id'].nunique()} tracks")
    
    # Import necessary modules
    try:
        from analysis_manager import AnalysisManager
        from state_manager import get_state_manager
        import streamlit as st
        
        # Initialize session state manually for testing
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __setitem__(self, key, value):
                self.data[key] = value
            
            def __contains__(self, key):
                return key in self.data
        
        # Mock streamlit session state
        if not hasattr(st, 'session_state'):
            st.session_state = MockSessionState()
        
        # Initialize managers
        state_manager = get_state_manager()
        analysis_manager = AnalysisManager()
        
        # Set up data
        state_manager.set_raw_tracks(primary_df)
        state_manager.set_secondary_channel_data(secondary_df, "Secondary Channel")
        
        # Test analysis types availability
        print("\nAvailable analysis types:")
        for analysis_type, config in analysis_manager.available_analyses.items():
            print(f"  - {analysis_type}: {config['name']}")
        
        # Test feasibility checking
        print("\nTesting analysis feasibility:")
        for analysis_type in ['correlative', 'multi_channel', 'channel_correlation']:
            feasibility = analysis_manager.check_analysis_feasibility(analysis_type)
            print(f"  {analysis_type}: {'✓' if feasibility['feasible'] else '✗'}")
            if not feasibility['feasible']:
                print(f"    Missing: {', '.join(feasibility['missing_requirements'])}")
        
        # Test correlative analysis
        print("\nTesting correlative analysis...")
        correlative_result = analysis_manager.run_correlative_analysis({
            'intensity_columns': ['ch1', 'Mean_ch1'],
            'lag_range': 3
        })
        print(f"  Correlative analysis result: {'✓' if correlative_result.get('success', False) else '✗'}")
        if not correlative_result.get('success', False):
            print(f"    Error: {correlative_result.get('error', 'Unknown error')}")
        
        # Test multi-channel analysis
        print("\nTesting multi-channel analysis...")
        multi_channel_result = analysis_manager.run_multi_channel_analysis({
            'distance_threshold': 1.0,
            'frame_tolerance': 1
        })
        print(f"  Multi-channel analysis result: {'✓' if multi_channel_result.get('success', False) else '✗'}")
        if not multi_channel_result.get('success', False):
            print(f"    Error: {multi_channel_result.get('error', 'Unknown error')}")
        
        # Test channel correlation analysis
        print("\nTesting channel correlation analysis...")
        channel_corr_result = analysis_manager.run_channel_correlation_analysis({
            'max_lag': 10,
            'primary_channel_name': 'Primary',
            'secondary_channel_name': 'Secondary'
        })
        print(f"  Channel correlation analysis result: {'✓' if channel_corr_result.get('success', False) else '✗'}")
        if not channel_corr_result.get('success', False):
            print(f"    Error: {channel_corr_result.get('error', 'Unknown error')}")
        
        # Test analysis pipeline
        print("\nTesting analysis pipeline...")
        pipeline_result = analysis_manager.execute_analysis_pipeline(
            ['correlative', 'multi_channel', 'channel_correlation'],
            {
                'correlative': {'intensity_columns': ['ch1'], 'lag_range': 3},
                'multi_channel': {'distance_threshold': 1.0},
                'channel_correlation': {'max_lag': 10}
            }
        )
        
        print(f"  Pipeline result: {'✓' if pipeline_result.get('success', False) else '✗'}")
        print(f"  Successful analyses: {pipeline_result.get('summary', {}).get('successful', 0)}")
        print(f"  Failed analyses: {pipeline_result.get('summary', {}).get('failed', 0)}")
        
        if pipeline_result.get('errors'):
            print("  Errors:")
            for error in pipeline_result['errors']:
                print(f"    - {error}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analysis_manager()
    sys.exit(0 if success else 1)
