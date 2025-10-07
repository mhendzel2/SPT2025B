"""
Test all fixed visualization methods to ensure they return single go.Figure objects.

This test validates that all previously list-returning methods now correctly
return single plotly Figure objects with proper subplot structures.
"""

import sys
import pandas as pd
import numpy as np
from enhanced_report_generator import EnhancedSPTReportGenerator

def create_test_tracks(n_tracks=10, n_frames=50):
    """Create synthetic track data for testing."""
    tracks = []
    for track_id in range(n_tracks):
        frames = np.arange(n_frames)
        x = np.cumsum(np.random.randn(n_frames) * 0.1)
        y = np.cumsum(np.random.randn(n_frames) * 0.1)
        
        track_data = pd.DataFrame({
            'track_id': track_id,
            'frame': frames,
            'x': x,
            'y': y
        })
        tracks.append(track_data)
    
    return pd.concat(tracks, ignore_index=True)

def test_plot_methods():
    """Test all fixed plot methods."""
    print("=" * 80)
    print("TESTING ALL FIXED VISUALIZATION METHODS")
    print("=" * 80)
    print()
    
    # Create test data
    print("Creating test track data...")
    tracks_df = create_test_tracks(n_tracks=20, n_frames=100)
    print(f"  Created {len(tracks_df['track_id'].unique())} tracks with {len(tracks_df)} total points")
    print()
    
    # Initialize report generator
    print("Initializing report generator...")
    generator = EnhancedSPTReportGenerator()
    print("  ✓ Generator initialized")
    print()
    
    tests_run = 0
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: _plot_crowding
    print("1. Testing _plot_crowding...")
    try:
        result = {
            'success': True,
            'corrections': [
                {'phi': 0.2, 'D_free': 0.15},
                {'phi': 0.3, 'D_free': 0.18},
                {'phi': 0.4, 'D_free': 0.22}
            ]
        }
        fig = generator._plot_crowding(result)
        tests_run += 1
        
        if fig is None:
            print("  ✗ FAIL: Returned None")
            tests_failed += 1
        elif isinstance(fig, list):
            print(f"  ✗ FAIL: Returned list with {len(fig)} items (should be single figure)")
            tests_failed += 1
        else:
            print(f"  ✓ PASS: Returns single Figure object")
            print(f"    - Type: {type(fig).__name__}")
            print(f"    - Number of traces: {len(fig.data)}")
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        tests_run += 1
        tests_failed += 1
    print()
    
    # Test 2: _plot_local_diffusion_map
    print("2. Testing _plot_local_diffusion_map...")
    try:
        D_map = np.random.rand(10, 10) * 0.5
        result = {
            'success': True,
            'full_results': {
                'D_map': D_map,
                'x_edges': np.linspace(0, 10, 11),
                'y_edges': np.linspace(0, 10, 11)
            }
        }
        fig = generator._plot_local_diffusion_map(result)
        tests_run += 1
        
        if fig is None:
            print("  ✗ FAIL: Returned None")
            tests_failed += 1
        elif isinstance(fig, list):
            print(f"  ✗ FAIL: Returned list with {len(fig)} items (should be single figure)")
            tests_failed += 1
        else:
            print(f"  ✓ PASS: Returns single Figure object")
            print(f"    - Type: {type(fig).__name__}")
            print(f"    - Number of traces: {len(fig.data)}")
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        tests_run += 1
        tests_failed += 1
    print()
    
    # Test 3: _plot_ctrw
    print("3. Testing _plot_ctrw...")
    try:
        result = {
            'success': True,
            'waiting_times': {'wait_times': np.random.exponential(0.1, 1000)},
            'jump_lengths': {'jump_lengths': np.random.exponential(0.5, 1000)}
        }
        fig = generator._plot_ctrw(result)
        tests_run += 1
        
        if fig is None:
            print("  ✗ FAIL: Returned None")
            tests_failed += 1
        elif isinstance(fig, list):
            print(f"  ✗ FAIL: Returned list with {len(fig)} items (should be single figure)")
            tests_failed += 1
        else:
            print(f"  ✓ PASS: Returns single Figure object")
            print(f"    - Type: {type(fig).__name__}")
            print(f"    - Number of traces: {len(fig.data)}")
            print(f"    - Has subplots: {hasattr(fig, 'layout') and 'xaxis2' in fig.layout}")
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        tests_run += 1
        tests_failed += 1
    print()
    
    # Test 4: _plot_fbm_enhanced
    print("4. Testing _plot_fbm_enhanced...")
    try:
        result = {
            'success': True,
            'n_valid': 20,
            'hurst_mean': 0.45,
            'hurst_std': 0.15,
            'D_mean': 0.0002,
            'D_std': 0.0001,
            'hurst_values': np.random.normal(0.45, 0.15, 20),
            'D_values': np.random.exponential(0.0002, 20)
        }
        fig = generator._plot_fbm_enhanced(result)
        tests_run += 1
        
        if fig is None:
            print("  ✗ FAIL: Returned None")
            tests_failed += 1
        elif isinstance(fig, list):
            print(f"  ✗ FAIL: Returned list with {len(fig)} items (should be single figure)")
            tests_failed += 1
        else:
            print(f"  ✓ PASS: Returns single Figure object")
            print(f"    - Type: {type(fig).__name__}")
            print(f"    - Number of traces: {len(fig.data)}")
            print(f"    - Has subplots: {hasattr(fig, 'layout') and 'xaxis2' in fig.layout}")
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        tests_run += 1
        tests_failed += 1
    print()
    
    # Test 5: _plot_track_quality
    print("5. Testing _plot_track_quality...")
    try:
        # Create realistic quality score data
        n_tracks = 20
        quality_df = pd.DataFrame({
            'track_id': range(n_tracks),
            'quality_score': np.random.uniform(0.3, 0.9, n_tracks),
            'length_score': np.random.uniform(0.4, 1.0, n_tracks),
            'completeness_score': np.random.uniform(0.6, 1.0, n_tracks),
            'snr_score': np.random.uniform(0.3, 0.8, n_tracks),
            'precision_score': np.random.uniform(0.4, 0.9, n_tracks),
            'smoothness_score': np.random.uniform(0.2, 0.7, n_tracks)
        })
        
        comp_df = pd.DataFrame({
            'track_id': range(n_tracks),
            'track_length': np.random.randint(30, 150, n_tracks),
            'completeness': np.random.uniform(0.6, 1.0, n_tracks)
        })
        
        result = {
            'success': True,
            'n_tracks': n_tracks,
            'quality_scores': quality_df,
            'completeness': comp_df,
            'summary': {
                'track_length': {'mean': 85.5},
                'completeness': {'mean': 0.82, 'tracks_above_70pct': 18},
                'quality_score': {'mean': 0.65, 'tracks_above_0.7': 12},
                'snr': {'mean': 2.5}
            }
        }
        
        fig = generator._plot_track_quality(result)
        tests_run += 1
        
        if fig is None:
            print("  ✗ FAIL: Returned None")
            tests_failed += 1
        elif isinstance(fig, list):
            print(f"  ✗ FAIL: Returned list with {len(fig)} items (should be single figure)")
            tests_failed += 1
        else:
            print(f"  ✓ PASS: Returns single Figure object")
            print(f"    - Type: {type(fig).__name__}")
            print(f"    - Number of traces: {len(fig.data)}")
            print(f"    - Has subplots: {hasattr(fig, 'layout') and 'xaxis2' in fig.layout}")
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        tests_run += 1
        tests_failed += 1
    print()
    
    # Final summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {tests_run}")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print()
    
    if tests_failed == 0:
        print("✓ ALL TESTS PASSED - All methods return single Figure objects!")
        return True
    else:
        print(f"✗ {tests_failed} TESTS FAILED - Some methods still return lists")
        return False

if __name__ == "__main__":
    success = test_plot_methods()
    sys.exit(0 if success else 1)
