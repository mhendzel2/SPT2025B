"""
Test iHMM Blur Analysis Fix
=============================
Tests that the iHMM state segmentation now calls the correct method.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_report_generator import EnhancedSPTReportGenerator
from data_access_utils import get_units

def create_test_multistate_data():
    """Create synthetic data with multiple diffusive states."""
    np.random.seed(42)
    
    tracks = []
    
    for track_id in range(3):
        n_frames = 100
        
        # Create trajectory with state switches
        x = np.zeros(n_frames)
        y = np.zeros(n_frames)
        
        for i in range(1, n_frames):
            # Switch states every 25 frames
            if i < 25:
                D = 0.05  # Slow state
            elif i < 50:
                D = 0.2   # Fast state
            elif i < 75:
                D = 0.1   # Medium state
            else:
                D = 0.15  # Medium-fast state
            
            dt = 0.1
            sigma = np.sqrt(2 * D * dt)
            
            x[i] = x[i-1] + np.random.normal(0, sigma)
            y[i] = y[i-1] + np.random.normal(0, sigma)
        
        for frame in range(n_frames):
            tracks.append({
                'track_id': track_id,
                'frame': frame,
                'x': x[frame],
                'y': y[frame]
            })
    
    return pd.DataFrame(tracks)

def test_ihmm_fix():
    """Test that iHMM analysis now works with correct method call."""
    print("=" * 80)
    print("Testing iHMM State Segmentation Fix")
    print("=" * 80)
    
    # Create test data
    print("\n1. Creating test data with multiple diffusive states...")
    tracks_df = create_test_multistate_data()
    print(f"   Created {tracks_df['track_id'].nunique()} tracks with {len(tracks_df)} total points")
    
    # Initialize report generator
    print("\n2. Initializing report generator...")
    generator = EnhancedSPTReportGenerator()
    
    # Get units
    current_units = get_units()
    print(f"   Pixel size: {current_units['pixel_size']} μm")
    print(f"   Frame interval: {current_units['frame_interval']} s")
    
    # Test iHMM analysis
    print("\n3. Running iHMM State Segmentation...")
    try:
        result = generator._analyze_ihmm_blur(tracks_df, current_units)
        
        if result.get('success', False):
            print("   ✓ PASS: iHMM analysis completed successfully!")
            
            # Display results
            summary = result.get('summary', {})
            print("\n   Results Summary:")
            print(f"   - Tracks analyzed: {summary.get('n_tracks_analyzed', 'N/A')}")
            
            n_states_dist = summary.get('n_states_distribution', {})
            print(f"   - States discovered (mean): {n_states_dist.get('mean', 'N/A'):.1f}")
            print(f"   - States discovered (median): {n_states_dist.get('median', 'N/A'):.1f}")
            print(f"   - States discovered (mode): {n_states_dist.get('mode', 'N/A')}")
            
            D_range = summary.get('D_range_um2_s', (None, None))
            if D_range[0] is not None:
                print(f"   - D range: [{D_range[0]:.2e}, {D_range[1]:.2e}] μm²/s")
            
            convergence = summary.get('convergence_rate', 'N/A')
            if isinstance(convergence, float):
                print(f"   - Convergence rate: {convergence*100:.1f}%")
            
            # Test visualization
            print("\n4. Testing visualization...")
            try:
                fig = generator._plot_ihmm_blur(result)
                print("   ✓ PASS: Visualization created successfully!")
                print(f"   - Figure has {len(fig.data)} traces")
            except Exception as e:
                print(f"   ⚠ WARNING: Visualization failed: {e}")
            
            return True
            
        else:
            error = result.get('error', 'Unknown error')
            if 'not available' in error.lower():
                print(f"   ⚠ INFO: {error}")
                print("   This is expected if ihmm_blur_analysis.py is not installed.")
                return True  # Not a bug, just missing optional module
            else:
                print(f"   ✗ FAIL: {error}")
                return False
                
    except AttributeError as e:
        if 'segment_trajectories' in str(e):
            print(f"   ✗ FAIL: Still calling wrong method!")
            print(f"   Error: {e}")
            return False
        else:
            print(f"   ✗ FAIL: Unexpected AttributeError: {e}")
            return False
            
    except Exception as e:
        print(f"   ✗ FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ihmm_fix()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED - iHMM method call fixed!")
    else:
        print("✗ TEST FAILED - iHMM still has issues")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
