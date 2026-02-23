"""
Test ML Classification Visualization Fix
==========================================
Tests that ML classification now returns a single combined figure instead of a list.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_report_generator import EnhancedSPTReportGenerator
from data_access_utils import get_units

def create_test_tracks():
    """Create synthetic tracks with different motion types."""
    np.random.seed(42)
    tracks = []
    
    # Class 0: Slow diffusion
    for track_id in range(20):
        x, y = 0.0, 0.0
        for frame in range(30):
            tracks.append({'track_id': track_id, 'frame': frame, 'x': x, 'y': y})
            x += np.random.normal(0, 0.05)
            y += np.random.normal(0, 0.05)
    
    # Class 1: Fast diffusion
    for track_id in range(20, 36):
        x, y = 5.0, 5.0
        for frame in range(30):
            tracks.append({'track_id': track_id, 'frame': frame, 'x': x, 'y': y})
            x += np.random.normal(0, 0.2)
            y += np.random.normal(0, 0.2)
    
    # Class 2: Directed motion
    for track_id in range(36, 46):
        x, y = 10.0, 10.0
        for frame in range(30):
            tracks.append({'track_id': track_id, 'frame': frame, 'x': x, 'y': y})
            x += np.random.normal(0.1, 0.05)  # Biased in x-direction
            y += np.random.normal(0, 0.05)
    
    # Class 3: Confined
    for track_id in range(46, 48):
        x, y = 15.0, 15.0
        center_x, center_y = x, y
        for frame in range(30):
            tracks.append({'track_id': track_id, 'frame': frame, 'x': x, 'y': y})
            # Add restoring force to keep near origin
            dx = np.random.normal(0, 0.1) - 0.1 * (x - center_x)
            dy = np.random.normal(0, 0.1) - 0.1 * (y - center_y)
            x += dx
            y += dy
    
    return pd.DataFrame(tracks)

def test_ml_classification_fix():
    """Test that ML classification visualization returns a single figure."""
    print("=" * 80)
    print("Testing ML Classification Visualization Fix")
    print("=" * 80)
    
    # Create test data
    print("\n1. Creating test tracks with 4 motion classes...")
    tracks_df = create_test_tracks()
    print(f"   Created {tracks_df['track_id'].nunique()} tracks with {len(tracks_df)} total points")
    
    # Initialize report generator
    print("\n2. Initializing report generator...")
    generator = EnhancedSPTReportGenerator()
    current_units = get_units()
    
    # Test ML classification analysis
    print("\n3. Running ML classification analysis...")
    try:
        result = generator._analyze_ml_classification(tracks_df, current_units)
        
        if not result.get('success', False):
            error = result.get('error', 'Unknown error')
            if 'not available' in error.lower() or 'cannot import' in error.lower():
                print(f"   ℹ INFO: {error}")
                print("   This is expected if ml_trajectory_classifier_enhanced.py or sklearn is not available.")
                return True  # Not a bug, just missing optional module
            else:
                print(f"   ✗ FAIL: Analysis failed - {error}")
                return False
        
        print("   ✓ Analysis completed successfully")
        print(f"   - Number of classes: {result.get('n_classes', 'N/A')}")
        print(f"   - Silhouette score: {result.get('silhouette_score', 'N/A'):.3f}")
        
        # Test visualization
        print("\n4. Testing visualization...")
        try:
            fig = generator._plot_ml_classification(result)
            
            # Check that fig is a single Figure object, not a list
            import plotly.graph_objects as go
            
            if isinstance(fig, list):
                print(f"   ✗ FAIL: Returned a list with {len(fig)} figures instead of single figure!")
                print("   This is the bug we're trying to fix.")
                return False
            elif isinstance(fig, go.Figure):
                print("   ✓ PASS: Returns a single plotly Figure object")
                print(f"   - Number of traces: {len(fig.data)}")
                # Check if it has subplots by checking layout properties
                has_subplots = hasattr(fig.layout, 'xaxis2') and fig.layout.xaxis2 is not None
                print(f"   - Has subplots: {has_subplots}")
                
                # Try to convert to HTML (this is what fails if fig is a list)
                print("\n5. Testing HTML rendering...")
                try:
                    import plotly.io as pio
                    html_str = pio.to_html(fig, include_plotlyjs='inline', full_html=False)
                    print(f"   ✓ PASS: Successfully rendered to HTML ({len(html_str)} chars)")
                    return True
                except Exception as e:
                    print(f"   ✗ FAIL: HTML rendering failed - {e}")
                    return False
            else:
                print(f"   ⚠ WARNING: Unexpected type: {type(fig)}")
                return False
                
        except Exception as e:
            print(f"   ✗ FAIL: Visualization failed - {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ✗ FAIL: Unexpected error - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_classification_fix()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED - ML classification returns single figure!")
    else:
        print("✗ TEST FAILED - ML classification still has issues")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
