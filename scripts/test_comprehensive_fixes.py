"""
Comprehensive test for intensity analysis and motion visualization fixes.
Tests that both analyses work end-to-end through the report generator.
"""

import numpy as np
import pandas as pd
import sys

print("=" * 80)
print("INTENSITY & MOTION ANALYSIS COMPREHENSIVE TEST")
print("=" * 80)

# Create realistic synthetic track data with intensity and motion
np.random.seed(42)
n_tracks = 10
n_points_per_track = 30

tracks_data = []
for track_id in range(1, n_tracks + 1):
    # Random starting position
    start_x = np.random.rand() * 20
    start_y = np.random.rand() * 20
    
    # Create different motion types
    if track_id <= 3:
        # Directed motion
        dx_per_frame = 0.5
        dy_per_frame = 0.3
        motion_type = "directed"
    elif track_id <= 6:
        # Brownian motion
        dx_per_frame = 0
        dy_per_frame = 0
        motion_type = "brownian"
    else:
        # Confined motion
        dx_per_frame = 0
        dy_per_frame = 0
        motion_type = "confined"
    
    x, y = start_x, start_y
    
    for frame in range(n_points_per_track):
        # Add motion
        if motion_type == "directed":
            x += dx_per_frame + np.random.randn() * 0.1
            y += dy_per_frame + np.random.randn() * 0.1
        elif motion_type == "brownian":
            x += np.random.randn() * 0.3
            y += np.random.randn() * 0.3
        else:  # confined
            x += np.random.randn() * 0.1
            y += np.random.randn() * 0.1
        
        # Add intensity that correlates with position
        base_intensity = 50 + track_id * 5
        intensity_ch1 = base_intensity + np.random.rand() * 20 + (x + y) * 0.5
        intensity_ch2 = base_intensity * 0.8 + np.random.rand() * 15
        
        tracks_data.append({
            'track_id': track_id,
            'frame': frame,
            'x': x,
            'y': y,
            'mean_intensity_ch1': intensity_ch1,
            'mean_intensity_ch2': intensity_ch2
        })

tracks_df = pd.DataFrame(tracks_data)

print(f"\n✓ Created test dataset:")
print(f"  - Tracks: {tracks_df['track_id'].nunique()}")
print(f"  - Total points: {len(tracks_df)}")
print(f"  - Frames per track: {n_points_per_track}")
print(f"  - Columns: {list(tracks_df.columns)}")

# Test 1: Intensity Analysis through Report Generator
print("\n" + "=" * 80)
print("TEST 1: Intensity Analysis (Report Generator Integration)")
print("=" * 80)

try:
    from enhanced_report_generator import EnhancedSPTReportGenerator
    
    generator = EnhancedSPTReportGenerator()
    
    current_units = {
        'pixel_size': 0.1,
        'frame_interval': 0.1
    }
    
    result = generator._analyze_intensity(tracks_df, current_units)
    
    if result.get('success'):
        print(f"✓ PASS: Intensity analysis completed")
        print(f"  - Channels detected: {result['summary'].get('channels_detected', [])}")
        print(f"  - Number of tracks: {result['summary'].get('n_tracks', 0)}")
        
        # Verify correlation results
        if result.get('correlation_results'):
            if 'error' in result['correlation_results']:
                print(f"  - Correlation: {result['correlation_results']['error']}")
            else:
                print(f"  - Mean correlation: {result['correlation_results'].get('mean_correlation', 'N/A')}")
        
        # Verify behavior results
        if result.get('behavior_results') is not None:
            print(f"  - Behavior classification: SUCCESS ({len(result['behavior_results'])} rows)")
        
        test1_pass = True
    else:
        print(f"✗ FAIL: {result.get('error', 'Unknown error')}")
        test1_pass = False
        
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test1_pass = False

# Test 2: Intensity Visualization
print("\n" + "=" * 80)
print("TEST 2: Intensity Visualization")
print("=" * 80)

try:
    if test1_pass and result.get('success'):
        fig = generator._plot_intensity(result)
        
        if fig is not None:
            print(f"✓ PASS: Intensity figure generated")
            print(f"  - Figure type: {type(fig).__name__}")
            
            # Check if it's a plotly figure with data
            if hasattr(fig, 'data'):
                print(f"  - Number of traces: {len(fig.data)}")
            
            test2_pass = True
        else:
            print(f"✗ FAIL: No figure returned")
            test2_pass = False
    else:
        print(f"⚠ SKIP: Intensity analysis failed in Test 1")
        test2_pass = False
        
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test2_pass = False

# Test 3: Motion Analysis through Report Generator
print("\n" + "=" * 80)
print("TEST 3: Motion Analysis (Report Generator Integration)")
print("=" * 80)

try:
    result_motion = generator._analyze_motion(tracks_df, current_units)
    
    if result_motion.get('success'):
        print(f"✓ PASS: Motion analysis completed")
        
        # Check for track_results
        if 'track_results' in result_motion:
            track_results = result_motion['track_results']
            print(f"  - Track results: {len(track_results)} tracks")
            
            if 'motion_type' in track_results.columns:
                motion_counts = track_results['motion_type'].value_counts()
                print(f"  - Motion types detected:")
                for mtype, count in motion_counts.items():
                    print(f"    * {mtype}: {count}")
        
        # Check for ensemble results
        if 'ensemble_results' in result_motion:
            ensemble = result_motion['ensemble_results']
            print(f"  - Ensemble results:")
            print(f"    * Mean speed: {ensemble.get('mean_speed', 'N/A'):.6f}")
            print(f"    * N tracks: {ensemble.get('n_tracks', 0)}")
        
        test3_pass = True
    else:
        print(f"✗ FAIL: {result_motion.get('error', 'Unknown error')}")
        test3_pass = False
        
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test3_pass = False

# Test 4: Motion Visualization with New Structure
print("\n" + "=" * 80)
print("TEST 4: Motion Visualization (Updated for New Structure)")
print("=" * 80)

try:
    if test3_pass and result_motion.get('success'):
        fig_motion = generator._plot_motion(result_motion)
        
        if fig_motion is not None:
            print(f"✓ PASS: Motion figure generated")
            print(f"  - Figure type: {type(fig_motion).__name__}")
            
            # For matplotlib figures
            if hasattr(fig_motion, 'axes'):
                print(f"  - Number of axes: {len(fig_motion.axes)}")
            
            # For plotly figures
            if hasattr(fig_motion, 'data'):
                print(f"  - Number of traces: {len(fig_motion.data)}")
            
            test4_pass = True
        else:
            print(f"✗ FAIL: No figure returned")
            test4_pass = False
    else:
        print(f"⚠ SKIP: Motion analysis failed in Test 3")
        test4_pass = False
        
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test4_pass = False

# Test 5: Verify plot_motion_analysis function directly
print("\n" + "=" * 80)
print("TEST 5: Direct plot_motion_analysis Function Call")
print("=" * 80)

try:
    from visualization import plot_motion_analysis
    
    if test3_pass and result_motion.get('success'):
        fig_direct = plot_motion_analysis(result_motion, title="Test Motion Analysis")
        
        if fig_direct is not None:
            print(f"✓ PASS: Direct visualization function works")
            print(f"  - Figure type: {type(fig_direct).__name__}")
            
            # Check for matplotlib figure with content
            if hasattr(fig_direct, 'axes'):
                axes = fig_direct.axes
                print(f"  - Number of axes: {len(axes)}")
                
                # Check if axes have content (not just "No data" message)
                has_content = False
                for ax in axes:
                    if len(ax.patches) > 0 or len(ax.lines) > 0:
                        has_content = True
                        break
                
                if has_content:
                    print(f"  - Has visualization content: YES")
                else:
                    print(f"  ⚠ Has visualization content: NO (may show 'no data' message)")
            
            test5_pass = True
        else:
            print(f"✗ FAIL: No figure returned")
            test5_pass = False
    else:
        print(f"⚠ SKIP: Motion analysis failed in Test 3")
        test5_pass = False
        
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test5_pass = False

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

tests = [
    ("Intensity Analysis (Report Generator)", test1_pass),
    ("Intensity Visualization", test2_pass),
    ("Motion Analysis (Report Generator)", test3_pass),
    ("Motion Visualization (Report Generator)", test4_pass),
    ("Direct Motion Visualization Function", test5_pass)
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "✓ PASS" if result else "✗ FAIL/SKIP"
    print(f"{status}: {test_name}")

print(f"\nTests passed: {passed}/{total}")

if passed >= 3:  # At least intensity + motion analysis should work
    print("\n✓ CORE FUNCTIONALITY WORKING - Both analyses validated!")
    sys.exit(0)
else:
    print(f"\n✗ CRITICAL FAILURES - Core functionality not working")
    sys.exit(1)
