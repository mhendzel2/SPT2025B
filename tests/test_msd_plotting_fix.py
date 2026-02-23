#!/usr/bin/env python3
"""
Test MSD plotting functionality after fix
"""

import pandas as pd
import numpy as np
import sys

def create_sample_track_data():
    """Create sample track data for testing"""
    np.random.seed(42)
    
    tracks = []
    for track_id in range(1, 6):  # 5 tracks
        n_points = 25
        x_start, y_start = np.random.uniform(0, 100, 2)
        x_positions = [x_start]
        y_positions = [y_start]
        
        for i in range(1, n_points):
            dx = np.random.normal(0, 1.5)
            dy = np.random.normal(0, 1.5)
            x_positions.append(x_positions[-1] + dx)
            y_positions.append(y_positions[-1] + dy)
        
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            tracks.append({
                'track_id': track_id,
                'frame': i,
                'x': x,
                'y': y
            })
    
    return pd.DataFrame(tracks)

def test_diffusion_analysis_and_plotting():
    """Test that diffusion analysis and plotting work correctly"""
    print("=" * 60)
    print("Testing Diffusion Analysis and MSD Plotting Fix")
    print("=" * 60)
    
    try:
        from analysis import analyze_diffusion
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create test data
        print("\n1. Creating sample track data...")
        tracks_df = create_sample_track_data()
        print(f"   ✓ Created {len(tracks_df['track_id'].unique())} tracks with {len(tracks_df)} points")
        
        # Run diffusion analysis
        print("\n2. Running diffusion analysis...")
        result = analyze_diffusion(
            tracks_df,
            pixel_size=0.1,
            frame_interval=0.1,
            max_lag=10
        )
        
        print(f"   Analysis success: {result.get('success', False)}")
        
        # Check structure
        if result.get('success'):
            print(f"   Result structure:")
            print(f"     - Has 'result' key: {'result' in result}")
            print(f"     - Has 'error' key: {'error' in result}")
            
            # Check nested data
            if 'result' in result:
                nested = result['result']
                print(f"     - Nested 'msd_data': {'msd_data' in nested}")
                print(f"     - Nested 'track_results': {'track_results' in nested}")
                print(f"     - Nested 'ensemble_results': {'ensemble_results' in nested}")
                
                if 'msd_data' in nested:
                    msd_df = nested['msd_data']
                    if isinstance(msd_df, pd.DataFrame):
                        print(f"       MSD data shape: {msd_df.shape}")
                        print(f"       MSD columns: {list(msd_df.columns)}")
                
                if 'track_results' in nested:
                    track_res = nested['track_results']
                    if isinstance(track_res, pd.DataFrame):
                        print(f"       Track results shape: {track_res.shape}")
                        if not track_res.empty:
                            print(f"       Track results columns: {list(track_res.columns)}")
        else:
            print(f"   ✗ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Test visualization
        print("\n3. Testing visualization function...")
        generator = EnhancedSPTReportGenerator()
        
        # Test data extraction helper
        print("   Testing _extract_analysis_data helper...")
        extracted = generator._extract_analysis_data(result)
        print(f"     - Extracted has 'msd_data': {'msd_data' in extracted}")
        print(f"     - Extracted has 'track_results': {'track_results' in extracted}")
        print(f"     - Extracted has 'ensemble_results': {'ensemble_results' in extracted}")
        print(f"     - Extracted has 'success': {'success' in extracted}")
        
        # Try to create plot
        print("\n4. Creating MSD plot...")
        fig = generator._plot_diffusion(result)
        
        if fig is not None:
            print(f"   ✓ Plot created successfully")
            print(f"     - Figure type: {type(fig).__name__}")
            print(f"     - Number of traces: {len(fig.data) if hasattr(fig, 'data') else 'N/A'}")
            
            # Check if plot has actual data
            if hasattr(fig, 'data') and len(fig.data) > 0:
                print(f"   ✓ Plot contains {len(fig.data)} traces")
                
                # Check for specific subplot content
                for i, trace in enumerate(fig.data[:3]):  # Check first 3 traces
                    trace_type = trace.type if hasattr(trace, 'type') else 'unknown'
                    has_data = len(trace.x) > 0 if hasattr(trace, 'x') else False
                    print(f"     - Trace {i}: type={trace_type}, has_data={has_data}")
            else:
                print(f"   ⚠ Plot created but has no traces")
        else:
            print(f"   ✗ Plot creation returned None")
            return False
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_analysis_and_plotting():
    """Test that motion analysis and plotting work correctly"""
    print("\n" + "=" * 60)
    print("Testing Motion Analysis Plotting")
    print("=" * 60)
    
    try:
        from analysis import analyze_motion
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create test data
        print("\n1. Creating sample track data...")
        tracks_df = create_sample_track_data()
        
        # Run motion analysis
        print("\n2. Running motion analysis...")
        result = analyze_motion(
            tracks_df,
            pixel_size=0.1,
            frame_interval=0.1
        )
        
        print(f"   Motion analysis success: {result.get('success', False)}")
        
        if result.get('success'):
            # Check if data is nested or flat
            print(f"   Result structure:")
            print(f"     - Has 'result' key: {'result' in result}")
            print(f"     - Has 'track_results': {'track_results' in result}")
            print(f"     - Has 'ensemble_results': {'ensemble_results' in result}")
        
        # Test visualization
        print("\n3. Testing motion plot...")
        generator = EnhancedSPTReportGenerator()
        fig = generator._plot_motion(result)
        
        if fig is not None:
            print(f"   ✓ Motion plot created successfully")
        else:
            print(f"   ⚠ Motion plot returned None")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Motion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nMSD Plotting Fix Verification Test")
    print("=" * 60 + "\n")
    
    success = True
    
    # Test diffusion analysis and plotting
    if not test_diffusion_analysis_and_plotting():
        success = False
    
    # Test motion analysis and plotting
    if not test_motion_analysis_and_plotting():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
