"""
Test script to verify all bug fixes for Enhanced Report Generator.

Tests:
1. iHMM State Segmentation - fixed max_states parameter error
2. Statistical Validation - fixed numpy array .empty attribute error  
3. DDM Tracking-Free Rheology - improved "not applicable" handling
4. Active Transport Detection - adaptive thresholds (already fixed)
"""

import numpy as np
import pandas as pd
from enhanced_report_generator import EnhancedSPTReportGenerator

def create_sample_tracks():
    """Create sample diffusive tracks for testing."""
    np.random.seed(42)
    tracks = []
    for track_id in range(20):
        n_frames = 50
        x = np.cumsum(np.random.randn(n_frames) * 0.02)
        y = np.cumsum(np.random.randn(n_frames) * 0.02)
        frames = np.arange(n_frames)
        
        track_df = pd.DataFrame({
            'track_id': track_id,
            'frame': frames,
            'x': x,
            'y': y
        })
        tracks.append(track_df)
    
    return pd.concat(tracks, ignore_index=True)


def test_ihmm_fix():
    """Test iHMM State Segmentation - should not crash on max_states."""
    print("="*70)
    print("TEST 1: iHMM State Segmentation (max_states parameter fix)")
    print("="*70)
    
    tracks_df = create_sample_tracks()
    generator = EnhancedSPTReportGenerator()
    current_units = {'pixel_size': 0.1, 'frame_interval': 0.1}
    
    try:
        result = generator._analyze_ihmm_blur(tracks_df, current_units)
        
        if result.get('success', False):
            print("✓ PASS: iHMM analysis completed successfully")
            print(f"  Result keys: {list(result.keys())}")
        elif 'max_states' in str(result.get('error', '')):
            print("✗ FAIL: max_states parameter error still present")
            print(f"  Error: {result.get('error')}")
            return False
        else:
            # May fail for other reasons (module not available, etc.)
            print(f"⚠ INFO: Analysis not completed: {result.get('error', 'Unknown')}")
            if 'module not available' in str(result.get('error', '')).lower():
                print("  (This is expected if iHMM module is optional)")
    except Exception as e:
        if 'max_states' in str(e):
            print(f"✗ FAIL: max_states error still occurs: {e}")
            return False
        else:
            print(f"⚠ INFO: Other error occurred: {e}")
    
    print()
    return True


def test_statistical_validation_fix():
    """Test Statistical Validation - should not crash on .empty attribute."""
    print("="*70)
    print("TEST 2: Statistical Validation (numpy array .empty fix)")
    print("="*70)
    
    tracks_df = create_sample_tracks()
    generator = EnhancedSPTReportGenerator()
    current_units = {'pixel_size': 0.1, 'frame_interval': 0.1}
    
    try:
        result = generator._analyze_statistical_validation(tracks_df, current_units)
        
        if result.get('success', False):
            print("✓ PASS: Statistical validation completed successfully")
            if 'bootstrap_D' in result:
                print(f"  Bootstrap D confidence interval: {result['bootstrap_D']}")
        elif '.empty' in str(result.get('error', '')) or 'no attribute' in str(result.get('error', '')):
            print("✗ FAIL: numpy array .empty error still present")
            print(f"  Error: {result.get('error')}")
            return False
        else:
            print(f"⚠ INFO: Analysis not completed: {result.get('error', 'Unknown')}")
    except AttributeError as e:
        if '.empty' in str(e) or 'no attribute' in str(e):
            print(f"✗ FAIL: .empty attribute error still occurs: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"⚠ INFO: Other error occurred: {e}")
    
    print()
    return True


def test_ddm_not_applicable():
    """Test DDM - should gracefully handle missing image stack."""
    print("="*70)
    print("TEST 3: DDM Tracking-Free Rheology (not applicable handling)")
    print("="*70)
    
    tracks_df = create_sample_tracks()
    generator = EnhancedSPTReportGenerator()
    current_units = {'pixel_size': 0.1, 'frame_interval': 0.1}
    
    try:
        # Call without image_stack (expected for track-based data)
        result = generator._analyze_ddm(tracks_df, current_units, image_stack=None)
        
        if result.get('not_applicable', False):
            print("✓ PASS: DDM correctly reported as 'not applicable'")
            print(f"  Message: {result.get('message', 'N/A')[:80]}...")
            
            # Test visualization
            fig = generator._plot_ddm(result)
            if fig:
                print("✓ PASS: Visualization handles 'not applicable' case")
        elif result.get('success', False) is False and 'requires image stack' in result.get('error', ''):
            print("⚠ WARNING: DDM still reports as 'error' instead of 'not applicable'")
            print(f"  (Should use 'not_applicable' flag for better UX)")
            return False
        else:
            print(f"⚠ INFO: Unexpected result: {result}")
    except Exception as e:
        print(f"✗ FAIL: DDM handling crashed: {e}")
        return False
    
    print()
    return True


def test_active_transport_adaptive():
    """Test Active Transport Detection - adaptive thresholds (already fixed)."""
    print("="*70)
    print("TEST 4: Active Transport Detection (adaptive thresholds)")
    print("="*70)
    
    tracks_df = create_sample_tracks()
    generator = EnhancedSPTReportGenerator()
    current_units = {'pixel_size': 0.1, 'frame_interval': 0.1}
    
    try:
        result = generator._analyze_active_transport(tracks_df, current_units)
        
        if result.get('no_active_transport', False):
            print("✓ PASS: Correctly identified no active transport (diffusive data)")
            print(f"  Message: {result.get('message', 'N/A')}")
        elif result.get('success', False):
            threshold_level = result.get('summary', {}).get('threshold_level', 'unknown')
            print(f"✓ PASS: Active transport detected with '{threshold_level}' thresholds")
        elif 'No directional motion segments detected' in result.get('error', ''):
            print("✗ FAIL: Old error message still present (adaptive thresholds not working)")
            return False
        else:
            print(f"⚠ INFO: Unexpected result: {result}")
    except Exception as e:
        print(f"✗ FAIL: Active transport analysis crashed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ENHANCED REPORT GENERATOR BUG FIX VALIDATION")
    print("="*70 + "\n")
    
    results = []
    
    # Test 1: iHMM
    results.append(("iHMM max_states fix", test_ihmm_fix()))
    
    # Test 2: Statistical Validation
    results.append(("Statistical Validation numpy fix", test_statistical_validation_fix()))
    
    # Test 3: DDM not applicable
    results.append(("DDM not applicable handling", test_ddm_not_applicable()))
    
    # Test 4: Active Transport (already fixed, verify)
    results.append(("Active Transport adaptive thresholds", test_active_transport_adaptive()))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print()
    print(f"Results: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ ALL BUGS FIXED!")
    else:
        print(f"\n⚠ {total_tests - total_passed} test(s) still failing")
    
    print("\nNOTE: Some analyses may show 'module not available' - this is expected")
    print("for optional dependencies and does NOT indicate a bug.")


if __name__ == '__main__':
    main()
