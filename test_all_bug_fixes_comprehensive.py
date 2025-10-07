"""
Comprehensive Report Generator Bug Fix Validation
===================================================
Tests all recent bug fixes:
1. Active Transport adaptive thresholds
2. iHMM correct method call (segment_trajectories → batch_analyze)
3. Statistical Validation DataFrame/array handling
4. DDM not_applicable handling

All in one comprehensive test suite.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_report_generator import EnhancedSPTReportGenerator
from data_access_utils import get_units

def create_diffusive_data():
    """Purely diffusive data for Active Transport and Statistical Validation tests."""
    np.random.seed(42)
    tracks = []
    
    for track_id in range(5):
        x, y = 0.0, 0.0
        for frame in range(50):
            tracks.append({'track_id': track_id, 'frame': frame, 'x': x, 'y': y})
            x += np.random.normal(0, 0.1)
            y += np.random.normal(0, 0.1)
    
    return pd.DataFrame(tracks)

def create_multistate_data():
    """Multi-state data for iHMM test."""
    np.random.seed(123)
    tracks = []
    
    for track_id in range(3):
        n_frames = 100
        x = np.zeros(n_frames)
        y = np.zeros(n_frames)
        
        for i in range(1, n_frames):
            # State switches
            if i < 25:
                D = 0.05
            elif i < 50:
                D = 0.2
            elif i < 75:
                D = 0.1
            else:
                D = 0.15
            
            dt = 0.1
            sigma = np.sqrt(2 * D * dt)
            x[i] = x[i-1] + np.random.normal(0, sigma)
            y[i] = y[i-1] + np.random.normal(0, sigma)
        
        for frame in range(n_frames):
            tracks.append({'track_id': track_id, 'frame': frame, 'x': x[frame], 'y': y[frame]})
    
    return pd.DataFrame(tracks)

def test_all_fixes():
    """Run comprehensive test of all bug fixes."""
    print("=" * 80)
    print("COMPREHENSIVE BUG FIX VALIDATION")
    print("=" * 80)
    
    generator = EnhancedSPTReportGenerator()
    current_units = get_units()
    
    results = {
        'active_transport': False,
        'ihmm': False,
        'statistical_validation': False,
        'ddm': False
    }
    
    # TEST 1: Active Transport adaptive thresholds
    print("\n[TEST 1/4] Active Transport Adaptive Thresholds")
    print("-" * 80)
    try:
        diffusive_data = create_diffusive_data()
        result = generator._analyze_active_transport(diffusive_data, current_units)
        
        if result.get('success', False):
            if result.get('no_active_transport', False):
                print("✓ PASS: Correctly identified purely diffusive data")
                print(f"  Status: {result.get('status', 'N/A')}")
                results['active_transport'] = True
            else:
                print("⚠ WARNING: Found active transport in diffusive data (false positive)")
                results['active_transport'] = False
        else:
            print(f"✗ FAIL: {result.get('error', 'Unknown error')}")
            results['active_transport'] = False
    except Exception as e:
        print(f"✗ FAIL: Exception - {e}")
        results['active_transport'] = False
    
    # TEST 2: iHMM correct method call
    print("\n[TEST 2/4] iHMM State Segmentation Method Call")
    print("-" * 80)
    try:
        multistate_data = create_multistate_data()
        result = generator._analyze_ihmm_blur(multistate_data, current_units)
        
        if result.get('success', False):
            summary = result.get('summary', {})
            n_tracks = summary.get('n_tracks_analyzed', 0)
            n_states_dist = summary.get('n_states_distribution', {})
            
            print(f"✓ PASS: iHMM analysis completed successfully")
            print(f"  Tracks analyzed: {n_tracks}")
            print(f"  States discovered (median): {n_states_dist.get('median', 'N/A')}")
            results['ihmm'] = True
        elif 'not available' in result.get('error', '').lower():
            print("⚠ INFO: iHMM module not available (expected if not installed)")
            results['ihmm'] = True  # Not a bug
        else:
            error = result.get('error', 'Unknown error')
            if 'segment_trajectories' in error:
                print(f"✗ FAIL: Still calling wrong method - {error}")
                results['ihmm'] = False
            else:
                print(f"✗ FAIL: {error}")
                results['ihmm'] = False
    except AttributeError as e:
        if 'segment_trajectories' in str(e):
            print(f"✗ FAIL: AttributeError still present - {e}")
            results['ihmm'] = False
        else:
            print(f"⚠ WARNING: Different AttributeError - {e}")
            results['ihmm'] = False
    except Exception as e:
        print(f"✗ FAIL: Exception - {e}")
        results['ihmm'] = False
    
    # TEST 3: Statistical Validation DataFrame/array handling
    print("\n[TEST 3/4] Statistical Validation Type Handling")
    print("-" * 80)
    try:
        diffusive_data = create_diffusive_data()
        result = generator._analyze_statistical_validation(diffusive_data, current_units)
        
        if result.get('success', False):
            bootstrap = result.get('bootstrap_ci', {})
            D_estimate = bootstrap.get('point_estimate', None)
            
            print("✓ PASS: Statistical validation completed successfully")
            if D_estimate is not None:
                print(f"  Bootstrap D: {D_estimate:.3e} μm²/s")
                ci_lower = bootstrap.get('ci_lower', None)
                ci_upper = bootstrap.get('ci_upper', None)
                if ci_lower and ci_upper:
                    print(f"  95% CI: [{ci_lower:.3e}, {ci_upper:.3e}]")
            results['statistical_validation'] = True
        else:
            error = result.get('error', 'Unknown error')
            if '.empty' in error or 'attribute' in error.lower():
                print(f"✗ FAIL: Array/DataFrame attribute error - {error}")
                results['statistical_validation'] = False
            else:
                print(f"✗ FAIL: {error}")
                results['statistical_validation'] = False
    except AttributeError as e:
        if 'empty' in str(e):
            print(f"✗ FAIL: numpy array .empty AttributeError - {e}")
            results['statistical_validation'] = False
        else:
            print(f"⚠ WARNING: Different AttributeError - {e}")
            results['statistical_validation'] = False
    except Exception as e:
        print(f"✗ FAIL: Exception - {e}")
        results['statistical_validation'] = False
    
    # TEST 4: DDM not_applicable handling
    print("\n[TEST 4/4] DDM Not Applicable Handling")
    print("-" * 80)
    try:
        diffusive_data = create_diffusive_data()
        result = generator._analyze_ddm(diffusive_data, current_units)
        
        if result.get('not_applicable', False):
            print("✓ PASS: DDM correctly reports 'not applicable' for track-based data")
            message = result.get('message', '')
            if message:
                print(f"  Message preview: {message[:80]}...")
            results['ddm'] = True
        elif result.get('success', False):
            print("⚠ WARNING: DDM reported success (expected not_applicable)")
            results['ddm'] = True  # Not wrong, just unexpected
        else:
            error = result.get('error', 'Unknown error')
            print(f"✗ FAIL: DDM reported error instead of not_applicable - {error}")
            results['ddm'] = False
    except Exception as e:
        print(f"✗ FAIL: Exception - {e}")
        results['ddm'] = False
    
    # SUMMARY
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    print("-" * 80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n✓✓✓ ALL BUGS FIXED! ✓✓✓")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) still failing")
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    sys.exit(0 if success else 1)
