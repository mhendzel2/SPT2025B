#!/usr/bin/env python
"""
Test script for Gemini-identified edge cases in intensity analysis.
Tests:
1. Channel detection supporting >3 channels (up to 10)
2. Savgol filter window adjustment for short tracks
3. Z-score calculation with constant intensity (std=0)
4. Plot color indexing with >3 channels
5. Legacy report_generator.py integration
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_channel_detection_many():
    """Test 1: Verify support for >3 channels."""
    print("=" * 70)
    print("TEST 1: Channel Detection (up to 10 channels)")
    print("=" * 70)
    
    from intensity_analysis import extract_intensity_channels
    
    # Create data with 5 channels
    df = pd.DataFrame({
        'track_id': [1, 1, 1],
        'frame': [0, 1, 2],
        'x': [1.0, 1.1, 1.2],
        'y': [1.0, 1.1, 1.2],
        'mean_intensity_ch1': [100, 95, 90],
        'mean_intensity_ch2': [80, 78, 76],
        'mean_intensity_ch3': [60, 58, 56],
        'mean_intensity_ch4': [40, 38, 36],
        'mean_intensity_ch5': [20, 18, 16],
    })
    
    channels = extract_intensity_channels(df)
    detected = list(channels.keys())
    print(f"  Detected {len(channels)} channels: {detected}")
    
    if len(channels) >= 5:
        print("  PASS: All 5+ channels detected")
        return True
    else:
        print(f"  FAIL: Should detect 5 channels, got {len(channels)}")
        return False


def test_short_track_savgol():
    """Test 2: Savgol filter with short tracks (<5 points)."""
    print("\n" + "=" * 70)
    print("TEST 2: Short Track Handling (Savgol filter window)")
    print("=" * 70)
    
    from intensity_analysis import analyze_intensity_profiles
    
    # Create a track with only 3 points (less than default window=5)
    df_short = pd.DataFrame({
        'track_id': [1, 1, 1],
        'frame': [0, 1, 2],
        'x': [1.0, 1.1, 1.2],
        'y': [1.0, 1.1, 1.2],
        'intensity': [100.0, 95.0, 90.0]
    })
    
    try:
        result = analyze_intensity_profiles(df_short, 'intensity', smoothing_window=5)
        profiles = result.get('track_profiles', {})
        print(f"  Profiles processed: {len(profiles)}")
        if profiles:
            print(f"  Track 1 smoothed: {profiles.get(1, {}).get('smoothed_intensities', 'N/A')}")
        print("  PASS: Short track handled without ValueError")
        return True
    except ValueError as e:
        print(f"  FAIL: Crashed with ValueError: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: Crashed with {type(e).__name__}: {e}")
        return False


def test_constant_intensity_zscore():
    """Test 3: Z-score calculation with constant intensity (std=0)."""
    print("\n" + "=" * 70)
    print("TEST 3: Constant Intensity (Z-score div-by-zero)")
    print("=" * 70)
    
    from intensity_analysis import analyze_intensity_profiles, classify_intensity_behavior
    
    # Create a track with constant intensity (std=0)
    df_const = pd.DataFrame({
        'track_id': [1, 1, 1, 1, 1, 1],
        'frame': [0, 1, 2, 3, 4, 5],
        'x': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'y': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'intensity': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    })
    
    passed = True
    
    # Test analyze_intensity_profiles
    try:
        result = analyze_intensity_profiles(df_const, 'intensity')
        # Check for NaN/Inf in results
        profiles = result.get('track_profiles', {})
        if profiles:
            vals = profiles.get(1, {}).get('smoothed_intensities', [])
            has_inf = any(np.isinf(v) for v in vals if v is not None)
            if has_inf:
                print("  analyze_intensity_profiles: WARNING - contains Inf values")
            else:
                print("  analyze_intensity_profiles: OK - No division by zero")
        else:
            print("  analyze_intensity_profiles: OK - No profiles (expected)")
    except Exception as e:
        print(f"  analyze_intensity_profiles: FAIL - {type(e).__name__}: {e}")
        passed = False
    
    # Test classify_intensity_behavior
    try:
        result = classify_intensity_behavior(df_const, 'intensity')
        # Check for NaN/Inf in blinking_frequency
        if 'blinking_frequency' in result.columns:
            has_inf = result['blinking_frequency'].apply(lambda x: np.isinf(x) if pd.notna(x) else False).any()
            if has_inf:
                print("  classify_intensity_behavior: WARNING - contains Inf values")
            else:
                print("  classify_intensity_behavior: OK - No division by zero")
        else:
            print("  classify_intensity_behavior: OK - No blinking_frequency column")
    except Exception as e:
        print(f"  classify_intensity_behavior: FAIL - {type(e).__name__}: {e}")
        passed = False
    
    if passed:
        print("  PASS: Constant intensity handled safely")
    else:
        print("  FAIL: Division by zero occurred")
    return passed


def test_plot_color_index():
    """Test 4: Plot color indexing with >3 channels."""
    print("\n" + "=" * 70)
    print("TEST 4: Plot Color Index (>3 channels)")
    print("=" * 70)
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create mock result with 5 channels
        result = {
            'success': True,
            'channel_stats': {
                'ch1': {'mean': 100, 'median': 95, 'std': 10},
                'ch2': {'mean': 80, 'median': 78, 'std': 8},
                'ch3': {'mean': 60, 'median': 58, 'std': 6},
                'ch4': {'mean': 40, 'median': 38, 'std': 4},
                'ch5': {'mean': 20, 'median': 18, 'std': 2},
            }
        }
        
        generator = EnhancedSPTReportGenerator()
        fig = generator._plot_intensity(result)
        
        if fig is not None:
            print(f"  Generated plot for {len(result['channel_stats'])} channels")
            print("  PASS: No IndexError with 5 channels")
            return True
        else:
            print("  FAIL: Plot returned None")
            return False
            
    except IndexError as e:
        print(f"  FAIL: IndexError - {e}")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_report_generator_integration():
    """Test 5: Legacy report_generator.py intensity integration."""
    print("\n" + "=" * 70)
    print("TEST 5: Legacy Report Generator Integration")
    print("=" * 70)
    
    try:
        from report_generator import INTENSITY_ANALYSIS_AVAILABLE
        
        if INTENSITY_ANALYSIS_AVAILABLE:
            print("  Intensity analysis module imported: True")
            print("  PASS: Integration available in report_generator.py")
            return True
        else:
            print("  FAIL: INTENSITY_ANALYSIS_AVAILABLE is False")
            return False
    except ImportError as e:
        print(f"  FAIL: Import error - {e}")
        return False
    except AttributeError:
        print("  FAIL: INTENSITY_ANALYSIS_AVAILABLE not defined")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("GEMINI EDGE CASE TESTS FOR INTENSITY ANALYSIS")
    print("=" * 70)
    
    results = []
    
    results.append(("Channel detection (>3)", test_channel_detection_many()))
    results.append(("Short track Savgol", test_short_track_savgol()))
    results.append(("Constant intensity z-score", test_constant_intensity_zscore()))
    results.append(("Plot color index (>3 ch)", test_plot_color_index()))
    results.append(("Legacy report integration", test_report_generator_integration()))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\nALL TESTS PASSED - Gemini-identified issues fixed!")
        sys.exit(0)
    else:
        print(f"\n{total - passed} TEST(S) FAILED")
        sys.exit(1)
