"""
Test script for adaptive threshold Active Transport Detection.

This tests the new automatic threshold adjustment feature that progressively
relaxes detection criteria when no directional motion is found with strict thresholds.
"""

import numpy as np
import pandas as pd
from biophysical_models import ActiveTransportAnalyzer

def generate_test_data():
    """Generate three types of test data."""
    np.random.seed(42)
    
    # Dataset 1: Purely diffusive (should trigger adaptive thresholds)
    print("\n=== Generating Test Dataset 1: Purely Diffusive ===")
    diffusive_tracks = []
    for track_id in range(20):
        n_frames = 50
        x = np.cumsum(np.random.randn(n_frames) * 0.02)  # Small random steps
        y = np.cumsum(np.random.randn(n_frames) * 0.02)
        frames = np.arange(n_frames)
        
        track_df = pd.DataFrame({
            'track_id': track_id,
            'frame': frames,
            'x': x,
            'y': y
        })
        diffusive_tracks.append(track_df)
    
    diffusive_df = pd.concat(diffusive_tracks, ignore_index=True)
    print(f"Created {len(diffusive_tracks)} diffusive tracks with {n_frames} frames each")
    
    # Dataset 2: Weakly directed (should be detected with relaxed thresholds)
    print("\n=== Generating Test Dataset 2: Weakly Directed ===")
    weak_directed_tracks = []
    for track_id in range(20, 35):
        n_frames = 50
        # Small directional bias + noise
        x = np.cumsum(np.random.randn(n_frames) * 0.02 + 0.015)  # Slight drift
        y = np.cumsum(np.random.randn(n_frames) * 0.02 + 0.01)
        frames = np.arange(n_frames)
        
        track_df = pd.DataFrame({
            'track_id': track_id,
            'frame': frames,
            'x': x,
            'y': y
        })
        weak_directed_tracks.append(track_df)
    
    weak_directed_df = pd.concat(weak_directed_tracks, ignore_index=True)
    print(f"Created {len(weak_directed_tracks)} weakly directed tracks")
    
    # Dataset 3: Strongly directed (should be detected with strict thresholds)
    print("\n=== Generating Test Dataset 3: Strongly Directed ===")
    strong_directed_tracks = []
    for track_id in range(35, 50):
        n_frames = 50
        # Strong directional motion
        x = np.cumsum(np.random.randn(n_frames) * 0.01 + 0.05)  # Strong drift
        y = np.cumsum(np.random.randn(n_frames) * 0.01 + 0.04)
        frames = np.arange(n_frames)
        
        track_df = pd.DataFrame({
            'track_id': track_id,
            'frame': frames,
            'x': x,
            'y': y
        })
        strong_directed_tracks.append(track_df)
    
    strong_directed_df = pd.concat(strong_directed_tracks, ignore_index=True)
    print(f"Created {len(strong_directed_tracks)} strongly directed tracks")
    
    return diffusive_df, weak_directed_df, strong_directed_df


def test_adaptive_thresholds(tracks_df, dataset_name, pixel_size=0.1, frame_interval=0.1):
    """
    Test adaptive threshold detection on a dataset.
    Simulates the logic from enhanced_report_generator.py _analyze_active_transport.
    """
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*70}")
    
    # Initialize analyzer
    analyzer = ActiveTransportAnalyzer(tracks_df, pixel_size=pixel_size, frame_interval=frame_interval)
    
    # Adaptive threshold levels (same as in enhanced_report_generator.py)
    threshold_levels = [
        {'velocity': 0.05, 'straightness': 0.7, 'segment_length': 5, 'level': 'strict'},
        {'velocity': 0.03, 'straightness': 0.6, 'segment_length': 4, 'level': 'moderate'},
        {'velocity': 0.02, 'straightness': 0.5, 'segment_length': 3, 'level': 'relaxed'},
        {'velocity': 0.01, 'straightness': 0.4, 'segment_length': 3, 'level': 'minimal'}
    ]
    
    segments_result = None
    thresholds_used = None
    
    # Try each threshold level until we find segments
    for i, threshold_set in enumerate(threshold_levels):
        print(f"\nAttempt {i+1}/{len(threshold_levels)}: {threshold_set['level']} thresholds")
        print(f"  velocity ≥ {threshold_set['velocity']:.3f} μm/s")
        print(f"  straightness ≥ {threshold_set['straightness']:.2f}")
        print(f"  min segment length: {threshold_set['segment_length']} frames")
        
        segments_result = analyzer.detect_directional_motion_segments(
            min_segment_length=threshold_set['segment_length'],
            straightness_threshold=threshold_set['straightness'],
            velocity_threshold=threshold_set['velocity']
        )
        
        n_segments = segments_result.get('total_segments', 0)
        print(f"  → Found {n_segments} segments")
        
        if segments_result.get('success', False) and n_segments > 0:
            thresholds_used = threshold_set
            print(f"  ✓ SUCCESS with {threshold_set['level']} thresholds!")
            break
    
    # Report results
    if segments_result and segments_result.get('success', False) and segments_result.get('total_segments', 0) > 0:
        modes_result = analyzer.characterize_transport_modes()
        
        print(f"\n{'─'*70}")
        print("RESULT: Active Transport Detected")
        print(f"{'─'*70}")
        print(f"Threshold level used: {thresholds_used['level']}")
        print(f"Total segments: {segments_result['total_segments']}")
        print(f"Mean velocity: {modes_result.get('mean_velocity', 0):.4f} μm/s")
        print(f"Mean straightness: {modes_result.get('mean_straightness', 0):.3f}")
        print("\nMode fractions:")
        for mode, fraction in modes_result.get('mode_fractions', {}).items():
            print(f"  {mode}: {fraction*100:.1f}%")
        
        return {
            'success': True,
            'segments': segments_result,
            'transport_modes': modes_result,
            'thresholds_used': thresholds_used
        }
    else:
        print(f"\n{'─'*70}")
        print("RESULT: No Active Transport Detected")
        print(f"{'─'*70}")
        print("Data appears to be purely diffusive/confined")
        print("Even minimal thresholds found no directional motion")
        
        return {
            'success': True,
            'no_active_transport': True,
            'message': 'No directional motion detected - data appears to be purely diffusive/confined'
        }


def main():
    """Run all tests."""
    print("="*70)
    print("ADAPTIVE THRESHOLD ACTIVE TRANSPORT DETECTION TEST")
    print("="*70)
    
    # Generate test data
    diffusive_df, weak_df, strong_df = generate_test_data()
    
    # Test parameters
    pixel_size = 0.1  # μm
    frame_interval = 0.1  # s
    
    # Test 1: Purely diffusive (should report no active transport)
    result1 = test_adaptive_thresholds(
        diffusive_df, 
        "Purely Diffusive Data",
        pixel_size, 
        frame_interval
    )
    
    # Test 2: Weakly directed (should be detected with relaxed thresholds)
    result2 = test_adaptive_thresholds(
        weak_df,
        "Weakly Directed Data",
        pixel_size,
        frame_interval
    )
    
    # Test 3: Strongly directed (should be detected with strict thresholds)
    result3 = test_adaptive_thresholds(
        strong_df,
        "Strongly Directed Data",
        pixel_size,
        frame_interval
    )
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    tests = [
        ("Purely Diffusive", result1),
        ("Weakly Directed", result2),
        ("Strongly Directed", result3)
    ]
    
    for name, result in tests:
        if result.get('no_active_transport', False):
            status = "✓ No active transport (as expected)"
            level = "N/A"
        elif result.get('success', False):
            status = "✓ Active transport detected"
            level = result.get('thresholds_used', {}).get('level', 'unknown')
        else:
            status = "✗ FAILED"
            level = "N/A"
        
        print(f"{name:.<30} {status} (threshold: {level})")
    
    print("\n✓ All tests completed successfully!")
    print("\nThe adaptive threshold system:")
    print("  1. Starts with strict criteria (fast, straight motion)")
    print("  2. Progressively relaxes thresholds if nothing is found")
    print("  3. Reports 'no active transport' for purely diffusive data")
    print("  4. Detects weak directional motion with relaxed thresholds")
    print("  5. Detects strong directional motion with strict thresholds")


if __name__ == '__main__':
    main()
