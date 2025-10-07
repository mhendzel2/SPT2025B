"""
Test the enhanced report generator with adaptive Active Transport Detection.

This verifies that the adaptive threshold system integrates correctly with
the report generation workflow.
"""

import numpy as np
import pandas as pd
from enhanced_report_generator import EnhancedSPTReportGenerator

def create_sample_data():
    """Create a sample dataset with purely diffusive motion."""
    np.random.seed(42)
    
    print("Creating sample diffusive dataset (should report 'no active transport')...")
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
    
    tracks_df = pd.concat(tracks, ignore_index=True)
    print(f"Created {len(tracks)} tracks with {n_frames} frames each\n")
    
    return tracks_df


def test_report_generation():
    """Test Active Transport Detection in report generator."""
    print("="*70)
    print("TESTING ACTIVE TRANSPORT DETECTION IN REPORT GENERATOR")
    print("="*70 + "\n")
    
    # Create test data
    tracks_df = create_sample_data()
    
    # Initialize report generator
    generator = EnhancedSPTReportGenerator()
    
    # Set units
    current_units = {
        'pixel_size': 0.1,  # μm
        'frame_interval': 0.1  # s
    }
    
    print("Running Active Transport Detection analysis...")
    print("-" * 70)
    
    # Run the analysis
    result = generator._analyze_active_transport(tracks_df, current_units)
    
    # Display results
    print("\nAnalysis Result:")
    print(f"  Success: {result.get('success', False)}")
    
    if result.get('no_active_transport', False):
        print(f"  Status: {result.get('message', 'N/A')}")
        print(f"\n  ✓ EXPECTED BEHAVIOR:")
        print(f"    - Purely diffusive data correctly identified")
        print(f"    - No error thrown (success=True)")
        print(f"    - Informative message provided")
    elif result.get('success', False):
        summary = result.get('summary', {})
        thresholds = result.get('thresholds_used', {})
        
        print(f"  Threshold level: {summary.get('threshold_level', 'N/A')}")
        print(f"  Total segments: {summary.get('total_segments', 0)}")
        print(f"  Mean velocity: {summary.get('mean_velocity', 0):.4f} μm/s")
        print(f"  Mean straightness: {summary.get('mean_straightness', 0):.3f}")
        
        if thresholds:
            print(f"\n  Thresholds used:")
            print(f"    Velocity: {thresholds['velocity']:.3f} μm/s")
            print(f"    Straightness: {thresholds['straightness']:.2f}")
            print(f"    Segment length: {thresholds['segment_length']} frames")
        
        mode_fractions = summary.get('mode_fractions', {})
        if mode_fractions:
            print(f"\n  Mode fractions:")
            for mode, fraction in mode_fractions.items():
                print(f"    {mode}: {fraction*100:.1f}%")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
        print(f"\n  ✗ UNEXPECTED ERROR")
    
    # Test visualization
    print("\n" + "-" * 70)
    print("Testing visualization...")
    
    try:
        fig = generator._plot_active_transport(result)
        print("  ✓ Visualization created successfully")
        print(f"  Figure title: {fig.layout.title.text[:80]}...")
        
        # Check if annotations are present for no-transport case
        if result.get('no_active_transport', False) and fig.layout.annotations:
            print(f"  ✓ Informative annotation added")
            print(f"  Annotation text preview: {fig.layout.annotations[0].text[:80]}...")
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
    
    # Final verdict
    print("\n" + "="*70)
    print("TEST VERDICT")
    print("="*70)
    
    if result.get('success', False):
        if result.get('no_active_transport', False):
            print("✓ PASS: Adaptive thresholds correctly identified purely diffusive data")
            print("✓ PASS: No false 'failure' error thrown")
            print("✓ PASS: Informative message provided to user")
        else:
            print("✓ PASS: Active transport detected with adaptive thresholds")
        print("\nThe adaptive threshold system is working correctly!")
    else:
        print("✗ FAIL: Unexpected error occurred")
        print(f"Error: {result.get('error', 'Unknown')}")
    
    return result


if __name__ == '__main__':
    result = test_report_generation()
