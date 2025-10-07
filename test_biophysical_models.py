"""
Test script for advanced biophysical models.
Tests Polymer Physics, Active Transport Analyzer, and Energy Landscape Mapper.
"""

import sys
import pandas as pd
import numpy as np

def generate_test_tracks(n_tracks=10, n_points=50):
    """Generate synthetic track data for testing."""
    tracks = []
    track_id = 0
    
    for i in range(n_tracks):
        # Half brownian, half directed
        if i < n_tracks // 2:
            # Brownian motion
            x = np.cumsum(np.random.randn(n_points)) * 0.1
            y = np.cumsum(np.random.randn(n_points)) * 0.1
        else:
            # Directed motion
            x = np.linspace(0, 5, n_points) + np.random.randn(n_points) * 0.05
            y = np.linspace(0, 5, n_points) + np.random.randn(n_points) * 0.05
        
        for frame in range(n_points):
            tracks.append({
                'track_id': track_id,
                'TRACK_ID': track_id,
                'frame': frame,
                'FRAME': frame,
                'x': x[frame],
                'y': y[frame],
                'POSITION_X': x[frame],
                'POSITION_Y': y[frame]
            })
        track_id += 1
    
    return pd.DataFrame(tracks)

def test_polymer_physics():
    """Test Polymer Physics Model."""
    print("\n" + "="*70)
    print("TESTING POLYMER PHYSICS MODEL")
    print("="*70)
    
    try:
        from biophysical_models import PolymerPhysicsModel
        from analysis import calculate_msd
        
        # Generate test data
        tracks_df = generate_test_tracks(n_tracks=10, n_points=50)
        
        # Calculate MSD
        print("\n1. Calculating MSD...")
        msd_result = calculate_msd(tracks_df, pixel_size=0.1, frame_interval=0.1)
        
        # Handle different return types
        if isinstance(msd_result, dict):
            if not msd_result.get('success'):
                print(f"   ❌ MSD calculation failed: {msd_result.get('error')}")
                return False
            msd_df = msd_result.get('ensemble_msd')
        else:
            # Direct DataFrame return
            msd_df = msd_result
        
        if msd_df is None or (isinstance(msd_df, pd.DataFrame) and msd_df.empty):
            print(f"   ❌ MSD calculation returned empty data")
            return False
        
        print(f"   ✓ MSD calculated with {len(msd_df)} time lags")
        
        # Initialize polymer model
        print("\n2. Fitting Rouse model...")
        polymer_model = PolymerPhysicsModel(
            msd_data=msd_df,
            pixel_size=0.1,
            frame_interval=0.1
        )
        
        # Fit Rouse model
        rouse_result = polymer_model.fit_rouse_model(fit_alpha=True)
        
        if not rouse_result.get('success'):
            print(f"   ❌ Rouse fit failed: {rouse_result.get('error')}")
            return False
        
        print(f"   ✓ Rouse model fitted successfully")
        params = rouse_result.get('parameters', rouse_result.get('params', {}))
        print(f"   Alpha: {params.get('alpha', 'N/A')}")
        print(f"   K: {params.get('K_rouse', 'N/A')}")
        if 'r_squared' in rouse_result:
            print(f"   R²: {rouse_result['r_squared']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_active_transport():
    """Test Active Transport Analyzer."""
    print("\n" + "="*70)
    print("TESTING ACTIVE TRANSPORT ANALYZER")
    print("="*70)
    
    try:
        from biophysical_models import ActiveTransportAnalyzer
        
        # Generate test data (mix of passive and active)
        tracks_df = generate_test_tracks(n_tracks=20, n_points=30)
        
        print("\n1. Initializing analyzer...")
        analyzer = ActiveTransportAnalyzer(
            tracks_df=tracks_df,
            pixel_size=0.1,
            frame_interval=0.1
        )
        print(f"   ✓ Analyzer initialized")
        
        print("\n2. Detecting active transport...")
        results = analyzer.detect_active_transport(
            speed_threshold=0.3,
            straightness_threshold=0.6,
            min_track_length=10
        )
        
        if not results.get('success'):
            print(f"   ❌ Detection failed: {results.get('error')}")
            return False
        
        print(f"   ✓ Active transport detection completed")
        
        summary = results['summary']
        print(f"\n   Summary:")
        print(f"   - Total tracks: {summary['total_tracks']}")
        print(f"   - Active tracks: {summary['active_tracks']}")
        print(f"   - Passive tracks: {summary['passive_tracks']}")
        print(f"   - Active fraction: {summary['active_fraction']:.1%}")
        
        stats = results['statistics']
        print(f"\n   Active Transport Statistics:")
        print(f"   - Mean speed: {stats['mean_speed']:.3f} μm/s")
        print(f"   - Max speed: {stats['max_speed']:.3f} μm/s")
        print(f"   - Mean straightness: {stats['mean_straightness']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_energy_landscape():
    """Test Energy Landscape Mapper."""
    print("\n" + "="*70)
    print("TESTING ENERGY LANDSCAPE MAPPER")
    print("="*70)
    
    try:
        from biophysical_models import EnergyLandscapeMapper
        
        # Generate test data with spatial structure
        tracks_df = generate_test_tracks(n_tracks=30, n_points=50)
        
        print("\n1. Initializing mapper...")
        mapper = EnergyLandscapeMapper(
            tracks_df=tracks_df,
            pixel_size=0.1,
            temperature=300.0
        )
        print(f"   ✓ Mapper initialized")
        
        print("\n2. Mapping energy landscape...")
        results = mapper.map_energy_landscape(
            resolution=15,
            method='boltzmann',
            smoothing=0.5,
            normalize=True
        )
        
        if not results.get('success'):
            print(f"   ❌ Mapping failed: {results.get('error')}")
            return False
        
        print(f"   ✓ Energy landscape mapped successfully")
        
        stats = results.get('statistics', {})
        print(f"\n   Landscape Statistics:")
        print(f"   - Min energy: {stats.get('min_energy', 0):.2f} kBT")
        print(f"   - Max energy: {stats.get('max_energy', 0):.2f} kBT")
        print(f"   - Energy range: {stats.get('energy_range', 0):.2f} kBT")
        print(f"   - Dwell regions: {stats.get('num_dwell_regions', 0)}")
        
        # Check required keys for UI
        required_keys = ['energy_map', 'x_coords', 'y_coords', 'force_field']
        missing_keys = [k for k in required_keys if k not in results]
        
        if missing_keys:
            print(f"   ⚠ Warning: Missing keys for UI: {missing_keys}")
        else:
            print(f"   ✓ All required keys present for UI")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔬 Testing Advanced Biophysical Models" + "\n")
    
    # Run tests
    test1_passed = test_polymer_physics()
    test2_passed = test_active_transport()
    test3_passed = test_energy_landscape()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Polymer Physics Model:      {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Active Transport Analyzer:  {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Energy Landscape Mapper:    {'✓ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n✅ All biophysical model tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Check the output above.")
        sys.exit(1)
