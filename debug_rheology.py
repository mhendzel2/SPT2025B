#!/usr/bin/env python3
"""
Debug script for microrheology analysis issues.
This script will help identify why the viscoelastic moduli are returning null values.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_microrheology_analysis():
    """Debug the microrheology analysis to find why it's returning null values."""
    print("=== Microrheology Analysis Debug ===")
    
    try:
        from rheology import MicrorheologyAnalyzer
        
        # Create test tracking data
        print("\n1. Creating test tracking data...")
        np.random.seed(42)
        n_tracks = 20
        n_points_per_track = 100
        
        tracks_data = []
        for track_id in range(n_tracks):
            # Create realistic particle trajectories with different diffusion coefficients
            D = np.random.uniform(0.1, 2.0) * 1e-12  # m²/s
            dt = 0.1  # 100ms frame interval
            
            x_pos = [np.random.uniform(0, 100)]  # Start position in pixels
            y_pos = [np.random.uniform(0, 100)]
            
            # Generate diffusive motion
            for frame in range(1, n_points_per_track):
                # Displacement step size in pixels (assume 100nm pixels)
                sigma_pixel = np.sqrt(2 * D * dt) / (100e-9)  # Convert to pixels
                
                dx = np.random.normal(0, sigma_pixel)
                dy = np.random.normal(0, sigma_pixel)
                
                x_pos.append(x_pos[-1] + dx)
                y_pos.append(y_pos[-1] + dy)
                
                tracks_data.append({
                    'track_id': track_id,
                    'frame': frame,
                    'x': x_pos[-1],
                    'y': y_pos[-1]
                })
        
        tracks_df = pd.DataFrame(tracks_data)
        print(f"Generated {len(tracks_df)} track points for {n_tracks} tracks")
        print(f"Average track length: {len(tracks_df) / n_tracks:.1f} points")
        
        # Test different parameter combinations
        test_cases = [
            {
                'name': 'Small particles (10nm)',
                'particle_radius_nm': 10.0,
                'pixel_size_um': 0.1,
                'frame_interval_s': 0.1,
                'temperature_K': 300.0
            },
            {
                'name': 'Medium particles (50nm)',
                'particle_radius_nm': 50.0,
                'pixel_size_um': 0.1,
                'frame_interval_s': 0.1,
                'temperature_K': 300.0
            },
            {
                'name': 'Large particles (200nm)',
                'particle_radius_nm': 200.0,
                'pixel_size_um': 0.1,
                'frame_interval_s': 0.1,
                'temperature_K': 300.0
            },
            {
                'name': 'High resolution (20nm pixels)',
                'particle_radius_nm': 50.0,
                'pixel_size_um': 0.02,
                'frame_interval_s': 0.1,
                'temperature_K': 300.0
            },
            {
                'name': 'Fast sampling (10ms)',
                'particle_radius_nm': 50.0,
                'pixel_size_um': 0.1,
                'frame_interval_s': 0.01,
                'temperature_K': 300.0
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{i+2}. Testing {test_case['name']}...")
            
            try:
                # Initialize analyzer
                particle_radius_m = test_case['particle_radius_nm'] * 1e-9
                analyzer = MicrorheologyAnalyzer(
                    particle_radius_m=particle_radius_m,
                    temperature_K=test_case['temperature_K']
                )
                
                print(f"   Particle radius: {test_case['particle_radius_nm']} nm")
                print(f"   Pixel size: {test_case['pixel_size_um']} μm")
                print(f"   Frame interval: {test_case['frame_interval_s']} s")
                
                # Calculate MSD
                msd_data = analyzer.calculate_msd_from_tracks(
                    tracks_df, 
                    pixel_size_um=test_case['pixel_size_um'],
                    frame_interval_s=test_case['frame_interval_s'],
                    max_lag_frames=20
                )
                
                print(f"   MSD data points: {len(msd_data)}")
                
                if len(msd_data) > 0:
                    print(f"   MSD range: {msd_data['msd_m2'].min():.2e} - {msd_data['msd_m2'].max():.2e} m²")
                    print(f"   Time range: {msd_data['lag_time_s'].min():.3f} - {msd_data['lag_time_s'].max():.3f} s")
                    
                    # Test power law fit
                    power_law_fit = analyzer.fit_power_law_msd(msd_data)
                    print(f"   Power law exponent: {power_law_fit['exponent']:.3f}")
                    print(f"   Power law amplitude: {power_law_fit['amplitude']:.2e}")
                    print(f"   Fit quality (R²): {power_law_fit['r_squared']:.3f}")
                    
                    # Test viscosity calculation
                    viscosity = analyzer.calculate_effective_viscosity(msd_data)
                    print(f"   Effective viscosity: {viscosity:.2e} Pa·s")
                    
                    # Test complex modulus calculation
                    omega = 2 * np.pi / (test_case['frame_interval_s'] * 10)
                    g_prime, g_double_prime = analyzer.calculate_complex_modulus_gser(msd_data, omega)
                    print(f"   G' (storage modulus): {g_prime:.2e} Pa")
                    print(f"   G\" (loss modulus): {g_double_prime:.2e} Pa")
                    
                    # Run full analysis
                    full_result = analyzer.analyze_microrheology(
                        tracks_df,
                        pixel_size_um=test_case['pixel_size_um'],
                        frame_interval_s=test_case['frame_interval_s'],
                        max_lag=20
                    )
                    
                    print(f"   Full analysis success: {full_result.get('success', False)}")
                    if full_result.get('success', False):
                        moduli = full_result.get('moduli', {})
                        print(f"   Mean G': {moduli.get('g_prime_mean_pa', 'N/A'):.2e} Pa")
                        print(f"   Mean G\": {moduli.get('g_double_prime_mean_pa', 'N/A'):.2e} Pa")
                        
                        # Check if values are actually zero or just very small
                        if moduli.get('g_prime_mean_pa', 0) == 0:
                            print("   ⚠️  WARNING: G' is exactly zero!")
                        if moduli.get('g_double_prime_mean_pa', 0) == 0:
                            print("   ⚠️  WARNING: G\" is exactly zero!")
                    else:
                        print(f"   ❌ Analysis failed: {full_result.get('error', 'Unknown error')}")
                else:
                    print("   ❌ No MSD data calculated")
                    
            except Exception as e:
                print(f"   ❌ Test case failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n=== Debug Complete ===")
        
    except Exception as e:
        print(f"Debug script failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_microrheology_analysis()
