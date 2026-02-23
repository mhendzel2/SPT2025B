#!/usr/bin/env python3
"""
Test enhanced visualizations: MSD with reference slopes, microrheology with crossover,
log-normal histograms, and temporal trajectory coloring.
"""

import pandas as pd
import numpy as np
import sys

def create_sample_data():
    """Create sample track data for testing"""
    np.random.seed(42)
    
    tracks = []
    for track_id in range(1, 11):  # 10 tracks
        n_points = np.random.randint(20, 40)
        x_start, y_start = np.random.uniform(0, 50, 2)
        
        x_positions = [x_start]
        y_positions = [y_start]
        
        # Simulate different motion types
        if track_id <= 3:  # Subdiffusive
            alpha = 0.5
        elif track_id <= 6:  # Brownian
            alpha = 1.0
        else:  # Superdiffusive/directed
            alpha = 1.5
        
        for i in range(1, n_points):
            step_size = 0.5 * (i ** (alpha/2)) / (n_points ** (alpha/2))
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
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

def test_enhanced_msd_plot():
    """Test enhanced MSD plotting with reference slopes"""
    print("\n" + "="*60)
    print("Testing Enhanced MSD Plot with Reference Slopes")
    print("="*60)
    
    try:
        from analysis import analyze_diffusion
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create test data
        print("\n1. Creating sample track data...")
        tracks_df = create_sample_data()
        print(f"   ✓ Created {len(tracks_df['track_id'].unique())} tracks")
        
        # Run diffusion analysis
        print("\n2. Running diffusion analysis...")
        result = analyze_diffusion(
            tracks_df,
            pixel_size=0.1,
            frame_interval=0.1,
            max_lag=15
        )
        
        if result.get('success'):
            print(f"   ✓ Analysis successful")
            
            # Generate enhanced plot
            print("\n3. Creating enhanced MSD plot...")
            generator = EnhancedSPTReportGenerator()
            fig = generator._plot_diffusion(result)
            
            if fig and hasattr(fig, 'data'):
                n_traces = len(fig.data)
                print(f"   ✓ Plot created with {n_traces} traces")
                
                # Check for specific features
                has_ensemble = any('Ensemble' in str(trace.name) for trace in fig.data if hasattr(trace, 'name'))
                has_ref_lines = any('α=' in str(trace.name) for trace in fig.data if hasattr(trace, 'name'))
                has_ci = any('CI' in str(trace.name) for trace in fig.data if hasattr(trace, 'name'))
                
                print(f"     - Ensemble average: {has_ensemble}")
                print(f"     - Reference lines (α): {has_ref_lines}")
                print(f"     - Confidence intervals: {has_ci}")
                
                if has_ensemble and has_ref_lines:
                    print("\n   ✓✓ All MSD enhancements present!")
                    return True
            else:
                print("   ✗ Plot creation failed")
                return False
        else:
            print(f"   ✗ Analysis failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n   ✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_microrheology():
    """Test enhanced microrheology with crossover detection"""
    print("\n" + "="*60)
    print("Testing Enhanced Microrheology Plot")
    print("="*60)
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create synthetic microrheology data
        print("\n1. Creating synthetic rheology data...")
        frequency = np.logspace(-2, 2, 50)  # 0.01 to 100 rad/s
        
        # Simulate viscoelastic material with crossover
        omega_c = 1.0  # Crossover frequency
        G_0 = 100.0  # Pa
        
        G_prime = G_0 * (frequency / omega_c)**0.5 / np.sqrt(1 + (frequency/omega_c)**2)
        G_double_prime = G_0 * (frequency / omega_c)**0.5 * (frequency/omega_c) / np.sqrt(1 + (frequency/omega_c)**2)
        eta_star = np.sqrt(G_prime**2 + G_double_prime**2) / frequency
        
        result = {
            'success': True,
            'frequency': frequency,
            'storage_modulus': G_prime,
            'loss_modulus': G_double_prime,
            'complex_viscosity': eta_star
        }
        print(f"   ✓ Created data with crossover at ω_c = {omega_c} rad/s")
        
        # Generate enhanced plot
        print("\n2. Creating enhanced microrheology plot...")
        generator = EnhancedSPTReportGenerator()
        fig = generator._plot_microrheology(result)
        
        if fig and hasattr(fig, 'data'):
            n_traces = len(fig.data)
            print(f"   ✓ Plot created with {n_traces} traces")
            
            # Check for features
            has_gprime = any("G'" in str(trace.name) or "Storage" in str(trace.name) for trace in fig.data if hasattr(trace, 'name'))
            has_gdouble = any('G"' in str(trace.name) or "Loss" in str(trace.name) for trace in fig.data if hasattr(trace, 'name'))
            has_tan_delta = any('tan' in str(trace.name) for trace in fig.data if hasattr(trace, 'name'))
            has_pie = any(trace.type == 'pie' for trace in fig.data)
            
            print(f"     - G' (Storage modulus): {has_gprime}")
            print(f"     - G'' (Loss modulus): {has_gdouble}")
            print(f"     - tan(δ) plot: {has_tan_delta}")
            print(f"     - Material state pie: {has_pie}")
            
            if all([has_gprime, has_gdouble, has_tan_delta, has_pie]):
                print("\n   ✓✓ All microrheology enhancements present!")
                return True
            else:
                print("\n   ⚠ Some features missing")
                return False
        else:
            print("   ✗ Plot creation failed")
            return False
            
    except Exception as e:
        print(f"\n   ✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_track_statistics():
    """Test enhanced track statistics with log-normal histograms"""
    print("\n" + "="*60)
    print("Testing Enhanced Track Statistics Plot")
    print("="*60)
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create synthetic statistics data
        print("\n1. Creating synthetic statistics data...")
        n_tracks = 100
        
        # Log-normal distribution for diffusion coefficients
        D_mean_log = -1.0  # log10(D) mean
        D_std_log = 0.5
        D_values = 10 ** np.random.normal(D_mean_log, D_std_log, n_tracks)
        
        stats_df = pd.DataFrame({
            'track_id': range(1, n_tracks+1),
            'track_length': np.random.randint(10, 50, n_tracks),
            'diffusion_coefficient': D_values,
            'total_displacement': np.random.gamma(2, 5, n_tracks),
            'mean_speed': D_values * np.random.uniform(0.5, 2.0, n_tracks)
        })
        
        result = {
            'success': True,
            'statistics_df': stats_df
        }
        print(f"   ✓ Created statistics for {n_tracks} tracks")
        
        # Generate enhanced plot
        print("\n2. Creating enhanced statistics plot...")
        generator = EnhancedSPTReportGenerator()
        fig = generator._plot_basic_statistics(result)
        
        if fig and hasattr(fig, 'data'):
            n_traces = len(fig.data)
            print(f"   ✓ Plot created with {n_traces} traces (4 histograms expected)")
            
            # Check for log scaling on D coefficient plot
            if hasattr(fig, 'layout') and hasattr(fig.layout, 'xaxis2'):
                x_axis_type = getattr(fig.layout.xaxis2, 'type', 'linear')
                print(f"     - D coefficient axis type: {x_axis_type}")
                
                if x_axis_type == 'log':
                    print("\n   ✓✓ Log-normal histogram for diffusion coefficients!")
                    return True
                else:
                    print("\n   ⚠ D coefficient plot not log-scaled")
                    return False
            else:
                print("   ⚠ Could not verify axis scaling")
                return False
        else:
            print("   ✗ Plot creation failed")
            return False
            
    except Exception as e:
        print(f"\n   ✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_trajectory_plot():
    """Test temporal color coding trajectory plot"""
    print("\n" + "="*60)
    print("Testing Temporal Trajectory Visualization")
    print("="*60)
    
    try:
        from enhanced_trajectory_viz import plot_trajectories_temporal_color
        
        # Create test data
        print("\n1. Creating sample track data...")
        tracks_df = create_sample_data()
        
        # Generate temporal color plot
        print("\n2. Creating temporal color trajectory plot...")
        fig = plot_trajectories_temporal_color(
            tracks_df,
            pixel_size=0.1,
            max_tracks=5
        )
        
        if fig and hasattr(fig, 'data'):
            n_traces = len(fig.data)
            print(f"   ✓ Plot created with {n_traces} traces")
            print(f"     (Includes trajectory segments + start/end markers)")
            
            # Check for colorbar (indicates temporal coloring)
            has_colorbar = any(hasattr(trace, 'marker') and 
                             hasattr(trace.marker, 'colorbar') 
                             for trace in fig.data)
            
            if has_colorbar:
                print("\n   ✓✓ Temporal color coding (dragon tails) implemented!")
                return True
            else:
                print("\n   ⚠ Colorbar not found")
                return False
        else:
            print("   ✗ Plot creation failed")
            return False
            
    except Exception as e:
        print(f"\n   ✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Enhanced Visualization Test Suite")
    print("="*60)
    
    success_count = 0
    total_tests = 4
    
    # Run all tests
    if test_enhanced_msd_plot():
        success_count += 1
    
    if test_enhanced_microrheology():
        success_count += 1
    
    if test_enhanced_track_statistics():
        success_count += 1
    
    if test_temporal_trajectory_plot():
        success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Test Results: {success_count}/{total_tests} passed")
    print("="*60)
    
    if success_count == total_tests:
        print("✓✓ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"⚠ {total_tests - success_count} test(s) failed")
        sys.exit(1)
