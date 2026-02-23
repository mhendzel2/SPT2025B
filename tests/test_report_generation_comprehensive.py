"""
Comprehensive Test Suite for Enhanced Report Generation
Tests both single-file and batch-file report generation with all new analyses.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path


def create_synthetic_test_data(n_tracks=20, track_length=100, filename="test_tracks.csv"):
    """
    Create synthetic tracking data for testing.
    
    Includes various motion types:
    - Brownian diffusion
    - Subdiffusive (H<0.5)
    - Superdiffusive (H>0.5)
    - Confined motion
    """
    all_tracks = []
    
    for track_id in range(n_tracks):
        # Vary motion type
        motion_type = track_id % 4
        
        if motion_type == 0:  # Brownian
            alpha = 1.0
            D = 0.1
        elif motion_type == 1:  # Subdiffusive
            alpha = 0.7
            D = 0.05
        elif motion_type == 2:  # Superdiffusive
            alpha = 1.3
            D = 0.15
        else:  # Confined
            alpha = 0.5
            D = 0.03
        
        # Generate trajectory
        dt = 0.1  # seconds
        
        # Start position
        x0, y0 = np.random.rand(2) * 10  # μm
        
        positions = [(x0, y0)]
        
        for frame in range(1, track_length):
            # Anomalous diffusion step
            step_size = np.sqrt(2 * D * dt) * (dt ** (alpha - 1))
            angle = np.random.rand() * 2 * np.pi
            
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)
            
            # Add confinement for confined motion
            if motion_type == 3:
                # Restoring force toward center
                x_center, y_center = positions[0]
                current_x, current_y = positions[-1]
                dx -= 0.1 * (current_x - x_center)
                dy -= 0.1 * (current_y - y_center)
            
            new_x = positions[-1][0] + dx
            new_y = positions[-1][1] + dy
            
            positions.append((new_x, new_y))
        
        # Create dataframe for this track
        for frame, (x, y) in enumerate(positions):
            all_tracks.append({
                'track_id': track_id,
                'frame': frame,
                'x': x / 0.1,  # Convert to pixels (pixel_size = 0.1 μm)
                'y': y / 0.1,
                'intensity': 1000 + np.random.randn() * 100
            })
    
    df = pd.DataFrame(all_tracks)
    df.to_csv(filename, index=False)
    print(f"✓ Created synthetic test data: {filename}")
    print(f"  - {n_tracks} tracks")
    print(f"  - {track_length} frames per track")
    print(f"  - Motion types: Brownian, subdiffusive, superdiffusive, confined")
    
    return df


def test_single_file_report():
    """Test report generation with a single file."""
    print("\n" + "="*80)
    print("TEST 1: Single File Report Generation")
    print("="*80)
    
    # Create test data
    test_file = "test_single_report.csv"
    df = create_synthetic_test_data(n_tracks=20, track_length=100, filename=test_file)
    
    # Initialize report generator
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        generator = EnhancedSPTReportGenerator()
        generator.track_df = df
        generator.pixel_size = 0.1
        generator.frame_interval = 0.1
        
        print("\n📊 Testing New Advanced Analyses:")
        print("-" * 80)
        
        # Test each new analysis
        new_analyses = [
            'percolation_analysis',
            'ctrw_analysis',
            'fbm_enhanced',
            'crowding_correction',
            'loop_extrusion',
            'territory_mapping',
            'local_diffusion_map'
        ]
        
        current_units = {
            'pixel_size': 0.1,
            'frame_interval': 0.1
        }
        
        results = {}
        
        for analysis_key in new_analyses:
            if analysis_key in generator.available_analyses:
                analysis_info = generator.available_analyses[analysis_key]
                print(f"\n🔬 Testing: {analysis_info['name']}")
                
                try:
                    # Run analysis
                    analyze_func = analysis_info['function']
                    result = analyze_func(df, current_units)
                    
                    results[analysis_key] = result
                    
                    if result.get('success', False):
                        print(f"   ✅ SUCCESS")
                        
                        # Print key metrics
                        if analysis_key == 'fbm_enhanced':
                            print(f"      N_Valid: {result.get('n_valid', 0)}")
                            print(f"      H_Mean: {result.get('hurst_mean', np.nan):.3f}")
                            print(f"      D_Mean: {result.get('D_mean', np.nan):.2e} μm²/s")
                        elif analysis_key == 'percolation_analysis':
                            print(f"      Percolation: {result.get('percolation_detected', False)}")
                            print(f"      Type: {result.get('percolation_type', 'N/A')}")
                        elif analysis_key == 'ctrw_analysis':
                            erg_results = result.get('ergodicity', {})
                            print(f"      Ergodic: {erg_results.get('is_ergodic', 'N/A')}")
                        elif analysis_key == 'crowding_correction':
                            print(f"      D_measured: {result.get('D_measured', 0):.2e} μm²/s")
                            corrections = result.get('corrections', [])
                            if corrections:
                                print(f"      D_free (φ=0.3): {corrections[1]['D_free']:.2e} μm²/s")
                        elif analysis_key == 'loop_extrusion':
                            print(f"      Loop Detected: {result.get('loop_detected', False)}")
                            print(f"      Confidence: {result.get('confidence', 'none')}")
                        elif analysis_key == 'territory_mapping':
                            print(f"      Territories: {result.get('num_territories', 0)}")
                        elif analysis_key == 'local_diffusion_map':
                            stats = result.get('statistics', {})
                            print(f"      Mean D: {stats.get('mean_D', 0):.2e} μm²/s")
                        
                        # Test visualization
                        viz_func = analysis_info['visualization']
                        figures = viz_func(result)
                        if figures and len(figures) > 0:
                            print(f"      📊 Generated {len(figures)} figure(s)")
                        else:
                            print(f"      ⚠️  No figures generated")
                    
                    else:
                        print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    print(f"   ❌ EXCEPTION: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Summary
        print("\n" + "="*80)
        print("SINGLE FILE TEST SUMMARY")
        print("="*80)
        
        total = len(new_analyses)
        successful = sum(1 for r in results.values() if r.get('success', False))
        
        print(f"Total Analyses: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {100*successful/total:.1f}%")
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return results
    
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_file_report():
    """Test batch report generation with multiple files."""
    print("\n" + "="*80)
    print("TEST 2: Batch File Report Generation")
    print("="*80)
    
    # Create multiple test datasets
    n_datasets = 3
    test_files = []
    
    print("\n📁 Creating test datasets...")
    
    for i in range(n_datasets):
        filename = f"test_batch_{i+1}.csv"
        # Vary parameters slightly for each dataset
        n_tracks = 15 + i * 5
        df = create_synthetic_test_data(
            n_tracks=n_tracks,
            track_length=80 + i * 20,
            filename=filename
        )
        test_files.append(filename)
    
    print(f"\n✓ Created {n_datasets} test datasets")
    
    # Test batch analysis
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        print("\n📊 Running Batch Analysis:")
        print("-" * 80)
        
        # Simulate batch processing
        all_results = {}
        
        for analysis_key in ['fbm_enhanced', 'percolation_analysis', 'crowding_correction']:
            print(f"\n🔬 Batch Analysis: {analysis_key}")
            
            analysis_results = []
            
            for idx, test_file in enumerate(test_files):
                df = pd.read_csv(test_file)
                
                generator = EnhancedSPTReportGenerator()
                generator.track_df = df
                
                current_units = {'pixel_size': 0.1, 'frame_interval': 0.1}
                
                if analysis_key in generator.available_analyses:
                    analyze_func = generator.available_analyses[analysis_key]['function']
                    result = analyze_func(df, current_units)
                    
                    if result.get('success', False):
                        analysis_results.append(result)
                        print(f"   ✓ Dataset {idx+1}: SUCCESS")
                    else:
                        print(f"   ✗ Dataset {idx+1}: FAILED - {result.get('error', 'Unknown')}")
            
            # Aggregate statistics
            if analysis_results:
                print(f"\n   📈 Aggregate Statistics ({len(analysis_results)}/{n_datasets} datasets):")
                
                if analysis_key == 'fbm_enhanced':
                    all_H = [r['hurst_mean'] for r in analysis_results if 'hurst_mean' in r]
                    all_D = [r['D_mean'] for r in analysis_results if 'D_mean' in r]
                    
                    if all_H:
                        print(f"      Hurst Exponent: {np.mean(all_H):.3f} ± {np.std(all_H):.3f}")
                        print(f"      D coefficient: {np.mean(all_D):.2e} ± {np.std(all_D):.2e} μm²/s")
                        print(f"      Range H: [{np.min(all_H):.3f}, {np.max(all_H):.3f}]")
                
                elif analysis_key == 'percolation_analysis':
                    n_percolating = sum(1 for r in analysis_results if r.get('percolation_detected', False))
                    print(f"      Percolating systems: {n_percolating}/{len(analysis_results)}")
                
                elif analysis_key == 'crowding_correction':
                    all_D_measured = [r['D_measured'] for r in analysis_results]
                    print(f"      D_measured: {np.mean(all_D_measured):.2e} ± {np.std(all_D_measured):.2e} μm²/s")
                
                all_results[analysis_key] = analysis_results
        
        # Summary
        print("\n" + "="*80)
        print("BATCH TEST SUMMARY")
        print("="*80)
        
        for analysis_key, results in all_results.items():
            print(f"{analysis_key}: {len(results)}/{n_datasets} successful")
        
        # Clean up
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        print("\n✓ Batch test complete, test files cleaned up")
        
        return all_results
    
    except Exception as e:
        print(f"\n❌ Batch test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_real_data_if_available():
    """Test with real data if Cell1_spots.csv exists."""
    print("\n" + "="*80)
    print("TEST 3: Real Data Testing (Cell1_spots.csv)")
    print("="*80)
    
    real_data_file = "Cell1_spots.csv"
    
    if not os.path.exists(real_data_file):
        print(f"\n⚠️  {real_data_file} not found. Skipping real data test.")
        return None
    
    print(f"\n✓ Found {real_data_file}")
    
    try:
        df = pd.read_csv(real_data_file)
        print(f"   Loaded: {len(df)} rows")
        print(f"   Tracks: {df['track_id'].nunique() if 'track_id' in df.columns else 'N/A'}")
        
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        generator = EnhancedSPTReportGenerator()
        generator.track_df = df
        
        current_units = {'pixel_size': 0.1, 'frame_interval': 0.1}
        
        # Test FBM on real data
        print("\n🔬 Testing FBM Enhanced on real data:")
        
        if 'fbm_enhanced' in generator.available_analyses:
            analyze_func = generator.available_analyses['fbm_enhanced']['function']
            result = analyze_func(df, current_units)
            
            if result.get('success', False):
                print(f"   ✅ SUCCESS")
                print(f"      N_Tracks: {result.get('n_tracks', 0)}")
                print(f"      N_Valid: {result.get('n_valid', 0)}")
                print(f"      H_Mean: {result.get('hurst_mean', np.nan):.3f}")
                print(f"      H_Std: {result.get('hurst_std', np.nan):.3f}")
                print(f"      D_Mean: {result.get('D_mean', np.nan):.2e} μm²/s")
            else:
                print(f"   ❌ FAILED: {result.get('error', 'Unknown')}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Real data test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE REPORT GENERATION TEST SUITE")
    print("="*80)
    print("Testing all new polymer physics analyses:")
    print("  1. Percolation Analysis")
    print("  2. CTRW Analysis")
    print("  3. Enhanced FBM")
    print("  4. Crowding Correction")
    print("  5. Loop Extrusion Detection")
    print("  6. Chromosome Territory Mapping")
    print("  7. Local Diffusion Map D(x,y)")
    print("="*80)
    
    # Run tests
    single_results = test_single_file_report()
    batch_results = test_batch_file_report()
    real_results = test_real_data_if_available()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    if single_results:
        print("✓ Single file test: PASSED")
    else:
        print("✗ Single file test: FAILED")
    
    if batch_results:
        print("✓ Batch file test: PASSED")
    else:
        print("✗ Batch file test: FAILED")
    
    if real_results:
        print("✓ Real data test: PASSED")
    else:
        print("⚠ Real data test: SKIPPED or FAILED")
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
