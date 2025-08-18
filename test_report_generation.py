
#!/usr/bin/env python3
"""
Comprehensive test for report generation functionality
"""

import pandas as pd
import numpy as np
import sys
import traceback

def create_sample_track_data():
    """Create sample track data for testing"""
    np.random.seed(42)  # For reproducible results
    
    tracks = []
    for track_id in range(1, 6):  # 5 tracks
        n_points = np.random.randint(15, 30)  # Random track length between 15-30
        
        # Create a realistic trajectory with some diffusion
        x_start, y_start = np.random.uniform(0, 100, 2)
        x_positions = [x_start]
        y_positions = [y_start]
        
        for i in range(1, n_points):
            # Add random diffusion step
            dx = np.random.normal(0, 2)
            dy = np.random.normal(0, 2)
            x_positions.append(x_positions[-1] + dx)
            y_positions.append(y_positions[-1] + dy)
        
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            tracks.append({
                'track_id': track_id,
                'frame': i,
                'x': x,
                'y': y,
                'intensity': np.random.uniform(100, 1000)  # Add intensity data
            })
    
    return pd.DataFrame(tracks)

def test_enhanced_report_generator():
    """Test enhanced report generator with comprehensive analysis"""
    print("Testing Enhanced Report Generator...")
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create test data
        tracks_df = create_sample_track_data()
        print(f"✓ Created sample data with {len(tracks_df)} points across {tracks_df['track_id'].nunique()} tracks")
        
        # Initialize generator
        generator = EnhancedSPTReportGenerator()
        print("✓ Report generator initialized")
        
        # Test individual analysis functions
        units = {'pixel_size': 0.1, 'frame_interval': 0.1}
        
        # Test basic statistics
        print("\nTesting basic statistics analysis...")
        basic_result = generator._analyze_basic_statistics(tracks_df, units)
        if basic_result.get('success'):
            print("✓ Basic statistics analysis successful")
            print(f"  - Total tracks: {basic_result.get('total_tracks')}")
            print(f"  - Mean track length: {basic_result.get('mean_track_length', 0):.2f}")
        else:
            print(f"✗ Basic statistics failed: {basic_result.get('error')}")
        
        # Test diffusion analysis
        print("\nTesting diffusion analysis...")
        diffusion_result = generator._analyze_diffusion(tracks_df, units)
        if diffusion_result.get('success'):
            print("✓ Diffusion analysis successful")
            print(f"  - Diffusion coefficient: {diffusion_result.get('diffusion_coefficient', 0):.2e}")
        else:
            print(f"✗ Diffusion analysis failed: {diffusion_result.get('error')}")
        
        # Test motion classification
        print("\nTesting motion classification...")
        motion_result = generator._analyze_motion(tracks_df, units)
        if motion_result.get('success'):
            print("✓ Motion classification successful")
            classifications = motion_result.get('motion_classification', [])
            print(f"  - Classified {len(classifications)} tracks")
        else:
            print(f"✗ Motion classification failed: {motion_result.get('error')}")
        
        # Test intensity analysis
        print("\nTesting intensity analysis...")
        intensity_result = generator._analyze_intensity(tracks_df, units)
        if intensity_result.get('success'):
            print("✓ Intensity analysis successful")
            print(f"  - Found intensity column: {intensity_result.get('intensity_column')}")
        else:
            print(f"✗ Intensity analysis failed: {intensity_result.get('error')}")
        
        # Test microrheology analysis
        print("\nTesting microrheology analysis...")
        microrheology_result = generator._analyze_microrheology(tracks_df, units)
        if microrheology_result.get('success'):
            print("✓ Microrheology analysis successful")
        else:
            print(f"✗ Microrheology analysis failed: {microrheology_result.get('error')}")
        
        # Test batch report generation
        print("\nTesting batch report generation...")
        selected_analyses = ['basic_statistics', 'diffusion_analysis', 'motion_classification']
        batch_result = generator.generate_batch_report(tracks_df, selected_analyses, "Test Condition")
        
        if batch_result.get('success'):
            print("✓ Batch report generation successful")
            print(f"  - Generated report for condition: {batch_result.get('condition_name')}")
            print(f"  - Analyses completed: {len(batch_result.get('analysis_results', {}))}")
        else:
            print(f"✗ Batch report generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced report generator test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_biophysical_models():
    """Test biophysical models functionality"""
    print("\nTesting Biophysical Models...")
    
    try:
        from biophysical_models import analyze_motion_models, PolymerPhysicsModel
        
        # Create test data
        tracks_df = create_sample_track_data()
        print(f"✓ Created sample data for biophysical models")
        
        # Test motion model analysis
        print("\nTesting motion model analysis...")
        motion_results = analyze_motion_models(tracks_df, min_track_length=10)
        
        if motion_results.get('success'):
            print("✓ Motion model analysis successful")
            classifications = motion_results.get('classifications', {})
            print(f"  - Classified {len(classifications)} tracks")
            
            # Print classification summary
            if 'summary' in motion_results:
                summary = motion_results['summary']
                model_counts = summary.get('model_counts', {})
                for model, count in model_counts.items():
                    print(f"  - {model}: {count} tracks")
        else:
            print(f"✗ Motion model analysis failed: {motion_results.get('error')}")
        
        # Test MSD calculation and polymer physics
        print("\nTesting polymer physics models...")
        try:
            from analysis import calculate_msd
            
            # Calculate MSD
            msd_data = calculate_msd(tracks_df, max_lag=10, pixel_size=0.1, frame_interval=0.1)
            
            if msd_data is not None and not msd_data.empty:
                print("✓ MSD calculation successful")
                print(f"  - Generated {len(msd_data)} MSD data points")
                
                # Test polymer physics model
                polymer_model = PolymerPhysicsModel(msd_data, pixel_size=0.1, frame_interval=0.1)
                
                # Test Rouse model
                rouse_result = polymer_model.fit_rouse_model()
                if rouse_result.get('success'):
                    print("✓ Rouse model fitting successful")
                    params = rouse_result.get('parameters', {})
                    print(f"  - Diffusion coefficient: {params.get('D_macro', 0):.2e}")
                    print(f"  - Alpha: {params.get('alpha', 0):.3f}")
                else:
                    print(f"✗ Rouse model fitting failed: {rouse_result.get('error')}")
                    return False
                
                # Test fractal analysis
                fractal_result = polymer_model.analyze_fractal_dimension()
                if fractal_result.get('success'):
                    print("✓ Fractal dimension analysis successful")
                else:
                    print(f"✗ Fractal dimension analysis failed: {fractal_result.get('error')}")
                    return False
                    
            else:
                print("✗ MSD calculation failed or returned empty data")
                return False
                
        except ImportError as e:
            print(f"⚠ Could not import analysis module: {e}")
            return False
        except Exception as e:
            print(f"✗ Polymer physics test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Biophysical models test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all report generation tests"""
    print("=" * 60)
    print("COMPREHENSIVE REPORT GENERATION TESTING")
    print("=" * 60)
    
    tests = [
        ("Enhanced Report Generator", test_enhanced_report_generator),
        ("Biophysical Models", test_biophysical_models),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Report generation is working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
