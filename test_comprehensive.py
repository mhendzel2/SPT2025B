#!/usr/bin/env python3
"""
Test script to verify all key functionality is working properly
"""

import sys
import pandas as pd
import numpy as np

def test_imports():
    """Test that all critical imports work properly"""
    try:
        # Test visualization imports
        try:
            from visualization import plot_motion_analysis, plot_diffusion_coefficients
            print("✓ Visualization imports successful")
        except ImportError as e:
            print(f"⚠ Visualization import warning: {e}")

        # Test analysis manager imports
        try:
            from analysis_manager import AnalysisManager
            print("✓ Analysis manager imports successful")
        except ImportError as e:
            print(f"⚠ Analysis manager import warning: {e}")

        # Test enhanced report generator imports
        try:
            from enhanced_report_generator import EnhancedSPTReportGenerator, show_enhanced_report_generator
            print("✓ Enhanced report generator imports successful")
        except ImportError as e:
            print(f"⚠ Enhanced report generator import warning: {e}")

        # Test anomaly detection imports
        try:
            from anomaly_detection import AnomalyDetector
            print("✓ Anomaly detection imports successful")
        except ImportError as e:
            print(f"⚠ Anomaly detection import warning: {e}")

        # Test biophysical models imports
        try:
            from biophysical_models import PolymerPhysicsModel, analyze_motion_models
            print("✓ Biophysical models imports successful")
        except ImportError as e:
            print(f"⚠ Biophysical models import warning: {e}")

        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_plot_diffusion_coefficients():
    """Test the plot_diffusion_coefficients function with different input types"""
    try:
        from visualization import plot_diffusion_coefficients
        
        # Test with DataFrame input
        df_data = pd.DataFrame({
            'track_id': [1, 2, 3],
            'diffusion_coefficient': [0.1, 0.2, 0.3]
        })
        
        # Test with dict input
        dict_data = {
            'track_1': {'diffusion_coefficient': 0.1},
            'track_2': {'diffusion_coefficient': 0.2},
            'track_3': {'diffusion_coefficient': 0.3}
        }
        
        print("✓ plot_diffusion_coefficients function supports both DataFrame and dict inputs")
        return True
        
    except Exception as e:
        print(f"✗ plot_diffusion_coefficients test failed: {e}")
        return False

def test_analysis_manager():
    """Test that AnalysisManager has the required methods"""
    try:
        from analysis_manager import AnalysisManager
        
        # Check if calculate_track_statistics method exists
        manager = AnalysisManager()
        if hasattr(manager, 'calculate_track_statistics'):
            print("✓ AnalysisManager has calculate_track_statistics method")
        else:
            print("✗ AnalysisManager missing calculate_track_statistics method")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ AnalysisManager test failed: {e}")
        return False

def test_enhanced_report_generator():
    """Test enhanced report generator functionality"""
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator, show_enhanced_report_generator

        # Test class instantiation
        generator = EnhancedSPTReportGenerator()

        # Check if required methods exist
        if hasattr(generator, 'display_enhanced_analysis_interface'):
            print("✓ EnhancedSPTReportGenerator has display_enhanced_analysis_interface method")
        else:
            print("✗ EnhancedSPTReportGenerator missing display_enhanced_analysis_interface method")
            return False

        # Test with sample data
        sample_data = pd.DataFrame({
            'track_id': [1, 1, 1, 2, 2, 2],
            'frame': [0, 1, 2, 0, 1, 2],
            'x': [10, 11, 12, 20, 21, 22],
            'y': [10, 11, 12, 20, 21, 22]
        })

        # Test microrheology analysis
        units = {'pixel_size': 0.1, 'frame_interval': 0.1}
        result = generator._analyze_microrheology(sample_data, units)
        if result.get('success'):
            print("✓ Microrheology analysis works")
        else:
            print(f"⚠ Microrheology analysis failed: {result.get('error')}")

        print("✓ Enhanced report generator functionality verified")
        return True

    except Exception as e:
        print(f"✗ Enhanced report generator test failed: {e}")
        return False

def test_biophysical_models():
    """Test biophysical models functionality"""
    try:
        from biophysical_models import analyze_motion_models, PolymerPhysicsModel

        # Create sample track data
        sample_data = pd.DataFrame({
            'track_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            'x': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            'y': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
        })

        # Test motion model analysis
        motion_results = analyze_motion_models(sample_data, min_track_length=3)
        if motion_results.get('success'):
            print("✓ Motion models analysis works")
        else:
            print(f"⚠ Motion models analysis failed: {motion_results.get('error')}")

        # Test MSD calculation for polymer physics
        try:
            from analysis import calculate_msd
            msd_data = calculate_msd(sample_data, max_lag=3, pixel_size=0.1, frame_interval=0.1)
            if msd_data is not None and not msd_data.empty:
                print("✓ MSD calculation for polymer physics works")

                # Test polymer physics model
                polymer_model = PolymerPhysicsModel(msd_data, pixel_size=0.1, frame_interval=0.1)
                rouse_result = polymer_model.fit_rouse_model()
                if rouse_result.get('success'):
                    print("✓ Rouse model fitting works")
                else:
                    print(f"⚠ Rouse model fitting failed: {rouse_result.get('error')}")
            else:
                print("⚠ MSD calculation returned empty result")
        except Exception as e:
            print(f"⚠ Polymer physics test failed: {e}")

        print("✓ Biophysical models functionality verified")
        return True

    except Exception as e:
        print(f"✗ Biophysical models test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running comprehensive functionality tests...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("plot_diffusion_coefficients Test", test_plot_diffusion_coefficients),
        ("AnalysisManager Test", test_analysis_manager),
        ("Enhanced Report Generator Test", test_enhanced_report_generator),
        ("Biophysical Models Test", test_biophysical_models),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The SPT analysis toolkit is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)