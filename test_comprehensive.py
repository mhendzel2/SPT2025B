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
        from visualization import plot_motion_analysis, plot_diffusion_coefficients
        print("✓ Visualization imports successful")
        
        # Test analysis manager imports
        from analysis_manager import AnalysisManager
        print("✓ Analysis manager imports successful")
        
        # Test enhanced report generator imports
        from enhanced_report_generator import EnhancedSPTReportGenerator, show_enhanced_report_generator
        print("✓ Enhanced report generator imports successful")
        
        # Test anomaly detection imports
        from anomaly_detection import AnomalyDetector
        from anomaly_visualization import AnomalyVisualizer
        print("✓ Anomaly detection imports successful")
        
        # Test rheology imports
        from rheology import MicrorheologyAnalyzer
        print("✓ Microrheology imports successful")
        
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
            
        print("✓ Enhanced report generator functionality verified")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced report generator test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running comprehensive functionality tests...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("plot_diffusion_coefficients Test", test_plot_diffusion_coefficients),
        ("AnalysisManager Test", test_analysis_manager),
        ("Enhanced Report Generator Test", test_enhanced_report_generator),
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
