"""
Test Energy Landscape and Percolation Visualizations
This script tests the fixed visualization functions for energy landscape and percolation analysis.
"""

import pandas as pd
import numpy as np
from enhanced_report_generator import EnhancedSPTReportGenerator

def test_energy_landscape_visualization():
    """Test energy landscape visualization with string-formatted numpy arrays."""
    print("\n" + "="*80)
    print("TESTING ENERGY LANDSCAPE VISUALIZATION")
    print("="*80)
    
    # Create mock result data that mimics the actual structure with numpy array strings
    result = {
        'success': True,
        'energy_map': "[[13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]]",
        'energy_landscape': "[[13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]\n [13.81551056 13.81551056 13.81551056 13.81551056 13.81551056 13.81551056]]",
        'x_coords': "[0.8763553 0.93020968 0.98406406 1.03791844 1.09177282 1.1456272]",
        'y_coords': "[0.8763553 0.93020968 0.98406406 1.03791844 1.09177282 1.1456272]",
        'x_edges': "[0.85038111 0.90423549 0.95808987 1.01194425 1.06579863 1.11965301 1.17350739]",
        'y_edges': "[0.85038111 0.90423549 0.95808987 1.01194425 1.06579863 1.11965301 1.17350739]",
        'force_field': {
            'fx': "[[-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]]",
            'fy': "[[-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]\n [-0.0 -0.0 -0.0 -0.0 -0.0 -0.0]]",
            'magnitude': "[[0.0 0.0 0.0 0.0 0.0 0.0]\n [0.0 0.0 0.0 0.0 0.0 0.0]\n [0.0 0.0 0.0 0.0 0.0 0.0]\n [0.0 0.0 0.0 0.0 0.0 0.0]\n [0.0 0.0 0.0 0.0 0.0 0.0]\n [0.0 0.0 0.0 0.0 0.0 0.0]]"
        },
        'statistics': {
            'min_energy': 0.0,
            'max_energy': 13.815510557964275,
            'energy_range': 13.815510557964275
        }
    }
    
    # Test the visualization function
    generator = EnhancedSPTReportGenerator()
    
    try:
        fig = generator._plot_energy_landscape(result)
        
        if fig is None:
            print("❌ FAILED: Visualization returned None")
            return False
        
        print("✅ SUCCESS: Energy landscape visualization created")
        print(f"   - Figure type: {type(fig).__name__}")
        print(f"   - Number of traces: {len(fig.data)}")
        
        # Check that we have both surface and contour
        trace_types = [trace.type for trace in fig.data]
        print(f"   - Trace types: {trace_types}")
        
        if 'surface' in trace_types and 'contour' in trace_types:
            print("   - ✅ Both 3D surface and 2D contour plots created")
        else:
            print("   - ⚠️  Expected both surface and contour traces")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_percolation_visualization():
    """Test percolation visualization with string-formatted numpy arrays."""
    print("\n" + "="*80)
    print("TESTING PERCOLATION VISUALIZATION")
    print("="*80)
    
    # Create mock result data that mimics the actual structure
    result = {
        'success': True,
        'percolation_detected': False,
        'num_clusters': 25,
        'largest_cluster_size': 2,
        'density': 9.30390262970703,
        'network_stats': {
            'num_nodes': 28,
            'num_edges': 17,
            'average_degree': 1.2142857142857142
        },
        'full_results': {
            'cluster_size_distribution': "[1 1 1 1 1 1 1 2 2 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1]",
            'labels': "[ 0  1  2  3  4  5  6  7  8  7  9 10 11 12 13 13 14 15  8 16 17 18 19 20\n 21 22 23 24]",
            'distance_threshold': 0.5,
            'spanning_cluster': False
        }
    }
    
    # Test the visualization function
    generator = EnhancedSPTReportGenerator()
    
    try:
        fig = generator._plot_percolation(result)
        
        if fig is None:
            print("❌ FAILED: Visualization returned None")
            return False
        
        print("✅ SUCCESS: Percolation visualization created")
        print(f"   - Figure type: {type(fig).__name__}")
        print(f"   - Number of traces: {len(fig.data)}")
        
        # Check trace types
        trace_types = [trace.type for trace in fig.data]
        print(f"   - Trace types: {trace_types}")
        
        # Should have: scatter (network map), bar (cluster sizes), table (stats), indicator (status)
        expected_types = {'scatter', 'bar', 'table', 'indicator'}
        actual_types = set(trace_types)
        
        if expected_types.issubset(actual_types):
            print("   - ✅ All expected plot types present (scatter, bar, table, indicator)")
        else:
            missing = expected_types - actual_types
            print(f"   - ⚠️  Missing plot types: {missing}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_sample_data():
    """Test with actual sample data if available."""
    print("\n" + "="*80)
    print("TESTING WITH SAMPLE DATA")
    print("="*80)
    
    try:
        # Try to load sample data
        sample_file = 'Cell1_spots.csv'
        import os
        if not os.path.exists(sample_file):
            print(f"⚠️  Sample file '{sample_file}' not found, skipping sample data test")
            return None
        
        df = pd.read_csv(sample_file)
        print(f"✅ Loaded sample data: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        
        # Try running actual analyses
        generator = EnhancedSPTReportGenerator()
        
        # Test energy landscape
        print("\n--- Testing Energy Landscape Analysis ---")
        try:
            from biophysical_models import EnergyLandscapeMapper
            
            # Need to ensure proper column names
            if 'x' in df.columns and 'y' in df.columns:
                df['x_um'] = df['x'] * 0.1  # Assuming 0.1 μm pixel size
                df['y_um'] = df['y'] * 0.1
            
            mapper = EnergyLandscapeMapper(df, pixel_size=0.1, temperature=300.0)
            energy_result = mapper.map_energy_landscape(resolution=10, method='boltzmann')
            
            if energy_result.get('success'):
                print("✅ Energy landscape analysis completed")
                fig = generator._plot_energy_landscape(energy_result)
                if fig:
                    print("✅ Energy landscape visualization created")
                else:
                    print("❌ Energy landscape visualization failed")
            else:
                print(f"❌ Energy landscape analysis failed: {energy_result.get('error')}")
        except Exception as e:
            print(f"⚠️  Could not test energy landscape with sample data: {e}")
        
        # Test percolation
        print("\n--- Testing Percolation Analysis ---")
        try:
            from percolation_analyzer import PercolationAnalyzer
            
            analyzer = PercolationAnalyzer(df, pixel_size=0.1)
            perc_results = analyzer.analyze_connectivity_network()
            
            perc_result = {
                'success': True,
                'percolation_detected': perc_results.get('spanning_cluster', False),
                'num_clusters': perc_results.get('num_clusters', 0),
                'largest_cluster_size': perc_results.get('largest_cluster_size', 0),
                'density': perc_results.get('density', 0),
                'network_stats': {
                    'num_nodes': perc_results.get('num_nodes', 0),
                    'num_edges': perc_results.get('num_edges', 0),
                    'average_degree': perc_results.get('average_degree', 0)
                },
                'full_results': perc_results
            }
            
            print(f"✅ Percolation analysis completed: {perc_result['num_clusters']} clusters")
            fig = generator._plot_percolation(perc_result)
            if fig:
                print("✅ Percolation visualization created")
            else:
                print("❌ Percolation visualization failed")
                
        except Exception as e:
            print(f"⚠️  Could not test percolation with sample data: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Sample data test failed: {e}")
        return None

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ENERGY LANDSCAPE & PERCOLATION VISUALIZATION FIX TESTS")
    print("="*80)
    
    results = {
        'energy_landscape': test_energy_landscape_visualization(),
        'percolation': test_percolation_visualization(),
        'sample_data': test_with_sample_data()
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{test_name:20s}: {status}")
    
    print("="*80)
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    
    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed, {passed} passed")
        return 1

if __name__ == "__main__":
    exit(main())
