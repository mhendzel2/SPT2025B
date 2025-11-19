"""
Quick GUI Integration Test for Percolation Analysis
===================================================

This script verifies that the new percolation analysis methods are 
accessible through the Enhanced Report Generator GUI.

Run this to verify integration:
    python test_gui_integration.py
"""

import sys
import pandas as pd
import numpy as np

def test_gui_integration():
    """Test that percolation methods are available in GUI."""
    print("="*70)
    print("  GUI Integration Test - Percolation Analysis Suite")
    print("="*70)
    
    # Test 1: Import EnhancedSPTReportGenerator
    print("\n[TEST 1] Importing EnhancedSPTReportGenerator...")
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        print("✅ Successfully imported EnhancedSPTReportGenerator")
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test 2: Initialize generator
    print("\n[TEST 2] Initializing report generator...")
    try:
        generator = EnhancedSPTReportGenerator()
        print("✅ Successfully initialized report generator")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Test 3: Check available analyses
    print("\n[TEST 3] Checking available percolation analyses...")
    percolation_methods = [
        'fractal_dimension',
        'connectivity_network',
        'anomalous_exponent_map',
        'obstacle_density'
    ]
    
    found_methods = []
    for method in percolation_methods:
        if method in generator.available_analyses:
            info = generator.available_analyses[method]
            print(f"✅ {method}")
            print(f"   Name: {info['name']}")
            print(f"   Category: {info['category']}")
            found_methods.append(method)
        else:
            print(f"❌ {method} NOT FOUND")
    
    if len(found_methods) != len(percolation_methods):
        print(f"\n⚠️  Warning: Only {len(found_methods)}/{len(percolation_methods)} methods found")
        return False
    
    # Test 4: Check that methods have implementations
    print("\n[TEST 4] Checking method implementations...")
    implementation_checks = []
    
    for method in percolation_methods:
        info = generator.available_analyses[method]
        has_function = info.get('function') is not None
        has_viz = info.get('visualization') is not None
        
        status = "✅" if (has_function and has_viz) else "❌"
        print(f"{status} {method}: function={has_function}, visualization={has_viz}")
        implementation_checks.append(has_function and has_viz)
    
    if not all(implementation_checks):
        print("\n⚠️  Warning: Some methods missing implementations")
        return False
    
    # Test 5: Test with synthetic data
    print("\n[TEST 5] Testing with synthetic data...")
    try:
        # Generate simple test data
        np.random.seed(42)
        n_tracks = 3
        n_steps = 30
        
        tracks = []
        for track_id in range(n_tracks):
            x = np.cumsum(np.random.randn(n_steps)) * 0.1
            y = np.cumsum(np.random.randn(n_steps)) * 0.1
            
            for i in range(n_steps):
                tracks.append({
                    'track_id': track_id,
                    'frame': i,
                    'x': x[i],
                    'y': y[i]
                })
        
        tracks_df = pd.DataFrame(tracks)
        current_units = {'pixel_size': 1.0, 'frame_interval': 0.05}
        
        print(f"   Generated {len(tracks_df)} data points for {n_tracks} tracks")
        
        # Test each analysis method
        test_results = []
        
        # Fractal dimension
        try:
            result = generator._analyze_fractal_dimension(tracks_df, current_units)
            success = result.get('success', False)
            print(f"   {'✅' if success else '❌'} Fractal dimension: {result.get('mean_fractal_dim', 'N/A')}")
            test_results.append(success)
        except Exception as e:
            print(f"   ❌ Fractal dimension failed: {e}")
            test_results.append(False)
        
        # Connectivity network
        try:
            result = generator._analyze_connectivity_network(tracks_df, current_units)
            success = result.get('success', False)
            print(f"   {'✅' if success else '❌'} Connectivity: Percolates={result.get('percolates', 'N/A')}")
            test_results.append(success)
        except Exception as e:
            print(f"   ❌ Connectivity network failed: {e}")
            test_results.append(False)
        
        # Anomalous exponent (just check it returns success)
        try:
            result = generator._analyze_anomalous_exponent(tracks_df, current_units)
            success = result.get('success', False)
            print(f"   {'✅' if success else '❌'} Anomalous exponent map")
            test_results.append(success)
        except Exception as e:
            print(f"   ❌ Anomalous exponent failed: {e}")
            test_results.append(False)
        
        # Obstacle density
        try:
            result = generator._analyze_obstacle_density(tracks_df, current_units)
            success = result.get('success', False)
            phi = result.get('obstacle_fraction', 0)
            print(f"   {'✅' if success else '❌'} Obstacle density: φ={phi:.1%}")
            test_results.append(success)
        except Exception as e:
            print(f"   ❌ Obstacle density failed: {e}")
            test_results.append(False)
        
        if not all(test_results):
            print(f"\n⚠️  Warning: {sum(test_results)}/{len(test_results)} analyses passed")
            return False
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Check category organization
    print("\n[TEST 6] Checking category organization...")
    categories = {}
    for key, info in generator.available_analyses.items():
        cat = info.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(key)
    
    if 'Percolation Analysis' in categories:
        print(f"✅ 'Percolation Analysis' category found with {len(categories['Percolation Analysis'])} methods:")
        for method in categories['Percolation Analysis']:
            print(f"   - {method}")
    else:
        print("❌ 'Percolation Analysis' category not found")
        print(f"   Available categories: {list(categories.keys())}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ ALL TESTS PASSED - Percolation Analysis GUI Integration Complete!")
    print("="*70)
    print("\nThe following methods are now accessible in the GUI:")
    print("  1. Fractal Dimension Analysis")
    print("  2. Spatial Connectivity Network")
    print("  3. Anomalous Exponent Map α(x,y)")
    print("  4. Obstacle Density Inference")
    print("\nTo use in the GUI:")
    print("  1. Run: streamlit run app.py")
    print("  2. Load tracking data")
    print("  3. Go to 'Enhanced Report Generator' tab")
    print("  4. Look for 'Percolation Analysis' category")
    print("  5. Select desired analyses and generate report")
    print()
    
    return True

if __name__ == '__main__':
    try:
        success = test_gui_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
