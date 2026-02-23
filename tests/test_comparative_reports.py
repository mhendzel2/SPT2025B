"""
Test comparative report generation for pooled condition data.
"""

import pandas as pd
import numpy as np
from enhanced_report_generator import EnhancedSPTReportGenerator

def create_synthetic_condition(n_tracks=10, n_frames=50, mean_speed=1.0, noise=0.5):
    """Create synthetic track data for a condition."""
    data = []
    for track_id in range(n_tracks):
        x, y = 0.0, 0.0
        for frame in range(n_frames):
            # Random walk with drift
            dx = np.random.normal(mean_speed, noise)
            dy = np.random.normal(mean_speed, noise)
            x += dx
            y += dy
            
            data.append({
                'track_id': track_id,
                'frame': frame,
                'x': x,
                'y': y
            })
    
    return pd.DataFrame(data)

def test_condition_reports():
    """Test generating reports for multiple conditions."""
    print("\n" + "="*70)
    print("TEST: Condition Report Generation")
    print("="*70)
    
    # Create synthetic data for 3 conditions
    condition_datasets = {
        'Control': create_synthetic_condition(n_tracks=10, n_frames=50, mean_speed=0.5, noise=0.3),
        'Treatment A': create_synthetic_condition(n_tracks=12, n_frames=50, mean_speed=1.0, noise=0.4),
        'Treatment B': create_synthetic_condition(n_tracks=8, n_frames=50, mean_speed=1.5, noise=0.5)
    }
    
    print(f"\nCreated {len(condition_datasets)} conditions:")
    for name, df in condition_datasets.items():
        print(f"  - {name}: {df['track_id'].nunique()} tracks, {len(df)} points")
    
    # Create generator
    generator = EnhancedSPTReportGenerator(pd.DataFrame(), pixel_size=0.1, frame_interval=0.1)
    
    # Select analyses
    selected_analyses = ['basic_statistics', 'diffusion_analysis', 'motion_classification']
    
    print(f"\nRunning {len(selected_analyses)} analyses on each condition...")
    
    # Generate reports
    try:
        results = generator.generate_condition_reports(
            condition_datasets,
            selected_analyses,
            pixel_size=0.1,
            frame_interval=0.1
        )
        
        print("\n✓ Report generation completed")
        print(f"  Success: {results.get('success', False)}")
        print(f"  Conditions processed: {results.get('n_conditions', 0)}")
        
        # Check individual condition results
        print("\n📊 Individual Condition Results:")
        for cond_name, cond_result in results['conditions'].items():
            success = cond_result.get('success', False)
            status = "✓" if success else "✗"
            print(f"  {status} {cond_name}")
            if success:
                n_analyses = len(cond_result.get('analysis_results', {}))
                n_figures = len(cond_result.get('figures', {}))
                print(f"    - Analyses: {n_analyses}")
                print(f"    - Figures: {n_figures}")
                
                # Show cache stats
                cache_stats = cond_result.get('cache_stats', {})
                if cache_stats:
                    print(f"    - Cache hits: {cache_stats.get('hits', 0)}, misses: {cache_stats.get('misses', 0)}")
        
        # Check comparison results
        if 'comparisons' in results and results['comparisons']:
            print("\n🔬 Comparison Analysis:")
            comparisons = results['comparisons']
            
            if comparisons.get('success', False):
                print("  ✓ Statistical comparisons completed")
                
                # Show metrics
                if 'metrics' in comparisons:
                    print("\n  Summary Metrics:")
                    for cond_name, metrics in comparisons['metrics'].items():
                        print(f"    {cond_name}:")
                        print(f"      - Mean track length: {metrics.get('mean_track_length', 0):.2f} frames")
                        print(f"      - Mean displacement: {metrics.get('mean_displacement', 0):.4f} μm")
                        print(f"      - Mean velocity: {metrics.get('mean_velocity', 0):.4f} μm/s")
                
                # Show statistical tests
                if 'statistical_tests' in comparisons and comparisons['statistical_tests']:
                    print("\n  Pairwise Statistical Tests:")
                    for comparison, tests in comparisons['statistical_tests'].items():
                        print(f"    {comparison}:")
                        for metric, test_results in tests.items():
                            if 't_test' in test_results:
                                p_val = test_results['t_test']['p_value']
                                significant = test_results.get('significant', False)
                                sig_mark = "*" if significant else "ns"
                                print(f"      - {metric}: p={p_val:.4f} ({sig_mark})")
                
                # Check figures
                if 'figures' in comparisons:
                    n_figs = len([f for f in comparisons['figures'].values() if f is not None])
                    print(f"\n  ✓ Generated {n_figs} comparison figures")
            else:
                print(f"  ✗ Comparison failed: {comparisons.get('error', 'Unknown error')}")
        
        print("\n" + "="*70)
        print("✓ TEST PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pairwise_comparisons():
    """Test that all pairwise comparisons are performed."""
    print("\n" + "="*70)
    print("TEST: Pairwise Comparisons")
    print("="*70)
    
    # Create 4 conditions for multiple comparisons
    condition_datasets = {
        'A': create_synthetic_condition(n_tracks=8, n_frames=30, mean_speed=0.5),
        'B': create_synthetic_condition(n_tracks=8, n_frames=30, mean_speed=1.0),
        'C': create_synthetic_condition(n_tracks=8, n_frames=30, mean_speed=1.5),
        'D': create_synthetic_condition(n_tracks=8, n_frames=30, mean_speed=2.0),
    }
    
    print(f"\nCreated {len(condition_datasets)} conditions")
    expected_comparisons = len(condition_datasets) * (len(condition_datasets) - 1) // 2
    print(f"Expected {expected_comparisons} pairwise comparisons")
    
    generator = EnhancedSPTReportGenerator(pd.DataFrame(), pixel_size=0.1, frame_interval=0.1)
    
    results = generator.generate_condition_reports(
        condition_datasets,
        ['basic_statistics'],
        pixel_size=0.1,
        frame_interval=0.1
    )
    
    if 'comparisons' in results and 'statistical_tests' in results['comparisons']:
        n_comparisons = len(results['comparisons']['statistical_tests'])
        print(f"Generated {n_comparisons} comparisons")
        
        if n_comparisons == expected_comparisons:
            print("✓ Correct number of pairwise comparisons")
            
            # List comparisons
            print("\nComparisons performed:")
            for comparison_key in results['comparisons']['statistical_tests'].keys():
                print(f"  - {comparison_key}")
            
            print("\n" + "="*70)
            print("✓ TEST PASSED")
            print("="*70)
            return True
        else:
            print(f"❌ Expected {expected_comparisons} comparisons, got {n_comparisons}")
            return False
    else:
        print("❌ No statistical tests found in results")
        return False

def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "="*70)
    print("COMPARATIVE REPORT GENERATION TEST SUITE")
    print("="*70)
    
    tests = [
        test_condition_reports,
        test_pairwise_comparisons
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
