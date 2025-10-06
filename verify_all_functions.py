#!/usr/bin/env python3
"""
Quick Verification Script for SPT2025B Functions and Report Generation

This script provides a fast verification that all analysis functions and
the automated report generation system are properly implemented and operational.

Usage:
    python verify_all_functions.py

Expected Output:
    - List of all 16 analysis functions
    - Verification that each has an implementation and visualization
    - Test of batch report generation
    - Summary of verification results
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime


def create_test_data(n_tracks=5, n_points_range=(20, 40)):
    """Create minimal test data for verification."""
    np.random.seed(42)
    tracks = []
    
    for track_id in range(1, n_tracks + 1):
        n_points = np.random.randint(*n_points_range)
        x_start, y_start = np.random.uniform(10, 90, 2)
        
        # Simple random walk
        x_steps = np.random.normal(0, 1.5, n_points)
        y_steps = np.random.normal(0, 1.5, n_points)
        x_positions = np.cumsum(np.concatenate([[x_start], x_steps]))
        y_positions = np.cumsum(np.concatenate([[y_start], y_steps]))
        
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            tracks.append({
                'track_id': track_id,
                'frame': i,
                'x': x,
                'y': y
            })
    
    return pd.DataFrame(tracks)


def verify_functions():
    """Verify all analysis functions are present and callable."""
    print("=" * 70)
    print("VERIFICATION: Analysis Functions")
    print("=" * 70)
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
    except ImportError as e:
        print(f"✗ FAILED: Cannot import EnhancedSPTReportGenerator: {e}")
        return False
    
    generator = EnhancedSPTReportGenerator()
    analyses = generator.available_analyses
    
    print(f"\nTotal Analysis Functions Found: {len(analyses)}")
    print("-" * 70)
    
    all_valid = True
    categories = {}
    
    for key, info in analyses.items():
        # Group by category
        category = info.get('category', 'Unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(info['name'])
        
        # Check function and visualization exist
        has_function = info.get('function') is not None
        has_viz = info.get('visualization') is not None
        
        if has_function and has_viz:
            status = "✓"
        else:
            status = "✗"
            all_valid = False
        
        print(f"{status} {info['name']:<45} [{category}]")
    
    print("\n" + "-" * 70)
    print("Analysis Functions by Category:")
    for category, names in sorted(categories.items()):
        print(f"  {category}: {len(names)} functions")
    
    return all_valid


def verify_report_generation():
    """Verify batch report generation works."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Automated Report Generation")
    print("=" * 70)
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create test data
        tracks_df = create_test_data()
        print(f"\n✓ Created test data: {len(tracks_df)} points, "
              f"{tracks_df['track_id'].nunique()} tracks")
        
        # Initialize generator
        generator = EnhancedSPTReportGenerator()
        print("✓ Initialized report generator")
        
        # Test batch report with subset of analyses
        selected = ['basic_statistics', 'diffusion_analysis', 'motion_classification']
        print(f"\n  Testing batch report with {len(selected)} analyses...")
        
        result = generator.generate_batch_report(
            tracks_df, 
            selected, 
            'Verification Test'
        )
        
        # Check results
        success = result.get('success', False)
        n_analyses = len(result.get('analysis_results', {}))
        n_figures = len(result.get('figures', {}))
        
        print(f"  - Success: {success}")
        print(f"  - Analyses completed: {n_analyses}/{len(selected)}")
        print(f"  - Figures generated: {n_figures}/{len(selected)}")
        
        if success and n_analyses == len(selected):
            print("\n✓ Batch report generation: PASSED")
            return True
        else:
            print("\n✗ Batch report generation: FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_export_capabilities():
    """Verify export functions are available."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Export Capabilities")
    print("=" * 70)
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        generator = EnhancedSPTReportGenerator()
        
        # Check for export methods
        export_methods = {
            'HTML': '_export_html_report',
            'CSV': '_generate_csv_summary',
            'Report Data': '_generate_report_data',
            'Display': '_display_generated_report'
        }
        
        all_present = True
        for name, method in export_methods.items():
            if hasattr(generator, method):
                print(f"✓ {name} export available")
            else:
                print(f"✗ {name} export missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"✗ Export verification failed: {e}")
        return False


def main():
    """Run all verifications."""
    print("\n" + "=" * 70)
    print("SPT2025B - Complete Verification Script")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'Functions': verify_functions(),
        'Report Generation': verify_report_generation(),
        'Export Capabilities': verify_export_capabilities()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED")
        print("\nAll analysis functions and automated report generation")
        print("are properly implemented and operational.")
        exit_code = 0
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("\nPlease review the failures above.")
        exit_code = 1
    
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
