"""
Quick Sample Data Validation Test
Tests import, analysis, visualization, and report generation for each subfolder.
Date: 2025-10-06
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("SPT2025B - Sample Data Comprehensive Test")
print("="*80)
print()

# Test files
test_files = {
    "C2C12_40nm_SC35": "sample data/C2C12_40nm_SC35/Cropped_spots_cell1.csv",
    "U2OS_40_SC35": "sample data/U2OS_40_SC35/Cropped_spots_cell2.csv",
    "U2OS_MS2": "sample data/U2OS_MS2/Cell1_spots.csv"
}

test_results = {}

for name, filepath in test_files.items():
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"File: {filepath}")
    print('='*80)
    
    results = {
        'import': False,
        'format': False,
        'analysis': False,
        'visualization': False,
        'report': False,
        'errors': []
    }
    
    # TEST 1: Import
    print("\n[1] Testing Data Import...")
    try:
        df = pd.read_csv(filepath)
        print(f"    ✓ Loaded {len(df)} rows")
        print(f"    ✓ Columns: {len(df.columns)}")
        results['import'] = True
        
        # Check for required columns
        required = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID']
        found = [col for col in required if col in df.columns]
        print(f"    ✓ Required columns found: {len(found)}/{len(required)}")
        
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        results['errors'].append(('import', str(e)))
        test_results[name] = results
        continue
    
    # TEST 2: Format
    print("\n[2] Testing Data Formatting...")
    try:
        from data_loader import format_track_data
        tracks_df = format_track_data(df)
        print(f"    ✓ Formatted to {len(tracks_df)} rows")
        
        # Verify columns
        expected = ['track_id', 'frame', 'x', 'y']
        if all(col in tracks_df.columns for col in expected):
            print(f"    ✓ Standard columns present: {expected}")
            results['format'] = True
            
            # Track statistics
            n_tracks = tracks_df['track_id'].nunique()
            mean_length = tracks_df.groupby('track_id').size().mean()
            print(f"    ✓ Tracks: {n_tracks} (mean length: {mean_length:.1f} frames)")
        else:
            print(f"    ✗ Missing columns after formatting")
            results['errors'].append(('format', 'Missing standard columns'))
            
    except Exception as e:
        print(f"    ✗ Formatting failed: {e}")
        results['errors'].append(('format', str(e)))
        test_results[name] = results
        continue
    
    # TEST 3: Analysis
    print("\n[3] Testing Analysis Functions...")
    try:
        from analysis import calculate_msd, classify_motion
        
        # MSD
        msd_result = calculate_msd(tracks_df, pixel_size=0.1, frame_interval=0.1)
        if msd_result.get('success'):
            D = msd_result.get('diffusion_coefficient', 0)
            print(f"    ✓ MSD calculated (D={D:.4f} μm²/s)")
        else:
            print(f"    ⚠ MSD warning: {msd_result.get('error', 'Unknown')}")
        
        # Motion classification
        motion_result = classify_motion(tracks_df, pixel_size=0.1, frame_interval=0.1)
        if motion_result.get('success'):
            alpha = motion_result.get('alpha', 0)
            print(f"    ✓ Motion classified (α={alpha:.3f})")
            results['analysis'] = True
        else:
            print(f"    ⚠ Motion classification warning: {motion_result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"    ✗ Analysis failed: {e}")
        results['errors'].append(('analysis', str(e)))
    
    # TEST 4: Visualization
    print("\n[4] Testing Visualization...")
    try:
        import plotly.graph_objects as go
        
        # Create simple trajectory plot
        fig = go.Figure()
        for track_id in tracks_df['track_id'].unique()[:3]:
            track = tracks_df[tracks_df['track_id'] == track_id]
            fig.add_trace(go.Scatter(x=track['x'], y=track['y'], mode='lines', name=f'Track {track_id}'))
        
        print(f"    ✓ Trajectory plot created (3 tracks)")
        results['visualization'] = True
        
    except Exception as e:
        print(f"    ✗ Visualization failed: {e}")
        results['errors'].append(('visualization', str(e)))
    
    # TEST 5: Report Generation
    print("\n[5] Testing Report Generation...")
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        generator = EnhancedSPTReportGenerator(
            tracks_df=tracks_df,
            project_name=f"Test_{name}",
            metadata={'test': True}
        )
        print(f"    ✓ Generator initialized")
        
        # Generate basic report
        report = generator.generate_batch_report(
            selected_analyses=['basic_statistics', 'diffusion_analysis'],
            current_units={'pixel_size': 0.1, 'frame_interval': 0.1}
        )
        
        success_count = sum(1 for a in report.get('analyses', {}).values() if a.get('success'))
        total_count = len(report.get('analyses', {}))
        print(f"    ✓ Report generated ({success_count}/{total_count} analyses succeeded)")
        
        if success_count > 0:
            results['report'] = True
        
    except Exception as e:
        print(f"    ✗ Report generation failed: {e}")
        results['errors'].append(('report', str(e)))
    
    test_results[name] = results

# FINAL SUMMARY
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

total_tests = 0
passed_tests = 0

for name, results in test_results.items():
    print(f"\n{name}:")
    tests = ['import', 'format', 'analysis', 'visualization', 'report']
    for test in tests:
        total_tests += 1
        status = "✓" if results[test] else "✗"
        print(f"  {status} {test.capitalize()}")
        if results[test]:
            passed_tests += 1
    
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")

success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
print(f"\n{'='*80}")
print(f"Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
print("="*80)

if success_rate >= 80:
    print("\n✓ RESULT: PASSED")
    sys.exit(0)
elif success_rate >= 60:
    print("\n⚠ RESULT: PARTIAL PASS")
    sys.exit(1)
else:
    print("\n✗ RESULT: FAILED")
    sys.exit(2)
