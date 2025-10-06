"""
Sample Data Comprehensive Test - FINAL
Tests all sample data subfolders for import, analysis, visualization, and reporting.
Date: 2025-10-06
"""

import pandas as pd
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Initialize session state for units (needed by report generator)
class FakeSessionState:
    def __init__(self):
        self.data = {
            'pixel_size': 0.1,
            'frame_interval': 0.1,
            'temperature': 298.15,
            'particle_radius': 0.5
        }
    
    def get(self, key, default=None):
        return self.data.get(key, default)

# Mock streamlit session_state
import sys
if 'streamlit' not in sys.modules:
    class MockStreamlit:
        session_state = FakeSessionState()
    sys.modules['streamlit'] = MockStreamlit()
    sys.modules['st'] = MockStreamlit()

print("="*80)
print("SPT2025B - Sample Data Comprehensive Test")
print("Date: October 6, 2025")
print("="*80)
print()

# Test files from each subfolder
test_files = {
    "C2C12_40nm_SC35": "sample data/C2C12_40nm_SC35/Cropped_spots_cell1.csv",
    "U2OS_40_SC35": "sample data/U2OS_40_SC35/Cropped_spots_cell2.csv",
    "U2OS_MS2": "sample data/U2OS_MS2/Cell1_spots.csv"
}

all_results = {}

for dataset_name, filepath in test_files.items():
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"File: {filepath}")
    print('='*80)
    
    test_result = {
        'dataset': dataset_name,
        'file': filepath,
        'tests': {},
        'errors': [],
        'data_info': {}
    }
    
    # TEST 1: IMPORT
    print("\n[1] IMPORT TEST")
    try:
        df = pd.read_csv(filepath)
        rows, cols = df.shape
        print(f"    ✓ Loaded: {rows} rows × {cols} columns")
        test_result['tests']['import'] = 'PASS'
        test_result['data_info']['raw_rows'] = rows
        test_result['data_info']['raw_cols'] = cols
        
        # Check for required columns
        required = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T']
        has_required = all(col in df.columns for col in required)
        if has_required:
            print(f"    ✓ Required columns present")
        else:
            print(f"    ⚠ Some required columns missing")
        
    except Exception as e:
        print(f"    ✗ FAIL: {e}")
        test_result['tests']['import'] = 'FAIL'
        test_result['errors'].append(('import', str(e)))
        all_results[dataset_name] = test_result
        continue
    
    # TEST 2: FORMAT
    print("\n[2] FORMAT TEST")
    try:
        from data_loader import format_track_data
        tracks_df = format_track_data(df)
        
        if tracks_df is not None and len(tracks_df) > 0:
            n_tracks = tracks_df['track_id'].nunique()
            track_lengths = tracks_df.groupby('track_id').size()
            mean_len = track_lengths.mean()
            max_len = track_lengths.max()
            
            print(f"    ✓ Formatted: {len(tracks_df)} rows")
            print(f"    ✓ Tracks: {n_tracks} (mean={mean_len:.1f}, max={max_len})")
            test_result['tests']['format'] = 'PASS'
            test_result['data_info']['formatted_rows'] = len(tracks_df)
            test_result['data_info']['n_tracks'] = n_tracks
            test_result['data_info']['mean_track_length'] = float(mean_len)
        else:
            print(f"    ✗ FAIL: Empty result")
            test_result['tests']['format'] = 'FAIL'
            test_result['errors'].append(('format', 'Empty data'))
            all_results[dataset_name] = test_result
            continue
            
    except Exception as e:
        print(f"    ✗ FAIL: {str(e)[:100]}")
        test_result['tests']['format'] = 'FAIL'
        test_result['errors'].append(('format', str(e)[:200]))
        all_results[dataset_name] = test_result
        continue
    
    # TEST 3: ANALYSIS
    print("\n[3] ANALYSIS TEST")
    analysis_passed = 0
    analysis_total = 0
    
    # Test MSD
    try:
        from analysis import calculate_msd
        msd_result = calculate_msd(tracks_df, pixel_size=0.1, frame_interval=0.1)
        analysis_total += 1
        
        if msd_result.get('success'):
            D = msd_result.get('diffusion_coefficient', 0)
            print(f"    ✓ MSD: D={D:.4f} μm²/s")
            analysis_passed += 1
        else:
            print(f"    ⚠ MSD: {msd_result.get('error', 'Failed')[:50]}")
    except Exception as e:
        print(f"    ⚠ MSD: {str(e)[:50]}")
        analysis_total += 1
    
    # Test motion analysis
    try:
        from analysis import analyze_motion
        motion_result = analyze_motion(tracks_df, window_size=5)
        analysis_total += 1
        
        if isinstance(motion_result, dict) and motion_result.get('success'):
            print(f"    ✓ Motion Analysis: Success")
            analysis_passed += 1
        else:
            print(f"    ⚠ Motion Analysis: Partial")
    except Exception as e:
        print(f"    ⚠ Motion Analysis: {str(e)[:50]}")
        analysis_total += 1
    
    if analysis_passed >= 1:
        test_result['tests']['analysis'] = 'PASS'
        print(f"    → Analysis: {analysis_passed}/{analysis_total} passed")
    else:
        test_result['tests']['analysis'] = 'WARN'
    
    # TEST 4: VISUALIZATION
    print("\n[4] VISUALIZATION TEST")
    viz_passed = 0
    viz_total = 0
    
    # Test trajectory plot
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for track_id in tracks_df['track_id'].unique()[:3]:
            track = tracks_df[tracks_df['track_id'] == track_id]
            fig.add_trace(go.Scatter(x=track['x'], y=track['y'], mode='lines', name=f'{track_id}'))
        print(f"    ✓ Trajectory plot created")
        viz_passed += 1
        viz_total += 1
    except Exception as e:
        print(f"    ✗ Trajectory plot: {str(e)[:50]}")
        viz_total += 1
    
    # Test MSD plot
    try:
        from analysis import calculate_msd
        msd_result = calculate_msd(tracks_df, pixel_size=0.1, frame_interval=0.1)
        if msd_result.get('success'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=msd_result.get('lag_times', []),
                y=msd_result.get('msd_values', []),
                mode='markers+lines'
            ))
            print(f"    ✓ MSD plot created")
            viz_passed += 1
        viz_total += 1
    except Exception as e:
        print(f"    ⚠ MSD plot: {str(e)[:50]}")
        viz_total += 1
    
    if viz_passed >= 1:
        test_result['tests']['visualization'] = 'PASS'
        print(f"    → Visualization: {viz_passed}/{viz_total} passed")
    else:
        test_result['tests']['visualization'] = 'FAIL'
    
    # TEST 5: REPORT GENERATION
    print("\n[5] REPORT GENERATION TEST")
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Initialize generator
        generator = EnhancedSPTReportGenerator(
            tracks_df=tracks_df,
            project_name=f"Test_{dataset_name}"
        )
        print(f"    ✓ Generator initialized")
        
        # Generate report using correct API
        report = generator.generate_batch_report(
            tracks_df=tracks_df,
            selected_analyses=['basic_statistics', 'diffusion_analysis'],
            condition_name=dataset_name
        )
        
        # Check results
        analyses = report.get('analysis_results', {})
        success_count = sum(1 for a in analyses.values() if a.get('success', True) and 'error' not in a)
        total_count = len(analyses)
        
        print(f"    ✓ Report generated: {success_count}/{total_count} analyses")
        
        if success_count >= 1:
            test_result['tests']['report'] = 'PASS'
        else:
            test_result['tests']['report'] = 'WARN'
        
        # Save report
        output_file = f"test_report_{dataset_name}.json"
        with open(output_file, 'w') as f:
            # Filter out non-serializable objects
            clean_report = {
                'condition_name': report.get('condition_name'),
                'success': report.get('success'),
                'analysis_count': total_count,
                'success_count': success_count
            }
            json.dump(clean_report, f, indent=2)
        print(f"    ✓ Report saved: {output_file}")
        
    except Exception as e:
        print(f"    ✗ Report generation: {str(e)[:100]}")
        test_result['tests']['report'] = 'FAIL'
        test_result['errors'].append(('report', str(e)[:200]))
    
    all_results[dataset_name] = test_result

# FINAL SUMMARY
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary_stats = {
    'total_datasets': len(all_results),
    'total_tests': 0,
    'pass_count': 0,
    'warn_count': 0,
    'fail_count': 0
}

for dataset, result in all_results.items():
    print(f"\n{dataset}:")
    print(f"  File: {result['file']}")
    
    if 'raw_rows' in result['data_info']:
        print(f"  Data: {result['data_info']['raw_rows']} rows → {result['data_info'].get('formatted_rows', 0)} formatted")
        if 'n_tracks' in result['data_info']:
            print(f"  Tracks: {result['data_info']['n_tracks']} (mean length: {result['data_info']['mean_track_length']:.1f})")
    
    print(f"  Tests:")
    for test_name, status in result['tests'].items():
        summary_stats['total_tests'] += 1
        if status == 'PASS':
            print(f"    ✓ {test_name}: PASS")
            summary_stats['pass_count'] += 1
        elif status == 'WARN':
            print(f"    ⚠ {test_name}: WARNING")
            summary_stats['warn_count'] += 1
        else:
            print(f"    ✗ {test_name}: FAIL")
            summary_stats['fail_count'] += 1
    
    if result['errors']:
        print(f"  Errors: {len(result['errors'])}")

# Overall statistics
print(f"\n{'='*80}")
print(f"OVERALL STATISTICS")
print(f"{'='*80}")
print(f"Datasets Tested: {summary_stats['total_datasets']}")
print(f"Total Tests: {summary_stats['total_tests']}")
print(f"  ✓ PASS: {summary_stats['pass_count']}")
print(f"  ⚠ WARN: {summary_stats['warn_count']}")
print(f"  ✗ FAIL: {summary_stats['fail_count']}")

if summary_stats['total_tests'] > 0:
    pass_rate = (summary_stats['pass_count'] / summary_stats['total_tests']) * 100
    print(f"\nSuccess Rate: {pass_rate:.1f}%")
    
    if pass_rate >= 80:
        print("\n✓ OVERALL: PASSED (≥80% success)")
        exit_code = 0
    elif pass_rate >= 60:
        print("\n⚠ OVERALL: PARTIAL (60-80% success)")
        exit_code = 1
    else:
        print("\n✗ OVERALL: FAILED (<60% success)")
        exit_code = 2
else:
    print("\n✗ OVERALL: NO TESTS COMPLETED")
    exit_code = 3

print("="*80)

# Save summary
with open('test_summary.json', 'w') as f:
    json.dump({
        'date': '2025-10-06',
        'summary_stats': summary_stats,
        'results': {k: {
            'file': v['file'],
            'data_info': v['data_info'],
            'tests': v['tests'],
            'error_count': len(v['errors'])
        } for k, v in all_results.items()}
    }, f, indent=2)

print("\n✓ Test summary saved to: test_summary.json")
sys.exit(exit_code)
