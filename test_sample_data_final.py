"""
Sample Data Validation Test - Final Version
Tests import, analysis, visualization, and report generation.
Date: 2025-10-06
"""

import pandas as pd
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("SPT2025B - Sample Data Validation Test")
print("="*80)
print()

# Test configuration
test_files = {
    "C2C12_40nm_SC35": "sample data/C2C12_40nm_SC35/Cropped_spots_cell1.csv",
    "U2OS_40_SC35": "sample data/U2OS_40_SC35/Cropped_spots_cell2.csv",
    "U2OS_MS2": "sample data/U2OS_MS2/Cell1_spots.csv"
}

test_units = {
    'pixel_size': 0.1,
    'frame_interval': 0.1,
    'temperature': 298.15,
    'particle_radius': 0.5
}

test_results = {}

for dataset_name, filepath in test_files.items():
    print(f"\n{'='*80}")
    print(f"TESTING: {dataset_name}")
    print(f"File: {filepath}")
    print('='*80)
    
    results = {'tests': {}, 'errors': [], 'warnings': []}
    
    # TEST 1: Import
    print("\n[1] Data Import")
    try:
        df = pd.read_csv(filepath)
        print(f"    ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        results['tests']['import'] = 'PASS'
        
        # Check columns
        required = ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"    ⚠ Missing columns: {missing}")
            results['warnings'].append(f"Missing columns: {missing}")
        else:
            print(f"    ✓ All required columns present")
        
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['tests']['import'] = 'FAIL'
        results['errors'].append(('import', str(e)))
        test_results[dataset_name] = results
        continue
    
    # TEST 2: Format
    print("\n[2] Data Formatting")
    try:
        from data_loader import format_track_data
        tracks_df = format_track_data(df)
        
        if tracks_df is not None and len(tracks_df) > 0:
            print(f"    ✓ Formatted {len(tracks_df)} rows")
            n_tracks = tracks_df['track_id'].nunique()
            track_lengths = tracks_df.groupby('track_id').size()
            print(f"    ✓ Tracks: {n_tracks} (mean length: {track_lengths.mean():.1f} frames)")
            results['tests']['format'] = 'PASS'
        else:
            print(f"    ✗ Formatting returned empty data")
            results['tests']['format'] = 'FAIL'
            results['errors'].append(('format', 'Empty data after formatting'))
            test_results[dataset_name] = results
            continue
            
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['tests']['format'] = 'FAIL'
        results['errors'].append(('format', str(e)))
        test_results[dataset_name] = results
        continue
    
    # TEST 3: Basic Analysis
    print("\n[3] Analysis Functions")
    try:
        from analysis import calculate_msd
        
        # MSD calculation
        msd_result = calculate_msd(
            tracks_df, 
            pixel_size=test_units['pixel_size'],
            frame_interval=test_units['frame_interval']
        )
        
        if msd_result.get('success'):
            D = msd_result.get('diffusion_coefficient', 0)
            print(f"    ✓ MSD calculated (D={D:.4f} μm²/s)")
            results['tests']['analysis_msd'] = 'PASS'
        else:
            print(f"    ⚠ MSD warning: {msd_result.get('error')}")
            results['tests']['analysis_msd'] = 'WARN'
            results['warnings'].append(f"MSD: {msd_result.get('error')}")
        
        # Motion analysis
        from analysis import analyze_motion_patterns
        motion_result = analyze_motion_patterns(
            tracks_df,
            pixel_size=test_units['pixel_size'],
            frame_interval=test_units['frame_interval']
        )
        
        if motion_result.get('success'):
            print(f"    ✓ Motion analysis completed")
            results['tests']['analysis_motion'] = 'PASS'
        else:
            print(f"    ⚠ Motion analysis warning")
            results['tests']['analysis_motion'] = 'WARN'
        
    except Exception as e:
        print(f"    ✗ Analysis FAILED: {e}")
        results['tests']['analysis'] = 'FAIL'
        results['errors'].append(('analysis', str(e)))
    
    # TEST 4: Visualization
    print("\n[4] Visualization")
    try:
        import plotly.graph_objects as go
        
        # Trajectory plot
        fig = go.Figure()
        sample_tracks = tracks_df['track_id'].unique()[:5]
        for track_id in sample_tracks:
            track = tracks_df[tracks_df['track_id'] == track_id]
            fig.add_trace(go.Scatter(
                x=track['x'], 
                y=track['y'], 
                mode='lines+markers',
                name=f'Track {track_id}',
                marker=dict(size=3)
            ))
        
        print(f"    ✓ Trajectory plot created ({len(sample_tracks)} tracks)")
        results['tests']['visualization'] = 'PASS'
        
    except Exception as e:
        print(f"    ✗ Visualization FAILED: {e}")
        results['tests']['visualization'] = 'FAIL'
        results['errors'].append(('visualization', str(e)))
    
    # TEST 5: Report Generation
    print("\n[5] Report Generation")
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Initialize generator
        generator = EnhancedSPTReportGenerator(
            tracks_df=tracks_df,
            project_name=f"Test_{dataset_name}",
            metadata={
                'dataset': dataset_name,
                'test_date': '2025-10-06',
                'file': filepath
            }
        )
        print(f"    ✓ Generator initialized")
        
        # Generate core analyses report
        core_analyses = ['basic_statistics', 'diffusion_analysis', 'motion_classification']
        report = generator.generate_batch_report(
            selected_analyses=core_analyses,
            current_units=test_units
        )
        
        # Check results
        analyses = report.get('analyses', {})
        success_count = sum(1 for a in analyses.values() if a.get('success'))
        total_count = len(analyses)
        
        print(f"    ✓ Report generated: {success_count}/{total_count} analyses succeeded")
        
        if success_count >= 2:
            results['tests']['report_core'] = 'PASS'
        else:
            results['tests']['report_core'] = 'WARN'
            results['warnings'].append(f"Only {success_count}/{total_count} core analyses succeeded")
        
        # Test microrheology analyses
        print("\n    Testing Advanced Microrheology...")
        micro_analyses = ['creep_compliance', 'relaxation_modulus']
        micro_report = generator.generate_batch_report(
            selected_analyses=micro_analyses,
            current_units=test_units
        )
        
        micro_success = sum(1 for a in micro_report.get('analyses', {}).values() if a.get('success'))
        micro_total = len(micro_report.get('analyses', {}))
        print(f"    ✓ Microrheology: {micro_success}/{micro_total} analyses succeeded")
        
        if micro_success > 0:
            results['tests']['report_micro'] = 'PASS'
        else:
            results['tests']['report_micro'] = 'WARN'
            results['warnings'].append("Microrheology analyses may need more data")
        
        # Export test report
        output_file = f"test_report_{dataset_name}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"    ✓ Report exported: {output_file}")
        
    except Exception as e:
        print(f"    ✗ Report generation FAILED: {e}")
        results['tests']['report'] = 'FAIL'
        results['errors'].append(('report', str(e)))
    
    test_results[dataset_name] = results

# FINAL SUMMARY
print("\n" + "="*80)
print("FINAL TEST SUMMARY")
print("="*80)

total_pass = 0
total_warn = 0
total_fail = 0

for dataset, results in test_results.items():
    print(f"\n{dataset}:")
    
    for test_name, status in results['tests'].items():
        if status == 'PASS':
            print(f"  ✓ {test_name}: PASS")
            total_pass += 1
        elif status == 'WARN':
            print(f"  ⚠ {test_name}: WARNING")
            total_warn += 1
        elif status == 'FAIL':
            print(f"  ✗ {test_name}: FAIL")
            total_fail += 1
    
    if results['warnings']:
        print(f"  Warnings: {len(results['warnings'])}")
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error_type, error_msg in results['errors'][:2]:  # Show first 2 errors
            print(f"    - {error_type}: {error_msg[:100]}")

total_tests = total_pass + total_warn + total_fail
pass_rate = (total_pass / total_tests * 100) if total_tests > 0 else 0

print(f"\n{'='*80}")
print(f"OVERALL RESULTS")
print(f"{'='*80}")
print(f"Total Tests: {total_tests}")
print(f"  PASS: {total_pass} ({pass_rate:.1f}%)")
print(f"  WARN: {total_warn}")
print(f"  FAIL: {total_fail}")

# Determine overall status
if total_fail == 0 and total_pass >= total_tests * 0.8:
    print(f"\n✓ OVERALL STATUS: PASSED")
    exit_code = 0
elif total_fail <= 2:
    print(f"\n⚠ OVERALL STATUS: PARTIAL PASS")
    exit_code = 1
else:
    print(f"\n✗ OVERALL STATUS: FAILED")
    exit_code = 2

print("="*80)
sys.exit(exit_code)
