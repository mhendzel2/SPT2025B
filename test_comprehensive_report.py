#!/usr/bin/env python3
"""
Comprehensive Report Generation Debug Test
Tests all analysis and visualization functions individually
"""

import pandas as pd
import numpy as np
import sys
import os
import io

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def safe_print(text):
    """Print text with proper encoding handling for Windows"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic unicode characters
        text = str(text).encode('ascii', 'replace').decode('ascii')
        print(text)

def create_realistic_track_data(n_tracks=10, min_length=20, max_length=50):
    """Create realistic diffusing particle tracks"""
    np.random.seed(42)
    
    tracks = []
    for track_id in range(1, n_tracks + 1):
        n_points = np.random.randint(min_length, max_length)
        
        # Random starting position
        x_start = np.random.uniform(10, 90)
        y_start = np.random.uniform(10, 90)
        
        # Create random walk (Brownian motion)
        x_steps = np.random.normal(0, 1.5, n_points)
        y_steps = np.random.normal(0, 1.5, n_points)
        
        x_positions = np.cumsum(np.concatenate([[x_start], x_steps]))
        y_positions = np.cumsum(np.concatenate([[y_start], y_steps]))
        
        # Add some directed motion for variety
        if track_id % 3 == 0:  # Every 3rd track is directed
            x_positions += np.linspace(0, 10, len(x_positions))
            y_positions += np.linspace(0, 5, len(y_positions))
        
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            # Add photobleaching effect
            base_intensity = np.random.uniform(800, 1200)
            bleach_factor = np.exp(-i * 0.02)
            
            tracks.append({
                'track_id': track_id,
                'frame': i,
                'x': x,
                'y': y,
                'z': np.random.normal(0, 0.5),  # Add z coordinate
                'intensity': base_intensity * bleach_factor,
                # Add intensity columns for channel 1
                'mean_intensity_ch1': base_intensity * bleach_factor,
                'Mean intensity ch1': base_intensity * bleach_factor,
                'contrast_ch1': np.random.uniform(0.3, 0.8),
                'snr_ch1': np.random.uniform(5, 20)
            })
    
    df = pd.DataFrame(tracks)
    return df

def test_all_analysis_functions():
    """Test each analysis function individually"""
    safe_print("\n" + "="*80)
    safe_print("TESTING ALL ANALYSIS FUNCTIONS")
    safe_print("="*80)
    
    from enhanced_report_generator import EnhancedSPTReportGenerator
    
    # Create realistic test data
    tracks_df = create_realistic_track_data(n_tracks=15, min_length=25, max_length=60)
    units = {'pixel_size': 0.1, 'frame_interval': 0.1}
    
    safe_print(f"\nTest Data Summary:")
    safe_print(f"  - Total points: {len(tracks_df)}")
    safe_print(f"  - Number of tracks: {tracks_df['track_id'].nunique()}")
    safe_print(f"  - Track lengths: {tracks_df.groupby('track_id').size().min()}-{tracks_df.groupby('track_id').size().max()} frames")
    safe_print(f"  - Spatial extent: x=[{tracks_df['x'].min():.1f}, {tracks_df['x'].max():.1f}], y=[{tracks_df['y'].min():.1f}, {tracks_df['y'].max():.1f}]")
    safe_print(f"  - Has intensity: {'intensity' in tracks_df.columns}")
    safe_print(f"  - Has z: {'z' in tracks_df.columns}")
    
    generator = EnhancedSPTReportGenerator()
    
    results = {}
    figures = {}
    
    # Get all available analyses
    analyses = generator.available_analyses
    
    safe_print(f"\nTesting {len(analyses)} available analyses...\n")
    
    for key, analysis in analyses.items():
        safe_print(f"{'='*80}")
        safe_print(f"Testing: {analysis['name']}")
        safe_print(f"Description: {analysis['description']}")
        safe_print(f"Category: {analysis['category']}")
        safe_print("-"*80)
        
        try:
            # Run analysis
            safe_print(f"          Running analysis function...")
            result = analysis['function'](tracks_df, units)
            
            # Check result structure
            if not isinstance(result, dict):
                safe_print(f"      ERROR: Result is not a dict, got {type(result)}")
                results[key] = {'success': False, 'error': 'Invalid result type'}
                continue
            
            success = result.get('success', True)
            has_error = 'error' in result
            
            safe_print(f"       Result keys: {list(result.keys())}")
            safe_print(f"       Success flag: {success}")
            
            if has_error:
                safe_print(f"          Error message: {result['error']}")
            
            results[key] = result
            
            # Test visualization if analysis succeeded
            if success and not has_error:
                safe_print(f"       Testing visualization function...")
                try:
                    fig = analysis['visualization'](result)
                    
                    if fig is None:
                        safe_print(f"          WARNING: Visualization returned None")
                        figures[key] = None
                    else:
                        safe_print(f"       Figure type: {type(fig).__name__}")
                        
                        # Check if it's a Plotly figure
                        if hasattr(fig, 'data'):
                            safe_print(f"       Figure has {len(fig.data)} traces")
                            if len(fig.data) == 0:
                                safe_print(f"          WARNING: Figure has no traces (blank graph)")
                            else:
                                trace_types = [type(trace).__name__ for trace in fig.data]
                                safe_print(f"       Trace types: {', '.join(set(trace_types))}")
                        
                        if hasattr(fig, 'layout'):
                            title = fig.layout.title.text if fig.layout.title else "No title"
                            safe_print(f"       Figure title: {title}")
                        
                        figures[key] = fig
                        safe_print(f"      Visualization PASSED")
                        
                except Exception as viz_error:
                    safe_print(f"      Visualization ERROR: {str(viz_error)}")
                    import traceback
                    safe_print(f"       Traceback:")
                    safe_print(traceback.format_exc())
                    figures[key] = None
            else:
                safe_print(f"          Skipping visualization (analysis failed)")
                figures[key] = None
                
            safe_print(f"      Analysis PASSED")
            
        except Exception as e:
            safe_print(f"      Analysis ERROR: {str(e)}")
            import traceback
            safe_print(f"       Traceback:")
            safe_print(traceback.format_exc())
            results[key] = {'success': False, 'error': str(e)}
            figures[key] = None
        
        safe_print("")
    
    # Summary
    safe_print("\n" + "="*80)
    safe_print("SUMMARY")
    safe_print("="*80)
    
    successful_analyses = sum(1 for r in results.values() if r.get('success', False))
    failed_analyses = len(results) - successful_analyses
    
    successful_viz = sum(1 for f in figures.values() if f is not None)
    failed_viz = len(figures) - successful_viz
    
    blank_figures = 0
    for key, fig in figures.items():
        if fig is not None and hasattr(fig, 'data') and len(fig.data) == 0:
            blank_figures += 1
    
    safe_print(f"\n     Analysis Results:")
    safe_print(f"      Successful: {successful_analyses}/{len(results)}")
    safe_print(f"      Failed: {failed_analyses}/{len(results)}")
    
    safe_print(f"\n     Visualization Results:")
    safe_print(f"      Successful: {successful_viz}/{len(figures)}")
    safe_print(f"      Failed (None): {failed_viz}/{len(figures)}")
    safe_print(f"          Blank (no traces): {blank_figures}/{len(figures)}")
    
    if failed_analyses > 0:
        safe_print(f"\n    Failed Analyses:")
        for key, result in results.items():
            if not result.get('success', False):
                analysis_name = analyses[key]['name']
                error = result.get('error', 'Unknown error')
                safe_print(f"  - {analysis_name}: {error}")
    
    if blank_figures > 0:
        safe_print(f"\n        Blank Figures:")
        for key, fig in figures.items():
            if fig is not None and hasattr(fig, 'data') and len(fig.data) == 0:
                analysis_name = analyses[key]['name']
                safe_print(f"  - {analysis_name}")
    
    # Overall result
    safe_print(f"\n{'='*80}")
    if failed_analyses == 0 and blank_figures == 0:
        safe_print("     ALL TESTS PASSED!")
        return True
    else:
        safe_print(f"        ISSUES FOUND: {failed_analyses} failed analyses, {blank_figures} blank figures")
        return False

def test_batch_report_workflow():
    """Test the complete batch report generation workflow"""
    safe_print("\n" + "="*80)
    safe_print("TESTING BATCH REPORT WORKFLOW")
    safe_print("="*80)
    
    from enhanced_report_generator import EnhancedSPTReportGenerator
    
    tracks_df = create_realistic_track_data(n_tracks=10)
    
    generator = EnhancedSPTReportGenerator()
    
    # Select a subset of analyses
    selected_analyses = [
        'basic_statistics',
        'diffusion_analysis',
        'motion_classification',
        'microrheology'
    ]
    
    safe_print(f"\n     Generating batch report with {len(selected_analyses)} analyses...")
    safe_print(f"  Selected: {', '.join(selected_analyses)}")
    
    try:
        results = generator.generate_batch_report(
            tracks_df, 
            selected_analyses, 
            "Comprehensive Test"
        )
        
        safe_print(f"\n    Batch report generated successfully")
        safe_print(f"  - Analysis results: {len(results.get('analysis_results', {}))}")
        safe_print(f"  - Figures generated: {len(results.get('figures', {}))}")
        
        # Check each analysis
        for key in selected_analyses:
            if key in results['analysis_results']:
                analysis_result = results['analysis_results'][key]
                has_figure = key in results['figures']
                
                success = analysis_result.get('success', True)
                has_error = 'error' in analysis_result
                
                status = "   " if (success and not has_error and has_figure) else "      "
                safe_print(f"  {status} {key}: success={success}, has_figure={has_figure}")
        
        return True
        
    except Exception as e:
        safe_print(f"    Batch report generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    safe_print("\n" + "="*80)
    safe_print("COMPREHENSIVE REPORT GENERATION DEBUG TEST")
    safe_print("="*80)
    
    # Test 1: All analysis functions
    test1_passed = test_all_analysis_functions()
    
    # Test 2: Batch report workflow  
    test2_passed = test_batch_report_workflow()
    
    # Final summary
    safe_print("\n" + "="*80)
    safe_print("FINAL TEST RESULTS")
    safe_print("="*80)
    safe_print(f"Test 1 (All Analyses): {'    PASSED' if test1_passed else '    FAILED'}")
    safe_print(f"Test 2 (Batch Workflow): {'    PASSED' if test2_passed else '    FAILED'}")
    safe_print("="*80)
    
    sys.exit(0 if (test1_passed and test2_passed) else 1)
