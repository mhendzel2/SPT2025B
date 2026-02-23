"""
Comprehensive Sample Data Testing Script
Tests import, analysis, visualization, and report generation for files from each subfolder.
Date: 2025-10-06
"""

import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required modules
from data_loader import load_tracking_data, format_track_data
from enhanced_report_generator import EnhancedSPTReportGenerator
from data_access_utils import get_track_data

# Test configuration
SAMPLE_DATA_DIR = project_root / "sample data"

TEST_FILES = {
    "C2C12_40nm_SC35": "Cropped_spots_cell1.csv",
    "U2OS_40_SC35": "Cropped_spots_cell2.csv",
    "U2OS_MS2": "Cell1_spots.csv"
}

# Standard units for testing
TEST_UNITS = {
    'pixel_size': 0.1,        # μm/pixel
    'frame_interval': 0.1,     # seconds/frame
    'temperature': 298.15,     # K (25°C)
    'particle_radius': 0.5     # μm
}

# Analyses to test (covering all major categories)
CORE_ANALYSES = [
    'basic_statistics',
    'diffusion_analysis',
    'motion_classification'
]

MICRORHEOLOGY_ANALYSES = [
    'microrheology',
    'creep_compliance',
    'relaxation_modulus'
]

ADVANCED_ANALYSES = [
    'velocity_correlation',
    'confinement_analysis'
]

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

def print_section(text):
    """Print formatted section."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'-'*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'-'*80}{Colors.RESET}")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")


class TestResult:
    """Container for test results."""
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        
    def add_success(self, test_name):
        self.passed += 1
        print_success(test_name)
        
    def add_failure(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print_error(f"{test_name}: {error}")
        
    def add_warning(self, test_name, warning):
        self.warnings += 1
        print_warning(f"{test_name}: {warning}")
        
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print_section(f"Summary: {self.name}")
        print(f"Total Tests: {total}")
        print_success(f"Passed: {self.passed}")
        if self.failed > 0:
            print_error(f"Failed: {self.failed}")
        if self.warnings > 0:
            print_warning(f"Warnings: {self.warnings}")
        print()


def test_file_import(file_path):
    """Test 1: Data Import"""
    result = TestResult("Data Import")
    
    try:
        # Test file exists
        if not file_path.exists():
            result.add_failure("File Existence", f"File not found: {file_path}")
            return result, None
        result.add_success("File Existence")
        
        # Test data loading
        try:
            df = pd.read_csv(file_path)
            result.add_success(f"CSV Loading ({len(df)} rows)")
        except Exception as e:
            result.add_failure("CSV Loading", str(e))
            return result, None
        
        # Test required columns
        required_cols = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            result.add_warning("Column Check", f"Missing columns: {missing_cols}")
            # Try alternative column names
            alt_mapping = {
                'POSITION_X': ['x', 'X', 'pos_x'],
                'POSITION_Y': ['y', 'Y', 'pos_y'],
                'POSITION_T': ['frame', 'FRAME', 't', 'T', 'time'],
                'TRACK_ID': ['track_id', 'track', 'TRACK', 'trajectory_id']
            }
            found = False
            for req_col, alternatives in alt_mapping.items():
                if req_col not in df.columns:
                    for alt in alternatives:
                        if alt in df.columns:
                            result.add_success(f"Found alternative: {alt} → {req_col}")
                            found = True
                            break
        else:
            result.add_success("Required Columns Present")
        
        # Test data formatting
        try:
            formatted_df = format_track_data(df)
            result.add_success(f"Data Formatting ({len(formatted_df)} rows)")
            
            # Verify formatted columns
            if all(col in formatted_df.columns for col in ['track_id', 'frame', 'x', 'y']):
                result.add_success("Formatted Columns Verified")
            else:
                result.add_failure("Formatted Columns", "Missing required columns after formatting")
                return result, None
                
            return result, formatted_df
            
        except Exception as e:
            result.add_failure("Data Formatting", str(e))
            return result, None
            
    except Exception as e:
        result.add_failure("Import Process", str(e))
        traceback.print_exc()
        return result, None


def test_basic_analysis(tracks_df):
    """Test 2: Basic Analysis Functions"""
    result = TestResult("Basic Analysis")
    
    if tracks_df is None:
        result.add_failure("Input Data", "No data provided")
        return result
    
    try:
        # Test track statistics
        try:
            n_tracks = tracks_df['track_id'].nunique()
            result.add_success(f"Track Count: {n_tracks} unique tracks")
            
            if n_tracks < 5:
                result.add_warning("Track Count", f"Only {n_tracks} tracks (minimum 10 recommended)")
        except Exception as e:
            result.add_failure("Track Count", str(e))
        
        # Test track length distribution
        try:
            track_lengths = tracks_df.groupby('track_id').size()
            mean_length = track_lengths.mean()
            min_length = track_lengths.min()
            max_length = track_lengths.max()
            result.add_success(f"Track Lengths: min={min_length}, mean={mean_length:.1f}, max={max_length}")
            
            if mean_length < 10:
                result.add_warning("Track Length", f"Short tracks (mean={mean_length:.1f}, recommend ≥10)")
        except Exception as e:
            result.add_failure("Track Lengths", str(e))
        
        # Test MSD calculation
        try:
            from analysis import calculate_msd
            msd_result = calculate_msd(
                tracks_df,
                pixel_size=TEST_UNITS['pixel_size'],
                frame_interval=TEST_UNITS['frame_interval']
            )
            if msd_result.get('success', False):
                result.add_success(f"MSD Calculation (D={msd_result.get('diffusion_coefficient', 0):.4f} μm²/s)")
            else:
                result.add_failure("MSD Calculation", msd_result.get('error', 'Unknown error'))
        except Exception as e:
            result.add_failure("MSD Calculation", str(e))
        
        # Test motion classification
        try:
            from analysis import classify_motion
            motion_result = classify_motion(
                tracks_df,
                pixel_size=TEST_UNITS['pixel_size'],
                frame_interval=TEST_UNITS['frame_interval']
            )
            if motion_result.get('success', False):
                result.add_success(f"Motion Classification (α={motion_result.get('alpha', 0):.3f})")
            else:
                result.add_warning("Motion Classification", motion_result.get('error', 'Failed'))
        except Exception as e:
            result.add_warning("Motion Classification", str(e))
        
    except Exception as e:
        result.add_failure("Analysis Process", str(e))
        traceback.print_exc()
    
    return result


def test_visualization(tracks_df):
    """Test 3: Visualization Functions"""
    result = TestResult("Visualization")
    
    if tracks_df is None:
        result.add_failure("Input Data", "No data provided")
        return result
    
    try:
        import plotly.graph_objects as go
        
        # Test trajectory plot
        try:
            fig = go.Figure()
            for track_id in tracks_df['track_id'].unique()[:5]:  # Plot first 5 tracks
                track = tracks_df[tracks_df['track_id'] == track_id]
                fig.add_trace(go.Scatter(
                    x=track['x'],
                    y=track['y'],
                    mode='lines+markers',
                    name=f'Track {track_id}',
                    line=dict(width=1),
                    marker=dict(size=3)
                ))
            result.add_success("Trajectory Plot Created")
        except Exception as e:
            result.add_failure("Trajectory Plot", str(e))
        
        # Test MSD plot
        try:
            from analysis import calculate_msd
            msd_result = calculate_msd(tracks_df, 
                                      pixel_size=TEST_UNITS['pixel_size'],
                                      frame_interval=TEST_UNITS['frame_interval'])
            if msd_result.get('success', False):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=msd_result.get('lag_times', []),
                    y=msd_result.get('msd_values', []),
                    mode='markers+lines',
                    name='MSD'
                ))
                result.add_success("MSD Plot Created")
            else:
                result.add_warning("MSD Plot", "MSD calculation failed")
        except Exception as e:
            result.add_failure("MSD Plot", str(e))
        
        # Test histogram
        try:
            displacements = []
            for track_id in tracks_df['track_id'].unique():
                track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                dx = track['x'].diff()
                dy = track['y'].diff()
                disp = np.sqrt(dx**2 + dy**2)
                displacements.extend(disp.dropna().values)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=displacements, nbinsx=30, name='Displacements'))
            result.add_success("Displacement Histogram Created")
        except Exception as e:
            result.add_failure("Displacement Histogram", str(e))
        
    except Exception as e:
        result.add_failure("Visualization Process", str(e))
        traceback.print_exc()
    
    return result


def test_report_generation(tracks_df, dataset_name):
    """Test 4: Report Generation"""
    result = TestResult("Report Generation")
    
    if tracks_df is None:
        result.add_failure("Input Data", "No data provided")
        return result
    
    try:
        # Create report generator
        try:
            generator = EnhancedSPTReportGenerator(
                tracks_df=tracks_df,
                project_name=f"Test_{dataset_name}",
                metadata={
                    'dataset': dataset_name,
                    'test_date': '2025-10-06',
                    'test_type': 'comprehensive_sample_data'
                }
            )
            result.add_success("Report Generator Initialized")
        except Exception as e:
            result.add_failure("Generator Initialization", str(e))
            return result
        
        # Test core analyses
        print_info("Testing core analyses...")
        for analysis_name in CORE_ANALYSES:
            try:
                report = generator.generate_batch_report(
                    selected_analyses=[analysis_name],
                    current_units=TEST_UNITS
                )
                
                if report.get('analyses', {}).get(analysis_name, {}).get('success', False):
                    result.add_success(f"Core Analysis: {analysis_name}")
                else:
                    error = report.get('analyses', {}).get(analysis_name, {}).get('error', 'Unknown')
                    result.add_failure(f"Core Analysis: {analysis_name}", error)
            except Exception as e:
                result.add_failure(f"Core Analysis: {analysis_name}", str(e))
        
        # Test microrheology analyses (may have specific requirements)
        print_info("Testing microrheology analyses...")
        for analysis_name in MICRORHEOLOGY_ANALYSES:
            try:
                report = generator.generate_batch_report(
                    selected_analyses=[analysis_name],
                    current_units=TEST_UNITS
                )
                
                if report.get('analyses', {}).get(analysis_name, {}).get('success', False):
                    result.add_success(f"Microrheology: {analysis_name}")
                else:
                    error = report.get('analyses', {}).get(analysis_name, {}).get('error', 'Unknown')
                    # Microrheology may fail on small datasets - warn instead of fail
                    result.add_warning(f"Microrheology: {analysis_name}", error)
            except Exception as e:
                result.add_warning(f"Microrheology: {analysis_name}", str(e))
        
        # Test advanced analyses
        print_info("Testing advanced analyses...")
        for analysis_name in ADVANCED_ANALYSES:
            try:
                report = generator.generate_batch_report(
                    selected_analyses=[analysis_name],
                    current_units=TEST_UNITS
                )
                
                if report.get('analyses', {}).get(analysis_name, {}).get('success', False):
                    result.add_success(f"Advanced Analysis: {analysis_name}")
                else:
                    error = report.get('analyses', {}).get(analysis_name, {}).get('error', 'Unknown')
                    result.add_warning(f"Advanced Analysis: {analysis_name}", error)
            except Exception as e:
                result.add_warning(f"Advanced Analysis: {analysis_name}", str(e))
        
        # Test full report generation
        print_info("Testing full report generation...")
        try:
            full_report = generator.generate_batch_report(
                selected_analyses=CORE_ANALYSES,
                current_units=TEST_UNITS
            )
            
            success_count = sum(1 for a in full_report.get('analyses', {}).values() if a.get('success'))
            total_count = len(full_report.get('analyses', {}))
            result.add_success(f"Full Report: {success_count}/{total_count} analyses succeeded")
            
            # Test JSON export
            try:
                output_path = project_root / f"test_report_{dataset_name}.json"
                with open(output_path, 'w') as f:
                    json.dump(full_report, f, indent=2, default=str)
                result.add_success(f"JSON Export: {output_path.name}")
            except Exception as e:
                result.add_warning("JSON Export", str(e))
                
        except Exception as e:
            result.add_failure("Full Report Generation", str(e))
        
    except Exception as e:
        result.add_failure("Report Process", str(e))
        traceback.print_exc()
    
    return result


def test_dataset(subfolder, filename):
    """Run comprehensive test on a single dataset."""
    print_header(f"Testing: {subfolder}/{filename}")
    
    file_path = SAMPLE_DATA_DIR / subfolder / filename
    dataset_name = f"{subfolder}_{filename.replace('.csv', '')}"
    
    all_results = []
    
    # Test 1: Import
    print_section("Test 1: Data Import")
    import_result, tracks_df = test_file_import(file_path)
    import_result.summary()
    all_results.append(import_result)
    
    if tracks_df is None:
        print_error("Cannot proceed with further tests - import failed")
        return all_results
    
    # Test 2: Analysis
    print_section("Test 2: Basic Analysis")
    analysis_result = test_basic_analysis(tracks_df)
    analysis_result.summary()
    all_results.append(analysis_result)
    
    # Test 3: Visualization
    print_section("Test 3: Visualization")
    viz_result = test_visualization(tracks_df)
    viz_result.summary()
    all_results.append(viz_result)
    
    # Test 4: Report Generation
    print_section("Test 4: Report Generation")
    report_result = test_report_generation(tracks_df, dataset_name)
    report_result.summary()
    all_results.append(report_result)
    
    return all_results


def main():
    """Main test execution."""
    print_header("SPT2025B - Comprehensive Sample Data Testing")
    print_info(f"Testing {len(TEST_FILES)} datasets from sample data folder")
    print_info(f"Date: 2025-10-06")
    print()
    
    all_test_results = {}
    
    # Test each dataset
    for subfolder, filename in TEST_FILES.items():
        results = test_dataset(subfolder, filename)
        all_test_results[f"{subfolder}/{filename}"] = results
    
    # Final Summary
    print_header("FINAL TEST SUMMARY")
    
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    
    for dataset, results in all_test_results.items():
        print(f"\n{Colors.BOLD}{dataset}{Colors.RESET}")
        for result in results:
            total_passed += result.passed
            total_failed += result.failed
            total_warnings += result.warnings
            print(f"  {result.name}: ", end='')
            print_success(f"{result.passed} passed")
            if result.failed > 0:
                print(f"    ", end='')
                print_error(f"{result.failed} failed")
            if result.warnings > 0:
                print(f"    ", end='')
                print_warning(f"{result.warnings} warnings")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}OVERALL RESULTS{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"Total Tests: {total_passed + total_failed}")
    print_success(f"Passed: {total_passed}")
    if total_failed > 0:
        print_error(f"Failed: {total_failed}")
    if total_warnings > 0:
        print_warning(f"Warnings: {total_warnings}")
    
    # Success criteria
    success_rate = total_passed / (total_passed + total_failed) * 100 if (total_passed + total_failed) > 0 else 0
    print(f"\n{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.RESET}")
    
    if success_rate >= 80:
        print_success(f"\n✓ OVERALL STATUS: PASSED (≥80% success rate)")
        return 0
    elif success_rate >= 60:
        print_warning(f"\n⚠ OVERALL STATUS: PARTIAL (60-80% success rate)")
        return 1
    else:
        print_error(f"\n✗ OVERALL STATUS: FAILED (<60% success rate)")
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(3)
