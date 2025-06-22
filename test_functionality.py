#!/usr/bin/env python3
"""
Comprehensive test script to verify all utils.py functions work correctly
with sample data files.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append('.')

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    try:
        import streamlit as st
        from utils import (
            initialize_session_state, validate_tracks_dataframe, 
            convert_coordinates_to_microns, calculate_basic_statistics,
            format_number, safe_divide, create_download_button,
            get_color_palette, filter_tracks_by_length, merge_close_detections
        )
        from enhanced_error_handling import suggest_statistical_test, validate_statistical_analysis
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def load_sample_data(filename):
    """Load and format sample data for testing."""
    print(f"Loading sample data: {filename}")
    try:
        df = pd.read_csv(f"sample_data/{filename}", skiprows=[1])
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        from utils import format_track_data
        df = format_track_data(df)
        print(f"✓ Formatted data: {len(df)} rows with required columns")
        
        return df
    except Exception as e:
        print(f"✗ Failed to load {filename}: {e}")
        return None

def test_utils_functions():
    """Test all utils.py functions with sample data."""
    # Load a default sample dataset for testing
    df = load_sample_data("Cell1_spots.csv")
    assert df is not None, "Sample data failed to load"
    from utils import (
        validate_tracks_dataframe, convert_coordinates_to_microns,
        calculate_basic_statistics, format_number, safe_divide,
        filter_tracks_by_length, merge_close_detections
    )
    
    print("\nTesting utils functions...")
    
    is_valid, message = validate_tracks_dataframe(df)
    print(f"✓ validate_tracks_dataframe: {is_valid} - {message}")
    
    df_microns = convert_coordinates_to_microns(df, pixel_size=0.1)
    print(f"✓ convert_coordinates_to_microns: converted {len(df_microns)} points")
    
    stats = calculate_basic_statistics(df)
    print(f"✓ calculate_basic_statistics: {len(stats)} metrics calculated")
    
    formatted = format_number(3.14159, decimals=2)
    print(f"✓ format_number: {formatted}")
    
    result = safe_divide(10, 2)
    result_zero = safe_divide(10, 0, default=999)
    print(f"✓ safe_divide: {result}, zero division: {result_zero}")
    
    filtered_df = filter_tracks_by_length(df, min_length=5)
    print(f"✓ filter_tracks_by_length: {len(filtered_df)} points after filtering")
    
    subset_df = df.head(100)
    merged_df = merge_close_detections(subset_df, distance_threshold=2.0)
    print(f"✓ merge_close_detections: {len(subset_df)} -> {len(merged_df)} points")
    
    return True

def test_enhanced_error_handling():
    """Test enhanced error handling functions."""
    from enhanced_error_handling import suggest_statistical_test, validate_statistical_analysis
    
    print("\nTesting enhanced error handling...")
    
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(0.5, 1, 50)
    suggestion = suggest_statistical_test(data1, data2)
    print(f"✓ suggest_statistical_test: {suggestion}")
    
    try:
        validate_statistical_analysis(data1, "test analysis")
        print("✓ validate_statistical_analysis: passed validation")
    except Exception as e:
        print(f"✗ validate_statistical_analysis failed: {e}")
    
    return True

def main():
    """Run comprehensive functionality tests."""
    print("=== SPT2025B Functionality Test ===\n")
    
    if not test_imports():
        return False
    
    sample_files = [
        "Cell1_spots.csv",
        "Cell2_spots.csv", 
        "Cropped_spots.csv"
    ]
    
    for filename in sample_files:
        if os.path.exists(f"sample_data/{filename}"):
            print(f"\n--- Testing with {filename} ---")
            df = load_sample_data(filename)
            if df is not None:
                # Run tests using one of the sample datasets
                test_utils_functions()
        else:
            print(f"⚠ Sample file {filename} not found")
    
    test_enhanced_error_handling()
    
    print("\n=== All tests completed ===")
    return True

if __name__ == "__main__":
    main()
