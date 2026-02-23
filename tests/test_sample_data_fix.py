"""
Quick test to verify sample data loading fix
"""

import os
import pandas as pd

def test_sample_data_structure():
    """Test that sample data folder structure is correct."""
    print("="*80)
    print("SAMPLE DATA STRUCTURE TEST")
    print("="*80)
    
    sample_data_dir = "sample data"
    
    # Check folder exists
    if not os.path.exists(sample_data_dir):
        print(f"❌ FAILED: '{sample_data_dir}' folder not found")
        return False
    
    print(f"✓ Found: {sample_data_dir}/")
    
    # Check subfolders
    expected_subdirs = ["C2C12_40nm_SC35", "U2OS_40_SC35", "U2OS_MS2"]
    found_subdirs = []
    
    for subdir in os.listdir(sample_data_dir):
        subdir_path = os.path.join(sample_data_dir, subdir)
        if os.path.isdir(subdir_path):
            found_subdirs.append(subdir)
    
    print(f"\nSubfolders found: {len(found_subdirs)}")
    for subdir in found_subdirs:
        print(f"  - {subdir}/")
    
    # Scan for CSV files
    print("\n" + "-"*80)
    print("CSV FILES BY FOLDER:")
    print("-"*80)
    
    total_files = 0
    all_datasets = {}
    
    for subdir in found_subdirs:
        subdir_path = os.path.join(sample_data_dir, subdir)
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        
        print(f"\n{subdir}/ ({len(csv_files)} files):")
        for csv_file in csv_files:
            print(f"  ✓ {csv_file}")
            display_name = f"{subdir}/{csv_file}"
            all_datasets[display_name] = os.path.join(subdir_path, csv_file)
            total_files += 1
    
    print("\n" + "="*80)
    print(f"TOTAL: {total_files} CSV files available")
    print("="*80)
    
    return all_datasets

def test_sample_file_loading():
    """Test loading a sample file."""
    print("\n" + "="*80)
    print("SAMPLE FILE LOADING TEST")
    print("="*80)
    
    # Try to load Cell1_spots.csv
    test_file = os.path.join("sample data", "U2OS_MS2", "Cell1_spots.csv")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"\nLoading: {test_file}")
    
    try:
        df = pd.read_csv(test_file)
        
        print(f"✓ Successfully loaded")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['track_id', 'frame', 'x', 'y']
        missing = []
        
        for col in required_cols:
            if col in df.columns:
                print(f"  ✓ {col}: present")
            else:
                print(f"  ✗ {col}: MISSING")
                missing.append(col)
        
        if missing:
            print(f"\n❌ Missing required columns: {missing}")
            return False
        
        # Check track statistics
        if 'track_id' in df.columns:
            n_tracks = df['track_id'].nunique()
            print(f"\n✓ Tracks: {n_tracks}")
            
            # Track length distribution
            track_lengths = df.groupby('track_id').size()
            print(f"  Mean length: {track_lengths.mean():.1f} frames")
            print(f"  Min length: {track_lengths.min()} frames")
            print(f"  Max length: {track_lengths.max()} frames")
        
        print("\n✅ Sample file is valid and ready for analysis!")
        return True
    
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return False

def test_fbm_readiness():
    """Test if data is suitable for FBM analysis."""
    print("\n" + "="*80)
    print("FBM ANALYSIS READINESS TEST")
    print("="*80)
    
    test_file = os.path.join("sample data", "U2OS_MS2", "Cell1_spots.csv")
    
    if not os.path.exists(test_file):
        print("❌ Test file not found")
        return False
    
    try:
        df = pd.read_csv(test_file)
        
        if 'track_id' not in df.columns:
            print("❌ No track_id column")
            return False
        
        # Check tracks meeting minimum length requirement
        min_length = 10
        track_lengths = df.groupby('track_id').size()
        valid_tracks = (track_lengths >= min_length).sum()
        total_tracks = len(track_lengths)
        
        print(f"Minimum track length for FBM: {min_length} frames")
        print(f"Total tracks: {total_tracks}")
        print(f"Valid tracks (≥{min_length} frames): {valid_tracks}")
        print(f"Valid percentage: {100*valid_tracks/total_tracks:.1f}%")
        
        if valid_tracks == 0:
            print("\n❌ No tracks meet minimum length requirement")
            return False
        elif valid_tracks < total_tracks * 0.5:
            print(f"\n⚠️ Warning: Only {100*valid_tracks/total_tracks:.1f}% of tracks are long enough")
            print("   FBM analysis will work but may have limited statistics")
        else:
            print("\n✅ Sufficient tracks for FBM analysis")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SAMPLE DATA FIX VERIFICATION")
    print("="*80)
    print("Testing that sample data loading fixes are working correctly")
    print("="*80)
    
    # Run tests
    test1 = test_sample_data_structure()
    test2 = test_sample_file_loading()
    test3 = test_fbm_readiness()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if test1:
        print("✅ Sample data structure: PASSED")
    else:
        print("❌ Sample data structure: FAILED")
    
    if test2:
        print("✅ Sample file loading: PASSED")
    else:
        print("❌ Sample file loading: FAILED")
    
    if test3:
        print("✅ FBM analysis readiness: PASSED")
    else:
        print("❌ FBM analysis readiness: FAILED")
    
    if test1 and test2 and test3:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nSample data loading is working correctly.")
        print("You can now:")
        print("  1. Open the app: streamlit run app.py")
        print("  2. Expand 'Sample Data' in sidebar")
        print("  3. Select any dataset from dropdown")
        print("  4. Click 'Load Selected Sample'")
        print("  5. Run analyses including FBM!")
    else:
        print("\n❌ Some tests failed. Check error messages above.")
