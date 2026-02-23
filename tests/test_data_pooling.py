"""
Test data pooling functionality with various edge cases.
"""

import pandas as pd
import numpy as np
import io
from batch_processing_utils import load_file_with_retry, pool_dataframes_efficiently

def test_duplicate_header_removal():
    """Test that duplicate header rows are removed."""
    print("\n" + "="*60)
    print("TEST: Duplicate Header Removal")
    print("="*60)
    
    # Create CSV data with duplicate header
    csv_data = """track_id,frame,x,y
track_id,frame,x,y
1,0,10.5,20.3
1,1,11.2,21.1
2,0,15.0,25.0
"""
    
    file_info = {
        'name': 'test_file.csv',
        'data': csv_data.encode('utf-8')
    }
    
    df, error = load_file_with_retry(file_info)
    
    if error:
        print(f"❌ Error loading file: {error}")
        return False
    
    print(f"Loaded DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Check that header was removed
    if len(df) == 3:  # Should have 3 data rows, not 4
        print("✓ Duplicate header row was removed")
    else:
        print(f"❌ Expected 3 rows, got {len(df)}")
        return False
    
    # Check that columns are numeric
    if pd.api.types.is_numeric_dtype(df['x']):
        print("✓ Column 'x' is numeric")
    else:
        print(f"❌ Column 'x' is not numeric: {df['x'].dtype}")
        return False
    
    print("\n✓ Test passed!")
    return True


def test_column_standardization():
    """Test that column names are standardized."""
    print("\n" + "="*60)
    print("TEST: Column Name Standardization")
    print("="*60)
    
    # Create dataframes with different column names
    df1 = pd.DataFrame({
        'Track': [1, 1, 2],
        'Frame': [0, 1, 0],
        'X': [10.0, 11.0, 15.0],
        'Y': [20.0, 21.0, 25.0]
    })
    
    df2 = pd.DataFrame({
        'TrackID': [3, 3, 4],
        'frame': [0, 1, 0],
        'x': [30.0, 31.0, 35.0],
        'y': [40.0, 41.0, 45.0]
    })
    
    pooled = pool_dataframes_efficiently([df1, df2], validate=True, deduplicate=False)
    
    print(f"Pooled DataFrame shape: {pooled.shape}")
    print(f"Columns: {list(pooled.columns)}")
    
    # Check standardized columns exist
    expected_cols = {'track_id', 'frame', 'x', 'y'}
    if expected_cols.issubset(set(pooled.columns)):
        print(f"✓ All expected columns present: {expected_cols}")
    else:
        missing = expected_cols - set(pooled.columns)
        print(f"❌ Missing columns: {missing}")
        return False
    
    # Check data was pooled correctly
    if len(pooled) == 6:
        print(f"✓ Correct number of rows: {len(pooled)}")
    else:
        print(f"❌ Expected 6 rows, got {len(pooled)}")
        return False
    
    print("\n✓ Test passed!")
    return True


def test_encoding_handling():
    """Test that different encodings are handled."""
    print("\n" + "="*60)
    print("TEST: Encoding Handling")
    print("="*60)
    
    # Create CSV with UTF-8 encoding
    csv_data = "track_id,frame,x,y\n1,0,10.5,20.3\n"
    
    file_info = {
        'name': 'test_utf8.csv',
        'data': csv_data.encode('utf-8')
    }
    
    df, error = load_file_with_retry(file_info)
    
    if error:
        print(f"❌ Error loading UTF-8 file: {error}")
        return False
    
    print(f"✓ UTF-8 file loaded successfully: {df.shape}")
    
    # Test string data (already encoded)
    file_info2 = {
        'name': 'test_string.csv',
        'data': csv_data  # String instead of bytes
    }
    
    df2, error2 = load_file_with_retry(file_info2)
    
    if error2:
        print(f"❌ Error loading string data: {error2}")
        return False
    
    print(f"✓ String data loaded successfully: {df2.shape}")
    
    print("\n✓ Test passed!")
    return True


def test_empty_row_removal():
    """Test that completely empty rows are removed."""
    print("\n" + "="*60)
    print("TEST: Empty Row Removal")
    print("="*60)
    
    # Create CSV with empty rows
    csv_data = """track_id,frame,x,y
1,0,10.5,20.3

1,1,11.2,21.1

2,0,15.0,25.0
"""
    
    file_info = {
        'name': 'test_empty_rows.csv',
        'data': csv_data.encode('utf-8')
    }
    
    df, error = load_file_with_retry(file_info)
    
    if error:
        print(f"❌ Error loading file: {error}")
        return False
    
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Check that empty rows were removed
    if len(df) == 3:  # Should have 3 data rows
        print("✓ Empty rows were removed")
    else:
        print(f"❌ Expected 3 rows, got {len(df)}")
        print(df)
        return False
    
    print("\n✓ Test passed!")
    return True


def test_duplicate_detection():
    """Test that duplicate rows are detected and removed."""
    print("\n" + "="*60)
    print("TEST: Duplicate Row Detection")
    print("="*60)
    
    # Create dataframes with overlapping data
    df1 = pd.DataFrame({
        'track_id': [1, 1, 2],
        'frame': [0, 1, 0],
        'x': [10.0, 11.0, 15.0],
        'y': [20.0, 21.0, 25.0]
    })
    
    df2 = pd.DataFrame({
        'track_id': [1, 2, 3],  # track_id=1,frame=0 and track_id=2,frame=0 are duplicates
        'frame': [0, 0, 0],
        'x': [10.0, 15.0, 30.0],
        'y': [20.0, 25.0, 40.0]
    })
    
    pooled = pool_dataframes_efficiently([df1, df2], validate=True, deduplicate=True)
    
    print(f"Pooled DataFrame shape: {pooled.shape}")
    print(f"Unique track_id+frame combinations: {len(pooled[['track_id', 'frame']].drop_duplicates())}")
    
    # Check that duplicates were removed
    # df1 has 3 rows: (1,0), (1,1), (2,0)
    # df2 has 3 rows: (1,0), (2,0), (3,0)
    # Duplicates: (1,0) and (2,0) appear in both
    # Result should have 4 unique rows: (1,0), (1,1), (2,0), (3,0)
    if len(pooled) == 4:
        print(f"✓ Duplicates removed correctly: {len(pooled)} unique rows")
    else:
        print(f"❌ Expected 4 rows, got {len(pooled)}")
        print(pooled)
        return False
    
    print("\n✓ Test passed!")
    return True


def run_all_tests():
    """Run all pooling tests."""
    print("\n" + "="*70)
    print("DATA POOLING TEST SUITE")
    print("="*70)
    
    tests = [
        test_duplicate_header_removal,
        test_column_standardization,
        test_encoding_handling,
        test_empty_row_removal,
        test_duplicate_detection
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
