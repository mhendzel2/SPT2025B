"""
Test script for load_precalculated_dwell_events function
"""
import pandas as pd
import sys
from analysis import load_precalculated_dwell_events

# Load the test data
print("Loading test data from CSV...")
df = pd.read_csv(r"c:\Users\mjhen\Downloads\2025-11-22T03-08_export.csv")

print("\n" + "="*60)
print("DATA INSPECTION")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "="*60)
print("CHECKING FOR DWELL-RELATED COLUMNS")
print("="*60)
dwell_cols = ['dwell_time', 'dwell_frames', 'start_frame', 'end_frame', 'track_id']
for col in dwell_cols:
    if col in df.columns:
        print(f"✓ {col}: FOUND")
        print(f"  - Data type: {df[col].dtype}")
        print(f"  - Non-null count: {df[col].notna().sum()}/{len(df)}")
        print(f"  - Sample values: {df[col].head(3).tolist()}")
    else:
        print(f"✗ {col}: MISSING")

print("\n" + "="*60)
print("TESTING load_precalculated_dwell_events")
print("="*60)

try:
    # Test with frame_interval = 0.1 (default)
    frame_interval = 0.1
    print(f"\nCalling load_precalculated_dwell_events with frame_interval={frame_interval}")
    
    result = load_precalculated_dwell_events(df, frame_interval=frame_interval)
    
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    
    if result.get('success'):
        print("✓ SUCCESS!")
        
        print("\n--- Ensemble Results ---")
        for key, value in result.get('ensemble_results', {}).items():
            print(f"  {key}: {value}")
        
        print("\n--- Dwell Stats (for UI) ---")
        for key, value in result.get('dwell_stats', {}).items():
            print(f"  {key}: {value}")
        
        print("\n--- Dwell Events DataFrame ---")
        if 'dwell_events' in result and not result['dwell_events'].empty:
            print(f"  Shape: {result['dwell_events'].shape}")
            print(f"  Columns: {list(result['dwell_events'].columns)}")
            print(f"\n  First 3 events:")
            print(result['dwell_events'].head(3))
        else:
            print("  WARNING: dwell_events is empty!")
        
        print("\n--- Track Results DataFrame ---")
        if 'track_results' in result and not result['track_results'].empty:
            print(f"  Shape: {result['track_results'].shape}")
            print(f"  Columns: {list(result['track_results'].columns)}")
            print(f"\n  First 3 tracks:")
            print(result['track_results'].head(3))
        else:
            print("  WARNING: track_results is empty!")
        
        if 'region_stats' in result:
            print("\n--- Region Stats ---")
            print(result['region_stats'])
    
    else:
        print("✗ FAILED!")
        print(f"  Error: {result.get('error', 'Unknown error')}")

except Exception as e:
    print(f"\n✗ EXCEPTION OCCURRED!")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    import traceback
    print("\n  Traceback:")
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
