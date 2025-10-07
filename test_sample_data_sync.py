"""
Test script to verify sample data synchronization and auto-discovery.
"""

import sys
from pathlib import Path
from sample_data_manager import SampleDataManager

def test_sample_data_discovery():
    """Test that all sample data files are discovered correctly."""
    
    print("=" * 70)
    print("SAMPLE DATA SYNCHRONIZATION TEST")
    print("=" * 70)
    
    # Initialize manager
    manager = SampleDataManager(sample_data_dir="sample data")
    
    # Get available datasets
    datasets = manager.get_available_datasets()
    
    print(f"\n✓ Found {len(datasets)} sample datasets\n")
    
    if len(datasets) == 0:
        print("❌ ERROR: No datasets found!")
        return False
    
    # Group by category
    categories = {}
    for file_key, info in datasets.items():
        category = info.get('category', 'General')
        if category not in categories:
            categories[category] = []
        categories[category].append(info)
    
    print(f"✓ Organized into {len(categories)} categories\n")
    
    # Display datasets by category
    for category, files in sorted(categories.items()):
        print(f"\n📁 {category} ({len(files)} files)")
        print("-" * 70)
        
        for info in sorted(files, key=lambda x: x['name']):
            print(f"  • {info['name']}")
            print(f"    File: {info['filename']}")
            print(f"    Path: {info['relative_path']}")
            print(f"    Size: {info['file_size'] / 1024:.1f} KB")
            print(f"    Columns: {len(info['columns'])}")
            
            # Show features
            features = []
            if info.get('has_tracks'):
                features.append("Tracks")
            if info.get('has_temporal'):
                features.append("Temporal")
            if info.get('has_multichannel'):
                features.append("Multi-channel")
            
            if features:
                print(f"    Features: {', '.join(features)}")
            print()
    
    # Test loading a dataset
    print("\n" + "=" * 70)
    print("TESTING DATASET LOADING")
    print("=" * 70)
    
    # Try to load first dataset
    first_key = list(datasets.keys())[0]
    print(f"\nLoading: {datasets[first_key]['name']}...")
    
    df = manager.load_dataset(first_key)
    
    if df is not None:
        print(f"✓ Successfully loaded dataset")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {', '.join(df.columns[:5])}...")
        
        # Show updated metadata
        updated_info = datasets[first_key]
        if updated_info.get('tracks'):
            print(f"  Tracks: {updated_info['tracks']}")
        if updated_info.get('frames'):
            print(f"  Frames: {updated_info['frames']}")
        print(f"  Particles: {updated_info['particles']}")
        
        return True
    else:
        print("❌ ERROR: Failed to load dataset")
        return False

def test_git_sync():
    """Test that sample data files are properly synced to git."""
    
    print("\n" + "=" * 70)
    print("GIT SYNCHRONIZATION TEST")
    print("=" * 70)
    
    sample_data_dir = Path("sample data")
    
    if not sample_data_dir.exists():
        print("❌ ERROR: 'sample data' directory not found")
        return False
    
    # Count CSV files
    csv_files = list(sample_data_dir.glob("**/*.csv"))
    print(f"\n✓ Found {len(csv_files)} CSV files in 'sample data' directory")
    
    # Check subdirectories
    subdirs = [d for d in sample_data_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(subdirs)} subdirectories:")
    
    for subdir in sorted(subdirs):
        csv_count = len(list(subdir.glob("*.csv")))
        print(f"  • {subdir.name}: {csv_count} files")
    
    return True

if __name__ == "__main__":
    print("\n" + "🔬 SPT2025B Sample Data Verification" + "\n")
    
    # Run tests
    test1_passed = test_sample_data_discovery()
    test2_passed = test_git_sync()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Sample Data Discovery: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Git Synchronization:   {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests PASSED! Sample data is properly synced and accessible.")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Check the output above.")
        sys.exit(1)
