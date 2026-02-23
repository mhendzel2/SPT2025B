"""
Test TrackMate XML import functionality.
"""

import pandas as pd
from special_file_handlers import load_trackmate_xml_file
import io

def test_trackmate_xml_import(file_path):
    """Test importing a TrackMate XML file."""
    print("\n" + "="*70)
    print(f"Testing TrackMate XML Import: {file_path}")
    print("="*70)
    
    try:
        # Read the file
        with open(file_path, 'rb') as f:
            content = f.read()
        
        print(f"✓ File loaded: {len(content)} bytes")
        
        # Parse with the loader
        file_stream = io.BytesIO(content)
        df = load_trackmate_xml_file(file_stream)
        
        print(f"✓ XML parsed successfully")
        print(f"\n📊 Imported Data Summary:")
        print(f"  - Total rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Number of tracks: {df['track_id'].nunique()}")
        print(f"  - Frame range: {df['frame'].min()} to {df['frame'].max()}")
        print(f"  - X range: {df['x'].min():.2f} to {df['x'].max():.2f}")
        print(f"  - Y range: {df['y'].min():.2f} to {df['y'].max():.2f}")
        
        if 'z' in df.columns:
            print(f"  - Z range: {df['z'].min():.2f} to {df['z'].max():.2f}")
        
        print(f"\n📈 Track Statistics:")
        track_lengths = df.groupby('track_id').size()
        print(f"  - Average track length: {track_lengths.mean():.1f} frames")
        print(f"  - Median track length: {track_lengths.median():.1f} frames")
        print(f"  - Min track length: {track_lengths.min()} frames")
        print(f"  - Max track length: {track_lengths.max()} frames")
        
        print(f"\n🔍 Sample Data (first 10 rows):")
        pd.set_option('display.width', 120)
        pd.set_option('display.max_columns', None)
        print(df.head(10).to_string(index=False))
        
        print(f"\n🔍 Sample Data (track statistics):")
        track_stats = df.groupby('track_id').agg({
            'frame': ['count', 'min', 'max'],
            'x': ['mean', 'std'],
            'y': ['mean', 'std']
        }).head(10)
        print(track_stats)
        
        # Check for data quality issues
        print(f"\n✅ Data Quality Checks:")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"  ⚠️ Found NaN values:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"    - {col}: {count} NaN values")
        else:
            print(f"  ✓ No NaN values found")
        
        # Check for negative coordinates
        if (df['x'] < 0).any() or (df['y'] < 0).any():
            print(f"  ⚠️ Found negative coordinates:")
            if (df['x'] < 0).any():
                print(f"    - X: {(df['x'] < 0).sum()} negative values")
            if (df['y'] < 0).any():
                print(f"    - Y: {(df['y'] < 0).sum()} negative values")
        else:
            print(f"  ✓ All coordinates are non-negative")
        
        # Check for duplicate track_id + frame combinations
        duplicates = df.duplicated(subset=['track_id', 'frame']).sum()
        if duplicates > 0:
            print(f"  ⚠️ Found {duplicates} duplicate track_id+frame combinations")
        else:
            print(f"  ✓ No duplicate track_id+frame combinations")
        
        # Check frame continuity
        print(f"\n🔍 Frame Continuity Check:")
        for track_id in df['track_id'].unique()[:5]:  # Check first 5 tracks
            track = df[df['track_id'] == track_id].sort_values('frame')
            frames = track['frame'].values
            if len(frames) > 1:
                gaps = frames[1:] - frames[:-1]
                if (gaps != 1).any():
                    gap_count = (gaps != 1).sum()
                    print(f"  ⚠️ Track {track_id}: {gap_count} frame gaps (max gap: {gaps.max()})")
                else:
                    print(f"  ✓ Track {track_id}: continuous frames")
        
        print("\n" + "="*70)
        print("✅ TEST PASSED - TrackMate XML import successful")
        print("="*70)
        
        return df
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_xml_structure(file_path):
    """Inspect the XML structure to understand the format."""
    import xml.etree.ElementTree as ET
    
    print("\n" + "="*70)
    print(f"Inspecting XML Structure: {file_path}")
    print("="*70)
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        print(f"\nRoot element: <{root.tag}>")
        print(f"Root attributes: {root.attrib}")
        
        # Find all unique element types
        all_elements = set()
        for elem in root.iter():
            all_elements.add(elem.tag)
        
        print(f"\nAll element types found ({len(all_elements)}):")
        for tag in sorted(all_elements):
            count = len(root.findall(f'.//{tag}'))
            print(f"  - <{tag}>: {count} occurrences")
        
        # Check for Spots
        spots = root.findall('.//Spot')
        if spots:
            print(f"\n✓ Found {len(spots)} <Spot> elements")
            if spots:
                print(f"  Sample Spot attributes: {spots[0].attrib}")
        else:
            print(f"\n⚠️ No <Spot> elements found in standard location")
            # Try alternative paths
            spots = root.findall('.//AllSpots/SpotsInFrame/Spot')
            if spots:
                print(f"  ✓ Found {len(spots)} spots in AllSpots/SpotsInFrame")
        
        # Check for Tracks
        tracks = root.findall('.//Track')
        if tracks:
            print(f"\n✓ Found {len(tracks)} <Track> elements")
            if tracks:
                print(f"  Sample Track attributes: {tracks[0].attrib}")
                edges = tracks[0].findall('.//Edge')
                if edges:
                    print(f"  Sample Track has {len(edges)} edges")
                    if edges:
                        print(f"    Sample Edge attributes: {edges[0].attrib}")
        else:
            print(f"\n⚠️ No <Track> elements found in standard location")
            # Try alternative
            tracks = root.findall('.//AllTracks/Track')
            if tracks:
                print(f"  ✓ Found {len(tracks)} tracks in AllTracks")
        
        # Check for alternative format
        particles = root.findall('.//particle')
        if particles:
            print(f"\n✓ Found {len(particles)} <particle> elements (alternative format)")
            if particles:
                detections = particles[0].findall('detection')
                if detections:
                    print(f"  Sample particle has {len(detections)} detections")
                    if detections:
                        print(f"    Sample detection attributes: {detections[0].attrib}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Failed to inspect XML: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with the provided file
    file_path = r"e:\OneDrive\Desktop\C2C12\Trackmate_tracks\23052025\Cell1_tracks.xml"
    
    print("\n" + "="*70)
    print("TRACKMATE XML IMPORT TEST")
    print("="*70)
    
    # First inspect the structure
    inspect_xml_structure(file_path)
    
    # Then test the import
    df = test_trackmate_xml_import(file_path)
    
    if df is not None:
        print("\n✅ Import successful! The file can be loaded accurately.")
        print(f"\nTo use this data in the application:")
        print(f"  1. Go to the 'Data Loading' tab")
        print(f"  2. Upload 'Cell1_tracks.xml'")
        print(f"  3. The system will auto-detect it as TrackMate XML format")
        print(f"  4. {len(df)} data points from {df['track_id'].nunique()} tracks will be loaded")
    else:
        print("\n❌ Import failed. See errors above.")
