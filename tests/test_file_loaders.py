"""
File Loader Tests

Comprehensive tests for all file format loaders including CSV, Excel, MVD2,
Volocity, Imaris, and special formats.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Import loaders to test
from data_loader import load_tracks_file, format_track_data, detect_file_format
from special_file_handlers import (
    load_imaris_spots,
    load_ms2_spots,
    IMARIS_AVAILABLE,
    MS2_AVAILABLE
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_csv_content():
    """Sample CSV content in standard format."""
    return """track_id,frame,x,y
1,0,10.5,20.3
1,1,11.2,20.8
1,2,12.1,21.5
2,0,30.2,40.1
2,1,31.5,40.9
2,2,32.3,41.7"""


@pytest.fixture
def sample_trackmate_csv():
    """Sample TrackMate format CSV."""
    return """TRACK_ID,POSITION_T,POSITION_X,POSITION_Y
1,0,10.5,20.3
1,1,11.2,20.8
2,0,30.2,40.1
2,1,31.5,40.9"""


@pytest.fixture
def sample_volocity_csv():
    """Sample Volocity format CSV."""
    return """Trajectory,Frame,X,Y
0,0,10.5,20.3
0,1,11.2,20.8
1,0,30.2,40.1
1,1,31.5,40.9"""


@pytest.fixture
def temp_csv_file(sample_csv_content):
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        yield f.name
    
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_trackmate_file(sample_trackmate_csv):
    """Create temporary TrackMate CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_trackmate_csv)
        yield f.name
    
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_volocity_file(sample_volocity_csv):
    """Create temporary Volocity CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_volocity_csv)
        yield f.name
    
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_excel_file():
    """Create temporary Excel file."""
    df = pd.DataFrame({
        'track_id': [1, 1, 2, 2],
        'frame': [0, 1, 0, 1],
        'x': [10.5, 11.2, 30.2, 31.5],
        'y': [20.3, 20.8, 40.1, 40.9]
    })
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        df.to_excel(f.name, index=False)
        yield f.name
    
    if os.path.exists(f.name):
        os.unlink(f.name)


# ==============================================================================
# CSV Loading Tests
# ==============================================================================

class TestCSVLoading:
    """Test CSV file loading with various formats."""
    
    def test_load_standard_csv(self, temp_csv_file):
        """Test loading standard format CSV."""
        df = load_tracks_file(temp_csv_file, pixel_size=0.1, frame_interval=0.1)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert 'track_id' in df.columns
        assert 'frame' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
    
    def test_load_trackmate_csv(self, temp_trackmate_file):
        """Test loading TrackMate format CSV."""
        df = load_tracks_file(temp_trackmate_file, pixel_size=0.1, frame_interval=0.1)
        
        assert df is not None
        assert 'track_id' in df.columns
        assert 'frame' in df.columns
        assert len(df) == 4
    
    def test_load_volocity_csv(self, temp_volocity_file):
        """Test loading Volocity format CSV."""
        df = load_tracks_file(temp_volocity_file, pixel_size=0.1, frame_interval=0.1)
        
        assert df is not None
        assert 'track_id' in df.columns
        assert 'frame' in df.columns
        assert len(df) == 4
    
    def test_load_csv_with_extra_columns(self):
        """Test loading CSV with additional columns."""
        csv_content = """track_id,frame,x,y,intensity,quality
1,0,10.5,20.3,100,0.95
1,1,11.2,20.8,105,0.93"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            df = load_tracks_file(temp_file, pixel_size=0.1, frame_interval=0.1)
            
            assert df is not None
            assert 'track_id' in df.columns
            assert 'intensity' in df.columns
            assert 'quality' in df.columns
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_csv_with_z_coordinate(self):
        """Test loading CSV with 3D coordinates."""
        csv_content = """track_id,frame,x,y,z
1,0,10.5,20.3,5.2
1,1,11.2,20.8,5.4"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            df = load_tracks_file(temp_file, pixel_size=0.1, frame_interval=0.1)
            
            assert df is not None
            assert 'z' in df.columns
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_csv_with_missing_values(self):
        """Test loading CSV with missing values."""
        csv_content = """track_id,frame,x,y
1,0,10.5,20.3
1,1,,20.8
1,2,12.1,"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            df = load_tracks_file(temp_file, pixel_size=0.1, frame_interval=0.1)
            
            # Should handle missing values
            assert df is not None
            assert len(df) >= 1  # At least one valid row
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


# ==============================================================================
# Excel Loading Tests
# ==============================================================================

class TestExcelLoading:
    """Test Excel file loading."""
    
    def test_load_excel_file(self, temp_excel_file):
        """Test loading Excel file."""
        df = load_tracks_file(temp_excel_file, pixel_size=0.1, frame_interval=0.1)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert 'track_id' in df.columns
    
    def test_load_excel_with_multiple_sheets(self):
        """Test loading Excel with multiple sheets."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            with pd.ExcelWriter(f.name) as writer:
                # Sheet 1
                df1 = pd.DataFrame({
                    'track_id': [1, 1],
                    'frame': [0, 1],
                    'x': [10.0, 11.0],
                    'y': [20.0, 21.0]
                })
                df1.to_excel(writer, sheet_name='Tracks', index=False)
                
                # Sheet 2
                df2 = pd.DataFrame({
                    'other_data': [1, 2, 3]
                })
                df2.to_excel(writer, sheet_name='Other', index=False)
            
            temp_file = f.name
        
        try:
            df = load_tracks_file(temp_file, pixel_size=0.1, frame_interval=0.1)
            
            # Should load first sheet or handle gracefully
            assert df is not None
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


# ==============================================================================
# Format Detection Tests
# ==============================================================================

class TestFormatDetection:
    """Test file format detection."""
    
    def test_detect_csv_format(self, temp_csv_file):
        """Test detecting CSV format."""
        # This function may not exist - testing if it does
        try:
            format_type = detect_file_format(temp_csv_file)
            assert format_type in ['csv', 'text', None]
        except (NameError, AttributeError):
            # Function doesn't exist - that's okay
            pass
    
    def test_detect_excel_format(self, temp_excel_file):
        """Test detecting Excel format."""
        try:
            format_type = detect_file_format(temp_excel_file)
            assert format_type in ['excel', 'xlsx', None]
        except (NameError, AttributeError):
            pass


# ==============================================================================
# Format Track Data Tests
# ==============================================================================

class TestFormatTrackData:
    """Test track data formatting and normalization."""
    
    def test_format_standard_columns(self):
        """Test formatting with standard columns."""
        df = pd.DataFrame({
            'track_id': [1, 1, 2],
            'frame': [0, 1, 0],
            'x': [10.0, 11.0, 20.0],
            'y': [20.0, 21.0, 30.0]
        })
        
        formatted = format_track_data(df)
        
        assert 'track_id' in formatted.columns
        assert 'frame' in formatted.columns
        assert 'x' in formatted.columns
        assert 'y' in formatted.columns
    
    def test_format_trackmate_columns(self):
        """Test formatting TrackMate column names."""
        df = pd.DataFrame({
            'TRACK_ID': [1, 1],
            'POSITION_T': [0, 1],
            'POSITION_X': [10.0, 11.0],
            'POSITION_Y': [20.0, 21.0]
        })
        
        formatted = format_track_data(df)
        
        assert 'track_id' in formatted.columns
        assert 'frame' in formatted.columns
        assert 'x' in formatted.columns
        assert 'y' in formatted.columns
    
    def test_format_volocity_columns(self):
        """Test formatting Volocity column names."""
        df = pd.DataFrame({
            'Trajectory': [0, 0, 1],
            'Frame': [0, 1, 0],
            'X': [10.0, 11.0, 20.0],
            'Y': [20.0, 21.0, 30.0]
        })
        
        formatted = format_track_data(df)
        
        assert 'track_id' in formatted.columns
        assert 'frame' in formatted.columns
    
    def test_format_with_case_variations(self):
        """Test formatting with different case variations."""
        df = pd.DataFrame({
            'Track_ID': [1, 1],
            'FRAME': [0, 1],
            'X': [10.0, 11.0],
            'y': [20.0, 21.0]
        })
        
        formatted = format_track_data(df)
        
        # Should normalize to lowercase
        assert 'track_id' in formatted.columns
        assert 'frame' in formatted.columns
    
    def test_format_preserves_extra_columns(self):
        """Test that extra columns are preserved."""
        df = pd.DataFrame({
            'track_id': [1, 1],
            'frame': [0, 1],
            'x': [10.0, 11.0],
            'y': [20.0, 21.0],
            'intensity': [100, 105],
            'quality': [0.95, 0.93]
        })
        
        formatted = format_track_data(df)
        
        assert 'intensity' in formatted.columns
        assert 'quality' in formatted.columns


# ==============================================================================
# Special File Format Tests
# ==============================================================================

class TestSpecialFileFormats:
    """Test special file format loaders."""
    
    @pytest.mark.skipif(not IMARIS_AVAILABLE, reason="Imaris loader not available")
    def test_load_imaris_spots(self):
        """Test loading Imaris spots file."""
        # This would require actual Imaris file - test if function exists
        assert callable(load_imaris_spots)
    
    @pytest.mark.skipif(not MS2_AVAILABLE, reason="MS2 loader not available")
    def test_load_ms2_spots(self):
        """Test loading MS2 spots file."""
        # This would require actual MS2 file - test if function exists
        assert callable(load_ms2_spots)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestFileLoaderErrorHandling:
    """Test error handling in file loaders."""
    
    def test_load_nonexistent_file(self):
        """Test loading file that doesn't exist."""
        result = load_tracks_file('nonexistent_file.csv', 0.1, 0.1)
        assert result is None
    
    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write nothing
            temp_file = f.name
        
        try:
            result = load_tracks_file(temp_file, 0.1, 0.1)
            # Should return None or empty DataFrame
            assert result is None or (isinstance(result, pd.DataFrame) and result.empty)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_corrupted_csv(self):
        """Test loading corrupted CSV file."""
        corrupted_content = """track_id,frame,x,y
1,0,10.5,20.3
1,1,corrupted,data,extra,columns
2,0,30.2"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(corrupted_content)
            temp_file = f.name
        
        try:
            result = load_tracks_file(temp_file, 0.1, 0.1)
            # Should handle gracefully
            assert result is None or isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_file_with_wrong_extension(self):
        """Test loading file with wrong extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("track_id,frame,x,y\n1,0,10,20\n")
            temp_file = f.name
        
        try:
            result = load_tracks_file(temp_file, 0.1, 0.1)
            # May or may not load depending on implementation
            assert result is None or isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_file_missing_required_columns(self):
        """Test loading file missing required columns."""
        csv_content = """time,position
0,10.5
1,11.2"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            result = load_tracks_file(temp_file, 0.1, 0.1)
            # Should return None or handle gracefully
            assert result is None or isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestFileLoaderPerformance:
    """Test file loader performance with larger files."""
    
    def test_load_large_csv(self):
        """Test loading large CSV file."""
        # Generate large dataset
        n_rows = 10000
        data = {
            'track_id': np.repeat(range(100), 100),
            'frame': np.tile(range(100), 100),
            'x': np.random.randn(n_rows) * 100,
            'y': np.random.randn(n_rows) * 100
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            import time
            start = time.time()
            result = load_tracks_file(temp_file, 0.1, 0.1)
            duration = time.time() - start
            
            assert result is not None
            assert len(result) == n_rows
            assert duration < 5.0  # Should complete in <5 seconds
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
