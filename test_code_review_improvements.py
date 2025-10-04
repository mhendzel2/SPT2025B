"""
Unit tests for code review improvements.
Tests data handling, batch processing, and visualization optimizations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import io


class TestDataHandling:
    """Test data handling improvements."""
    
    def test_duplicate_column_detection(self):
        """Test that duplicate columns are detected and handled."""
        # Create DataFrame with duplicate columns
        df = pd.DataFrame({
            'track_id': [1, 2, 3],
            'frame': [0, 1, 2],
            'x': [10, 20, 30],
            'y': [15, 25, 35]
        })
        
        # Manually create duplicate column
        df['x_2'] = [11, 21, 31]
        df.columns = ['track_id', 'frame', 'x', 'y', 'x']
        
        # Check duplicate detection
        assert df.columns.duplicated().any()
        
        # Remove duplicates
        df_clean = df.loc[:, ~df.columns.duplicated(keep='first')]
        assert not df_clean.columns.duplicated().any()
        assert len(df_clean.columns) == 4
    
    def test_type_validation_reporting(self):
        """Test that type coercion is properly reported."""
        df = pd.DataFrame({
            'track_id': ['1', '2', 'invalid', '4'],
            'frame': ['0', '1', '2', '3'],
            'x': ['10.5', '20.3', '30.1', 'bad'],
            'y': ['15.2', '25.8', '35.4', '45.6']
        })
        
        # Convert with coercion
        for col in ['track_id', 'frame', 'x', 'y']:
            original_count = len(df[col])
            converted = pd.to_numeric(df[col], errors='coerce')
            nan_count = converted.isna().sum()
            
            # Verify conversion tracking
            if col == 'track_id':
                assert nan_count == 1  # 'invalid'
            elif col == 'x':
                assert nan_count == 1  # 'bad'
            else:
                assert nan_count == 0
    
    def test_enhanced_track_validation(self):
        """Test enhanced track validation."""
        from utils import validate_tracks_dataframe
        
        # Valid tracks
        valid_df = pd.DataFrame({
            'track_id': [1, 1, 2, 2],
            'frame': [0, 1, 0, 1],
            'x': [10.0, 11.0, 20.0, 21.0],
            'y': [15.0, 16.0, 25.0, 26.0]
        })
        
        is_valid, message = validate_tracks_dataframe(valid_df)
        assert is_valid
        assert "2 tracks" in message
        
        # Invalid: duplicate track/frame
        invalid_df = pd.DataFrame({
            'track_id': [1, 1, 1],
            'frame': [0, 0, 1],
            'x': [10.0, 10.5, 11.0],
            'y': [15.0, 15.5, 16.0]
        })
        
        is_valid, message = validate_tracks_dataframe(invalid_df, check_duplicates=True)
        assert not is_valid
        assert "duplicate" in message.lower()
        
        # Invalid: negative frames
        negative_frame_df = pd.DataFrame({
            'track_id': [1, 1],
            'frame': [-1, 0],
            'x': [10.0, 11.0],
            'y': [15.0, 16.0]
        })
        
        is_valid, message = validate_tracks_dataframe(negative_frame_df)
        assert not is_valid
        assert "negative" in message.lower()


class TestBatchProcessing:
    """Test batch processing improvements."""
    
    def test_parallel_file_loading(self):
        """Test parallel processing of multiple files."""
        from batch_processing_utils import parallel_process_files
        
        # Create mock file data
        files = []
        for i in range(5):
            df = pd.DataFrame({
                'track_id': [i] * 10,
                'frame': range(10),
                'x': np.random.randn(10),
                'y': np.random.randn(10)
            })
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            files.append({
                'file_name': f'test_{i}.csv',
                'data': csv_buffer.getvalue()
            })
        
        # Define simple loader
        def load_func(file_info):
            try:
                df = pd.read_csv(io.BytesIO(file_info['data']))
                return df, None
            except Exception as e:
                return None, str(e)
        
        # Process in parallel
        results, errors = parallel_process_files(
            files,
            load_func,
            max_workers=2,
            show_progress=False
        )
        
        assert len(results) == 5
        assert len(errors) == 0
    
    def test_batch_progress_tracking(self):
        """Test progress tracking during batch operations."""
        from batch_processing_utils import BatchProcessingProgress
        
        # Create progress tracker (without UI)
        progress = BatchProcessingProgress(total_items=10, description="Testing")
        
        # Simulate processing
        for i in range(10):
            if i == 5:
                progress.update(error="Test error")
            else:
                progress.update()
        
        assert progress.completed_items == 10
        assert progress.failed_items == 1
        assert len(progress.errors) == 1
        
        progress.complete()
    
    def test_dataframe_pooling(self):
        """Test efficient DataFrame pooling."""
        from batch_processing_utils import pool_dataframes_efficiently
        
        # Create multiple DataFrames
        dfs = []
        for i in range(5):
            df = pd.DataFrame({
                'track_id': [i] * 10,
                'frame': range(10),
                'x': np.random.randn(10),
                'y': np.random.randn(10)
            })
            dfs.append(df)
        
        # Pool with validation
        pooled = pool_dataframes_efficiently(dfs, validate=True, deduplicate=True)
        
        assert len(pooled) == 50  # 5 * 10
        assert set(pooled.columns) == {'track_id', 'frame', 'x', 'y'}
        assert pooled['track_id'].nunique() == 5


class TestVisualizationOptimization:
    """Test visualization optimization."""
    
    def test_track_downsampling(self):
        """Test track downsampling."""
        from visualization_optimization import downsample_tracks
        
        # Create large dataset
        df = pd.DataFrame({
            'track_id': np.repeat(range(100), 1000),
            'frame': np.tile(range(1000), 100),
            'x': np.random.randn(100000),
            'y': np.random.randn(100000)
        })
        
        # Downsample
        downsampled = downsample_tracks(
            df,
            max_tracks=50,
            max_points_per_track=100
        )
        
        assert downsampled['track_id'].nunique() <= 50
        for track_id in downsampled['track_id'].unique():
            track = downsampled[downsampled['track_id'] == track_id]
            assert len(track) <= 100
    
    def test_plot_caching(self):
        """Test plot caching mechanism."""
        from visualization_optimization import PlotCache, get_plot_cache_key
        
        cache = PlotCache(max_size=5)
        
        # Create test data
        df = pd.DataFrame({
            'track_id': [1, 1],
            'frame': [0, 1],
            'x': [10.0, 11.0],
            'y': [15.0, 16.0]
        })
        
        # Generate cache keys
        key1 = get_plot_cache_key(df, 'tracks', {'max_tracks': 50})
        key2 = get_plot_cache_key(df, 'tracks', {'max_tracks': 100})
        
        # Keys should be different
        assert key1 != key2
        
        # Test cache operations
        import plotly.graph_objects as go
        fig = go.Figure()
        
        cache.set(key1, fig)
        assert cache.get(key1) is not None
        assert cache.get(key2) is None
        
        # Test cache size limit
        for i in range(10):
            cache.set(f"key_{i}", fig)
        
        assert cache.size() <= 5
    
    def test_colorblind_palettes(self):
        """Test colorblind-safe color palettes."""
        from visualization_optimization import COLORBLIND_SAFE_PALETTES, apply_colorblind_palette
        import plotly.graph_objects as go
        
        # Check palettes exist
        assert 'default' in COLORBLIND_SAFE_PALETTES
        assert 'contrast' in COLORBLIND_SAFE_PALETTES
        assert 'viridis' in COLORBLIND_SAFE_PALETTES
        
        # Test applying palette
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        
        fig_colored = apply_colorblind_palette(fig, 'default')
        assert fig_colored is not None
    
    def test_plot_metadata(self):
        """Test plot metadata annotation."""
        from visualization_optimization import add_plot_metadata
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig = add_plot_metadata(
            fig,
            n_tracks=50,
            n_points=5000,
            downsampled=True,
            cached=False
        )
        
        # Check annotation was added
        assert len(fig.layout.annotations) > 0
        annotation_text = fig.layout.annotations[0].text
        assert '50' in annotation_text
        assert '5000' in annotation_text or '5,000' in annotation_text


class TestLogging:
    """Test logging utilities."""
    
    def test_logging_setup(self):
        """Test logging initialization."""
        from logging_utils import setup_logging, get_logger
        import logging
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = setup_logging(
                log_dir=tmpdir,
                log_level="INFO",
                console_output=False,
                log_to_file=True
            )
            
            assert log_file.exists()
            
            # Test getting logger
            logger = get_logger(__name__)
            assert logger is not None
            
            # Test logging
            logger.info("Test message")
            
            # Check log file content
            log_content = log_file.read_text()
            assert "Test message" in log_content
    
    def test_dataframe_logging(self):
        """Test DataFrame logging utility."""
        from logging_utils import log_dataframe_info
        
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, None]
        })
        
        # Should not raise exception
        log_dataframe_info(df, "Test DataFrame")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
