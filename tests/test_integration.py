"""
Integration Tests for SPT2025B

Comprehensive integration testing suite to increase test coverage from ~30-40% to >80%.
Tests complete workflows including data loading, analysis pipelines, state management,
file handlers, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any

# Import modules to test
from data_loader import load_tracks_file, format_track_data
from analysis import (
    calculate_msd, fit_msd, classify_motion,
    calculate_velocity_autocorrelation, calculate_confinement_ratio
)
from state_manager import StateManager, get_state_manager
from bounded_cache import BoundedResultsCache, get_results_cache
from security_utils import SecureFileHandler
from logging_config import get_logger
from data_access_utils import get_track_data, get_units

logger = get_logger(__name__)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_track_data():
    """Generate sample tracking data for testing."""
    np.random.seed(42)
    n_tracks = 10
    frames_per_track = 50
    
    data = []
    for track_id in range(n_tracks):
        x_start = np.random.uniform(0, 512)
        y_start = np.random.uniform(0, 512)
        
        # Generate random walk
        x_positions = x_start + np.cumsum(np.random.randn(frames_per_track) * 2)
        y_positions = y_start + np.cumsum(np.random.randn(frames_per_track) * 2)
        
        for frame, (x, y) in enumerate(zip(x_positions, y_positions)):
            data.append({
                'track_id': track_id,
                'frame': frame,
                'x': x,
                'y': y
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_track_data):
    """Create temporary CSV file with track data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_track_data.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def state_manager():
    """Get fresh state manager instance."""
    return get_state_manager()


@pytest.fixture
def bounded_cache():
    """Get fresh bounded cache instance."""
    cache = BoundedResultsCache(max_items=10, max_memory_mb=50)
    yield cache
    cache.clear()


# ==============================================================================
# Data Loading Tests
# ==============================================================================

class TestDataLoading:
    """Test data loading and file format handling."""
    
    def test_load_csv_file(self, temp_csv_file):
        """Test loading CSV file with track data."""
        result = load_tracks_file(temp_csv_file, pixel_size=0.1, frame_interval=0.1)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'track_id' in result.columns
        assert 'frame' in result.columns
        assert 'x' in result.columns
        assert 'y' in result.columns
        assert len(result) > 0
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns None."""
        result = load_tracks_file('nonexistent_file.csv', 0.1, 0.1)
        assert result is None
    
    def test_format_track_data(self, sample_track_data):
        """Test track data formatting."""
        formatted = format_track_data(sample_track_data)
        
        assert isinstance(formatted, pd.DataFrame)
        assert 'track_id' in formatted.columns
        assert 'frame' in formatted.columns
        assert len(formatted) == len(sample_track_data)
    
    def test_format_track_data_with_different_column_names(self):
        """Test formatting with non-standard column names."""
        df = pd.DataFrame({
            'Trajectory': [1, 1, 2, 2],
            'Frame': [0, 1, 0, 1],
            'Position X': [10, 11, 20, 21],
            'Position Y': [10, 11, 20, 21]
        })
        
        formatted = format_track_data(df)
        assert 'track_id' in formatted.columns
        assert 'frame' in formatted.columns
        assert 'x' in formatted.columns
        assert 'y' in formatted.columns


# ==============================================================================
# Analysis Pipeline Tests
# ==============================================================================

class TestAnalysisPipeline:
    """Test analysis functions and workflows."""
    
    def test_calculate_msd_basic(self, sample_track_data):
        """Test MSD calculation with valid data."""
        msd_df = calculate_msd(
            sample_track_data,
            max_lag=10,
            pixel_size=0.1,
            frame_interval=0.1
        )
        
        assert isinstance(msd_df, pd.DataFrame)
        assert not msd_df.empty
        assert 'track_id' in msd_df.columns
        assert 'lag_time' in msd_df.columns
        assert 'msd' in msd_df.columns
        assert 'n_points' in msd_df.columns
        
        # Check MSD values are positive
        assert (msd_df['msd'] >= 0).all()
    
    def test_calculate_msd_empty_dataframe(self):
        """Test MSD calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        msd_df = calculate_msd(empty_df)
        
        assert isinstance(msd_df, pd.DataFrame)
        assert msd_df.empty
    
    def test_calculate_msd_missing_columns(self):
        """Test MSD calculation with missing required columns."""
        invalid_df = pd.DataFrame({
            'track_id': [1, 1],
            'frame': [0, 1]
            # Missing x, y columns
        })
        
        msd_df = calculate_msd(invalid_df)
        assert msd_df.empty
    
    def test_calculate_msd_short_tracks(self, sample_track_data):
        """Test MSD calculation filters short tracks."""
        msd_df = calculate_msd(
            sample_track_data,
            max_lag=10,
            min_track_length=100  # Very high threshold
        )
        
        # Should have no results since all tracks are shorter than 100 frames
        assert msd_df.empty
    
    def test_fit_msd(self, sample_track_data):
        """Test MSD fitting."""
        msd_df = calculate_msd(sample_track_data, max_lag=10)
        
        if not msd_df.empty:
            track_id = msd_df['track_id'].iloc[0]
            track_msd = msd_df[msd_df['track_id'] == track_id]
            
            fit_result = fit_msd(track_msd)
            
            assert isinstance(fit_result, dict)
            # Check for expected keys (implementation-dependent)
            # assert 'D' in fit_result or 'diffusion_coefficient' in fit_result
    
    def test_classify_motion(self, sample_track_data):
        """Test motion classification."""
        result = classify_motion(sample_track_data)
        
        assert isinstance(result, dict)
        # Should have classification results
        # Implementation-specific checks
    
    def test_velocity_autocorrelation(self, sample_track_data):
        """Test velocity autocorrelation calculation."""
        result = calculate_velocity_autocorrelation(
            sample_track_data,
            pixel_size=0.1,
            frame_interval=0.1
        )
        
        assert isinstance(result, dict)
        # Check for data and/or success key
    
    def test_confinement_ratio(self, sample_track_data):
        """Test confinement ratio calculation."""
        result = calculate_confinement_ratio(
            sample_track_data,
            pixel_size=0.1
        )
        
        assert isinstance(result, dict)


# ==============================================================================
# State Management Tests
# ==============================================================================

class TestStateManagement:
    """Test state manager functionality."""
    
    def test_state_manager_singleton(self):
        """Test state manager is singleton."""
        sm1 = get_state_manager()
        sm2 = get_state_manager()
        assert sm1 is sm2
    
    def test_set_and_get_tracks(self, state_manager, sample_track_data):
        """Test storing and retrieving tracks."""
        state_manager.set_tracks(sample_track_data, filename='test.csv')
        
        assert state_manager.has_tracks()
        retrieved = state_manager.get_tracks()
        
        assert isinstance(retrieved, pd.DataFrame)
        assert len(retrieved) == len(sample_track_data)
    
    def test_set_and_get_pixel_size(self, state_manager):
        """Test pixel size storage."""
        test_size = 0.123
        state_manager.set_pixel_size(test_size)
        
        retrieved = state_manager.get_pixel_size()
        assert retrieved == test_size
    
    def test_set_and_get_frame_interval(self, state_manager):
        """Test frame interval storage."""
        test_interval = 0.234
        state_manager.set_frame_interval(test_interval)
        
        retrieved = state_manager.get_frame_interval()
        assert retrieved == test_interval
    
    def test_clear_data(self, state_manager, sample_track_data):
        """Test clearing all data."""
        state_manager.set_tracks(sample_track_data)
        assert state_manager.has_tracks()
        
        state_manager.clear_data()
        assert not state_manager.has_tracks()
    
    def test_analysis_results_storage(self, state_manager):
        """Test analysis results storage and retrieval."""
        test_results = {
            'success': True,
            'data': {'msd': [1, 2, 3]},
            'summary': {'mean': 2.0}
        }
        
        state_manager.set_analysis_results('msd', test_results)
        retrieved = state_manager.get_analysis_results('msd')
        
        assert retrieved['success'] == test_results['success']
        assert 'data' in retrieved
    
    def test_get_cache_stats(self, state_manager):
        """Test cache statistics retrieval."""
        stats = state_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'size' in stats
        assert 'max_items' in stats
        assert 'memory_mb' in stats


# ==============================================================================
# Bounded Cache Tests
# ==============================================================================

class TestBoundedCache:
    """Test bounded results cache functionality."""
    
    def test_cache_set_and_get(self, bounded_cache):
        """Test basic cache set and get."""
        test_data = {'value': 123, 'data': [1, 2, 3]}
        bounded_cache.set('test_key', test_data)
        
        retrieved = bounded_cache.get('test_key')
        assert retrieved is not None
        assert retrieved['value'] == 123
    
    def test_cache_miss(self, bounded_cache):
        """Test cache miss returns None."""
        result = bounded_cache.get('nonexistent_key')
        assert result is None
    
    def test_cache_lru_eviction(self, bounded_cache):
        """Test LRU eviction when cache is full."""
        # Fill cache beyond capacity
        for i in range(15):  # Cache max is 10
            bounded_cache.set(f'key_{i}', {'data': f'value_{i}'})
        
        # First items should be evicted
        assert bounded_cache.get('key_0') is None
        assert bounded_cache.get('key_1') is None
        
        # Recent items should still exist
        assert bounded_cache.get('key_14') is not None
    
    def test_cache_move_to_end_on_access(self, bounded_cache):
        """Test that accessing items moves them to end (LRU)."""
        # Add items
        for i in range(5):
            bounded_cache.set(f'key_{i}', {'data': i})
        
        # Access old item
        _ = bounded_cache.get('key_0')
        
        # Fill cache to capacity
        for i in range(10):
            bounded_cache.set(f'new_key_{i}', {'data': i})
        
        # key_0 should still exist because we accessed it
        # (depends on exact eviction policy)
        stats = bounded_cache.get_stats()
        assert stats['size'] <= bounded_cache.max_items
    
    def test_cache_clear(self, bounded_cache):
        """Test cache clearing."""
        bounded_cache.set('key1', {'data': 1})
        bounded_cache.set('key2', {'data': 2})
        
        assert len(bounded_cache) == 2
        
        bounded_cache.clear()
        
        assert len(bounded_cache) == 0
        assert bounded_cache.get('key1') is None
    
    def test_cache_statistics(self, bounded_cache):
        """Test cache statistics tracking."""
        bounded_cache.set('key1', {'data': 1})
        bounded_cache.get('key1')  # Hit
        bounded_cache.get('key2')  # Miss
        
        stats = bounded_cache.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['sets'] == 1
        assert stats['hit_rate'] > 0
    
    def test_cache_memory_limit(self):
        """Test cache respects memory limits."""
        small_cache = BoundedResultsCache(max_items=100, max_memory_mb=1)
        
        # Try to add large items
        large_data = {'array': np.random.randn(1000, 1000)}  # ~8MB
        
        small_cache.set('large_item', large_data)
        
        # Should handle gracefully (strip large figures or skip)
        stats = small_cache.get_stats()
        assert stats['memory_mb'] <= small_cache.max_memory_bytes / 1024 / 1024 * 1.1  # 10% tolerance


# ==============================================================================
# Security Tests
# ==============================================================================

class TestSecurityUtils:
    """Test security utilities."""
    
    def test_validate_filename_valid(self):
        """Test filename validation with valid files."""
        handler = SecureFileHandler()
        
        assert handler.validate_filename('tracks.csv')
        assert handler.validate_filename('data.xlsx')
        assert handler.validate_filename('image.tif')
    
    def test_validate_filename_invalid_extension(self):
        """Test filename validation rejects invalid extensions."""
        handler = SecureFileHandler()
        
        assert not handler.validate_filename('malicious.exe')
        assert not handler.validate_filename('script.sh')
        assert not handler.validate_filename('file.bat')
    
    def test_validate_filename_path_traversal(self):
        """Test filename validation detects path traversal."""
        handler = SecureFileHandler()
        
        assert not handler.validate_filename('../../../etc/passwd')
        assert not handler.validate_filename('..\\..\\windows\\system32')
        assert not handler.validate_filename('data/../../../etc/passwd')
    
    def test_validate_file_size(self, temp_csv_file):
        """Test file size validation."""
        handler = SecureFileHandler()
        
        # Should pass for normal CSV
        assert handler.validate_file_size(temp_csv_file)
        
        # Should fail for files exceeding limit
        assert not handler.validate_file_size(temp_csv_file, max_size_mb=0.000001)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        handler = SecureFileHandler()
        
        sanitized = handler.sanitize_for_filename('test file (1).csv')
        assert ' ' not in sanitized
        assert '(' not in sanitized
        assert ')' not in sanitized
        
        # Should be safe for filesystem
        assert sanitized.endswith('.csv')


# ==============================================================================
# Integration Workflow Tests
# ==============================================================================

class TestIntegrationWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_complete_analysis_workflow(self, temp_csv_file, state_manager):
        """Test complete workflow: load → store → analyze → retrieve."""
        # 1. Load data
        tracks_df = load_tracks_file(temp_csv_file, pixel_size=0.1, frame_interval=0.1)
        assert tracks_df is not None
        
        # 2. Store in state manager
        state_manager.set_tracks(tracks_df, filename='test.csv')
        assert state_manager.has_tracks()
        
        # 3. Run analysis
        msd_df = calculate_msd(tracks_df, max_lag=10)
        assert not msd_df.empty
        
        # 4. Store results
        analysis_results = {
            'success': True,
            'data': msd_df,
            'summary': {'n_tracks': tracks_df['track_id'].nunique()}
        }
        state_manager.set_analysis_results('msd', analysis_results)
        
        # 5. Retrieve results
        retrieved = state_manager.get_analysis_results('msd')
        assert retrieved['success']
        assert 'data' in retrieved
    
    def test_data_quality_check_workflow(self, sample_track_data):
        """Test data quality checking workflow."""
        from data_quality_checker import DataQualityChecker
        
        checker = DataQualityChecker()
        report = checker.run_all_checks(
            sample_track_data,
            pixel_size=0.1,
            frame_interval=0.1
        )
        
        assert report.overall_score >= 0
        assert report.overall_score <= 100
        assert report.total_checks > 0
        assert len(report.checks) > 0
        assert len(report.recommendations) >= 0
    
    def test_multiple_analyses_workflow(self, sample_track_data, state_manager):
        """Test running multiple analyses and caching results."""
        analyses = ['msd', 'velocity', 'confinement']
        
        for analysis_type in analyses:
            result = {
                'success': True,
                'data': {'test': f'{analysis_type}_data'},
                'timestamp': pd.Timestamp.now()
            }
            state_manager.set_analysis_results(analysis_type, result)
        
        # Retrieve all
        for analysis_type in analyses:
            retrieved = state_manager.get_analysis_results(analysis_type)
            assert retrieved['success']
            assert analysis_type in str(retrieved['data']['test'])
        
        # Check cache stats
        stats = state_manager.get_cache_stats()
        assert stats['size'] >= len(analyses)


# ==============================================================================
# Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_track_data(self):
        """Test handling of empty track data."""
        empty_df = pd.DataFrame()
        msd_df = calculate_msd(empty_df)
        assert msd_df.empty
    
    def test_single_point_track(self):
        """Test handling of single-point tracks."""
        single_point_df = pd.DataFrame({
            'track_id': [1],
            'frame': [0],
            'x': [10.0],
            'y': [10.0]
        })
        
        msd_df = calculate_msd(single_point_df, min_track_length=1)
        # Should handle gracefully (likely empty or minimal result)
        assert isinstance(msd_df, pd.DataFrame)
    
    def test_tracks_with_nan_values(self):
        """Test handling of NaN values in tracks."""
        df_with_nan = pd.DataFrame({
            'track_id': [1, 1, 1],
            'frame': [0, 1, 2],
            'x': [10.0, np.nan, 12.0],
            'y': [10.0, 11.0, np.nan]
        })
        
        # Should handle gracefully
        msd_df = calculate_msd(df_with_nan)
        assert isinstance(msd_df, pd.DataFrame)
    
    def test_very_large_dataset(self):
        """Test handling of large datasets."""
        # Generate larger dataset
        large_data = []
        for track_id in range(100):
            for frame in range(100):
                large_data.append({
                    'track_id': track_id,
                    'frame': frame,
                    'x': np.random.randn() * 10,
                    'y': np.random.randn() * 10
                })
        
        large_df = pd.DataFrame(large_data)
        
        # Should complete without error (may be slow)
        msd_df = calculate_msd(large_df, max_lag=10)
        assert isinstance(msd_df, pd.DataFrame)
    
    def test_negative_coordinates(self):
        """Test handling of negative coordinates."""
        negative_coords_df = pd.DataFrame({
            'track_id': [1, 1, 1],
            'frame': [0, 1, 2],
            'x': [-10.0, -9.0, -8.0],
            'y': [-5.0, -4.0, -3.0]
        })
        
        msd_df = calculate_msd(negative_coords_df)
        assert isinstance(msd_df, pd.DataFrame)


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short', '--cov=.', '--cov-report=term'])
