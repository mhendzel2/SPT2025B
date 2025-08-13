import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import load_tracks_file
from analysis import calculate_msd, analyze_diffusion, analyze_motion
from visualization import plot_tracks, plot_msd_curves, plot_diffusion_coefficients
from utils import format_track_data, calculate_track_statistics

@pytest.fixture
def sample_track_data():
    """Fixture to load sample track data."""
    # Create a dummy file in memory
    from io import StringIO

    csv_data = """track_id,frame,x,y,intensity
1,0,10,10,100
1,1,11,12,110
1,2,12,14,120
1,3,13,16,130
1,4,14,18,140
2,0,20,20,150
2,1,21,22,160
2,2,22,24,170
2,3,23,26,180
2,4,24,28,190
3,0,30,30,200
3,1,31,32,210
3,2,32,34,220
3,3,33,36,230
3,4,34,38,240
"""
    # Create a dummy file object
    from unittest.mock import MagicMock
    file_mock = MagicMock()
    file_mock.name = "sample_tracks.csv"
    file_mock.getvalue.return_value = csv_data.encode('utf-8')

    return load_tracks_file(file_mock)

def test_load_sample_data(sample_track_data):
    """Test that the sample data is loaded correctly."""
    df = sample_track_data
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_columns = ['track_id', 'frame', 'x', 'y', 'intensity']
    assert all(col in df.columns for col in expected_columns)
    assert df['track_id'].nunique() == 3
    assert len(df) == 15

def test_calculate_msd(sample_track_data):
    """Test the calculate_msd function."""
    msd_df = calculate_msd(sample_track_data, max_lag=2, frame_interval=0.1)
    assert isinstance(msd_df, pd.DataFrame)
    assert not msd_df.empty
    expected_columns = ['track_id', 'lag_time', 'msd', 'n_points']
    assert all(col in msd_df.columns for col in expected_columns)
    # Check that the number of tracks is correct
    assert msd_df['track_id'].nunique() == 3
    # Check that the lag times are correct
    assert msd_df['lag_time'].nunique() == 2
    assert msd_df['lag_time'].min() == 0.1
    assert msd_df['lag_time'].max() == 0.2

def test_analyze_diffusion(sample_track_data):
    """Test the analyze_diffusion function."""
    diffusion_results = analyze_diffusion(sample_track_data, min_track_length=5)
    assert isinstance(diffusion_results, dict)
    assert diffusion_results['success']
    assert 'msd_data' in diffusion_results
    assert 'track_results' in diffusion_results
    assert 'ensemble_results' in diffusion_results
    assert isinstance(diffusion_results['track_results'], pd.DataFrame)
    assert not diffusion_results['track_results'].empty
    assert 'diffusion_coefficient' in diffusion_results['track_results'].columns
    if 'alpha' in diffusion_results['track_results'].columns:
        assert 'alpha' in diffusion_results['track_results'].columns

def test_analyze_motion(sample_track_data):
    """Test the analyze_motion function."""
    motion_results = analyze_motion(sample_track_data, min_track_length=5)
    assert isinstance(motion_results, dict)
    assert motion_results['success']
    assert 'track_results' in motion_results
    assert 'ensemble_results' in motion_results
    assert isinstance(motion_results['track_results'], pd.DataFrame)
    assert not motion_results['track_results'].empty
    assert 'mean_speed' in motion_results['track_results'].columns

def test_plot_tracks(sample_track_data):
    """Test the plot_tracks function."""
    fig = plot_tracks(sample_track_data)
    assert fig is not None
    assert isinstance(fig, go.Figure)

def test_plot_msd_curves(sample_track_data):
    """Test the plot_msd_curves function."""
    msd_df = calculate_msd(sample_track_data)
    fig = plot_msd_curves(msd_df)
    assert fig is not None
    assert isinstance(fig, go.Figure)

def test_plot_diffusion_coefficients(sample_track_data):
    """Test the plot_diffusion_coefficients function."""
    diffusion_results = analyze_diffusion(sample_track_data, min_track_length=5)
    fig = plot_diffusion_coefficients(diffusion_results['track_results'])
    assert fig is not None
    assert isinstance(fig, go.Figure)
