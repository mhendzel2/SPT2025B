"""Regression tests for biophysical utilities merged from corrected module."""

import numpy as np
import pandas as pd

from biophysical_models import calculate_msd_variance


def _make_tracks(n_tracks: int = 5, n_frames: int = 50, D: float = 0.1, dt: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    sigma_step = np.sqrt(2 * D * dt)
    rows = []

    for track_id in range(n_tracks):
        x = np.cumsum(rng.normal(0, sigma_step, size=n_frames))
        y = np.cumsum(rng.normal(0, sigma_step, size=n_frames))
        for frame in range(n_frames):
            rows.append(
                {
                    "track_id": track_id,
                    "frame": frame,
                    "x": x[frame] / 0.1,  # convert to pixels for pixel_size=0.1
                    "y": y[frame] / 0.1,
                }
            )

    return pd.DataFrame(rows)


def test_calculate_msd_variance_shapes():
    tracks_df = _make_tracks()
    lag_times, msd_mean, msd_var = calculate_msd_variance(
        tracks_df, pixel_size=0.1, frame_interval=0.1, max_lag=20
    )

    assert len(lag_times) == 20
    assert len(msd_mean) == 20
    assert len(msd_var) == 20
    assert np.isfinite(msd_mean).sum() > 0


def test_calculate_msd_variance_monotonic_lags():
    tracks_df = _make_tracks()
    lag_times, _, _ = calculate_msd_variance(
        tracks_df, pixel_size=0.1, frame_interval=0.2, max_lag=15
    )

    assert np.all(np.diff(lag_times) > 0)
    assert lag_times[0] == 0.2
