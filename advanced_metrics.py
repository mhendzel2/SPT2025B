import numpy as np
import pandas as pd
from typing import Dict

def non_gaussian_parameter_2d(displacements: np.ndarray) -> float:
    dr2 = displacements**2
    num = np.mean(dr2**2)
    den = 2.0 * (np.mean(dr2)**2 + 1e-30)
    return float(num / den - 1.0)

def van_hove_distribution(dx: np.ndarray, bins: int = 50) -> Dict[str, np.ndarray]:
    hist, edges = np.histogram(dx, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return {"centers": centers, "density": hist}

def time_vs_ensemble_msds(tracks_df: pd.DataFrame, lag: int, pixel_size: float = 1.0) -> Dict[str, float]:
    if lag < 1:
        return {"TAMSD": np.nan, "EAMSD": np.nan, "EB_ratio": np.nan}
    dr2_all = []
    tamsd_vals = []
    for _, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame")
        x = g["x"].values * pixel_size
        y = g["y"].values * pixel_size
        if len(x) <= lag:
            continue
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        dr2 = dx*dx + dy*dy
        dr2_all.append(dr2)
        tamsd_vals.append(float(np.mean(dr2)))
    if not dr2_all:
        return {"TAMSD": np.nan, "EAMSD": np.nan, "EB_ratio": np.nan}
    cat = np.concatenate(dr2_all)
    eamsd = float(np.mean(cat))
    tamsd = float(np.mean(tamsd_vals))
    return {"TAMSD": tamsd, "EAMSD": eamsd, "EB_ratio": tamsd / (eamsd + 1e-30)}

def calculate_full_vacf(tracks_df: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0) -> pd.DataFrame:
    """
    Calculate the ensemble-averaged Velocity Autocorrelation Function (VACF)
    for both positive and negative lags.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns 'track_id', 'frame', 'x', 'y'.
    pixel_size : float
        Size of a pixel in microns (or other unit).
    frame_interval : float
        Time between frames in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'lag_time' and 'vacf'.
        Includes negative lags (symmetric to positive lags).
    """
    all_vacfs = {} # lag_index -> list of vacf values

    # Check if we have necessary columns
    if tracks_df.empty or not all(c in tracks_df.columns for c in ['track_id', 'frame', 'x', 'y']):
        return pd.DataFrame(columns=['lag_time', 'vacf'])

    for track_id, track in tracks_df.groupby('track_id'):
        if len(track) < 3:
            continue

        # Sort by frame just in case
        track = track.sort_values('frame')

        # Calculate velocities
        x = track['x'].values * pixel_size
        y = track['y'].values * pixel_size
        t = track['frame'].values * frame_interval

        # Simple finite difference velocity
        # v(t) = (r(t+dt) - r(t)) / dt
        # This velocity is defined for the interval between frames
        dt = np.diff(t)

        # Filter out jumps with zero time difference to avoid division by zero
        valid_steps = dt > 0
        if not np.any(valid_steps):
            continue

        vx = np.diff(x)[valid_steps] / dt[valid_steps]
        vy = np.diff(y)[valid_steps] / dt[valid_steps]

        # Assuming roughly constant frame interval for lag calculation based on index
        # This approximates lag_time = lag_index * mean(frame_interval)

        n = len(vx)
        # Calculate for lags up to n/2
        max_lag = max(1, n // 2)

        for lag in range(max_lag + 1):
            if lag == 0:
                val = np.mean(vx*vx + vy*vy)
            else:
                # Correlate v[i] with v[i+lag]
                if n > lag:
                    val = np.mean(vx[:-lag]*vx[lag:] + vy[:-lag]*vy[lag:])
                else:
                    continue

            if lag not in all_vacfs:
                all_vacfs[lag] = []
            all_vacfs[lag].append(val)

    # Ensemble average
    lags = []
    vacf_values = []

    sorted_lags = sorted(all_vacfs.keys())
    for lag in sorted_lags:
        if all_vacfs[lag]:
            lags.append(lag * frame_interval)
            vacf_values.append(np.mean(all_vacfs[lag]))

    if not lags:
        return pd.DataFrame(columns=['lag_time', 'vacf'])

    # Create symmetric dataframe
    # Convert lists to arrays
    pos_lags = np.array(lags)
    pos_vacf = np.array(vacf_values)

    # Negative lags (excluding 0)
    if len(pos_lags) > 1:
        neg_lags = -pos_lags[1:][::-1]
        neg_vacf = pos_vacf[1:][::-1]

        full_lags = np.concatenate([neg_lags, pos_lags])
        full_vacf = np.concatenate([neg_vacf, pos_vacf])
    else:
        full_lags = pos_lags
        full_vacf = pos_vacf

    return pd.DataFrame({'lag_time': full_lags, 'vacf': full_vacf})
