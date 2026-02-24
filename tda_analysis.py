import numpy as np
import pandas as pd
from typing import Dict, Any, List
import warnings

try:
    from giotto_tda.homology import VietorisRipsPersistence
    from giotto_tda.diagrams import PersistenceDiagram
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser not available. Time-windowed persistence analysis is disabled.")

def perform_tda(points: np.ndarray, max_edge_length: float = 10.0) -> Dict[str, Any]:
    """
    Perform Topological Data Analysis (TDA) on a point cloud.

    Args:
        points: A NumPy array of shape (n_points, n_dims) representing the point cloud.
        max_edge_length: The maximum edge length for the Vietoris-Rips complex.

    Returns:
        A dictionary containing the persistence diagram.
    """
    if not TDA_AVAILABLE:
        raise RuntimeError("giotto-tda is not available.")

    if points.shape[0] < 3:
        return {'success': False, 'error': 'Not enough points for TDA.'}

    # Vietoris-Rips persistence
    vr_persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=max_edge_length)
    diagrams = vr_persistence.fit_transform([points])

    return {
        'success': True,
        'diagram': diagrams[0],
        'homology_dimensions': [0, 1, 2]
    }


def compute_time_windowed_persistence(df: pd.DataFrame, window_size: int = 50) -> List[Dict[str, Any]]:
    """
    Compute rolling-window persistence summaries from spatiotemporal SPT data.

    For each overlapping temporal window, the function builds a point cloud from
    all active particle positions in ``(x, y)``, computes persistent homology via
    ``ripser``, and reports maximum Betti-1 persistence lifespan.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing at least ``frame``, ``x``, and ``y`` columns.
    window_size : int, optional
        Number of unique frames per window, by default 50.

    Returns
    -------
    list of dict
        One dictionary per window with frame span and Betti-1 lifespan summary.
    """
    required_cols = {'frame', 'x', 'y'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"df missing required columns: {missing}")
    if window_size < 2:
        raise ValueError("window_size must be >= 2")

    if not RIPSER_AVAILABLE:
        warnings.warn("compute_time_windowed_persistence skipped because ripser is unavailable.")
        return []

    work_df = df[['frame', 'x', 'y']].dropna().copy()
    if work_df.empty:
        return []

    work_df = work_df.sort_values('frame')
    unique_frames = np.sort(work_df['frame'].unique())
    if unique_frames.size < window_size:
        return []

    stride = max(1, window_size // 2)
    summaries: List[Dict[str, Any]] = []

    for start_idx in range(0, unique_frames.size - window_size + 1, stride):
        window_frames = unique_frames[start_idx:start_idx + window_size]
        frame_start = float(window_frames[0])
        frame_end = float(window_frames[-1])

        mask = (work_df['frame'] >= frame_start) & (work_df['frame'] <= frame_end)
        points = work_df.loc[mask, ['x', 'y']].to_numpy(dtype=float)

        if points.shape[0] < 3:
            summaries.append({
                'window_index': int(start_idx // stride),
                'frame_start': frame_start,
                'frame_end': frame_end,
                'n_points': int(points.shape[0]),
                'max_betti0_lifespan': 0.0,
                'max_betti1_lifespan': 0.0,
                'betti1_feature_count': 0,
                'success': False,
                'error': 'Insufficient points for persistence computation'
            })
            continue

        try:
            result = ripser(points, maxdim=1)
            diagrams = result.get('dgms', [])
            dgms0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))
            dgms1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))

            if dgms0.size > 0:
                b0_lifespans = dgms0[:, 1] - dgms0[:, 0]
                b0_lifespans = b0_lifespans[np.isfinite(b0_lifespans)]
                max_betti0 = float(np.max(b0_lifespans)) if b0_lifespans.size else 0.0
            else:
                max_betti0 = 0.0

            if dgms1.size > 0:
                b1_lifespans = dgms1[:, 1] - dgms1[:, 0]
                b1_lifespans = b1_lifespans[np.isfinite(b1_lifespans)]
                max_betti1 = float(np.max(b1_lifespans)) if b1_lifespans.size else 0.0
                b1_count = int(b1_lifespans.size)
            else:
                max_betti1 = 0.0
                b1_count = 0

            summaries.append({
                'window_index': int(start_idx // stride),
                'frame_start': frame_start,
                'frame_end': frame_end,
                'n_points': int(points.shape[0]),
                'max_betti0_lifespan': max_betti0,
                'max_betti1_lifespan': max_betti1,
                'betti1_feature_count': b1_count,
                'success': True
            })
        except Exception as exc:
            summaries.append({
                'window_index': int(start_idx // stride),
                'frame_start': frame_start,
                'frame_end': frame_end,
                'n_points': int(points.shape[0]),
                'max_betti0_lifespan': np.nan,
                'max_betti1_lifespan': np.nan,
                'betti1_feature_count': 0,
                'success': False,
                'error': str(exc)
            })

    return summaries
