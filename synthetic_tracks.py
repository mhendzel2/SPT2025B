from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Phase:
    n: int          # frames
    D: float        # μm²/s
    vx: float = 0.0 # μm/s
    vy: float = 0.0 # μm/s

def simulate_piecewise_track(
    phases: List[Phase],
    dt: float = 0.1,
    pixel_size: float = 0.1,
    sigma_loc: float = 0.0,
    track_id: int = 1,
    seed: Optional[int] = 1234,
    x0_um: float = 0.0,
    y0_um: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        track dataframe (pixels) + truth segment dataframe
    """
    rng = np.random.default_rng(seed)
    xs, ys, frames = [], [], []
    x, y = x0_um, y0_um
    truth_rows = []
    t0 = 0
    for ph in phases:
        n = int(ph.n)
        if n <= 0:
            continue
        sig = np.sqrt(2.0 * ph.D * dt)
        for _ in range(n):
            dx = ph.vx * dt + sig * rng.standard_normal()
            dy = ph.vy * dt + sig * rng.standard_normal()
            x += dx; y += dy
            if sigma_loc > 0:
                x_obs = x + sigma_loc * rng.standard_normal()
                y_obs = y + sigma_loc * rng.standard_normal()
            else:
                x_obs, y_obs = x, y
            xs.append(x_obs / pixel_size)
            ys.append(y_obs / pixel_size)
            frames.append(len(frames))
        truth_rows.append({
            'start_frame': t0,
            'end_frame': t0 + n - 1,
            'D': ph.D,
            'vx': ph.vx,
            'vy': ph.vy
        })
        t0 += n
    df = pd.DataFrame({
        'track_id': track_id,
        'frame': np.asarray(frames, dtype=int),
        'x': np.asarray(xs, dtype=float),
        'y': np.asarray(ys, dtype=float)
    })
    truth = pd.DataFrame(truth_rows)
    return df, truth

def make_dataset(
    n_tracks: int = 3,
    phases: Optional[List[Phase]] = None,
    dt: float = 0.1,
    pixel_size: float = 0.1,
    sigma_loc: float = 0.02,
    seed: int = 1234
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    if phases is None:
        phases = [
            Phase(n=120, D=0.02),
            Phase(n=120, D=0.06),
            Phase(n=120, D=0.02, vx=0.20)
        ]
    rng = np.random.default_rng(seed)
    all_tracks = []
    truths: Dict[int, pd.DataFrame] = {}
    for tid in range(1, n_tracks + 1):
        df, truth = simulate_piecewise_track(
            phases=phases, dt=dt, pixel_size=pixel_size,
            sigma_loc=sigma_loc, track_id=tid,
            seed=int(rng.integers(0, 1_000_000))
        )
        all_tracks.append(df)
        truths[tid] = truth
    return pd.concat(all_tracks, ignore_index=True), truths

# ---------------- Metrics ----------------

def boundaries_from_segments(segments: pd.DataFrame) -> List[int]:
    if segments is None or segments.empty:
        return []
    ends = sorted(int(e) for e in segments['end_frame'])
    return ends[:-1] if len(ends) > 0 else []

def boundaries_from_truth(truth: pd.DataFrame) -> List[int]:
    if truth is None or truth.empty:
        return []
    ends = sorted(int(e) for e in truth['end_frame'])
    return ends[:-1] if len(ends) > 0 else []

def boundary_f1(truth_bounds: List[int], pred_bounds: List[int], tol: int = 3) -> Dict[str, float]:
    tb = sorted(truth_bounds)
    pb = sorted(pred_bounds)
    if not tb and not pb:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not pb:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    used = set()
    tp = 0
    for b in pb:
        candidates = [t for t in tb if abs(t - b) <= tol and t not in used]
        if not candidates:
            continue
        tbest = min(candidates, key=lambda t: abs(t - b))
        used.add(tbest)
        tp += 1
    fp = len(pb) - tp
    fn = len(tb) - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1}
