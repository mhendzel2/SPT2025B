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
