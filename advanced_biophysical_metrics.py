from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class MetricConfig:
    pixel_size: float = 1.0
    frame_interval: float = 1.0
    min_track_length: int = 5
    max_lag: int = 20
    log_lag: bool = True
    n_hist_bins: int = 60
    seed: Optional[int] = None
    n_bootstrap: int = 500  # 0 disables CIs

class AdvancedMetricsAnalyzer:
    """
    Advanced SPT metrics:
      NGP (1D/2D), van Hove, TAMSD/EAMSD, ergodicity (EB ratio & parameter),
      VACF, turning angles, Hurst exponent from TAMSD scaling.
    tracks_df columns required: track_id, frame, x, y (pixel units).
    """
    def __init__(self, tracks_df: pd.DataFrame, config: MetricConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        needed = {'track_id','frame','x','y'}
        if not needed.issubset(tracks_df.columns):
            missing = needed - set(tracks_df.columns)
            raise ValueError(f"tracks_df missing columns: {missing}")
        df = (tracks_df[['track_id','frame','x','y']].dropna().copy())
        df['frame'] = df['frame'].astype(int)
        df = df.sort_values(['track_id','frame'])
        df = df.groupby('track_id').filter(lambda g: len(g) >= self.cfg.min_track_length)
        self.df = df.reset_index(drop=True)
        if self.df.empty:
            self.lags = np.array([], dtype=int)
            return
        self.df['x_um'] = self.df['x'] * self.cfg.pixel_size
        self.df['y_um'] = self.df['y'] * self.cfg.pixel_size
        self.lags = self._choose_lags()

    # -------- lag selection --------
    def _choose_lags(self) -> np.ndarray:
        max_len = self.df.groupby('track_id').size().max()
        max_possible = int(min(self.cfg.max_lag, max(1, max_len - 1)))
        if max_possible < 1:
            return np.array([], dtype=int)
        if (not self.cfg.log_lag) or max_possible <= 6:
            return np.arange(1, max_possible+1, dtype=int)
        lags = np.unique(np.geomspace(1, max_possible, num=min(15, max_possible)).astype(int))
        return lags[lags >= 1]

    # -------- displacement helpers --------
    def _displacements_for_lag(self, g: pd.DataFrame, lag: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        x = g['x_um'].values; y = g['y_um'].values
        if len(x) <= lag:
            return np.array([]), np.array([]), np.array([])
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        dr = np.sqrt(dx*dx + dy*dy)
        return dx, dy, dr

    def _collect_displacements(self, lag: int):
        dx_all, dy_all, dr_all = [], [], []
        n_pairs = 0
        for _, g in self.df.groupby('track_id'):
            dx, dy, dr = self._displacements_for_lag(g, lag)
            if dx.size:
                dx_all.append(dx); dy_all.append(dy); dr_all.append(dr)
                n_pairs += dx.size
        if not dx_all:
            return np.array([]), np.array([]), np.array([]), 0
        return (np.concatenate(dx_all),
                np.concatenate(dy_all),
                np.concatenate(dr_all),
                n_pairs)

    # -------- TAMSD / EAMSD --------
    def tamsd_eamsd(self):
        rows_t, rows_e = [], []
        for lag in self.lags:
            dr2_all = []
            tau_s = lag * self.cfg.frame_interval
            for tid, g in self.df.groupby('track_id'):
                dx, dy, dr = self._displacements_for_lag(g, lag)
                if not dr.size:
                    continue
                dr2 = dr*dr
                rows_t.append({'track_id': tid, 'lag': lag, 'tau_s': tau_s, 'tamsd': float(np.mean(dr2))})
                dr2_all.append(dr2)
            if dr2_all:
                rows_e.append({'lag': lag, 'tau_s': tau_s, 'eamsd': float(np.mean(np.concatenate(dr2_all)))})
        return pd.DataFrame(rows_t), pd.DataFrame(rows_e)

    # -------- Ergodicity --------
    def ergodicity_measures(self, tamsd_df: pd.DataFrame, eamsd_df: pd.DataFrame):
        out = []
        for lag in self.lags:
            tau_s = lag * self.cfg.frame_interval
            eamsd_row = eamsd_df[eamsd_df['lag']==lag]
            if eamsd_row.empty:
                out.append({'lag': lag, 'tau_s': tau_s, 'EB_ratio': np.nan, 'EB_parameter': np.nan})
                continue
            eamsd = float(eamsd_row['eamsd'])
            vals = tamsd_df.loc[tamsd_df['lag']==lag, 'tamsd'].values
            if not vals.size or not np.isfinite(eamsd):
                out.append({'lag': lag, 'tau_s': tau_s, 'EB_ratio': np.nan, 'EB_parameter': np.nan})
                continue
            mean_tamsd = float(np.mean(vals))
            eb_ratio = mean_tamsd / (eamsd + 1e-30)
            norm = vals / (mean_tamsd + 1e-30)
            eb_param = float(np.mean((norm - 1.0)**2))
            row = {'lag': lag, 'tau_s': tau_s, 'EB_ratio': eb_ratio, 'EB_parameter': eb_param}
            if self.cfg.n_bootstrap and vals.size > 5:
                eb_r_s, eb_p_s = [], []
                for _ in range(self.cfg.n_bootstrap):
                    bs = self.rng.choice(vals, size=vals.size, replace=True)
                    m = float(np.mean(bs))
                    eb_r_s.append(m / (eamsd + 1e-30))
                    eb_p_s.append(np.mean((bs/(m+1e-30) - 1.0)**2))
                lo_r, hi_r = np.percentile(eb_r_s, [2.5,97.5])
                lo_p, hi_p = np.percentile(eb_p_s, [2.5,97.5])
                row.update({'EB_ratio_CI_low': float(lo_r), 'EB_ratio_CI_high': float(hi_r),
                            'EB_param_CI_low': float(lo_p), 'EB_param_CI_high': float(hi_p)})
            out.append(row)
        return pd.DataFrame(out)

    # -------- NGP --------
    @staticmethod
    def _ngp_1d(dx: np.ndarray) -> float:
        if dx.size == 0: return np.nan
        m2 = np.mean(dx*dx)
        if m2 <= 0: return np.nan
        m4 = np.mean((dx*dx)**2)
        return float(m4 / (3*m2*m2) - 1.0)
    @staticmethod
    def _ngp_2d(dr: np.ndarray) -> float:
        if dr.size == 0: return np.nan
        r2 = dr*dr
        m2 = np.mean(r2)
        if m2 <= 0: return np.nan
        m4 = np.mean(r2*r2)
        return float(m4 / (2*m2*m2) - 1.0)

    def ngp_vs_lag(self):
        rows = []
        for lag in self.lags:
            dx, dy, dr, n = self._collect_displacements(lag)
            tau_s = lag * self.cfg.frame_interval
            rows.append({'lag': lag, 'tau_s': tau_s,
                         'NGP_1D': self._ngp_1d(dx),
                         'NGP_2D': self._ngp_2d(dr),
                         'n_steps': n})
        return pd.DataFrame(rows)

    # -------- van Hove --------
    def van_hove(self, lag: int, bins: Optional[int] = None, clip_sigma: Optional[float] = None):
        dx, dy, dr, n = self._collect_displacements(lag)
        if n == 0:
            return {"dx_centers": np.array([]), "dx_density": np.array([]),
                    "r_centers": np.array([]), "r_density": np.array([])}
        if bins is None:
            bins = self.cfg.n_hist_bins
        if clip_sigma is not None:
            sx = np.std(dx) + 1e-12
            sr = np.std(dr) + 1e-12
            dx = dx[np.abs(dx) <= clip_sigma * sx]
            dr = dr[dr <= clip_sigma * sr]
        dx_hist, dx_edges = np.histogram(dx, bins=bins, density=True)
        r_hist, r_edges = np.histogram(dr, bins=bins, density=True)
        return {
            "dx_centers": 0.5*(dx_edges[:-1]+dx_edges[1:]),
            "dx_density": dx_hist,
            "r_centers": 0.5*(r_edges[:-1]+r_edges[1:]),
            "r_density": r_hist
        }

    # -------- VACF --------
    def vacf(self, max_lag: Optional[int] = None):
        if max_lag is None:
            max_lag = min(self.cfg.max_lag, 30)
        rows = []
        dt = self.cfg.frame_interval
        for lag in range(0, max_lag+1):
            num_sum = den_sum = 0.0
            n_used = 0
            for _, g in self.df.groupby('track_id'):
                x = g['x_um'].values; y = g['y_um'].values
                if len(x) < lag + 2: continue
                vx = np.diff(x)/dt; vy = np.diff(y)/dt
                if lag == 0:
                    num = np.mean(vx*vx + vy*vy); den = num
                else:
                    num = np.mean(vx[:-lag]*vx[lag:] + vy[:-lag]*vy[lag:])
                    den = np.mean(vx*vx + vy*vy)
                if np.isfinite(num) and np.isfinite(den) and den > 0:
                    num_sum += num; den_sum += den; n_used += 1
            vacf = (num_sum/den_sum) if den_sum > 0 else np.nan
            rows.append({'lag': lag, 'tau_s': lag*dt, 'VACF': float(vacf), 'n_tracks_used': n_used})
        return pd.DataFrame(rows)

    # -------- Turning angles --------
    def turning_angles(self):
        angles = []
        for _, g in self.df.groupby('track_id'):
            x = g['x_um'].values; y = g['y_um'].values
            if len(x) < 3: continue
            dx = np.diff(x); dy = np.diff(y)
            v1x = dx[:-1]; v1y = dy[:-1]
            v2x = dx[1:];  v2y = dy[1:]
            num = v1x*v2x + v1y*v2y
            den = np.sqrt(v1x*v1x + v1y*v1y)*np.sqrt(v2x*v2x + v2y*v2y) + 1e-30
            cosang = np.clip(num/den, -1, 1)
            theta = np.arccos(cosang)  # [0, π]
            sgn = np.sign(v1x*v2y - v1y*v2x)
            angles.append(theta * sgn)  # signed (-π, π)
        return pd.DataFrame({'angle_rad': np.concatenate(angles)}) if angles else pd.DataFrame({'angle_rad': []})

    # -------- Hurst (from TAMSD slope) --------
    @staticmethod
    def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if np.count_nonzero(mask) < 2:
            return np.nan
        b, a = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
        return float(b)

    def hurst_from_tamsd(self, tamsd_df: pd.DataFrame):
        rows = []
        for tid, g in tamsd_df.groupby('track_id'):
            tau = g['tau_s'].values; msd = g['tamsd'].values
            alpha = self._robust_slope(tau, msd)
            rows.append({'track_id': tid, 'alpha': alpha, 'H': alpha/2 if np.isfinite(alpha) else np.nan})
        return pd.DataFrame(rows)

    # -------- Orchestrator --------
    def compute_all(self) -> Dict[str, object]:
        if self.df.empty or self.lags.size == 0:
            return {'success': False, 'error': 'No tracks after filtering or no valid lags.'}
        tamsd_df, eamsd_df = self.tamsd_eamsd()
        erg_df = self.ergodicity_measures(tamsd_df, eamsd_df)
        ngp_df = self.ngp_vs_lag()
        vacf_df = self.vacf()
        ang_df = self.turning_angles()
        hurst_df = self.hurst_from_tamsd(tamsd_df)
        summary = {
            'n_tracks': int(self.df['track_id'].nunique()),
            'median_track_len': int(self.df.groupby('track_id').size().median()),
            'lags': self.lags.tolist(),
            'H_median': float(np.nanmedian(hurst_df['H'])) if not hurst_df.empty else np.nan
        }
        return {
            'success': True,
            'config': self.cfg,
            'summary': summary,
            'tamsd': tamsd_df,
            'eamsd': eamsd_df,
            'ergodicity': erg_df,
            'ngp': ngp_df,
            'vacf': vacf_df,
            'turning_angles': ang_df,
            'hurst': hurst_df
        }
