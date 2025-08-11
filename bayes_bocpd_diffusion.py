from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class BOCPDConfig:
    pixel_size: float = 1.0
    frame_interval: float = 1.0
    lag_frames: int = 1
    hazard_tau: float = 50.0     # expected run length
    alpha0: float = 1.0          # Gamma prior shape
    beta0: float = 1e-3          # Gamma prior rate
    rmax: int = 1000             # max run-length tracked
    min_segment_len: int = 5
    credible_q: Tuple[float, float] = (2.5, 97.5)

class BOCPDDiffusion:
    """
    Bayesian Online Changepoint Detection on squared radial steps r^2 (μm^2).
    Model: r^2 ~ Exp(λ); λ ~ Gamma(α, β).
    Diffusion estimate per segment: D = 1/(4 * Δt * lag_frames * λ_mean).
    """
    def __init__(self, tracks_df: pd.DataFrame, cfg: BOCPDConfig):
        req = {'track_id','frame','x','y'}
        miss = req - set(tracks_df.columns)
        if miss:
            raise ValueError(f"tracks_df missing columns: {miss}")
        self.cfg = cfg
        df = tracks_df.loc[:, ['track_id','frame','x','y']].dropna().copy()
        df['frame'] = df['frame'].astype(int)
        df.sort_values(['track_id','frame'], inplace=True)
        df['x_um'] = df['x'] * cfg.pixel_size
        df['y_um'] = df['y'] * cfg.pixel_size
        self.df = df

    def _track_steps(self, g: pd.DataFrame):
        L = self.cfg.lag_frames
        x = g['x_um'].values; y = g['y_um'].values; fr = g['frame'].values
        if len(x) <= L:
            return np.array([]), np.array([]), np.array([]), np.array([])
        dx = x[L:] - x[:-L]; dy = y[L:] - y[:-L]
        r2 = dx*dx + dy*dy
        frames_mid = (fr[L:] + fr[:-L]) // 2
        return frames_mid, dx, dy, r2

    def _bocpd_exponential(self, r2: np.ndarray) -> Dict[str, np.ndarray]:
        T = len(r2)
        if T == 0:
            return {'R': np.zeros((0,1)), 'cp': np.zeros(0, dtype=int),
                    'alpha': np.zeros((0,1)), 'beta': np.zeros((0,1))}
        rmax = min(self.cfg.rmax, T)
        H = 1.0 / max(1.0, self.cfg.hazard_tau)
        R_prev = np.zeros(rmax+1); R_prev[0] = 1.0
        a_prev = np.full(rmax+1, self.cfg.alpha0)
        b_prev = np.full(rmax+1, self.cfg.beta0)
        R = np.zeros((T, rmax+1))
        alpha = np.zeros((T, rmax+1))
        beta = np.zeros((T, rmax+1))
        back = np.zeros((T, rmax+1), dtype=int)
        for t in range(T):
            x = r2[t]
            log_pred = np.log(a_prev) + a_prev*np.log(b_prev) - (a_prev+1)*np.log(b_prev + x)
            log_growth = np.log(1-H) + (np.log(R_prev + 1e-300) + log_pred)
            m = np.max(np.log(R_prev + 1e-300) + log_pred)
            cp_mass = np.exp(np.log(H) + m + np.log(np.sum(np.exp((np.log(R_prev + 1e-300) + log_pred) - m))))
            R_t = np.zeros(rmax+1)
            R_t[0] = cp_mass
            R_t[1:] = np.exp(log_growth[:-1])
            tot = R_t.sum()
            if tot <= 0:
                R_t = np.zeros_like(R_t); R_t[0] = 1.0
            else:
                R_t /= tot
            R[t] = R_t
            prev_scores = np.full(rmax+1, -np.inf)
            prev_scores[1:] = log_growth[:-1]
            prev_scores[0] = np.log(H) + m
            back[t] = np.argmax(prev_scores)
            # posterior update
            a_new = np.zeros_like(a_prev); b_new = np.zeros_like(b_prev)
            a_new[0] = self.cfg.alpha0; b_new[0] = self.cfg.beta0
            a_new[1:] = a_prev[:-1] + 1
            b_new[1:] = b_prev[:-1] + x
            alpha[t] = a_new; beta[t] = b_new
            a_prev, b_prev, R_prev = a_new, b_new, R_t
        cp = np.zeros(T, dtype=int)
        cp[-1] = int(np.argmax(R[-1]))
        for t in range(T-2, -1, -1):
            cp[t] = back[t+1, cp[t+1]]
        return {'R': R, 'cp': cp, 'alpha': alpha, 'beta': beta}

    def _segments(self, frames_mid, dx, dy, r2, cp, alpha, beta):
        if len(cp) == 0:
            return pd.DataFrame(columns=['start_frame','end_frame','n_steps','D_mean','D_CI_low','D_CI_high',
                                         'lambda_mean','vx_um_s','vy_um_s','speed_um_s'])
        seg_bounds = []
        s = 0
        for t in range(1, len(cp)):
            if cp[t] <= cp[t-1]:
                seg_bounds.append((s, t-1)); s = t
        seg_bounds.append((s, len(cp)-1))
        rows = []
        dtL = self.cfg.frame_interval * self.cfg.lag_frames
        try:
            from scipy.stats import gamma as _Gamma
            sample_gamma = True
        except Exception:
            sample_gamma = False
        for s, e in seg_bounds:
            n = e - s + 1
            if n < self.cfg.min_segment_len:
                continue
            a = float(alpha[e, cp[e]])
            b = float(beta[e, cp[e]])
            lam_mean = a / b
            if sample_gamma and a > 0 and b > 0:
                draws = _Gamma.rvs(a, scale=1/b, size=4000, random_state=0)
                qlo, qhi = np.percentile(draws, list(self.cfg.credible_q))
            else:
                qlo, qhi = np.nan, np.nan
            D_mean = 1.0 / (4.0 * dtL * lam_mean + 1e-30)
            D_lo = 1.0 / (4.0 * dtL * qhi + 1e-30) if np.isfinite(qhi) and qhi>0 else np.nan
            D_hi = 1.0 / (4.0 * dtL * qlo + 1e-30) if np.isfinite(qlo) and qlo>0 else np.nan
            vx = float(np.mean(dx[s:e+1])) / dtL
            vy = float(np.mean(dy[s:e+1])) / dtL
            rows.append({
                'start_frame': int(frames_mid[s]),
                'end_frame': int(frames_mid[e]),
                'n_steps': int(n),
                'D_mean': float(D_mean),
                'D_CI_low': float(D_lo),
                'D_CI_high': float(D_hi),
                'lambda_mean': float(lam_mean),
                'vx_um_s': vx,
                'vy_um_s': vy,
                'speed_um_s': float(np.hypot(vx, vy))
            })
        return pd.DataFrame(rows)

    def segment_all(self) -> Dict[str, object]:
        out = {'success': True, 'tracks': {}, 'config': self.cfg.__dict__}
        for tid, g in self.df.groupby('track_id'):
            frames_mid, dx, dy, r2 = self._track_steps(g)
            R = self._bocpd_exponential(r2)
            segs = self._segments(frames_mid, dx, dy, r2,
                                  R['cp'], R['alpha'], R['beta'])
            out['tracks'][tid] = {'segments': segs, 'n_segments': int(len(segs))}
        return out
