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
    def _robust_slope(x: np.ndarray, y: np.ndarray, min_points: int = 5) -> float:
        """Calculate slope from log-log linear regression with minimum point requirement."""
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if np.count_nonzero(mask) < min_points:
            return np.nan
        b, a = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
        return float(b)

    def hurst_from_tamsd(self, tamsd_df: pd.DataFrame, min_points: int = 5):
        """
        Calculate Hurst exponent from TAMSD power-law scaling.
        
        Parameters
        ----------
        tamsd_df : pd.DataFrame
            Time-averaged MSD data with track_id, tau_s, tamsd columns
        min_points : int
            Minimum number of valid lag points required for robust slope calculation.
            Default is 5. Tracks with fewer points will return NaN.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with track_id, alpha (scaling exponent), and H (Hurst exponent)
        """
        rows = []
        for tid, g in tamsd_df.groupby('track_id'):
            tau = g['tau_s'].values
            msd = g['tamsd'].values
            alpha = self._robust_slope(tau, msd, min_points=min_points)
            H_value = alpha/2 if np.isfinite(alpha) else np.nan
            rows.append({
                'track_id': tid, 
                'alpha': alpha, 
                'H': H_value,
                'n_lag_points': int(np.count_nonzero(np.isfinite(tau) & np.isfinite(msd)))
            })
        return pd.DataFrame(rows)

    def fbm_analysis(self, min_track_length: int = 30):
        """
        Perform Fractional Brownian Motion (fBm) analysis on all tracks.
        
        Parameters
        ----------
        min_track_length : int
            Minimum number of frames required for robust FBM estimation.
            Default is 30. Shorter tracks will be skipped and marked with error.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with track_id, H_fbm (Hurst), D_fbm (diffusion), 
            fbm_fit_error (error message if any), and track_length
        """
        rows = []
        for tid, g in self.df.groupby('track_id'):
            track_len = len(g)
            if track_len < min_track_length:
                row = {
                    'track_id': tid, 
                    'H_fbm': np.nan, 
                    'D_fbm': np.nan, 
                    'fbm_fit_error': f'Track too short ({track_len} < {min_track_length} frames)',
                    'track_length': track_len
                }
            else:
                result = fit_fbm_model(g, self.cfg.pixel_size, self.cfg.frame_interval)
                row = {
                    'track_id': tid, 
                    'H_fbm': result.get('H'), 
                    'D_fbm': result.get('D'), 
                    'fbm_fit_error': result.get('error', None),
                    'track_length': track_len
                }
            rows.append(row)
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
        hurst_df = self.hurst_from_tamsd(tamsd_df, min_points=5)
        fbm_df = self.fbm_analysis(min_track_length=30)
        
        # Calculate filtering statistics
        n_tracks = int(self.df['track_id'].nunique())
        n_valid_hurst = int(hurst_df['H'].notna().sum())
        n_valid_fbm = int(fbm_df['H_fbm'].notna().sum())
        
        summary = {
            'n_tracks': n_tracks,
            'median_track_len': int(self.df.groupby('track_id').size().median()),
            'lags': self.lags.tolist(),
            'H_median': float(np.nanmedian(hurst_df['H'])) if not hurst_df.empty else np.nan,
            'H_fbm_median': float(np.nanmedian(fbm_df['H_fbm'])) if not fbm_df.empty else np.nan,
            'n_valid_hurst': n_valid_hurst,
            'n_excluded_hurst': n_tracks - n_valid_hurst,
            'n_valid_fbm': n_valid_fbm,
            'n_excluded_fbm': n_tracks - n_valid_fbm
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
            'hurst': hurst_df,
            'fbm': fbm_df
        }

def fit_fbm_model(track_df: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0):
    """
    Fit a Fractional Brownian Motion (fBm) model to a single trajectory.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame for a single track with 'x', 'y', and 'frame' columns.
    pixel_size : float, optional
        Pixel size in micrometers, by default 1.0.
    frame_interval : float, optional
        Time between frames in seconds, by default 1.0.

    Returns
    -------
    dict
        A dictionary with the estimated Hurst parameter 'H' and diffusion coefficient 'D'.
    """
    try:
        import fbm
        from scipy.optimize import curve_fit
    except ImportError:
        return {'H': np.nan, 'D': np.nan, 'error': 'fbm or scipy not installed.'}

    if track_df.empty or len(track_df) < 10: # Need more points for robust estimation
        return {'H': np.nan, 'D': np.nan, 'error': 'Track too short for fitting.'}

    # Prepare data
    track = track_df.sort_values('frame').copy()
    track['x_um'] = track['x'] * pixel_size
    track['y_um'] = track['y'] * pixel_size

    results = {}

    try:
        # Estimate Hurst exponent from the trajectory increments
        dx = np.diff(track['x_um'].values)
        dy = np.diff(track['y_um'].values)

        # Check if we have valid displacement data
        if len(dx) < 5 or len(dy) < 5:
            return {'H': np.nan, 'D': np.nan, 'error': 'Insufficient displacement data.'}
        
        # Remove any NaN or infinite values
        valid_dx = dx[np.isfinite(dx)]
        valid_dy = dy[np.isfinite(dy)]
        
        if len(valid_dx) < 5 or len(valid_dy) < 5:
            return {'H': np.nan, 'D': np.nan, 'error': 'Too many invalid displacements.'}

        # fbm.hurst expects a 1D array of increments
        # The fbm.hurst function can fail, so we need to catch that
        try:
            h_x = fbm.hurst(valid_dx)
            h_y = fbm.hurst(valid_dy)
            
            # Validate Hurst values
            if not np.isfinite(h_x) or not np.isfinite(h_y):
                # Fall back to MSD-based estimation
                H = np.nan
            else:
                H = (h_x + h_y) / 2.0
        except Exception as hurst_error:
            # If fbm.hurst fails, try alternative method: fit MSD power law
            H = np.nan

        # Now, calculate MSD and fit for the diffusion coefficient D
        # MSD(t) = 4*D*t^H for 2D fBm
        msd_values = []
        lag_times = []
        max_lag = min(len(track) - 1, 20)
        for lag in range(1, max_lag + 1):
            if len(track) > lag:
                disp_x = track['x_um'].values[lag:] - track['x_um'].values[:-lag]
                disp_y = track['y_um'].values[lag:] - track['y_um'].values[:-lag]
                sq_disp = disp_x**2 + disp_y**2
                valid_sq_disp = sq_disp[np.isfinite(sq_disp)]
                if len(valid_sq_disp) > 0:
                    msd_values.append(np.mean(valid_sq_disp))
                    lag_times.append(lag * frame_interval)

        if len(msd_values) < 3:
            return {'H': np.nan, 'D': np.nan, 'error': 'Insufficient MSD data points.'}

        msd_values = np.array(msd_values)
        lag_times = np.array(lag_times)

        # If H wasn't calculated above, estimate it from MSD slope in log-log space
        if not np.isfinite(H):
            try:
                # log(MSD) = log(4*D) + H*log(t)
                log_msd = np.log(msd_values[msd_values > 0])
                log_t = np.log(lag_times[msd_values > 0])
                if len(log_msd) >= 3:
                    coeffs = np.polyfit(log_t, log_msd, 1)
                    H = coeffs[0]  # Slope is the Hurst exponent
                else:
                    H = 0.5  # Default to Brownian motion
            except Exception:
                H = 0.5  # Default to Brownian motion

        # Ensure H is in valid range [0, 1]
        H = np.clip(H, 0.0, 1.0)

        def msd_model(t, D):
            return 4 * D * (t ** H)

        try:
            # Use bounds to ensure positive D
            popt, _ = curve_fit(msd_model, lag_times, msd_values, 
                               bounds=(0, np.inf), maxfev=5000)
            D = popt[0]
        except Exception:
            # Fallback: estimate D from first MSD point assuming MSD = 4*D*t^H
            if len(msd_values) > 0 and lag_times[0] > 0:
                D = msd_values[0] / (4 * lag_times[0] ** H)
            else:
                D = np.nan

        results['H'] = float(H) if np.isfinite(H) else np.nan
        results['D'] = float(D) if np.isfinite(D) else np.nan
        results['error'] = None

    except Exception as e:
        results['H'] = np.nan
        results['D'] = np.nan
        results['error'] = str(e)

    return results


def analyze_turning_angles(
    tracks_df: pd.DataFrame,
    min_track_length: int = 3
) -> pd.DataFrame:
    """
    Calculates the turning angle for each step in every track.

    The turning angle is the angle between two consecutive displacement vectors
    in a trajectory. It ranges from -pi to +pi.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with track data including 'track_id', 'frame', 'x', 'y'.
    min_track_length : int, optional
        Minimum number of points in a track to be included, by default 3.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single column 'angle_rad' containing all calculated
        turning angles in radians.
    """
    if not {'track_id', 'frame', 'x', 'y'}.issubset(tracks_df.columns):
        raise ValueError("Input DataFrame must contain 'track_id', 'frame', 'x', and 'y' columns.")

    df = tracks_df[['track_id', 'frame', 'x', 'y']].dropna().copy()
    df = df.sort_values(['track_id', 'frame'])

    # Filter for tracks with at least min_track_length points
    df = df.groupby('track_id').filter(lambda g: len(g) >= min_track_length)

    if df.empty:
        return pd.DataFrame({'angle_rad': []})

    angles = []
    for _, g in df.groupby('track_id'):
        x = g['x'].values
        y = g['y'].values

        if len(x) < 3: continue

        # Calculate displacement vectors
        dx = np.diff(x)
        dy = np.diff(y)

        # Create vectors for angle calculation
        v1x, v1y = dx[:-1], dy[:-1]
        v2x, v2y = dx[1:], dy[1:]

        # Calculate dot product and magnitudes
        dot_product = v1x * v2x + v1y * v2y
        mag_v1 = np.sqrt(v1x**2 + v1y**2)
        mag_v2 = np.sqrt(v2x**2 + v2y**2)

        # Avoid division by zero
        denominator = mag_v1 * mag_v2

        # Filter out zero-magnitude vectors
        valid_indices = denominator > 1e-9
        if not np.any(valid_indices):
            continue

        dot_product = dot_product[valid_indices]
        denominator = denominator[valid_indices]
        v1x = v1x[valid_indices]
        v1y = v1y[valid_indices]
        v2x = v2x[valid_indices]
        v2y = v2y[valid_indices]

        # Calculate cosine of the angle
        cos_angle = np.clip(dot_product / denominator, -1.0, 1.0)

        # Get angle in [0, pi]
        angle = np.arccos(cos_angle)

        # Determine the sign of the angle using the 2D cross-product
        cross_product_sign = np.sign(v1x * v2y - v1y * v2x)

        # Apply the sign to get the angle in [-pi, pi]
        signed_angle = angle * cross_product_sign

        angles.append(signed_angle)

    if not angles:
        return pd.DataFrame({'angle_rad': []})

    return pd.DataFrame({'angle_rad': np.concatenate(angles)})


def calculate_ergodicity_breaking(
    tamsd_df: pd.DataFrame,
    eamsd_df: pd.DataFrame,
    n_bootstrap: int = 0
) -> pd.DataFrame:
    """
    Calculates the Ergodicity Breaking (EB) parameter over lag times.

    This function compares the time-averaged MSD (TAMSD) for individual
    trajectories with the ensemble-averaged MSD (EAMSD).

    Parameters
    ----------
    tamsd_df : pd.DataFrame
        A DataFrame containing time-averaged MSDs for each track.
        Must contain 'lag', 'tau_s', and 'tamsd' columns.
    eamsd_df : pd.DataFrame
        A DataFrame containing the ensemble-averaged MSD.
        Must contain 'lag', 'tau_s', and 'eamsd' columns.
    n_bootstrap : int, optional
        Number of bootstrap samples for confidence interval estimation.
        If 0, no confidence intervals are calculated. By default 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with ergodicity breaking metrics for each lag time,
        including 'EB_ratio' and 'EB_parameter'.
    """
    required_tamsd = {'lag', 'tau_s', 'tamsd'}
    required_eamsd = {'lag', 'tau_s', 'eamsd'}

    if not required_tamsd.issubset(tamsd_df.columns):
        raise ValueError(f"tamsd_df is missing required columns: {required_tamsd - set(tamsd_df.columns)}")
    if not required_eamsd.issubset(eamsd_df.columns):
        raise ValueError(f"eamsd_df is missing required columns: {required_eamsd - set(eamsd_df.columns)}")

    eamsd_lookup = eamsd_df.set_index('lag')

    output_rows = []

    common_lags = sorted(list(set(tamsd_df['lag'].unique()) & set(eamsd_lookup.index)))

    rng = np.random.default_rng()

    for lag in common_lags:
        tau_s = eamsd_lookup.loc[lag, 'tau_s']
        eamsd = eamsd_lookup.loc[lag, 'eamsd']

        tamsd_values = tamsd_df[tamsd_df['lag'] == lag]['tamsd'].values

        if tamsd_values.size == 0 or not np.isfinite(eamsd):
            continue

        mean_tamsd = np.mean(tamsd_values)

        eb_ratio = mean_tamsd / (eamsd + 1e-30)

        normalized_tamsd = tamsd_values / (mean_tamsd + 1e-30)
        eb_parameter = np.mean((normalized_tamsd - 1.0)**2)

        row = {
            'lag': lag,
            'tau_s': tau_s,
            'EB_ratio': eb_ratio,
            'EB_parameter': eb_parameter
        }

        if n_bootstrap > 0 and tamsd_values.size > 5:
            eb_r_samples, eb_p_samples = [], []
            for _ in range(n_bootstrap):
                bootstrap_sample = rng.choice(tamsd_values, size=tamsd_values.size, replace=True)

                m_bs = np.mean(bootstrap_sample)
                eb_r_samples.append(m_bs / (eamsd + 1e-30))

                norm_bs = bootstrap_sample / (m_bs + 1e-30)
                eb_p_samples.append(np.mean((norm_bs - 1.0)**2))

            lo_r, hi_r = np.percentile(eb_r_samples, [2.5, 97.5])
            lo_p, hi_p = np.percentile(eb_p_samples, [2.5, 97.5])

            row.update({
                'EB_ratio_CI_low': lo_r, 'EB_ratio_CI_high': hi_r,
                'EB_param_CI_low': lo_p, 'EB_param_CI_high': hi_p
            })

        output_rows.append(row)

    return pd.DataFrame(output_rows)
