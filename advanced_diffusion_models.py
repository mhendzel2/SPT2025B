"""
Advanced Diffusion Models for SPT2025B

Implements:
1. Continuous Time Random Walk (CTRW) analysis
2. Fractional Brownian Motion (FBM) explicit fitting
3. Aging and non-ergodicity analysis

References:
- Metzler & Klafter (2000): "The Random Walk's Guide to Anomalous Diffusion"
- Metzler et al. (2014): "Anomalous diffusion models and their properties"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, kstest, expon, powerlaw
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List, Optional
import warnings


class CTRWAnalyzer:
    """
    Continuous Time Random Walk (CTRW) Analysis.
    
    CTRW is characterized by:
    - Distribution of waiting times ψ(t)
    - Distribution of jump lengths λ(r)
    - Possible coupling between waiting time and jump length
    
    In biological systems, waiting times often reflect binding/unbinding events.
    """
    
    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 0.1, frame_interval: float = 0.1):
        """
        Initialize CTRW analyzer.
        
        Parameters:
        -----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Time between frames in seconds
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        
        # Convert to physical units
        self.tracks_df['x_um'] = self.tracks_df['x'] * pixel_size
        self.tracks_df['y_um'] = self.tracks_df['y'] * pixel_size
        self.tracks_df['time_s'] = self.tracks_df['frame'] * frame_interval
    
    def analyze_waiting_time_distribution(
        self,
        min_pause_threshold: float = 0.01,
        fit_distribution: str = 'auto'
    ) -> Dict:
        """
        Analyze distribution of waiting times between movement events.
        
        Heavy-tailed waiting time distributions indicate CTRW behavior.
        
        Parameters:
        -----------
        min_pause_threshold : float
            Minimum displacement (μm) to consider as movement
        fit_distribution : str
            'exponential', 'powerlaw', or 'auto'
        
        Returns:
        --------
        dict with keys:
            - 'waiting_times': array of waiting times (s)
            - 'distribution_type': str ('exponential', 'powerlaw', or 'mixed')
            - 'mean_waiting_time': float
            - 'alpha_exponent': float (power-law exponent if applicable)
            - 'is_heavy_tailed': bool
            - 'ks_statistic': float
        """
        waiting_times = []
        
        # Analyze each track
        for track_id in self.tracks_df['track_id'].unique():
            track = self.tracks_df[self.tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track) < 3:
                continue
            
            # Calculate displacements
            dx = np.diff(track['x_um'].values)
            dy = np.diff(track['y_um'].values)
            displacements = np.sqrt(dx**2 + dy**2)
            
            # Find pauses (displacements below threshold)
            is_paused = displacements < min_pause_threshold
            
            # Calculate waiting times
            current_wait = 0
            for i, paused in enumerate(is_paused):
                if paused:
                    current_wait += self.frame_interval
                else:
                    if current_wait > 0:
                        waiting_times.append(current_wait)
                    current_wait = 0
            
            # Add final waiting time if ended in pause
            if current_wait > 0:
                waiting_times.append(current_wait)
        
        waiting_times = np.array(waiting_times)
        
        if len(waiting_times) == 0:
            return {
                'waiting_times': waiting_times,
                'distribution_type': 'none',
                'mean_waiting_time': 0.0,
                'alpha_exponent': np.nan,
                'is_heavy_tailed': False,
                'ks_statistic': np.nan
            }
        
        mean_wait = np.mean(waiting_times)
        
        # Test for exponential distribution
        ks_exp_stat, ks_exp_p = kstest(waiting_times, 'expon', args=(0, mean_wait))
        
        # Test for power-law distribution
        alpha_exponent = np.nan
        ks_pl_stat = 1.0
        
        if fit_distribution in ['powerlaw', 'auto']:
            # Fit power law: P(t) ~ t^(-alpha)
            # Use only t > min_wait for better fit
            min_wait = np.percentile(waiting_times, 25)
            tail_times = waiting_times[waiting_times > min_wait]
            
            if len(tail_times) > 10:
                try:
                    # MLE for power law
                    alpha_exponent = 1 + len(tail_times) / np.sum(np.log(tail_times / min_wait))
                    
                    # KS test for power law
                    pl_cdf = lambda t: 1 - (t / min_wait)**(-alpha_exponent + 1)
                    ks_pl_stat = np.max(np.abs(
                        np.arange(1, len(tail_times) + 1) / len(tail_times) -
                        pl_cdf(np.sort(tail_times))
                    ))
                except Exception as e:
                    warnings.warn(f"Power-law fitting failed: {e}")
        
        # Determine distribution type
        if fit_distribution == 'auto':
            if ks_exp_p > 0.05:
                distribution_type = 'exponential'
                ks_statistic = ks_exp_stat
            elif not np.isnan(alpha_exponent) and ks_pl_stat < ks_exp_stat:
                distribution_type = 'powerlaw'
                ks_statistic = ks_pl_stat
            else:
                distribution_type = 'mixed'
                ks_statistic = min(ks_exp_stat, ks_pl_stat)
        elif fit_distribution == 'exponential':
            distribution_type = 'exponential'
            ks_statistic = ks_exp_stat
        else:
            distribution_type = 'powerlaw'
            ks_statistic = ks_pl_stat
        
        # Heavy tail if power-law with alpha < 3
        is_heavy_tailed = (distribution_type == 'powerlaw' and alpha_exponent < 3)
        
        return {
            'waiting_times': waiting_times,
            'distribution_type': distribution_type,
            'mean_waiting_time': mean_wait,
            'alpha_exponent': alpha_exponent,
            'is_heavy_tailed': is_heavy_tailed,
            'ks_statistic': ks_statistic,
            'ks_exponential_p': ks_exp_p
        }
    
    def analyze_jump_length_distribution(
        self,
        remove_zeros: bool = True,
        fit_distribution: str = 'auto'
    ) -> Dict:
        """
        Analyze distribution of jump lengths (step sizes).
        
        Levy flights show power-law jump length distributions.
        
        Parameters:
        -----------
        remove_zeros : bool
            Remove pauses (zero displacements)
        fit_distribution : str
            'gaussian', 'exponential', 'powerlaw', 'levy', or 'auto'
        
        Returns:
        --------
        dict with keys:
            - 'jump_lengths': array (μm)
            - 'distribution_type': str
            - 'mean_jump_length': float
            - 'levy_exponent': float (if power-law)
            - 'is_levy_flight': bool
        """
        jump_lengths = []
        
        # Calculate jump lengths for all tracks
        for track_id in self.tracks_df['track_id'].unique():
            track = self.tracks_df[self.tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track) < 2:
                continue
            
            dx = np.diff(track['x_um'].values)
            dy = np.diff(track['y_um'].values)
            jumps = np.sqrt(dx**2 + dy**2)
            
            jump_lengths.extend(jumps)
        
        jump_lengths = np.array(jump_lengths)
        
        if remove_zeros:
            jump_lengths = jump_lengths[jump_lengths > 0]
        
        if len(jump_lengths) == 0:
            return {
                'jump_lengths': jump_lengths,
                'distribution_type': 'none',
                'mean_jump_length': 0.0,
                'levy_exponent': np.nan,
                'is_levy_flight': False
            }
        
        mean_jump = np.mean(jump_lengths)
        
        # Test for Gaussian (Brownian motion)
        # For Brownian: P(r) ~ exp(-r²)
        theoretical_mean_2d = np.sqrt(np.pi/2) * np.std(jump_lengths)
        is_gaussian = np.abs(mean_jump - theoretical_mean_2d) / theoretical_mean_2d < 0.2
        
        # Test for power law (Levy flight): P(r) ~ r^(-(1+beta))
        levy_exponent = np.nan
        
        if fit_distribution in ['powerlaw', 'levy', 'auto']:
            # Fit power law to tail
            tail_threshold = np.percentile(jump_lengths, 50)
            tail_jumps = jump_lengths[jump_lengths > tail_threshold]
            
            if len(tail_jumps) > 10:
                try:
                    log_r = np.log(tail_jumps)
                    log_P = -np.log(np.arange(len(tail_jumps), 0, -1))
                    
                    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_P)
                    levy_exponent = -slope - 1  # Convert to beta
                except Exception as e:
                    warnings.warn(f"Levy exponent fitting failed: {e}")
        
        # Determine distribution type
        if fit_distribution == 'auto':
            if is_gaussian:
                distribution_type = 'gaussian'
            elif not np.isnan(levy_exponent) and 0 < levy_exponent < 2:
                distribution_type = 'levy'
            else:
                distribution_type = 'exponential'
        else:
            distribution_type = fit_distribution
        
        # Levy flight if 0 < beta < 2
        is_levy_flight = (distribution_type == 'levy' and 
                         not np.isnan(levy_exponent) and 
                         0 < levy_exponent < 2)
        
        return {
            'jump_lengths': jump_lengths,
            'distribution_type': distribution_type,
            'mean_jump_length': mean_jump,
            'levy_exponent': levy_exponent,
            'is_levy_flight': is_levy_flight
        }
    
    def test_ergodicity(
        self,
        n_segments: int = 4,
        lag_times: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Test for ergodicity by comparing time-averaged and ensemble-averaged MSD.
        
        Non-ergodic systems show aging: MSD depends on measurement start time.
        
        Parameters:
        -----------
        n_segments : int
            Number of temporal segments to analyze
        lag_times : array, optional
            Lag times to compute MSD (default: auto-generate)
        
        Returns:
        --------
        dict with keys:
            - 'is_ergodic': bool
            - 'ergodicity_breaking_parameter': float (0=ergodic, 1=fully non-ergodic)
            - 'aging_coefficient': float
            - 'time_avg_msd': array
            - 'ensemble_avg_msd': array
        """
        from analysis import calculate_msd
        
        # Calculate ensemble MSD for full dataset
        ensemble_msd = calculate_msd(
            self.tracks_df,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval
        )
        
        if isinstance(ensemble_msd, dict):
            ensemble_msd_df = ensemble_msd.get('ensemble_msd')
        else:
            ensemble_msd_df = ensemble_msd
        
        if ensemble_msd_df is None or ensemble_msd_df.empty:
            return {
                'is_ergodic': True,
                'ergodicity_breaking_parameter': 0.0,
                'aging_coefficient': 0.0,
                'time_avg_msd': np.array([]),
                'ensemble_avg_msd': np.array([])
            }
        
        # Divide data into temporal segments
        total_frames = self.tracks_df['frame'].max()
        segment_size = total_frames // n_segments
        
        segment_msds = []
        
        for i in range(n_segments):
            start_frame = i * segment_size
            end_frame = (i + 1) * segment_size if i < n_segments - 1 else total_frames
            
            segment_data = self.tracks_df[
                (self.tracks_df['frame'] >= start_frame) &
                (self.tracks_df['frame'] < end_frame)
            ].copy()
            
            # Renumber frames for this segment
            segment_data['frame'] = segment_data['frame'] - start_frame
            
            if len(segment_data) < 10:
                continue
            
            seg_msd = calculate_msd(
                segment_data,
                pixel_size=self.pixel_size,
                frame_interval=self.frame_interval
            )
            
            if isinstance(seg_msd, dict):
                seg_msd_df = seg_msd.get('ensemble_msd')
            else:
                seg_msd_df = seg_msd
            
            if seg_msd_df is not None and not seg_msd_df.empty:
                segment_msds.append(seg_msd_df['msd'].values)
        
        if len(segment_msds) < 2:
            return {
                'is_ergodic': True,
                'ergodicity_breaking_parameter': 0.0,
                'aging_coefficient': 0.0,
                'time_avg_msd': np.array([]),
                'ensemble_avg_msd': ensemble_msd_df['msd'].values
            }
        
        # Calculate variance across segments (ergodicity breaking)
        min_len = min(len(msd) for msd in segment_msds)
        segment_msds_trimmed = np.array([msd[:min_len] for msd in segment_msds])
        
        time_avg_msd = np.mean(segment_msds_trimmed, axis=0)
        msd_variance = np.var(segment_msds_trimmed, axis=0)
        
        # Ergodicity breaking parameter (EB)
        # EB = 0: ergodic, EB = 1: fully non-ergodic
        ensemble_msd_trimmed = ensemble_msd_df['msd'].values[:min_len]
        
        if np.sum(ensemble_msd_trimmed**2) > 0:
            eb_parameter = np.mean(msd_variance / (ensemble_msd_trimmed**2 + 1e-10))
            eb_parameter = np.clip(eb_parameter, 0, 1)
        else:
            eb_parameter = 0.0
        
        # Aging coefficient: change in MSD over time
        if len(segment_msds_trimmed) > 1:
            first_segment = segment_msds_trimmed[0]
            last_segment = segment_msds_trimmed[-1]
            
            # Calculate relative change
            aging_coef = np.mean((last_segment - first_segment) / (first_segment + 1e-10))
        else:
            aging_coef = 0.0
        
        is_ergodic = eb_parameter < 0.2  # Threshold for ergodicity
        
        return {
            'is_ergodic': is_ergodic,
            'ergodicity_breaking_parameter': eb_parameter,
            'aging_coefficient': aging_coef,
            'time_avg_msd': time_avg_msd,
            'ensemble_avg_msd': ensemble_msd_trimmed,
            'segment_msds': segment_msds_trimmed
        }
    
    def analyze_coupling(
        self,
        min_pause_threshold: float = 0.01
    ) -> Dict:
        """
        Test for coupling between waiting times and jump lengths.
        
        Coupling indicates dependence between binding duration and subsequent motion.
        
        Returns:
        --------
        dict with keys:
            - 'correlation_coefficient': float
            - 'p_value': float
            - 'is_coupled': bool
        """
        waiting_times = []
        subsequent_jumps = []
        
        for track_id in self.tracks_df['track_id'].unique():
            track = self.tracks_df[self.tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track) < 3:
                continue
            
            # Calculate displacements
            dx = np.diff(track['x_um'].values)
            dy = np.diff(track['y_um'].values)
            displacements = np.sqrt(dx**2 + dy**2)
            
            # Find pause-movement pairs
            current_wait = 0
            for i in range(len(displacements)):
                if displacements[i] < min_pause_threshold:
                    current_wait += self.frame_interval
                else:
                    if current_wait > 0:
                        waiting_times.append(current_wait)
                        subsequent_jumps.append(displacements[i])
                    current_wait = 0
        
        if len(waiting_times) < 3:
            return {
                'correlation_coefficient': 0.0,
                'p_value': 1.0,
                'is_coupled': False
            }
        
        # Calculate correlation
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(waiting_times, subsequent_jumps)
        
        is_coupled = (p_value < 0.05) and (abs(corr) > 0.3)
        
        return {
            'correlation_coefficient': corr,
            'p_value': p_value,
            'is_coupled': is_coupled,
            'n_pairs': len(waiting_times)
        }


def fit_fbm_model(
    tracks_df: pd.DataFrame,
    pixel_size: float = 0.1,
    frame_interval: float = 0.1
) -> Dict:
    """
    Fit Fractional Brownian Motion (FBM) model explicitly.
    
    FBM is characterized by Hurst exponent H:
    - H = 0.5: Standard Brownian motion
    - H < 0.5: Anti-persistent (returns tend to reverse)
    - H > 0.5: Persistent (trends continue)
    
    MSD scaling: MSD ~ t^(2H)
    
    Parameters:
    -----------
    tracks_df : pd.DataFrame
        Track data
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Time between frames in seconds
    
    Returns:
    --------
    dict with keys:
        - 'hurst_exponent': float
        - 'diffusion_coefficient': float
        - 'persistence_type': str
        - 'r_squared': float
        - 'interpretation': str
    """
    from analysis import calculate_msd
    
    # Calculate MSD
    msd_results = calculate_msd(tracks_df, max_lag=100, pixel_size=pixel_size, frame_interval=frame_interval)
    
    if isinstance(msd_results, dict):
        msd_df = msd_results.get('ensemble_msd')
    else:
        msd_df = msd_results
    
    if msd_df is None or msd_df.empty or len(msd_df) < 3:
        return {
            'success': False,
            'error': 'Insufficient MSD data'
        }
    
    # Fit MSD = 2*d*D*t^(2H) where d is dimensionality (2 for 2D)
    # log(MSD) = log(2*d*D) + 2H*log(t)
    
    lag_times = msd_df['lag_time'].values[1:10]  # Use first few points
    msd_values = msd_df['msd'].values[1:10]
    
    # Remove any invalid values
    valid = (lag_times > 0) & (msd_values > 0)
    lag_times = lag_times[valid]
    msd_values = msd_values[valid]
    
    if len(lag_times) < 3:
        return {
            'success': False,
            'error': 'Insufficient valid data points'
        }
    
    # Fit in log-log space
    log_t = np.log(lag_times)
    log_msd = np.log(msd_values)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_msd)
    
    # Extract Hurst exponent: slope = 2H
    hurst = slope / 2.0
    
    # Extract diffusion coefficient: intercept = log(2*d*D)
    d = 2  # 2D
    log_D = (intercept - np.log(2 * d)) / 1.0
    D = np.exp(log_D)
    
    # Classify persistence type
    if abs(hurst - 0.5) < 0.1:
        persistence_type = "Brownian (H ≈ 0.5)"
        interpretation = "Normal diffusion without memory"
    elif hurst < 0.5:
        persistence_type = "Anti-persistent (H < 0.5)"
        interpretation = "Motion tends to reverse direction (negative correlation)"
    else:
        persistence_type = "Persistent (H > 0.5)"
        interpretation = "Motion tends to continue in same direction (positive correlation)"
    
    return {
        'success': True,
        'hurst_exponent': hurst,
        'diffusion_coefficient': D,
        'persistence_type': persistence_type,
        'r_squared': r_value**2,
        'interpretation': interpretation,
        'fitted_slope': slope,
        'fitted_intercept': intercept
    }
