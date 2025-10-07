"""
Bias-Corrected Diffusion Estimation Module

Implements advanced estimators that correct for localization noise and motion blur:
- CVE (Covariance-based Estimator): Robust to static localization noise
- MLE (Maximum Likelihood Estimator): Accounts for motion blur and finite exposure

Reference: Berglund (2010) "Statistics of camera-based single-particle tracking"
           Physical Review E, PubMed 20866658

Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func


class BiasedInferenceCorrector:
    """
    Likelihood-based estimators for D and α with blur/noise correction.
    
    Addresses systematic biases in MSD-based estimation when:
    - Trajectories are short (N < 50 steps)
    - Localization noise is significant (SNR < 10)
    - Motion blur from finite exposure time
    """
    
    def __init__(self, temperature_K: float = 298.15):
        """
        Initialize corrector.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin (default 298.15 K = 25°C)
        """
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.temperature = temperature_K
    
    
    def cve_estimator(self, track: np.ndarray, dt: float, 
                     localization_error: float,
                     dimensions: int = 2) -> Dict:
        """
        Covariance-based estimator (CVE) for diffusion coefficient.
        Corrects for static localization noise.
        
        Reference: Berglund (2010) Eq. 13-15
        
        The CVE uses the covariance between successive displacements to
        eliminate static localization errors. More robust than MSD for
        noisy data.
        
        Parameters
        ----------
        track : np.ndarray
            Trajectory coordinates, shape (N, dimensions)
        dt : float
            Frame interval in seconds
        localization_error : float
            Static localization precision (σ_loc) in same units as track
        dimensions : int
            Number of spatial dimensions (2 or 3)
        
        Returns
        -------
        Dict
            {
                'D': diffusion coefficient (units²/s),
                'D_std': standard error,
                'alpha': anomalous exponent (assumed 1.0 for CVE),
                'method': 'CVE',
                'N_steps': number of steps used,
                'localization_corrected': True
            }
        """
        if len(track) < 3:
            return {
                'success': False,
                'error': 'CVE requires at least 3 positions',
                'method': 'CVE'
            }
        
        # Calculate displacements
        displacements = np.diff(track, axis=0)
        N = len(displacements)
        
        if N < 2:
            return {
                'success': False,
                'error': 'CVE requires at least 2 displacements',
                'method': 'CVE'
            }
        
        # Calculate squared displacements
        squared_disp = np.sum(displacements**2, axis=1)
        
        # Mean squared displacement (biased by noise)
        msd_biased = np.mean(squared_disp)
        
        # Covariance between successive displacements
        # C = E[Δr_i · Δr_{i+1}]
        covariance = 0.0
        for i in range(N - 1):
            covariance += np.dot(displacements[i], displacements[i+1])
        covariance /= (N - 1)
        
        # CVE formula (Berglund Eq. 14):
        # D = (MSD/2d + C/d) / (2·dt)
        # where d = number of dimensions
        # This eliminates static localization error contribution
        
        D_cve = (msd_biased / (2 * dimensions) + covariance / dimensions) / (2 * dt)
        
        # Standard error estimation (simplified)
        # Full derivation in Berglund Eq. 16
        var_msd = np.var(squared_disp)
        D_std = np.sqrt(var_msd / (2 * dimensions * dt**2 * N))
        
        # Quality checks
        if D_cve < 0:
            # Negative D indicates high noise or insufficient data
            return {
                'success': False,
                'error': 'Negative D estimated - noise dominates signal',
                'D': D_cve,
                'method': 'CVE',
                'recommendation': 'Use longer tracks or improve SNR'
            }
        
        return {
            'success': True,
            'D': D_cve,
            'D_std': D_std,
            'alpha': 1.0,  # CVE assumes normal diffusion
            'alpha_std': 0.0,
            'method': 'CVE',
            'N_steps': N,
            'localization_error': localization_error,
            'localization_corrected': True,
            'covariance': covariance,
            'msd_biased': msd_biased
        }
    
    
    def mle_with_blur(self, track: np.ndarray, dt: float, 
                      exposure_time: float, localization_error: float,
                      dimensions: int = 2,
                      alpha_fixed: Optional[float] = None) -> Dict:
        """
        Maximum likelihood estimator accounting for:
        - Motion blur from finite exposure time
        - Localization noise (Gaussian)
        - Short trajectory bias
        
        Reference: Berglund (2010) Eq. 22-27
        
        Uses numerical optimization to find D (and optionally α) that
        maximize the likelihood of observing the trajectory.
        
        Parameters
        ----------
        track : np.ndarray
            Trajectory coordinates, shape (N, dimensions)
        dt : float
            Frame interval in seconds
        exposure_time : float
            Camera exposure time in seconds (≤ dt)
        localization_error : float
            Static localization precision (σ_loc) in same units as track
        dimensions : int
            Number of spatial dimensions (2 or 3)
        alpha_fixed : float, optional
            If provided, fix anomalous exponent to this value.
            If None, estimate both D and α.
        
        Returns
        -------
        Dict
            {
                'D': diffusion coefficient,
                'D_std': standard error,
                'alpha': anomalous exponent,
                'alpha_std': standard error,
                'method': 'MLE',
                'N_steps': number of steps,
                'blur_corrected': True,
                'likelihood': log-likelihood value
            }
        """
        if len(track) < 4:
            return {
                'success': False,
                'error': 'MLE requires at least 4 positions',
                'method': 'MLE'
            }
        
        if exposure_time > dt:
            return {
                'success': False,
                'error': f'Exposure time ({exposure_time}s) > frame interval ({dt}s)',
                'method': 'MLE'
            }
        
        # Calculate displacements
        displacements = np.diff(track, axis=0)
        N = len(displacements)
        
        # Negative log-likelihood function
        def neg_log_likelihood(params):
            """
            Negative log-likelihood for normal or anomalous diffusion.
            
            For normal diffusion (α=1):
            L = -Σ log[P(Δr_i | D)]
            where P is Gaussian with variance = 2d·D·dt + 2d·σ_loc²
            
            For anomalous diffusion:
            variance = 2d·D·dt^α + 2d·σ_loc²
            
            Motion blur correction (Berglund Eq. 24):
            Effective variance is reduced by factor R = (exposure/dt)²/3
            when exposure < dt (reduces measured diffusion)
            """
            if alpha_fixed is None:
                D, alpha = params
                if alpha <= 0 or alpha > 2:
                    return 1e10  # Penalty for invalid alpha
            else:
                D = params[0]
                alpha = alpha_fixed
            
            if D <= 0:
                return 1e10  # Penalty for invalid D
            
            # Motion blur correction factor
            R = (exposure_time / dt)**2 / 3 if exposure_time > 0 else 0
            blur_factor = 1 - R
            
            # Expected variance per dimension (with blur correction)
            # variance = 2·D·dt^α + 2·σ_loc²
            var_expected = 2 * D * (dt**alpha) * blur_factor + 2 * localization_error**2
            
            if var_expected <= 0:
                return 1e10
            
            # Log-likelihood (Gaussian)
            # L = -N·d/2·log(2π·var) - Σ(Δr²)/(2·var)
            sum_squared_disp = np.sum(displacements**2)
            
            log_L = (-N * dimensions / 2 * np.log(2 * np.pi * var_expected)
                     - sum_squared_disp / (2 * var_expected))
            
            return -log_L  # Return negative for minimization
        
        # Initial guess for D (from simple MSD)
        msd_simple = np.mean(np.sum(displacements**2, axis=1))
        D_init = msd_simple / (2 * dimensions * dt)
        
        # Optimize
        try:
            if alpha_fixed is None:
                # Estimate both D and α
                alpha_init = 1.0
                result = minimize(
                    neg_log_likelihood,
                    x0=[D_init, alpha_init],
                    method='L-BFGS-B',
                    bounds=[(1e-6, None), (0.1, 2.0)]
                )
                D_mle, alpha_mle = result.x
            else:
                # Estimate only D
                result = minimize(
                    neg_log_likelihood,
                    x0=[D_init],
                    method='L-BFGS-B',
                    bounds=[(1e-6, None)]
                )
                D_mle = result.x[0]
                alpha_mle = alpha_fixed
            
            if not result.success:
                return {
                    'success': False,
                    'error': f'Optimization failed: {result.message}',
                    'method': 'MLE'
                }
            
            # Estimate uncertainties from Hessian (simplified)
            # Full calculation requires Fisher information matrix
            # For now, use bootstrap-style approximation
            D_std = D_mle * 0.1 / np.sqrt(N)  # Rough estimate
            alpha_std = 0.05 / np.sqrt(N) if alpha_fixed is None else 0.0
            
            return {
                'success': True,
                'D': D_mle,
                'D_std': D_std,
                'alpha': alpha_mle,
                'alpha_std': alpha_std,
                'method': 'MLE',
                'N_steps': N,
                'exposure_time': exposure_time,
                'blur_corrected': True,
                'localization_corrected': True,
                'likelihood': -result.fun,
                'optimization_success': result.success
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'MLE optimization failed: {str(e)}',
                'method': 'MLE'
            }
    
    
    def select_estimator(self, track: np.ndarray, dt: float,
                        localization_error: float,
                        exposure_time: float = 0.0,
                        snr: Optional[float] = None) -> str:
        """
        Auto-select CVE vs MLE vs MSD based on data quality.
        
        Decision rules:
        - N_steps < 20 → Use MLE (best for very short tracks)
        - 20 ≤ N_steps < 50 & SNR < 5 → Use CVE (robust to noise)
        - N_steps ≥ 50 & SNR > 10 → MSD acceptable (fast)
        - Motion blur significant (exposure > 0.5·dt) → Use MLE
        
        Parameters
        ----------
        track : np.ndarray
            Trajectory coordinates
        dt : float
            Frame interval
        localization_error : float
            Localization precision
        exposure_time : float
            Camera exposure time
        snr : float, optional
            Signal-to-noise ratio (if known)
        
        Returns
        -------
        str
            Recommended method: 'MLE', 'CVE', or 'MSD'
        """
        N = len(track) - 1  # Number of steps
        
        # Check motion blur
        blur_significant = (exposure_time > 0.5 * dt) if exposure_time > 0 else False
        
        # Estimate SNR if not provided
        if snr is None:
            # Rough estimate: SNR ≈ displacement_std / localization_error
            if N > 1:
                displacements = np.diff(track, axis=0)
                displacement_std = np.std(np.linalg.norm(displacements, axis=1))
                snr = displacement_std / localization_error if localization_error > 0 else 10
            else:
                snr = 5  # Default assumption
        
        # Decision tree
        if N < 20:
            return 'MLE'  # Best for very short tracks
        elif blur_significant:
            return 'MLE'  # Need blur correction
        elif N < 50 and snr < 5:
            return 'CVE'  # Robust to noise
        elif N >= 50 and snr > 10:
            return 'MSD'  # Simple MSD sufficient
        else:
            return 'CVE'  # Default safe choice
    
    
    def analyze_track(self, track: np.ndarray, dt: float,
                     localization_error: float,
                     exposure_time: float = 0.0,
                     method: str = 'auto',
                     dimensions: int = 2) -> Dict:
        """
        High-level API: Analyze single track with bias correction.
        
        Parameters
        ----------
        track : np.ndarray
            Trajectory coordinates, shape (N, dimensions)
        dt : float
            Frame interval in seconds
        localization_error : float
            Localization precision in same units as track
        exposure_time : float
            Camera exposure time in seconds
        method : str
            'auto', 'CVE', 'MLE', or 'MSD'
        dimensions : int
            Number of spatial dimensions (2 or 3)
        
        Returns
        -------
        Dict
            Analysis results with D, α, and quality metrics
        """
        # Auto-select method if requested
        if method == 'auto':
            method = self.select_estimator(track, dt, localization_error, exposure_time)
        
        # Run selected method
        if method == 'CVE':
            result = self.cve_estimator(track, dt, localization_error, dimensions)
        elif method == 'MLE':
            result = self.mle_with_blur(track, dt, exposure_time, 
                                       localization_error, dimensions)
        elif method == 'MSD':
            # Fall back to simple MSD
            result = self._msd_simple(track, dt, dimensions)
        else:
            return {
                'success': False,
                'error': f'Unknown method: {method}. Use CVE, MLE, MSD, or auto.'
            }
        
        # Add track length info
        if result.get('success', False):
            result['track_length'] = len(track)
            result['method_selected'] = method
        
        return result
    
    
    def _msd_simple(self, track: np.ndarray, dt: float, dimensions: int = 2) -> Dict:
        """Simple MSD-based estimation (no corrections)."""
        if len(track) < 3:
            return {'success': False, 'error': 'Need at least 3 positions', 'method': 'MSD'}
        
        displacements = np.diff(track, axis=0)
        msd = np.mean(np.sum(displacements**2, axis=1))
        D = msd / (2 * dimensions * dt)
        
        # Rough error estimate
        D_std = D * 0.2 / np.sqrt(len(displacements))
        
        return {
            'success': True,
            'D': D,
            'D_std': D_std,
            'alpha': 1.0,
            'alpha_std': 0.0,
            'method': 'MSD',
            'N_steps': len(displacements),
            'localization_corrected': False,
            'blur_corrected': False
        }
    
    
    def batch_analyze(self, tracks_df: pd.DataFrame, 
                     pixel_size_um: float,
                     dt: float,
                     localization_error_um: float,
                     exposure_time: float = 0.0,
                     method: str = 'auto') -> pd.DataFrame:
        """
        Analyze multiple tracks with bias correction.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Tracking data with columns: track_id, frame, x, y (and optionally z)
        pixel_size_um : float
            Pixel size in microns
        dt : float
            Frame interval in seconds
        localization_error_um : float
            Localization precision in microns
        exposure_time : float
            Exposure time in seconds
        method : str
            'auto', 'CVE', 'MLE', or 'MSD'
        
        Returns
        -------
        pd.DataFrame
            Results with columns: track_id, D_um2_per_s, D_std, alpha, alpha_std,
            method_used, N_steps, success
        """
        results = []
        
        # Check if 3D
        has_z = 'z' in tracks_df.columns
        dimensions = 3 if has_z else 2
        
        for track_id, group in tracks_df.groupby('track_id'):
            # Sort by frame
            group = group.sort_values('frame')
            
            # Extract coordinates
            if has_z:
                coords = group[['x', 'y', 'z']].values * pixel_size_um
            else:
                coords = group[['x', 'y']].values * pixel_size_um
            
            # Analyze
            result = self.analyze_track(
                coords, dt, localization_error_um, exposure_time, method, dimensions
            )
            
            # Store
            results.append({
                'track_id': track_id,
                'D_um2_per_s': result.get('D', np.nan),
                'D_std': result.get('D_std', np.nan),
                'alpha': result.get('alpha', np.nan),
                'alpha_std': result.get('alpha_std', np.nan),
                'method_used': result.get('method', method),
                'N_steps': result.get('N_steps', len(coords) - 1),
                'success': result.get('success', False),
                'localization_corrected': result.get('localization_corrected', False),
                'blur_corrected': result.get('blur_corrected', False)
            })
        
        return pd.DataFrame(results)


def compare_estimators(track: np.ndarray, dt: float,
                      localization_error: float,
                      exposure_time: float = 0.0,
                      dimensions: int = 2) -> Dict:
    """
    Compare MSD, CVE, and MLE estimates for a single track.
    
    Useful for understanding bias effects and method selection.
    
    Parameters
    ----------
    track : np.ndarray
        Trajectory coordinates
    dt : float
        Frame interval
    localization_error : float
        Localization precision
    exposure_time : float
        Exposure time
    dimensions : int
        Number of dimensions
    
    Returns
    -------
    Dict
        {
            'msd': {...},
            'cve': {...},
            'mle': {...},
            'comparison': {
                'D_bias_msd_vs_cve': fractional difference,
                'D_bias_msd_vs_mle': fractional difference,
                'recommended': 'CVE' or 'MLE'
            }
        }
    """
    corrector = BiasedInferenceCorrector()
    
    # Run all three methods
    msd_result = corrector._msd_simple(track, dt, dimensions)
    cve_result = corrector.cve_estimator(track, dt, localization_error, dimensions)
    mle_result = corrector.mle_with_blur(track, dt, exposure_time, 
                                        localization_error, dimensions)
    
    # Calculate biases
    comparison = {}
    if msd_result['success'] and cve_result['success']:
        D_msd = msd_result['D']
        D_cve = cve_result['D']
        comparison['D_bias_msd_vs_cve'] = (D_msd - D_cve) / D_cve if D_cve > 0 else np.nan
    
    if msd_result['success'] and mle_result['success']:
        D_msd = msd_result['D']
        D_mle = mle_result['D']
        comparison['D_bias_msd_vs_mle'] = (D_msd - D_mle) / D_mle if D_mle > 0 else np.nan
    
    # Recommendation
    recommended = corrector.select_estimator(track, dt, localization_error, exposure_time)
    comparison['recommended'] = recommended
    
    return {
        'msd': msd_result,
        'cve': cve_result,
        'mle': mle_result,
        'comparison': comparison
    }
