"""
Bias-Corrected Diffusion Estimation Module

Implements advanced estimators that correct for localization noise and motion blur:
- CVE (Covariance-based Estimator): Robust to static localization noise
- MLE (Maximum Likelihood Estimator): Accounts for motion blur and finite exposure
- Spot-On style population inference: Multi-population diffusion analysis with bias correction

References:
- Berglund (2010) "Statistics of camera-based single-particle tracking"
  Physical Review E, PubMed 20866658
- Hansen et al. (2018) "Robust model-based analysis of single-particle tracking experiments
  with Spot-On" eLife 7:e33125

Author: SPT2025B Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
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
        
        # Standard error estimation via Fisher information matrix
        # Cramér-Rao bound: Var(D) ≥ 1/I(D)
        # Fisher information for CVE (Berglund Eq. 16-17)
        
        # Variance of MSD and covariance
        var_msd = np.var(squared_disp)
        
        # Simplified Fisher information for normal diffusion:
        # I(D) ≈ N / (2·D²·dt²)
        fisher_info = N / (2 * D_cve**2 * dt**2) if D_cve > 0 else 1e-10
        D_std = np.sqrt(1.0 / fisher_info) if fisher_info > 0 else np.inf
        
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
            
            # Estimate uncertainties from Fisher information matrix
            # Cramér-Rao lower bound provides minimum variance
            
            # Fisher information matrix (diagonal approximation)
            # I_DD = ∂²(-log L)/∂D²
            # I_αα = ∂²(-log L)/∂α²
            
            # For Gaussian likelihood with variance σ² = 2D·dt^α + 2σ_loc²:
            # I_DD ≈ N·d / σ⁴ · (dt^α)²
            # I_αα ≈ N·d / σ⁴ · (D·dt^α·ln(dt))²
            
            R = (exposure_time / dt)**2 / 3 if exposure_time > 0 else 0
            blur_factor = 1 - R
            var_est = 2 * D_mle * (dt**alpha_mle) * blur_factor + 2 * localization_error**2
            
            if var_est > 0:
                fisher_D = N * dimensions / (var_est**2) * ((dt**alpha_mle) * blur_factor)**2
                D_std = np.sqrt(1.0 / fisher_D) if fisher_D > 0 else D_mle * 0.1
                
                if alpha_fixed is None:
                    fisher_alpha = N * dimensions / (var_est**2) * (D_mle * (dt**alpha_mle) * np.log(dt) * blur_factor)**2
                    alpha_std = np.sqrt(1.0 / fisher_alpha) if fisher_alpha > 0 else 0.05
                else:
                    alpha_std = 0.0
            else:
                D_std = D_mle * 0.1 / np.sqrt(N)
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


    def bootstrap_confidence_intervals(self, track: np.ndarray, dt: float,
                                       method: str = 'CVE',
                                       exposure_time: float = 0.0,
                                       localization_error: float = 0.0,
                                       dimensions: int = 2,
                                       n_bootstrap: int = 1000,
                                       confidence: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence intervals for D and α.
        
        Parameters
        ----------
        track : np.ndarray
            Trajectory coordinates
        dt : float
            Frame interval (s)
        method : str
            'CVE' or 'MLE'
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level (e.g., 0.95 for 95%)
        
        Returns
        -------
        Dict with confidence intervals
        """
        N = len(track)
        D_samples = []
        alpha_samples = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(N, size=N, replace=True)
            track_boot = track[indices]
            
            if method.upper() == 'CVE':
                result = self.cve_estimator(track_boot, dt, localization_error, dimensions)
            elif method.upper() == 'MLE':
                result = self.mle_with_blur(track_boot, dt, exposure_time, localization_error, dimensions)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if result.get('success', True):
                D_samples.append(result['D'])
                alpha_samples.append(result.get('alpha', 1.0))
        
        if len(D_samples) == 0:
            return {'success': False, 'error': 'All bootstrap samples failed'}
        
        # Calculate percentile-based confidence intervals
        alpha_level = (1 - confidence) / 2
        
        return {
            'success': True,
            'D_mean': np.mean(D_samples),
            'D_ci_lower': np.percentile(D_samples, alpha_level * 100),
            'D_ci_upper': np.percentile(D_samples, (1 - alpha_level) * 100),
            'D_std': np.std(D_samples),
            'alpha_mean': np.mean(alpha_samples),
            'alpha_ci_lower': np.percentile(alpha_samples, alpha_level * 100),
            'alpha_ci_upper': np.percentile(alpha_samples, (1 - alpha_level) * 100),
            'alpha_std': np.std(alpha_samples),
            'n_bootstrap': len(D_samples),
            'confidence': confidence,
            'method': f'{method}_bootstrap'
        }
    
    
    def detect_anisotropic_diffusion(self, track: np.ndarray, dt: float,
                                    dimensions: int = 2) -> Dict:
        """
        Detect anisotropic diffusion by analyzing 2D covariance matrix.
        
        Parameters
        ----------
        track : np.ndarray
            Trajectory coordinates
        dt : float
            Frame interval (s)
        dimensions : int
            Number of dimensions (2 or 3)
        
        Returns
        -------
        Dict with anisotropy metrics
        """
        if dimensions not in [2, 3]:
            return {'success': False, 'error': 'Need 2D or 3D tracks'}
        
        if len(track) < 10:
            return {'success': False, 'error': 'Need at least 10 positions'}
        
        # Calculate displacements
        displacements = np.diff(track, axis=0)
        
        # Covariance matrix
        cov_matrix = np.cov(displacements.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Convert to diffusion coefficients
        D_values = eigenvalues / (2 * dt)
        
        # Anisotropy ratio
        anisotropy_ratio = D_values[0] / D_values[-1] if D_values[-1] > 0 else np.inf
        
        # Statistical test
        D_mean = np.mean(D_values)
        chi2_stat = np.sum((D_values - D_mean)**2) / (D_mean**2) if D_mean > 0 else 0
        threshold = dimensions * 1.5
        p_value = np.exp(-chi2_stat / threshold) if chi2_stat > 0 else 1.0
        
        isotropic = (anisotropy_ratio < 2.0) and (p_value > 0.05)
        
        return {
            'success': True,
            'isotropic': isotropic,
            'anisotropy_ratio': anisotropy_ratio,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'D_values': D_values,
            'D_mean': D_mean,
            'D_max': D_values[0],
            'D_min': D_values[-1],
            'principal_direction': eigenvectors[:, 0],
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'interpretation': 'Isotropic' if isotropic else f'Anisotropic (ratio={anisotropy_ratio:.2f})'
        }


def compare_estimators(track: np.ndarray, dt: float,
                      localization_error: float,
                      exposure_time: float = 0.0,
                      dimensions: int = 2) -> Dict:
    """
    Compare MSD, CVE, and MLE estimates for a single track.
    
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
    Dict with comparison results
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


# ============================================================================
# Spot-On Style Population Inference
# ============================================================================

class SpotOnPopulationInference:
    """
    Spot-On style bias-aware diffusion population inference.
    
    Infers fractions and diffusion coefficients of multiple diffusing populations
    while correcting for:
    - Out-of-focus bias (axial detection range)
    - Motion blur from finite exposure
    - Localization noise
    - Track length bias
    
    Based on Hansen et al. (2018) eLife 7:e33125
    """
    
    def __init__(
        self,
        frame_interval: float,
        pixel_size: float = 0.1,
        localization_error: float = 0.03,
        exposure_time: Optional[float] = None,
        axial_detection_range: Optional[float] = None,
        max_jump_gap: int = 0,
        dimensions: int = 2
    ):
        """
        Initialize Spot-On population inference.
        
        Parameters
        ----------
        frame_interval : float
            Time between frames (seconds)
        pixel_size : float
            Pixel size (micrometers)
        localization_error : float
            Localization precision (micrometers)
        exposure_time : float, optional
            Camera exposure time (seconds). Defaults to frame_interval.
        axial_detection_range : float, optional
            Axial detection range (micrometers, for 3D correction in 2D imaging)
        max_jump_gap : int
            Maximum allowed gap in tracking (frames)
        dimensions : int
            Spatial dimensions (2 or 3)
        """
        self.frame_interval = frame_interval
        self.pixel_size = pixel_size
        self.localization_error = localization_error
        self.exposure_time = exposure_time if exposure_time is not None else frame_interval
        self.axial_detection_range = axial_detection_range
        self.max_jump_gap = max_jump_gap
        self.dimensions = dimensions
        
        # Motion blur correction factor
        R = self.exposure_time / self.frame_interval
        self.blur_factor = 1.0 - (R**2) / 6.0 if R < 1 else 0.5
        
    def _out_of_focus_probability(self, D: float, dz: float) -> float:
        """
        Calculate probability of remaining in focus for one time step.
        
        For 2D imaging with finite axial detection range dz,
        fast particles can diffuse out of focus between frames.
        
        P_in_focus = erf(dz / sqrt(4*D*dt))
        
        Parameters
        ----------
        D : float
            Diffusion coefficient (µm²/s)
        dz : float
            Axial detection range (µm)
        
        Returns
        -------
        float
            Probability of staying in focus
        """
        if dz is None or dz == 0:
            return 1.0  # No out-of-focus correction
        
        from scipy.special import erf
        sigma_z = np.sqrt(2 * D * self.frame_interval)
        if sigma_z == 0:
            return 1.0
        
        return float(erf(dz / (2 * sigma_z)))
    
    def _jump_length_pdf(
        self,
        r: np.ndarray,
        D: float,
        dt: float,
        sigma_loc: float,
        dim: int = 2
    ) -> np.ndarray:
        """
        Probability density function for jump lengths.
        
        Accounts for diffusion + localization noise.
        For 2D: P(r) = r/(2σ²) * exp(-r²/(4σ²))
        where σ² = 2*D*dt + 2*σ_loc²
        
        Parameters
        ----------
        r : np.ndarray
            Jump distances (µm)
        D : float
            Diffusion coefficient (µm²/s)
        dt : float
            Time interval (s)
        sigma_loc : float
            Localization error (µm)
        dim : int
            Spatial dimensions
        
        Returns
        -------
        np.ndarray
            Probability densities
        """
        # Variance per dimension
        var_diffusion = 2 * D * dt * self.blur_factor
        var_total = var_diffusion + 2 * sigma_loc**2
        
        if var_total <= 0:
            return np.zeros_like(r)
        
        if dim == 2:
            # 2D Rayleigh distribution
            return (r / var_total) * np.exp(-r**2 / (2 * var_total))
        elif dim == 3:
            # 3D Maxwell-Boltzmann distribution
            return (4 * np.pi * r**2 / (2 * np.pi * var_total)**(3/2)) * np.exp(-r**2 / (2 * var_total))
        else:
            raise ValueError(f"Unsupported dimensions: {dim}")
    
    def fit_populations(
        self,
        jump_distances: np.ndarray,
        n_populations: int = 2,
        D_bounds: Optional[List[Tuple[float, float]]] = None,
        fraction_bounds: Optional[List[Tuple[float, float]]] = None,
        initial_guess: Optional[Dict] = None
    ) -> Dict:
        """
        Fit multiple diffusing populations to jump distance distribution.
        
        Uses maximum likelihood estimation to infer:
        - Fraction of each population
        - Diffusion coefficient of each population
        
        Parameters
        ----------
        jump_distances : np.ndarray
            Observed jump distances (µm)
        n_populations : int
            Number of populations to fit (typically 2-3)
        D_bounds : list of tuples, optional
            Bounds for diffusion coefficients [(D1_min, D1_max), ...]
        fraction_bounds : list of tuples, optional
            Bounds for fractions [(f1_min, f1_max), ...]
        initial_guess : dict, optional
            Initial parameter values
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'D_values': [D1, D2, ...],
                'fractions': [f1, f2, ...],
                'D_std': [std1, std2, ...],
                'fraction_std': [std1, std2, ...],
                'log_likelihood': float,
                'AIC': float,
                'BIC': float,
                'n_data': int
            }
        """
        from scipy.optimize import minimize
        
        if len(jump_distances) < 10:
            return {
                'success': False,
                'error': 'Need at least 10 jump distances'
            }
        
        # Remove invalid values
        jump_distances = jump_distances[np.isfinite(jump_distances) & (jump_distances > 0)]
        n_data = len(jump_distances)
        
        # Default bounds
        if D_bounds is None:
            D_bounds = [(1e-4, 0.01), (0.01, 10.0)] + [(0.01, 100.0)] * (n_populations - 2)
        
        if fraction_bounds is None:
            fraction_bounds = [(0.01, 0.99)] * (n_populations - 1)
        
        # Negative log-likelihood
        def neg_log_likelihood(params):
            # Extract parameters: [D1, D2, ..., Dn, f1, f2, ..., f(n-1)]
            # Last fraction is 1 - sum(others)
            D_values = params[:n_populations]
            fractions_partial = params[n_populations:]
            fractions = np.append(fractions_partial, 1 - np.sum(fractions_partial))
            
            # Validate
            if np.any(D_values <= 0) or np.any(fractions < 0) or fractions[-1] < 0:
                return 1e10
            
            # Calculate mixture PDF
            pdf_total = np.zeros_like(jump_distances)
            for D, f in zip(D_values, fractions):
                if f > 0:
                    # Apply out-of-focus correction
                    if self.axial_detection_range is not None:
                        p_focus = self._out_of_focus_probability(D, self.axial_detection_range)
                        f_effective = f * p_focus
                    else:
                        f_effective = f
                    
                    pdf = self._jump_length_pdf(
                        jump_distances, D, self.frame_interval,
                        self.localization_error, self.dimensions
                    )
                    pdf_total += f_effective * pdf
            
            # Avoid log(0)
            pdf_total = np.maximum(pdf_total, 1e-300)
            
            # Negative log-likelihood
            return -np.sum(np.log(pdf_total))
        
        # Initial guess
        if initial_guess is None:
            # Heuristic: use quantiles of jump distances
            quantiles = np.quantile(jump_distances, [0.25, 0.75])
            D_init = [(q**2) / (4 * self.frame_interval) for q in quantiles]
            D_init = D_init + [D_init[-1] * 10] * (n_populations - 2)
            fractions_init = [1.0 / n_populations] * (n_populations - 1)
            x0 = D_init + fractions_init
        else:
            x0 = list(initial_guess.get('D_values', [])) + list(initial_guess.get('fractions', []))
        
        # Bounds
        bounds = D_bounds + fraction_bounds
        
        # Optimize
        try:
            result = minimize(
                neg_log_likelihood,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if not result.success:
                return {
                    'success': False,
                    'error': f'Optimization failed: {result.message}'
                }
            
            # Extract results
            D_values = result.x[:n_populations]
            fractions_partial = result.x[n_populations:]
            fractions = np.append(fractions_partial, 1 - np.sum(fractions_partial))
            
            # Sort by D (slow to fast)
            idx = np.argsort(D_values)
            D_values = D_values[idx]
            fractions = fractions[idx]
            
            # Estimate uncertainties from Hessian
            # Use finite differences for diagonal elements
            epsilon = 1e-6
            hessian_diag = np.zeros(len(result.x))
            for i in range(len(result.x)):
                x_plus = result.x.copy()
                x_minus = result.x.copy()
                x_plus[i] += epsilon
                x_minus[i] -= epsilon
                
                hessian_diag[i] = (
                    neg_log_likelihood(x_plus) - 2 * neg_log_likelihood(result.x) + 
                    neg_log_likelihood(x_minus)
                ) / (epsilon**2)
            
            # Standard errors (inverse of Hessian diagonal, Cramér-Rao bound)
            std_errors = np.sqrt(1.0 / np.maximum(hessian_diag, 1e-10))
            
            D_std = std_errors[:n_populations][idx]
            fraction_std = np.append(std_errors[n_populations:], np.sqrt(np.sum(std_errors[n_populations:]**2)))[idx]
            
            # Model selection criteria
            log_likelihood = -result.fun
            n_params = len(result.x)
            AIC = 2 * n_params - 2 * log_likelihood
            BIC = n_params * np.log(n_data) - 2 * log_likelihood
            
            return {
                'success': True,
                'D_values': D_values.tolist(),
                'fractions': fractions.tolist(),
                'D_std': D_std.tolist(),
                'fraction_std': fraction_std.tolist(),
                'log_likelihood': float(log_likelihood),
                'AIC': float(AIC),
                'BIC': float(BIC),
                'n_data': int(n_data),
                'n_populations': n_populations,
                'blur_corrected': True,
                'localization_corrected': True,
                'out_of_focus_corrected': self.axial_detection_range is not None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Population inference failed: {str(e)}'
            }
    
    def model_selection(
        self,
        jump_distances: np.ndarray,
        max_populations: int = 4
    ) -> Dict:
        """
        Automatically select optimal number of populations using BIC.
        
        Parameters
        ----------
        jump_distances : np.ndarray
            Observed jump distances
        max_populations : int
            Maximum number of populations to test
        
        Returns
        -------
        Dict
            {
                'optimal_n': int,
                'results': {1: result1, 2: result2, ...},
                'BIC_values': [BIC1, BIC2, ...],
                'best_result': Dict
            }
        """
        results = {}
        BIC_values = []
        
        for n_pop in range(1, max_populations + 1):
            result = self.fit_populations(jump_distances, n_populations=n_pop)
            
            if result['success']:
                results[n_pop] = result
                BIC_values.append(result['BIC'])
            else:
                BIC_values.append(np.inf)
        
        # Find optimal
        if len(BIC_values) > 0:
            optimal_idx = np.argmin(BIC_values)
            optimal_n = optimal_idx + 1
            
            return {
                'success': True,
                'optimal_n': optimal_n,
                'results': results,
                'BIC_values': BIC_values,
                'best_result': results.get(optimal_n, {})
            }
        else:
            return {
                'success': False,
                'error': 'All models failed to converge'
            }
