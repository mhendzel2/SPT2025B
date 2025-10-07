"""
Differential Dynamic Microscopy (DDM) Analysis Module

DDM enables tracking-free microrheology from image sequences alone.
Critical for:
- Dense samples where tracking fails
- High particle density (>10 particles/100μm²)
- Systems with frequent overlap/crossings
- Validation of tracking-based methods

Algorithm: Fourier analysis of intensity differences between frames
Result: Image structure function → MSD(q,τ) → G*(ω)

Reference: BioRxiv 2025.01.09.632077 (Wilson et al. 2025)
Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from scipy.fft import fft2, ifft2, fftfreq
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, List
import warnings


class DDMAnalyzer:
    """
    Differential Dynamic Microscopy analyzer.
    
    Computes ensemble MSD from image sequences without tracking.
    """
    
    def __init__(self, pixel_size_um: float = 0.1, 
                 frame_interval_s: float = 0.1,
                 temperature_k: float = 298.15):
        """
        Initialize DDM analyzer.
        
        Parameters
        ----------
        pixel_size_um : float
            Pixel size in micrometers
        frame_interval_s : float
            Time between frames in seconds
        temperature_k : float
            Temperature in Kelvin (for rheology)
        """
        self.pixel_size_um = pixel_size_um
        self.frame_interval_s = frame_interval_s
        self.temperature_k = temperature_k
        
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
    
    
    def compute_image_structure_function(self, 
                                         image_stack: np.ndarray,
                                         lag_frames: Optional[List[int]] = None,
                                         q_range_um_inv: Optional[Tuple[float, float]] = None,
                                         subtract_background: bool = True,
                                         background_method: str = 'temporal_median') -> Dict:
        """
        Compute image structure function D(q, τ).
        
        D(q,τ) = <|δI(q,τ)|²> where δI is Fourier transform of intensity difference
        
        Parameters
        ----------
        image_stack : np.ndarray
            3D array (n_frames, height, width) of grayscale images
        lag_frames : list of int, optional
            Lag times in frames. Default: [1, 2, 5, 10, 20, 50]
        q_range_um_inv : tuple, optional
            (q_min, q_max) in μm⁻¹. Default: use all available q
        subtract_background : bool
            Whether to subtract background before DDM analysis
        background_method : str
            'temporal_median': Use temporal median as background
            'temporal_mean': Use temporal mean
            'rolling_ball': Use rolling ball algorithm (spatial)
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'q_values_um_inv': ndarray,  # Wavevectors
                'lag_times_s': ndarray,       # Lag times
                'D_q_tau': ndarray,           # Structure function (n_q, n_tau)
                'A_q': ndarray,               # Amplitude factor (n_q,)
                'B_q': ndarray,               # Background (n_q,)
                'background_subtracted': bool
            }
        """
        if image_stack.ndim != 3:
            return {
                'success': False,
                'error': 'image_stack must be 3D array (n_frames, height, width)'
            }
        
        n_frames, height, width = image_stack.shape
        
        if n_frames < 10:
            return {
                'success': False,
                'error': 'Need at least 10 frames for DDM analysis'
            }
        
        # Background subtraction
        if subtract_background:
            if background_method == 'temporal_median':
                background = np.median(image_stack, axis=0)
                image_stack_corrected = image_stack - background[np.newaxis, :, :]
            elif background_method == 'temporal_mean':
                background = np.mean(image_stack, axis=0)
                image_stack_corrected = image_stack - background[np.newaxis, :, :]
            elif background_method == 'rolling_ball':
                # Simple rolling ball: subtract minimum over sliding window
                from scipy.ndimage import uniform_filter
                window_size = min(height, width) // 10
                background = np.zeros_like(image_stack)
                for i in range(n_frames):
                    background[i] = uniform_filter(image_stack[i], size=window_size, mode='reflect')
                image_stack_corrected = image_stack - background
            else:
                warnings.warn(f'Unknown background method: {background_method}, using temporal_median')
                background = np.median(image_stack, axis=0)
                image_stack_corrected = image_stack - background[np.newaxis, :, :]
            
            # Ensure non-negative intensities
            image_stack_corrected = np.maximum(image_stack_corrected, 0)
        else:
            image_stack_corrected = image_stack.copy()
        
        # Default lag times
        if lag_frames is None:
            max_lag = min(n_frames // 4, 100)
            lag_frames = np.unique(np.logspace(0, np.log10(max_lag), 15).astype(int))
            lag_frames = lag_frames[lag_frames > 0]
        
        lag_times_s = np.array(lag_frames) * self.frame_interval_s
        
        # Compute Fourier frequencies
        qy = fftfreq(height, d=self.pixel_size_um) * 2 * np.pi
        qx = fftfreq(width, d=self.pixel_size_um) * 2 * np.pi
        qy_grid, qx_grid = np.meshgrid(qy, qx, indexing='ij')
        q_magnitude = np.sqrt(qx_grid**2 + qy_grid**2)
        
        # Filter by q_range
        if q_range_um_inv is not None:
            q_min, q_max = q_range_um_inv
            q_mask = (q_magnitude >= q_min) & (q_magnitude <= q_max)
        else:
            # Default: exclude DC and very high q (noise)
            q_min = 2 * np.pi / (min(height, width) * self.pixel_size_um)
            q_max = np.pi / (2 * self.pixel_size_um)  # Nyquist / 2
            q_mask = (q_magnitude >= q_min) & (q_magnitude <= q_max)
        
        # Radial binning of q values
        q_valid = q_magnitude[q_mask]
        n_q_bins = 20
        q_bins = np.logspace(np.log10(q_valid.min()), np.log10(q_valid.max()), n_q_bins + 1)
        q_values_um_inv = (q_bins[:-1] + q_bins[1:]) / 2
        
        # Initialize structure function
        D_q_tau = np.zeros((len(q_values_um_inv), len(lag_frames)))
        counts = np.zeros((len(q_values_um_inv), len(lag_frames)))
        
        # Compute for each lag time
        for tau_idx, tau in enumerate(lag_frames):
            if tau >= n_frames:
                continue
            
            # Accumulate differences
            delta_I_squared_accum = np.zeros((height, width), dtype=complex)
            n_pairs = 0
            
            for t in range(n_frames - tau):
                # Intensity difference (use background-corrected images)
                delta_I = image_stack_corrected[t + tau] - image_stack_corrected[t]
                
                # Fourier transform
                delta_I_fft = fft2(delta_I)
                
                # Accumulate |δI(q,τ)|²
                delta_I_squared_accum += np.abs(delta_I_fft)**2
                n_pairs += 1
            
            # Average
            if n_pairs > 0:
                delta_I_squared_avg = delta_I_squared_accum / n_pairs
                
                # Radial binning
                for q_idx in range(len(q_values_um_inv)):
                    q_low = q_bins[q_idx]
                    q_high = q_bins[q_idx + 1]
                    bin_mask = (q_magnitude >= q_low) & (q_magnitude < q_high) & q_mask
                    
                    if np.sum(bin_mask) > 0:
                        D_q_tau[q_idx, tau_idx] = np.mean(delta_I_squared_avg[bin_mask].real)
                        counts[q_idx, tau_idx] = np.sum(bin_mask)
        
        # Remove bins with insufficient data
        valid_q = np.sum(counts > 0, axis=1) > len(lag_frames) // 2
        q_values_um_inv = q_values_um_inv[valid_q]
        D_q_tau = D_q_tau[valid_q]
        
        # Fit to model: D(q,τ) = A(q) * [1 - exp(-τ/τ_q)] + B(q)
        # This extracts A(q) and B(q)
        A_q = np.zeros(len(q_values_um_inv))
        B_q = np.zeros(len(q_values_um_inv))
        
        for q_idx in range(len(q_values_um_inv)):
            curve = D_q_tau[q_idx]
            if np.all(curve > 0):
                A_q[q_idx] = np.max(curve) - np.min(curve)
                B_q[q_idx] = np.min(curve)
            else:
                A_q[q_idx] = 0
                B_q[q_idx] = 0
        
        return {
            'success': True,
            'q_values_um_inv': q_values_um_inv,
            'lag_times_s': lag_times_s,
            'D_q_tau': D_q_tau,
            'A_q': A_q,
            'B_q': B_q,
            'n_frames_analyzed': n_frames,
            'spatial_resolution_um': (height * self.pixel_size_um, width * self.pixel_size_um)
        }
    
    
    def extract_msd_from_structure_function(self, ddm_result: Dict) -> Dict:
        """
        Convert D(q,τ) to ensemble MSD(τ).
        
        For Brownian diffusion: D(q,τ) = A(q) * [1 - exp(-q²·MSD(τ)/d)]
        Where d = dimensionality (2 or 3)
        
        Parameters
        ----------
        ddm_result : Dict
            Output from compute_image_structure_function()
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'lag_times_s': ndarray,
                'msd_um2': ndarray,
                'diffusion_coeff_um2_s': float,
                'alpha_exponent': float (from power law fit)
            }
        """
        if not ddm_result.get('success', False):
            return {
                'success': False,
                'error': 'Invalid DDM result'
            }
        
        q_values = ddm_result['q_values_um_inv']
        lag_times = ddm_result['lag_times_s']
        D_q_tau = ddm_result['D_q_tau']
        A_q = ddm_result['A_q']
        
        # For each τ, fit D(q,τ) vs q² to extract MSD(τ)
        # Model: D(q,τ) ≈ A(q) * [1 - exp(-q²·MSD/d)]
        # For small q²·MSD: D ≈ A(q) * q² * MSD / d
        
        msd_values = np.zeros(len(lag_times))
        
        for tau_idx in range(len(lag_times)):
            curve = D_q_tau[:, tau_idx]
            
            # Use linear regime (small q)
            # D/A ≈ q²·MSD/d for q²·MSD << d
            q_squared = q_values**2
            
            # Normalize by A_q
            with np.errstate(divide='ignore', invalid='ignore'):
                y = curve / A_q
            
            # Fit linear part (first few points)
            valid = np.isfinite(y) & (q_squared > 0)
            if np.sum(valid) >= 3:
                # Use first 5 points or half of data
                n_fit = min(5, np.sum(valid) // 2)
                x_fit = q_squared[valid][:n_fit]
                y_fit = y[valid][:n_fit]
                
                if len(x_fit) >= 2:
                    # Linear fit: y = (MSD/d) * x
                    # Assume 2D → d=4
                    d = 4  # 2D diffusion
                    slope, _ = np.polyfit(x_fit, y_fit, 1)
                    msd_values[tau_idx] = slope * d
        
        # Remove invalid MSDs
        valid_msd = msd_values > 0
        if np.sum(valid_msd) < 3:
            return {
                'success': False,
                'error': 'Insufficient valid MSD points'
            }
        
        lag_times_valid = lag_times[valid_msd]
        msd_valid = msd_values[valid_msd]
        
        # Fit power law: MSD = 4·D·τ^α (2D)
        log_tau = np.log10(lag_times_valid)
        log_msd = np.log10(msd_valid)
        
        coeffs = np.polyfit(log_tau, log_msd, 1)
        alpha = coeffs[0]
        log_D = (coeffs[1] - np.log10(4)) / alpha
        D = 10**log_D
        
        return {
            'success': True,
            'lag_times_s': lag_times,
            'msd_um2': msd_values,
            'lag_times_valid_s': lag_times_valid,
            'msd_valid_um2': msd_valid,
            'diffusion_coeff_um2_s': D,
            'alpha_exponent': alpha,
            'fit_quality': np.corrcoef(log_tau, log_msd)[0, 1]**2  # R²
        }
    
    
    def compute_rheology_from_ddm(self, msd_result: Dict, 
                                  particle_radius_um: float,
                                  frequency_range_hz: Tuple[float, float] = (0.1, 100)) -> Dict:
        """
        Convert DDM-derived MSD to viscoelastic moduli.
        
        Uses GSER (Generalized Stokes-Einstein Relation):
        G*(ω) = k_B·T / [π·a·iω·MSD*(ω)]
        
        Parameters
        ----------
        msd_result : Dict
            Output from extract_msd_from_structure_function()
        particle_radius_um : float
            Effective particle radius in micrometers
        frequency_range_hz : tuple
            (f_min, f_max) for rheology calculation
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'frequencies_hz': ndarray,
                'g_prime_pa': ndarray,      # Storage modulus
                'g_double_prime_pa': ndarray  # Loss modulus
            }
        """
        if not msd_result.get('success', False):
            return {
                'success': False,
                'error': 'Invalid MSD result'
            }
        
        # Use valid MSD data
        if 'lag_times_valid_s' in msd_result:
            lag_times = msd_result['lag_times_valid_s']
            msd_um2 = msd_result['msd_valid_um2']
        else:
            lag_times = msd_result['lag_times_s']
            msd_um2 = msd_result['msd_um2']
        
        # Filter valid MSD
        valid = (msd_um2 > 0) & np.isfinite(msd_um2)
        if np.sum(valid) < 3:
            return {
                'success': False,
                'error': 'Insufficient valid MSD data for rheology'
            }
        
        lag_times = lag_times[valid]
        msd_um2 = msd_um2[valid]
        
        # Convert to SI units
        msd_m2 = msd_um2 * 1e-12
        particle_radius_m = particle_radius_um * 1e-6
        
        # Fourier transform MSD to get MSD*(ω)
        # Use interpolation for frequency range
        f_min, f_max = frequency_range_hz
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 30)
        omega = 2 * np.pi * frequencies
        
        # Interpolate MSD in log space
        log_tau = np.log10(lag_times)
        log_msd = np.log10(msd_m2)
        
        # Ensure monotonic for interpolation
        sort_idx = np.argsort(log_tau)
        log_tau = log_tau[sort_idx]
        log_msd = log_msd[sort_idx]
        
        # Interpolate at ω positions
        tau_at_omega = 1 / (2 * np.pi * frequencies)
        log_tau_omega = np.log10(tau_at_omega)
        
        # Extrapolate if needed
        log_msd_at_omega = np.interp(log_tau_omega, log_tau, log_msd, 
                                      left=log_msd[0], right=log_msd[-1])
        msd_at_omega = 10**log_msd_at_omega
        
        # Get alpha (slope in log-log)
        alpha = msd_result.get('alpha_exponent', 1.0)
        
        # GSER formula
        # G*(ω) = (k_B·T) / (π·a·iω·MSD*(ω))
        # For power law MSD ∝ τ^α:
        # G' ∝ cos(πα/2) / ω^α
        # G'' ∝ sin(πα/2) / ω^α
        
        prefactor = (self.k_B * self.temperature_k) / (np.pi * particle_radius_m)
        
        # Complex modulus magnitude
        G_magnitude = prefactor / (omega * msd_at_omega)
        
        # Phase from alpha
        phase_angle = np.pi * alpha / 2
        
        g_prime = G_magnitude * np.cos(phase_angle)
        g_double_prime = G_magnitude * np.sin(phase_angle)
        
        return {
            'success': True,
            'frequencies_hz': frequencies,
            'g_prime_pa': g_prime,
            'g_double_prime_pa': g_double_prime,
            'loss_tangent': g_double_prime / (g_prime + 1e-12),
            'method': 'DDM_GSER',
            'particle_radius_um': particle_radius_um,
            'alpha_exponent': alpha
        }
    
    
    def analyze_image_stack(self, image_stack: np.ndarray,
                           particle_radius_um: float,
                           lag_frames: Optional[List[int]] = None) -> Dict:
        """
        Full DDM analysis pipeline: images → structure function → MSD → G*(ω).
        
        Parameters
        ----------
        image_stack : np.ndarray
            3D array of images (n_frames, height, width)
        particle_radius_um : float
            Particle radius for rheology calculation
        lag_frames : list, optional
            Lag times in frames
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'ddm_structure': Dict (D(q,τ) data),
                'msd': Dict (MSD vs τ),
                'rheology': Dict (G' and G'' vs f),
                'summary': Dict (key statistics)
            }
        """
        # Step 1: Compute structure function
        ddm_result = self.compute_image_structure_function(
            image_stack=image_stack,
            lag_frames=lag_frames
        )
        
        if not ddm_result['success']:
            return {
                'success': False,
                'error': ddm_result.get('error', 'DDM structure function failed')
            }
        
        # Step 2: Extract MSD
        msd_result = self.extract_msd_from_structure_function(ddm_result)
        
        if not msd_result['success']:
            return {
                'success': False,
                'error': msd_result.get('error', 'MSD extraction failed'),
                'ddm_structure': ddm_result
            }
        
        # Step 3: Compute rheology
        rheology_result = self.compute_rheology_from_ddm(
            msd_result=msd_result,
            particle_radius_um=particle_radius_um
        )
        
        # Summary statistics
        summary = {
            'n_frames': ddm_result['n_frames_analyzed'],
            'diffusion_coeff_um2_s': msd_result.get('diffusion_coeff_um2_s', np.nan),
            'alpha_exponent': msd_result.get('alpha_exponent', np.nan),
            'spatial_resolution_um': ddm_result['spatial_resolution_um'],
            'q_range_um_inv': (
                ddm_result['q_values_um_inv'].min(),
                ddm_result['q_values_um_inv'].max()
            ),
            'time_range_s': (
                ddm_result['lag_times_s'].min(),
                ddm_result['lag_times_s'].max()
            ),
            'rheology_success': rheology_result.get('success', False)
        }
        
        return {
            'success': True,
            'ddm_structure': ddm_result,
            'msd': msd_result,
            'rheology': rheology_result if rheology_result['success'] else None,
            'summary': summary
        }


def quick_ddm_analysis(image_stack: np.ndarray,
                      pixel_size_um: float,
                      frame_interval_s: float,
                      particle_radius_um: float) -> Dict:
    """
    One-liner DDM analysis.
    
    Parameters
    ----------
    image_stack : np.ndarray
        3D image array
    pixel_size_um : float
        Pixel size
    frame_interval_s : float
        Frame interval
    particle_radius_um : float
        Particle radius
    
    Returns
    -------
    Dict
        Complete analysis results
    """
    analyzer = DDMAnalyzer(
        pixel_size_um=pixel_size_um,
        frame_interval_s=frame_interval_s
    )
    
    return analyzer.analyze_image_stack(
        image_stack=image_stack,
        particle_radius_um=particle_radius_um
    )
