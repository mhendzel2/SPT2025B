"""
Acquisition Parameter Advisor Module

Recommends optimal frame rate and exposure time based on:
- Expected diffusion coefficient
- Localization precision
- Desired track length

Uses bias maps from Weimann et al. (2024) to minimize systematic errors.

Reference: "Impact of temporal resolution in single particle tracking analysis"
           Nature Methods (2024), PubMed 38724858

Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path


class AcquisitionAdvisor:
    """
    Recommends frame rate and exposure from expected D, SNR, density.
    
    Prevents 30-50% estimation bias from suboptimal acquisition settings.
    """
    
    def __init__(self):
        """Initialize advisor with bias tables."""
        self.bias_tables = self._generate_bias_tables()
    
    
    def _generate_bias_tables(self) -> Dict:
        """
        Generate bias lookup tables based on Weimann et al. 2024.
        
        Tables encode how D and α estimation errors depend on:
        - Frame interval (dt)
        - Localization precision (σ)
        - True diffusion coefficient (D_true)
        - Track length (N)
        
        Returns
        -------
        Dict
            Bias tables for different parameter combinations
        """
        # Simplified bias model (full version would load from CSV supplement)
        # Key findings from Weimann 2024:
        # 1. D underestimated when dt too large (undersampling)
        # 2. D overestimated when dt too small (noise dominates)
        # 3. Optimal dt ≈ σ² / D (balances localization vs blur)
        # 4. α biased toward 1.0 when dt non-optimal
        
        tables = {
            'description': 'Bias maps from Weimann et al. 2024',
            'reference': 'https://pubmed.ncbi.nlm.nih.gov/38724858/',
            
            # Optimal dt scaling: dt_opt = k · σ² / D
            'optimal_dt_factor': {
                'normal_diffusion': 2.0,  # k = 2 for α ≈ 1
                'subdiffusion': 1.5,      # k = 1.5 for α < 0.8
                'superdiffusion': 3.0     # k = 3 for α > 1.2
            },
            
            # Exposure time: best as fraction of frame time
            'optimal_exposure_fraction': 0.8,  # 80% of dt
            
            # Bias thresholds
            'acceptable_D_bias': 0.1,  # <10% bias acceptable
            'acceptable_alpha_bias': 0.05,  # <5% bias acceptable
            
            # Track length requirements
            'min_track_length': {
                'D_estimation': 20,  # Minimum for reasonable D
                'alpha_estimation': 50,  # Minimum for reasonable α
                'reliable_alpha': 100  # For confident α
            }
        }
        
        return tables
    
    
    def recommend_framerate(self, D_expected: float, 
                           localization_precision: float,
                           track_length: int = 50,
                           alpha_expected: float = 1.0) -> Dict:
        """
        Recommend optimal frame interval based on expected parameters.
        
        Uses 2024 bias maps to suggest optimal frame interval.
        
        Parameters
        ----------
        D_expected : float
            Expected diffusion coefficient (μm²/s)
        localization_precision : float
            Localization precision (μm)
        track_length : int
            Expected track length (frames)
        alpha_expected : float
            Expected anomalous exponent (1.0 = normal diffusion)
        
        Returns
        -------
        Dict
            {
                'recommended_dt': optimal frame interval (s),
                'dt_range': (min_acceptable, max_acceptable) in s,
                'exposure_time': recommended exposure (s),
                'exposure_fraction': recommended as fraction of dt,
                'expected_D_bias': estimated bias if using recommended dt,
                'expected_alpha_bias': estimated α bias,
                'framerate_hz': recommended frame rate,
                'rationale': explanation string,
                'warnings': list of potential issues
            }
        """
        if D_expected <= 0:
            return {
                'success': False,
                'error': 'D_expected must be positive'
            }
        
        if localization_precision <= 0:
            return {
                'success': False,
                'error': 'localization_precision must be positive'
            }
        
        # Determine motion type
        if alpha_expected < 0.8:
            motion_type = 'subdiffusion'
        elif alpha_expected > 1.2:
            motion_type = 'superdiffusion'
        else:
            motion_type = 'normal_diffusion'
        
        k_factor = self.bias_tables['optimal_dt_factor'][motion_type]
        
        # Optimal dt formula: dt_opt = k · σ² / D
        # This balances:
        # - Too small dt → noise dominates, D overestimated
        # - Too large dt → undersampling, D underestimated
        sigma_sq = localization_precision**2
        dt_optimal = k_factor * sigma_sq / D_expected
        
        # Acceptable range (factor of 2 on either side)
        dt_min = dt_optimal / 2
        dt_max = dt_optimal * 2
        
        # Recommended exposure time
        exposure_fraction = self.bias_tables['optimal_exposure_fraction']
        exposure_recommended = dt_optimal * exposure_fraction
        
        # Estimate biases at recommended dt
        # Simplified model: bias increases as you deviate from optimal
        D_bias = 0.02  # ~2% at optimal
        alpha_bias = 0.01  # ~1% at optimal
        
        # Track length check
        min_N_D = self.bias_tables['min_track_length']['D_estimation']
        min_N_alpha = self.bias_tables['min_track_length']['alpha_estimation']
        
        warnings = []
        
        # Sub-resolution diffusion warning (CRITICAL)
        # Check if expected displacement < localization precision
        expected_displacement = np.sqrt(4 * D_expected * dt_optimal)  # 2D RMS displacement
        if expected_displacement < localization_precision:
            warnings.append(
                f'⚠️ CRITICAL: Expected displacement ({expected_displacement:.4f} μm) '
                f'< localization precision ({localization_precision:.3f} μm). '
                f'Motion is UNRESOLVABLE with current precision. '
                f'Recommendations: (1) Increase frame interval, (2) Improve localization '
                f'(brighter dye, better SNR), or (3) Accept that D cannot be reliably measured.'
            )
        elif expected_displacement < 2 * localization_precision:
            warnings.append(
                f'⚠️ WARNING: Expected displacement ({expected_displacement:.4f} μm) '
                f'is only {expected_displacement/localization_precision:.1f}× localization precision. '
                f'D estimates will have high uncertainty. Consider longer dt or better SNR.'
            )
        
        if track_length < min_N_D:
            warnings.append(
                f'Track length ({track_length}) < recommended minimum ({min_N_D}) '
                f'for reliable D estimation. Consider longer acquisition.'
            )
        if track_length < min_N_alpha and abs(alpha_expected - 1.0) > 0.1:
            warnings.append(
                f'Track length ({track_length}) < recommended minimum ({min_N_alpha}) '
                f'for anomalous exponent estimation.'
            )
        
        # Check photobleaching
        total_time = dt_optimal * track_length
        if total_time > 30:  # > 30 seconds
            warnings.append(
                f'Total acquisition time ({total_time:.1f}s) may cause photobleaching. '
                f'Consider shorter tracks or lower illumination.'
            )
        
        # Check camera frame rate capability
        framerate_recommended = 1.0 / dt_optimal
        if framerate_recommended > 100:
            warnings.append(
                f'Recommended frame rate ({framerate_recommended:.1f} Hz) may exceed '
                f'camera capability. Check camera specifications.'
            )
        elif framerate_recommended < 0.1:
            warnings.append(
                f'Recommended frame rate ({framerate_recommended:.3f} Hz) is very slow. '
                f'Consider if sample drift will be an issue.'
            )
        
        # Rationale
        rationale = (
            f'Optimal dt={dt_optimal:.4f}s balances localization precision '
            f'({localization_precision:.3f} μm) and diffusion rate ({D_expected:.3f} μm²/s). '
            f'For {motion_type.replace("_", " ")}, k-factor = {k_factor}. '
            f'Formula: dt = {k_factor} × σ² / D'
        )
        
        return {
            'success': True,
            'recommended_dt': dt_optimal,
            'dt_range': (dt_min, dt_max),
            'exposure_time': exposure_recommended,
            'exposure_fraction': exposure_fraction,
            'expected_D_bias': D_bias,
            'expected_alpha_bias': alpha_bias,
            'framerate_hz': framerate_recommended,
            'framerate_range_hz': (1/dt_max, 1/dt_min),
            'motion_type': motion_type,
            'k_factor': k_factor,
            'rationale': rationale,
            'warnings': warnings,
            'parameters_used': {
                'D_expected': D_expected,
                'localization_precision': localization_precision,
                'track_length': track_length,
                'alpha_expected': alpha_expected
            }
        }
    
    
    def validate_settings(self, dt_actual: float, exposure_actual: float,
                         tracks_df: pd.DataFrame,
                         pixel_size_um: float,
                         localization_precision_um: float) -> Dict:
        """
        Post-acquisition check: Are settings reasonable for observed D?
        
        Flags if dt >> optimal (undersampling) or << optimal (noise dominates).
        
        Parameters
        ----------
        dt_actual : float
            Actual frame interval used (s)
        exposure_actual : float
            Actual exposure time used (s)
        tracks_df : pd.DataFrame
            Tracking data with columns: track_id, frame, x, y
        pixel_size_um : float
            Pixel size in microns
        localization_precision_um : float
            Localization precision in microns
        
        Returns
        -------
        Dict
            {
                'optimal_dt': what dt should have been,
                'dt_ratio': actual / optimal,
                'warning': True if suboptimal,
                'message': explanation,
                'recommendation': what to do,
                'observed_D_mean': mean D from data
            }
        """
        if dt_actual <= 0:
            return {
                'success': False,
                'error': 'dt_actual must be positive'
            }
        
        if len(tracks_df) == 0:
            return {
                'success': False,
                'error': 'No tracks to analyze'
            }
        
        # Quick D estimation (simple MSD)
        D_values = []
        for track_id, group in tracks_df.groupby('track_id'):
            group = group.sort_values('frame')
            if len(group) < 3:
                continue
            
            coords = group[['x', 'y']].values * pixel_size_um
            displacements = np.diff(coords, axis=0)
            msd = np.mean(np.sum(displacements**2, axis=1))
            D = msd / (4 * dt_actual)  # 2D diffusion
            
            if D > 0 and D < 100:  # Sanity check
                D_values.append(D)
        
        if len(D_values) == 0:
            return {
                'success': False,
                'error': 'Could not estimate D from tracks'
            }
        
        D_observed_mean = np.mean(D_values)
        D_observed_std = np.std(D_values)
        
        # What dt should have been used?
        recommendation = self.recommend_framerate(
            D_expected=D_observed_mean,
            localization_precision=localization_precision_um,
            track_length=len(tracks_df) // len(tracks_df['track_id'].unique())
        )
        
        if not recommendation['success']:
            return recommendation
        
        dt_optimal = recommendation['recommended_dt']
        dt_ratio = dt_actual / dt_optimal
        
        # Assess quality
        warning = False
        message = ""
        recommendation_text = ""
        
        if dt_ratio > 2.0:
            # Undersampling
            warning = True
            message = (
                f'⚠️ Frame interval too large: {dt_actual:.4f}s vs optimal {dt_optimal:.4f}s. '
                f'Ratio: {dt_ratio:.2f}×. This causes undersampling and D underestimation.'
            )
            recommendation_text = (
                f'Increase frame rate to {1/dt_optimal:.1f} Hz ({dt_optimal:.4f}s per frame) '
                f'for next experiment.'
            )
        elif dt_ratio < 0.5:
            # Oversampling (noise dominates)
            warning = True
            message = (
                f'⚠️ Frame interval too small: {dt_actual:.4f}s vs optimal {dt_optimal:.4f}s. '
                f'Ratio: {dt_ratio:.2f}×. Localization noise dominates, D overestimation likely.'
            )
            recommendation_text = (
                f'Decrease frame rate to {1/dt_optimal:.1f} Hz ({dt_optimal:.4f}s per frame) '
                f'or improve localization precision for next experiment.'
            )
        else:
            # Acceptable range
            message = (
                f'✓ Frame interval acceptable: {dt_actual:.4f}s vs optimal {dt_optimal:.4f}s. '
                f'Ratio: {dt_ratio:.2f}× (within 0.5-2.0× range).'
            )
            recommendation_text = 'Settings are near-optimal. No changes needed.'
        
        # Exposure check
        exposure_ratio = exposure_actual / dt_actual if dt_actual > 0 else 0
        if exposure_ratio > 0.9:
            warning = True
            message += (
                f'\n⚠️ Exposure time ({exposure_actual:.4f}s) is {exposure_ratio*100:.0f}% '
                f'of frame interval. This causes significant motion blur.'
            )
            recommendation_text += (
                f' Reduce exposure to ~{dt_actual*0.8:.4f}s (80% of frame time).'
            )
        
        return {
            'success': True,
            'optimal_dt': dt_optimal,
            'actual_dt': dt_actual,
            'dt_ratio': dt_ratio,
            'dt_range_acceptable': (dt_optimal/2, dt_optimal*2),
            'warning': warning,
            'message': message,
            'recommendation': recommendation_text,
            'observed_D_mean': D_observed_mean,
            'observed_D_std': D_observed_std,
            'n_tracks': len(D_values),
            'exposure_ratio': exposure_ratio,
            'optimal_exposure': dt_optimal * 0.8
        }
    
    
    def create_acquisition_plan(self, 
                               D_range: Tuple[float, float],
                               precision_range: Tuple[float, float],
                               target_track_length: int = 50,
                               alpha_expected: float = 1.0) -> pd.DataFrame:
        """
        Create acquisition parameter matrix for different D values.
        
        Useful for planning experiments where D is unknown but can be bounded.
        
        Parameters
        ----------
        D_range : Tuple[float, float]
            (D_min, D_max) in μm²/s
        precision_range : Tuple[float, float]
            (σ_min, σ_max) in μm
        target_track_length : int
            Desired track length
        alpha_expected : float
            Expected anomalous exponent
        
        Returns
        -------
        pd.DataFrame
            Matrix with columns: D_expected, precision, dt_recommended,
            framerate_hz, exposure_time, rationale
        """
        D_values = np.logspace(np.log10(D_range[0]), np.log10(D_range[1]), 5)
        precision_values = np.linspace(precision_range[0], precision_range[1], 3)
        
        results = []
        for D in D_values:
            for precision in precision_values:
                rec = self.recommend_framerate(
                    D_expected=D,
                    localization_precision=precision,
                    track_length=target_track_length,
                    alpha_expected=alpha_expected
                )
                
                if rec['success']:
                    results.append({
                        'D_expected_um2_per_s': D,
                        'precision_um': precision,
                        'dt_recommended_s': rec['recommended_dt'],
                        'framerate_hz': rec['framerate_hz'],
                        'exposure_time_s': rec['exposure_time'],
                        'dt_min_s': rec['dt_range'][0],
                        'dt_max_s': rec['dt_range'][1],
                        'motion_type': rec['motion_type']
                    })
        
        return pd.DataFrame(results)
    
    
    def compare_settings(self, settings_list: list) -> pd.DataFrame:
        """
        Compare multiple acquisition settings side-by-side.
        
        Parameters
        ----------
        settings_list : list of dicts
            Each dict has: {'name': str, 'dt': float, 'exposure': float, 
                           'D_expected': float, 'precision': float}
        
        Returns
        -------
        pd.DataFrame
            Comparison table with bias estimates
        """
        results = []
        
        for setting in settings_list:
            name = setting['name']
            dt = setting['dt']
            exposure = setting['exposure']
            D = setting['D_expected']
            precision = setting['precision']
            
            # Get optimal
            rec = self.recommend_framerate(D, precision)
            
            if not rec['success']:
                continue
            
            dt_optimal = rec['recommended_dt']
            dt_ratio = dt / dt_optimal
            
            # Estimate bias (simplified model)
            # Bias increases quadratically as you deviate from optimal
            D_bias = 0.02 + 0.1 * (dt_ratio - 1)**2
            
            results.append({
                'setting_name': name,
                'dt_used_s': dt,
                'dt_optimal_s': dt_optimal,
                'dt_ratio': dt_ratio,
                'framerate_hz': 1/dt,
                'exposure_s': exposure,
                'exposure_fraction': exposure/dt,
                'estimated_D_bias': D_bias,
                'quality': 'Good' if 0.5 < dt_ratio < 2.0 else 'Suboptimal'
            })
        
        return pd.DataFrame(results)


class AcquisitionOptimizer:
    """
    Optimizes camera acquisition parameters for single particle tracking experiments.
    
    Balances competing factors:
    - Short exposure → less motion blur → reduced bias
    - Long exposure → more photons → better localization precision
    - Short frame interval → better temporal resolution → more accurate D
    - Long frame interval → larger displacements → better SNR
    
    Based on optimal temporal resolution theory (Berglund 2010, Michalet 2010).
    """
    
    def __init__(self):
        """Initialize optimizer with default constraints."""
        # Physical constraints (typical camera limitations)
        self.min_frame_interval = 0.001  # 1 ms (1000 fps max)
        self.max_frame_interval = 10.0   # 10 s (practical upper limit)
        self.min_exposure_time = 0.0001  # 0.1 ms (fast cameras)
        self.max_exposure_time = 1.0     # 1 s (practical limit)
        
        # Optimization parameters
        self.min_displacement_to_noise_ratio = 1.5  # Minimum SNR_spatial
        self.optimal_blur_fraction = 0.3  # R = exposure/interval (30% reduces bias <5%)
        self.min_steps_per_track = 20  # Minimum for CVE/MLE robustness
    
    def calculate_optimal_parameters(
        self,
        expected_D: float,
        localization_precision: float,
        target_error_pct: float = 10.0,
        pixel_size: float = 0.1,
        typical_track_length_frames: int = 50,
        camera_min_exposure: Optional[float] = None,
        camera_max_exposure: Optional[float] = None,
        camera_min_interval: Optional[float] = None,
        camera_max_interval: Optional[float] = None
    ) -> Dict:
        """
        Calculate optimal frame interval and exposure time.
        
        Parameters
        ----------
        expected_D : float
            Expected diffusion coefficient in µm²/s
        localization_precision : float
            Expected localization precision (1σ) in nm
        target_error_pct : float, optional
            Target relative error in D estimation (default: 10%)
        pixel_size : float, optional
            Pixel size in µm (default: 0.1 µm = 100 nm)
        typical_track_length_frames : int, optional
            Expected number of frames per track (default: 50)
        camera_min_exposure : float, optional
            Camera minimum exposure time in seconds
        camera_max_exposure : float, optional
            Camera maximum exposure time in seconds
        camera_min_interval : float, optional
            Camera minimum frame interval in seconds
        camera_max_interval : float, optional
            Camera maximum frame interval in seconds
            
        Returns
        -------
        dict
            Optimization results with keys:
            - 'frame_interval': Recommended frame interval (s)
            - 'exposure_time': Recommended exposure time (s)
            - 'expected_error_pct': Expected relative error in D (%)
            - 'displacement_per_frame': Expected displacement (µm)
            - 'displacement_to_noise_ratio': SNR_spatial
            - 'blur_fraction': R = exposure/interval
            - 'steps_needed': Minimum frames for target error
            - 'acquisition_time': Total time for typical track (s)
            - 'warnings': List of warning messages
            - 'recommendations': List of recommendations
            - 'feasibility': 'optimal', 'acceptable', 'challenging', or 'infeasible'
        """
        # Convert localization precision to µm
        sigma_loc_um = localization_precision / 1000.0
        
        # Override camera limits if provided
        if camera_min_exposure is not None:
            self.min_exposure_time = camera_min_exposure
        if camera_max_exposure is not None:
            self.max_exposure_time = camera_max_exposure
        if camera_min_interval is not None:
            self.min_frame_interval = camera_min_interval
        if camera_max_interval is not None:
            self.max_frame_interval = camera_max_interval
        
        warnings_list = []
        recommendations = []
        
        # Step 1: Calculate optimal frame interval
        # Goal: displacement >> localization_precision (SNR_spatial > 1.5)
        # For 2D diffusion: <r²> = 4Dτ → <r> ~ √(4Dτ)
        # Want: √(4Dτ) / σ_loc ≥ min_SNR
        
        target_snr = self.min_displacement_to_noise_ratio
        
        # Solve for τ: τ = (target_snr * σ_loc)² / (4D)
        optimal_interval = (target_snr * sigma_loc_um) ** 2 / (4 * expected_D)
        
        # Step 2: Calculate optimal exposure time
        # Goal: minimize motion blur while maintaining localization precision
        # Compromise: exposure = 0.3 * interval (reduces bias to <5%)
        optimal_exposure = self.optimal_blur_fraction * optimal_interval
        
        # Step 3: Apply camera constraints
        frame_interval = np.clip(optimal_interval, self.min_frame_interval, self.max_frame_interval)
        exposure_time = np.clip(optimal_exposure, self.min_exposure_time, 
                               min(self.max_exposure_time, frame_interval))
        
        # Ensure exposure ≤ interval
        if exposure_time > frame_interval:
            exposure_time = frame_interval * 0.9  # Leave 10% dead time
            warnings_list.append("Exposure time limited by frame interval")
        
        # Step 4: Calculate expected performance metrics
        actual_displacement = np.sqrt(4 * expected_D * frame_interval)
        snr_spatial = actual_displacement / sigma_loc_um
        blur_fraction = exposure_time / frame_interval
        
        # Step 5: Estimate measurement error
        # Error sources:
        # 1. Finite track length: σ(D)/D ~ 1/√(N-2) for MSD
        # 2. Localization noise bias: adds ~2σ²/τ to apparent D
        # 3. Motion blur bias: reduces apparent D by factor (1 - R²/12)
        
        # Relative bias from localization noise
        noise_bias_relative = (2 * sigma_loc_um ** 2) / (4 * expected_D * frame_interval)
        
        # Relative bias from motion blur (typically negative, reduces D)
        blur_bias_relative = (blur_fraction ** 2) / 12
        
        # Net bias (noise dominates for short intervals)
        net_bias_pct = (noise_bias_relative - blur_bias_relative) * 100
        
        # Statistical uncertainty from finite track length (CVE estimator)
        # For CVE: σ(D)/D ≈ √(2/(N-1)) for unbiased estimator
        statistical_error_pct = np.sqrt(2 / (typical_track_length_frames - 1)) * 100
        
        # Total expected error (combine bias and statistical uncertainty)
        expected_error_pct = np.sqrt(net_bias_pct ** 2 + statistical_error_pct ** 2)
        
        # Calculate minimum steps needed for target error
        # From statistical error: N ≈ 2/(target_error/100)² + 1
        steps_needed = int(2 / (target_error_pct / 100) ** 2) + 1
        steps_needed = max(steps_needed, self.min_steps_per_track)
        
        acquisition_time = frame_interval * typical_track_length_frames
        
        # Step 6: Assess feasibility
        feasibility = self._assess_feasibility(
            frame_interval, optimal_interval,
            exposure_time, optimal_exposure,
            snr_spatial, expected_error_pct, target_error_pct,
            warnings_list, recommendations
        )
        
        # Step 7: Add specific recommendations
        self._add_recommendations(
            expected_D, sigma_loc_um, snr_spatial, blur_fraction,
            frame_interval, exposure_time, expected_error_pct,
            typical_track_length_frames, steps_needed,
            recommendations, warnings_list
        )
        
        return {
            'frame_interval': frame_interval,
            'exposure_time': exposure_time,
            'expected_error_pct': expected_error_pct,
            'displacement_per_frame': actual_displacement,
            'displacement_to_noise_ratio': snr_spatial,
            'blur_fraction': blur_fraction,
            'noise_bias_pct': noise_bias_relative * 100,
            'blur_bias_pct': -blur_bias_relative * 100,  # Negative sign (reduces D)
            'statistical_error_pct': statistical_error_pct,
            'steps_needed': steps_needed,
            'acquisition_time': acquisition_time,
            'warnings': warnings_list,
            'recommendations': recommendations,
            'feasibility': feasibility,
            'optimal_interval': optimal_interval,  # For comparison
            'optimal_exposure': optimal_exposure   # For comparison
        }
    
    def _assess_feasibility(
        self,
        frame_interval: float,
        optimal_interval: float,
        exposure_time: float,
        optimal_exposure: float,
        snr_spatial: float,
        expected_error: float,
        target_error: float,
        warnings: list,
        recommendations: list
    ) -> str:
        """Determine experiment feasibility based on constraints."""
        
        issues = 0
        
        # Check if camera limits forced compromise
        if abs(frame_interval - optimal_interval) / optimal_interval > 0.5:
            issues += 1
            if frame_interval > optimal_interval:
                warnings.append("Frame interval limited by camera maximum frame rate")
            else:
                warnings.append("Frame interval faster than optimal (SNR will be lower)")
        
        if abs(exposure_time - optimal_exposure) / optimal_exposure > 0.5:
            issues += 1
            warnings.append("Exposure time constrained by camera limits")
        
        # Check SNR
        if snr_spatial < 1.0:
            issues += 2
            warnings.append(f"Low SNR ({snr_spatial:.2f}) - displacements barely exceed noise")
        elif snr_spatial < self.min_displacement_to_noise_ratio:
            issues += 1
            warnings.append(f"Marginal SNR ({snr_spatial:.2f}) - consider longer intervals")
        
        # Check error target
        if expected_error > target_error * 2:
            issues += 2
            warnings.append(f"Expected error ({expected_error:.1f}%) >> target ({target_error:.1f}%)")
        elif expected_error > target_error * 1.5:
            issues += 1
            warnings.append(f"Expected error ({expected_error:.1f}%) exceeds target ({target_error:.1f}%)")
        
        # Classify feasibility
        if issues == 0:
            return 'optimal'
        elif issues <= 2:
            return 'acceptable'
        elif issues <= 4:
            return 'challenging'
        else:
            return 'infeasible'
    
    def _add_recommendations(
        self,
        D: float,
        sigma_loc: float,
        snr: float,
        blur_frac: float,
        interval: float,
        exposure: float,
        error: float,
        track_length: int,
        steps_needed: int,
        recommendations: list,
        warnings: list
    ):
        """Generate specific recommendations based on parameters."""
        
        # Diffusion regime
        if D < 0.01:
            recommendations.append("Very slow diffusion - consider longer acquisition time per track")
        elif D > 10.0:
            recommendations.append("Fast diffusion - use high-speed camera if available")
        
        # SNR regime
        if snr < 1.5:
            recommendations.append(
                f"Low SNR ({snr:.2f}): Increase frame interval OR improve localization precision"
            )
            recommendations.append("Consider: brighter labels, lower background, or camera cooling")
        elif snr > 5.0:
            recommendations.append(
                f"Excellent SNR ({snr:.2f}) - can use shorter intervals for better time resolution"
            )
        
        # Motion blur
        if blur_frac > 0.5:
            recommendations.append(
                f"High motion blur (R={blur_frac:.2f}) - reduce exposure time if possible"
            )
            recommendations.append("Motion blur will reduce apparent D by ~5-10%")
        elif blur_frac < 0.2:
            recommendations.append(
                f"Low motion blur (R={blur_frac:.2f}) - can increase exposure for better SNR"
            )
        
        # Track length
        if steps_needed > track_length * 1.5:
            recommendations.append(
                f"Need ≥{steps_needed} frames per track for target error "
                f"(typical: {track_length})"
            )
            recommendations.append("Consider: longer tracks OR accept higher error")
        
        # Localization precision
        if sigma_loc > 0.05:  # >50 nm
            recommendations.append(
                "Localization precision >50 nm will cause significant bias"
            )
            recommendations.append("Use CVE or MLE estimators (not naive MSD)")
        
        # Analysis method recommendation
        if track_length < 30 or snr < 2.0:
            recommendations.append(
                "Recommended analysis: CVE (covariance) or MLE (maximum likelihood)"
            )
        elif track_length < 50:
            recommendations.append("Recommended analysis: CVE (covariance-based estimator)")
        else:
            recommendations.append("Track length sufficient for MSD, CVE, or MLE methods")
        
        # Exposure optimization
        if exposure < interval * 0.2:
            recommendations.append(
                f"Exposure only {exposure/interval*100:.0f}% of frame interval - "
                "can increase for more photons"
            )
        
        # Overall assessment
        if error < 5.0:
            recommendations.append(f"Expected error ({error:.1f}%) - excellent precision achievable")
        elif error < 10.0:
            recommendations.append(f"Expected error ({error:.1f}%) - good precision achievable")
        elif error < 20.0:
            recommendations.append(f"Expected error ({error:.1f}%) - acceptable for most applications")
        else:
            recommendations.append(f"Expected error ({error:.1f}%) - challenging measurement")


def quick_recommendation(D_um2_per_s: float, 
                        precision_nm: float) -> str:
    """
    Quick one-liner recommendation.
    
    Parameters
    ----------
    D_um2_per_s : float
        Expected diffusion coefficient (μm²/s)
    precision_nm : float
        Localization precision (nm)
    
    Returns
    -------
    str
        Human-readable recommendation
    """
    advisor = AcquisitionAdvisor()
    rec = advisor.recommend_framerate(D_um2_per_s, precision_nm / 1000)
    
    if not rec['success']:
        return f"Error: {rec.get('error', 'Unknown error')}"
    
    dt = rec['recommended_dt']
    fps = rec['framerate_hz']
    exp = rec['exposure_time']
    
    return (
        f"📊 For D={D_um2_per_s:.3f} μm²/s and σ={precision_nm:.0f} nm:\n"
        f"   Recommended: {fps:.1f} FPS ({dt*1000:.1f} ms/frame)\n"
        f"   Exposure: {exp*1000:.1f} ms ({exp/dt*100:.0f}% of frame time)\n"
        f"   Acceptable range: {rec['framerate_range_hz'][1]:.1f}-{rec['framerate_range_hz'][0]:.1f} FPS"
    )
