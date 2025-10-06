"""
Microrheology Analysis Module

Calculates G' (storage modulus), G" (loss modulus), and effective viscosity
from single particle tracking data using dual frame rate measurements.

This module implements microrheology principles to extract mechanical
properties of the cellular environment from particle motion.
"""

from typing import Dict, Tuple, Optional, List
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import gamma
from analysis import calculate_msd

# Added imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

class MicrorheologyAnalyzer:
    """
    Advanced microrheology analysis using dual frame rate tracking data.
    
    Calculates viscoelastic moduli (G', G") and effective viscosity from
    mean squared displacement data at different time scales.
    """
    
    def __init__(self, particle_radius_m: float, temperature_K: float = 300.0):
        """
        Initialize the microrheology analyzer.
        
        Parameters
        ----------
        particle_radius_m : float
            Radius of tracked particles in meters
        temperature_K : float
            Temperature in Kelvin (default: 300K = 27°C)
        """
        self.particle_radius_m = particle_radius_m
        self.temperature_K = temperature_K
        self.kB = 1.380649e-23  # Boltzmann constant in J/K

    def calculate_complex_modulus_gser(self, msd_df: pd.DataFrame,
                                       omega_rad_s: float) -> Tuple[float, float]:
        """
        Calculate complex modulus G*(ω) using the GSER (Generalized Stokes-Einstein Relation) approach.
        
        The GSER relates the complex modulus to the Fourier transform of the MSD:
        G*(ω) = (kB*T) / (3*π*a*s*Γ(1+α)) * (iω)^α
        
        Where α is the local logarithmic slope of MSD at time τ = 1/ω
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data with columns: lag_time_s, msd_m2
        omega_rad_s : float
            Angular frequency in rad/s
            
        Returns
        -------
        Tuple[float, float]
            G' (storage modulus) and G" (loss modulus) in Pa
        """
        if msd_df is None or len(msd_df) < 3 or omega_rad_s <= 0:
            return np.nan, np.nan

        # Calculate characteristic time τ = 1/ω
        tau = 1.0 / omega_rad_s
        
        # Find the closest time point and estimate local slope α
        if tau < msd_df['lag_time_s'].min() or tau > msd_df['lag_time_s'].max():
            return np.nan, np.nan
            
        # Interpolate MSD at τ and estimate local slope
        log_times = np.log(msd_df['lag_time_s'])
        log_msd = np.log(msd_df['msd_m2'])
        
        # Find closest point to tau
        idx = np.argmin(np.abs(msd_df['lag_time_s'] - tau))
        
        # Estimate local slope α using neighboring points
        if idx == 0:
            alpha = (log_msd[idx + 1] - log_msd[idx]) / (log_times[idx + 1] - log_times[idx])
        elif idx == len(msd_df) - 1:
            alpha = (log_msd[idx] - log_msd[idx - 1]) / (log_times[idx] - log_times[idx - 1])
        else:
            # Use symmetric difference for better accuracy
            alpha = (log_msd[idx + 1] - log_msd[idx - 1]) / (log_times[idx + 1] - log_times[idx - 1])
        
        # Ensure alpha is physically reasonable (0 < α < 2)
        alpha = np.clip(alpha, 0.01, 1.99)
        
        # Interpolate MSD at tau
        msd_at_tau = np.interp(tau, msd_df['lag_time_s'], msd_df['msd_m2'])
        
        # Calculate the prefactor using corrected GSER formula
        # G*(ω) = (kB*T) / (3*π*a*<Δr²(τ)>) * Γ(1+α) * (iω*τ)^α
        prefactor = (self.kB * self.temperature_K) / (3 * np.pi * self.particle_radius_m * msd_at_tau)
        
        # Gamma function factor
        gamma_factor = gamma(1 + alpha)
        
        # Complex frequency factor (iω*τ)^α = (ωτ)^α * e^(iα*π/2)
        omega_tau_alpha = (omega_rad_s * tau) ** alpha
        phase = alpha * np.pi / 2
        
        # Calculate G' and G"
        g_prime = prefactor * gamma_factor * omega_tau_alpha * np.cos(phase)
        g_double_prime = prefactor * gamma_factor * omega_tau_alpha * np.sin(phase)
        
        return g_prime, g_double_prime

    def calculate_effective_viscosity(self, msd_df: pd.DataFrame, 
                                      lag_time_range_s: Tuple[float, float] = None) -> float:
        """
        Calculate effective viscosity from MSD slope using Stokes-Einstein relation.
        
        For 2D projected motion: η = kB*T / (4*π*D*a)
        Where D is the diffusion coefficient from MSD slope: D = slope/4 (2D)
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data
        lag_time_range_s : Tuple[float, float], optional
            Time range for slope calculation. If None, uses initial linear region.
            
        Returns
        -------
        float
            Effective viscosity in Pa·s
        """
        if len(msd_df) < 2:
            return np.nan
        
        # Filter data for slope calculation
        if lag_time_range_s:
            mask = (msd_df['lag_time_s'] >= lag_time_range_s[0]) & \
                   (msd_df['lag_time_s'] <= lag_time_range_s[1])
            slope_data = msd_df[mask]
        else:
            # Use initial linear region (first 20% of data or first 5 points)
            n_points = min(max(2, len(msd_df) // 5), 5)
            slope_data = msd_df.head(n_points)
        
        if len(slope_data) < 2:
            return np.nan
        
        # Linear fit to get slope in log-log space to check for pure diffusion
        try:
            # First check if behavior is diffusive (slope ≈ 1 in log-log)
            log_slope = np.polyfit(np.log(slope_data['lag_time_s']), 
                                 np.log(slope_data['msd_m2']), 1)[0]
            
            # If close to diffusive behavior, use linear fit
            if 0.8 <= log_slope <= 1.2:
                slope, _ = np.polyfit(slope_data['lag_time_s'], slope_data['msd_m2'], 1)
            else:
                # For non-diffusive behavior, use power law fit at short times
                # MSD = 4*D*t^α, so D_eff = MSD(t)/(4*t) at short times
                t_ref = float(slope_data['lag_time_s'].iloc[0])
                msd_ref = float(slope_data['msd_m2'].iloc[0])
                slope = msd_ref / max(t_ref, 1e-12)
        except Exception:
            return np.nan
        
        if slope <= 0:
            return np.inf
        
        # Calculate effective diffusion coefficient
        # For 2D projected motion: D = slope/4
        D_eff = slope / 4.0
        
        # Calculate viscosity using Stokes-Einstein relation
        # For 2D: η = kB*T / (4*π*D*a)
        # For 3D: η = kB*T / (6*π*D*a)
        # Use 2D formula since we're analyzing projected motion
        viscosity_eff = (self.kB * self.temperature_K) / (4 * np.pi * D_eff * self.particle_radius_m)
        
        return viscosity_eff
    
    def calculate_frequency_dependent_viscosity(self, msd_df: pd.DataFrame, 
                                              omega_rad_s: float) -> float:
        """
        Calculate frequency-dependent viscosity using the complex modulus.
        
        η*(ω) = G*(ω) / (iω)
        |η*(ω)| = |G*(ω)| / ω
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data
        omega_rad_s : float
            Angular frequency in rad/s
            
        Returns
        -------
        float
            Frequency-dependent viscosity magnitude in Pa·s
        """
        g_prime, g_double_prime = self.calculate_complex_modulus_gser(msd_df, omega_rad_s)
        
        if np.isnan(g_prime) or np.isnan(g_double_prime):
            return np.nan
        
        # Calculate complex viscosity magnitude
        g_star_magnitude = np.sqrt(g_prime**2 + g_double_prime**2)
        eta_star_magnitude = g_star_magnitude / omega_rad_s
        
        return eta_star_magnitude
    
    def fit_power_law_msd(self, msd_df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit power law to MSD data: MSD(t) = A * t^α
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data
            
        Returns
        -------
        Dict[str, float]
            Dictionary with 'amplitude', 'exponent', and 'r_squared'
        """
        if len(msd_df) < 3:
            return {'amplitude': np.nan, 'exponent': np.nan, 'r_squared': np.nan}
        
        try:
            # Fit in log space: log(MSD) = log(A) + α*log(t)
            log_times = np.log(msd_df['lag_time_s'])
            log_msd = np.log(msd_df['msd_m2'])
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_times) & np.isfinite(log_msd)
            if np.sum(valid_mask) < 3:
                return {'amplitude': np.nan, 'exponent': np.nan, 'r_squared': np.nan}
            
            log_times = log_times[valid_mask]
            log_msd = log_msd[valid_mask]
            
            # Linear fit
            coeffs = np.polyfit(log_times, log_msd, 1)
            alpha = coeffs[0]  # Exponent
            log_A = coeffs[1]  # Log amplitude
            A = np.exp(log_A)  # Amplitude
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, log_times)
            ss_res = np.sum((log_msd - y_pred)**2)
            ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'amplitude': A,
                'exponent': alpha,
                'r_squared': r_squared
            }
            
        except:
            return {'amplitude': np.nan, 'exponent': np.nan, 'r_squared': np.nan}
    
    def multi_dataset_analysis(self, track_datasets: List[pd.DataFrame], 
                              frame_intervals_s: List[float],
                              pixel_size_um: float,
                              omega_ranges: List[List[float]] = None) -> Dict:
        """
        Perform complete microrheology analysis using multiple datasets with different sampling rates.
        
        Parameters
        ----------
        track_datasets : List[pd.DataFrame]
            List of tracking data from the same sample at different sampling rates
        frame_intervals_s : List[float]
            Frame intervals for each dataset in seconds
        pixel_size_um : float
            Pixel size in micrometers
        omega_ranges : List[List[float]], optional
            Angular frequencies (rad/s) to calculate for each dataset.
            If None, automatically determined from frame intervals.
            
        Returns
        -------
        Dict
            Complete microrheology analysis results for all datasets
        """
        if len(track_datasets) != len(frame_intervals_s):
            return {
                'success': False,
                'error': 'Number of datasets must match number of frame intervals'
            }
        
        if len(track_datasets) == 0:
            return {
                'success': False,
                'error': 'No datasets provided'
            }
        
        results = {
            'success': False,
            'error': None,
            'datasets': [],
            'combined_frequency_response': {},
            'dataset_comparison': {}
        }
        
        try:
            all_frequencies_hz, all_gp, all_gpp, all_eta = [], [], [], []

            for i, (tracks_df, dt) in enumerate(zip(track_datasets, frame_intervals_s)):
                dataset_label = f"Dataset_{i+1}_{dt:.3f}s"
                msd_data = calculate_msd(tracks_df, pixel_size=pixel_size_um, frame_interval=dt)
                if msd_data is None or msd_data.empty:
                    continue
                msd_data = msd_data.rename(columns={'lag_time': 'lag_time_s', 'msd': 'msd_m2'})
                msd_data['msd_m2'] = msd_data['msd_m2'] * 1e-12

                fit = self.fit_power_law_msd(msd_data)

                if omega_ranges and i < len(omega_ranges) and omega_ranges[i]:
                    omega_list = np.asarray(omega_ranges[i], dtype=float)
                else:
                    # frequency bounds from MSD time window
                    tmin = float(msd_data['lag_time_s'].min())
                    tmax = float(msd_data['lag_time_s'].max())
                    if tmin <= 0 or tmax <= 0 or tmax <= tmin:
                        omega_list = np.logspace(-1, 2, 20)
                    else:
                        omega_list = np.logspace(
                            np.log10(2*np.pi/(tmax*0.5)),
                            np.log10(2*np.pi/(tmin*2.0)),
                            20
                        )

                gp_list, gpp_list, eta_list, f_hz = [], [], [], []
                for omega in omega_list:
                    gp, gpp = self.calculate_complex_modulus_gser(msd_data, omega)
                    eta = self.calculate_frequency_dependent_viscosity(msd_data, omega)
                    if not (np.isnan(gp) or np.isnan(gpp) or np.isnan(eta)):
                        gp_list.append(gp)
                        gpp_list.append(gpp)
                        eta_list.append(eta)
                        f_hz.append(omega/(2*np.pi))

                eff_eta = self.calculate_effective_viscosity(msd_data)

                ds = {
                    'label': dataset_label,
                    'frame_interval_s': dt,
                    'msd_data': msd_data,
                    'power_law_fit': fit,
                    'frequencies_hz': f_hz,
                    'g_prime_pa': gp_list,
                    'g_double_prime_pa': gpp_list,
                    'frequency_dependent_viscosity_pa_s': eta_list,
                    'effective_viscosity_pa_s': eff_eta,
                    'omega_range_rad_s': omega_list.tolist()
                }
                # stats for dataset
                if gp_list:
                    ds['g_prime_mean_pa'] = float(np.mean(gp_list))
                    ds['g_double_prime_mean_pa'] = float(np.mean(gpp_list))
                    ds['loss_tangent'] = (float(np.mean(gpp_list)) / float(np.mean(gp_list))) if np.mean(gp_list) > 0 else np.inf

                results['datasets'].append(ds)
                all_frequencies_hz.extend(f_hz)
                all_gp.extend(gp_list)
                all_gpp.extend(gpp_list)
                all_eta.extend(eta_list)

            if all_frequencies_hz:
                order = np.argsort(all_frequencies_hz)
                results['combined_frequency_response'] = {
                    'frequencies_hz': [all_frequencies_hz[i] for i in order],
                    'g_prime_pa': [all_gp[i] for i in order],
                    'g_double_prime_pa': [all_gpp[i] for i in order],
                    'viscosity_pa_s': [all_eta[i] for i in order],
                    'g_prime_overall_mean_pa': float(np.mean(all_gp)) if all_gp else np.nan,
                    'g_double_prime_overall_mean_pa': float(np.mean(all_gpp)) if all_gpp else np.nan,
                    'viscosity_overall_mean_pa_s': float(np.mean(all_eta)) if all_eta else np.nan,
                    'frequency_range_hz': [float(min(all_frequencies_hz)), float(max(all_frequencies_hz))]
                }

            if len(results['datasets']) > 1:
                visc = [ds.get('effective_viscosity_pa_s', np.nan) for ds in results['datasets']]
                visc = [v for v in visc if np.isfinite(v)]
                if len(visc) > 1:
                    mean_v = float(np.mean(visc))
                    std_v = float(np.std(visc))
                    results['dataset_comparison'] = {
                        'mean_viscosity_pa_s': mean_v,
                        'viscosity_variation_coefficient': (std_v/mean_v) if mean_v > 0 else np.nan,
                        'viscosity_range_pa_s': [float(np.min(visc)), float(np.max(visc))],
                        'frequency_dependent_behavior': bool(std_v/mean_v > 0.5) if mean_v > 0 else False
                    }

            results['success'] = True
        except Exception as e:
            results['error'] = str(e)
        return results

    def analyze_microrheology(self, tracks_df: pd.DataFrame, pixel_size_um: float,
                              frame_interval_s: float, max_lag: int = 20) -> Dict:
        """High level single dataset microrheology analysis."""

        msd_df = calculate_msd(
            tracks_df, pixel_size=pixel_size_um, frame_interval=frame_interval_s, max_lag=max_lag
        )
        msd_df = msd_df.rename(columns={'lag_time': 'lag_time_s', 'msd': 'msd_m2'})
        msd_df['msd_m2'] = msd_df['msd_m2'] * 1e-12

        if msd_df.empty:
            return {'success': False, 'error': 'Insufficient data for MSD calculation'}

        # Fit power law to MSD
        power_law_fit = self.fit_power_law_msd(msd_df)

        # Determine frequency range based on sampling
        min_time = msd_df['lag_time_s'].min()
        max_time = msd_df['lag_time_s'].max()
        omega_list = np.logspace(
            np.log10(2 * np.pi / (max_time * 0.5)),
            np.log10(2 * np.pi / (min_time * 2)),
            20
        )

        g_prime_values = []
        g_double_prime_values = []
        viscosity_values = []
        frequencies_hz = []

        for omega in omega_list:
            g_p, g_pp = self.calculate_complex_modulus_gser(msd_df, omega)
            viscosity = self.calculate_frequency_dependent_viscosity(msd_df, omega)
            
            if not (np.isnan(g_p) or np.isnan(g_pp)):
                g_prime_values.append(g_p)
                g_double_prime_values.append(g_pp)
                viscosity_values.append(viscosity)
                frequencies_hz.append(omega / (2 * np.pi))

        effective_viscosity = self.calculate_effective_viscosity(msd_df)

        return {
            'success': True,
            'msd_data': msd_df,
            'power_law_fit': power_law_fit,
            'frequency_response': {
                'frequencies_hz': frequencies_hz,
                'g_prime_pa': g_prime_values,
                'g_double_prime_pa': g_double_prime_values,
                'viscosity_pa_s': viscosity_values,
            },
            'viscosity': {'effective_pa_s': effective_viscosity},
            'moduli': {
                'g_prime_mean_pa': np.mean(g_prime_values) if g_prime_values else np.nan,
                'g_prime_std_pa': np.std(g_prime_values) if g_prime_values else np.nan,
                'g_double_prime_mean_pa': np.mean(g_double_prime_values) if g_double_prime_values else np.nan,
                'g_double_prime_std_pa': np.std(g_double_prime_values) if g_double_prime_values else np.nan,
                'loss_tangent': np.mean(g_double_prime_values) / np.mean(g_prime_values) if g_prime_values and np.mean(g_prime_values) > 0 else np.inf
            }
        }

    def analyze_multi_dataset_rheology(self, track_datasets: List[pd.DataFrame],
                                       pixel_sizes: List[float],
                                       frame_intervals: List[float],
                                       max_lag: int = 20) -> Dict:
        """Wrapper for multi-dataset microrheology analysis."""

        if len(track_datasets) != len(frame_intervals):
            return {'success': False, 'error': 'Number of datasets must match number of frame intervals'}

        pixel_size = pixel_sizes[0] if isinstance(pixel_sizes, list) else pixel_sizes

        return self.multi_dataset_analysis(
            track_datasets=track_datasets,
            frame_intervals_s=frame_intervals,
            pixel_size_um=pixel_size
        )

    def analyze_viscoelasticity(self, tracks_df: pd.DataFrame,
                                pixel_size_um: float = 0.1,
                                frame_interval_s: float = 0.1,
                                max_lag: int = 20) -> Dict:
        """Backward compatible API used by older components."""
        return self.analyze_microrheology(
            tracks_df,
            pixel_size_um=pixel_size_um,
            frame_interval_s=frame_interval_s,
            max_lag=max_lag,
        )
    
    def calculate_creep_compliance(self, msd_df: pd.DataFrame, 
                                  time_points_s: np.ndarray = None) -> Dict:
        """
        Calculate creep compliance J(t) from MSD data.
        
        Creep compliance describes the time-dependent deformation under constant stress.
        For passive microrheology: J(t) ≈ <Δr²(t)> / (6*kB*T*a)
        
        This is a simplified calculation valid for isotropic materials. More sophisticated
        approaches would use the inverse Laplace transform of 1/(s*G*(s)).
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data with columns: lag_time_s, msd_m2
        time_points_s : np.ndarray, optional
            Specific time points to calculate. If None, uses MSD time points.
            
        Returns
        -------
        Dict
            Creep compliance data including time series and power-law fit
        """
        if msd_df is None or len(msd_df) < 3:
            return {'success': False, 'error': 'Insufficient MSD data'}
        
        try:
            # Use provided time points or MSD times
            if time_points_s is None:
                times = msd_df['lag_time_s'].values
                msd_vals = msd_df['msd_m2'].values
            else:
                # Interpolate MSD at requested times
                times = time_points_s
                msd_vals = np.interp(times, msd_df['lag_time_s'], msd_df['msd_m2'])
            
            # Calculate creep compliance
            # J(t) = <Δr²(t)> / (6*kB*T*a) for 3D
            # J(t) = <Δr²(t)> / (4*kB*T*a) for 2D projected
            prefactor = 1.0 / (4.0 * self.kB * self.temperature_K * self.particle_radius_m)
            creep_compliance = prefactor * msd_vals
            
            # Fit power law: J(t) = J₀ * t^β
            try:
                log_t = np.log(times[times > 0])
                log_j = np.log(creep_compliance[times > 0])
                valid = np.isfinite(log_t) & np.isfinite(log_j)
                
                if np.sum(valid) >= 3:
                    coeffs = np.polyfit(log_t[valid], log_j[valid], 1)
                    beta = coeffs[0]
                    log_j0 = coeffs[1]
                    j0 = np.exp(log_j0)
                    
                    # Calculate R²
                    j_pred = np.exp(np.polyval(coeffs, log_t[valid]))
                    ss_res = np.sum((creep_compliance[times > 0][valid] - j_pred)**2)
                    ss_tot = np.sum((creep_compliance[times > 0][valid] - 
                                   np.mean(creep_compliance[times > 0][valid]))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                else:
                    beta, j0, r_squared = np.nan, np.nan, np.nan
            except:
                beta, j0, r_squared = np.nan, np.nan, np.nan
            
            # Material classification based on β
            if np.isfinite(beta):
                if beta < 0.3:
                    material_type = "Elastic solid"
                elif beta < 0.7:
                    material_type = "Viscoelastic solid"
                elif beta < 1.3:
                    material_type = "Viscoelastic fluid"
                else:
                    material_type = "Viscous fluid"
            else:
                material_type = "Unknown"
            
            return {
                'success': True,
                'time_s': times,
                'creep_compliance_pa_inv': creep_compliance,
                'power_law_fit': {
                    'j0_pa_inv': j0,
                    'beta': beta,
                    'r_squared': r_squared
                },
                'material_type': material_type,
                'mean_compliance': float(np.mean(creep_compliance)),
                'units': '1/Pa'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Creep compliance calculation failed: {str(e)}'}
    
    def calculate_relaxation_modulus(self, msd_df: pd.DataFrame,
                                    time_points_s: np.ndarray = None,
                                    use_approximation: bool = True) -> Dict:
        """
        Calculate relaxation modulus G(t) from MSD data.
        
        Relaxation modulus describes stress decay under constant strain.
        Approximation: G(t) ≈ kB*T / (π*a*<Δr²(t)>)
        
        More accurate methods would use inverse Fourier transform of G*(ω),
        but require data at multiple frequencies.
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data with columns: lag_time_s, msd_m2
        time_points_s : np.ndarray, optional
            Specific time points to calculate. If None, uses MSD time points.
        use_approximation : bool
            If True, uses simplified formula. If False, attempts
            frequency-domain calculation (requires more data).
            
        Returns
        -------
        Dict
            Relaxation modulus data including time series and decay parameters
        """
        if msd_df is None or len(msd_df) < 3:
            return {'success': False, 'error': 'Insufficient MSD data'}
        
        try:
            # Use provided time points or MSD times
            if time_points_s is None:
                times = msd_df['lag_time_s'].values
                msd_vals = msd_df['msd_m2'].values
            else:
                times = time_points_s
                msd_vals = np.interp(times, msd_df['lag_time_s'], msd_df['msd_m2'])
            
            if use_approximation:
                # Simplified approximation: G(t) ≈ kB*T / (π*a*<Δr²(t)>)
                prefactor = (self.kB * self.temperature_K) / (np.pi * self.particle_radius_m)
                relaxation_modulus = prefactor / msd_vals
                
            else:
                # More sophisticated: Use G' and G" to estimate G(t) via inverse FT
                # This requires calculating G*(ω) at multiple frequencies
                omega_range = np.logspace(-2, 2, 50)  # 50 frequency points
                g_prime_vals = []
                g_double_prime_vals = []
                
                for omega in omega_range:
                    gp, gpp = self.calculate_complex_modulus_gser(msd_df, omega)
                    if not (np.isnan(gp) or np.isnan(gpp)):
                        g_prime_vals.append(gp)
                        g_double_prime_vals.append(gpp)
                    else:
                        g_prime_vals.append(0)
                        g_double_prime_vals.append(0)
                
                # Approximate inverse FT using numerical integration
                relaxation_modulus = np.zeros_like(times)
                for i, t in enumerate(times):
                    if t > 0:
                        # G(t) = integral of G'(ω)*cos(ωt) + G"(ω)*sin(ωt) over ω
                        integrand = np.array([gp * np.cos(omega * t) + gpp * np.sin(omega * t)
                                            for gp, gpp, omega in zip(g_prime_vals, 
                                                                     g_double_prime_vals, 
                                                                     omega_range)])
                        relaxation_modulus[i] = integrate.trapz(integrand, omega_range) * 2 / np.pi
            
            # Fit exponential decay: G(t) = G₀ * exp(-t/τ) + G_inf
            try:
                valid_mask = (times > 0) & np.isfinite(relaxation_modulus) & (relaxation_modulus > 0)
                if np.sum(valid_mask) >= 4:
                    t_valid = times[valid_mask]
                    g_valid = relaxation_modulus[valid_mask]
                    
                    # Estimate parameters
                    g_inf = np.min(g_valid) if len(g_valid) > 0 else 0
                    g0_est = np.max(g_valid) - g_inf
                    tau_est = t_valid[len(t_valid)//2]  # Rough estimate
                    
                    # Fit exponential
                    def exp_decay(t, g0, tau, g_inf):
                        return g0 * np.exp(-t / tau) + g_inf
                    
                    try:
                        from scipy.optimize import curve_fit
                        popt, _ = curve_fit(exp_decay, t_valid, g_valid, 
                                          p0=[g0_est, tau_est, g_inf],
                                          bounds=([0, 0, 0], [np.inf, np.inf, np.max(g_valid)]),
                                          maxfev=1000)
                        g0_fit, tau_fit, g_inf_fit = popt
                        
                        # Calculate R²
                        g_pred = exp_decay(t_valid, *popt)
                        ss_res = np.sum((g_valid - g_pred)**2)
                        ss_tot = np.sum((g_valid - np.mean(g_valid))**2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    except:
                        g0_fit, tau_fit, g_inf_fit, r_squared = np.nan, np.nan, np.nan, np.nan
                else:
                    g0_fit, tau_fit, g_inf_fit, r_squared = np.nan, np.nan, np.nan, np.nan
            except:
                g0_fit, tau_fit, g_inf_fit, r_squared = np.nan, np.nan, np.nan, np.nan
            
            # Calculate loss tangent estimate
            if np.isfinite(g0_fit) and np.isfinite(g_inf_fit) and g_inf_fit > 0:
                loss_tangent_approx = (g0_fit - g_inf_fit) / g_inf_fit
            else:
                loss_tangent_approx = np.nan
            
            return {
                'success': True,
                'time_s': times,
                'relaxation_modulus_pa': relaxation_modulus,
                'exponential_fit': {
                    'g0_pa': g0_fit,
                    'tau_s': tau_fit,
                    'g_inf_pa': g_inf_fit,
                    'r_squared': r_squared
                },
                'mean_modulus_pa': float(np.mean(relaxation_modulus[np.isfinite(relaxation_modulus)])),
                'loss_tangent_approx': loss_tangent_approx,
                'method': 'approximation' if use_approximation else 'frequency_domain'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Relaxation modulus calculation failed: {str(e)}'}
    
    def two_point_microrheology(self, tracks_df: pd.DataFrame,
                               pixel_size_um: float,
                               frame_interval_s: float,
                               distance_bins_um: np.ndarray = None,
                               max_lag: int = 20) -> Dict:
        """
        Two-point microrheology: analyze cross-correlations between particle pairs.
        
        This method calculates distance-dependent viscoelastic properties by analyzing
        the cross-correlation of particle displacements as a function of separation distance.
        Reveals spatial heterogeneity in the medium.
        
        Theory: For particles separated by distance r, the cross-MSD is:
        <Δr₁(t)·Δr₂(t)> which decays with distance for elastic media.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
        pixel_size_um : float
            Pixel size in micrometers
        frame_interval_s : float
            Time between frames in seconds
        distance_bins_um : np.ndarray, optional
            Distance bins for pair correlation. If None, uses automatic binning.
        max_lag : int
            Maximum lag time for MSD calculation
            
        Returns
        -------
        Dict
            Two-point microrheology results including distance-dependent moduli
        """
        if tracks_df is None or len(tracks_df) < 10:
            return {'success': False, 'error': 'Insufficient tracking data'}
        
        try:
            # Convert coordinates to micrometers
            tracks_um = tracks_df.copy()
            tracks_um['x'] = tracks_um['x'] * pixel_size_um
            tracks_um['y'] = tracks_um['y'] * pixel_size_um
            
            # Get unique track IDs and frames
            track_ids = tracks_um['track_id'].unique()
            
            if len(track_ids) < 2:
                return {'success': False, 'error': 'Need at least 2 tracks for two-point analysis'}
            
            # Calculate initial positions and average positions for all tracks
            track_positions = {}
            for tid in track_ids:
                track_data = tracks_um[tracks_um['track_id'] == tid].sort_values('frame')
                if len(track_data) >= max_lag:
                    track_positions[tid] = {
                        'x_avg': track_data['x'].mean(),
                        'y_avg': track_data['y'].mean(),
                        'data': track_data
                    }
            
            valid_tracks = list(track_positions.keys())
            if len(valid_tracks) < 2:
                return {'success': False, 'error': 'Insufficient tracks with enough frames'}
            
            # Calculate pairwise distances and cross-MSDs
            if distance_bins_um is None:
                # Auto-generate distance bins
                all_distances = []
                for i, tid1 in enumerate(valid_tracks):
                    for tid2 in valid_tracks[i+1:]:
                        dx = track_positions[tid1]['x_avg'] - track_positions[tid2]['x_avg']
                        dy = track_positions[tid1]['y_avg'] - track_positions[tid2]['y_avg']
                        dist = np.sqrt(dx**2 + dy**2)
                        all_distances.append(dist)
                
                if len(all_distances) < 5:
                    return {'success': False, 'error': 'Too few particle pairs'}
                
                # Create bins based on distance distribution
                min_dist = np.min(all_distances)
                max_dist = np.max(all_distances)
                distance_bins_um = np.linspace(min_dist, max_dist, 6)  # 5 bins
            
            # Initialize results storage
            distance_results = []
            
            for i_bin in range(len(distance_bins_um) - 1):
                bin_min = distance_bins_um[i_bin]
                bin_max = distance_bins_um[i_bin + 1]
                bin_center = (bin_min + bin_max) / 2
                
                # Find pairs within this distance bin
                cross_msds = []
                
                for i, tid1 in enumerate(valid_tracks):
                    for tid2 in valid_tracks[i+1:]:
                        # Calculate average separation
                        dx = track_positions[tid1]['x_avg'] - track_positions[tid2]['x_avg']
                        dy = track_positions[tid1]['y_avg'] - track_positions[tid2]['y_avg']
                        separation = np.sqrt(dx**2 + dy**2)
                        
                        if bin_min <= separation < bin_max:
                            # Calculate cross-MSD for this pair
                            data1 = track_positions[tid1]['data']
                            data2 = track_positions[tid2]['data']
                            
                            # Find common frames
                            common_frames = sorted(set(data1['frame']).intersection(set(data2['frame'])))
                            
                            if len(common_frames) >= max_lag:
                                # Calculate cross-MSD
                                cross_msd_vals = []
                                for lag in range(1, min(max_lag, len(common_frames))):
                                    displacements = []
                                    for start_frame in common_frames[:-lag]:
                                        if start_frame + lag in common_frames:
                                            # Get positions at start and end
                                            pos1_start = data1[data1['frame'] == start_frame][['x', 'y']].values
                                            pos1_end = data1[data1['frame'] == start_frame + lag][['x', 'y']].values
                                            pos2_start = data2[data2['frame'] == start_frame][['x', 'y']].values
                                            pos2_end = data2[data2['frame'] == start_frame + lag][['x', 'y']].values
                                            
                                            if len(pos1_start) > 0 and len(pos1_end) > 0 and \
                                               len(pos2_start) > 0 and len(pos2_end) > 0:
                                                # Calculate displacements
                                                dr1 = pos1_end[0] - pos1_start[0]
                                                dr2 = pos2_end[0] - pos2_start[0]
                                                # Cross-correlation (dot product)
                                                cross_disp = np.dot(dr1, dr2)
                                                displacements.append(cross_disp)
                                    
                                    if len(displacements) > 0:
                                        cross_msd_vals.append(np.mean(displacements))
                                
                                if len(cross_msd_vals) >= 3:
                                    cross_msds.append(cross_msd_vals)
                
                if len(cross_msds) > 0:
                    # Calculate average cross-MSD for this distance bin
                    # Make all arrays same length
                    min_len = min(len(msd) for msd in cross_msds)
                    cross_msds_array = np.array([msd[:min_len] for msd in cross_msds])
                    avg_cross_msd = np.mean(cross_msds_array, axis=0)
                    lag_times = np.arange(1, len(avg_cross_msd) + 1) * frame_interval_s
                    
                    # Create MSD DataFrame for this bin
                    msd_df_bin = pd.DataFrame({
                        'lag_time_s': lag_times,
                        'msd_m2': np.abs(avg_cross_msd) * 1e-12  # Convert μm² to m², use abs
                    })
                    
                    # Calculate G' and G" at representative frequency
                    if len(lag_times) > 1:
                        omega = 2 * np.pi / (lag_times[len(lag_times)//2])
                        g_prime, g_double_prime = self.calculate_complex_modulus_gser(msd_df_bin, omega)
                    else:
                        g_prime, g_double_prime = np.nan, np.nan
                    
                    distance_results.append({
                        'distance_um': bin_center,
                        'distance_range': (bin_min, bin_max),
                        'n_pairs': len(cross_msds),
                        'cross_msd': avg_cross_msd,
                        'lag_times_s': lag_times,
                        'g_prime_pa': g_prime,
                        'g_double_prime_pa': g_double_prime,
                        'correlation_strength': avg_cross_msd[0] if len(avg_cross_msd) > 0 else np.nan
                    })
            
            if len(distance_results) == 0:
                return {'success': False, 'error': 'No valid distance bins with sufficient pairs'}
            
            # Analyze spatial correlation length
            distances = [r['distance_um'] for r in distance_results]
            correlations = [r['correlation_strength'] for r in distance_results]
            
            # Fit exponential decay: C(r) = C₀ * exp(-r/ξ) to estimate correlation length ξ
            try:
                valid_corr = np.isfinite(correlations) & (np.array(correlations) > 0)
                if np.sum(valid_corr) >= 3:
                    from scipy.optimize import curve_fit
                    def exp_decay(r, c0, xi):
                        return c0 * np.exp(-r / xi)
                    
                    popt, _ = curve_fit(exp_decay, 
                                      np.array(distances)[valid_corr],
                                      np.abs(np.array(correlations)[valid_corr]),
                                      p0=[np.abs(correlations[0]), distances[len(distances)//2]],
                                      bounds=([0, 0], [np.inf, max(distances)*2]),
                                      maxfev=1000)
                    correlation_length = popt[1]
                else:
                    correlation_length = np.nan
            except:
                correlation_length = np.nan
            
            return {
                'success': True,
                'distance_bins': distance_results,
                'correlation_length_um': correlation_length,
                'n_distance_bins': len(distance_results),
                'distance_range_um': (min(distances), max(distances)),
                'method': 'two_point_microrheology'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Two-point microrheology failed: {str(e)}'}
    
    def spatial_microrheology_map(self, tracks_df: pd.DataFrame,
                                 pixel_size_um: float,
                                 frame_interval_s: float,
                                 grid_size_um: float = 10.0,
                                 max_lag: int = 20,
                                 min_tracks_per_bin: int = 3) -> Dict:
        """
        Generate spatial map of microrheological properties.
        
        Divides the field of view into spatial bins and calculates G', G", and viscosity
        for particles in each bin. Reveals spatial heterogeneity in mechanical properties.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
        pixel_size_um : float
            Pixel size in micrometers
        frame_interval_s : float
            Time between frames in seconds
        grid_size_um : float
            Size of spatial bins in micrometers
        max_lag : int
            Maximum lag for MSD calculation
        min_tracks_per_bin : int
            Minimum number of tracks required per bin
            
        Returns
        -------
        Dict
            Spatial map of viscoelastic properties
        """
        if tracks_df is None or len(tracks_df) < min_tracks_per_bin * 4:
            return {'success': False, 'error': 'Insufficient tracking data'}
        
        try:
            # Convert coordinates to micrometers
            tracks_um = tracks_df.copy()
            tracks_um['x'] = tracks_um['x'] * pixel_size_um
            tracks_um['y'] = tracks_um['y'] * pixel_size_um
            
            # Determine field of view bounds
            x_min, x_max = tracks_um['x'].min(), tracks_um['x'].max()
            y_min, y_max = tracks_um['y'].min(), tracks_um['y'].max()
            
            # Create spatial grid
            x_bins = np.arange(x_min, x_max + grid_size_um, grid_size_um)
            y_bins = np.arange(y_min, y_max + grid_size_um, grid_size_um)
            
            if len(x_bins) < 2 or len(y_bins) < 2:
                return {'success': False, 'error': 'Field of view too small for spatial mapping'}
            
            # Initialize results storage
            spatial_results = []
            
            # Calculate properties for each spatial bin
            for i in range(len(x_bins) - 1):
                for j in range(len(y_bins) - 1):
                    x_start, x_end = x_bins[i], x_bins[i+1]
                    y_start, y_end = y_bins[j], y_bins[j+1]
                    
                    # Find tracks in this bin (based on average position)
                    bin_tracks = []
                    for track_id in tracks_um['track_id'].unique():
                        track_data = tracks_um[tracks_um['track_id'] == track_id]
                        x_avg = track_data['x'].mean()
                        y_avg = track_data['y'].mean()
                        
                        if x_start <= x_avg < x_end and y_start <= y_avg < y_end:
                            if len(track_data) >= max_lag:
                                bin_tracks.append(track_id)
                    
                    if len(bin_tracks) >= min_tracks_per_bin:
                        # Extract data for tracks in this bin
                        bin_data = tracks_um[tracks_um['track_id'].isin(bin_tracks)].copy()
                        
                        # Convert back to pixels for MSD calculation
                        bin_data['x'] = bin_data['x'] / pixel_size_um
                        bin_data['y'] = bin_data['y'] / pixel_size_um
                        
                        # Calculate MSD for this bin
                        try:
                            msd_result = calculate_msd(bin_data, max_lag=max_lag)
                            
                            if 'lag_time' in msd_result.columns and 'msd' in msd_result.columns:
                                # Group by lag_time and average
                                msd_avg = msd_result.groupby('lag_time')['msd'].mean()
                                
                                if len(msd_avg) >= 3:
                                    # Convert to physical units
                                    msd_df = pd.DataFrame({
                                        'lag_time_s': msd_avg.index * frame_interval_s,
                                        'msd_m2': msd_avg.values * (pixel_size_um * 1e-6)**2
                                    })
                                    
                                    # Calculate viscoelastic properties
                                    if len(msd_df) > 1:
                                        omega = 2 * np.pi / (msd_df['lag_time_s'].iloc[len(msd_df)//2])
                                        g_prime, g_double_prime = self.calculate_complex_modulus_gser(msd_df, omega)
                                        viscosity = self.calculate_effective_viscosity(msd_df)
                                    else:
                                        g_prime, g_double_prime, viscosity = np.nan, np.nan, np.nan
                                    
                                    spatial_results.append({
                                        'x_center_um': (x_start + x_end) / 2,
                                        'y_center_um': (y_start + y_end) / 2,
                                        'x_range': (x_start, x_end),
                                        'y_range': (y_start, y_end),
                                        'n_tracks': len(bin_tracks),
                                        'g_prime_pa': g_prime,
                                        'g_double_prime_pa': g_double_prime,
                                        'viscosity_pa_s': viscosity,
                                        'loss_tangent': g_double_prime / g_prime if (np.isfinite(g_prime) and g_prime > 0) else np.nan
                                    })
                        except Exception as e:
                            # Skip bins with calculation errors
                            continue
            
            if len(spatial_results) == 0:
                return {'success': False, 'error': 'No spatial bins with sufficient data'}
            
            # Calculate statistics across the field
            g_prime_vals = [r['g_prime_pa'] for r in spatial_results if np.isfinite(r['g_prime_pa'])]
            g_double_prime_vals = [r['g_double_prime_pa'] for r in spatial_results if np.isfinite(r['g_double_prime_pa'])]
            viscosity_vals = [r['viscosity_pa_s'] for r in spatial_results if np.isfinite(r['viscosity_pa_s'])]
            
            # Calculate spatial heterogeneity (coefficient of variation)
            def calc_cv(vals):
                if len(vals) > 1:
                    return np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else np.nan
                return np.nan
            
            return {
                'success': True,
                'spatial_bins': spatial_results,
                'grid_size_um': grid_size_um,
                'n_bins': len(spatial_results),
                'field_of_view_um': {
                    'x_range': (x_min, x_max),
                    'y_range': (y_min, y_max)
                },
                'global_statistics': {
                    'g_prime_mean_pa': np.mean(g_prime_vals) if g_prime_vals else np.nan,
                    'g_prime_std_pa': np.std(g_prime_vals) if g_prime_vals else np.nan,
                    'g_prime_cv': calc_cv(g_prime_vals),
                    'g_double_prime_mean_pa': np.mean(g_double_prime_vals) if g_double_prime_vals else np.nan,
                    'g_double_prime_std_pa': np.std(g_double_prime_vals) if g_double_prime_vals else np.nan,
                    'g_double_prime_cv': calc_cv(g_double_prime_vals),
                    'viscosity_mean_pa_s': np.mean(viscosity_vals) if viscosity_vals else np.nan,
                    'viscosity_std_pa_s': np.std(viscosity_vals) if viscosity_vals else np.nan,
                    'viscosity_cv': calc_cv(viscosity_vals)
                },
                'heterogeneity_index': {
                    'g_prime': calc_cv(g_prime_vals),
                    'g_double_prime': calc_cv(g_double_prime_vals),
                    'viscosity': calc_cv(viscosity_vals)
                },
                'method': 'spatial_microrheology_map'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Spatial microrheology mapping failed: {str(e)}'}


def create_rheology_plots(analysis_results: Dict) -> Dict[str, go.Figure]:
    """
    Create comprehensive plots for microrheology analysis results.
    
    Parameters
    ----------
    analysis_results : Dict
        Results from multi_dataset_analysis or single dataset analysis
        
    Returns
    -------
    Dict[str, go.Figure]
        Dictionary of plotly figures
    """
    figures = {}
    
    # Handle both old and new result structures
    if 'datasets' in analysis_results:
        # New multi-dataset structure
        datasets = analysis_results['datasets']
        
        # MSD comparison plot for multiple datasets
        if len(datasets) > 0:
            fig_msd = go.Figure()
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, dataset in enumerate(datasets):
                if 'msd_data' in dataset and not dataset['msd_data'].empty:
                    col = colors[i % len(colors)]
                    fig_msd.add_trace(go.Scatter(
                        x=dataset['msd_data']['lag_time_s'],
                        y=dataset['msd_data']['msd_m2'],
                        mode='lines+markers',
                        name=f"{dataset['label']} MSD",
                        line=dict(color=col)
                    ))
                    
                    # Add power law fit if available
                    if 'power_law_fit' in dataset and not np.isnan(dataset['power_law_fit']['amplitude']):
                        fit_data = dataset['power_law_fit']
                        t_fit = np.logspace(np.log10(dataset['msd_data']['lag_time_s'].min()), 
                                          np.log10(dataset['msd_data']['lag_time_s'].max()), 50)
                        msd_fit = fit_data['amplitude'] * (t_fit ** fit_data['exponent'])
                        
                        fig_msd.add_trace(go.Scatter(
                            x=t_fit,
                            y=msd_fit,
                            mode='lines',
                            name=f"{dataset['label']} fit (α={fit_data['exponent']:.2f})",
                            line=dict(color=col, dash='dash'),
                            showlegend=True
                        ))
            
            fig_msd.update_layout(
                title='Mean Squared Displacement Comparison Across Datasets',
                xaxis_title='Lag Time (s)',
                yaxis_title='MSD (m²)',
                xaxis_type='log',
                yaxis_type='log',
                template='plotly_white'
            )
            
            figures['msd_comparison'] = fig_msd
        
        # Combined frequency response plot
        if 'combined_frequency_response' in analysis_results:
            freq_data = analysis_results['combined_frequency_response']
            
            if freq_data and 'frequencies_hz' in freq_data:
                fig_freq = go.Figure()
                
                # G' (storage modulus)
                fig_freq.add_trace(go.Scatter(
                    x=freq_data['frequencies_hz'],
                    y=freq_data['g_prime_pa'],
                    mode='lines+markers',
                    name="G' (Storage Modulus)",
                    line=dict(color='green'),
                    yaxis='y'
                ))
                
                # G" (loss modulus)
                fig_freq.add_trace(go.Scatter(
                    x=freq_data['frequencies_hz'],
                    y=freq_data['g_double_prime_pa'],
                    mode='lines+markers',
                    name='G" (Loss Modulus)',
                    line=dict(color='orange'),
                    yaxis='y'
                ))
                
                # Viscosity if available
                if 'viscosity_pa_s' in freq_data:
                    fig_freq.add_trace(go.Scatter(
                        x=freq_data['frequencies_hz'],
                        y=freq_data['viscosity_pa_s'],
                        mode='lines+markers',
                        name='|η*| (Complex Viscosity)',
                        line=dict(color='purple'),
                        yaxis='y2'
                    ))
                
                fig_freq.update_layout(
                    title='Combined Complex Modulus vs Frequency',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Modulus (Pa)',
                    yaxis2=dict(
                        title='Viscosity (Pa·s)',
                        overlaying='y',
                        side='right'
                    ),
                    xaxis_type='log',
                    yaxis_type='log',
                    template='plotly_white'
                )
                
                figures['frequency_response'] = fig_freq
        
        # Individual dataset frequency responses
        fig_individual = go.Figure()
        colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown']
        
        for i, dataset in enumerate(datasets):
            if 'frequencies_hz' in dataset and len(dataset['frequencies_hz']) > 0:
                color = colors[i % len(colors)]
                
                # G' for this dataset
                fig_individual.add_trace(go.Scatter(
                    x=dataset['frequencies_hz'],
                    y=dataset['g_prime_pa'],
                    mode='lines+markers',
                    name=f"{dataset['label']} G'",
                    line=dict(color=color, dash='solid'),
                    legendgroup=dataset['label']
                ))
                
                # G" for this dataset
                fig_individual.add_trace(go.Scatter(
                    x=dataset['frequencies_hz'],
                    y=dataset['g_double_prime_pa'],
                    mode='lines+markers',
                    name=f"{dataset['label']} G\"",
                    line=dict(color=color, dash='dash'),
                    legendgroup=dataset['label']
                ))
        
        if len(fig_individual.data) > 0:
            fig_individual.update_layout(
                title='Individual Dataset Frequency Responses',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Modulus (Pa)',
                xaxis_type='log',
                yaxis_type='log',
                template='plotly_white'
            )
            
            figures['individual_frequency_response'] = fig_individual
    
    else:
        # Legacy single dataset structure (backwards compatibility)
        # MSD plot
        if 'msd_data' in analysis_results:
            fig_msd = go.Figure()
            msd_data = analysis_results['msd_data']
            
            fig_msd.add_trace(go.Scatter(
                x=msd_data['lag_time_s'],
                y=msd_data['msd_m2'],
                mode='lines+markers',
                name='MSD',
                line=dict(color='blue'),
                error_y=dict(
                    type='data',
                    array=msd_data.get('std_msd_m2', []),
                    visible=True
                )
            ))
            
            # Add power law fit if available
            if 'power_law_fit' in analysis_results and not np.isnan(analysis_results['power_law_fit']['amplitude']):
                fit_data = analysis_results['power_law_fit']
                t_fit = np.logspace(np.log10(msd_data['lag_time_s'].min()), 
                                  np.log10(msd_data['lag_time_s'].max()), 50)
                msd_fit = fit_data['amplitude'] * (t_fit ** fit_data['exponent'])
                
                fig_msd.add_trace(go.Scatter(
                    x=t_fit,
                    y=msd_fit,
                    mode='lines',
                    name=f"Power law fit (α={fit_data['exponent']:.2f})",
                    line=dict(color='red', dash='dash')
                ))
            
            fig_msd.update_layout(
                title='Mean Squared Displacement',
                xaxis_title='Lag Time (s)',
                yaxis_title='MSD (m²)',
                xaxis_type='log',
                yaxis_type='log',
                template='plotly_white'
            )
            
            figures['msd_comparison'] = fig_msd
        
        # Frequency response plot
        if 'frequency_response' in analysis_results and analysis_results['frequency_response']:
            freq_data = analysis_results['frequency_response']
            
            fig_freq = go.Figure()
            
            # G' (storage modulus)
            fig_freq.add_trace(go.Scatter(
                x=freq_data['frequencies_hz'],
                y=freq_data['g_prime_pa'],
                mode='lines+markers',
                name="G' (Storage Modulus)",
                line=dict(color='green'),
                yaxis='y'
            ))
            
            # G" (loss modulus)
            fig_freq.add_trace(go.Scatter(
                x=freq_data['frequencies_hz'],
                y=freq_data['g_double_prime_pa'],
                mode='lines+markers',
                name='G" (Loss Modulus)',
                line=dict(color='orange'),
                yaxis='y'
            ))
            
            # Viscosity if available
            if 'viscosity_pa_s' in freq_data:
                fig_freq.add_trace(go.Scatter(
                    x=freq_data['frequencies_hz'],
                    y=freq_data['viscosity_pa_s'],
                    mode='lines+markers',
                    name='|η*| (Complex Viscosity)',
                    line=dict(color='purple'),
                    yaxis='y2'
                ))
            
            fig_freq.update_layout(
                title='Complex Modulus vs Frequency',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Modulus (Pa)',
                yaxis2=dict(
                    title='Viscosity (Pa·s)',
                    overlaying='y',
                    side='right'
                ),
                xaxis_type='log',
                yaxis_type='log',
                template='plotly_white'
            )
            
            figures['frequency_response'] = fig_freq
    
    return figures


def display_rheology_summary(analysis_results: Dict) -> None:
    """
    Display summary statistics for microrheology analysis.
    
    Parameters
    ----------
    analysis_results : Dict
        Results from multi_dataset_analysis or single dataset analysis
    """
    if not analysis_results:
        st.error("No microrheology analysis results available")
        return
        
    if not analysis_results.get('success', False):
        st.error(f"Microrheology analysis failed: {analysis_results.get('error', 'Unknown error')}")
        return
    
    # Store the available analyses in session state so other components can check it
    if 'available_rheology_analyses' not in st.session_state:
        st.session_state.available_rheology_analyses = []
    
    # Mark analyses as available
    if 'datasets' in analysis_results and len(analysis_results['datasets']) > 0:
        st.session_state.available_rheology_analyses.append('datasets')
    if 'combined_frequency_response' in analysis_results:
        st.session_state.available_rheology_analyses.append('frequency_response')
    if 'dataset_comparison' in analysis_results:
        st.session_state.available_rheology_analyses.append('comparison')
    
    st.subheader("Microrheology Analysis Summary")
    
    # Handle multi-dataset structure
    if 'datasets' in analysis_results:
        datasets = analysis_results['datasets']
        
        st.write(f"**Number of Datasets:** {len(datasets)}")
        
        # Dataset overview table
        if len(datasets) > 0:
            dataset_summary = []
            for dataset in datasets:
                power_law = dataset.get('power_law_fit', {})
                summary_row = {
                    'Dataset': dataset['label'],
                    'Frame Interval (s)': f"{dataset['frame_interval_s']:.3f}",
                    'MSD Exponent (α)': f"{power_law.get('exponent', 0):.3f}" if not np.isnan(power_law.get('exponent', np.nan)) else "N/A",
                    'R² (fit)': f"{power_law.get('r_squared', 0):.3f}" if not np.isnan(power_law.get('r_squared', np.nan)) else "N/A",
                    'Effective Viscosity (Pa·s)': f"{dataset.get('effective_viscosity_pa_s', 0):.2e}" if not np.isnan(dataset.get('effective_viscosity_pa_s', np.nan)) else "N/A",
                    'G\' Mean (Pa)': f"{dataset.get('g_prime_mean_pa', 0):.2e}" if dataset.get('g_prime_mean_pa') is not None else "N/A",
                    'G\" Mean (Pa)': f"{dataset.get('g_double_prime_mean_pa', 0):.2e}" if dataset.get('g_double_prime_mean_pa') is not None else "N/A",
                    'Loss Tangent': f"{dataset.get('loss_tangent', 0):.3f}" if dataset.get('loss_tangent') is not None and not np.isinf(dataset.get('loss_tangent', np.inf)) else "N/A"
                }
                dataset_summary.append(summary_row)
            
            st.table(pd.DataFrame(dataset_summary))
        
        # Combined frequency response summary
        if 'combined_frequency_response' in analysis_results and analysis_results['combined_frequency_response']:
            freq_data = analysis_results['combined_frequency_response']
            
            st.subheader("Combined Frequency Response")
            
            if 'frequency_range_hz' in freq_data:
                freq_range = freq_data['frequency_range_hz']
                st.write(f"**Frequency Range:** {freq_range[0]:.2e} - {freq_range[1]:.2e} Hz")
            
            if 'frequencies_hz' in freq_data:
                st.write(f"**Total Data Points:** {len(freq_data['frequencies_hz'])}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'g_prime_overall_mean_pa' in freq_data:
                    st.metric("Mean G' (Pa)", f"{freq_data['g_prime_overall_mean_pa']:.2e}")
            
            with col2:
                if 'g_double_prime_overall_mean_pa' in freq_data:
                    st.metric("Mean G\" (Pa)", f"{freq_data['g_double_prime_overall_mean_pa']:.2e}")
            
            with col3:
                if 'viscosity_overall_mean_pa_s' in freq_data:
                    st.metric("Mean |η*| (Pa·s)", f"{freq_data['viscosity_overall_mean_pa_s']:.2e}")
        
        # Dataset comparison
        if 'dataset_comparison' in analysis_results and analysis_results['dataset_comparison']:
            comparison = analysis_results['dataset_comparison']
            
            st.subheader("Dataset Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'viscosity_variation_coefficient' in comparison:
                    st.metric(
                        label="Viscosity Variation Coefficient",
                        value=f"{comparison['viscosity_variation_coefficient']:.3f}",
                        help="Standard deviation / Mean of viscosity across datasets"
                    )
            
            with col2:
                if 'mean_viscosity_pa_s' in comparison:
                    st.metric(
                        label="Mean Viscosity Across Datasets",
                        value=f"{comparison['mean_viscosity_pa_s']:.2e} Pa·s",
                        delta=f"±{comparison.get('std_viscosity_pa_s', 0):.2e}"
                    )
            
            if comparison.get('frequency_dependent_behavior', False):
                st.warning("⚠️ Significant frequency-dependent behavior detected across datasets")
            else:
                st.success("✅ Consistent behavior across different sampling rates")
            
            if 'viscosity_range_pa_s' in comparison:
                visc_range = comparison['viscosity_range_pa_s']
                st.write(f"**Viscosity Range:** {visc_range[0]:.2e} - {visc_range[1]:.2e} Pa·s")
    
    else:
        # Legacy single dataset structure (backwards compatibility)
        # Power law fit summary
        if 'power_law_fit' in analysis_results:
            power_law = analysis_results['power_law_fit']
            
            st.subheader("MSD Power Law Fit")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Exponent (α)",
                    f"{power_law.get('exponent', 0):.3f}",
                    help="α = 1: normal diffusion, α < 1: subdiffusion, α > 1: superdiffusion"
                )
            
            with col2:
                st.metric(
                    "Amplitude (A)",
                    f"{power_law.get('amplitude', 0):.2e}",
                    help="Prefactor in MSD = A * t^α"
                )
            
            with col3:
                st.metric(
                    "R² (fit quality)",
                    f"{power_law.get('r_squared', 0):.3f}",
                    help="Coefficient of determination for power law fit"
                )
            
            # Interpret the exponent
            alpha = power_law.get('exponent', 1.0)
            if 0.9 <= alpha <= 1.1:
                st.success("✅ Normal diffusion detected (α ≈ 1)")
            elif alpha < 0.9:
                st.warning(f"⚠️ Subdiffusion detected (α = {alpha:.3f} < 1)")
            else:
                st.info(f"ℹ️ Superdiffusion detected (α = {alpha:.3f} > 1)")
        
        # Moduli summary
        if 'moduli' in analysis_results:
            moduli = analysis_results['moduli']
            
            st.subheader("Viscoelastic Moduli")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Storage Modulus (G')",
                    f"{moduli.get('g_prime_mean_pa', 0):.2e} Pa",
                    delta=f"±{moduli.get('g_prime_std_pa', 0):.2e}"
                )
            
            with col2:
                st.metric(
                    "Loss Modulus (G\")",
                    f"{moduli.get('g_double_prime_mean_pa', 0):.2e} Pa",
                    delta=f"±{moduli.get('g_double_prime_std_pa', 0):.2e}"
                )
            
            with col3:
                st.metric(
                    "Loss Tangent (G\"/G')",
                    f"{moduli.get('loss_tangent', np.nan):.3f}" if np.isfinite(moduli.get('loss_tangent', np.inf)) else "N/A"
                )
        
        # Viscosity summary
        if 'viscosity' in analysis_results:
            viscosity = analysis_results['viscosity']
            
            st.subheader("Effective Viscosity")
            
            st.metric(
                "Effective Viscosity",
                f"{viscosity.get('effective_pa_s', 0):.2e} Pa·s",
                help="Calculated from MSD slope using Stokes-Einstein relation"
            )