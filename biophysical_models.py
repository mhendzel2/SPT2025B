"""
Advanced biophysical models for single particle tracking analysis.
Specialized for nucleosome diffusion in chromatin and polymer physics modeling.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

class PolymerPhysicsModel:
    """
    Contains models and fitting routines from polymer physics, specifically for 
    nucleosome diffusion in chromatin fiber modeling using Rouse model dynamics.
    """
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def rouse_msd_model_phenomenological(t: np.ndarray, D_macro: float, Gamma: float, alpha: float = 0.5) -> np.ndarray:
        """
        Phenomenological Rouse-like MSD model for nucleosome diffusion in chromatin.
        
        MSD(t) = 4 * D_macro * t + Gamma * t^alpha
        
        The factor of 4 (for 2D) is included for proper dimensionality.
        For nucleosome diffusion:
        - D_macro: macroscopic diffusion coefficient (long-time behavior)
        - Gamma: amplitude of subdiffusive component (chromatin constraints)
        - alpha: anomalous diffusion exponent (typically 0.5 for Rouse dynamics)
        
        Parameters
        ----------
        t : np.ndarray
            Time lags
        D_macro : float
            Macroscopic diffusion coefficient (μm²/s)
        Gamma : float
            Amplitude of subdiffusive component
        alpha : float
            Anomalous diffusion exponent (0.5 for Rouse model)
            
        Returns
        -------
        np.ndarray
            Theoretical MSD values
        """
        # Ensure positive parameters during fitting and include proper 2D factor
        return 4 * np.abs(D_macro) * t + np.abs(Gamma) * (t ** np.abs(alpha))
    
    @staticmethod
    def rouse_msd_model_fixed_alpha(t: np.ndarray, D_macro: float, Gamma: float) -> np.ndarray:
        """Rouse MSD model with alpha fixed to 0.5 (classic Rouse dynamics)."""
        return PolymerPhysicsModel.rouse_msd_model_phenomenological(t, D_macro, Gamma, alpha=0.5)
    
    @staticmethod
    def confined_diffusion_model(t: np.ndarray, D_free: float, L_conf: float) -> np.ndarray:
        """
        Confined diffusion model for nucleosomes in chromatin loops.
        
        MSD(t) = L_conf^2 * (1 - exp(-12*D_free*t/L_conf^2))
        
        This is the correct 2D formula for confined diffusion.
        
        Parameters
        ----------
        t : np.ndarray
            Time lags
        D_free : float
            Free diffusion coefficient
        L_conf : float
            Confinement length scale
            
        Returns
        -------
        np.ndarray
            Theoretical MSD values for confined diffusion
        """
        L_conf = np.abs(L_conf)
        D_free = np.abs(D_free)
        
        # Avoid numerical issues with very small L_conf
        if L_conf < 1e-9:
            L_conf = 1e-9
            
        tau = L_conf**2 / (12 * D_free) if D_free > 0 else np.inf
        
        # Handle cases where tau is very small or very large
        if tau == 0:
            return np.zeros_like(t)
        elif tau == np.inf:
            return 4 * D_free * t  # Free diffusion with 2D factor
        else:
            return L_conf**2 * (1 - np.exp(-t / tau))

    @staticmethod
    def anomalous_diffusion_model(t: np.ndarray, K_alpha: float, alpha: float) -> np.ndarray:
        """
        Pure anomalous diffusion model.
        
        MSD(t) = K_alpha * t^alpha
        
        Parameters
        ----------
        t : np.ndarray
            Time lags
        K_alpha : float
            Generalized diffusion coefficient
        alpha : float
            Anomalous diffusion exponent
            
        Returns
        -------
        np.ndarray
            Theoretical MSD values
        """
        return np.abs(K_alpha) * (t ** np.abs(alpha))
    
    @staticmethod
    def fractional_brownian_motion_model(t: np.ndarray, D_H: float, H: float) -> np.ndarray:
        """
        Fractional Brownian Motion model for chromatin dynamics.
        
        MSD(t) = 2 * D_H * t^(2*H)
        
        Parameters
        ----------
        t : np.ndarray
            Time lags
        D_H : float
            Generalized diffusion coefficient
        H : float
            Hurst exponent (0 < H < 1)
            
        Returns
        -------
        np.ndarray
            Theoretical MSD values
        """
        H_bounded = np.clip(np.abs(H), 0.01, 0.99)  # Keep H in valid range
        return 2 * np.abs(D_H) * (t ** (2 * H_bounded))
    
    def fit_rouse_model_to_msd(self, 
                               time_lags: np.ndarray, 
                               msd_values: np.ndarray, 
                               fit_alpha_exponent: bool = False, 
                               initial_guess: Optional[List[float]] = None, 
                               bounds: Optional[Tuple[List[float], List[float]]] = None) -> Dict[str, Any]:
        """
        Fits the Rouse MSD model to experimental nucleosome tracking data.
        
        Parameters
        ----------
        time_lags : np.ndarray
            Array of time lags
        msd_values : np.ndarray
            Array of corresponding MSD values
        fit_alpha_exponent : bool
            If True, fits alpha; otherwise alpha fixed to 0.5
        initial_guess : Optional[List[float]]
            Initial parameter guesses
        bounds : Optional[Tuple[List[float], List[float]]]
            Parameter bounds
            
        Returns
        -------
        Dict[str, Any]
            Fitting results including parameters, errors, and goodness of fit
        """
        if len(time_lags) != len(msd_values):
            return {'success': False, 'error': 'time_lags and msd_values must have the same length.'}
        
        # Remove any zero or negative time lags and corresponding MSD values
        valid_mask = (time_lags > 0) & (msd_values > 0)
        time_lags = time_lags[valid_mask]
        msd_values = msd_values[valid_mask]
        
        if len(time_lags) < 3:
            return {'success': False, 'error': 'Not enough valid data points for fitting.'}
        
        num_params_expected = 3 if fit_alpha_exponent else 2
        if len(time_lags) < num_params_expected:
            return {'success': False, 'error': f'Not enough data points ({len(time_lags)}) to fit {num_params_expected} parameters.'}

        model_to_fit = self.rouse_msd_model_phenomenological if fit_alpha_exponent else self.rouse_msd_model_fixed_alpha

        if initial_guess is None:
            # Intelligent initial guesses for nucleosome diffusion
            # D_macro from long-time slope (divide by 4 to account for 2D factor)
            if len(time_lags) > 2:
                n_points = max(2, len(time_lags) // 4)
                slope_times = time_lags[-n_points:]
                slope_msds = msd_values[-n_points:]
                
                if len(slope_times) > 1 and slope_times[-1] > slope_times[0]:
                    slope = (slope_msds[-1] - slope_msds[0]) / (slope_times[-1] - slope_times[0])
                    D_macro_guess = slope / 4  # Account for the 4D factor in the model
                else:
                    D_macro_guess = msd_values[-1] / (4 * time_lags[-1]) if time_lags[-1] > 0 else 0.01
            else:
                D_macro_guess = 0.01
            
            D_macro_guess = max(D_macro_guess, 1e-9)
            
            # Gamma from early time behavior (subdiffusive component)
            early_idx = min(len(time_lags) // 3, len(time_lags) - 1)
            early_idx = max(1, early_idx)
            
            if time_lags[early_idx] > 0:
                msd_residual = msd_values[early_idx] - 4 * D_macro_guess * time_lags[early_idx]
                Gamma_guess = msd_residual / (time_lags[early_idx] ** 0.5) if msd_residual > 0 else 0.01
            else:
                Gamma_guess = 0.01
            
            Gamma_guess = max(Gamma_guess, 1e-9)
            
            if fit_alpha_exponent:
                initial_guess = [D_macro_guess, Gamma_guess, 0.5]
            else:
                initial_guess = [D_macro_guess, Gamma_guess]
        
        if bounds is None:
            if fit_alpha_exponent:  # D_macro, Gamma, alpha
                lower_bounds = [0, 0, 0.1]
                upper_bounds = [np.inf, np.inf, 1.0]
            else:  # D_macro, Gamma
                lower_bounds = [0, 0]
                upper_bounds = [np.inf, np.inf]
            bounds = (lower_bounds, upper_bounds)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, covariance = curve_fit(
                    model_to_fit,
                    time_lags,
                    msd_values,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=20000,
                    ftol=1e-8,
                    xtol=1e-8
                )
            
            # Calculate goodness of fit
            fitted_msd = model_to_fit(time_lags, *params)
            residuals = msd_values - fitted_msd
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            
            # Calculate reduced chi-squared
            dof = len(time_lags) - len(params)
            reduced_chi_squared = ss_res / dof if dof > 0 else np.inf
            
            param_names = ['D_macro', 'Gamma', 'alpha'] if fit_alpha_exponent else ['D_macro', 'Gamma']
            fitted_params_dict = {name: val for name, val in zip(param_names, params)}
            
            param_errors = np.sqrt(np.diag(covariance)) if covariance is not None else [np.nan] * len(params)
            param_errors_dict = {name: err for name, err in zip(param_names, param_errors)}
            
            # Calculate confidence intervals (95%)
            confidence_intervals = {}
            for i, name in enumerate(param_names):
                if not np.isnan(param_errors[i]):
                    confidence_intervals[name] = {
                        'lower': params[i] - 1.96 * param_errors[i],
                        'upper': params[i] + 1.96 * param_errors[i]
                    }
                else:
                    confidence_intervals[name] = {'lower': np.nan, 'upper': np.nan}

            return {
                'success': True,
                'model_type': 'rouse',
                'params': fitted_params_dict,
                'param_errors': param_errors_dict,
                'confidence_intervals': confidence_intervals,
                'covariance_matrix': covariance.tolist() if covariance is not None else None,
                'r_squared': r_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'fitted_msd_values': fitted_msd.tolist(),
                'residuals': residuals.tolist(),
                'time_lags': time_lags.tolist(),
                'original_msd_values': msd_values.tolist()
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Fitting failed: {str(e)}'}
    
    def fit_confined_diffusion_model(self, 
                                   time_lags: np.ndarray, 
                                   msd_values: np.ndarray,
                                   initial_guess: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Fit confined diffusion model to MSD data.
        
        Parameters
        ----------
        time_lags : np.ndarray
            Time lags
        msd_values : np.ndarray
            MSD values
        initial_guess : Optional[List[float]]
            Initial parameter guesses [D_free, L_conf]
            
        Returns
        -------
        Dict[str, Any]
            Fitting results
        """
        if len(time_lags) < 3:
            return {'success': False, 'error': 'Not enough data points for confined diffusion fitting.'}
        
        if initial_guess is None:
            # Estimate plateau value for L_conf
            L_conf_guess = np.sqrt(np.max(msd_values))
            # Estimate D_free from initial slope
            if len(time_lags) > 1:
                D_free_guess = (msd_values[1] - msd_values[0]) / (time_lags[1] - time_lags[0])
            else:
                D_free_guess = 0.1
            initial_guess = [max(D_free_guess, 1e-9), max(L_conf_guess, 1e-9)]
        
        bounds = ([0, 0], [np.inf, np.inf])
        
        try:
            params, covariance = curve_fit(
                self.confined_diffusion_model,
                time_lags,
                msd_values,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            fitted_msd = self.confined_diffusion_model(time_lags, *params)
            residuals = msd_values - fitted_msd
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            
            param_names = ['D_free', 'L_conf']
            fitted_params_dict = {name: val for name, val in zip(param_names, params)}
            
            param_errors = np.sqrt(np.diag(covariance)) if covariance is not None else [np.nan] * len(params)
            param_errors_dict = {name: err for name, err in zip(param_names, param_errors)}
            
            return {
                'success': True,
                'model_type': 'confined_diffusion',
                'params': fitted_params_dict,
                'param_errors': param_errors_dict,
                'r_squared': r_squared,
                'fitted_msd_values': fitted_msd.tolist(),
                'residuals': residuals.tolist()
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Confined diffusion fitting failed: {str(e)}'}
    
    def analyze_chromatin_dynamics(self, 
                                 time_lags: np.ndarray, 
                                 msd_values: np.ndarray,
                                 models_to_fit: List[str] = ['rouse', 'confined', 'anomalous']) -> Dict[str, Any]:
        """
        Comprehensive analysis of chromatin dynamics using multiple models.
        
        Parameters
        ----------
        time_lags : np.ndarray
            Time lags
        msd_values : np.ndarray
            MSD values
        models_to_fit : List[str]
            List of models to fit ['rouse', 'confined', 'anomalous', 'fbm']
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive fitting results for all models
        """
        results = {
            'success': True,
            'models_fitted': {},
            'best_fit_model': None,
            'model_comparison': {}
        }
        
        best_r_squared = -np.inf
        best_model = None
        
        # Fit Rouse model
        if 'rouse' in models_to_fit:
            rouse_result = self.fit_rouse_model_to_msd(time_lags, msd_values, fit_alpha_exponent=True)
            results['models_fitted']['rouse'] = rouse_result
            
            if rouse_result['success'] and rouse_result['r_squared'] > best_r_squared:
                best_r_squared = rouse_result['r_squared']
                best_model = 'rouse'
        
        # Fit confined diffusion model
        if 'confined' in models_to_fit:
            confined_result = self.fit_confined_diffusion_model(time_lags, msd_values)
            results['models_fitted']['confined'] = confined_result
            
            if confined_result['success'] and confined_result['r_squared'] > best_r_squared:
                best_r_squared = confined_result['r_squared']
                best_model = 'confined'
        
        # Fit anomalous diffusion model
        if 'anomalous' in models_to_fit:
            try:
                params, covariance = curve_fit(
                    self.anomalous_diffusion_model,
                    time_lags,
                    msd_values,
                    p0=[1.0, 0.5],
                    bounds=([0, 0.1], [np.inf, 2.0]),
                    maxfev=10000
                )
                
                fitted_msd = self.anomalous_diffusion_model(time_lags, *params)
                ss_res = np.sum((msd_values - fitted_msd)**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                
                anomalous_result = {
                    'success': True,
                    'model_type': 'anomalous',
                    'params': {'K_alpha': params[0], 'alpha': params[1]},
                    'param_errors': {'K_alpha': np.sqrt(covariance[0,0]), 'alpha': np.sqrt(covariance[1,1])},
                    'r_squared': r_squared,
                    'fitted_msd_values': fitted_msd.tolist()
                }
                
                results['models_fitted']['anomalous'] = anomalous_result
                
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_model = 'anomalous'
                    
            except Exception as e:
                results['models_fitted']['anomalous'] = {'success': False, 'error': str(e)}
        
        results['best_fit_model'] = best_model
        results['best_r_squared'] = best_r_squared
        
        # Create model comparison table
        comparison_data = []
        for model_name, model_result in results['models_fitted'].items():
            if model_result['success']:
                comparison_data.append({
                    'model': model_name,
                    'r_squared': model_result['r_squared'],
                    'num_parameters': len(model_result['params'])
                })
        
        if comparison_data:
            results['model_comparison'] = pd.DataFrame(comparison_data).sort_values('r_squared', ascending=False)
        
        return results


class EnergyLandscapeMapper:
    """
    Energy landscape analysis for particle tracking data using Boltzmann inversion.
    Maps spatial probability distributions to potential energy landscapes.
    """
    
    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0):
        self.tracks_df = tracks_df.copy()
        self.tracks_df['x_um'] = self.tracks_df['x'] * pixel_size
        self.tracks_df['y_um'] = self.tracks_df['y'] * pixel_size
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.results = {}

    def calculate_boltzmann_inversion(self, bins: int = 50, temperature: float = 300.0) -> Dict[str, Any]:
        """
        Estimate 2D potential energy landscape using Boltzmann inversion from particle positions.
        U(x,y) = -kT * ln(P(x,y))
        
        Parameters
        ----------
        bins : int
            Number of bins for spatial histogram
        temperature : float
            System temperature in Kelvin
            
        Returns
        -------
        Dict[str, Any]
            Results containing histogram, edges, and potential energy map
        """
        if self.tracks_df.empty:
            return {'success': False, 'error': 'Track data is empty.'}

        k_B = 1.380649e-23  # Boltzmann constant in J/K
        kT = k_B * temperature

        # Create 2D histogram of particle positions
        x_coords = self.tracks_df['x_um'].values
        y_coords = self.tracks_df['y_um'].values

        hist, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=bins, density=True)
        
        # Avoid log(0) by adding small epsilon
        hist_smooth = np.where(hist > 1e-9, hist, 1e-9) 
        
        # Calculate potential energy U(x,y) = -kT * ln(P(x,y))
        potential_energy_map = -kT * np.log(hist_smooth)
        
        # Normalize to have minimum energy at 0
        potential_energy_map -= np.min(potential_energy_map) 

        self.results['boltzmann_inversion'] = {
            'success': True,
            'histogram': hist,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'potential_energy_map': potential_energy_map,
            'units': 'Joules (relative to min)',
            'temperature_K': temperature,
            'bins': bins
        }
        return self.results['boltzmann_inversion']

    def analyze_dwell_regions(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, 
                              energy_threshold_factor: float = 0.5) -> Dict[str, Any]:
        """
        Identify significant dwell regions (potential wells) from the energy map.
        
        Parameters
        ----------
        potential_map : np.ndarray
            The 2D potential energy map
        x_edges, y_edges : np.ndarray
            Bin edges for the map
        energy_threshold_factor : float
            Factor to determine energy cutoff relative to max depth
            
        Returns
        -------
        Dict[str, Any]
            Information about identified dwell regions
        """
        try:
            from skimage import measure
        except ImportError:
            return {'success': False, 'error': 'scikit-image package required for region analysis'}

        if potential_map is None or potential_map.size == 0:
             return {'success': False, 'error': 'Potential map not provided or empty.'}

        # Invert map to find wells (low energy regions)
        inverted_energy_map = np.max(potential_map) - potential_map
        
        # Threshold to identify significant wells
        threshold = energy_threshold_factor * np.max(inverted_energy_map)
        binary_map = inverted_energy_map > threshold
        
        labeled_wells = measure.label(binary_map, connectivity=2)
        regions = measure.regionprops(labeled_wells, intensity_image=potential_map)
        
        dwell_regions_info = []
        for region in regions:
            min_val, max_val, mean_val = region.min_intensity, region.max_intensity, region.mean_intensity
            
            # Centroid in pixel/bin coordinates
            yc, xc = region.centroid 
            # Convert centroid to physical units
            centroid_x_um = x_edges[0] + (xc + 0.5) * (x_edges[1] - x_edges[0])
            centroid_y_um = y_edges[0] + (yc + 0.5) * (y_edges[1] - y_edges[0])
            
            area_pixels = region.area
            area_um2 = area_pixels * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])

            dwell_regions_info.append({
                'label': region.label,
                'centroid_x_um': centroid_x_um,
                'centroid_y_um': centroid_y_um,
                'area_pixels': area_pixels,
                'area_um2': area_um2,
                'min_potential_J': min_val,
                'mean_potential_J': mean_val,
                'bounding_box': region.bbox
            })
            
        self.results['dwell_regions'] = {
            'success': True,
            'regions': dwell_regions_info,
            'threshold_factor': energy_threshold_factor
        }
        return self.results['dwell_regions']

    def calculate_force_field(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> Dict[str, Any]:
        """
        Calculate the force field from the potential energy landscape.
        F = -∇U
        
        Parameters
        ----------
        potential_map : np.ndarray
            The 2D potential energy map
        x_edges, y_edges : np.ndarray
            Bin edges for the map
            
        Returns
        -------
        Dict[str, Any]
            Force field components Fx and Fy
        """
        if potential_map is None or potential_map.size == 0:
            return {'success': False, 'error': 'Potential map not provided or empty.'}

        # Calculate spatial derivatives
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        
        # Use numpy gradient for central differences
        grad_y, grad_x = np.gradient(potential_map, dy, dx)
        
        # Force is negative gradient
        Fx = -grad_x
        Fy = -grad_y
        
        # Calculate force magnitude
        force_magnitude = np.sqrt(Fx**2 + Fy**2)
        
        self.results['force_field'] = {
            'success': True,
            'Fx': Fx,
            'Fy': Fy,
            'force_magnitude': force_magnitude,
            'dx': dx,
            'dy': dy,
            'units': 'N (force per unit mass)'
        }
        return self.results['force_field']


class ActiveTransportAnalyzer:
    """
    Advanced analyzer for active transport detection and characterization.
    Integrates with biophysical models for comprehensive transport analysis.
    """
    
    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0):
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.results = {}
    
    def detect_directional_motion_segments(self, min_segment_length: int = 5, 
                                          straightness_threshold: float = 0.8,
                                          velocity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect segments of directional motion within tracks.
        
        Parameters
        ----------
        min_segment_length : int
            Minimum length of directional segments
        straightness_threshold : float
            Minimum straightness for directional motion
        velocity_threshold : float
            Minimum velocity for active transport (μm/s)
            
        Returns
        -------
        Dict[str, Any]
            Information about detected directional segments
        """
        try:
            from analysis import analyze_active_transport
        except ImportError:
            return {'success': False, 'error': 'Required module "analysis" not found. Please ensure it is installed and available.'}
        
        # Use existing active transport analysis
        transport_results = analyze_active_transport(
            self.tracks_df,
            min_track_length=min_segment_length,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval,
            straightness_threshold=straightness_threshold,
            min_segment_length=min_segment_length
        )
        
        if not transport_results.get('success', False):
            return {'success': False, 'error': 'Active transport analysis failed'}
        
        # Extract directional segments with velocity filtering
        segments = []
        track_results = transport_results.get('track_results', pd.DataFrame())
        
        # Vectorized filtering for tracks above velocity threshold
        fast_tracks = track_results[track_results.get('mean_speed', pd.Series(dtype=float)).fillna(0) >= velocity_threshold]
        
        # Vectorized segment creation - much faster than iterrows()
        if not fast_tracks.empty:
            segments = [
                {
                    'track_id': row.get('track_id'),
                    'mean_velocity': row.get('mean_speed', 0),
                    'straightness': row.get('straightness', 0),
                    'transport_type': 'active' if row.get('mean_speed', 0) > velocity_threshold else 'passive'
                }
                for _, row in fast_tracks.iterrows()
            ]
        else:
            segments = []
        
        self.results['directional_segments'] = {
            'success': True,
            'segments': segments,
            'total_segments': len(segments),
            'parameters': {
                'min_segment_length': min_segment_length,
                'straightness_threshold': straightness_threshold,
                'velocity_threshold': velocity_threshold
            }
        }
        return self.results['directional_segments']
    
    def characterize_transport_modes(self) -> Dict[str, Any]:
        """
        Characterize different modes of transport in the data.
        
        Returns
        -------
        Dict[str, Any]
            Classification of transport modes
        """
        if 'directional_segments' not in self.results:
            return {'success': False, 'error': 'Run detect_directional_motion_segments first'}
        
        segments = self.results['directional_segments']['segments']
        
        if not segments:
            return {'success': False, 'error': 'No directional segments found'}
        
        # Classify transport modes
        velocities = [s['mean_velocity'] for s in segments]
        straightness_values = [s['straightness'] for s in segments]
        
        # Define thresholds for classification
        slow_threshold = 0.1  # μm/s
        fast_threshold = 0.5  # μm/s
        high_straightness = 0.8
        
        transport_modes = {
            'diffusive': 0,
            'slow_directed': 0,
            'fast_directed': 0,
            'mixed': 0
        }
        
        for velocity, straightness in zip(velocities, straightness_values):
            if velocity < slow_threshold:
                transport_modes['diffusive'] += 1
            elif velocity < fast_threshold and straightness > high_straightness:
                transport_modes['slow_directed'] += 1
            elif velocity >= fast_threshold and straightness > high_straightness:
                transport_modes['fast_directed'] += 1
            else:
                transport_modes['mixed'] += 1
        
        total_segments = len(segments)
        transport_fractions = {mode: count/total_segments for mode, count in transport_modes.items()}
        
        self.results['transport_modes'] = {
            'success': True,
            'mode_counts': transport_modes,
            'mode_fractions': transport_fractions,
            'total_analyzed': total_segments,
            'mean_velocity': np.mean(velocities),
            'mean_straightness': np.mean(straightness_values)
        }
        return self.results['transport_modes']

def analyze_motion_models(tracks_df, models=None, min_track_length=10, time_window=None):
    """
    Analyze track data using different motion models to classify motion types.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with columns 'track_id', 'frame', 'x', 'y'
    models : list, optional
        List of motion models to test. Default is ['brownian', 'directed', 'confined']
    min_track_length : int, optional
        Minimum track length required for analysis, by default 10
    time_window : int, optional
        Number of frames to use for local analysis, by default None (use entire track)
    
    Returns
    -------
    dict
        Dictionary containing model fit results and classifications
    """
    if models is None:
        models = ['brownian', 'directed', 'confined']
    
    # Group tracks by track_id
    grouped = tracks_df.groupby('track_id')
    
    # Filter for minimum track length
    long_enough = [tid for tid, group in grouped if len(group) >= min_track_length]
    
    results = {
        'track_ids': long_enough,
        'models': models,
        'classifications': {},
        'model_parameters': {},
        'best_model': {},
        'error_metrics': {},
    }
    
    for track_id in long_enough:
        track_data = grouped.get_group(track_id).sort_values('frame')
        
        # Extract positions and frames
        positions = track_data[['x', 'y']].values
        frames = track_data['frame'].values
        
        # Handle time_window if specified
        if time_window and len(positions) > time_window:
            model_fits = {}
            model_errors = {}
            
            # Analyze sliding windows
            for i in range(len(positions) - time_window + 1):
                window_pos = positions[i:i+time_window]
                window_frames = frames[i:i+time_window]
                
                window_fits, window_errors = _fit_motion_models(window_pos, window_frames, models)
                
                for model, fit in window_fits.items():
                    if model not in model_fits:
                        model_fits[model] = []
                    model_fits[model].append(fit)
                
                for model, error in window_errors.items():
                    if model not in model_errors:
                        model_errors[model] = []
                    model_errors[model].append(error)
            
            # Aggregate window results
            best_model, best_score = _determine_best_model(model_errors)
            
            # Store results
            results['model_parameters'][track_id] = {
                model: {param: np.mean([fit[param] for fit in fits]) 
                       for param in fits[0]}
                for model, fits in model_fits.items()
            }
            
            results['error_metrics'][track_id] = {
                model: np.mean(errors) for model, errors in model_errors.items()
            }
            
        else:
            # Analyze whole track
            model_fits, model_errors = _fit_motion_models(positions, frames, models)
            best_model, best_score = _determine_best_model(model_errors)
            
            results['model_parameters'][track_id] = model_fits
            results['error_metrics'][track_id] = model_errors
        
        results['best_model'][track_id] = best_model
        results['classifications'][track_id] = best_model
    
    # Generate summary statistics
    results['summary'] = _summarize_motion_analysis(results)
    
    return results

def _fit_motion_models(positions, frames, models):
    """
    Fit different motion models to position data.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of x,y positions
    frames : np.ndarray
        Array of frame numbers
    models : list
        List of motion model names to fit
    
    Returns
    -------
    tuple
        (model_fits, model_errors) dictionaries
    """
    model_fits = {}
    model_errors = {}
    
    # Calculate displacements and time intervals
    displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    dt = np.diff(frames)
    
    # Brownian motion model
    if 'brownian' in models:
        D, brownian_error = _fit_brownian_motion(displacements, dt)
        model_fits['brownian'] = {'D': D}
        model_errors['brownian'] = brownian_error
    
    # Directed motion model
    if 'directed' in models:
        D, v, directed_error = _fit_directed_motion(displacements, dt, positions)
        model_fits['directed'] = {'D': D, 'v': v}
        model_errors['directed'] = directed_error
    
    # Confined motion model
    if 'confined' in models:
        D, L, confined_error = _fit_confined_motion(displacements, dt, positions)
        model_fits['confined'] = {'D': D, 'L': L}
        model_errors['confined'] = confined_error
    
    return model_fits, model_errors

def _fit_brownian_motion(displacements, dt):
    """Fit Brownian motion model to displacements."""
    # For pure Brownian motion: MSD = 4*D*t
    # Estimate diffusion coefficient
    msd = np.mean(displacements**2)
    mean_dt = np.mean(dt)
    D = msd / (4 * mean_dt)
    
    # Calculate error as residual between observed and expected displacements
    expected_disp = np.sqrt(4 * D * dt)
    error = np.mean((displacements - expected_disp)**2)
    
    return D, error

def _fit_directed_motion(displacements, dt, positions):
    """Fit directed motion model to displacements and positions."""
    # For directed motion: MSD = 4*D*t + (v*t)^2
    
    # First, estimate direction using linear regression
    t = np.cumsum(np.concatenate([[0], dt]))
    x = positions[:, 0]
    y = positions[:, 1]
    
    vx, _ = np.polyfit(t, x, 1)
    vy, _ = np.polyfit(t, y, 1)
    
    v = np.sqrt(vx**2 + vy**2)
    
    # Now estimate diffusion coefficient
    msd = np.mean(displacements**2)
    mean_dt = np.mean(dt)
    
    # MSD = 4*D*t + (v*t)^2
    # Solve for D
    D = (msd - (v * mean_dt)**2) / (4 * mean_dt)
    D = max(0, D)  # Ensure non-negative diffusion coefficient
    
    # Calculate error
    expected_disp = np.sqrt(4 * D * dt + (v * dt)**2)
    error = np.mean((displacements - expected_disp)**2)
    
    return D, v, error

def _fit_confined_motion(displacements, dt, positions):
    """Fit confined motion model to displacements and positions."""
    # For confined motion: MSD = L^2 * [1 - exp(-4*D*t/L^2)]
    
    # Initial estimates
    msd = np.mean(displacements**2)
    mean_dt = np.mean(dt)
    D_initial = msd / (4 * mean_dt)
    
    # Estimate confinement size as 2x standard deviation of positions
    std_pos = np.std(positions, axis=0)
    L_initial = 2 * np.mean(std_pos)
    
    # Define confined motion model function
    def confined_msd(t, D, L):
        return L**2 * (1 - np.exp(-4 * D * t / L**2))
    
    # Function to minimize
    def objective(params):
        D, L = params
        expected_msd = confined_msd(dt, D, L)
        return np.mean((displacements**2 - expected_msd)**2)
    
    # Optimize parameters
    from scipy.optimize import minimize
    
    bounds = [(0, None), (0, None)]  # D and L must be positive
    result = minimize(objective, [D_initial, L_initial], bounds=bounds)
    
    D, L = result.x
    error = result.fun
    
    return D, L, error

def _determine_best_model(model_errors):
    """Determine best model based on error metrics."""
    # Simple approach: choose model with lowest error
    best_model = min(model_errors, key=model_errors.get)
    best_score = model_errors[best_model]
    
    # Apply a bias to favor simpler models
    # If Brownian motion is within 10% of best model, prefer it
    if 'brownian' in model_errors and best_model != 'brownian':
        if model_errors['brownian'] < 1.1 * best_score:
            best_model = 'brownian'
            best_score = model_errors['brownian']
    
    return best_model, best_score

def _summarize_motion_analysis(results):
    """Generate summary statistics for motion analysis."""
    summary = {
        'total_tracks': len(results['track_ids']),
        'model_counts': {},
        'model_parameters': {}
    }
    
    # Count occurrences of each model
    for model in results['best_model'].values():
        if model not in summary['model_counts']:
            summary['model_counts'][model] = 0
        summary['model_counts'][model] += 1
    
    # Calculate average parameters for each model
    for model in results['models']:
        # Collect parameters for this model from all tracks
        params = {}
        for track_id, model_params in results['model_parameters'].items():
            if model in model_params:
                for param, value in model_params[model].items():
                    if param not in params:
                        params[param] = []
                    params[param].append(value)
        
        # Calculate averages
        if params:
            summary['model_parameters'][model] = {
                param: np.mean(values) for param, values in params.items()
            }
    
    # Calculate fractions
    for model, count in summary['model_counts'].items():
        summary['model_counts'][f"{model}_fraction"] = count / summary['total_tracks']
    
    return summary
