"""
Advanced biophysical models for single particle tracking analysis.
Specialized for nucleosome diffusion in chromatin and polymer physics modeling.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from typing import Any, Dict, List, Tuple, Optional
try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # fallback so type hints referencing go.Figure don't raise NameError

# Import data access utilities
try:
    from data_access_utils import get_track_data, check_data_availability, get_units, display_data_summary
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

def show_biophysical_models():
    """Main interface for biophysical models."""
    st.title("🧬 Biophysical Models")
    
    # Check for data availability
    if DATA_UTILS_AVAILABLE:
        if not check_data_availability():
            return
        tracks_df, _ = get_track_data()
        units = get_units()
    else:
        # Fallback to direct access
        tracks_df = st.session_state.get('tracks_df') or st.session_state.get('raw_tracks')
        if tracks_df is None or tracks_df.empty:
            st.error("No track data loaded. Please load data first.")
            return
        units = {
            'pixel_size': st.session_state.get('pixel_size', 0.1),
            'frame_interval': st.session_state.get('frame_interval', 0.1)
        }
    
    # Display data summary
    if DATA_UTILS_AVAILABLE:
        display_data_summary()
    
    # ...existing code...
    # Continue with the rest of the biophysical models logic using tracks_df

class PolymerPhysicsModel:
    """Polymer physics model implementation."""
    
    def __init__(self, msd_data=None, pixel_size=1.0, frame_interval=0.1, lag_units='frames'):
        self.msd_data = msd_data
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.lag_units = lag_units

    def fit_rouse_model(self, fit_alpha=False):
        """
        Fits the Rouse model to the provided MSD data.
        The Rouse model predicts MSD ~ t^0.5.
        """
        if self.msd_data is None or self.msd_data.empty:
            return {'success': False, 'error': 'MSD data not available'}

        lag_time = self.msd_data['lag_time'].values
        msd = self.msd_data['msd'].values

        # Avoid log(0) issues by filtering out non-positive values
        valid_indices = (lag_time > 0) & (msd > 0)
        if not np.any(valid_indices):
            return {'success': False, 'error': 'No positive lag time and MSD data available for fitting.'}

        lag_time = lag_time[valid_indices]
        msd = msd[valid_indices]

        log_lag_time = np.log(lag_time)
        log_msd = np.log(msd)

        params = {}

        if fit_alpha:
            # Fit for both alpha and K: log(MSD) = alpha * log(t) + log(K)
            try:
                # A robust way to handle potential issues with polyfit
                if len(log_lag_time) < 2:
                    return {'success': False, 'error': 'Not enough data points to fit model.'}
                p = np.polyfit(log_lag_time, log_msd, 1)
                alpha = p[0]
                log_K = p[1]
                K = np.exp(log_K)
                params['alpha'] = alpha
                params['K_rouse'] = K
            except np.linalg.LinAlgError as e:
                return {'success': False, 'error': f'Failed to fit model: {e}'}

        else:
            # Fixed alpha = 0.5
            alpha = 0.5
            # MSD = K * t^0.5  => K = MSD / t^0.5
            # We can calculate K for each point and take the average for a simple estimate
            with np.errstate(divide='ignore', invalid='ignore'):
                K_values = msd / (lag_time ** alpha)

            # Use nanmean to ignore potential NaNs or Infs from division issues
            K = np.nanmean(K_values[np.isfinite(K_values)])

            if np.isnan(K):
                 return {'success': False, 'error': 'Could not determine a valid K_rouse parameter.'}

            params['alpha'] = alpha
            params['K_rouse'] = K

        return {
            'success': True,
            'parameters': params
        }

    def fit_zimm_model(self, fit_alpha=False, solvent_viscosity=0.001, 
                       hydrodynamic_radius=5e-9, temperature=300.0):
        """
        Fits the Zimm model to the provided MSD data.
        The Zimm model includes hydrodynamic interactions: MSD ~ t^(2/3).
        
        Parameters
        ----------
        fit_alpha : bool
            If True, fit alpha exponent. If False, use theoretical value (0.667)
        solvent_viscosity : float
            Solvent viscosity in Pa·s (default: 0.001 for water at 25°C)
        hydrodynamic_radius : float
            Hydrodynamic radius in meters (default: 5 nm)
        temperature : float
            Temperature in Kelvin (default: 300 K)
            
        Returns
        -------
        dict
            Results with success status, parameters, and fitted curve
        """
        if self.msd_data is None or self.msd_data.empty:
            return {'success': False, 'error': 'MSD data not available'}

        lag_time = self.msd_data['lag_time'].values
        msd = self.msd_data['msd'].values

        # Filter valid data
        valid_indices = (lag_time > 0) & (msd > 0)
        if not np.any(valid_indices):
            return {'success': False, 'error': 'No positive lag time and MSD data available for fitting.'}

        lag_time = lag_time[valid_indices]
        msd = msd[valid_indices]

        log_lag_time = np.log(lag_time)
        log_msd = np.log(msd)

        params = {}
        kB = 1.38e-23  # Boltzmann constant (J/K)

        if fit_alpha:
            # Fit for both alpha and K
            try:
                if len(log_lag_time) < 2:
                    return {'success': False, 'error': 'Not enough data points to fit model.'}
                p = np.polyfit(log_lag_time, log_msd, 1)
                alpha = p[0]
                log_K = p[1]
                K = np.exp(log_K)
                params['alpha'] = alpha
                params['K_zimm'] = K
            except np.linalg.LinAlgError as e:
                return {'success': False, 'error': f'Failed to fit model: {e}'}
        else:
            # Theoretical Zimm exponent
            alpha = 2.0/3.0
            
            # Calculate K from hydrodynamic theory
            # MSD = (kB*T)/(3*pi*eta*Rh) * t^(2/3) * prefactor
            with np.errstate(divide='ignore', invalid='ignore'):
                K_values = msd / (lag_time ** alpha)
            
            K = np.nanmean(K_values[np.isfinite(K_values)])
            
            if np.isnan(K):
                return {'success': False, 'error': 'Could not determine a valid K_zimm parameter.'}
            
            params['alpha'] = alpha
            params['K_zimm'] = K
            
            # Theoretical diffusion coefficient
            D_zimm = (kB * temperature) / (6 * np.pi * solvent_viscosity * hydrodynamic_radius)
            params['D_zimm_theory'] = D_zimm

        # Calculate fitted curve
        fitted_curve = {
            'lag_time': lag_time,
            'msd_fit': params.get('K_zimm', 0) * (lag_time ** params['alpha'])
        }

        return {
            'success': True,
            'parameters': params,
            'fitted_curve': fitted_curve,
            'model': 'Zimm'
        }

    def fit_reptation_model(self, temperature=300.0, tube_diameter=100e-9, 
                           contour_length=1000e-9):
        """
        Fits the reptation model (de Gennes) to the provided MSD data.
        Reptation describes polymer motion in entangled networks.
        
        Early time: MSD ~ t^0.25 (Rouse-like within tube)
        Late time: MSD ~ t^0.5 (tube escape, reptation)
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin (default: 300 K)
        tube_diameter : float
            Tube diameter in meters (default: 100 nm)
        contour_length : float
            Polymer contour length in meters (default: 1000 nm)
            
        Returns
        -------
        dict
            Results with success status, parameters, and regime classification
        """
        if self.msd_data is None or self.msd_data.empty:
            return {'success': False, 'error': 'MSD data not available'}

        lag_time = self.msd_data['lag_time'].values
        msd = self.msd_data['msd'].values

        # Filter valid data
        valid_indices = (lag_time > 0) & (msd > 0)
        if not np.any(valid_indices):
            return {'success': False, 'error': 'No positive lag time and MSD data available for fitting.'}

        lag_time = lag_time[valid_indices]
        msd = msd[valid_indices]

        # Fit log-log to determine exponent
        log_lag_time = np.log(lag_time)
        log_msd = np.log(msd)

        try:
            if len(log_lag_time) < 2:
                return {'success': False, 'error': 'Not enough data points to fit model.'}
            
            # Fit overall exponent
            p = np.polyfit(log_lag_time, log_msd, 1)
            alpha = p[0]
            log_K = p[1]
            K = np.exp(log_K)
            
            # Classify regime based on alpha
            if alpha < 0.35:
                regime = "Early reptation (Rouse-like, confined)"
                regime_phase = "early"
            elif alpha < 0.6:
                regime = "Transition to tube escape"
                regime_phase = "transition"
            else:
                regime = "Late reptation (tube escape)"
                regime_phase = "late"
            
            # Estimate tube parameters from MSD
            # Tube diameter ≈ sqrt(MSD_plateau) (confinement)
            # Find plateau if present (where d(log MSD)/d(log t) is minimal)
            if len(msd) > 5:
                # Look for minimal slope region
                window = 3
                slopes = []
                for i in range(window, len(log_lag_time) - window):
                    local_slope = np.polyfit(
                        log_lag_time[i-window:i+window],
                        log_msd[i-window:i+window],
                        1
                    )[0]
                    slopes.append(local_slope)
                
                if slopes:
                    min_slope_idx = np.argmin(slopes) + window
                    tube_diameter_estimated = np.sqrt(msd[min_slope_idx])
                else:
                    tube_diameter_estimated = tube_diameter
            else:
                tube_diameter_estimated = tube_diameter
            
            # Reptation time (tau_rep)
            kB = 1.38e-23  # Boltzmann constant
            # tau_rep ~ L^3 / (d^2 * D_local) where L is contour length, d is tube diameter
            # Rough estimate
            D_local = K / 4  # Approximate local diffusion coefficient
            if D_local > 0 and tube_diameter > 0:
                tau_rep = (contour_length**3) / (tube_diameter**2 * D_local)
            else:
                tau_rep = None
            
            params = {
                'alpha': alpha,
                'K_reptation': K,
                'regime': regime,
                'regime_phase': regime_phase,
                'tube_diameter_estimated': tube_diameter_estimated,
                'tube_diameter_input': tube_diameter,
                'contour_length': contour_length,
                'reptation_time': tau_rep
            }
            
            # Calculate fitted curve
            fitted_curve = {
                'lag_time': lag_time,
                'msd_fit': K * (lag_time ** alpha)
            }
            
            return {
                'success': True,
                'parameters': params,
                'fitted_curve': fitted_curve,
                'model': 'Reptation',
                'interpretation': f"{regime} (α={alpha:.3f})"
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Reptation fit failed: {str(e)}'}

    def analyze_fractal_dimension(self):
        """
        Calculate fractal dimension of MSD trajectory using box-counting method.
        
        Fractal dimension (Df) characterizes the space-filling property:
        - Df = 1: Straight line (ballistic)
        - Df ≈ 1.5: Anomalous diffusion
        - Df = 2: Space-filling (Brownian motion)
        
        Returns:
            dict: Contains success status, fractal dimension value, and interpretation
        """
        if self.msd_data is None or self.msd_data.empty:
            return {'success': False, 'error': 'MSD data not available for fractal dimension analysis'}
        
        try:
            # Extract lag time and MSD values
            lag_time = self.msd_data['lag_time'].values
            msd = self.msd_data['msd'].values
            
            # Filter valid data
            valid_indices = (lag_time > 0) & (msd > 0) & np.isfinite(lag_time) & np.isfinite(msd)
            if not np.any(valid_indices):
                return {'success': False, 'error': 'No valid data for fractal dimension analysis'}
            
            lag_time = lag_time[valid_indices]
            msd = msd[valid_indices]
            
            if len(lag_time) < 5:
                return {'success': False, 'error': 'Insufficient data points for fractal dimension calculation'}
            
            # Use box-counting method on log-log space
            # For MSD data, we analyze the trajectory in (log(t), log(MSD)) space
            log_t = np.log(lag_time)
            log_msd = np.log(msd)
            
            # Calculate bounding box
            t_min, t_max = np.min(log_t), np.max(log_t)
            msd_min, msd_max = np.min(log_msd), np.max(log_msd)
            
            range_t = t_max - t_min
            range_msd = msd_max - msd_min
            max_range = max(range_t, range_msd)
            
            if max_range == 0:
                return {'success': False, 'error': 'MSD data has no variation'}
            
            # Use logarithmically spaced scales
            n_scales = 10
            scales = np.logspace(np.log10(max_range/50), np.log10(max_range/2), n_scales)
            
            counts = []
            valid_scales = []
            
            for scale in scales:
                # Assign each point to a box
                box_indices = set()
                for lt, lm in zip(log_t, log_msd):
                    ix = int((lt - t_min) / scale)
                    iy = int((lm - msd_min) / scale)
                    box_indices.add((ix, iy))
                
                n_boxes = len(box_indices)
                if n_boxes > 0:
                    counts.append(n_boxes)
                    valid_scales.append(scale)
            
            if len(counts) < 3:
                return {'success': False, 'error': 'Insufficient scale range for fractal dimension calculation'}
            
            # Fit log(N) = -Df * log(ε) + c
            
        except Exception as e:
            return {'success': False, 'error': f'Fractal dimension calculation failed: {str(e)}'}

    def analyze_polymer_dynamics(self, tracks_df=None):
        """Analyze polymer dynamics from tracking data."""
        # Get data if not provided
        if tracks_df is None:
            if DATA_UTILS_AVAILABLE:
                tracks_df, has_data = get_track_data()
                if not has_data:
                    return {'error': 'No track data available', 'success': False}
            else:
                tracks_df = st.session_state.get('tracks_df') or st.session_state.get('raw_tracks')
                if tracks_df is None or tracks_df.empty:
                    return {'error': 'No track data available', 'success': False}
        
        # ...existing code...
        # Continue with analysis using tracks_df
    
    def correct_for_crowding(
        self,
        D_measured: float,
        phi_crowding: float = 0.3,
        obstacle_shape: str = 'spherical'
    ) -> Dict:
        """
        Correct measured diffusion coefficient for macromolecular crowding effects.
        
        The nucleus is a crowded environment with volume fraction φ ≈ 0.2-0.4.
        Crowding reduces effective diffusion coefficient.
        
        Parameters:
        -----------
        D_measured : float
            Measured diffusion coefficient (μm²/s) in crowded environment
        phi_crowding : float
            Volume fraction occupied by obstacles (0-1)
            Typical nuclear values: 0.2-0.4
        obstacle_shape : str
            'spherical' or 'rod-like' (affects scaling)
        
        Returns:
        --------
        dict with keys:
            - 'D_free': float (diffusion in dilute solution)
            - 'D_measured': float (input value)
            - 'crowding_factor': float (D_measured / D_free)
            - 'phi_crowding': float
            - 'effective_viscosity_ratio': float
        
        Reference:
        - Minton (2001): "The Influence of Macromolecular Crowding"
        """
        # Scaling factor depends on obstacle shape
        if obstacle_shape == 'spherical':
            alpha = 1.5  # Hard spheres
        elif obstacle_shape == 'rod-like':
            alpha = 2.0  # Rod-like obstacles
        else:
            alpha = 1.5  # Default
        
        # Scaled particle theory: D_eff = D_free * exp(-α*φ)
        D_free = D_measured / np.exp(-alpha * phi_crowding)
        
        crowding_factor = D_measured / D_free
        
        # Effective viscosity ratio (Stokes-Einstein: D ~ 1/η)
        viscosity_ratio = D_free / D_measured
        
        # Interpretation
        if phi_crowding < 0.2:
            crowding_level = "Low crowding"
        elif phi_crowding < 0.35:
            crowding_level = "Moderate crowding (typical nuclear)"
        else:
            crowding_level = "High crowding"
        
        return {
            'success': True,
            'D_free': D_free,
            'D_measured': D_measured,
            'crowding_factor': crowding_factor,
            'phi_crowding': phi_crowding,
            'effective_viscosity_ratio': viscosity_ratio,
            'crowding_level': crowding_level,
            'interpretation': f"{crowding_level}: Diffusion reduced to {crowding_factor*100:.1f}% of free value"
        }
    
    def calculate_local_diffusion_map(
        self,
        tracks_df: pd.DataFrame,
        grid_resolution: int = 20,
        window_size: int = 5,
        min_points: int = 5
    ) -> Dict:
        """
        Calculate spatially-resolved diffusion coefficient D(x,y).
        
        Divides space into grid and calculates local D from tracks passing
        through each cell.
        
        Parameters:
        -----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
        grid_resolution : int
            Number of grid cells in each direction
        window_size : int
            Number of frames for local MSD calculation
        min_points : int
            Minimum number of displacements required per cell
        
        Returns:
        --------
        dict with keys:
            - 'D_map': 2D array of D values (μm²/s)
            - 'x_coords': array of x coordinates (μm)
            - 'y_coords': array of y coordinates (μm)
            - 'confidence_map': R² values for each fit
            - 'count_map': Number of measurements per cell
        """
        # Convert to physical units
        tracks_df = tracks_df.copy()
        tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
        tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
        tracks_df['time_s'] = tracks_df['frame'] * self.frame_interval
        
        # Define grid
        x_min, x_max = tracks_df['x_um'].min(), tracks_df['x_um'].max()
        y_min, y_max = tracks_df['y_um'].min(), tracks_df['y_um'].max()
        
        x_edges = np.linspace(x_min, x_max, grid_resolution + 1)
        y_edges = np.linspace(y_min, y_max, grid_resolution + 1)
        
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # Initialize output maps
        D_map = np.full((grid_resolution, grid_resolution), np.nan)
        confidence_map = np.full((grid_resolution, grid_resolution), np.nan)
        count_map = np.zeros((grid_resolution, grid_resolution), dtype=int)
        
        # Calculate local D for each grid cell
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Find tracks in this cell
                in_cell = (
                    (tracks_df['x_um'] >= x_edges[i]) &
                    (tracks_df['x_um'] < x_edges[i+1]) &
                    (tracks_df['y_um'] >= y_edges[j]) &
                    (tracks_df['y_um'] < y_edges[j+1])
                )
                
                cell_tracks = tracks_df[in_cell]
                
                if len(cell_tracks) < min_points:
                    continue
                
                # Calculate local displacements
                displacements_squared = []
                time_lags = []
                
                for track_id in cell_tracks['track_id'].unique():
                    track = cell_tracks[cell_tracks['track_id'] == track_id].sort_values('frame')
                    
                    if len(track) < 2:
                        continue
                    
                    # Calculate displacements up to window_size
                    for lag in range(1, min(window_size + 1, len(track))):
                        dx = track['x_um'].values[lag:] - track['x_um'].values[:-lag]
                        dy = track['y_um'].values[lag:] - track['y_um'].values[:-lag]
                        dt = track['time_s'].values[lag:] - track['time_s'].values[:-lag]
                        
                        r_squared = dx**2 + dy**2
                        
                        displacements_squared.extend(r_squared)
                        time_lags.extend(dt)
                
                if len(displacements_squared) < min_points:
                    continue
                
                count_map[j, i] = len(displacements_squared)
                
                # Fit D: <r²> = 4*D*t (2D)
                try:
                    from scipy.stats import linregress
                    
                    time_lags = np.array(time_lags)
                    displacements_squared = np.array(displacements_squared)
                    
                    # Remove outliers (optional)
                    valid = displacements_squared < np.percentile(displacements_squared, 95)
                    
                    if np.sum(valid) < min_points:
                        continue
                    
                    slope, intercept, r_value, p_value, std_err = linregress(
                        time_lags[valid],
                        displacements_squared[valid]
                    )
                    
                    D_local = slope / 4.0  # 2D: <r²> = 4*D*t
                    
                    # Store results
                    D_map[j, i] = D_local
                    confidence_map[j, i] = r_value**2
                    
                except Exception as e:
                    continue
        
        return {
            'success': True,
            'D_map': D_map,
            'x_coords': x_centers,
            'y_coords': y_centers,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'confidence_map': confidence_map,
            'count_map': count_map,
            'grid_resolution': grid_resolution
        }

class EnergyLandscapeMapper:
    """
    Class for mapping energy landscapes from particle trajectories.
    """

    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 0.1, temperature: float = 300.0):
        """
        Initialize the energy landscape mapper.

        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns 'track_id', 'frame', 'x', 'y'
        pixel_size : float
            Pixel size in micrometers
        temperature : float
            Temperature in Kelvin
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        self.temperature = temperature
        self.kB = 1.38e-23  # Boltzmann constant, J/K
        self.kBT = self.kB * self.temperature  # Thermal energy
        self.results = {}

        # Convert coordinates to micrometers if needed
        for col in ['x', 'y']:
            if col in self.tracks_df.columns:
                self.tracks_df[f'{col}_um'] = self.tracks_df[col] * self.pixel_size

    def map_energy_landscape(self, resolution: int = 20, method: str = "boltzmann", 
                           smoothing: float = 0.5, normalize: bool = True) -> Dict[str, Any]:
        """
        Map energy landscape from particle positions.

        Parameters
        ----------
        resolution : int
            Grid resolution for energy landscape
        method : str
            Method for energy calculation ('boltzmann', 'drift', 'kramers')
        smoothing : float
            Smoothing factor for the landscape
        normalize : bool
            Whether to normalize energies to kBT units

        Returns
        -------
        Dict[str, Any]
            Energy landscape mapping results
        """
        # Extract position data
        x = self.tracks_df['x_um'].values
        y = self.tracks_df['y_um'].values

        try:
            # Determine the spatial extent
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # Create grid for energy landscape
            x_edges = np.linspace(x_min, x_max, resolution + 1)
            y_edges = np.linspace(y_min, y_max, resolution + 1)

            # Calculate position histogram (particle density)
            H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

            # Add smoothing if requested
            if smoothing > 0:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=smoothing)

            # Small constant to avoid log(0)
            epsilon = H.max() * 1e-6
            H[H < epsilon] = epsilon

            if normalize:
                U = U  # Already normalized in kBT units by using log
                energy_units = "kBT"
            else:
                U = U * self.kBT  # Convert to actual energy in Joules
                energy_units = "J"

            # Calculate force field from the energy landscape
            force_field = self.calculate_force_field(U, x_edges, y_edges)

            # Create visualization
            fig = self.visualize_energy_landscape(U, x_edges, y_edges, force_field, energy_units)

            # Analyze dwell regions
            dwell_regions = self.analyze_dwell_regions(U, x_edges, y_edges)

            # Calculate statistics
            min_energy = np.min(U)
            max_energy = np.max(U)
            energy_range = max_energy - min_energy

            # Store results
            results = {
                'success': True,
                'energy_map': U,
                'energy_landscape': U,  # Keep for backward compatibility
                'x_coords': (x_edges[:-1] + x_edges[1:]) / 2,  # Bin centers
                'y_coords': (y_edges[:-1] + y_edges[1:]) / 2,  # Bin centers
                'x_edges': x_edges,
                'y_edges': y_edges,
                'force_field': force_field,
                'dwell_regions': dwell_regions,
                'statistics': {
                    'min_energy': min_energy,
                    'max_energy': max_energy,
                    'energy_range': energy_range,
                    'num_dwell_regions': len(dwell_regions) if dwell_regions else 0
                },
                'parameters': {
                    'method': method,
                    'resolution': resolution,
                    'smoothing': smoothing,
                    'normalize': normalize,
                    'energy_units': energy_units
                },
                'visualization': fig
            }

        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }

        # Store results and return
        self.results['energy_landscape'] = results
        return results

    def analyze_dwell_regions(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, 
                              energy_threshold_factor: float = 0.5) -> Dict[str, Any]:
        """
        Identify and analyze potential dwell regions (energy minima).

        Parameters
        ----------
        potential_map : np.ndarray
            Energy landscape
        x_edges : np.ndarray
            X-axis bin edges
        y_edges : np.ndarray
            Y-axis bin edges
        energy_threshold_factor : float
            Energy threshold factor for identifying wells

        Returns
        -------
        Dict[str, Any]
            Dwell region analysis results
        """
        try:
            # Find local minima in the potential map
            from scipy.ndimage import minimum_filter

            # Apply minimum filter to find local minima
            min_filtered = minimum_filter(potential_map, size=3)
            minima = (potential_map == min_filtered) & (potential_map < potential_map.mean())

            # Label connected regions of minima
            from scipy.ndimage import label
            labeled_minima, num_minima = label(minima)

            # Calculate properties of each minimum
            minima_properties = []

            for i in range(1, num_minima + 1):
                # Get coordinates of this minimum
                y_indices, x_indices = np.where(labeled_minima == i)

                if len(y_indices) > 0:
                    # Calculate centroid
                    y_center = y_indices.mean()
                    x_center = x_indices.mean()

                    # Convert to physical coordinates
                    x_pos = np.interp(x_center, np.arange(len(x_edges) - 1), x_edges[:-1])
                    y_pos = np.interp(y_center, np.arange(len(y_edges) - 1), y_edges[:-1])

                    # Get the energy value at this minimum
                    energy_value = potential_map[y_indices[0], x_indices[0]]

                    # Calculate size (area) of the minimum region
                    size = len(y_indices)

                    # Add to properties list
                    minima_properties.append({
                        'id': i,
                        'x_position': x_pos,
                        'y_position': y_pos,
                        'energy': energy_value,
                        'size': size
                    })

            # Calculate energy barriers between minima (if multiple minima exist)
            energy_barriers = []

            if len(minima_properties) > 1:
                for i in range(len(minima_properties)):
                    for j in range(i + 1, len(minima_properties)):
                        min1 = minima_properties[i]
                        min2 = minima_properties[j]

                        # Calculate the straight-line path between the two minima
                        num_steps = 20
                        x_path = np.linspace(min1['x_position'], min2['x_position'], num_steps)
                        y_path = np.linspace(min1['y_position'], min2['y_position'], num_steps)

                        # Calculate energy along this path
                        path_energies = []
                        for x, y in zip(x_path, y_path):
                            # Convert positions to indices
                            x_idx = np.interp(x, x_edges[:-1], np.arange(len(x_edges) - 1))
                            y_idx = np.interp(y, y_edges[:-1], np.arange(len(y_edges) - 1))

                            # Ensure indices are within bounds
                            x_idx = int(min(max(0, x_idx), potential_map.shape[1] - 1))
                            y_idx = int(min(max(0, y_idx), potential_map.shape[0] - 1))

                            # Get energy at this point
                            energy = potential_map[y_idx, x_idx]
                            path_energies.append(energy)

                        # Calculate barrier height
                        energy_min = min(min1['energy'], min2['energy'])
                        barrier_height = max(path_energies) - energy_min

                        energy_barriers.append({
                            'from_id': min1['id'],
                            'to_id': min2['id'],
                            'barrier_height': barrier_height,
                            'distance': np.sqrt((min1['x_position'] - min2['x_position'])**2 + 
                                              (min1['y_position'] - min2['y_position'])**2)
                        })

            # Return dwell region analysis
            return {
                'num_minima': num_minima,
                'minima_properties': minima_properties,
                'energy_barriers': energy_barriers
            }

        except Exception as e:
            return {
                'error': str(e),
                'num_minima': 0,
                'minima_properties': [],
                'energy_barriers': []
            }

    def calculate_force_field(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> Dict[str, Any]:
        """
        Calculate force field from energy landscape gradient.

        Parameters
        ----------
        potential_map : np.ndarray
            Energy landscape
        x_edges : np.ndarray
            X-axis bin edges
        y_edges : np.ndarray
            Y-axis bin edges

        Returns
        -------
        Dict[str, Any]
            Force field data
        """
        try:
            # Calculate gradient of the potential map
            from scipy.ndimage import sobel

            # Calculate gradients using Sobel operator
            dy = sobel(potential_map, axis=0)
            dx = sobel(potential_map, axis=1)

            # Force is negative gradient
            fx = -dx
            fy = -dy

            # Calculate force magnitude
            f_magnitude = np.sqrt(fx**2 + fy**2)

            # Create grid coordinates
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers)

            # Subsample for visualization
            n_samples = min(20, min(len(x_centers), len(y_centers)))
            step_x = max(1, len(x_centers) // n_samples)
            step_y = max(1, len(y_centers) // n_samples)

            X_sub = X[::step_y, ::step_x]
            Y_sub = Y[::step_y, ::step_x]
            fx_sub = fx[::step_y, ::step_x]
            fy_sub = fy[::step_y, ::step_x]
            f_mag_sub = f_magnitude[::step_y, ::step_x]

            return {
                'fx': fx,
                'fy': fy,
                'magnitude': f_magnitude,
                'X_viz': X_sub,
                'Y_viz': Y_sub,
                'fx_viz': fx_sub,
                'fy_viz': fy_sub,
                'magnitude_viz': f_mag_sub,
                'x_centers': x_centers,
                'y_centers': y_centers
            }

        except Exception as e:
            return {
                'error': str(e)
            }

    def visualize_energy_landscape(self, potential_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                                 force_field: Dict[str, Any], energy_units: str) -> go.Figure:
        """
        Create interactive visualization of energy landscape and force field.

        Parameters
        ----------
        potential_map : np.ndarray
            Energy landscape
        x_edges : np.ndarray
            X-axis bin edges
        y_edges : np.ndarray
            Y-axis bin edges
        force_field : Dict[str, Any]
            Force field data
        energy_units : str
            Units for energy display

        Returns
        -------
        go.Figure
            Plotly figure with energy landscape and force field
        """
        from plotly.subplots import make_subplots

        # Create subplots: energy surface and force field
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Energy Landscape", "Force Field"],
            specs=[[{"type": "surface"}, {"type": "contour"}]]
        )

        # Create x and y grids for surface plot
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        # Add energy landscape surface
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=potential_map,
                colorscale='Viridis',
                colorbar=dict(title=f'Energy ({energy_units})'),
                showscale=True,
                contours={
                    "z": {
                        "show": True,
                        "start": np.min(potential_map),
                        "end": np.max(potential_map),
                        "size": (np.max(potential_map) - np.min(potential_map)) / 10,
                    }
                }
            ),
            row=1, col=1
        )

        # Add contour plot with force vectors
        fig.add_trace(
            go.Contour(
                z=potential_map,
                x=x_centers,
                y=y_centers,
                colorscale='Viridis',
                showscale=False
            ),
            row=1, col=2
        )

        # Add force vectors if available
        if 'X_viz' in force_field:
            # Scale arrows
            max_mag = np.max(force_field['magnitude_viz']) if force_field['magnitude_viz'].size > 0 else 1
            if max_mag > 0:
                scale_factor = 0.3 * (x_edges.max() - x_edges.min()) / max_mag
            else:
                scale_factor = 1.0

            # Normalize to get unit vectors for direction
            with np.errstate(divide='ignore', invalid='ignore'):
                fx_norm = np.divide(force_field['fx_viz'], force_field['magnitude_viz'])
                fy_norm = np.divide(force_field['fy_viz'], force_field['magnitude_viz'])
                fx_norm[~np.isfinite(fx_norm)] = 0
                fy_norm[~np.isfinite(fy_norm)] = 0

            # Create quiver plot
            fig.add_trace(
                go.Scatter(
                    x=force_field['X_viz'].flatten(),
                    y=force_field['Y_viz'].flatten(),
                    mode='markers',
                    marker=dict(
                        symbol='arrow',
                        size=5,
                        color=force_field['magnitude_viz'].flatten(),
                        colorscale='Viridis',
                        showscale=False,
                        opacity=0.8
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title="Energy Landscape and Force Field",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title=f"Energy ({energy_units})"
            ),
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=1000,
            height=500
        )

        return fig


class ActiveTransportAnalyzer:
    """
    Advanced analyzer for active transport detection and characterization.
    Integrates with biophysical models for comprehensive transport analysis.
    """

    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0):
        """
        Initialize the active transport analyzer.

        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns 'track_id', 'frame', 'x', 'y'
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.results = {}

        # Convert coordinates to micrometers if needed
        for col in ['x', 'y']:
            if col in self.tracks_df.columns:
                self.tracks_df[f'{col}_um'] = self.tracks_df[col] * self.pixel_size

        # Preprocess: calculate velocities and accelerations
        self._preprocess_tracks()

    def _preprocess_tracks(self) -> None:
        """
        Preprocess tracks to calculate velocities, accelerations, and other features.
        """
        # Group by track_id
        grouped = self.tracks_df.groupby('track_id')

        # List to store processed tracks
        processed_tracks = []

        for track_id, track_data in grouped:
            if len(track_data) < 3:
                continue

            # Sort by frame
            track = track_data.sort_values('frame').copy()

            # Calculate displacements
            track['dx'] = track['x_um'].diff()
            track['dy'] = track['y_um'].diff()

            # Calculate time differences
            track['dt'] = track['frame'].diff() * self.frame_interval

            # Calculate speed and velocity
            track['speed'] = np.sqrt(track['dx']**2 + track['dy']**2) / track['dt']
            track['vx'] = track['dx'] / track['dt']
            track['vy'] = track['dy'] / track['dt']

            # Calculate acceleration
            track['ax'] = track['vx'].diff() / track['dt'].shift(-1)
            track['ay'] = track['vy'].diff() / track['dt'].shift(-1)
            track['acceleration'] = np.sqrt(track['ax']**2 + track['ay']**2)

            # Calculate angle changes (direction)
            v_angles = np.arctan2(track['vy'], track['vx'])
            track['angle_change'] = np.abs(np.diff(v_angles, append=v_angles.iloc[-1]))

            # Fix angle wraparound (ensure angle differences are between 0 and pi)
            track['angle_change'] = np.minimum(track['angle_change'], 2*np.pi - track['angle_change'])

            # Calculate straightness (distance traveled / path length)
            end_to_end_dist = np.sqrt(
                (track['x_um'].iloc[-1] - track['x_um'].iloc[0])**2 +
                (track['y_um'].iloc[-1] - track['y_um'].iloc[0])**2
            )
            path_length = np.sum(np.sqrt(track['dx']**2 + track['dy']**2))

            if path_length > 0:
                straightness = end_to_end_dist / path_length
            else:
                straightness = 0

            track['straightness'] = straightness

            processed_tracks.append(track)

        # Combine processed tracks
        if processed_tracks:
            self.processed_tracks = pd.concat(processed_tracks)

            # Compute track-level statistics
            track_stats = []

            for track_id, track_data in self.processed_tracks.groupby('track_id'):
                # Calculate mean statistics
                mean_speed = track_data['speed'].mean()
                mean_acc = track_data['acceleration'].mean()
                mean_angle_change = track_data['angle_change'].mean()
                straightness = track_data['straightness'].iloc[0]  # Same for all rows

                # Calculate max speed
                max_speed = track_data['speed'].max()

                track_stats.append({
                    'track_id': track_id,
                    'mean_speed': mean_speed,
                    'max_speed': max_speed,
                    'mean_acceleration': mean_acc,
                    'mean_angle_change': mean_angle_change,
                    'straightness': straightness,
                    'track_length': len(track_data)
                })

            self.track_results = pd.DataFrame(track_stats)

            # Store basic results
            self.results['basic_stats'] = {
                'success': True,
                'track_results': self.track_results,
                'processed_tracks': self.processed_tracks,
                'parameters': {
                    'pixel_size': self.pixel_size,
                    'frame_interval': self.frame_interval
                }
            }
        else:
            # No valid tracks after preprocessing
            self.processed_tracks = pd.DataFrame()
            self.track_results = pd.DataFrame()

            # Store error result
            self.results['basic_stats'] = {
                'success': False,
                'error': 'No valid tracks after preprocessing',
                'parameters': {
                    'pixel_size': self.pixel_size,
                    'frame_interval': self.frame_interval
                }
            }

    def detect_active_transport(self, speed_threshold: float = 0.5, 
                               straightness_threshold: float = 0.7,
                               min_track_length: int = 10) -> Dict[str, Any]:
        """
        Detect active transport in tracks based on speed and straightness criteria.

        Parameters
        ----------
        speed_threshold : float
            Minimum speed (μm/s) to classify as active transport
        straightness_threshold : float
            Minimum straightness (0-1) for directed motion
        min_track_length : int
            Minimum number of frames required for analysis

        Returns
        -------
        Dict[str, Any]
            Active transport detection results with summary statistics
        """
        # Check if we have preprocessed tracks
        if not hasattr(self, 'track_results') or self.track_results.empty:
            return {
                'success': False,
                'error': 'No track data available. Preprocessing may have failed.'
            }

        # Filter tracks by length
        valid_tracks = self.track_results[
            self.track_results['track_length'] >= min_track_length
        ].copy()

        if valid_tracks.empty:
            return {
                'success': False,
                'error': f'No tracks meet minimum length requirement ({min_track_length} frames)'
            }

        # Classify tracks as active if they meet both criteria
        valid_tracks['is_active'] = (
            (valid_tracks['mean_speed'] >= speed_threshold) &
            (valid_tracks['straightness'] >= straightness_threshold)
        )

        # Calculate summary statistics
        total_tracks = len(valid_tracks)
        active_tracks = valid_tracks['is_active'].sum()
        active_fraction = active_tracks / total_tracks if total_tracks > 0 else 0

        # Statistics for active tracks
        active_data = valid_tracks[valid_tracks['is_active']]
        
        if not active_data.empty:
            statistics = {
                'mean_speed': active_data['mean_speed'].mean(),
                'max_speed': active_data['max_speed'].max(),
                'mean_straightness': active_data['straightness'].mean(),
                'mean_acceleration': active_data['mean_acceleration'].mean(),
                'mean_path_length': 0  # Will be calculated if needed
            }
        else:
            statistics = {
                'mean_speed': 0,
                'max_speed': 0,
                'mean_straightness': 0,
                'mean_acceleration': 0,
                'mean_path_length': 0
            }

        results = {
            'success': True,
            'summary': {
                'total_tracks': total_tracks,
                'active_tracks': int(active_tracks),
                'passive_tracks': total_tracks - int(active_tracks),
                'active_fraction': active_fraction
            },
            'statistics': statistics,
            'classified_tracks': valid_tracks,
            'parameters': {
                'speed_threshold': speed_threshold,
                'straightness_threshold': straightness_threshold,
                'min_track_length': min_track_length
            }
        }

        self.results['active_transport'] = results
        return results

    def detect_directional_motion_segments(self, min_segment_length: int = 5, 
                                          straightness_threshold: float = 0.8,
                                          velocity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect segments of directed motion within tracks.

        Parameters
        ----------
        min_segment_length : int
            Minimum number of frames for a valid segment
        straightness_threshold : float
            Minimum straightness value for directional segments (0-1)
        velocity_threshold : float
            Minimum velocity for active transport (μm/s)

        Returns
        -------
        Dict[str, Any]
            Directional segment detection results
        """
        # Check if we have basic results
        transport_results = self.results.get('basic_stats', {})

        if not transport_results.get('success', False):
            return {
                'success': False,
                'error': 'Basic track analysis failed. No valid tracks available.'
            }

        # Extract directional segments with velocity filtering
        segments = []
        track_results = transport_results.get('track_results', pd.DataFrame())

        # Vectorized filtering for tracks above velocity threshold
        fast_tracks = track_results[track_results.get('mean_speed', pd.Series(dtype=float)).fillna(0) >= velocity_threshold]

        # Vectorized segment creation - much faster than iterrows()
        if not fast_tracks.empty:
            for _, track in fast_tracks.iterrows():
                track_id = track['track_id']

                # Get processed track data
                track_data = self.processed_tracks[self.processed_tracks['track_id'] == track_id]

                if len(track_data) < min_segment_length:
                    continue

                # Check for straightness
                if track['straightness'] >= straightness_threshold:
                    # Calculate mean velocity vector
                    mean_vx = track_data['vx'].mean()
                    mean_vy = track_data['vy'].mean()
                    mean_velocity = np.sqrt(mean_vx**2 + mean_vy**2)
                    mean_direction = np.arctan2(mean_vy, mean_vx)

                    # Calculate metrics
                    duration = (track_data['frame'].max() - track_data['frame'].min()) * self.frame_interval
                    distance = np.sqrt(
                        (track_data['x_um'].iloc[-1] - track_data['x_um'].iloc[0])**2 +
                        (track_data['y_um'].iloc[-1] - track_data['y_um'].iloc[0])**2
                    )

                    segments.append({
                        'track_id': track_id,
                        'start_frame': track_data['frame'].min(),
                        'end_frame': track_data['frame'].max(),
                        'duration': duration,
                        'distance': distance,
                        'mean_velocity': mean_velocity,
                        'direction': mean_direction,
                        'straightness': track['straightness'],
                        'num_points': len(track_data)
                    })
        else:
            # No tracks met the velocity threshold
            pass

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
            return {
                'success': False,
                'error': 'Run detect_directional_motion_segments() first to identify segments'
            }

        segments = self.results['directional_segments']['segments']

        if not segments:
            return {
                'success': False,
                'error': 'No directional segments detected'
            }

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
            elif velocity < fast_threshold:
                transport_modes['slow_directed'] += 1
            elif straightness >= high_straightness:
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

def analyze_motion_models(tracks_df, models=None, min_track_length=10, time_window=None, pixel_size=1.0, frame_interval=1.0):
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
    pixel_size : float, optional
        Pixel size in μm, by default 1.0
    frame_interval : float, optional
        Time between frames in seconds, by default 1.0

    Returns
    -------
    dict
        Dictionary containing model fit results and classifications
    """
    if models is None:
        models = ['brownian', 'directed', 'confined']

    # Validate input data
    if tracks_df is None or tracks_df.empty:
        return {'success': False, 'error': 'Empty or invalid tracks dataframe provided'}

    required_columns = ['track_id', 'frame', 'x', 'y']
    if not all(col in tracks_df.columns for col in required_columns):
        return {'success': False, 'error': f'Tracks dataframe missing required columns: {required_columns}'}

    # Convert coordinates to physical units
    tracks_df = tracks_df.copy()
    if 'x_um' not in tracks_df.columns:
        tracks_df['x_um'] = tracks_df['x'] * pixel_size
        tracks_df['y_um'] = tracks_df['y'] * pixel_size

    # Group tracks by track_id
    grouped = tracks_df.groupby('track_id')

    # Filter for minimum track length
    long_enough = [tid for tid, group in grouped if len(group) >= min_track_length]

    if not long_enough:
        return {
            'success': False,
            'error': f'No tracks meet the minimum length requirement of {min_track_length}'
        }

    results = {
        'success': True,
        'track_ids': long_enough,
        'models': models,
        'classifications': {},
        'model_parameters': {},
        'best_model': {},
        'error_metrics': {},
    }

    try:
        for track_id in long_enough:
            track_data = grouped.get_group(track_id).sort_values('frame')

            # Extract positions and frames - use um coordinates if available
            positions = track_data[['x_um', 'y_um']].values if 'x_um' in track_data.columns else track_data[['x', 'y']].values
            frames = track_data['frame'].values

            # Convert frames to time if frame_interval is provided
            times = frames * frame_interval

            # Handle time_window if specified
            if time_window and len(positions) > time_window:
                model_fits = {}
                model_errors = {}

                # Analyze sliding windows
                for i in range(len(positions) - time_window + 1):
                    window_pos = positions[i:i+time_window]
                    window_times = times[i:i+time_window]

                    try:
                        window_fits, window_errors = _fit_motion_models(window_pos, window_times, models)

                        for model, fit in window_fits.items():
                            if model not in model_fits:
                                model_fits[model] = []
                            model_fits[model].append(fit)

                        for model, error in window_errors.items():
                            if model not in model_errors:
                                model_errors[model] = []
                            model_errors[model].append(error)
                    except Exception as e:
                        print(f"Warning: Error fitting window for track {track_id}: {str(e)}")
                        continue

                if not model_errors:
                    # Skip this track if all window fits failed
                    continue

                # Aggregate window results - handle case where some fits may have failed
                best_model, best_score = _determine_best_model(model_errors)

                # Store results - calculate mean of successful fits for each parameter
                results['model_parameters'][track_id] = {}
                for model, fits in model_fits.items():
                    if fits:  # Check if any fits exist for this model
                        # Get all parameters from first fit
                        param_names = list(fits[0].keys())
                        results['model_parameters'][track_id][model] = {
                            param: np.mean([fit.get(param, 0) for fit in fits if param in fit]) 
                            for param in param_names
                        }

                # Calculate mean error for each model
                results['error_metrics'][track_id] = {
                    model: np.mean(errors) for model, errors in model_errors.items() if errors
                }

            else:
                # Analyze whole track
                try:
                    model_fits, model_errors = _fit_motion_models(positions, times, models)
                    best_model, best_score = _determine_best_model(model_errors)

                    results['model_parameters'][track_id] = model_fits
                    results['error_metrics'][track_id] = model_errors
                except Exception as e:
                    print(f"Warning: Error fitting track {track_id}: {str(e)}")
                    continue

            results['best_model'][track_id] = best_model
            results['classifications'][track_id] = best_model

        # Generate summary statistics only if we have results
        if results['classifications']:
            results['summary'] = _summarize_motion_analysis(results)
        else:
            results['success'] = False
            results['error'] = 'No tracks could be successfully analyzed'

    except Exception as e:
        results['success'] = False
        results['error'] = f'Error in motion model analysis: {str(e)}'

    return results

def _determine_best_model(model_errors):
    """Determine the best model based on the lowest error."""
    if not model_errors:
        return 'unknown', float('inf')

    # Using a robust min operation that handles if a model's error is not calculated
    best_model = min(model_errors, key=lambda model: model_errors.get(model, float('inf')))
    return best_model, model_errors[best_model]

def _summarize_motion_analysis(results):
    """Summarize motion analysis results."""
    summary = {}
    if 'classifications' in results:
        classifications = list(results['classifications'].values())
        summary['total_tracks_analyzed'] = len(classifications)
        summary['model_counts'] = {model: classifications.count(model) for model in set(classifications)}
    return summary

def _fit_motion_models(positions, times, models):
    """
    Fit different motion models to position data.

    Parameters
    ----------
    positions : np.ndarray
        Array of x,y positions
    times : np.ndarray
        Array of times (seconds)
    models : list
        List of motion model names to fit

    Returns
    -------
    tuple
        (model_fits, model_errors) dictionaries
    """
    model_fits = {}
    model_errors = {}

    # Check for sufficient data points
    if len(positions) < 4:
        raise ValueError("At least 4 positions are required for model fitting")

    # Calculate displacements and time intervals
    displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    dt = np.diff(times)

    # Handle edge case with zero time intervals
    if np.any(dt <= 0):
        # Replace zero time intervals with minimum positive value to avoid division by zero
        min_positive_dt = np.min(dt[dt > 0]) if np.any(dt > 0) else 1e-6
        dt = np.maximum(dt, min_positive_dt)

    # Brownian motion model
    if 'brownian' in models:
        try:
            D, brownian_error = _fit_brownian_motion(displacements, dt)
            model_fits['brownian'] = {'D': D}
            model_errors['brownian'] = brownian_error
        except Exception as e:
            print(f"Warning: Failed to fit Brownian motion model: {str(e)}")

    # Directed motion model
    if 'directed' in models:
        try:
            D, v, directed_error = _fit_directed_motion(displacements, dt, positions, times)
            model_fits['directed'] = {'D': D, 'v': v}
            model_errors['directed'] = directed_error
        except Exception as e:
            print(f"Warning: Failed to fit directed motion model: {str(e)}")

    # Confined motion model
    if 'confined' in models:
        try:
            D, L, confined_error = _fit_confined_motion(displacements, dt, positions)
            model_fits['confined'] = {'D': D, 'L': L}
            model_errors['confined'] = confined_error
        except Exception as e:
            print(f"Warning: Failed to fit confined motion model: {str(e)}")

    # If no models were successfully fit, raise exception
    if not model_fits:
        raise ValueError("Failed to fit any motion models to the data")

    return model_fits, model_errors

def _fit_brownian_motion(displacements, dt):
    """Fit Brownian motion model to displacements."""
    # For pure Brownian motion: MSD = 4*D*t
    # Estimate diffusion coefficient

    # Calculate MSD for different time lags
    max_lag = min(20, len(displacements) // 4)  # Use at most 1/4 of the track length
    max_lag = max(2, max_lag)  # At least 2 lags

    msd_by_lag = []
    for lag in range(1, max_lag + 1):
        if lag >= len(displacements):
            break
        # Get displacements for this lag
        lag_disps = displacements[0:len(displacements)-lag+1]
        lag_dts = dt[0:len(dt)-lag+1].sum()
        msd = np.mean(lag_disps**2)
        msd_by_lag.append((lag_dts, msd))

    # Fit MSD = 4*D*t
    times = np.array([t for t, _ in msd_by_lag])
    msds = np.array([m for _, m in msd_by_lag])

    # Linear fit through origin
    D = np.sum(msds * times) / (4 * np.sum(times**2))
    D = max(D, 1e-9)  # Ensure positive diffusion coefficient

    # Calculate error as residual between observed and expected MSDs
    expected_msds = 4 * D * times
    error = np.mean((msds - expected_msds)**2)

    return D, error

def _fit_directed_motion(displacements, dt, positions, times):
    """Fit directed motion model to displacements and positions."""
    # For directed motion: MSD = 4*D*t + (v*t)^2

    # First, estimate direction using linear regression
    t = times - times[0]  # Start from zero
    x = positions[:, 0]
    y = positions[:, 1]

    # Use try-except to handle singular matrices
    try:
        vx, _ = np.polyfit(t, x, 1)
        vy, _ = np.polyfit(t, y, 1)
    except np.linalg.LinAlgError:
        # Fall back to simple difference if polyfit fails
        vx = (x[-1] - x[0]) / (t[-1] - t[0] + 1e-9)
        vy = (y[-1] - y[0]) / (t[-1] - t[0] + 1e-9)

    v = np.sqrt(vx**2 + vy**2)

    # Calculate MSD for different time lags
    max_lag = min(20, len(displacements) // 4)
    max_lag = max(2, max_lag)

    msd_by_lag = []
    for lag in range(1, max_lag + 1):
        if lag >= len(displacements):
            break
        lag_disps = displacements[0:len(displacements)-lag+1]
        lag_dts = dt[0:len(dt)-lag+1].sum()
        msd = np.mean(lag_disps**2)
        msd_by_lag.append((lag_dts, msd))

    times = np.array([t for t, _ in msd_by_lag])
    msds = np.array([m for _, m in msd_by_lag])

    # Fit MSD = 4*D*t + (v*t)^2
    # Use linear regression: MSD = 4*D*t + (v^2)*t^2
    A = np.column_stack((times, times**2))
    try:
        params, residuals, _, _ = np.linalg.lstsq(A, msds, rcond=None)
        D = params[0] / 4  # Extract diffusion coefficient
        v_squared = params[1]  # Extract velocity squared
    except np.linalg.LinAlgError:
        # Fall back to simple estimates if lstsq fails
        D = np.mean(msds) / (4 * np.mean(times))
        v_squared = v**2  # Use previously estimated velocity

    # Ensure non-negative diffusion coefficient
    D = max(0, D)

    # Validate velocity: use sqrt of v_squared if positive, otherwise use previously estimated v
    v_fit = np.sqrt(max(0, v_squared)) if v_squared > 0 else v

    # Calculate error
    expected_msds = 4 * D * times + (v_fit * times)**2
    error = np.mean((msds - expected_msds)**2)

    return D, v_fit, error

def _fit_confined_motion(displacements, dt, positions):
    """Fit confined motion model to displacements and positions."""
    # For confined motion: MSD = L^2 * [1 - exp(-4*D*t/L^2)]

    # Initial estimates
    msd = np.mean(displacements**2)
    mean_dt = np.mean(dt)
    D_initial = msd / (4 * mean_dt)

    # Ensure D_initial is positive
    D_initial = max(D_initial, 1e-9)

    # Estimate confinement size as 2x standard deviation of positions
    std_pos = np.std(positions, axis=0)
    L_initial = 2 * np.mean(std_pos)

    # Ensure L_initial is positive
    L_initial = max(L_initial, 1e-9)

    # Time vector for fitting
    t_fit = np.linspace(0, np.sum(dt), len(displacements))

    # Define the model function
    def confined_model(t, D, L):
        return L**2 * (1 - np.exp(-4 * D * t / L**2))

    # Fit the model to the data
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(confined_model, t_fit, displacements, p0=[D_initial, L_initial])
        D_confined, L_confined = popt
    except Exception as e:
        D_confined, L_confined = np.nan, np.nan
        print(f"Warning: Confined motion fit failed: {str(e)}")

    # Calculate error as residual between observed and expected MSDs
    expected_msds = confined_model(t_fit, D_confined, L_confined)
    error = np.mean((displacements - expected_msds)**2)

    return D_confined, L_confined, error

# Bayesian segmentation helpers (lazy import to avoid mandatory dependency during normal use)
def run_bocpd_segmentation(tracks_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience wrapper for Bayesian Online Changepoint Detection on diffusion steps.
    Parameters
    ----------
    tracks_df : DataFrame with columns track_id, frame, x, y
    config : dict of BOCPDConfig overrides
    """
    try:
        from bayes_bocpd_diffusion import BOCPDDiffusion, BOCPDConfig
    except ImportError:
        return {'success': False, 'error': 'bayes_bocpd_diffusion module not found'}
    cfg_kwargs = config or {}
    cfg = BOCPDConfig(**{k: v for k, v in cfg_kwargs.items() if k in BOCPDConfig().__dict__})
    model = BOCPDDiffusion(tracks_df, cfg)
    return model.segment_all()

def run_hmm_segmentation(tracks_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience wrapper for Bayesian (MAP) HMM diffusion / drift state segmentation.
    Parameters
    ----------
    tracks_df : DataFrame with columns track_id, frame, x, y
    config : dict of HMMConfig overrides
    """
    try:
        from bayes_hmm_diffusion import BayesHMMDiffusion, HMMConfig
    except ImportError:
        return {'success': False, 'error': 'bayes_hmm_diffusion module not found'}
    cfg_kwargs = config or {}
    cfg = HMMConfig(**{k: v for k, v in cfg_kwargs.items() if k in HMMConfig().__dict__})
    model = BayesHMMDiffusion(tracks_df, cfg)
    return model.segment_all()


def fit_ornstein_uhlenbeck_model(
    track_df: pd.DataFrame,
    pixel_size: float = 1.0,
    frame_interval: float = 1.0
) -> Dict[str, float]:
    """
    Fits the Ornstein-Uhlenbeck model to a single track's velocity autocorrelation function (VACF).

    The OU process models a particle in a harmonic potential. This function calculates
    the VACF for a single trajectory, fits it to an exponential decay, and extracts
    key physical parameters.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame for a single track, containing 'frame', 'x', and 'y' columns.
        It is assumed that this DataFrame contains data for only one track_id.
    pixel_size : float, optional
        The size of a pixel in physical units (e.g., micrometers), by default 1.0.
    frame_interval : float
        The time interval between consecutive frames in seconds.

    Returns
    -------
    dict[str, float]
        A dictionary containing the fitted OU parameters:
        - 'relaxation_time' (tau): The characteristic time for velocity decorrelation.
        - 'kt_m' (<v^2>): The variance of the velocity distribution, proportional to temperature.
        - 'gamma_m' (gamma/m): The friction coefficient per unit mass.
        - 'k_m' (k/m): The trap stiffness per unit mass.
        Returns an empty dictionary if the fit is unsuccessful.
    """
    if track_df.empty or len(track_df) < 4:
        return {}

    # --- 1. Calculate VACF for the single track ---
    track = track_df.sort_values('frame').copy()
    x = track['x'].values * pixel_size
    y = track['y'].values * pixel_size
    t = track['frame'].values * frame_interval

    # Calculate velocities, handling potential non-uniform frame intervals
    dt = np.diff(t)
    if np.any(dt <= 0):
        return {}

    vx = np.diff(x) / dt
    vy = np.diff(y) / dt

    if len(vx) < 2:
        return {}

    vacf_values = []
    max_lag = len(vx) // 2
    lags = np.arange(max_lag) * frame_interval

    for lag_idx in range(max_lag):
        if lag_idx == 0:
            vacf_values.append(np.mean(vx*vx + vy*vy))
        else:
            v_t = vx[:-lag_idx]*vx[lag_idx:] + vy[:-lag_idx]*vy[lag_idx:]
            vacf_values.append(np.mean(v_t))

    vacf = np.array(vacf_values)

    # --- 2. Fit VACF to an exponential decay ---
    def _exponential_decay(t, tau, A):
        return A * np.exp(-t / tau)

    try:
        p0 = (1.0, vacf[0])
        popt, _ = curve_fit(_exponential_decay, lags, vacf, p0=p0)
        tau, A = popt

        # --- 3. Extract physical parameters ---
        kt_m = A
        gamma_m = 1 / tau
        k_m = 1 / (tau**2)

        return {
            'relaxation_time': tau,
            'kt_m': kt_m,
            'gamma_m': gamma_m,
            'k_m': k_m
        }
    except (RuntimeError, ValueError):
        return {}