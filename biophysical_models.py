"""
Advanced biophysical models for single particle tracking analysis.
Specialized for nucleosome diffusion in chromatin and polymer physics modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import linregress
import os
import tempfile

class PolymerPhysicsModel:
    """
    Class for analyzing particle motion using polymer physics models.
    This includes Rouse model, Zimm model, Reptation, and Fractal analysis.
    """
    
    def __init__(self, msd_data: pd.DataFrame, pixel_size: float = 0.1, frame_interval: float = 0.1):
        """
        Initialize polymer physics model analyzer.
        
        Parameters
        ----------
        msd_data : pd.DataFrame
            Mean squared displacement data with columns 'track_id', 'lag_time', 'msd'
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
        """
        self.msd_data = msd_data
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.results = {}
        
    def fit_rouse_model(self, fit_alpha: bool = False, temperature: float = 300.0, 
                       n_beads: int = 100, friction_coefficient: float = 1e-8) -> Dict[str, Any]:
        """
        Fit MSD data to Rouse model.
        
        Parameters
        ----------
        fit_alpha : bool
            Whether to fit anomalous diffusion exponent (True) or fix at 0.5 (False)
        temperature : float
            Temperature in Kelvin
        n_beads : int
            Number of beads in the Rouse chain
        friction_coefficient : float
            Bead friction coefficient
            
        Returns
        -------
        Dict[str, Any]
            Rouse model fitting results
        """
        # Calculate ensemble MSD
        ensemble_msd = self.msd_data.groupby('lag_time')['msd'].mean().reset_index()
        
        # Convert lag time to seconds
        ensemble_msd['lag_time_seconds'] = ensemble_msd['lag_time'] * self.frame_interval
        
        # Initialize parameters
        kB = 1.38e-23  # Boltzmann constant, J/K
        
        if fit_alpha:
            # Fit MSD = 4*D*t^alpha
            from scipy.optimize import curve_fit
            
            def power_law(t, D, alpha):
                return 4 * D * np.power(t, alpha)
            
            # Initial parameter guess
            p0 = [1e-2, 0.5]  # D, alpha
            
            # Fit the model
            try:
                popt, pcov = curve_fit(power_law, ensemble_msd['lag_time_seconds'], ensemble_msd['msd'], p0=p0)
                D_macro, alpha = popt
                
                # Calculate Gamma coefficient (depends on the polymer model)
                Gamma = D_macro / (kB * temperature) * (6 * np.pi * n_beads * friction_coefficient)
                
                # Calculate fit curve
                ensemble_msd['msd_fit'] = power_law(ensemble_msd['lag_time_seconds'], D_macro, alpha)
                
                # Calculate R-squared
                residuals = ensemble_msd['msd'] - ensemble_msd['msd_fit']
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((ensemble_msd['msd'] - np.mean(ensemble_msd['msd']))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store results
                results = {
                    'success': True,
                    'parameters': {
                        'D_macro': D_macro,
                        'alpha': alpha,
                        'Gamma': Gamma,
                        'temperature': temperature,
                        'n_beads': n_beads,
                        'friction_coefficient': friction_coefficient
                    },
                    'ensemble_msd': ensemble_msd,
                    'r_squared': r_squared
                }
                
                # Create visualization
                results['visualization'] = self._create_rouse_visualization(ensemble_msd, fit_alpha)
                
            except Exception as e:
                results = {
                    'success': False,
                    'error': str(e)
                }
        else:
            # Fix alpha = 0.5 (standard Rouse model)
            try:
                # For Rouse model with fixed alpha=0.5, fit MSD = 4*D*t^0.5
                from scipy.optimize import curve_fit
                
                def rouse_law(t, D):
                    return 4 * D * np.power(t, 0.5)
                
                # Initial parameter guess
                p0 = [1e-2]  # D
                
                # Fit the model
                popt, pcov = curve_fit(rouse_law, ensemble_msd['lag_time_seconds'], ensemble_msd['msd'], p0=p0)
                D_macro = popt[0]
                alpha = 0.5  # Fixed for standard Rouse model
                
                # Calculate Gamma coefficient
                Gamma = D_macro / (kB * temperature) * (6 * np.pi * n_beads * friction_coefficient)
                
                # Calculate fit curve
                ensemble_msd['msd_fit'] = rouse_law(ensemble_msd['lag_time_seconds'], D_macro)
                
                # Calculate R-squared
                residuals = ensemble_msd['msd'] - ensemble_msd['msd_fit']
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((ensemble_msd['msd'] - np.mean(ensemble_msd['msd']))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store results
                results = {
                    'success': True,
                    'parameters': {
                        'D_macro': D_macro,
                        'alpha': alpha,  # Fixed at 0.5
                        'Gamma': Gamma,
                        'temperature': temperature,
                        'n_beads': n_beads,
                        'friction_coefficient': friction_coefficient
                    },
                    'ensemble_msd': ensemble_msd,
                    'r_squared': r_squared
                }
                
                # Create visualization
                results['visualization'] = self._create_rouse_visualization(ensemble_msd, fit_alpha)
                
            except Exception as e:
                results = {
                    'success': False,
                    'error': str(e)
                }
        
        # Store results and return
        self.results['rouse_model'] = results
        return results
    
    def fit_zimm_model(self, temperature: float = 300.0, solvent_viscosity: float = 0.001, 
                      hydrodynamic_radius: float = 5e-9) -> Dict[str, Any]:
        """
        Fit MSD data to Zimm model (includes hydrodynamic interactions).
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        solvent_viscosity : float
            Solvent viscosity in Pa·s
        hydrodynamic_radius : float
            Hydrodynamic radius in meters
            
        Returns
        -------
        Dict[str, Any]
            Zimm model fitting results
        """
        # Calculate ensemble MSD
        ensemble_msd = self.msd_data.groupby('lag_time')['msd'].mean().reset_index()
        
        # Convert lag time to seconds
        ensemble_msd['lag_time_seconds'] = ensemble_msd['lag_time'] * self.frame_interval
        
        # Initialize parameters
        kB = 1.38e-23  # Boltzmann constant, J/K
        
        try:
            # For Zimm model, fit MSD = 4*D*t^(2/3)
            from scipy.optimize import curve_fit
            
            def zimm_law(t, D):
                return 4 * D * np.power(t, 2/3)
            
            # Initial parameter guess
            p0 = [1e-2]  # D
            
            # Fit the model
            popt, pcov = curve_fit(zimm_law, ensemble_msd['lag_time_seconds'], ensemble_msd['msd'], p0=p0)
            D_zimm = popt[0]
            alpha = 2/3  # Fixed for standard Zimm model
            
            # Calculate radius of gyration (Rg) from Stokes-Einstein relation
            # D = kB * T / (6 * pi * eta * Rh)
            Rg = kB * temperature / (6 * np.pi * solvent_viscosity * D_zimm)
            
            # Calculate fit curve
            ensemble_msd['msd_fit'] = zimm_law(ensemble_msd['lag_time_seconds'], D_zimm)
            
            # Calculate R-squared
            residuals = ensemble_msd['msd'] - ensemble_msd['msd_fit']
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ensemble_msd['msd'] - np.mean(ensemble_msd['msd']))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store results
            results = {
                'success': True,
                'parameters': {
                    'D_zimm': D_zimm,
                    'alpha': alpha,  # Fixed at 2/3
                    'Rg': Rg,
                    'temperature': temperature,
                    'solvent_viscosity': solvent_viscosity,
                    'hydrodynamic_radius': hydrodynamic_radius
                },
                'ensemble_msd': ensemble_msd,
                'r_squared': r_squared
            }
            
            # Create visualization
            fig = go.Figure()
            
            # Plot original MSD data
            fig.add_trace(go.Scatter(
                x=ensemble_msd['lag_time_seconds'],
                y=ensemble_msd['msd'],
                mode='markers',
                name='MSD Data',
                marker=dict(color='blue')
            ))
            
            # Plot fitted curve
            fig.add_trace(go.Scatter(
                x=ensemble_msd['lag_time_seconds'],
                y=ensemble_msd['msd_fit'],
                mode='lines',
                name=f'Zimm Model Fit (α=2/3)',
                line=dict(color='red')
            ))
            
            # Update layout
            fig.update_layout(
                title='Zimm Model Fit (MSD vs Time)',
                xaxis_title='Lag Time (s)',
                yaxis_title='MSD (μm²)',
                xaxis_type="log",
                yaxis_type="log",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            fig.update_xaxes(exponentformat='power')
            fig.update_yaxes(exponentformat='power')
            
            results['visualization'] = fig
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
        
        # Store results and return
        self.results['zimm_model'] = results
        return results
    
    def fit_reptation_model(self, temperature: float = 300.0, tube_diameter: float = 100e-9,
                          contour_length: float = 1000e-9) -> Dict[str, Any]:
        """
        Fit MSD data to reptation model (tube-like motion in entangled solutions).
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        tube_diameter : float
            Diameter of confining tube in meters
        contour_length : float
            Total contour length of the polymer in meters
            
        Returns
        -------
        Dict[str, Any]
            Reptation model fitting results
        """
        # Calculate ensemble MSD
        ensemble_msd = self.msd_data.groupby('lag_time')['msd'].mean().reset_index()
        
        # Convert lag time to seconds
        ensemble_msd['lag_time_seconds'] = ensemble_msd['lag_time'] * self.frame_interval
        
        # Initialize parameters
        kB = 1.38e-23  # Boltzmann constant, J/K
        
        try:
            # For reptation model, fit MSD = 4*D*t^0.25 for intermediate times
            from scipy.optimize import curve_fit
            
            def reptation_law(t, D):
                return 4 * D * np.power(t, 0.25)
            
            # Initial parameter guess
            p0 = [1e-3]  # D
            
            # Fit the model to intermediate time region (assuming the data is in this regime)
            # For a complete model, we would need to determine which regime we're in
            popt, pcov = curve_fit(reptation_law, ensemble_msd['lag_time_seconds'], ensemble_msd['msd'], p0=p0)
            D_rep = popt[0]
            alpha = 0.25  # Fixed for reptation model intermediate times
            
            # Calculate key parameters
            # Rough estimate of disengagement time (tube renewal)
            disengagement_time = (contour_length**2) / D_rep
            
            # Rough estimate of entanglement time
            entanglement_time = (tube_diameter**4) / (D_rep * contour_length**2)
            
            # Calculate fit curve
            ensemble_msd['msd_fit'] = reptation_law(ensemble_msd['lag_time_seconds'], D_rep)
            
            # Calculate R-squared
            residuals = ensemble_msd['msd'] - ensemble_msd['msd_fit']
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ensemble_msd['msd'] - np.mean(ensemble_msd['msd']))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store results
            results = {
                'success': True,
                'parameters': {
                    'D_rep': D_rep,
                    'alpha': alpha,  # Fixed at 0.25
                    'disengagement_time': disengagement_time,
                    'entanglement_time': entanglement_time,
                    'temperature': temperature,
                    'tube_diameter': tube_diameter,
                    'contour_length': contour_length
                },
                'ensemble_msd': ensemble_msd,
                'r_squared': r_squared
            }
            
            # Create visualization
            fig = go.Figure()
            
            # Plot original MSD data
            fig.add_trace(go.Scatter(
                x=ensemble_msd['lag_time_seconds'],
                y=ensemble_msd['msd'],
                mode='markers',
                name='MSD Data',
                marker=dict(color='blue')
            ))
            
            # Plot fitted curve
            fig.add_trace(go.Scatter(
                x=ensemble_msd['lag_time_seconds'],
                y=ensemble_msd['msd_fit'],
                mode='lines',
                name=f'Reptation Model Fit (α=0.25)',
                line=dict(color='red')
            ))
            
            # Update layout
            fig.update_layout(
                title='Reptation Model Fit (MSD vs Time)',
                xaxis_title='Lag Time (s)',
                yaxis_title='MSD (μm²)',
                xaxis_type="log",
                yaxis_type="log",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            fig.update_xaxes(exponentformat='power')
            fig.update_yaxes(exponentformat='power')
            
            results['visualization'] = fig
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
        
        # Store results and return
        self.results['reptation_model'] = results
        return results
    
    def analyze_fractal_dimension(self, min_scale: int = 1, max_scale: int = 20) -> Dict[str, Any]:
        """
        Analyze fractal dimension of tracks using box-counting method.
        
        Parameters
        ----------
        min_scale : int
            Minimum scale for box counting
        max_scale : int
            Maximum scale for box counting
            
        Returns
        -------
        Dict[str, Any]
            Fractal analysis results
        """
        try:
            # Group MSD data by track_id
            track_groups = self.msd_data.groupby('track_id')
            
            # Calculate fractal dimensions for each track
            fractal_dimensions = []
            
            for track_id, group in track_groups:
                # For box counting, we need raw trajectory data
                # This is a placeholder, as fractal dimension calculation requires full trajectories
                
                # Simulate a fractal dimension calculation based on MSD scaling
                # In a real implementation, this would use the trajectory and box-counting method
                track_msd = group.sort_values('lag_time')
                lag_times = track_msd['lag_time'].values
                msd_values = track_msd['msd'].values
                
                if len(lag_times) > 5:
                    # Estimate fractal dimension from MSD power law
                    # log(MSD) ∝ alpha * log(t), and fractal dimension D ≈ 2/alpha
                    log_lag = np.log10(lag_times[1:])  # Skip first point at lag=0
                    log_msd = np.log10(msd_values[1:])
                    
                    # Linear fit to get power law exponent
                    slope, intercept, r_value, p_value, std_err = linregress(log_lag, log_msd)
                    
                    alpha = slope
                    if alpha > 0:
                        # Calculate fractal dimension
                        track_fractal_dim = 2 / alpha
                        
                        # Typical fractal dimensions are between 1 and 2
                        if 1 <= track_fractal_dim <= 2:
                            fractal_dimensions.append(track_fractal_dim)
            
            # Calculate average fractal dimension
            if fractal_dimensions:
                mean_fractal_dim = np.mean(fractal_dimensions)
                std_fractal_dim = np.std(fractal_dimensions)
            else:
                mean_fractal_dim = float('nan')
                std_fractal_dim = float('nan')
            
            # Create visualization
            fig = go.Figure()
            
            # Histogram of fractal dimensions
            if fractal_dimensions:
                fig.add_trace(go.Histogram(
                    x=fractal_dimensions,
                    nbinsx=20,
                    marker_color='rgb(55, 83, 109)',
                    name='Fractal Dimension'
                ))
                
                # Add vertical line for mean
                fig.add_vline(
                    x=mean_fractal_dim,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_fractal_dim:.3f}",
                    annotation_position="top right"
                )
                
                # Update layout
                fig.update_layout(
                    title='Fractal Dimension Distribution',
                    xaxis_title='Fractal Dimension',
                    yaxis_title='Count',
                    showlegend=False
                )
            else:
                fig.add_annotation(
                    text="Insufficient data for fractal analysis",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
            
            # Store results
            results = {
                'success': True,
                'parameters': {
                    'fractal_dimension': mean_fractal_dim,
                    'fractal_dimension_std': std_fractal_dim,
                    'min_scale': min_scale,
                    'max_scale': max_scale
                },
                'r_squared': r_value**2 if 'r_value' in locals() else float('nan'),
                'visualization': fig
            }
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e)
            }
        
        # Store results and return
        self.results['fractal_analysis'] = results
        return results
    
    def _create_rouse_visualization(self, ensemble_msd: pd.DataFrame, fit_alpha: bool = False) -> go.Figure:
        """
        Create visualization for Rouse model fit.
        
        Parameters
        ----------
        ensemble_msd : pd.DataFrame
            DataFrame with MSD data and fit
        fit_alpha : bool
            Whether alpha was fitted or fixed
            
        Returns
        -------
        go.Figure
            Plotly figure with MSD data and fit
        """
        fig = go.Figure()
        
        # Plot original MSD data
        fig.add_trace(go.Scatter(
            x=ensemble_msd['lag_time_seconds'],
            y=ensemble_msd['msd'],
            mode='markers',
            name='MSD Data',
            marker=dict(color='blue')
        ))
        
        # Plot fitted curve
        alpha_value = self.results['rouse_model']['parameters']['alpha']
        alpha_text = f"α={alpha_value:.3f}" if fit_alpha else f"α=0.5 (fixed)"
        
        fig.add_trace(go.Scatter(
            x=ensemble_msd['lag_time_seconds'],
            y=ensemble_msd['msd_fit'],
            mode='lines',
            name=f'Rouse Model Fit ({alpha_text})',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Rouse Model Fit (MSD vs Time)',
            xaxis_title='Lag Time (s)',
            yaxis_title='MSD (μm²)',
            xaxis_type="log",
            yaxis_type="log",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )
        
        fig.update_xaxes(exponentformat='power')
        fig.update_yaxes(exponentformat='power')
        
        return fig


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
            
            # Calculate potential energy using appropriate method
            if method == "boltzmann":
                # Boltzmann inversion: U = -kBT * ln(P)
                U = -np.log(H / H.max())
                
            elif method == "drift":
                # Drift-based method requires velocity data
                # Calculate the mean drift at each position
                U = np.zeros_like(H)
                
                # Placeholder - would need to calculate drift field from the data
                # This is a simplification - not an actual drift calculation
                U = -np.log(H / H.max())  # Substitute with Boltzmann for now
                
            elif method == "kramers":
                # Kramers-Moyal expansion requires more detailed dynamics
                # This would require analysis of transition probabilities
                U = np.zeros_like(H)
                
                # Placeholder - not an actual Kramers-Moyal implementation
                U = -np.log(H / H.max())  # Substitute with Boltzmann for now
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Normalize energy to kBT units if requested
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
            
            # Store results
            results = {
                'success': True,
                'energy_landscape': U,
                'x_edges': x_edges,
                'y_edges': y_edges,
                'force_field': force_field,
                'dwell_regions': dwell_regions,
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
    
    # Ensure L_initial is positive and reasonable
    L_initial = max(L_initial, 0.1)
    
    # Calculate MSD for different time lags
    max_lag = min(20, len(displacements) // 4)
    max_lag = max(3, max_lag)  # Need more lags for confined motion
    
    msd_by_lag = []
    for lag in range(1, max_lag + 1):
        if lag >= len(displacements):
            break
        lag_disps = displacements[0:len(displacements)-lag+1]
        lag_dts = np.sum(dt[0:len(dt)-lag+1])
        msd = np.mean(lag_disps**2)
        msd_by_lag.append((lag_dts, msd))
    
    times = np.array([t for t, _ in msd_by_lag])
    msds = np.array([m for _, m in msd_by_lag])
    
    # Define confined motion model function
    def confined_msd(t, D, L):
        # Handle potential numerical issues
        if L <= 0:
            return np.zeros_like(t)
            
        # Avoid overflow in exp
        exponent = np.minimum(-4 * D * t / (L**2), 709)  # max value for np.exp
        return L**2 * (1 - np.exp(exponent))
    
    # Function to minimize
    def objective(params):
        D, L = params
        # Add penalty for negative values
        if D <= 0 or L <= 0:
            return 1e6
        expected_msd = confined_msd(times, D, L)
        return np.mean((msds - expected_msd)**2)
    
    # Optimize parameters
    try:
        from scipy.optimize import minimize
        
        bounds = [(1e-9, None), (1e-9, None)]  # D and L must be positive
        result = minimize(objective, [D_initial, L_initial], bounds=bounds, method='L-BFGS-B')
        
        D, L = result.x
        error = result.fun
        
        # Ensure parameters are physically reasonable
        D = max(D, 1e-9)
        L = max(L, 1e-9)
    except Exception:
        # Fallback if optimization fails
        D = D_initial
        L = L_initial
        error = objective([D, L])
    
    return D, L, error

def _determine_best_model(model_errors):
    """Determine best model based on error metrics and complexity."""
    if not model_errors:
        return 'unknown', float('inf')
    
    # Calculate Bayesian Information Criterion (BIC) for model selection
    # BIC = n*ln(RSS/n) + k*ln(n), where n is number of data points and k is number of parameters
    # We can simplify since n is the same for all models
    bic = {}
    param_count = {'brownian': 1, 'directed': 2, 'confined': 2}
    n = 1  # placeholder, will cancel out in comparison
    
    for model, error in model_errors.items():
        k = param_count.get(model, 1)
        bic[model] = n * np.log(error) + k * np.log(n)
    
    # Choose model with lowest BIC
    best_model = min(bic, key=bic.get)
    best_score = model_errors[best_model]
    
    # If errors are very close, prefer simpler model
    error_ratio_threshold = 1.1
    
    if 'brownian' in model_errors and best_model != 'brownian':
        if model_errors['brownian'] < error_ratio_threshold * best_score:
            best_model = 'brownian'
            best_score = model_errors['brownian']
    
    return best_model, best_score

def _summarize_motion_analysis(results):
    """Generate summary statistics for motion analysis."""
    summary = {
        'total_tracks': len(results['track_ids']),
        'analyzed_tracks': len(results['classifications']),
        'model_counts': {},
        'model_parameters': {},
        'fractions': {}
    }
    
    # Count occurrences of each model
    for model in results['best_model'].values():
        if model not in summary['model_counts']:
            summary['model_counts'][model] = 0
        summary['model_counts'][model] += 1
    
    # Calculate model fractions
    total_classified = len(results['classifications'])
    if total_classified > 0:
        for model, count in summary['model_counts'].items():
            summary['fractions'][model] = count / total_classified
    
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
        
        # Calculate averages and standard deviations
        if params:
            summary['model_parameters'][model] = {
                f"{param}_mean": np.mean(values) for param, values in params.items()
            }
            summary['model_parameters'][model].update({
                f"{param}_std": np.std(values) for param, values in params.items() if len(values) > 1
            })
    
    # Add detailed track-by-track classifications for visualization
    summary['track_classifications'] = {track_id: model for track_id, model in results['classifications'].items()}
    
    # Prepare parameter distributions for visualization
    parameter_distributions = {}
    for model in results['models']:
        model_params = {}
        for param_name in ['D', 'v', 'L', 'alpha', 'K_alpha']:  # Common parameters across models
            values = []
            for track_id, model_data in results['model_parameters'].items():
                if model in model_data and param_name in model_data[model]:
                    values.append({
                        'track_id': track_id,
                        'value': model_data[model][param_name]
                    })
            if values:
                model_params[param_name] = values
        if model_params:
            parameter_distributions[model] = model_params
    
    summary['parameter_distributions'] = parameter_distributions
    
    return summary

def prepare_motion_visualization_data(motion_analysis_results):
    """
    Prepare motion analysis results for visualization.
    
    Parameters
    ----------
    motion_analysis_results : dict
        Results from analyze_motion_models function
    
    Returns
    -------
    dict
        Data formatted for visualization
    """
    if not motion_analysis_results.get('success', False):
        return {'success': False, 'error': motion_analysis_results.get('error', 'Unknown error')}
    
    viz_data = {
        'success': True,
        'motion_types': {
            'counts': motion_analysis_results.get('summary', {}).get('model_counts', {}),
            'fractions': motion_analysis_results.get('summary', {}).get('fractions', {})
        },
        'track_classifications': motion_analysis_results.get('summary', {}).get('track_classifications', {}),
        'parameter_distributions': motion_analysis_results.get('summary', {}).get('parameter_distributions', {})
    }
    
    # Generate color mapping for tracks
    track_colors = {}
    color_map = {
        'brownian': '#1f77b4',  # blue
        'directed': '#ff7f0e',  # orange
        'confined': '#2ca02c',  # green
        'anomalous': '#d62728'  # red
    }
    
    for track_id, model in viz_data['track_classifications'].items():
        track_colors[track_id] = color_map.get(model, '#7f7f7f')  # default gray
    
    viz_data['track_colors'] = track_colors
    
    # Calculate mean parameter values for each motion type
    motion_parameters = {}
    if 'summary' in motion_analysis_results and 'model_parameters' in motion_analysis_results['summary']:
        for model, params in motion_analysis_results['summary']['model_parameters'].items():
            motion_parameters[model] = {
                param.split('_')[0]: value 
                for param, value in params.items() 
                if param.endswith('_mean')
            }
    
    viz_data['motion_parameters'] = motion_parameters
    
    # Add speed distributions by motion type if available
    if motion_analysis_results.get('speed_distributions', {}):
        viz_data['speed_distributions'] = motion_analysis_results['speed_distributions']
    
    return viz_data

def calculate_motion_speed_distributions(tracks_df, motion_analysis_results, pixel_size=1.0, frame_interval=1.0):
    """
    Calculate speed distributions for different motion types.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data
    motion_analysis_results : dict
        Results from analyze_motion_models function
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Time between frames in seconds
    
    Returns
    -------
    dict
        Speed distributions by motion type
    """
    if not motion_analysis_results.get('success', False) or tracks_df.empty:
        return {'success': False}
    
    # Get track classifications
    classifications = motion_analysis_results.get('classifications', {})
    if not classifications:
        return {'success': False, 'error': 'No track classifications found'}
    
    # Create empty result structure
    speed_distributions = {model: [] for model in motion_analysis_results.get('models', [])}
    
    # Convert coordinates to physical units if needed
    if 'x_um' not in tracks_df.columns:
        tracks_df = tracks_df.copy()
        tracks_df['x_um'] = tracks_df['x'] * pixel_size
        tracks_df['y_um'] = tracks_df['y'] * pixel_size
    
    # Calculate instantaneous speeds for each track
    for track_id, model in classifications.items():
        if model not in speed_distributions:
            continue
        
        # Get track data
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        if len(track_data) < 2:
            continue
        
        # Calculate displacements
        dx = np.diff(track_data['x_um'].values)
        dy = np.diff(track_data['y_um'].values)
        dt = np.diff(track_data['frame'].values) * frame_interval
        
        # Calculate speeds (μm/s)
        speeds = np.sqrt(dx**2 + dy**2) / dt
        
        # Filter out unrealistic speeds (e.g., from tracking errors)
        speeds = speeds[speeds < 100]  # 100 μm/s is a reasonable upper limit
        
        # Add to distribution
        speed_distributions[model].extend(speeds.tolist())
    
    # Add summary statistics
    for model, speeds in speed_distributions.items():
        if speeds:
            speed_distributions[f"{model}_mean"] = np.mean(speeds)
            speed_distributions[f"{model}_median"] = np.median(speeds)
            speed_distributions[f"{model}_std"] = np.std(speeds)
    
    return {'success': True, 'speed_distributions': speed_distributions}

def export_motion_analysis_results(motion_analysis_results, output_format='csv', output_path=None):
    """
    Export motion analysis results to file.
    
    Parameters
    ----------
    motion_analysis_results : dict
        Results from analyze_motion_models function
    output_format : str
        Format to export ('csv' or 'json')
    output_path : str
        Path to save the output file
    
    Returns
    -------
    bool
        True if export was successful
    """
    if not motion_analysis_results.get('success', False):
        return False
    
    if output_path is None:
        import tempfile
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, f"motion_analysis_results.{output_format}")
    
    try:
        if output_format.lower() == 'csv':
            # Extract track classifications
            classifications_df = pd.DataFrame([
                {'track_id': track_id, 'motion_type': model}
                for track_id, model in motion_analysis_results.get('classifications', {}).items()
            ])
            
            # Add parameters if available
            for track_id, model_params in motion_analysis_results.get('model_parameters', {}).items():
                for model, params in model_params.items():
                    for param_name, value in params.items():
                        column_name = f"{model}_{param_name}"
                        mask = classifications_df['track_id'] == track_id
                        if any(mask):
                                                       classifications_df.loc[mask, column_name] = value
            
            # Save to CSV
            classifications_df.to_csv(output_path, index=False)
            
        elif output_format.lower() == 'json':
            import json
            
            # Prepare JSON serializable data
            json_data = {
                'summary': motion_analysis_results.get('summary', {}),
                'classifications': motion_analysis_results.get('classifications', {}),
                'model_parameters': {}
            }
            
            # Convert numpy types to native Python types
            for track_id, model_params in motion_analysis_results.get('model_parameters', {}).items():
                json_data['model_parameters'][str(track_id)] = {}
                for model, params in model_params.items():
                    json_data['model_parameters'][str(track_id)][model] = {
                        param: float(value) for param, value in params.items()
                    }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        else:
            return False
            
        return True
        
    except Exception as e:
        print(f"Error exporting motion analysis results: {str(e)}")
        return False
    except Exception as e:
        print(f"Error exporting motion analysis results: {str(e)}")
        return False

def analyze_whole_image_diffusion(tracks_df, pixel_size=1.0, frame_interval=1.0, max_time_points=None, min_tracks_per_point=3):
    """
    Analyze diffusion across the whole image by aligning all tracks to start at time zero
    and averaging the squared displacement for each time step.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with columns 'track_id', 'frame', 'x', 'y'
    pixel_size : float, optional
        Pixel size in μm, by default 1.0
    frame_interval : float, optional
        Time between frames in seconds, by default 1.0
    max_time_points : int, optional
        Maximum number of time points to analyze, by default None (use all available)
    min_tracks_per_point : int, optional
        Minimum number of tracks required for a time point to be included, by default 3
    
    Returns
    -------
    dict
        Dictionary containing diffusion analysis results
    """
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
    track_groups = tracks_df.groupby('track_id')
    
    # Process each track to calculate squared displacements from starting point
    all_displacements = []
    
    for track_id, track_data in track_groups:
        # Sort by frame
        track = track_data.sort_values('frame').copy()
        
        if len(track) < 2:
            continue
        
        # Get starting position
        start_x = track['x_um'].iloc[0]
        start_y = track['y_um'].iloc[0]
        start_frame = track['frame'].iloc[0]
        
        # Calculate displacement from start for each point
        track['time_from_start'] = (track['frame'] - start_frame) * frame_interval
        track['dx_from_start'] = track['x_um'] - start_x
        track['dy_from_start'] = track['y_um'] - start_y
        track['squared_displacement'] = track['dx_from_start']**2 + track['dy_from_start']**2
        
        # Add to collection
        all_displacements.append(track[['time_from_start', 'squared_displacement']])
    
    if not all_displacements:
        return {'success': False, 'error': 'No valid tracks for displacement analysis'}
    
    # Combine all displacement data
    displacement_df = pd.concat(all_displacements, ignore_index=True)
    
    # Group by time from start and calculate mean squared displacement (MSD)
    msd_by_time = displacement_df.groupby('time_from_start').agg(
        msd=('squared_displacement', 'mean'),
        msd_std=('squared_displacement', 'std'),
        n_tracks=('squared_displacement', 'count')
    ).reset_index()
    
    # Filter time points with too few tracks
    msd_by_time = msd_by_time[msd_by_time['n_tracks'] >= min_tracks_per_point].copy()
    
    if msd_by_time.empty:
        return {'success': False, 'error': 'No time points with sufficient track data'}
    
    # Limit to max_time_points if specified
    if max_time_points is not None and len(msd_by_time) > max_time_points:
        msd_by_time = msd_by_time.iloc[:max_time_points].copy()
    
    # Fit diffusion model: MSD = 4*D*t + offset
    try:
        # Simple linear fit with sklearn
        from sklearn.linear_model import LinearRegression
        
        # Prepare data for fitting
        X = msd_by_time['time_from_start'].values.reshape(-1, 1)
        y = msd_by_time['msd'].values
        
        # Fit linear model
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # Extract parameters
        D = model.coef_[0] / 4  # Diffusion coefficient
        offset = model.intercept_  # Offset (ideally should be close to 0)
        
        # Calculate R-squared
        y_pred = model.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Add fit curve to results
        msd_by_time['msd_fit'] = y_pred
        
        # Create visualization
        fig = go.Figure()
        
        # Add MSD data points with error bars
        fig.add_trace(go.Scatter(
            x=msd_by_time['time_from_start'],
            y=msd_by_time['msd'],
            mode='markers',
            name='MSD Data',
            error_y=dict(
                type='data',
                array=msd_by_time['msd_std'] / np.sqrt(msd_by_time['n_tracks']),
                visible=True
            ),
            marker=dict(size=8, color='blue')
        ))
        
        # Add fit line
        fig.add_trace(go.Scatter(
            x=msd_by_time['time_from_start'],
            y=msd_by_time['msd_fit'],
            mode='lines',
            name=f'Fit: D={D:.4e} μm²/s, R²={r_squared:.4f}',
            line=dict(color='red', width=2)
        ))
        
        # Add number of tracks as a secondary y-axis
        fig.add_trace(go.Scatter(
            x=msd_by_time['time_from_start'],
            y=msd_by_time['n_tracks'],
            mode='lines+markers',
            name='Number of Tracks',
            yaxis='y2',
            marker=dict(size=6, color='green'),
            line=dict(color='green', width=1, dash='dot')
        ))
        
        # Update layout
        fig.update_layout(
            title='Whole Image Diffusion Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Mean Squared Displacement (μm²)',
            yaxis2=dict(
                title='Number of Tracks',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='closest'
        )
        
        return {
            'success': True,
            'D': D,
            'offset': offset,
            'r_squared': r_squared,
            'msd_data': msd_by_time,
            'n_tracks_total': len(track_groups),
            'n_tracks_used': msd_by_time['n_tracks'].iloc[0],
            'visualization': fig
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f'Error fitting diffusion model: {str(e)}',
            'traceback': traceback.format_exc()
        }

def check_data_availability(tracks_df=None, motion_results=None):
    """
    Check if track data and motion analysis results are available for visualization.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame, optional
        Track data
    motion_results : dict, optional
        Motion analysis results
        
    Returns
    -------
    dict
        Status of data availability
    """
    status = {
        'tracks_available': False,
        'motion_analysis_available': False,
        'tracks_count': 0,
        'analyzed_tracks_count': 0,
        'error_messages': []
    }
    
    # Check track data
    if tracks_df is not None and not tracks_df.empty:
        required_columns = ['track_id', 'frame', 'x', 'y']
        if all(col in tracks_df.columns for col in required_columns):
            status['tracks_available'] = True
            status['tracks_count'] = tracks_df['track_id'].nunique()
        else:
            status['error_messages'].append(f"Track data missing required columns: {required_columns}")
    else:
        status['error_messages'].append("No track data loaded")
    
    # Check motion analysis results
    if motion_results is not None and isinstance(motion_results, dict):
        if motion_results.get('success', False):
            status['motion_analysis_available'] = True
            status['analyzed_tracks_count'] = len(motion_results.get('classifications', {}))
        else:
            status['error_messages'].append(f"Motion analysis failed: {motion_results.get('error', 'Unknown error')}")
    else:
        status['error_messages'].append("No motion analysis results available")
    
    return status

def get_motion_analysis_summary(motion_results):
    """
    Get a summary of motion analysis results for reporting.
    
    Parameters
    ----------
    motion_results : dict
        Motion analysis results
        
    Returns
    -------
    dict
        Summary data for reports
    """
    if not motion_results or not motion_results.get('success', False):
        return {'success': False, 'error': 'No valid motion analysis results'}
    
    summary = motion_results.get('summary', {})
    
    report_data = {
        'success': True,
        'total_tracks': summary.get('total_tracks', 0),
        'analyzed_tracks': summary.get('analyzed_tracks', 0),
        'model_counts': summary.get('model_counts', {}),
        'model_fractions': summary.get('fractions', {}),
        'model_parameters': summary.get('model_parameters', {}),
        'track_classifications': summary.get('track_classifications', {}),
        'parameter_distributions': summary.get('parameter_distributions', {})
    }
    
    # Calculate additional statistics
    if report_data['model_counts']:
        most_common_model = max(report_data['model_counts'], key=report_data['model_counts'].get)
        report_data['most_common_model'] = most_common_model
        report_data['most_common_fraction'] = report_data['model_fractions'].get(most_common_model, 0)
    
    return report_data
