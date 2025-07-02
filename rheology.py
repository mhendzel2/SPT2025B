"""
Microrheology Analysis Module

Calculates G' (storage modulus), G" (loss modulus), and effective viscosity
from single particle tracking data using dual frame rate measurements.

This module implements microrheology principles to extract mechanical
properties of the cellular environment from particle motion.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional, List
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import gamma
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


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
        
    def calculate_msd_from_tracks(self, tracks_df: pd.DataFrame, 
                                  pixel_size_um: float, frame_interval_s: float,
                                  max_lag_frames: int = 50) -> pd.DataFrame:
        """
        Calculate ensemble mean squared displacement from track data.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
        pixel_size_um : float
            Pixel size in micrometers
        frame_interval_s : float
            Time interval between frames in seconds
        max_lag_frames : int
            Maximum lag time in frames to calculate
            
        Returns
        -------
        pd.DataFrame
            MSD data with columns: lag_time_s, msd_m2, std_msd_m2, n_tracks
        """
        msd_results = []
        
        # Group by track_id
        tracks_grouped = tracks_df.groupby('track_id')
        
        # Calculate MSD for each lag time
        for lag in range(1, min(max_lag_frames + 1, tracks_df['frame'].nunique())):
            lag_time_s = lag * frame_interval_s
            displacements_squared = []
            
            for track_id, track_data in tracks_grouped:
                if len(track_data) <= lag:
                    continue
                    
                # Sort by frame to ensure proper ordering
                track_data = track_data.sort_values('frame')
                
                # Calculate displacements for this lag
                for i in range(len(track_data) - lag):
                    x1, y1 = track_data.iloc[i]['x'], track_data.iloc[i]['y']
                    x2, y2 = track_data.iloc[i + lag]['x'], track_data.iloc[i + lag]['y']
                    
                    # Convert to meters and calculate squared displacement
                    dx_m = (x2 - x1) * pixel_size_um * 1e-6
                    dy_m = (y2 - y1) * pixel_size_um * 1e-6
                    disp_sq = dx_m**2 + dy_m**2
                    
                    displacements_squared.append(disp_sq)
            
            if len(displacements_squared) > 0:
                msd_m2 = np.mean(displacements_squared)
                std_msd_m2 = np.std(displacements_squared)
                n_tracks = len(set([track_id for track_id, _ in tracks_grouped if len(_) > lag]))
                
                msd_results.append({
                    'lag_time_s': lag_time_s,
                    'msd_m2': msd_m2,
                    'std_msd_m2': std_msd_m2,
                    'n_tracks': n_tracks
                })
        
        return pd.DataFrame(msd_results)
    
    # In rheology.py, within the MicrorheologyAnalyzer class

    def calculate_complex_modulus_gser(self, msd_df: pd.DataFrame,
                                       omega_rad_s: float) -> Tuple[float, float]:
        """Calculate complex modulus G*(ω) using the GSER approach."""
        if len(msd_df) < 3:
            return np.nan, np.nan

        # Estimate alpha from local slope of the MSD curve
        log_times = np.log(msd_df['lag_time_s'])
        log_msd = np.log(msd_df['msd_m2'])
        idx = np.argmin(np.abs(msd_df['lag_time_s'] - (1.0 / omega_rad_s)))
        if idx == 0 or idx == len(msd_df) - 1:
            alpha = 1.0
        else:
            alpha = (log_msd[idx + 1] - log_msd[idx - 1]) / (log_times[idx + 1] - log_times[idx - 1])

        from scipy.special import gamma
        gamma_factor = gamma(1 + alpha)

        msd_at_tau = np.interp(1.0 / omega_rad_s, msd_df['lag_time_s'], msd_df['msd_m2'])
        prefactor = (self.kB * self.temperature_K) / (3 * np.pi * self.particle_radius_m * msd_at_tau)

        omega_alpha = omega_rad_s ** alpha
        phase = alpha * np.pi / 2

        g_star_real = prefactor * gamma_factor * np.cos(phase)
        g_star_imag = prefactor * gamma_factor * np.sin(phase)

        return g_star_real, g_star_imag

    def calculate_effective_viscosity(self, msd_df: pd.DataFrame, 
                                      lag_time_range_s: Tuple[float, float] = None) -> float:
        """
        Calculate effective viscosity from MSD slope.
        
        Parameters
        ----------
        msd_df : pd.DataFrame
            MSD data
        lag_time_range_s : Tuple[float, float], optional
            Time range for slope calculation. If None, uses initial slope.
            
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
            # Use first few points for initial slope
            slope_data = msd_df.head(min(5, len(msd_df)))
        
        if len(slope_data) < 2:
            return np.nan
        
        # Linear fit to get slope
        try:
            slope, _ = np.polyfit(slope_data['lag_time_s'], slope_data['msd_m2'], 1)
        except:
            return np.nan
        
        if slope <= 0:
            return np.inf
        
        # Effective diffusion coefficient: D_eff = slope / 4 (for 2D)
        D_eff = slope / 4.0
        
        # Stokes-Einstein: η = kB*T / (6*π*D*a) for 3D
        # For 2D projected motion: η = kB*T / (4*π*D*a)
        viscosity_eff = (self.kB * self.temperature_K) / (4 * np.pi * D_eff * self.particle_radius_m)
        
        return viscosity_eff
    
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
            # Process each dataset
            all_frequencies_hz = []
            all_g_prime_values = []
            all_g_double_prime_values = []
            dataset_labels = []
            
            for i, (tracks_df, frame_interval_s) in enumerate(zip(track_datasets, frame_intervals_s)):
                dataset_label = f"Dataset_{i+1}_{frame_interval_s:.3f}s"
                dataset_labels.append(dataset_label)
                
                # Calculate MSD for this dataset
                msd_data = self.calculate_msd_from_tracks(
                    tracks_df, pixel_size_um, frame_interval_s
                )
                
                if len(msd_data) == 0:
                    results['error'] = f"Insufficient data for MSD calculation in {dataset_label}"
                    return results
                
                # Determine frequency range for this dataset
                if omega_ranges and i < len(omega_ranges):
                    omega_list = omega_ranges[i]
                else:
                    # Auto-determine frequency range based on accessible time scales
                    min_time = frame_interval_s
                    max_time = msd_data['lag_time_s'].max()
                    omega_max = 2 * np.pi / (min_time * 2)  # High frequency limit
                    omega_min = 2 * np.pi / (max_time * 0.5)  # Low frequency limit
                    omega_list = np.logspace(np.log10(omega_min), np.log10(omega_max), 10)
                
                # Calculate complex moduli for this dataset
                g_prime_values = []
                g_double_prime_values = []
                frequencies_hz = []
                
                for omega_rad_s in omega_list:
                    g_prime, g_double_prime = self.calculate_complex_modulus_gser(msd_data, omega_rad_s)
                    
                    if not (np.isnan(g_prime) or np.isnan(g_double_prime)):
                        g_prime_values.append(g_prime)
                        g_double_prime_values.append(g_double_prime)
                        frequencies_hz.append(omega_rad_s / (2 * np.pi))
                        
                        # Add to combined lists for overall frequency response
                        all_frequencies_hz.append(omega_rad_s / (2 * np.pi))
                        all_g_prime_values.append(g_prime)
                        all_g_double_prime_values.append(g_double_prime)
                
                # Calculate effective viscosity for this dataset
                viscosity = self.calculate_effective_viscosity(msd_data)
                
                # Store dataset results
                dataset_results = {
                    'label': dataset_label,
                    'frame_interval_s': frame_interval_s,
                    'msd_data': msd_data,
                    'frequencies_hz': frequencies_hz,
                    'g_prime_pa': g_prime_values,
                    'g_double_prime_pa': g_double_prime_values,
                    'effective_viscosity_pa_s': viscosity,
                    'omega_range_rad_s': omega_list.tolist() if hasattr(omega_list, 'tolist') else list(omega_list)
                }
                
                if len(g_prime_values) > 0:
                    dataset_results.update({
                        'g_prime_mean_pa': np.mean(g_prime_values),
                        'g_prime_std_pa': np.std(g_prime_values),
                        'g_double_prime_mean_pa': np.mean(g_double_prime_values),
                        'g_double_prime_std_pa': np.std(g_double_prime_values),
                        'loss_tangent': np.mean(g_double_prime_values) / np.mean(g_prime_values) if np.mean(g_prime_values) > 0 else np.inf
                    })
                
                results['datasets'].append(dataset_results)
            
            # Create combined frequency response
            if len(all_frequencies_hz) > 0:
                # Sort by frequency for better visualization
                sorted_indices = np.argsort(all_frequencies_hz)
                results['combined_frequency_response'] = {
                    'frequencies_hz': [all_frequencies_hz[i] for i in sorted_indices],
                    'g_prime_pa': [all_g_prime_values[i] for i in sorted_indices],
                    'g_double_prime_pa': [all_g_double_prime_values[i] for i in sorted_indices],
                    'dataset_labels': [dataset_labels[i // 10] for i in sorted_indices]  # Approximate dataset assignment
                }
                
                # Overall statistics
                results['combined_frequency_response'].update({
                    'g_prime_overall_mean_pa': np.mean(all_g_prime_values),
                    'g_double_prime_overall_mean_pa': np.mean(all_g_double_prime_values),
                    'frequency_range_hz': [min(all_frequencies_hz), max(all_frequencies_hz)]
                })
            
            # Dataset comparison metrics
            if len(results['datasets']) > 1:
                viscosities = [ds.get('effective_viscosity_pa_s', np.nan) for ds in results['datasets']]
                valid_viscosities = [v for v in viscosities if not np.isnan(v)]
                
                if len(valid_viscosities) > 1:
                    results['dataset_comparison'] = {
                        'viscosity_variation_coefficient': np.std(valid_viscosities) / np.mean(valid_viscosities),
                        'frequency_dependent_behavior': np.std(valid_viscosities) / np.mean(valid_viscosities) > 0.2,
                        'viscosity_range_pa_s': [min(valid_viscosities), max(valid_viscosities)]
                    }
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)

        return results

    def analyze_microrheology(self, tracks_df: pd.DataFrame, pixel_size_um: float,
                              frame_interval_s: float, max_lag: int = 20) -> Dict:
        """High level single dataset microrheology analysis."""

        msd_df = self.calculate_msd_from_tracks(
            tracks_df, pixel_size_um, frame_interval_s, max_lag_frames=max_lag
        )

        if msd_df.empty:
            return {'success': False, 'error': 'Insufficient data for MSD calculation'}

        # Determine frequency range based on sampling
        min_time = frame_interval_s
        max_time = msd_df['lag_time_s'].max()
        omega_list = np.logspace(
            np.log10(2 * np.pi / (max_time * 0.5)),
            np.log10(2 * np.pi / (min_time * 2)),
            20
        )

        g_prime_values = []
        g_double_prime_values = []
        frequencies_hz = []

        for omega in omega_list:
            g_p, g_pp = self.calculate_complex_modulus_gser(msd_df, omega)
            if not (np.isnan(g_p) or np.isnan(g_pp)):
                g_prime_values.append(g_p)
                g_double_prime_values.append(g_pp)
                frequencies_hz.append(omega / (2 * np.pi))

        viscosity = self.calculate_effective_viscosity(msd_df)

        return {
            'success': True,
            'msd_data': msd_df,
            'frequency_response': {
                'frequencies_hz': frequencies_hz,
                'g_prime_pa': g_prime_values,
                'g_double_prime_pa': g_double_prime_values,
            },
            'viscosity': {'effective_pa_s': viscosity},
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
                    msd_data = dataset['msd_data']
                    color = colors[i % len(colors)]
                    
                    fig_msd.add_trace(go.Scatter(
                        x=msd_data['lag_time_s'],
                        y=msd_data['msd_m2'],
                        mode='lines+markers',
                        name=dataset['label'],
                        line=dict(color=color),
                        error_y=dict(
                            type='data',
                            array=msd_data.get('std_msd_m2', []),
                            visible=True
                        )
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
                
                fig_freq.update_layout(
                    title='Combined Complex Modulus vs Frequency',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Modulus (Pa)',
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
        # MSD comparison plot
        if analysis_results.get('msd_high_freq') is not None and analysis_results.get('msd_low_freq') is not None:
            fig_msd = go.Figure()
            
            msd_high = analysis_results['msd_high_freq']
            msd_low = analysis_results['msd_low_freq']
            
            # High frequency MSD
            fig_msd.add_trace(go.Scatter(
                x=msd_high['lag_time_s'],
                y=msd_high['msd_m2'],
                mode='lines+markers',
                name='High Frequency',
                line=dict(color='red'),
                error_y=dict(
                    type='data',
                    array=msd_high.get('std_msd_m2', []),
                    visible=True
                )
            ))
            
            # Low frequency MSD
            fig_msd.add_trace(go.Scatter(
                x=msd_low['lag_time_s'],
                y=msd_low['msd_m2'],
                mode='lines+markers',
                name='Low Frequency',
                line=dict(color='blue'),
                error_y=dict(
                    type='data',
                    array=msd_low.get('std_msd_m2', []),
                    visible=True
                )
            ))
            
            fig_msd.update_layout(
                title='Mean Squared Displacement Comparison',
                xaxis_title='Lag Time (s)',
                yaxis_title='MSD (m²)',
                xaxis_type='log',
                yaxis_type='log',
                template='plotly_white'
            )
            
            figures['msd_comparison'] = fig_msd
        
        # Single dataset MSD plot
        elif 'msd_data' in analysis_results:
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
            
            fig_freq.update_layout(
                title='Complex Modulus vs Frequency',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Modulus (Pa)',
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
    if not analysis_results.get('success', False):
        st.error(f"Microrheology analysis failed: {analysis_results.get('error', 'Unknown error')}")
        return
    
    st.subheader("Microrheology Analysis Summary")
    
    # Handle multi-dataset structure
    if 'datasets' in analysis_results:
        datasets = analysis_results['datasets']
        
        st.write(f"**Number of Datasets:** {len(datasets)}")
        
        # Dataset overview table
        if len(datasets) > 0:
            dataset_summary = []
            for dataset in datasets:
                summary_row = {
                    'Dataset': dataset['label'],
                    'Frame Interval (s)': f"{dataset['frame_interval_s']:.3f}",
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
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'g_prime_overall_mean_pa' in freq_data:
                    st.metric(
                        label="Overall G' Mean",
                        value=f"{freq_data['g_prime_overall_mean_pa']:.2e} Pa"
                    )
            
            with col2:
                if 'g_double_prime_overall_mean_pa' in freq_data:
                    st.metric(
                        label="Overall G\" Mean",
                        value=f"{freq_data['g_double_prime_overall_mean_pa']:.2e} Pa"
                    )
        
        # Dataset comparison
        if 'dataset_comparison' in analysis_results and analysis_results['dataset_comparison']:
            comparison = analysis_results['dataset_comparison']
            
            st.subheader("Dataset Comparison")
            
            if 'viscosity_variation_coefficient' in comparison:
                st.metric(
                    label="Viscosity Variation Coefficient",
                    value=f"{comparison['viscosity_variation_coefficient']:.3f}"
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
        col1, col2, col3 = st.columns(3)
        
        # Moduli summary
        if 'moduli' in analysis_results:
            moduli = analysis_results['moduli']
            
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
                    "Loss Tangent (tan δ)",
                    f"{moduli.get('loss_tangent', 0):.3f}",
                    help="G\"/G' - indicates viscoelastic behavior"
                )
        
        # Viscosity summary
        if 'viscosity' in analysis_results:
            viscosity = analysis_results['viscosity']
            
            st.subheader("Effective Viscosity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "High Frequency Viscosity",
                    f"{viscosity.get('high_freq_pa_s', 0):.2e} Pa·s"
                )
            
            with col2:
                st.metric(
                    "Low Frequency Viscosity",
                    f"{viscosity.get('low_freq_pa_s', 0):.2e} Pa·s"
            )
        
        if viscosity.get('frequency_dependent', False):
            st.warning("⚠️ Frequency-dependent viscosity detected - indicates viscoelastic behavior")
        else:
            st.info("✅ Frequency-independent viscosity - indicates Newtonian behavior")