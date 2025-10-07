"""
Loop Extrusion Detection Module for SPT2025B

Detects cohesin-mediated loop extrusion signatures in chromatin dynamics
through pattern recognition in single particle tracking data.

References:
- Fudenberg et al. (2016): "Formation of Chromosomal Domains by Loop Extrusion"
- Hansen et al. (2017): "CTCF and cohesin regulate chromatin loop stability"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, periodogram
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple
import warnings


class LoopExtrusionDetector:
    """
    Detect loop extrusion signatures in particle tracking data.
    
    Loop extrusion causes:
    1. Periodic confinement in MSD (oscillations)
    2. Restricted motion with characteristic length scale (loop size)
    3. Return-to-origin behavior
    4. Bimodal diffusion (free vs confined)
    """
    
    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 0.1, frame_interval: float = 0.1):
        """
        Initialize loop extrusion detector.
        
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
    
    def detect_loop_signatures(
        self,
        min_track_length: int = 50,
        confinement_threshold: float = 0.5
    ) -> Dict:
        """
        Detect loop extrusion signatures across all tracks.
        
        Parameters:
        -----------
        min_track_length : int
            Minimum track length for analysis
        confinement_threshold : float
            Maximum MSD plateau for confinement (μm²)
        
        Returns:
        --------
        dict with keys:
            - 'n_tracks_analyzed': int
            - 'n_confined_tracks': int
            - 'confinement_fraction': float
            - 'mean_loop_size': float (μm)
            - 'periodic_tracks': list of track IDs
            - 'track_results': DataFrame with per-track analysis
        """
        results_list = []
        
        for track_id in self.tracks_df['track_id'].unique():
            track = self.tracks_df[self.tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track) < min_track_length:
                continue
            
            # Analyze individual track
            track_result = self._analyze_single_track(track, confinement_threshold)
            track_result['track_id'] = track_id
            results_list.append(track_result)
        
        if len(results_list) == 0:
            return {
                'n_tracks_analyzed': 0,
                'n_confined_tracks': 0,
                'confinement_fraction': 0.0,
                'mean_loop_size': 0.0,
                'periodic_tracks': [],
                'track_results': pd.DataFrame()
            }
        
        results_df = pd.DataFrame(results_list)
        
        # Summary statistics
        n_confined = np.sum(results_df['is_confined'])
        n_periodic = np.sum(results_df['has_periodicity'])
        
        confined_tracks = results_df[results_df['is_confined']]
        mean_loop_size = confined_tracks['confinement_radius'].mean() if len(confined_tracks) > 0 else 0.0
        
        periodic_track_ids = results_df[results_df['has_periodicity']]['track_id'].tolist()
        
        return {
            'n_tracks_analyzed': len(results_df),
            'n_confined_tracks': n_confined,
            'n_periodic_tracks': n_periodic,
            'confinement_fraction': n_confined / len(results_df),
            'periodicity_fraction': n_periodic / len(results_df),
            'mean_loop_size': mean_loop_size,
            'periodic_tracks': periodic_track_ids,
            'track_results': results_df
        }
    
    def _analyze_single_track(self, track: pd.DataFrame, confinement_threshold: float) -> Dict:
        """Analyze single track for loop signatures."""
        
        # Calculate MSD for this track
        msd_values, lag_times = self._calculate_track_msd(track)
        
        # Check for MSD plateau (confinement)
        is_confined, plateau_value = self._detect_msd_plateau(msd_values, lag_times, confinement_threshold)
        
        # Estimate confinement radius from plateau
        if is_confined and plateau_value > 0:
            # MSD_plateau ≈ L²/3 for square confinement, L²/4 for circular
            confinement_radius = np.sqrt(plateau_value * 4)  # Assume circular
        else:
            confinement_radius = 0.0
        
        # Check for periodicity in trajectory
        has_periodicity, period = self._detect_periodicity(track)
        
        # Calculate return-to-origin tendency
        return_tendency = self._calculate_return_tendency(track)
        
        return {
            'is_confined': is_confined,
            'confinement_radius': confinement_radius,
            'plateau_msd': plateau_value,
            'has_periodicity': has_periodicity,
            'period': period,
            'return_tendency': return_tendency
        }
    
    def _calculate_track_msd(self, track: pd.DataFrame, max_lag: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MSD for a single track."""
        x = track['x_um'].values
        y = track['y_um'].values
        
        msd_values = []
        lag_times = []
        
        for lag in range(1, min(max_lag, len(x) // 3)):
            displacements_sq = []
            
            for i in range(len(x) - lag):
                dx = x[i + lag] - x[i]
                dy = y[i + lag] - y[i]
                displacements_sq.append(dx**2 + dy**2)
            
            if len(displacements_sq) > 0:
                msd_values.append(np.mean(displacements_sq))
                lag_times.append(lag * self.frame_interval)
        
        return np.array(msd_values), np.array(lag_times)
    
    def _detect_msd_plateau(
        self,
        msd_values: np.ndarray,
        lag_times: np.ndarray,
        threshold: float
    ) -> Tuple[bool, float]:
        """Detect if MSD reaches a plateau (confined motion)."""
        
        if len(msd_values) < 10:
            return False, 0.0
        
        # Check if MSD levels off
        # Compare early vs late MSD slope
        mid_point = len(msd_values) // 2
        
        early_msd = msd_values[:mid_point]
        late_msd = msd_values[mid_point:]
        
        # Fit linear to each half
        from scipy.stats import linregress
        
        try:
            early_slope, _, _, _, _ = linregress(range(len(early_msd)), early_msd)
            late_slope, _, _, _, _ = linregress(range(len(late_msd)), late_msd)
            
            # Plateau if late slope much smaller than early
            slope_ratio = abs(late_slope) / (abs(early_slope) + 1e-10)
            
            # Also check if final MSD is below threshold
            final_msd = msd_values[-5:].mean()
            
            is_confined = (slope_ratio < 0.3) and (final_msd < threshold)
            plateau_value = final_msd if is_confined else 0.0
            
            return is_confined, plateau_value
            
        except Exception:
            return False, 0.0
    
    def _detect_periodicity(self, track: pd.DataFrame) -> Tuple[bool, float]:
        """Detect periodic behavior in trajectory using FFT."""
        
        if len(track) < 20:
            return False, 0.0
        
        # Calculate distance from starting position
        x = track['x_um'].values
        y = track['y_um'].values
        x0, y0 = x[0], y[0]
        
        distances = np.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Remove trend
        from scipy.signal import detrend
        distances_detrended = detrend(distances)
        
        # Compute power spectrum
        try:
            freqs, power = periodogram(distances_detrended, fs=1.0/self.frame_interval)
            
            # Find peaks in power spectrum (excluding DC component)
            if len(power) > 5:
                peaks, properties = find_peaks(power[1:], height=np.max(power[1:]) * 0.3)
                
                if len(peaks) > 0:
                    # Get dominant frequency
                    dominant_peak_idx = peaks[np.argmax(properties['peak_heights'])]
                    dominant_freq = freqs[dominant_peak_idx + 1]  # +1 because we excluded DC
                    period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
                    
                    # Consider periodic if period is reasonable (1-100 seconds)
                    has_periodicity = 1.0 < period < 100.0
                    
                    return has_periodicity, period
        except Exception:
            pass
        
        return False, 0.0
    
    def _calculate_return_tendency(self, track: pd.DataFrame) -> float:
        """Calculate tendency to return to starting position."""
        
        x = track['x_um'].values
        y = track['y_um'].values
        
        if len(x) < 10:
            return 0.0
        
        x0, y0 = x[0], y[0]
        
        # Calculate distances from origin at different times
        distances = np.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Return tendency: correlation between time and distance
        # Negative correlation = returning
        from scipy.stats import spearmanr
        
        try:
            corr, p_value = spearmanr(np.arange(len(distances)), distances)
            
            # Return tendency: -corr (so positive value means returning)
            return -corr if p_value < 0.05 else 0.0
        except Exception:
            return 0.0
    
    def estimate_loop_parameters(self, confined_tracks_only: bool = True) -> Dict:
        """
        Estimate loop extrusion parameters from confined tracks.
        
        Returns:
        --------
        dict with keys:
            - 'mean_loop_size': float (μm)
            - 'std_loop_size': float
            - 'loop_lifetime': float (s) - based on periodicity
            - 'extrusion_velocity': float (μm/s) - estimated
        """
        results = self.detect_loop_signatures()
        
        if results['n_confined_tracks'] == 0:
            return {
                'mean_loop_size': 0.0,
                'std_loop_size': 0.0,
                'loop_lifetime': 0.0,
                'extrusion_velocity': 0.0,
                'n_loops_detected': 0
            }
        
        track_results = results['track_results']
        
        if confined_tracks_only:
            confined = track_results[track_results['is_confined']]
        else:
            confined = track_results
        
        loop_sizes = confined['confinement_radius'].values
        periods = confined[confined['has_periodicity']]['period'].values
        
        mean_size = np.mean(loop_sizes)
        std_size = np.std(loop_sizes)
        mean_period = np.mean(periods) if len(periods) > 0 else 0.0
        
        # Estimate extrusion velocity: v ~ loop_size / loop_lifetime
        if mean_period > 0:
            extrusion_velocity = mean_size / mean_period
        else:
            extrusion_velocity = 0.0
        
        return {
            'mean_loop_size': mean_size,
            'std_loop_size': std_size,
            'loop_lifetime': mean_period,
            'extrusion_velocity': extrusion_velocity,
            'n_loops_detected': len(confined)
        }
    
    def visualize_loop_detection(self, show_top_n: int = 5) -> go.Figure:
        """
        Visualize loop extrusion detection results.
        
        Shows trajectories colored by confinement status.
        """
        results = self.detect_loop_signatures()
        track_results = results['track_results']
        
        if len(track_results) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No tracks analyzed", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Sort by confinement strength
        track_results = track_results.sort_values('confinement_radius', ascending=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confined vs Free Tracks',
                'Loop Size Distribution',
                'MSD Example (Confined)',
                'Trajectory Example (Confined)'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # Plot 1: Scatter of all tracks
        confined_mask = track_results['is_confined']
        
        for track_id in track_results['track_id'].iloc[:20]:  # Show top 20
            track = self.tracks_df[self.tracks_df['track_id'] == track_id]
            is_conf = track_results[track_results['track_id'] == track_id]['is_confined'].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=track['x_um'],
                    y=track['y_um'],
                    mode='lines',
                    line=dict(color='red' if is_conf else 'blue', width=1),
                    showlegend=False,
                    hovertemplate=f'Track {track_id}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Loop size histogram
        confined_sizes = track_results[confined_mask]['confinement_radius']
        
        if len(confined_sizes) > 0:
            fig.add_trace(
                go.Histogram(
                    x=confined_sizes,
                    nbinsx=20,
                    name='Loop Sizes',
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        # Plot 3: Example MSD from most confined track
        if len(track_results) > 0:
            top_track_id = track_results.iloc[0]['track_id']
            top_track = self.tracks_df[self.tracks_df['track_id'] == top_track_id].sort_values('frame')
            
            msd_vals, lag_times = self._calculate_track_msd(top_track)
            
            fig.add_trace(
                go.Scatter(
                    x=lag_times,
                    y=msd_vals,
                    mode='markers+lines',
                    name=f'Track {top_track_id}',
                    marker=dict(color='red', size=6)
                ),
                row=2, col=1
            )
            
            # Plot 4: Trajectory of same track
            fig.add_trace(
                go.Scatter(
                    x=top_track['x_um'],
                    y=top_track['y_um'],
                    mode='lines+markers',
                    name=f'Track {top_track_id}',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="X (μm)", row=1, col=1)
        fig.update_yaxes(title_text="Y (μm)", row=1, col=1)
        fig.update_xaxes(title_text="Loop Size (μm)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Lag Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="MSD (μm²)", row=2, col=1)
        fig.update_xaxes(title_text="X (μm)", row=2, col=2)
        fig.update_yaxes(title_text="Y (μm)", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text=f"Loop Extrusion Detection<br>" +
                      f"Confined: {results['n_confined_tracks']}/{results['n_tracks_analyzed']} " +
                      f"({results['confinement_fraction']*100:.1f}%)",
            showlegend=False
        )
        
        return fig
    
    def classify_motion_regimes(self) -> Dict:
        """
        Classify tracks into motion regimes.
        
        Returns:
        --------
        dict with keys:
            - 'free_diffusion': list of track IDs
            - 'confined_loops': list of track IDs
            - 'periodic_motion': list of track IDs
            - 'mixed_regime': list of track IDs
        """
        results = self.detect_loop_signatures()
        track_results = results['track_results']
        
        free_diffusion = []
        confined_loops = []
        periodic_motion = []
        mixed_regime = []
        
        for _, row in track_results.iterrows():
            track_id = row['track_id']
            
            if row['is_confined'] and row['has_periodicity']:
                confined_loops.append(track_id)
            elif row['is_confined']:
                mixed_regime.append(track_id)
            elif row['has_periodicity']:
                periodic_motion.append(track_id)
            else:
                free_diffusion.append(track_id)
        
        return {
            'free_diffusion': free_diffusion,
            'confined_loops': confined_loops,
            'periodic_motion': periodic_motion,
            'mixed_regime': mixed_regime,
            'summary': {
                'n_free': len(free_diffusion),
                'n_confined_loops': len(confined_loops),
                'n_periodic': len(periodic_motion),
                'n_mixed': len(mixed_regime)
            }
        }
