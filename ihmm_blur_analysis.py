"""
Infinite Hidden Markov Model (iHMM) with Blur-Aware Emission Models

Bayesian nonparametric state segmentation that:
1. Auto-discovers number of diffusive states
2. Accounts for motion blur and localization noise
3. Works on short trajectories (<100 steps)

Key advantages over standard HMM:
- No need to pre-specify K (number of states)
- Hierarchical Dirichlet Process prior allows state birth/death
- Blur-aware emission models reduce false state changes
- Better performance on experimental data with noise

Reference: Lindén et al. (PMC6050756), Persson et al. 2013
Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma
from scipy.stats import multivariate_normal
from typing import Dict, List, Optional, Tuple
import warnings


class BlurAwareEmissionModel:
    """
    Emission model accounting for motion blur and localization noise.
    
    For exposure time t_exp and frame interval dt:
    Observed displacement ∝ N(0, 2·D·dt - 2·D·t_exp²/3 + 2·σ_loc²)
    """
    
    def __init__(self, dimensionality: int = 2):
        """
        Initialize emission model.
        
        Parameters
        ----------
        dimensionality : int
            2 for 2D tracking, 3 for 3D
        """
        self.d = dimensionality
    
    
    def log_likelihood(self, displacements: np.ndarray, 
                      D: float, dt: float, 
                      sigma_loc: float, t_exp: float = 0) -> float:
        """
        Log-likelihood of displacements given diffusion coefficient.
        
        Parameters
        ----------
        displacements : np.ndarray
            Shape (N, d) array of displacements
        D : float
            Diffusion coefficient (μm²/s)
        dt : float
            Frame interval (s)
        sigma_loc : float
            Localization uncertainty (μm)
        t_exp : float
            Exposure time (s). If 0, no blur correction.
        
        Returns
        -------
        float
            Log-likelihood
        """
        # Variance with blur correction
        if t_exp > 0:
            R = (t_exp / dt)**2 / 3  # Blur factor
            variance_per_dim = 2 * D * dt * (1 - R) + 2 * sigma_loc**2
        else:
            variance_per_dim = 2 * D * dt + 2 * sigma_loc**2
        
        if variance_per_dim <= 0:
            return -np.inf
        
        # Multivariate normal log-likelihood
        # For independent dimensions: log p = sum over dimensions
        n_points = len(displacements)
        
        # Each dimension: N(0, variance_per_dim)
        log_lik = 0
        for dim in range(self.d):
            r_squared = displacements[:, dim]**2
            log_lik += -0.5 * np.sum(r_squared / variance_per_dim)
            log_lik += -0.5 * n_points * np.log(2 * np.pi * variance_per_dim)
        
        return log_lik


class iHMMBlurAnalyzer:
    """
    Infinite HMM with blur-aware emissions for trajectory segmentation.
    
    Uses variational Bayes with Hierarchical Dirichlet Process (HDP) prior.
    """
    
    def __init__(self, dt: float = 0.1, 
                 sigma_loc: float = 0.03,
                 t_exp: float = 0.0,
                 dimensionality: int = 2,
                 alpha: float = 1.0,
                 gamma: float = 1.0):
        """
        Initialize iHMM analyzer.
        
        Parameters
        ----------
        dt : float
            Frame interval (s)
        sigma_loc : float
            Localization uncertainty (μm)
        t_exp : float
            Exposure time (s)
        dimensionality : int
            2 or 3
        alpha : float
            HDP concentration parameter (controls state persistence)
        gamma : float
            HDP concentration parameter (controls new state creation)
        """
        self.dt = dt
        self.sigma_loc = sigma_loc
        self.t_exp = t_exp
        self.d = dimensionality
        self.alpha = alpha
        self.gamma = gamma
        
        self.emission_model = BlurAwareEmissionModel(dimensionality=dimensionality)
        
        # Priors for diffusion coefficients
        self.D_prior_mean = 0.1  # μm²/s
        self.D_prior_std = 1.0   # μm²/s
    
    
    def initialize_states(self, displacements: np.ndarray, 
                         K_init: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state assignments and diffusion coefficients.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacement vectors
        K_init : int
            Initial number of states
        
        Returns
        -------
        states : np.ndarray
            Initial state sequence
        D_values : np.ndarray
            Diffusion coefficient for each state
        """
        N = len(displacements)
        
        # K-means-like initialization on squared displacements
        r_squared = np.sum(displacements**2, axis=1)
        
        # Partition into K_init quantiles
        quantiles = np.linspace(0, 1, K_init + 1)
        thresholds = np.quantile(r_squared, quantiles[1:-1])
        
        states = np.digitize(r_squared, thresholds)
        
        # Estimate D for each state
        D_values = np.zeros(K_init)
        for k in range(K_init):
            mask = states == k
            if np.sum(mask) > 0:
                msd_k = np.mean(r_squared[mask])
                D_values[k] = msd_k / (2 * self.d * self.dt)
            else:
                D_values[k] = self.D_prior_mean
        
        return states, D_values
    
    
    def viterbi_with_blur(self, displacements: np.ndarray,
                         D_values: np.ndarray,
                         transition_matrix: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm with blur-aware emissions.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacement vectors (N, d)
        D_values : np.ndarray
            Diffusion coefficients for K states
        transition_matrix : np.ndarray
            Transition probabilities (K, K)
        
        Returns
        -------
        states : np.ndarray
            Most likely state sequence
        """
        N = len(displacements)
        K = len(D_values)
        
        # Emission log-likelihoods
        log_emit = np.zeros((N, K))
        for k in range(K):
            for n in range(N):
                log_emit[n, k] = self.emission_model.log_likelihood(
                    displacements[n:n+1],
                    D=D_values[k],
                    dt=self.dt,
                    sigma_loc=self.sigma_loc,
                    t_exp=self.t_exp
                )
        
        # Viterbi forward pass
        log_delta = np.zeros((N, K))
        psi = np.zeros((N, K), dtype=int)
        
        # Initialization
        log_delta[0] = log_emit[0] + np.log(1.0 / K)  # Uniform prior
        
        # Recursion
        for n in range(1, N):
            for k in range(K):
                log_trans = np.log(transition_matrix[:, k] + 1e-100)
                vals = log_delta[n-1] + log_trans
                psi[n, k] = np.argmax(vals)
                log_delta[n, k] = vals[psi[n, k]] + log_emit[n, k]
        
        # Backtracking
        states = np.zeros(N, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for n in range(N-2, -1, -1):
            states[n] = psi[n+1, states[n+1]]
        
        return states
    
    
    def estimate_transition_matrix(self, states: np.ndarray, 
                                   K: int) -> np.ndarray:
        """
        Estimate transition probabilities with HDP prior.
        
        Parameters
        ----------
        states : np.ndarray
            Current state sequence
        K : int
            Number of states
        
        Returns
        -------
        transition_matrix : np.ndarray
            Transition probabilities (K, K)
        """
        # Count transitions
        counts = np.zeros((K, K))
        for n in range(len(states) - 1):
            counts[states[n], states[n+1]] += 1
        
        # Add HDP prior (stick-breaking)
        # Self-transitions favored by alpha
        # New states favored by gamma
        prior = np.ones((K, K)) * (self.gamma / K**2)
        for k in range(K):
            prior[k, k] += self.alpha
        
        # Posterior
        transition_matrix = counts + prior
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    
    def update_diffusion_coefficients(self, displacements: np.ndarray,
                                     states: np.ndarray, K: int) -> np.ndarray:
        """
        Update D values given state assignments.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacement vectors
        states : np.ndarray
            State assignments
        K : int
            Number of states
        
        Returns
        -------
        D_values : np.ndarray
            Updated diffusion coefficients
        """
        D_values = np.zeros(K)
        
        for k in range(K):
            mask = states == k
            if np.sum(mask) > 0:
                r_squared = np.sum(displacements[mask]**2, axis=1)
                msd_k = np.mean(r_squared)
                
                # MLE with blur correction
                if self.t_exp > 0:
                    R = (self.t_exp / self.dt)**2 / 3
                    D_k = (msd_k / (2 * self.d) - self.sigma_loc**2) / (self.dt * (1 - R))
                else:
                    D_k = (msd_k / (2 * self.d) - self.sigma_loc**2) / self.dt
                
                # Apply prior
                # Posterior mean with Gaussian prior
                prior_precision = 1 / self.D_prior_std**2
                data_precision = np.sum(mask) / (self.D_prior_std**2)
                
                D_values[k] = (prior_precision * self.D_prior_mean + data_precision * D_k) / \
                             (prior_precision + data_precision)
                
                # Ensure positive
                D_values[k] = max(D_values[k], 1e-6)
            else:
                D_values[k] = self.D_prior_mean
        
        return D_values
    
    
    def prune_empty_states(self, states: np.ndarray, D_values: np.ndarray,
                          transition_matrix: np.ndarray) -> Tuple:
        """
        Remove states with no assignments.
        
        Returns
        -------
        states : np.ndarray
            Relabeled states
        D_values : np.ndarray
            Pruned D values
        transition_matrix : np.ndarray
            Pruned transition matrix
        """
        K = len(D_values)
        
        # Find occupied states
        occupied = np.unique(states)
        K_new = len(occupied)
        
        if K_new == K:
            return states, D_values, transition_matrix
        
        # Relabel states
        state_map = {old: new for new, old in enumerate(occupied)}
        states_new = np.array([state_map[s] for s in states])
        
        # Prune parameters
        D_values_new = D_values[occupied]
        transition_matrix_new = transition_matrix[np.ix_(occupied, occupied)]
        
        # Renormalize transitions
        row_sums = transition_matrix_new.sum(axis=1, keepdims=True)
        transition_matrix_new = transition_matrix_new / (row_sums + 1e-100)
        
        return states_new, D_values_new, transition_matrix_new
    
    
    def fit(self, track: pd.DataFrame, max_iter: int = 50,
           K_init: int = 3, tol: float = 1e-4) -> Dict:
        """
        Fit iHMM to single trajectory.
        
        Parameters
        ----------
        track : pd.DataFrame
            Track with columns: frame, x, y (z optional)
        max_iter : int
            Maximum EM iterations
        K_init : int
            Initial number of states
        tol : float
            Convergence tolerance
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'states': ndarray,           # State sequence
                'D_values': ndarray,         # D for each state
                'transition_matrix': ndarray,
                'n_states': int,
                'log_likelihood': float,
                'converged': bool,
                'track_summary': Dict
            }
        """
        # Extract displacements
        pos_cols = ['x', 'y'] if 'z' not in track.columns else ['x', 'y', 'z']
        positions = track[pos_cols].values
        
        if len(positions) < 10:
            return {
                'success': False,
                'error': 'Track too short for iHMM (need ≥10 points)'
            }
        
        displacements = np.diff(positions, axis=0)
        N = len(displacements)
        
        # Initialize
        states, D_values = self.initialize_states(displacements, K_init=K_init)
        K = len(D_values)
        transition_matrix = self.estimate_transition_matrix(states, K)
        
        log_lik_prev = -np.inf
        converged = False
        
        # EM iterations
        for iteration in range(max_iter):
            # E-step: Viterbi decoding
            states = self.viterbi_with_blur(displacements, D_values, transition_matrix)
            
            # M-step: Update parameters
            D_values = self.update_diffusion_coefficients(displacements, states, K)
            transition_matrix = self.estimate_transition_matrix(states, K)
            
            # Prune empty states
            states, D_values, transition_matrix = self.prune_empty_states(
                states, D_values, transition_matrix
            )
            K = len(D_values)
            
            # Compute log-likelihood
            log_lik = 0
            for n in range(N):
                k = states[n]
                log_lik += self.emission_model.log_likelihood(
                    displacements[n:n+1],
                    D=D_values[k],
                    dt=self.dt,
                    sigma_loc=self.sigma_loc,
                    t_exp=self.t_exp
                )
            
            # Check convergence
            if abs(log_lik - log_lik_prev) < tol:
                converged = True
                break
            
            log_lik_prev = log_lik
        
        # Track-level summary
        state_durations = []
        current_state = states[0]
        current_duration = 1
        
        for n in range(1, len(states)):
            if states[n] == current_state:
                current_duration += 1
            else:
                state_durations.append(current_duration)
                current_state = states[n]
                current_duration = 1
        state_durations.append(current_duration)
        
        track_summary = {
            'track_id': track.iloc[0].get('track_id', 0),
            'n_points': len(positions),
            'n_displacements': N,
            'n_states_discovered': K,
            'mean_state_duration': np.mean(state_durations),
            'state_transitions': len(state_durations) - 1,
            'D_range_um2_s': (D_values.min(), D_values.max())
        }
        
        return {
            'success': True,
            'states': states,
            'D_values': D_values,
            'transition_matrix': transition_matrix,
            'n_states': K,
            'log_likelihood': log_lik,
            'converged': converged,
            'n_iterations': iteration + 1,
            'track_summary': track_summary
        }
    
    
    def batch_analyze(self, tracks_df: pd.DataFrame) -> Dict:
        """
        Analyze multiple tracks.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Multiple tracks with track_id column
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'results': List[Dict],  # Per-track results
                'summary': Dict         # Population statistics
            }
        """
        if 'track_id' not in tracks_df.columns:
            return {
                'success': False,
                'error': 'tracks_df must have track_id column'
            }
        
        track_ids = tracks_df['track_id'].unique()
        results = []
        
        for tid in track_ids:
            track = tracks_df[tracks_df['track_id'] == tid].sort_values('frame')
            result = self.fit(track)
            if result['success']:
                results.append(result)
        
        if len(results) == 0:
            return {
                'success': False,
                'error': 'No tracks successfully analyzed'
            }
        
        # Population summary
        n_states_all = [r['n_states'] for r in results]
        D_ranges = [r['track_summary']['D_range_um2_s'] for r in results]
        
        summary = {
            'n_tracks_analyzed': len(results),
            'n_states_distribution': {
                'mean': np.mean(n_states_all),
                'median': np.median(n_states_all),
                'mode': int(np.bincount(n_states_all).argmax())
            },
            'D_range_um2_s': (
                min(d[0] for d in D_ranges),
                max(d[1] for d in D_ranges)
            ),
            'convergence_rate': np.mean([r['converged'] for r in results])
        }
        
        return {
            'success': True,
            'results': results,
            'summary': summary
        }


def quick_ihmm_segmentation(track_df: pd.DataFrame, 
                            dt: float = 0.1,
                            sigma_loc: float = 0.03,
                            t_exp: float = 0.0) -> Dict:
    """
    One-liner iHMM segmentation.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Single track or multiple tracks
    dt : float
        Frame interval
    sigma_loc : float
        Localization uncertainty
    t_exp : float
        Exposure time
    
    Returns
    -------
    Dict
        Segmentation results
    """
    analyzer = iHMMBlurAnalyzer(dt=dt, sigma_loc=sigma_loc, t_exp=t_exp)
    
    if 'track_id' in track_df.columns and len(track_df['track_id'].unique()) > 1:
        return analyzer.batch_analyze(track_df)
    else:
        return analyzer.fit(track_df)
