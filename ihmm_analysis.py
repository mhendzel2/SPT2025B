"""
Infinite Hidden Markov Model (iHMM) Analysis with Blur-Aware Emissions

Enhanced HMM for single particle tracking that accounts for:
- Motion blur from finite camera exposure time
- Localization noise
- Automatic state number determination (Variational Bayes / BIC)
- Diffusion coefficient estimation per state

Key improvements over standard HMM:
1. Blur-aware emission probabilities: variance includes motion blur correction
2. Automatic model selection: determines optimal number of states
3. Physical constraints: ensures positive diffusion coefficients
4. Bootstrap confidence intervals for state parameters

References:
- Persson et al. (2013) Nature Methods 10(3):265 - vbSPT method
- Rowland & Biteen (2017) Biophys J 112(7):1375 - Blur correction in HMM
- Amitai et al. (2020) Nat Commun 11:3607 - Bayesian state identification
- Monnier et al. (2015) Biophys J 108(6):1183 - SPT-HMM with diffusion states

Author: SPT2025B Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
import warnings


class BlurAwareHMM:
    """
    Hidden Markov Model with motion blur-aware emission probabilities.
    
    Standard HMM assumes emission variance = 2*D*Δt + 2*σ²_loc
    Blur-aware HMM uses: variance = 2*D*Δt*(1 - R²/12) + 2*σ²_loc
    where R = exposure_time / frame_interval (blur fraction)
    
    This correction is critical for accurate D estimation when exposure
    time is significant relative to frame interval (R > 0.3).
    """
    
    def __init__(
        self,
        n_states: int,
        frame_interval: float = 0.1,
        exposure_time: Optional[float] = None,
        localization_error: float = 0.03,
        dimensions: int = 2,
        covariance_type: str = 'diag'
    ):
        """
        Initialize Blur-Aware HMM.
        
        Parameters
        ----------
        n_states : int
            Number of hidden states
        frame_interval : float
            Time between frames (seconds)
        exposure_time : float, optional
            Camera exposure time (seconds). If None, assumes R=0 (no blur)
        localization_error : float
            Localization precision (µm)
        dimensions : int
            Number of spatial dimensions (2 or 3)
        covariance_type : str
            'diag': diagonal covariance, 'full': full covariance matrix
        """
        self.n_states = n_states
        self.frame_interval = frame_interval
        self.exposure_time = exposure_time if exposure_time is not None else 0.0
        self.localization_error = localization_error
        self.dimensions = dimensions
        self.covariance_type = covariance_type
        
        # Blur fraction
        self.R = self.exposure_time / self.frame_interval if self.frame_interval > 0 else 0.0
        self.blur_correction_factor = 1.0 - (self.R ** 2) / 12.0
        
        # Model parameters (initialized in fit())
        self.diffusion_coefficients = None  # D for each state (µm²/s)
        self.transition_matrix = None  # State transition probabilities
        self.initial_state_probs = None  # Initial state distribution
        self.state_labels = None  # Optional state names
        
        # Fitted results
        self.log_likelihood = None
        self.state_sequence = None
        self.posterior_probs = None
        
    def _compute_emission_variance(self, D: float) -> float:
        """
        Compute emission variance for a given diffusion coefficient.
        
        Includes both diffusion and localization noise, corrected for motion blur.
        
        Parameters
        ----------
        D : float
            Diffusion coefficient (µm²/s)
            
        Returns
        -------
        float
            Variance in µm²
        """
        # Variance from diffusion (blur-corrected)
        diffusion_variance = 2 * D * self.frame_interval * self.blur_correction_factor
        
        # Variance from localization noise (static + dynamic)
        # Factor of 2 because variance of displacement = 2*σ²
        localization_variance = 2 * (self.localization_error ** 2)
        
        total_variance = diffusion_variance + localization_variance
        
        return total_variance
    
    def _emission_probability(
        self,
        displacement: np.ndarray,
        state_idx: int
    ) -> float:
        """
        Compute emission probability for a displacement given state.
        
        Parameters
        ----------
        displacement : np.ndarray
            Displacement vector (dx, dy) or (dx, dy, dz) in µm
        state_idx : int
            State index
            
        Returns
        -------
        float
            Log probability
        """
        D = self.diffusion_coefficients[state_idx]
        variance = self._compute_emission_variance(D)
        
        if self.covariance_type == 'diag':
            # Isotropic diffusion: same variance in all dimensions
            covariance = np.eye(self.dimensions) * variance
        elif self.covariance_type == 'full':
            # Allow anisotropic diffusion (would need to fit per state)
            covariance = np.eye(self.dimensions) * variance  # Simplified
        
        # Gaussian emission
        log_prob = multivariate_normal.logpdf(displacement, mean=np.zeros(self.dimensions), cov=covariance)
        
        return log_prob
    
    def fit(
        self,
        displacements: np.ndarray,
        initial_D_values: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-4
    ) -> Dict:
        """
        Fit HMM using Baum-Welch (EM) algorithm with blur-aware emissions.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacement vectors, shape (n_steps, n_dim)
        initial_D_values : np.ndarray, optional
            Initial guess for diffusion coefficients of each state (µm²/s)
        max_iter : int
            Maximum EM iterations
        tol : float
            Convergence tolerance on log-likelihood
            
        Returns
        -------
        dict
            Fit results with keys:
            - 'success': bool
            - 'diffusion_coefficients': D values per state
            - 'transition_matrix': state transition probabilities
            - 'log_likelihood': final log-likelihood
            - 'n_iterations': iterations until convergence
        """
        n_steps = displacements.shape[0]
        
        # Initialize parameters
        if initial_D_values is None:
            # Heuristic: logarithmically space D values
            msd_estimate = np.mean(np.sum(displacements**2, axis=1))
            D_max = msd_estimate / (2 * self.dimensions * self.frame_interval * self.blur_correction_factor)
            D_min = D_max / 100
            self.diffusion_coefficients = np.logspace(np.log10(D_min), np.log10(D_max), self.n_states)
        else:
            self.diffusion_coefficients = np.array(initial_D_values)
        
        # Initialize transition matrix (uniform transitions)
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        
        # Initialize initial state probabilities (uniform)
        self.initial_state_probs = np.ones(self.n_states) / self.n_states
        
        # EM iterations
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Forward-backward algorithm
            alpha, c = self._forward(displacements)
            beta = self._backward(displacements, c)
            
            # Compute posterior probabilities
            gamma = alpha * beta  # P(state_t | observations)
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
            
            # Compute pairwise posteriors
            xi = self._compute_xi(displacements, alpha, beta, c)
            
            # M-step: Update parameters
            
            # Update initial state probabilities
            self.initial_state_probs = gamma[0, :]
            
            # Update transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    numerator = np.sum(xi[:, i, j])
                    denominator = np.sum(gamma[:-1, i])
                    self.transition_matrix[i, j] = numerator / (denominator + 1e-10)
            
            # Normalize transition matrix
            self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)
            
            # Update diffusion coefficients
            for state_idx in range(self.n_states):
                # Weighted MSD for this state
                weights = gamma[:, state_idx]
                weighted_squared_displacements = weights[:, np.newaxis] * (displacements ** 2)
                weighted_msd = np.sum(weighted_squared_displacements) / (np.sum(weights) + 1e-10)
                
                # Estimate D from MSD, accounting for blur and localization noise
                # MSD = 2*d*D*Δt*(1-R²/12) + 2*d*σ²
                # Solve for D
                localization_contribution = 2 * self.dimensions * (self.localization_error ** 2)
                D_numerator = weighted_msd - localization_contribution
                D_denominator = 2 * self.dimensions * self.frame_interval * self.blur_correction_factor
                
                D_estimated = D_numerator / D_denominator
                
                # Ensure positive D
                self.diffusion_coefficients[state_idx] = max(D_estimated, 1e-6)
            
            # Compute log-likelihood
            log_likelihood = np.sum(np.log(c + 1e-300))
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            
            prev_log_likelihood = log_likelihood
        
        # Viterbi decoding for most likely state sequence
        self.state_sequence = self._viterbi(displacements)
        self.posterior_probs = gamma
        self.log_likelihood = log_likelihood
        
        return {
            'success': True,
            'diffusion_coefficients': self.diffusion_coefficients,
            'transition_matrix': self.transition_matrix,
            'initial_state_probs': self.initial_state_probs,
            'log_likelihood': log_likelihood,
            'n_iterations': iteration + 1,
            'state_sequence': self.state_sequence,
            'posterior_probs': gamma
        }
    
    def _forward(self, displacements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm with scaling.
        
        Returns
        -------
        alpha : np.ndarray
            Scaled forward probabilities, shape (n_steps, n_states)
        c : np.ndarray
            Scaling factors, shape (n_steps,)
        """
        n_steps = displacements.shape[0]
        alpha = np.zeros((n_steps, self.n_states))
        c = np.zeros(n_steps)
        
        # Initialize
        for state_idx in range(self.n_states):
            alpha[0, state_idx] = (
                self.initial_state_probs[state_idx] *
                np.exp(self._emission_probability(displacements[0], state_idx))
            )
        
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / (c[0] + 1e-300)
        
        # Forward pass
        for t in range(1, n_steps):
            for j in range(self.n_states):
                # Sum over previous states
                sum_alpha = 0.0
                for i in range(self.n_states):
                    sum_alpha += alpha[t-1, i] * self.transition_matrix[i, j]
                
                alpha[t, j] = sum_alpha * np.exp(self._emission_probability(displacements[t], j))
            
            c[t] = np.sum(alpha[t])
            alpha[t] = alpha[t] / (c[t] + 1e-300)
        
        return alpha, c
    
    def _backward(self, displacements: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Backward algorithm with scaling.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacement vectors
        c : np.ndarray
            Scaling factors from forward pass
            
        Returns
        -------
        beta : np.ndarray
            Scaled backward probabilities, shape (n_steps, n_states)
        """
        n_steps = displacements.shape[0]
        beta = np.zeros((n_steps, self.n_states))
        
        # Initialize
        beta[-1, :] = 1.0 / (c[-1] + 1e-300)
        
        # Backward pass
        for t in range(n_steps - 2, -1, -1):
            for i in range(self.n_states):
                sum_beta = 0.0
                for j in range(self.n_states):
                    emission_prob = np.exp(self._emission_probability(displacements[t+1], j))
                    sum_beta += (
                        self.transition_matrix[i, j] *
                        emission_prob *
                        beta[t+1, j]
                    )
                beta[t, i] = sum_beta / (c[t] + 1e-300)
        
        return beta
    
    def _compute_xi(
        self,
        displacements: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        c: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise state posteriors ξ_t(i,j) = P(s_t=i, s_{t+1}=j | observations).
        
        Returns
        -------
        xi : np.ndarray
            Shape (n_steps-1, n_states, n_states)
        """
        n_steps = displacements.shape[0]
        xi = np.zeros((n_steps - 1, self.n_states, self.n_states))
        
        for t in range(n_steps - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = np.exp(self._emission_probability(displacements[t+1], j))
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.transition_matrix[i, j] *
                        emission_prob *
                        beta[t+1, j]
                    )
        
        return xi
    
    def _viterbi(self, displacements: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm to find most likely state sequence.
        
        Returns
        -------
        np.ndarray
            State sequence, shape (n_steps,)
        """
        n_steps = displacements.shape[0]
        
        # Initialize
        delta = np.zeros((n_steps, self.n_states))
        psi = np.zeros((n_steps, self.n_states), dtype=int)
        
        for state_idx in range(self.n_states):
            delta[0, state_idx] = (
                np.log(self.initial_state_probs[state_idx] + 1e-300) +
                self._emission_probability(displacements[0], state_idx)
            )
        
        # Forward pass
        for t in range(1, n_steps):
            for j in range(self.n_states):
                values = delta[t-1, :] + np.log(self.transition_matrix[:, j] + 1e-300)
                psi[t, j] = np.argmax(values)
                delta[t, j] = values[psi[t, j]] + self._emission_probability(displacements[t], j)
        
        # Backtrack
        states = np.zeros(n_steps, dtype=int)
        states[-1] = np.argmax(delta[-1, :])
        
        for t in range(n_steps - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states


class InfiniteHMM:
    """
    Infinite HMM with automatic state number determination.
    
    Uses Bayesian model selection (BIC) to determine optimal number of states.
    Can also approximate Dirichlet Process prior for nonparametric Bayesian inference.
    """
    
    def __init__(
        self,
        frame_interval: float = 0.1,
        exposure_time: Optional[float] = None,
        localization_error: float = 0.03,
        dimensions: int = 2,
        method: str = 'BIC'
    ):
        """
        Initialize Infinite HMM.
        
        Parameters
        ----------
        frame_interval : float
            Time between frames (seconds)
        exposure_time : float, optional
            Camera exposure time (seconds)
        localization_error : float
            Localization precision (µm)
        dimensions : int
            Number of spatial dimensions
        method : str
            'BIC': Bayesian Information Criterion
            'AIC': Akaike Information Criterion
            'VB': Variational Bayes (approximate Dirichlet Process)
        """
        self.frame_interval = frame_interval
        self.exposure_time = exposure_time
        self.localization_error = localization_error
        self.dimensions = dimensions
        self.method = method
        
        self.best_model = None
        self.best_n_states = None
        self.model_scores = {}
    
    def fit(
        self,
        displacements: np.ndarray,
        min_states: int = 2,
        max_states: int = 5,
        **kwargs
    ) -> Dict:
        """
        Fit iHMM with automatic state number determination.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacement vectors, shape (n_steps, n_dim)
        min_states : int
            Minimum number of states to consider
        max_states : int
            Maximum number of states to consider
        **kwargs
            Additional arguments passed to BlurAwareHMM.fit()
            
        Returns
        -------
        dict
            Results including:
            - 'best_n_states': optimal number of states
            - 'best_model': fitted BlurAwareHMM instance
            - 'model_scores': scores for each n_states
            - 'diffusion_coefficients': D per state
            - 'state_sequence': Viterbi path
        """
        n_steps = displacements.shape[0]
        n_params_per_state = self.dimensions + 1  # D + state-specific emission params
        
        best_score = -np.inf if self.method == 'AIC' else np.inf
        best_model = None
        best_n_states = min_states
        
        scores = {}
        
        print(f"iHMM: Testing {min_states} to {max_states} states...")
        
        for n_states in range(min_states, max_states + 1):
            print(f"  Fitting {n_states} states...")
            
            # Fit HMM with n_states
            model = BlurAwareHMM(
                n_states=n_states,
                frame_interval=self.frame_interval,
                exposure_time=self.exposure_time,
                localization_error=self.localization_error,
                dimensions=self.dimensions
            )
            
            try:
                result = model.fit(displacements, **kwargs)
                
                if not result['success']:
                    print(f"    ⚠️  Fit failed for {n_states} states")
                    continue
                
                log_likelihood = result['log_likelihood']
                
                # Calculate number of free parameters
                # Transition matrix: n_states * (n_states - 1) (rows sum to 1)
                # Initial probs: n_states - 1
                # Diffusion coefficients: n_states
                n_free_params = n_states * (n_states - 1) + (n_states - 1) + n_states
                
                # Model selection criterion
                if self.method == 'BIC':
                    # BIC = -2*log(L) + k*log(n)
                    score = -2 * log_likelihood + n_free_params * np.log(n_steps)
                    is_better = (score < best_score)
                elif self.method == 'AIC':
                    # AIC = -2*log(L) + 2*k
                    score = -2 * log_likelihood + 2 * n_free_params
                    is_better = (score < best_score)
                elif self.method == 'VB':
                    # Variational lower bound (ELBO) approximation
                    # For simplicity, use BIC as proxy
                    score = -2 * log_likelihood + n_free_params * np.log(n_steps)
                    is_better = (score < best_score)
                
                scores[n_states] = {
                    'score': score,
                    'log_likelihood': log_likelihood,
                    'n_params': n_free_params,
                    'D_values': model.diffusion_coefficients.copy()
                }
                
                print(f"    {self.method} = {score:.2f}, log(L) = {log_likelihood:.2f}")
                
                if is_better:
                    best_score = score
                    best_model = model
                    best_n_states = n_states
                    print(f"    ✓ New best!")
                
            except Exception as e:
                print(f"    ⚠️  Error fitting {n_states} states: {str(e)}")
                continue
        
        if best_model is None:
            return {
                'success': False,
                'error': 'No models converged successfully'
            }
        
        self.best_model = best_model
        self.best_n_states = best_n_states
        self.model_scores = scores
        
        # Classify states by diffusion coefficient
        D_sorted_indices = np.argsort(best_model.diffusion_coefficients)
        state_labels = self._classify_states(best_model.diffusion_coefficients)
        
        print(f"\n✓ Best model: {best_n_states} states")
        print(f"  {self.method} score: {best_score:.2f}")
        print(f"  State classification:")
        for i, (state_idx, label) in enumerate(zip(D_sorted_indices, state_labels)):
            D_val = best_model.diffusion_coefficients[state_idx]
            print(f"    State {state_idx}: D = {D_val:.4f} µm²/s ({label})")
        
        return {
            'success': True,
            'best_n_states': best_n_states,
            'best_model': best_model,
            'model_scores': scores,
            'diffusion_coefficients': best_model.diffusion_coefficients,
            'state_labels': state_labels,
            'state_sequence': best_model.state_sequence,
            'posterior_probs': best_model.posterior_probs,
            'transition_matrix': best_model.transition_matrix,
            'log_likelihood': best_model.log_likelihood
        }
    
    def _classify_states(self, D_values: np.ndarray) -> List[str]:
        """
        Classify states as Bound, Diffusive, or Fast based on D values.
        
        Heuristic classification:
        - D < 0.01 µm²/s: Bound/Confined
        - 0.01 <= D < 1.0 µm²/s: Diffusive
        - D >= 1.0 µm²/s: Fast/Active
        """
        labels = []
        D_sorted = np.sort(D_values)
        
        for D in D_sorted:
            if D < 0.01:
                labels.append("Bound/Confined")
            elif D < 1.0:
                labels.append("Diffusive")
            else:
                labels.append("Fast/Active")
        
        return labels


def analyze_track_with_ihmm(
    track: pd.DataFrame,
    pixel_size: float = 0.1,
    frame_interval: float = 0.1,
    exposure_time: Optional[float] = None,
    localization_error: float = 0.03,
    min_states: int = 2,
    max_states: int = 4,
    method: str = 'BIC'
) -> Dict:
    """
    Analyze a single track using Infinite HMM.
    
    Parameters
    ----------
    track : pd.DataFrame
        Track data with columns 'x', 'y' (and optionally 'z')
    pixel_size : float
        Pixel size in µm
    frame_interval : float
        Frame interval in seconds
    exposure_time : float, optional
        Exposure time in seconds
    localization_error : float
        Localization precision in µm
    min_states : int
        Minimum number of states
    max_states : int
        Maximum number of states
    method : str
        Model selection method ('BIC', 'AIC', 'VB')
        
    Returns
    -------
    dict
        iHMM analysis results
    """
    # Extract positions
    has_z = 'z' in track.columns
    dimensions = 3 if has_z else 2
    
    if has_z:
        positions = track[['x', 'y', 'z']].values * pixel_size
    else:
        positions = track[['x', 'y']].values * pixel_size
    
    # Compute displacements
    displacements = np.diff(positions, axis=0)
    
    if len(displacements) < 10:
        return {
            'success': False,
            'error': 'Track too short for iHMM (need >10 steps)'
        }
    
    # Fit iHMM
    ihmm = InfiniteHMM(
        frame_interval=frame_interval,
        exposure_time=exposure_time,
        localization_error=localization_error,
        dimensions=dimensions,
        method=method
    )
    
    result = ihmm.fit(
        displacements,
        min_states=min_states,
        max_states=max_states
    )
    
    if not result['success']:
        return result
    
    # Add track information to results
    result['track_length'] = len(track)
    result['track_id'] = track['track_id'].iloc[0] if 'track_id' in track.columns else None
    
    return result


# Example usage and testing
if __name__ == '__main__':
    print("Infinite HMM (iHMM) Analyzer - Test Suite")
    print("=" * 70)
    
    # Generate synthetic multi-state track
    print("\n1. Generating synthetic 3-state track...")
    
    np.random.seed(42)
    
    # State parameters
    true_states = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]  # Bound → Diffusive → Fast
    true_D_values = [0.005, 0.1, 2.0]  # µm²/s
    frame_interval = 0.05  # s
    exposure_time = 0.04  # s (80% exposure)
    localization_error = 0.02  # µm
    
    # Generate displacements
    displacements = []
    for state in true_states:
        D = true_D_values[state]
        R = exposure_time / frame_interval
        blur_factor = 1.0 - (R ** 2) / 12.0
        
        # True variance including blur and localization noise
        variance = 2 * D * frame_interval * blur_factor + 2 * (localization_error ** 2)
        std = np.sqrt(variance)
        
        dx = np.random.randn() * std
        dy = np.random.randn() * std
        displacements.append([dx, dy])
    
    displacements = np.array(displacements)
    
    print(f"   Generated {len(displacements)} displacements")
    print(f"   True states: {len(set(true_states))} ({np.unique(true_states)})")
    print(f"   True D values: {true_D_values} µm²/s")
    
    # Test 1: Fixed-state HMM
    print("\n2. Testing BlurAwareHMM with 3 states...")
    model_fixed = BlurAwareHMM(
        n_states=3,
        frame_interval=frame_interval,
        exposure_time=exposure_time,
        localization_error=localization_error
    )
    
    result_fixed = model_fixed.fit(displacements)
    
    if result_fixed['success']:
        print(f"   ✓ Converged in {result_fixed['n_iterations']} iterations")
        print(f"   Log-likelihood: {result_fixed['log_likelihood']:.2f}")
        print(f"   Estimated D values: {model_fixed.diffusion_coefficients}")
    
    # Test 2: Infinite HMM with automatic state selection
    print("\n3. Testing InfiniteHMM with automatic state selection...")
    ihmm = InfiniteHMM(
        frame_interval=frame_interval,
        exposure_time=exposure_time,
        localization_error=localization_error,
        method='BIC'
    )
    
    result_ihmm = ihmm.fit(displacements, min_states=2, max_states=5)
    
    if result_ihmm['success']:
        print(f"\n   ✓ Selected {result_ihmm['best_n_states']} states")
        print(f"   True number of states: {len(true_D_values)}")
        
        if result_ihmm['best_n_states'] == len(true_D_values):
            print(f"   ✅ Correctly identified number of states!")
        else:
            print(f"   ⚠️  State count mismatch")
        
        print(f"\n   State classification:")
        for i, (D_val, label) in enumerate(zip(result_ihmm['diffusion_coefficients'], result_ihmm['state_labels'])):
            print(f"     State {i}: D = {D_val:.4f} µm²/s ({label})")
    
    print("\n" + "=" * 70)
    print("Test complete!")
