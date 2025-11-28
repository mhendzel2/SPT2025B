import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

try:
    from hmmlearn import hmm
    _HMMLEARN_AVAILABLE = True
except ImportError as e:
    _HMMLEARN_AVAILABLE = False
    _hmm_import_error = e

def ensure_hmmlearn():
    if not _HMMLEARN_AVAILABLE:
        raise RuntimeError(
            "hmmlearn is required for HMM functionality but is not installed. "
            "Install with: pip install hmmlearn\nOriginal error: "
            f"{_hmm_import_error}"
        )

def fit_hmm(tracks_df: pd.DataFrame, n_states: int = 3, n_iter: int = 100):
    """
    Fit a Hidden Markov Model to track data.

    Args:
        tracks_df: DataFrame of tracks with 'track_id', 'x', and 'y' columns.
        n_states: The number of hidden states to model.
        n_iter: The number of iterations to perform.

    Returns:
        A tuple containing:
            - model: The fitted HMM model.
            - predictions: A dictionary mapping track_id to the predicted state sequence.
    """
    if not _HMMLEARN_AVAILABLE:
        # Lightweight graceful fallback (or call ensure_hmmlearn() to hard fail)
        raise RuntimeError("fit_hmm unavailable: hmmlearn not installed.")

    # Prepare the data for hmmlearn
    # We'll use the displacements (dx, dy) as our observed features
    all_displacements = []
    track_lengths = []
    track_ids = []

    for track_id, track in tracks_df.groupby('track_id'):
        displacements = np.diff(track[['x', 'y']].values, axis=0)
        if len(displacements) > 0:
            all_displacements.append(displacements)
            track_lengths.append(len(displacements))
            track_ids.append(track_id)

    if not all_displacements:
        return None, None

    X = np.concatenate(all_displacements)

    # Create and fit the HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter)
    model.fit(X, lengths=track_lengths)

    # Get the predicted states for each track
    predictions = {}
    for i, track_id in enumerate(track_ids):
        track_displacements = all_displacements[i]
        states = model.predict(track_displacements)
        predictions[track_id] = states

    return model, predictions


def optimize_hmm_states(tracks_df: pd.DataFrame, max_states: int = 5, n_iter: int = 100) -> Tuple[int, Dict[int, float]]:
    """
    Determine optimal number of HMM states using BIC.
    
    Args:
        tracks_df: DataFrame of tracks
        max_states: Maximum number of states to test
        n_iter: Number of iterations for HMM fitting
        
    Returns:
        Tuple of (optimal_n_states, bic_scores)
    """
    ensure_hmmlearn()
    
    # Prepare data (same as fit_hmm)
    all_displacements = []
    track_lengths = []
    
    for track_id, track in tracks_df.groupby('track_id'):
        displacements = np.diff(track[['x', 'y']].values, axis=0)
        if len(displacements) > 0:
            all_displacements.append(displacements)
            track_lengths.append(len(displacements))
            
    if not all_displacements:
        return 1, {}
        
    X = np.concatenate(all_displacements)
    
    bic_scores = {}
    best_bic = float('inf')
    best_n = 1
    
    for n in range(1, max_states + 1):
        try:
            model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=n_iter)
            model.fit(X, lengths=track_lengths)
            
            # Calculate BIC
            # BIC = -2 * log_likelihood + k * log(N)
            log_likelihood = model.score(X, lengths=track_lengths)
            n_features = X.shape[1]
            
            # Number of parameters:
            # Transition matrix: n * (n - 1)
            # Means: n * n_features
            # Covariances (full): n * n_features * (n_features + 1) / 2
            n_params = n * (n - 1) + n * n_features + n * n_features * (n_features + 1) / 2
            n_samples = len(X)
            
            bic = -2 * log_likelihood + n_params * np.log(n_samples)
            bic_scores[n] = bic
            
            if bic < best_bic:
                best_bic = bic
                best_n = n
                
        except Exception as e:
            # Log error but continue trying other states
            print(f"Failed to fit HMM with {n} states: {e}")
            continue
            
    return best_n, bic_scores
