import numpy as np
import pandas as pd
from hmmlearn import hmm

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
