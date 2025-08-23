import numpy as np
import pandas as pd
from typing import List

def generate_brownian_motion(n_steps: int = 100, D: float = 1.0, dt: float = 1.0, n_tracks: int = 1) -> pd.DataFrame:
    """
    Generate 2D Brownian motion trajectories.

    Args:
        n_steps: Number of steps in the trajectory.
        D: Diffusion coefficient.
        dt: Time step.
        n_tracks: Number of tracks to generate.

    Returns:
        A pandas DataFrame with the generated tracks.
    """
    tracks = []
    for i in range(n_tracks):
        displacements = np.random.normal(0, np.sqrt(2 * D * dt), (n_steps, 2))
        positions = np.cumsum(displacements, axis=0)

        df = pd.DataFrame(positions, columns=['x', 'y'])
        df['frame'] = np.arange(n_steps)
        df['track_id'] = i
        df['label'] = 'brownian'
        tracks.append(df)

    return pd.concat(tracks, ignore_index=True)

def generate_confined_motion(n_steps: int = 100, D: float = 1.0, dt: float = 1.0, L: float = 10.0, n_tracks: int = 1) -> pd.DataFrame:
    """
    Generate 2D confined motion trajectories within a square box of side 2L.

    Args:
        n_steps: Number of steps in the trajectory.
        D: Diffusion coefficient.
        dt: Time step.
        L: Confinement radius (half-side of the box).
        n_tracks: Number of tracks to generate.

    Returns:
        A pandas DataFrame with the generated tracks.
    """
    tracks = []
    for i in range(n_tracks):
        positions = np.zeros((n_steps, 2))
        for j in range(1, n_steps):
            step = np.random.normal(0, np.sqrt(2 * D * dt), 2)
            new_pos = positions[j-1] + step

            # Confinement
            new_pos = np.clip(new_pos, -L, L)
            positions[j] = new_pos

        df = pd.DataFrame(positions, columns=['x', 'y'])
        df['frame'] = np.arange(n_steps)
        df['track_id'] = i + n_tracks # To avoid track_id collision with brownian
        df['label'] = 'confined'
        tracks.append(df)

    return pd.concat(tracks, ignore_index=True)

def generate_directed_motion(n_steps: int = 100, D: float = 1.0, dt: float = 1.0, v: float = 1.0, n_tracks: int = 1) -> pd.DataFrame:
    """
    Generate 2D directed motion trajectories.

    Args:
        n_steps: Number of steps in the trajectory.
        D: Diffusion coefficient.
        dt: Time step.
        v: Drift velocity.
        n_tracks: Number of tracks to generate.

    Returns:
        A pandas DataFrame with the generated tracks.
    """
    tracks = []
    for i in range(n_tracks):
        displacements = np.random.normal(0, np.sqrt(2 * D * dt), (n_steps, 2))

        # Add drift
        drift = np.zeros((n_steps, 2))
        angle = np.random.uniform(0, 2 * np.pi)
        drift[:, 0] = v * dt * np.cos(angle)
        drift[:, 1] = v * dt * np.sin(angle)

        positions = np.cumsum(displacements + drift, axis=0)

        df = pd.DataFrame(positions, columns=['x', 'y'])
        df['frame'] = np.arange(n_steps)
        df['track_id'] = i + 2 * n_tracks # To avoid track_id collision
        df['label'] = 'directed'
        tracks.append(df)

    return pd.concat(tracks, ignore_index=True)
