import numpy as np
import pandas as pd
from typing import Dict, Any

try:
    from giotto_tda.homology import VietorisRipsPersistence
    from giotto_tda.diagrams import PersistenceDiagram
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

def perform_tda(points: np.ndarray, max_edge_length: float = 10.0) -> Dict[str, Any]:
    """
    Perform Topological Data Analysis (TDA) on a point cloud.

    Args:
        points: A NumPy array of shape (n_points, n_dims) representing the point cloud.
        max_edge_length: The maximum edge length for the Vietoris-Rips complex.

    Returns:
        A dictionary containing the persistence diagram.
    """
    if not TDA_AVAILABLE:
        raise RuntimeError("giotto-tda is not available.")

    if points.shape[0] < 3:
        return {'success': False, 'error': 'Not enough points for TDA.'}

    # Vietoris-Rips persistence
    vr_persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=max_edge_length)
    diagrams = vr_persistence.fit_transform([points])

    return {
        'success': True,
        'diagram': diagrams[0],
        'homology_dimensions': [0, 1, 2]
    }
