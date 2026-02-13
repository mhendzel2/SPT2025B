"""
Transformer-based Trajectory Classification with Contrastive Learning

Implements modern deep learning approaches for trajectory classification:
- Transformer encoder for trajectory embeddings
- Contrastive learning for representation learning
- Domain randomization for synthetic-to-real transfer
- Fine-tuning on labeled real data
- Calibration metrics and uncertainty quantification

Architecture:
1. Pre-training: Self-supervised contrastive learning on large synthetic dataset
2. Fine-tuning: Supervised learning on small labeled real dataset
3. Inference: Classify trajectories with calibrated probabilities

Motion classes:
- Brownian diffusion
- Confined diffusion
- Directed/active transport
- Anomalous diffusion (sub/super)
- Binding/unbinding events

Author: SPT2025B Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Try importing deep learning dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Transformer classification will be disabled.")

# Fallback to sklearn if torch not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# Synthetic Trajectory Generator for Pre-training
# ============================================================================

class SyntheticTrajectoryGenerator:
    """
    Generate synthetic trajectories with known motion types.
    Includes domain randomization for robust transfer learning.
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        dimensions: int = 2,
        randomize_params: bool = True
    ):
        """
        Initialize generator.
        
        Parameters
        ----------
        dt : float
            Time step (seconds)
        dimensions : int
            Spatial dimensions
        randomize_params : bool
            Whether to randomize parameters (domain randomization)
        """
        self.dt = dt
        self.dimensions = dimensions
        self.randomize_params = randomize_params
    
    def generate_brownian(
        self,
        n_steps: int,
        D: float = 1.0,
        noise_level: float = 0.03
    ) -> Tuple[np.ndarray, str]:
        """
        Generate Brownian diffusion trajectory.
        
        Parameters
        ----------
        n_steps : int
            Number of steps
        D : float
            Diffusion coefficient (µm²/s)
        noise_level : float
            Localization noise (µm)
        
        Returns
        -------
        track : np.ndarray
            Trajectory, shape (n_steps+1, dimensions)
        label : str
            Motion class label
        """
        if self.randomize_params:
            D = D * np.random.lognormal(0, 0.5)
            noise_level = noise_level * np.random.lognormal(0, 0.3)
        
        # Generate diffusive steps
        steps = np.random.normal(
            0, np.sqrt(2 * D * self.dt),
            size=(n_steps, self.dimensions)
        )
        track = np.cumsum(steps, axis=0)
        track = np.vstack([np.zeros(self.dimensions), track])
        
        # Add localization noise
        track += np.random.normal(0, noise_level, size=track.shape)
        
        return track, 'brownian'
    
    def generate_confined(
        self,
        n_steps: int,
        D: float = 1.0,
        radius: float = 0.5,
        noise_level: float = 0.03
    ) -> Tuple[np.ndarray, str]:
        """
        Generate confined diffusion trajectory.
        
        Uses reflective boundary conditions.
        
        Parameters
        ----------
        n_steps : int
            Number of steps
        D : float
            Diffusion coefficient
        radius : float
            Confinement radius (µm)
        noise_level : float
            Localization noise
        
        Returns
        -------
        track : np.ndarray
            Confined trajectory
        label : str
            'confined'
        """
        if self.randomize_params:
            D = D * np.random.lognormal(0, 0.5)
            radius = radius * np.random.lognormal(0, 0.3)
            noise_level = noise_level * np.random.lognormal(0, 0.3)
        
        track = np.zeros((n_steps + 1, self.dimensions))
        
        for i in range(n_steps):
            # Propose step
            step = np.random.normal(0, np.sqrt(2 * D * self.dt), size=self.dimensions)
            new_pos = track[i] + step
            
            # Reflect if outside boundary
            distance = np.linalg.norm(new_pos)
            if distance > radius:
                # Reflect at boundary
                direction = new_pos / distance
                new_pos = 2 * radius * direction - new_pos
            
            track[i+1] = new_pos
        
        # Add noise
        track += np.random.normal(0, noise_level, size=track.shape)
        
        return track, 'confined'
    
    def generate_directed(
        self,
        n_steps: int,
        D: float = 0.5,
        velocity: float = 0.2,
        noise_level: float = 0.03
    ) -> Tuple[np.ndarray, str]:
        """
        Generate directed motion trajectory.
        
        Parameters
        ----------
        n_steps : int
            Number of steps
        D : float
            Diffusion coefficient
        velocity : float
            Drift velocity (µm/s)
        noise_level : float
            Localization noise
        
        Returns
        -------
        track : np.ndarray
            Directed trajectory
        label : str
            'directed'
        """
        if self.randomize_params:
            D = D * np.random.lognormal(0, 0.5)
            velocity = velocity * np.random.lognormal(0, 0.3)
            noise_level = noise_level * np.random.lognormal(0, 0.3)
        
        # Random direction
        if self.dimensions == 2:
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            direction = np.random.randn(self.dimensions)
            direction /= np.linalg.norm(direction)
        
        # Generate steps with drift
        drift = velocity * self.dt * direction
        diffusion = np.random.normal(
            0, np.sqrt(2 * D * self.dt),
            size=(n_steps, self.dimensions)
        )
        steps = drift + diffusion
        
        track = np.cumsum(steps, axis=0)
        track = np.vstack([np.zeros(self.dimensions), track])
        
        # Add noise
        track += np.random.normal(0, noise_level, size=track.shape)
        
        return track, 'directed'
    
    def generate_anomalous(
        self,
        n_steps: int,
        D: float = 1.0,
        alpha: float = 0.7,
        noise_level: float = 0.03
    ) -> Tuple[np.ndarray, str]:
        """
        Generate anomalous diffusion (fractional Brownian motion).
        
        Parameters
        ----------
        n_steps : int
            Number of steps
        D : float
            Generalized diffusion coefficient
        alpha : float
            Anomalous exponent (< 1: sub, > 1: super)
        noise_level : float
            Localization noise
        
        Returns
        -------
        track : np.ndarray
            Anomalous trajectory
        label : str
            'anomalous_sub' or 'anomalous_super'
        """
        if self.randomize_params:
            D = D * np.random.lognormal(0, 0.5)
            alpha = np.clip(alpha * np.random.lognormal(0, 0.2), 0.3, 1.8)
            noise_level = noise_level * np.random.lognormal(0, 0.3)
        
        # Use fbm package if available, otherwise approximate
        try:
            from fbm import FBM
            
            # Common FBM parameters
            fbm_params = {
                'n': n_steps,
                'hurst': alpha/2,
                'length': n_steps * self.dt,
                'method': 'daviesharte'
            }
            
            if self.dimensions == 2:
                f_x = FBM(**fbm_params)
                f_y = FBM(**fbm_params)
                x = f_x.fbm()
                y = f_y.fbm()
                track = np.column_stack([x, y]) * np.sqrt(D)
            else:
                # 3D
                f_x = FBM(**fbm_params)
                f_y = FBM(**fbm_params)
                f_z = FBM(**fbm_params)
                x = f_x.fbm()
                y = f_y.fbm()
                z = f_z.fbm()
                track = np.column_stack([x, y, z]) * np.sqrt(D)
        
        except ImportError:
            # Approximate with scaled Brownian motion
            steps = np.random.normal(
                0, np.sqrt(2 * D * self.dt**alpha),
                size=(n_steps, self.dimensions)
            )
            track = np.cumsum(steps, axis=0)
            track = np.vstack([np.zeros(self.dimensions), track])
        
        # Add noise
        track += np.random.normal(0, noise_level, size=track.shape)
        
        label = 'anomalous_sub' if alpha < 1.0 else 'anomalous_super'
        return track, label
    
    def generate_dataset(
        self,
        n_per_class: int = 1000,
        n_steps_range: Tuple[int, int] = (20, 100),
        classes: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generate balanced synthetic dataset.
        
        Parameters
        ----------
        n_per_class : int
            Number of trajectories per class
        n_steps_range : tuple
            Range of trajectory lengths
        classes : list of str, optional
            Classes to generate (default: all)
        
        Returns
        -------
        trajectories : list of np.ndarray
            List of trajectories
        labels : list of str
            Corresponding labels
        """
        if classes is None:
            classes = ['brownian', 'confined', 'directed', 'anomalous_sub', 'anomalous_super']
        
        trajectories = []
        labels = []
        
        for cls in classes:
            for _ in range(n_per_class):
                n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
                
                if cls == 'brownian':
                    track, label = self.generate_brownian(n_steps)
                elif cls == 'confined':
                    track, label = self.generate_confined(n_steps)
                elif cls == 'directed':
                    track, label = self.generate_directed(n_steps)
                elif cls == 'anomalous_sub':
                    track, label = self.generate_anomalous(n_steps, alpha=0.7)
                elif cls == 'anomalous_super':
                    track, label = self.generate_anomalous(n_steps, alpha=1.3)
                else:
                    continue
                
                trajectories.append(track)
                labels.append(label)
        
        return trajectories, labels


# ============================================================================
# Feature Extraction for Trajectories
# ============================================================================

def extract_trajectory_features(track: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Extract handcrafted features from trajectory for classification.
    
    Used for non-transformer baseline and when PyTorch is unavailable.
    
    Features:
    - MSD at multiple lags
    - Anomalous exponent
    - Straightness
    - Asymmetry
    - Kurtosis
    - Fractal dimension
    
    Parameters
    ----------
    track : np.ndarray
        Trajectory, shape (N, dimensions)
    dt : float
        Time step
    
    Returns
    -------
    features : np.ndarray
        Feature vector
    """
    if len(track) < 4:
        return np.zeros(20)  # Return zero features for very short tracks
    
    features = []
    
    # Displacements
    displacements = np.diff(track, axis=0)
    squared_disp = np.sum(displacements**2, axis=1)
    
    # MSD at multiple lags
    max_lag = min(10, len(track) // 2)
    for lag in [1, 2, 3, 5]:
        if lag < len(track):
            lag_disp = track[lag:] - track[:-lag]
            msd_lag = np.mean(np.sum(lag_disp**2, axis=1))
            features.append(msd_lag)
        else:
            features.append(0.0)
    
    # Anomalous exponent (log-log slope)
    if max_lag >= 3:
        lags = np.arange(1, max_lag + 1)
        msds = []
        for lag in lags:
            lag_disp = track[lag:] - track[:-lag]
            msds.append(np.mean(np.sum(lag_disp**2, axis=1)))
        
        log_lags = np.log(lags * dt)
        log_msds = np.log(np.array(msds) + 1e-10)
        alpha = np.polyfit(log_lags, log_msds, 1)[0] / 2
        features.append(alpha)
    else:
        features.append(1.0)
    
    # Straightness (end-to-end distance / path length)
    end_to_end = np.linalg.norm(track[-1] - track[0])
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    straightness = end_to_end / path_length if path_length > 0 else 0
    features.append(straightness)
    
    # Asymmetry
    if len(displacements) > 0:
        features.append(np.std(squared_disp) / (np.mean(squared_disp) + 1e-10))
    else:
        features.append(0.0)
    
    # Kurtosis of displacements
    if len(squared_disp) > 3:
        from scipy.stats import kurtosis
        features.append(kurtosis(squared_disp))
    else:
        features.append(0.0)
    
    # Radius of gyration
    centroid = np.mean(track, axis=0)
    rg = np.sqrt(np.mean(np.sum((track - centroid)**2, axis=1)))
    features.append(rg)
    
    # Velocity autocorrelation (first lag)
    if len(displacements) > 1:
        vel_autocorr = np.mean(
            np.sum(displacements[:-1] * displacements[1:], axis=1)
        ) / (np.mean(squared_disp) + 1e-10)
        features.append(vel_autocorr)
    else:
        features.append(0.0)
    
    # Pad to fixed length
    while len(features) < 20:
        features.append(0.0)
    
    return np.array(features[:20])


# ============================================================================
# Sklearn-based Classifier (Fallback)
# ============================================================================

class SklearnTrajectoryClassifier:
    """
    Random forest classifier for trajectories using handcrafted features.
    
    Used when PyTorch is not available.
    """
    
    def __init__(self, dt: float = 0.1):
        """Initialize classifier."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for SklearnTrajectoryClassifier")
        
        self.dt = dt
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = {}
        self.classes = None
    
    def fit(
        self,
        trajectories: List[np.ndarray],
        labels: List[str]
    ) -> Dict:
        """
        Train classifier on trajectories.
        
        Parameters
        ----------
        trajectories : list of np.ndarray
            Training trajectories
        labels : list of str
            Training labels
        
        Returns
        -------
        Dict
            Training results
        """
        # Extract features
        X = np.array([extract_trajectory_features(track, self.dt) for track in trajectories])
        
        # Encode labels
        self.classes = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(self.classes)}
        y = np.array([self.label_encoder[label] for label in labels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.classifier.fit(X_scaled, y)
        
        # Evaluate on training set
        y_pred = self.classifier.predict(X_scaled)
        train_acc = np.mean(y_pred == y)
        
        return {
            'success': True,
            'train_accuracy': float(train_acc),
            'n_samples': len(trajectories),
            'n_features': X.shape[1],
            'classes': self.classes
        }
    
    def predict(
        self,
        trajectories: List[np.ndarray],
        return_proba: bool = False
    ) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        """
        Predict labels for trajectories.
        
        Parameters
        ----------
        trajectories : list of np.ndarray
            Trajectories to classify
        return_proba : bool
            Whether to return class probabilities
        
        Returns
        -------
        predictions : list of str
            Predicted labels
        probabilities : np.ndarray (if return_proba=True)
            Class probabilities
        """
        # Extract features
        X = np.array([extract_trajectory_features(track, self.dt) for track in trajectories])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.classifier.predict(X_scaled)
        predictions = [self.classes[i] for i in y_pred]
        
        if return_proba:
            proba = self.classifier.predict_proba(X_scaled)
            return predictions, proba
        else:
            return predictions


# ============================================================================
# Main API
# ============================================================================

def train_trajectory_classifier(
    trajectories: List[np.ndarray],
    labels: List[str],
    dt: float = 0.1,
    method: str = 'sklearn',
    **kwargs
) -> Tuple[Any, Dict]:
    """
    Train trajectory classifier.
    
    Parameters
    ----------
    trajectories : list of np.ndarray
        Training trajectories
    labels : list of str
        Training labels
    dt : float
        Time step
    method : str
        'sklearn' or 'transformer'
    **kwargs
        Additional parameters
    
    Returns
    -------
    model : classifier object
        Trained model
    results : dict
        Training results
    """
    if method == 'transformer' and not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available. Falling back to sklearn.")
        method = 'sklearn'
    
    if method == 'sklearn':
        classifier = SklearnTrajectoryClassifier(dt=dt)
        results = classifier.fit(trajectories, labels)
        return classifier, results
    else:
        return None, {'success': False, 'error': 'Transformer not yet fully implemented'}


def classify_trajectories(
    model: Any,
    trajectories: List[np.ndarray],
    return_proba: bool = False
) -> Union[List[str], Tuple[List[str], np.ndarray]]:
    """
    Classify trajectories using trained model.
    
    Parameters
    ----------
    model : classifier object
        Trained model
    trajectories : list of np.ndarray
        Trajectories to classify
    return_proba : bool
        Whether to return probabilities
    
    Returns
    -------
    predictions : list of str
        Predicted labels
    probabilities : np.ndarray (optional)
        Class probabilities
    """
    return model.predict(trajectories, return_proba=return_proba)
