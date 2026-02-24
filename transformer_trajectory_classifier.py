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
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.calibration import calibration_curve
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


if TORCH_AVAILABLE:
    class MultiTaskTrajectoryTransformer(nn.Module):
        """
        Multi-task transformer for trajectory classification and fBm parameter inference.

        Outputs both class logits and two continuous values:
        - Hurst exponent H in [0, 1] via sigmoid
        - log(D_alpha) as unconstrained real value
        """

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 3,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            max_seq_len: int = 256
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.d_model = d_model
            self.max_seq_len = max_seq_len

            self.input_projection = nn.Linear(input_dim, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classification_head = nn.Linear(d_model, num_classes)
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)
            )

        def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor with shape (batch, seq_len, input_dim).
            padding_mask : torch.Tensor, optional
                Boolean mask with shape (batch, seq_len), where True marks padded
                positions to ignore.

            Returns
            -------
            dict
                Dictionary with keys ``class_logits`` and ``regression`` where
                ``regression`` has columns [H, log_D_alpha].
            """
            batch_size, seq_len, _ = x.shape
            seq_len = min(seq_len, self.max_seq_len)
            x = x[:, :seq_len, :]

            if padding_mask is not None:
                padding_mask = padding_mask[:, :seq_len]

            x_proj = self.input_projection(x)
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, x_proj], dim=1)

            pos = self.position_embedding[:, :tokens.size(1), :]
            tokens = tokens + pos

            if padding_mask is not None:
                cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding_mask.device)
                src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            else:
                src_key_padding_mask = None

            encoded = self.transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
            pooled = encoded[:, 0, :]

            class_logits = self.classification_head(pooled)
            regression_raw = self.regression_head(pooled)
            hurst = torch.sigmoid(regression_raw[:, 0:1])
            log_d_alpha = regression_raw[:, 1:2]
            regression = torch.cat([hurst, log_d_alpha], dim=1)

            return {
                'class_logits': class_logits,
                'regression': regression
            }


    class TransformerTrajectoryClassifier:
        """
        Wrapper for multi-task transformer trajectory modeling.

        Supports:
        - Class prediction over known motion classes
        - Continuous inference of ``H`` and ``log(D_alpha)``
        - Out-of-distribution (OoD) detection from confidence and entropy
        """

        def __init__(
            self,
            dt: float = 0.1,
            dimensions: int = 2,
            class_names: Optional[List[str]] = None,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 3,
            max_seq_len: int = 256,
            device: Optional[str] = None
        ) -> None:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")

            self.dt = dt
            self.dimensions = dimensions
            self.class_names = class_names or ['brownian', 'confined', 'directed', 'anomalous_sub', 'anomalous_super']
            self.label_to_index = {label: index for index, label in enumerate(self.class_names)}
            self.index_to_label = {index: label for label, index in self.label_to_index.items()}
            self.max_seq_len = max_seq_len
            self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

            self.model = MultiTaskTrajectoryTransformer(
                input_dim=self.dimensions,
                num_classes=len(self.class_names),
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                max_seq_len=max_seq_len
            ).to(self.device)
            self.is_trained = False

        def _to_tensor_sequence(self, trajectory: np.ndarray) -> torch.Tensor:
            array = np.asarray(trajectory, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError("trajectory must have shape (N, dimensions)")

            if array.shape[1] < self.dimensions:
                pad = np.zeros((array.shape[0], self.dimensions - array.shape[1]), dtype=np.float32)
                array = np.hstack([array, pad])
            elif array.shape[1] > self.dimensions:
                array = array[:, :self.dimensions]

            if array.shape[0] > self.max_seq_len:
                array = array[:self.max_seq_len]

            return torch.tensor(array, dtype=torch.float32)

        def _batchify(self, trajectories: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
            sequences = [self._to_tensor_sequence(track) for track in trajectories]
            lengths = [seq.shape[0] for seq in sequences]
            max_len = int(min(max(lengths), self.max_seq_len))
            batch_size = len(sequences)

            batch = torch.zeros((batch_size, max_len, self.dimensions), dtype=torch.float32)
            padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

            for i, seq in enumerate(sequences):
                current_len = min(seq.shape[0], max_len)
                batch[i, :current_len, :] = seq[:current_len]
                padding_mask[i, :current_len] = False

            return batch.to(self.device), padding_mask.to(self.device)

        def fit(
            self,
            trajectories: List[np.ndarray],
            labels: List[str],
            regression_targets: Optional[np.ndarray] = None,
            epochs: int = 15,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            alpha_regression: float = 1.0
        ) -> Dict[str, Any]:
            """
            Train multi-task transformer.

            Parameters
            ----------
            trajectories : list of np.ndarray
                Input trajectories.
            labels : list of str
                Motion class labels.
            regression_targets : np.ndarray, optional
                Array with shape (n_samples, 2) containing [H, log(D_alpha)].
                If omitted, regression loss is skipped.
            epochs : int
                Number of optimization epochs.
            learning_rate : float
                Optimizer learning rate.
            batch_size : int
                Batch size.
            alpha_regression : float
                Weight multiplier for regression loss.

            Returns
            -------
            dict
                Training summary with final losses.
            """
            if len(trajectories) != len(labels):
                return {'success': False, 'error': 'trajectories and labels length mismatch'}
            if len(trajectories) == 0:
                return {'success': False, 'error': 'No training data provided'}

            unknown = sorted(set(labels) - set(self.class_names))
            if unknown:
                return {'success': False, 'error': f'Unknown labels encountered: {unknown}'}

            y_class = torch.tensor([self.label_to_index[label] for label in labels], dtype=torch.long, device=self.device)

            use_regression = regression_targets is not None
            if use_regression:
                reg_targets = np.asarray(regression_targets, dtype=np.float32)
                if reg_targets.shape != (len(trajectories), 2):
                    return {'success': False, 'error': 'regression_targets must have shape (n_samples, 2)'}
                y_reg = torch.tensor(reg_targets, dtype=torch.float32, device=self.device)
            else:
                y_reg = None

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            ce_loss_fn = nn.CrossEntropyLoss()
            mse_loss_fn = nn.MSELoss()

            n_samples = len(trajectories)
            indices = np.arange(n_samples)
            final_total_loss = np.nan
            final_class_loss = np.nan
            final_reg_loss = np.nan

            self.model.train()
            for _ in range(epochs):
                np.random.shuffle(indices)
                epoch_total_losses = []
                epoch_class_losses = []
                epoch_reg_losses = []

                for start in range(0, n_samples, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_tracks = [trajectories[i] for i in batch_idx]
                    batch_x, batch_mask = self._batchify(batch_tracks)
                    batch_y_class = y_class[batch_idx]

                    outputs = self.model(batch_x, padding_mask=batch_mask)
                    class_logits = outputs['class_logits']
                    reg_out = outputs['regression']

                    class_loss = ce_loss_fn(class_logits, batch_y_class)
                    if use_regression:
                        batch_y_reg = y_reg[batch_idx]
                        reg_loss = mse_loss_fn(reg_out, batch_y_reg)
                        total_loss = class_loss + alpha_regression * reg_loss
                    else:
                        reg_loss = torch.tensor(0.0, device=self.device)
                        total_loss = class_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    epoch_total_losses.append(float(total_loss.detach().cpu().item()))
                    epoch_class_losses.append(float(class_loss.detach().cpu().item()))
                    epoch_reg_losses.append(float(reg_loss.detach().cpu().item()))

                final_total_loss = float(np.mean(epoch_total_losses)) if epoch_total_losses else np.nan
                final_class_loss = float(np.mean(epoch_class_losses)) if epoch_class_losses else np.nan
                final_reg_loss = float(np.mean(epoch_reg_losses)) if epoch_reg_losses else np.nan

            self.is_trained = True

            return {
                'success': True,
                'n_samples': n_samples,
                'n_classes': len(self.class_names),
                'use_regression_targets': use_regression,
                'final_total_loss': final_total_loss,
                'final_classification_loss': final_class_loss,
                'final_regression_loss': final_reg_loss
            }

        def predict(
            self,
            trajectories: List[np.ndarray],
            return_proba: bool = False
        ) -> Union[List[str], Tuple[List[str], np.ndarray]]:
            """
            Predict class labels for trajectory list.

            Parameters
            ----------
            trajectories : list of np.ndarray
                Input trajectories.
            return_proba : bool
                Whether to also return class probabilities.

            Returns
            -------
            list[str] or tuple
                Predicted labels, optionally with probability matrix.
            """
            self.model.eval()
            with torch.no_grad():
                x, mask = self._batchify(trajectories)
                outputs = self.model(x, padding_mask=mask)
                probs = F.softmax(outputs['class_logits'], dim=1).cpu().numpy()
                pred_idx = np.argmax(probs, axis=1)
                labels = [self.index_to_label[int(i)] for i in pred_idx]

            if return_proba:
                return labels, probs
            return labels

        def predict_with_ood(
            self,
            trajectory: torch.Tensor,
            confidence_threshold: float = 0.6,
            entropy_threshold: float = 0.75
        ) -> Dict[str, Any]:
            """
            Predict class and continuous parameters with OoD detection.

            OoD is flagged when either:
            - maximum class probability is below ``confidence_threshold``
            - normalized predictive entropy exceeds ``entropy_threshold``

            Parameters
            ----------
            trajectory : torch.Tensor
                Single trajectory tensor with shape (N, D).
            confidence_threshold : float
                Minimum confidence for in-distribution prediction.
            entropy_threshold : float
                Maximum normalized entropy for in-distribution prediction.

            Returns
            -------
            dict
                Prediction summary including OoD decision and [H, log(D_alpha)].
            """
            if trajectory.ndim != 2:
                raise ValueError("trajectory must have shape (N, D)")

            np_track = trajectory.detach().cpu().numpy()
            batch_x, batch_mask = self._batchify([np_track])

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(batch_x, padding_mask=batch_mask)
                logits = outputs['class_logits'][0]
                regression = outputs['regression'][0]

                probs = F.softmax(logits, dim=0)
                max_prob_value, max_index = torch.max(probs, dim=0)

                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                normalized_entropy = float((entropy / np.log(len(self.class_names))).cpu().item())
                max_prob = float(max_prob_value.cpu().item())

                is_ood = (max_prob < confidence_threshold) or (normalized_entropy > entropy_threshold)
                predicted_label = 'Out-of-Distribution' if is_ood else self.index_to_label[int(max_index.cpu().item())]

                hurst = float(regression[0].cpu().item())
                log_d_alpha = float(regression[1].cpu().item())

            return {
                'predicted_label': predicted_label,
                'ood': bool(is_ood),
                'max_probability': max_prob,
                'predictive_entropy': normalized_entropy,
                'confidence_threshold': float(confidence_threshold),
                'entropy_threshold': float(entropy_threshold),
                'class_probabilities': {
                    label: float(prob.cpu().item())
                    for label, prob in zip(self.class_names, probs)
                },
                'H': hurst,
                'log_D_alpha': log_d_alpha
            }


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
        classifier = TransformerTrajectoryClassifier(
            dt=dt,
            dimensions=kwargs.get('dimensions', 2),
            class_names=kwargs.get('class_names'),
            d_model=kwargs.get('d_model', 128),
            n_heads=kwargs.get('n_heads', 4),
            n_layers=kwargs.get('n_layers', 3),
            max_seq_len=kwargs.get('max_seq_len', 256),
            device=kwargs.get('device')
        )
        fit_results = classifier.fit(
            trajectories=trajectories,
            labels=labels,
            regression_targets=kwargs.get('regression_targets'),
            epochs=kwargs.get('epochs', 15),
            learning_rate=kwargs.get('learning_rate', 1e-3),
            batch_size=kwargs.get('batch_size', 32),
            alpha_regression=kwargs.get('alpha_regression', 1.0)
        )
        return classifier, fit_results


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
