"""
Enhanced Machine Learning Trajectory Classifier
Provides comprehensive motion classification with multiple ML approaches
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

# Try importing ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML classification features will be limited.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Deep learning classification will not be available.")


# ==================== FEATURE EXTRACTION ====================

def extract_trajectory_features(track_df: pd.DataFrame, pixel_size: float = 0.1, 
                               frame_interval: float = 0.1) -> np.ndarray:
    """
    Extract comprehensive features from a single trajectory for ML classification.
    
    Features extracted:
    - MSD-based: diffusion coefficient, anomalous exponent, linearity
    - Velocity: mean speed, speed variance, directional persistence
    - Geometry: radius of gyration, asphericity, efficiency
    - Dynamics: turning angles, confinement ratio
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Single track with columns [x, y, frame]
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Time between frames in seconds
        
    Returns
    -------
    np.ndarray
        Feature vector (22 features)
    """
    features = []
    
    # Extract coordinates
    coords = track_df[['x', 'y']].values * pixel_size
    n_points = len(coords)
    
    if n_points < 3:
        return np.full(22, np.nan)
    
    # === 1. MSD-based features ===
    max_lag = min(n_points // 4, 50)
    msds = []
    for lag in range(1, max_lag + 1):
        disps = coords[lag:] - coords[:-lag]
        msd = np.mean(np.sum(disps**2, axis=1))
        msds.append(msd)
    
    msds = np.array(msds)
    lags = np.arange(1, max_lag + 1) * frame_interval
    
    # Diffusion coefficient from linear fit
    if len(lags) >= 5:
        D_fit = np.polyfit(lags[:5], msds[:5], 1)
        D_coeff = D_fit[0] / 4.0  # 2D diffusion
    else:
        D_coeff = np.nan
    
    # Anomalous exponent from log-log fit
    if len(lags) >= 5 and np.all(msds > 0):
        alpha_fit = np.polyfit(np.log(lags), np.log(msds), 1)
        alpha = alpha_fit[0]
    else:
        alpha = 1.0
    
    # MSD non-linearity (deviation from linear)
    if len(lags) >= 5:
        linear_pred = D_fit[0] * lags + D_fit[1]
        msd_nonlinearity = np.mean((msds - linear_pred)**2)
    else:
        msd_nonlinearity = 0.0
    
    features.extend([D_coeff, alpha, msd_nonlinearity])
    
    # === 2. Velocity features ===
    velocities = np.diff(coords, axis=0) / frame_interval
    speeds = np.linalg.norm(velocities, axis=1)
    
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)
    
    # Directional persistence (velocity autocorrelation)
    if len(velocities) >= 2:
        v_norm = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-10)
        vac = np.mean([np.dot(v_norm[i], v_norm[i+1]) for i in range(len(v_norm)-1)])
    else:
        vac = 0.0
    
    features.extend([mean_speed, std_speed, max_speed, vac])
    
    # === 3. Geometric features ===
    # Radius of gyration
    centroid = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
    
    # Asphericity (shape anisotropy)
    centered = coords - centroid
    gyration_tensor = np.dot(centered.T, centered) / n_points
    eigenvalues = np.linalg.eigvalsh(gyration_tensor)
    asphericity = (eigenvalues[1] - eigenvalues[0])**2 / (eigenvalues[1] + eigenvalues[0])**2
    
    # Path efficiency (end-to-end distance / path length)
    path_length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    end_to_end = np.linalg.norm(coords[-1] - coords[0])
    efficiency = end_to_end / (path_length + 1e-10)
    
    # Confinement ratio
    max_dist_from_centroid = np.max(np.linalg.norm(coords - centroid, axis=1))
    confinement_ratio = rg / (max_dist_from_centroid + 1e-10)
    
    features.extend([rg, asphericity, efficiency, confinement_ratio])
    
    # === 4. Dynamics features ===
    # Turning angles
    if len(coords) >= 3:
        v1 = coords[1:-1] - coords[:-2]
        v2 = coords[2:] - coords[1:-1]
        v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-10)
        cos_angles = np.sum(v1_norm * v2_norm, axis=1)
        cos_angles = np.clip(cos_angles, -1, 1)
        angles = np.arccos(cos_angles)
        
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
    else:
        mean_angle = 0.0
        std_angle = 0.0
    
    # Fractal dimension (box counting approximation)
    fractal_dim = estimate_fractal_dimension(coords)
    
    # Kurtosis of displacement distribution
    disps_all = coords[1:] - coords[:-1]
    disp_magnitudes = np.linalg.norm(disps_all, axis=1)
    if len(disp_magnitudes) >= 4:
        kurtosis = np.mean((disp_magnitudes - mean_speed)**4) / (std_speed**4 + 1e-10)
    else:
        kurtosis = 0.0
    
    features.extend([mean_angle, std_angle, fractal_dim, kurtosis])
    
    # === 5. Statistical features ===
    # Autocorrelation of x and y positions
    if n_points >= 10:
        x_autocorr = np.corrcoef(coords[:-5, 0], coords[5:, 0])[0, 1]
        y_autocorr = np.corrcoef(coords[:-5, 1], coords[5:, 1])[0, 1]
    else:
        x_autocorr = 0.0
        y_autocorr = 0.0
    
    # Track straightness
    straightness = end_to_end / (path_length + 1e-10)
    
    # Bounding box aspect ratio
    bbox_width = np.ptp(coords[:, 0])
    bbox_height = np.ptp(coords[:, 1])
    bbox_aspect = bbox_width / (bbox_height + 1e-10)
    
    # Number of trajectory points (log scale for normalization)
    log_n_points = np.log10(n_points + 1)
    
    features.extend([x_autocorr, y_autocorr, straightness, bbox_aspect, log_n_points])
    
    return np.array(features)


def estimate_fractal_dimension(coords: np.ndarray, max_boxes: int = 10) -> float:
    """Estimate fractal dimension using box-counting method"""
    try:
        # Normalize coordinates
        coords_norm = (coords - coords.min(axis=0)) / (coords.ptp(axis=0) + 1e-10)
        
        scales = []
        counts = []
        
        for i in range(3, max_boxes):
            box_size = 1.0 / i
            boxes = set()
            for point in coords_norm:
                box_idx = tuple((point // box_size).astype(int))
                boxes.add(box_idx)
            
            if len(boxes) > 0:
                scales.append(box_size)
                counts.append(len(boxes))
        
        if len(scales) >= 3:
            # Fractal dimension from slope of log-log plot
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return -slope
        else:
            return 1.5  # Default for 2D
    except:
        return 1.5


def extract_features_from_tracks_df(tracks_df: pd.DataFrame, pixel_size: float = 0.1, 
                                    frame_interval: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all tracks in a DataFrame.
    
    Returns
    -------
    features : np.ndarray
        Feature matrix (n_tracks, n_features)
    track_ids : np.ndarray
        Track IDs corresponding to each row
    """
    track_ids = tracks_df['track_id'].unique()
    features_list = []
    valid_track_ids = []
    
    for track_id in track_ids:
        track_data = tracks_df[tracks_df['track_id'] == track_id].copy()
        
        if len(track_data) < 3:
            continue
        
        features = extract_trajectory_features(track_data, pixel_size, frame_interval)
        
        if not np.any(np.isnan(features)):
            features_list.append(features)
            valid_track_ids.append(track_id)
    
    if not features_list:
        return np.array([]), np.array([])
    
    return np.array(features_list), np.array(valid_track_ids)


# ==================== SUPERVISED CLASSIFICATION ====================

class TrajectoryClassifier:
    """
    Ensemble trajectory classifier supporting multiple ML algorithms.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier.
        
        Parameters
        ----------
        model_type : str
            One of: 'random_forest', 'gradient_boosting', 'svm', 'lstm'
        """
        if not SKLEARN_AVAILABLE and model_type != 'lstm':
            raise RuntimeError("scikit-learn is required for this classifier type")
        
        if not TENSORFLOW_AVAILABLE and model_type == 'lstm':
            raise RuntimeError("TensorFlow is required for LSTM classifier")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Get descriptive names for all features"""
        return [
            'diffusion_coeff', 'anomalous_exponent', 'msd_nonlinearity',
            'mean_speed', 'std_speed', 'max_speed', 'velocity_persistence',
            'radius_gyration', 'asphericity', 'path_efficiency', 'confinement_ratio',
            'mean_turning_angle', 'std_turning_angle', 'fractal_dimension', 'kurtosis',
            'x_autocorr', 'y_autocorr', 'straightness', 'bbox_aspect', 'log_n_points'
        ]
    
    def _create_model(self, n_classes: int):
        """Create the ML model based on model_type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'lstm':
            # LSTM model (requires sequential data)
            model = Sequential([
                LSTM(64, input_shape=(None, 22), return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(n_classes, activation='softmax')
            ])
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        labels : np.ndarray
            Class labels for each sample
        validation_split : float
            Fraction of data to use for validation
            
        Returns
        -------
        dict
            Training results including accuracy, confusion matrix, etc.
        """
        if len(features) == 0:
            return {'success': False, 'error': 'No features provided'}
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, y_encoded, test_size=validation_split, 
            random_state=42, stratify=y_encoded
        )
        
        # Scale features (except for LSTM which handles raw data)
        if self.model_type != 'lstm':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Create and train model
        self.model = self._create_model(n_classes)
        
        if self.model_type == 'lstm':
            # For LSTM, convert to one-hot and add sequence dimension
            y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
            y_val_cat = tf.keras.utils.to_categorical(y_val, n_classes)
            
            # Add sequence dimension
            X_train_seq = X_train_scaled[:, np.newaxis, :]
            X_val_seq = X_val_scaled[:, np.newaxis, :]
            
            history = self.model.fit(
                X_train_seq, y_train_cat,
                validation_data=(X_val_seq, y_val_cat),
                epochs=50,
                batch_size=32,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0
            )
            
            # Predictions
            y_pred = np.argmax(self.model.predict(X_val_seq, verbose=0), axis=1)
        else:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        class_report = classification_report(
            y_val, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Feature importance (if available)
        feature_importance = None
        if self.model_type in ['random_forest', 'gradient_boosting']:
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        
        return {
            'success': True,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'feature_importance': feature_importance,
            'n_classes': n_classes,
            'class_names': list(self.label_encoder.classes_)
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes for new samples.
        
        Returns
        -------
        labels : np.ndarray
            Predicted class labels
        probabilities : np.ndarray
            Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Scale features
        if self.model_type != 'lstm':
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features[:, np.newaxis, :]
        
        # Predict
        if self.model_type == 'lstm':
            probs = self.model.predict(features_scaled, verbose=0)
            predictions = np.argmax(probs, axis=1)
        else:
            predictions = self.model.predict(features_scaled)
            probs = self.model.predict_proba(features_scaled)
        
        # Decode labels
        labels = self.label_encoder.inverse_transform(predictions)
        
        return labels, probs


# ==================== UNSUPERVISED CLUSTERING ====================

class TrajectoryClusterer:
    """
    Unsupervised clustering for trajectory classification.
    """
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 4):
        """
        Parameters
        ----------
        method : str
            'kmeans' or 'dbscan'
        n_clusters : int
            Number of clusters (for kmeans)
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for clustering")
        
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Fit clustering model and return cluster assignments.
        
        Returns
        -------
        dict
            Clustering results including labels, silhouette score, etc.
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = self.model.fit_predict(features_scaled)
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
            labels = self.model.fit_predict(features_scaled)
            self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Calculate silhouette score
        if self.n_clusters > 1:
            silhouette = silhouette_score(features_scaled, labels)
        else:
            silhouette = -1.0
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        return {
            'success': True,
            'labels': labels,
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Assign new samples to clusters"""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features_scaled = self.scaler.transform(features)
        
        if self.method == 'kmeans':
            return self.model.predict(features_scaled)
        elif self.method == 'dbscan':
            # DBSCAN doesn't support predict, use closest cluster
            from sklearn.metrics.pairwise import euclidean_distances
            core_samples = self.model.components_
            distances = euclidean_distances(features_scaled, core_samples)
            return self.model.labels_[np.argmin(distances, axis=1)]


# ==================== HIGH-LEVEL API ====================

def classify_motion_types(tracks_df: pd.DataFrame, pixel_size: float = 0.1, 
                         frame_interval: float = 0.1,
                         method: str = 'unsupervised',
                         model_type: str = 'random_forest',
                         labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    High-level function to classify trajectory motion types.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks with columns [track_id, frame, x, y]
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    method : str
        'supervised' or 'unsupervised'
    model_type : str
        For supervised: 'random_forest', 'gradient_boosting', 'svm', 'lstm'
        For unsupervised: 'kmeans', 'dbscan'
    labels : np.ndarray, optional
        Ground truth labels for supervised learning
        
    Returns
    -------
    dict
        Classification results with predicted labels and metrics
    """
    # Extract features
    features, track_ids = extract_features_from_tracks_df(
        tracks_df, pixel_size, frame_interval
    )
    
    if len(features) == 0:
        return {
            'success': False,
            'error': 'No valid trajectories for classification'
        }
    
    if method == 'supervised':
        if labels is None:
            return {
                'success': False,
                'error': 'Labels required for supervised learning'
            }
        
        classifier = TrajectoryClassifier(model_type=model_type)
        train_result = classifier.train(features, labels)
        
        if not train_result['success']:
            return train_result
        
        # Predict on all data
        pred_labels, probs = classifier.predict(features)
        
        return {
            'success': True,
            'method': 'supervised',
            'model_type': model_type,
            'track_ids': track_ids,
            'predicted_labels': pred_labels,
            'probabilities': probs,
            'training_results': train_result
        }
        
    elif method == 'unsupervised':
        n_clusters = 4 if model_type == 'kmeans' else None
        clusterer = TrajectoryClusterer(method=model_type, n_clusters=n_clusters)
        cluster_result = clusterer.fit(features)
        
        return {
            'success': True,
            'method': 'unsupervised',
            'model_type': model_type,
            'track_ids': track_ids,
            'predicted_labels': cluster_result['labels'],
            'clustering_results': cluster_result,
            'features': features
        }
    
    else:
        return {
            'success': False,
            'error': f'Unknown method: {method}'
        }
