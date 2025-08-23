import numpy as np
import pandas as pd
from typing import List, Dict, Any

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def create_lstm_model(n_features: int, n_timesteps: int, n_classes: int) -> 'tf.keras.Model':
    """
    Create a simple LSTM model for trajectory classification.
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    model = Sequential([
        LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_classifier(tracks: List[np.ndarray], labels: List[str]):
    """
    Train the LSTM classifier on trajectory data.

    Args:
        tracks: A list of trajectories, where each trajectory is a NumPy array of shape (n_points, n_features).
        labels: A list of labels corresponding to each trajectory.
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    # Preprocess data
    # Padding sequences to the same length
    padded_tracks = pad_sequences(tracks, padding='post', dtype='float32')

    # Encode labels
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = tf.keras.utils.to_categorical(integer_encoded)

    n_timesteps, n_features = padded_tracks.shape[1], padded_tracks.shape[2]
    n_classes = onehot_encoder.shape[1]

    # Create model
    model = create_lstm_model(n_features, n_timesteps, n_classes)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(padded_tracks, onehot_encoder, test_size=0.2, random_state=42)

    # Train model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    return model, history, label_encoder

def classify_trajectories(model, tracks: List[np.ndarray], label_encoder) -> List[str]:
    """
    Classify new trajectories using the trained model.
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    padded_tracks = pad_sequences(tracks, padding='post', dtype='float32', maxlen=model.input_shape[1])

    predictions = model.predict(padded_tracks)
    predicted_classes = np.argmax(predictions, axis=1)

    # Decode labels
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    return predicted_labels
