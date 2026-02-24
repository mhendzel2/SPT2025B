import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def calculate_population_metrics(tracks_df, pixel_size=0.1, frame_interval=0.1):
    """
    Calculate population-level metrics from track data.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data
    pixel_size : float
        Pixel size in micrometers
    frame_interval : float
        Frame interval in seconds

    Returns
    -------
    dict
        Dictionary with population metrics
    """
    if tracks_df is None or len(tracks_df) == 0:
        return {
            'mean_displacement': 0,
            'mean_velocity': 0,
            'total_tracks': 0,
            'total_points': 0,
            'avg_track_length': 0
        }

    # Get basic statistics
    unique_tracks = tracks_df['track_id'].nunique()
    total_points = len(tracks_df)
    avg_track_length = total_points / unique_tracks if unique_tracks > 0 else 0

    # Calculate displacements for each track
    displacements = []
    velocities = []

    # Group by track_id to calculate metrics per track
    for track_id, track_data in tracks_df.groupby('track_id'):
        if len(track_data) < 2:
            continue

        # Sort by frame
        track_data = track_data.sort_values('frame')

        # Calculate displacements
        dx = np.diff(track_data['x'].values) * pixel_size
        dy = np.diff(track_data['y'].values) * pixel_size
        dt = np.diff(track_data['frame'].values) * frame_interval

        # Calculate step displacements
        step_displacements = np.sqrt(dx**2 + dy**2)
        displacements.extend(step_displacements)

        # Calculate velocities (displacement/time)
        step_velocities = step_displacements / dt
        velocities.extend(step_velocities)

    # Calculate population means
    mean_displacement = np.mean(displacements) if displacements else 0
    mean_velocity = np.mean(velocities) if velocities else 0

    return {
        'mean_displacement': mean_displacement,
        'mean_velocity': mean_velocity,
        'total_tracks': unique_tracks,
        'total_points': total_points,
        'avg_track_length': avg_track_length
    }

def perform_class_based_analysis(tracks_df: pd.DataFrame, mask_name: str, classes: list,
                                analysis_type: str, segmentation_method: str):
    """Perform comprehensive analysis for each class from the selected segmentation method."""
    results = {}

    # Add class information to tracks
    tracks_with_classes = apply_mask_to_tracks(tracks_df, mask_name, classes)

    if 'class' not in tracks_with_classes.columns:
        return {"error": "Failed to apply mask classification to tracks"}

    # Get class names based on segmentation method
    if segmentation_method == "Simple Segmentation (Binary)":
        class_names = {0: "Background", 1: "Nucleus"}
    elif segmentation_method == "Two-Step Segmentation (3 Classes)":
        class_names = {0: "Background", 1: "Class 1", 2: "Class 2"}
    else:  # Nuclear Density Mapping
        class_names = {0: "Background", 1: "Low Density", 2: "High Density"}

    # Analyze each class separately
    for class_id in classes:
        class_name = class_names.get(class_id, f"Class {class_id}")
        class_tracks = tracks_with_classes[tracks_with_classes['class'] == class_id]

        if len(class_tracks) == 0:
            results[class_name] = {"error": f"No tracks found in {class_name}"}
            continue

        # Basic statistics
        n_tracks = len(np.unique(class_tracks['track_id']))
        n_points = len(class_tracks)
        avg_track_length = n_points / n_tracks if n_tracks > 0 else 0

        # Calculate basic motion statistics
        class_results = {
            "n_tracks": n_tracks,
            "n_points": n_points,
            "avg_track_length": avg_track_length
        }

        results[class_name] = class_results

    # Add overall statistics
    results["overall"] = {
        "segmentation_method": segmentation_method,
        "total_tracks": len(np.unique(tracks_with_classes['track_id'])),
        "total_points": len(tracks_with_classes),
        "classes_analyzed": list(class_names.values())
    }

    return results

def apply_mask_to_tracks(tracks_df: pd.DataFrame, mask_name: str, selected_classes: list):
    """Add class labels to tracks and optionally filter by selected classes."""
    if mask_name not in st.session_state.available_masks:
        return tracks_df

    mask = st.session_state.available_masks[mask_name]

    # Add class information to all track points
    tracks_with_classes = tracks_df.copy()
    tracks_with_classes['class'] = 0  # Default background class

    # Get current pixel size for coordinate conversion
    pixel_size = st.session_state.get('global_pixel_size', st.session_state.get('current_pixel_size', 0.1))
    track_coords_in_microns = st.session_state.get('track_coordinates_in_microns', False)

    for idx, row in tracks_with_classes.iterrows():
        # Convert coordinates to pixel indices for mask lookup
        if track_coords_in_microns:
            # Track coordinates are in microns - convert to pixels for mask indexing
            x_pixel = int(row['x'] / pixel_size)
            y_pixel = int(row['y'] / pixel_size)
        else:
            # Track coordinates are already in pixels
            x_pixel = int(row['x'])
            y_pixel = int(row['y'])

        # Check bounds and assign class
        if 0 <= x_pixel < mask.shape[1] and 0 <= y_pixel < mask.shape[0]:
            tracks_with_classes.loc[idx, 'class'] = mask[y_pixel, x_pixel]

    # Filter by selected classes if specified
    if selected_classes:
        filtered_tracks = tracks_with_classes[tracks_with_classes['class'].isin(selected_classes)]
        return filtered_tracks
    else:
        return tracks_with_classes

def apply_mask_to_image_analysis(image: np.ndarray, mask_name: str, selected_classes: list):
    """Apply mask to image for region-specific analysis."""
    if mask_name not in st.session_state.available_masks:
        return image

    mask = st.session_state.available_masks[mask_name]

    # Create combined mask for selected classes
    combined_mask = np.zeros_like(mask, dtype=bool)
    for class_val in selected_classes:
        combined_mask |= (mask == class_val)

    # Apply mask to image
    masked_image = image.copy()
    masked_image[~combined_mask] = 0  # Set pixels outside mask to 0

    return masked_image

def plot_density_map(tracks_df):
    """Create a density map visualization for particle tracks"""
    try:
        if tracks_df.empty:
            st.warning("No track data available for density map.")
            return None

        # Get position data
        x = tracks_df['x'].values
        y = tracks_df['y'].values

        if len(x) < 3:
            st.warning("Need at least 3 points for density map.")
            return None

        # Create 2D histogram for density
        import numpy as np

        # Calculate optimal number of bins
        n_bins = min(50, int(np.sqrt(len(x))))

        # Create histogram
        counts, xedges, yedges = np.histogram2d(x, y, bins=n_bins)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=counts.T,
            x=xedges[:-1],
            y=yedges[:-1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Particle Count")
        ))

        # Add scatter overlay
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(color='white', size=2, opacity=0.5),
            name='Particles',
            showlegend=True
        ))

        fig.update_layout(
            title="Particle Density Map",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=600,
            height=500
        )

        return fig

    except Exception as e:
        st.error(f"Error creating density map: {str(e)}")
        return None

def convert_coordinates_to_microns(tracks_df, pixel_size):
    """
    Convert coordinates to microns based on track coordinate units setting.

    Parameters:
    -----------
    tracks_df : pd.DataFrame
        Track data with x, y coordinates
    pixel_size : float
        Pixel size in microns

    Returns:
    --------
    pd.DataFrame
        Track data with coordinates in microns
    """
    if tracks_df is None or len(tracks_df) == 0:
        return tracks_df

    tracks_converted = tracks_df.copy()

    # Check if track coordinates are already in microns
    track_coords_in_microns = st.session_state.get('track_coordinates_in_microns', False)

    if not track_coords_in_microns:
        # Track coordinates are in pixels - convert to microns
        for col in ['x', 'y', 'z']:
            if col in tracks_converted.columns:
                tracks_converted[col] = tracks_converted[col] * pixel_size

    # If track coordinates are already in microns, no conversion needed for tracks
    # Image pixel coordinates will still need conversion when comparing with tracks

    return tracks_converted

def normalize_image_for_display(image):
    """Normalize image for proper display in Streamlit."""
    if image is None:
        return None

    # Handle different image shapes
    if len(image.shape) == 3:
        if image.shape[2] == 2:
            # Convert 2-channel to single channel by taking the first channel
            image = image[:, :, 0]
        elif image.shape[2] > 4:
            # Convert multi-channel to single channel by taking the first channel
            image = image[:, :, 0]

    # Normalize to uint8 if needed
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        img_min, img_max = np.min(image), np.max(image)
        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        return normalized
    return image

def process_image_data(image_data):
    """Process image data to ensure it's a proper numpy array for analysis."""
    if image_data is None:
        return None

    try:
        # Handle different image data formats
        if isinstance(image_data, list):
            if len(image_data) == 0:
                return None
            # If it's a list (multi-channel/multi-frame), take the first channel/frame
            first_item = image_data[0]
            if isinstance(first_item, np.ndarray):
                current_image = first_item
            else:
                current_image = np.array(first_item)
        elif isinstance(image_data, np.ndarray):
            current_image = image_data
        else:
            current_image = np.array(image_data)

        # Ensure it's a valid array
        if current_image.size == 0:
            return None

        # Ensure it's a 2D array for processing
        if current_image.ndim > 2:
            current_image = current_image[:, :, 0] if current_image.shape[2] > 0 else current_image.squeeze()
        elif current_image.ndim == 1:
            # If it's 1D, can't process as an image
            return None

        return current_image
    except Exception as e:
        st.error(f"Error processing image data: {str(e)}")
        return None
