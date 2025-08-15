import btrack
import deeptrack as dt
import numpy as np
import pandas as pd

def run_btrack(detections: pd.DataFrame, config_path: str = "models/cell_config.json"):
    """
    Run btrack on a pandas DataFrame of detections.

    Args:
        detections: DataFrame with columns 'x', 'y', 'z', 't', and 'label'.
        config_path: Path to the btrack configuration file.

    Returns:
        A tuple containing:
            - tracks: A list of btrack.Tracklet objects.
            - properties: A dictionary of track properties.
            - graph: A dictionary representing the tracking graph.
    """
    # Convert detections to btrack objects
    objects = btrack.utils.segmentation_to_objects(detections, properties=('label',))

    # Initialize btrack session
    with btrack.BayesianTracker() as tracker:
        # Configure tracker
        tracker.configure_from_file(config_path)
        tracker.max_search_radius = 50

        # Append objects
        tracker.append(objects)

        # Set volume
        tracker.volume = ((0, 1200), (0, 1600), (-1e5, 1e5))

        # Track
        tracker.track_interactive(step_size=100)

        # Generate hypotheses and optimize
        tracker.optimize()

        # Get tracks
        tracks = tracker.tracks

        # Get properties and graph
        properties = tracker.properties
        graph = tracker.graph

        return tracks, properties, graph


def preprocess_with_deeptrack(image: np.ndarray):
    """
    Preprocess an image using DeepTrack2.

    Args:
        image: The image to preprocess.

    Returns:
        The preprocessed image.
    """
    # Create a DeepTrack2 pipeline
    pipeline = dt.Pipeline()
    pipeline.add(dt.features.Normalize())
    pipeline.add(dt.features.Gaussian(sigma=1))

    # Preprocess the image
    preprocessed_image = pipeline.resolve(image)

    return preprocessed_image
