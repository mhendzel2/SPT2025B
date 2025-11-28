"""
Constants and configuration parameters for SPT Analysis application.
Centralizes magic numbers and default values for better maintainability.
"""

# UI Page Names
PAGE_HOME = "Home"
PAGE_DATA_LOADING = "Data Loading"
PAGE_ANALYSIS = "Analysis"
PAGE_VISUALIZATION = "Visualization"
PAGE_PROJECT_MANAGEMENT = "Project Management"
PAGE_MD_INTEGRATION = "MD Integration"
PAGE_TEST_DATA = "Test Data"

# Analysis Types
ANALYSIS_DIFFUSION = "Diffusion Analysis"
ANALYSIS_MOTION = "Motion Analysis"
ANALYSIS_TRACK_STATS = "Track Statistics"
ANALYSIS_SPATIAL_DIST = "Spatial Distribution"
ANALYSIS_CLUSTERING = "Clustering"
ANALYSIS_ANOMALY = "Anomaly Detection"
ANALYSIS_CHANGEPOINT = "Changepoint Detection"
ANALYSIS_CORRELATIVE = "Correlative Analysis"

# Segmentation Methods
SEGMENTATION_OTSU = "Otsu"
SEGMENTATION_ADAPTIVE = "Adaptive"
SEGMENTATION_WATERSHED = "Watershed"
SEGMENTATION_CUSTOM = "Custom Threshold"

# Detection Methods
DETECTION_LOG = "Laplacian of Gaussian"
DETECTION_DOG = "Difference of Gaussians"
DETECTION_ENHANCED = "Enhanced Detection"

# File Types
SUPPORTED_IMAGE_FORMATS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
SUPPORTED_TRACK_FORMATS = ['.csv', '.xlsx', '.xml']
SUPPORTED_SPECIAL_FORMATS = ['.mvd2', '.aiix', '.aisf']

# Default Parameters
DEFAULT_PIXEL_SIZE = 0.1  # microns
DEFAULT_FRAME_INTERVAL = 0.1  # seconds
DEFAULT_MIN_TRACK_LENGTH = 5
DEFAULT_MAX_LAG_TIME = 20
DEFAULT_PARTICLE_SIZE = 3.0  # pixels
DEFAULT_SEARCH_RADIUS = 20.0  # pixels

# Mathematical Constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# Diffusion Analysis Constants
DEFAULT_SHORT_LAG_CUTOFF = 4
MIN_POINTS_ANOMALOUS = 5
MIN_POINTS_CONFINEMENT = 8
ALPHA_SUBDIFFUSIVE_THRESHOLD = 0.9
ALPHA_SUPERDIFFUSIVE_THRESHOLD = 1.1
REVERSAL_ANGLE_THRESHOLD = 1.5708  # pi/2 radians

# UI Configuration
MAX_CHANNELS_TO_DETECT = 4
MAX_SAMPLE_TRACKS_DISPLAY = 1000
DEFAULT_DENSITY_CLASSES = 3
MAX_TIFF_FRAMES = 10000

# Analysis Thresholds
MIN_TRACK_LENGTH_FOR_MSD = 10
MIN_TRACK_LENGTH_FOR_MOTION = 5
MIN_TRACK_LENGTH_FOR_CLUSTERING = 3
CHANGEPOINT_PERSISTENCE_THRESHOLD = 0.5
SHORT_LAG_CUTOFF_MIN = 2

# File Processing
HEADER_DETECTION_LINES = 15
MAX_HEADER_ROWS = 10
CSV_CHUNK_SIZE = 1000

# Numerical Stability
DIVISION_BY_ZERO_REPLACEMENT = 1e-9
MIN_CONFINEMENT_LENGTH = 1e-9
MIN_DIFFUSION_COEFFICIENT = 1e-15

# Optimization Parameters
KMEANS_N_INIT = 10
OPTIMIZATION_MAX_WORKERS = 4
PARAMETER_GRID_POINTS = 6

# Visualization
DEFAULT_COLORMAP = "viridis"
ANOMALY_COLORS = {
    'normal': 'blue',
    'velocity_anomaly': 'red',
    'confinement': 'orange',
    'directed_motion': 'green',
    'immobile': 'gray'
}

# Session State Keys
SESSION_PIXEL_SIZE = 'global_pixel_size'
SESSION_FRAME_INTERVAL = 'global_frame_interval'
SESSION_CURRENT_PIXEL_SIZE = 'current_pixel_size'
SESSION_CURRENT_FRAME_INTERVAL = 'current_frame_interval'
SESSION_TRACK_COORDS_IN_MICRONS = 'track_coordinates_in_microns'
SESSION_TRACKS_DATA = 'tracks_data'
SESSION_IMAGE_DATA = 'image_data'
SESSION_AVAILABLE_MASKS = 'available_masks'
SESSION_ANALYSIS_RESULTS = 'analysis_results'

# Error Messages
ERROR_NO_TRACKS = "No track data loaded. Please load track data first."
ERROR_NO_IMAGE = "No image data loaded. Please load image data first."
ERROR_SHORT_TRACKS = "Tracks too short for analysis. Minimum length: {}"
ERROR_INVALID_PARAMETERS = "Invalid analysis parameters provided."
ERROR_FILE_LOAD_FAILED = "Failed to load file: {}"

# Success Messages
SUCCESS_FILE_LOADED = "File loaded successfully: {}"
SUCCESS_ANALYSIS_COMPLETE = "Analysis completed successfully"
SUCCESS_PARAMETERS_UPDATED = "Parameters updated successfully"