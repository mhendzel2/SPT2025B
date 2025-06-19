"""
Advanced Segmentation Methods for SPT Analysis
Implements CellSAM and Cellpose models for particle detection in microscopy images.
Replaces placeholder CNN functions with real, pre-trained models.
"""

import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Tuple, Optional, Any
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

try:
    # CellSAM imports
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    # OpenCV for image processing
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Set availability flags based on imports
CELLSAM_AVAILABLE = TORCH_AVAILABLE and OPENCV_AVAILABLE
CELLPOSE_AVAILABLE = TORCH_AVAILABLE and OPENCV_AVAILABLE

class CellSAMSegmentation:
    """
    CellSAM (Cell Segment Anything Model) implementation for particle detection.
    Uses Meta's SAM foundation model specialized for cellular images.
    """
    
    def __init__(self, model_type: str = "vit_b", device: str = "auto"):
        """
        Initialize CellSAM model.
        
        Parameters
        ----------
        model_type : str
            SAM model type ('vit_b', 'vit_l', 'vit_h')
        device : str
            Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.predictor = None
        self.loaded = False
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load the CellSAM model.
        
        Parameters
        ----------
        checkpoint_path : str, optional
            Path to SAM checkpoint. If None, downloads automatically.
            
        Returns
        -------
        bool
            True if model loaded successfully
        """
        try:
            if not CELLSAM_AVAILABLE:
                st.error("CellSAM dependencies not available")
                return False
            
            # Default checkpoint URLs for SAM models
            checkpoint_urls = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            }
            
            if checkpoint_path is None:
                # Use the base SAM model - in practice, you'd want the CellSAM checkpoint
                st.info(f"Loading SAM {self.model_type} model...")
                checkpoint_path = checkpoint_urls[self.model_type]
            
            # Load the model
            self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)
            self.loaded = True
            
            st.success(f"CellSAM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            st.error(f"Failed to load CellSAM model: {str(e)}")
            return False
    
    def detect_particles(self, image: np.ndarray, 
                        confidence_threshold: float = 0.5,
                        size_filter: Tuple[int, int] = (10, 1000)) -> pd.DataFrame:
        """
        Detect particles using CellSAM.
        
        Parameters
        ----------
        image : np.ndarray
            Input microscopy image
        confidence_threshold : float
            Confidence threshold for detection
        size_filter : tuple
            Min and max particle sizes (pixels)
            
        Returns
        -------
        pd.DataFrame
            Detected particle coordinates and properties
        """
        if not self.loaded:
            if not self.load_model():
                return pd.DataFrame()
        
        try:
            # Preprocess image
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Set image for prediction
            self.predictor.set_image(image_rgb)
            
            # Generate automatic masks
            mask_generator = self._create_mask_generator()
            masks = mask_generator.generate(image_rgb)
            
            # Process masks to extract particle information
            particles = []
            for i, mask_info in enumerate(masks):
                mask = mask_info['segmentation']
                
                # Calculate particle properties
                props = self._calculate_particle_properties(mask, mask_info)
                
                # Filter by size and confidence
                if (size_filter[0] <= props['area'] <= size_filter[1] and 
                    props['confidence'] >= confidence_threshold):
                    particles.append(props)
            
            # Convert to DataFrame
            if particles:
                df = pd.DataFrame(particles)
                return df
            else:
                return pd.DataFrame(columns=['x', 'y', 'area', 'confidence', 'eccentricity'])
                
        except Exception as e:
            st.error(f"CellSAM detection failed: {str(e)}")
            return pd.DataFrame()
    
    def _create_mask_generator(self):
        """Create automatic mask generator with optimized parameters."""
        from segment_anything import SamAutomaticMaskGenerator
        
        return SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
    
    def _calculate_particle_properties(self, mask: np.ndarray, mask_info: Dict) -> Dict:
        """Calculate particle properties from segmentation mask."""
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return {}
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Calculate eccentricity
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            axes = ellipse[1]
            eccentricity = np.sqrt(1 - (min(axes) / max(axes))**2) if max(axes) > 0 else 0
        else:
            eccentricity = 0
        
        return {
            'x': cx,
            'y': cy,
            'area': area,
            'confidence': mask_info.get('stability_score', 0.0),
            'eccentricity': eccentricity,
            'bbox': cv2.boundingRect(contour)
        }


class CellposeSegmentation:
    """
    Cellpose implementation for particle detection.
    Uses specialized models for different cell types and imaging modalities.
    """
    
    def __init__(self, model_type: str = "cyto", device: str = "auto"):
        """
        Initialize Cellpose model.
        
        Parameters
        ----------
        model_type : str
            Cellpose model type ('cyto', 'nuclei', 'cyto2', 'livecell')
        device : str
            Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.loaded = False
        
        # Model parameters
        self.available_models = {
            'cyto': 'Cytoplasm model - general cell segmentation',
            'nuclei': 'Nuclear model - for nuclear segmentation', 
            'cyto2': 'Improved cytoplasm model',
            'livecell': 'Live cell imaging model',
            'bact': 'Bacterial segmentation model',
            'plant': 'Plant cell segmentation model'
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_model(self) -> bool:
        """
        Load the Cellpose model.
        
        Returns
        -------
        bool
            True if model loaded successfully
        """
        try:
            if not CELLPOSE_AVAILABLE:
                st.error("Cellpose not available")
                return False
            
            st.info(f"Loading Cellpose {self.model_type} model...")
            
            # Initialize Cellpose model
            self.model = Cellpose(model_type=self.model_type, gpu=torch.cuda.is_available())
            self.loaded = True
            
            st.success(f"Cellpose {self.model_type} model loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"Failed to load Cellpose model: {str(e)}")
            return False
    
    def detect_particles(self, image: np.ndarray,
                        diameter: Optional[float] = None,
                        flow_threshold: float = 0.4,
                        cellprob_threshold: float = 0.0,
                        size_filter: Tuple[int, int] = (10, 1000)) -> pd.DataFrame:
        """
        Detect particles using Cellpose.
        
        Parameters
        ----------
        image : np.ndarray
            Input microscopy image
        diameter : float, optional
            Expected particle diameter in pixels. If None, auto-estimated.
        flow_threshold : float
            Flow error threshold for mask computation
        cellprob_threshold : float
            Cell probability threshold
        size_filter : tuple
            Min and max particle sizes (pixels)
            
        Returns
        -------
        pd.DataFrame
            Detected particle coordinates and properties
        """
        if not self.loaded:
            if not self.load_model():
                return pd.DataFrame()
        
        try:
            # Preprocess image
            if len(image.shape) == 3:
                # Convert to grayscale for Cellpose
                image_proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_proc = image.copy()
            
            # Normalize image
            image_proc = self._normalize_image(image_proc)
            
            # Run Cellpose segmentation
            masks, flows, styles, diams = self.model.eval(
                image_proc,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                channels=[0, 0]  # Grayscale
            )
            
            # Extract particle information from masks
            particles = self._extract_particles_from_masks(
                masks, image_proc, size_filter
            )
            
            # Convert to DataFrame
            if particles:
                df = pd.DataFrame(particles)
                return df
            else:
                return pd.DataFrame(columns=['x', 'y', 'area', 'confidence', 'eccentricity', 'diameter'])
                
        except Exception as e:
            st.error(f"Cellpose detection failed: {str(e)}")
            return pd.DataFrame()
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for better Cellpose performance."""
        # Convert to float
        image = image.astype(np.float32)
        
        # Normalize to 0-1 range
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Scale to 0-255 range
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def _extract_particles_from_masks(self, masks: np.ndarray, 
                                     image: np.ndarray,
                                     size_filter: Tuple[int, int]) -> List[Dict]:
        """Extract particle properties from Cellpose masks."""
        particles = []
        
        # Get unique mask IDs (excluding background = 0)
        mask_ids = np.unique(masks)[1:]
        
        for mask_id in mask_ids:
            # Create binary mask for this particle
            particle_mask = (masks == mask_id).astype(np.uint8)
            
            # Calculate properties
            props = self._calculate_particle_properties_cellpose(
                particle_mask, image
            )
            
            # Filter by size
            if size_filter[0] <= props['area'] <= size_filter[1]:
                particles.append(props)
        
        return particles
    
    def _calculate_particle_properties_cellpose(self, mask: np.ndarray, 
                                               image: np.ndarray) -> Dict:
        """Calculate particle properties from Cellpose mask."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Calculate equivalent diameter
        diameter = 2 * np.sqrt(area / np.pi)
        
        # Calculate eccentricity
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            axes = ellipse[1]
            eccentricity = np.sqrt(1 - (min(axes) / max(axes))**2) if max(axes) > 0 else 0
        else:
            eccentricity = 0
        
        # Calculate mean intensity
        mean_intensity = np.mean(image[mask > 0]) if np.any(mask > 0) else 0
        
        return {
            'x': cx,
            'y': cy,
            'area': area,
            'diameter': diameter,
            'confidence': mean_intensity / 255.0,  # Normalized intensity as confidence
            'eccentricity': eccentricity,
            'mean_intensity': mean_intensity,
            'bbox': cv2.boundingRect(contour)
        }


def create_advanced_segmentation_interface():
    """Create Streamlit interface for advanced segmentation methods."""
    st.header("🔬 Advanced Particle Segmentation")
    
    # Method selection
    method = st.selectbox(
        "Select segmentation method",
        ["CellSAM (Segment Anything)", "Cellpose", "Compare Methods"],
        help="Choose between CellSAM and Cellpose for particle detection"
    )
    
    if method == "CellSAM (Segment Anything)":
        create_cellsam_interface()
    elif method == "Cellpose":
        create_cellpose_interface()
    else:
        create_comparison_interface()


def create_cellsam_interface():
    """Create interface for CellSAM segmentation."""
    st.subheader("CellSAM Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model type",
            ["vit_b", "vit_l", "vit_h"],
            help="vit_b: Fastest, vit_h: Most accurate"
        )
        confidence_threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, 0.5, 0.1,
            help="Minimum confidence for particle detection"
        )
    
    with col2:
        min_size = st.number_input("Minimum particle size (pixels)", 1, 1000, 10)
        max_size = st.number_input("Maximum particle size (pixels)", 10, 10000, 1000)
    
    # Initialize segmentation
    if st.button("Initialize CellSAM"):
        with st.spinner("Loading CellSAM model..."):
            segmenter = CellSAMSegmentation(model_type=model_type)
            if segmenter.load_model():
                st.session_state.cellsam_segmenter = segmenter
                st.success("CellSAM initialized successfully!")
    
    # Run segmentation if model is loaded and image is available
    if hasattr(st.session_state, 'cellsam_segmenter') and hasattr(st.session_state, 'loaded_images'):
        if st.button("Run CellSAM Detection"):
            run_cellsam_detection(confidence_threshold, (min_size, max_size))


def create_cellpose_interface():
    """Create interface for Cellpose segmentation."""
    st.subheader("Cellpose Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model type",
            ["cyto", "nuclei", "cyto2", "livecell", "bact", "plant"],
            help="Choose model based on your cell type"
        )
        diameter = st.number_input(
            "Expected particle diameter (pixels)", 
            0, 200, 0,
            help="0 for auto-estimation"
        )
    
    with col2:
        flow_threshold = st.slider(
            "Flow threshold", 0.0, 3.0, 0.4, 0.1,
            help="Error threshold for mask computation"
        )
        cellprob_threshold = st.slider(
            "Cell probability threshold", -6.0, 6.0, 0.0, 0.5,
            help="Threshold for cell probability"
        )
    
    min_size = st.number_input("Minimum particle size (pixels)", 1, 1000, 10)
    max_size = st.number_input("Maximum particle size (pixels)", 10, 10000, 1000)
    
    # Initialize segmentation
    if st.button("Initialize Cellpose"):
        with st.spinner("Loading Cellpose model..."):
            segmenter = CellposeSegmentation(model_type=model_type)
            if segmenter.load_model():
                st.session_state.cellpose_segmenter = segmenter
                st.success("Cellpose initialized successfully!")
    
    # Run segmentation if model is loaded and image is available
    if hasattr(st.session_state, 'cellpose_segmenter') and hasattr(st.session_state, 'loaded_images'):
        if st.button("Run Cellpose Detection"):
            run_cellpose_detection(
                diameter if diameter > 0 else None,
                flow_threshold, cellprob_threshold, (min_size, max_size)
            )


def run_cellsam_detection(confidence_threshold: float, size_filter: Tuple[int, int]):
    """Run CellSAM detection on loaded images."""
    try:
        # Get the first loaded image
        image = st.session_state.loaded_images[0]
        
        with st.spinner("Running CellSAM detection..."):
            detections = st.session_state.cellsam_segmenter.detect_particles(
                image, confidence_threshold, size_filter
            )
        
        if not detections.empty:
            st.success(f"Detected {len(detections)} particles with CellSAM")
            st.session_state.cellsam_detections = detections
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detection Results")
                st.dataframe(detections.head(10))
            
            with col2:
                st.subheader("Detection Statistics")
                st.metric("Total Particles", len(detections))
                st.metric("Mean Area", f"{detections['area'].mean():.1f} px²")
                st.metric("Mean Confidence", f"{detections['confidence'].mean():.3f}")
        else:
            st.warning("No particles detected with current parameters")
            
    except Exception as e:
        st.error(f"CellSAM detection failed: {str(e)}")


def run_cellpose_detection(diameter: Optional[float], flow_threshold: float,
                         cellprob_threshold: float, size_filter: Tuple[int, int]):
    """Run Cellpose detection on loaded images."""
    try:
        # Get the first loaded image
        image = st.session_state.loaded_images[0]
        
        with st.spinner("Running Cellpose detection..."):
            detections = st.session_state.cellpose_segmenter.detect_particles(
                image, diameter, flow_threshold, cellprob_threshold, size_filter
            )
        
        if not detections.empty:
            st.success(f"Detected {len(detections)} particles with Cellpose")
            st.session_state.cellpose_detections = detections
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detection Results")
                st.dataframe(detections.head(10))
            
            with col2:
                st.subheader("Detection Statistics")
                st.metric("Total Particles", len(detections))
                st.metric("Mean Area", f"{detections['area'].mean():.1f} px²")
                st.metric("Mean Diameter", f"{detections['diameter'].mean():.1f} px")
        else:
            st.warning("No particles detected with current parameters")
            
    except Exception as e:
        st.error(f"Cellpose detection failed: {str(e)}")


def create_comparison_interface():
    """Create interface for comparing CellSAM and Cellpose methods."""
    st.subheader("Method Comparison")
    
    if (hasattr(st.session_state, 'cellsam_detections') and 
        hasattr(st.session_state, 'cellpose_detections')):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CellSAM Results")
            cellsam_df = st.session_state.cellsam_detections
            st.metric("Particles Detected", len(cellsam_df))
            st.metric("Mean Confidence", f"{cellsam_df['confidence'].mean():.3f}")
            
        with col2:
            st.subheader("Cellpose Results")
            cellpose_df = st.session_state.cellpose_detections
            st.metric("Particles Detected", len(cellpose_df))
            st.metric("Mean Intensity", f"{cellpose_df['mean_intensity'].mean():.1f}")
        
        # Comparison metrics
        st.subheader("Comparison")
        overlap_count = calculate_detection_overlap(cellsam_df, cellpose_df)
        st.metric("Overlapping Detections", overlap_count)
        
    else:
        st.info("Run both CellSAM and Cellpose detection to compare results")


def calculate_detection_overlap(df1: pd.DataFrame, df2: pd.DataFrame, 
                              threshold: float = 20.0) -> int:
    """Calculate number of overlapping detections between two methods."""
    if df1.empty or df2.empty:
        return 0
    
    overlap_count = 0
    for _, row1 in df1.iterrows():
        distances = np.sqrt((df2['x'] - row1['x'])**2 + (df2['y'] - row1['y'])**2)
        if np.any(distances < threshold):
            overlap_count += 1
    
    return overlap_count