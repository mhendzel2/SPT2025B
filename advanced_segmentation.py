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
import io
import matplotlib.pyplot as plt
import os
import tempfile
import sys

# Try to import SAM3
try:
    # Attempt to import from installed package
    import sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    # Fallback: check if local directory exists and add to path (dev mode)
    if os.path.exists("sam3_lib"):
        sys.path.append("sam3_lib")
        try:
            import sam3
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            SAM3_AVAILABLE = True
        except ImportError:
            SAM3_AVAILABLE = False
    else:
        SAM3_AVAILABLE = False

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

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from cellpose import models
    CellposeModel = models.CellposeModel
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

# Set availability flags based on imports
CELLSAM_AVAILABLE = TORCH_AVAILABLE and OPENCV_AVAILABLE and SAM_AVAILABLE
CELLPOSE_AVAILABLE = TORCH_AVAILABLE and OPENCV_AVAILABLE and 'CellposeModel' in locals()
# SAM3 availability is determined above

class SAM3Segmentation:
    """
    SAM3 (Segment Anything Model 3) implementation for particle detection.
    Supports text-based prompting for concept segmentation.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize SAM3 model wrapper.

        Parameters
        ----------
        device : str
            Device to run on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.loaded = False

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self, checkpoint_path: str) -> bool:
        """
        Load the SAM3 model from a local checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Local path to the SAM3 checkpoint file (.pt)

        Returns
        -------
        bool
            True if model loaded successfully
        """
        try:
            if not SAM3_AVAILABLE:
                st.error("SAM3 dependencies not available.")
                return False

            if not os.path.exists(checkpoint_path):
                st.error(f"Checkpoint file not found at: {checkpoint_path}")
                return False

            st.info(f"Loading SAM3 model from {checkpoint_path} on {self.device}...")

            # Use bfloat16 or float16 for memory efficiency if on CUDA
            # However, build_sam3_image_model doesn't expose dtype directly,
            # but we can cast after loading if needed.
            # For now, we stick to default loading.

            with torch.no_grad():
                self.model = build_sam3_image_model(
                    checkpoint_path=checkpoint_path,
                    device=self.device,
                    eval_mode=True,
                    load_from_HF=False  # We use local path
                )

                self.processor = Sam3Processor(
                    self.model,
                    device=self.device
                )

            self.loaded = True
            st.success(f"SAM3 model loaded successfully on {self.device}")
            return True

        except Exception as e:
            st.error(f"Failed to load SAM3 model: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return False

    def detect_particles(self, image: np.ndarray,
                        prompt: str = "object",
                        confidence_threshold: float = 0.4,
                        size_filter: Tuple[int, int] = (10, 1000)) -> pd.DataFrame:
        """
        Detect particles using SAM3 with text prompt.

        Parameters
        ----------
        image : np.ndarray
            Input image (RGB or Grayscale)
        prompt : str
            Text prompt for detection (e.g., "cell", "nucleus", "particle")
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
            st.error("SAM3 model not loaded.")
            return pd.DataFrame()

        try:
            # Preprocess image to RGB PIL Image
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.ndim == 3:
                    image_rgb = image  # Assume RGB
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")

                pil_image = Image.fromarray(image_rgb)
            else:
                # Assume it's already PIL or compatible
                pil_image = image

            with torch.no_grad():
                # 1. Set Image
                self.processor.set_image(pil_image)

                # 2. Set Confidence Threshold
                # We can set it on the processor, or filter later.
                # The processor has a method set_confidence_threshold which might affect internal logic.
                # But setting it too high might miss things, setting it too low might OOM.
                # Let's use a moderate threshold internally and filter output.
                self.processor.set_confidence_threshold(confidence_threshold)

                # 3. Run Inference with Text Prompt
                # The processor returns state dict with 'masks', 'boxes', 'scores'
                state = {}
                # Note: set_image returns the state, so we should capture it
                state = self.processor.set_image(pil_image, state=state)

                # Run prompt
                results_state = self.processor.set_text_prompt(prompt, state)

                # Extract results
                # masks: (N, H, W) bool
                # boxes: (N, 4) float [x1, y1, x2, y2]
                # scores: (N,) float

                masks = results_state.get("masks")
                boxes = results_state.get("boxes")
                scores = results_state.get("scores")

                if masks is None or len(masks) == 0:
                    return pd.DataFrame(columns=['x', 'y', 'area', 'confidence', 'eccentricity', 'bbox'])

                # Move to CPU/Numpy
                masks_np = masks.cpu().numpy().astype(np.uint8)
                scores_np = scores.cpu().numpy()
                boxes_np = boxes.cpu().numpy()

                particles = []

                for i in range(len(masks_np)):
                    mask = masks_np[i]
                    score = float(scores_np[i])
                    box = boxes_np[i] # [x1, y1, x2, y2]

                    # Calculate properties
                    # We can use opencv on the mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    # Largest contour
                    contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour)

                    # Filter by size
                    if not (size_filter[0] <= area <= size_filter[1]):
                        continue

                    # Centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)

                    # Eccentricity
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        axes = ellipse[1]
                        eccentricity = np.sqrt(1 - (min(axes) / max(axes))**2) if max(axes) > 0 else 0
                    else:
                        eccentricity = 0

                    particles.append({
                        'x': cx,
                        'y': cy,
                        'area': area,
                        'confidence': score,
                        'eccentricity': eccentricity,
                        'bbox': (box[0], box[1], box[2]-box[0], box[3]-box[1]) # x, y, w, h
                    })

                if particles:
                    return pd.DataFrame(particles)
                else:
                    return pd.DataFrame(columns=['x', 'y', 'area', 'confidence', 'eccentricity', 'bbox'])

        except torch.cuda.OutOfMemoryError:
            st.error("CUDA Out of Memory. Try reducing image size or running on CPU.")
            # Optionally clear cache
            torch.cuda.empty_cache()
            return pd.DataFrame()
        except Exception as e:
            st.error(f"SAM3 inference failed: {str(e)}")
            return pd.DataFrame()

    def visualize_detections(self, image: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
        """
        Visualize detected particles on the image.

        Parameters
        ----------
        image : np.ndarray
            Input image
        detections : pd.DataFrame
            Detected particles

        Returns
        -------
        np.ndarray
            Image with visualized detections
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()

        # Normalize if not uint8
        if vis_image.dtype != np.uint8:
            vis_image = ((vis_image - vis_image.min()) / (vis_image.max() - vis_image.min()) * 255).astype(np.uint8)

        # Draw detected particles
        for _, row in detections.iterrows():
            x, y = int(row['x']), int(row['y'])
            radius = int(np.sqrt(row['area'] / np.pi))
            confidence = row['confidence']

            # Color based on confidence (cyan -> blue)
            # SAM3 usually gives good confidence, so let's use a distinct color
            color = (255, 255, 0) # Cyan/Yellow mix (BGR: 0, 255, 255 is yellow in BGR? No BGR 0,255,255 is yellow)
            # Let's use Red for visibility
            color = (0, 0, 255)

            # Draw circle at particle position
            cv2.circle(vis_image, (x, y), radius, color, 2)

            # Draw bounding box if available
            if 'bbox' in row and row['bbox'] is not None:
                bbox = row['bbox']
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x1, y1, w, h = bbox
                    x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                    cv2.rectangle(vis_image, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), 1)

        return vis_image


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
        self.checkpoint_dir = os.path.join(tempfile.gettempdir(), "cellsam_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
                # Download the checkpoint if needed
                checkpoint_path = os.path.join(self.checkpoint_dir, f"sam_{self.model_type}.pth")
                
                if not os.path.exists(checkpoint_path):
                    st.info(f"Downloading SAM {self.model_type} model (this may take a few minutes)...")
                    # Download the file
                    import urllib.request
                    urllib.request.urlretrieve(checkpoint_urls[self.model_type], checkpoint_path)
                
                st.info(f"Loading SAM {self.model_type} model...")
            
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
            Input image
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
                image_rgb = np.stack([image, image, image], axis=-1)
            
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
                if (size_filter[0] <= props.get('area', 0) <= size_filter[1] and 
                    props.get('confidence', 0) >= confidence_threshold):
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
            'bbox': mask_info.get('bbox', (0, 0, 0, 0))
        }

    def visualize_detections(self, image: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
        """
        Visualize detected particles on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        detections : pd.DataFrame
            Detected particles
            
        Returns
        -------
        np.ndarray
            Image with visualized detections
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()
            
        # Normalize if not uint8
        if vis_image.dtype != np.uint8:
            vis_image = ((vis_image - vis_image.min()) / (vis_image.max() - vis_image.min()) * 255).astype(np.uint8)
        
        # Draw detected particles
        for _, row in detections.iterrows():
            x, y = int(row['x']), int(row['y'])
            radius = int(np.sqrt(row['area'] / np.pi))
            confidence = row['confidence']
            
            # Color based on confidence (red->yellow->green)
            color = (
                int(255 * (1 - confidence)),  # B
                int(255 * confidence),        # G
                int(255 * (1 - confidence))   # R
            )
            
            # Draw circle at particle position
            cv2.circle(vis_image, (x, y), radius, color, 2)
            
            # Draw bounding box if available
            if 'bbox' in row and row['bbox'] is not None:
                bbox = row['bbox']
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x1, y1, w, h = bbox
                    cv2.rectangle(vis_image, (x1, y1), (x1 + w, y1 + h), color, 1)
        
        return vis_image


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
            self.model = CellposeModel(model_type=self.model_type, gpu=torch.cuda.is_available())
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

    def visualize_detections(self, image: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
        """
        Visualize detected particles on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        detections : pd.DataFrame
            Detected particles
            
        Returns
        -------
        np.ndarray
            Image with visualized detections
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()
            
        # Normalize if not uint8
        if vis_image.dtype != np.uint8:
            vis_image = ((vis_image - vis_image.min()) / (vis_image.max() - vis_image.min()) * 255).astype(np.uint8)
        
        # Draw detected particles
        for _, row in detections.iterrows():
            x, y = int(row['x']), int(row['y'])
            diameter = int(row['diameter']) if 'diameter' in row else int(2 * np.sqrt(row['area'] / np.pi))
            
            # Use different colors for different particles
            color = (0, 255, 0)  # Green by default
            
            # Draw circle at particle position
            cv2.circle(vis_image, (x, y), diameter // 2, color, 2)
            
            # Add label with diameter
            cv2.putText(vis_image, f"{diameter}px", (x + 5, y + 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image


def create_advanced_segmentation_interface():
    """Create Streamlit interface for advanced segmentation methods."""
    st.header("🔬 Advanced Particle Segmentation")
    
    # Method selection
    method = st.selectbox(
        "Select segmentation method",
        ["SAM3 (Segment Anything 3)", "CellSAM (Segment Anything)", "Cellpose", "Compare Methods"],
        help="Choose between SAM3, CellSAM, and Cellpose for particle detection"
    )
    
    if method == "SAM3 (Segment Anything 3)":
        create_sam3_interface()
    elif method == "CellSAM (Segment Anything)":
        create_cellsam_interface()
    elif method == "Cellpose":
        create_cellpose_interface()
    else:
        create_comparison_interface()

def create_sam3_interface():
    """Create interface for SAM3 segmentation."""
    st.subheader("SAM3 Configuration")

    st.markdown("""
    **SAM3** uses text prompts to find specific concepts (e.g., "cell", "nucleus", "particle")
    or generic objects in the image.
    """)

    col1, col2 = st.columns(2)
    with col1:
        checkpoint_path = st.text_input(
            "Path to SAM3 Checkpoint (.pt)",
            value=os.path.join(os.getcwd(), "sam3_checkpoint.pt"),
            help="Absolute path to the downloaded sam3.pt file"
        )
        text_prompt = st.text_input(
            "Text Prompt",
            value="cell",
            help="What do you want to detect? (e.g., 'cell', 'nucleus', 'mitochondria')"
        )

    with col2:
        confidence_threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, 0.4, 0.05,
            help="Minimum confidence for particle detection"
        )

        min_size = st.number_input("Minimum particle size (pixels)", 1, 1000, 10, key="sam3_min")
        max_size = st.number_input("Maximum particle size (pixels)", 10, 10000, 1000, key="sam3_max")

    # Initialize segmentation
    if st.button("Initialize SAM3"):
        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint file not found at {checkpoint_path}. Please download it from HuggingFace (facebook/sam3) and provide the correct path.")
        else:
            with st.spinner("Loading SAM3 model..."):
                segmenter = SAM3Segmentation()
                if segmenter.load_model(checkpoint_path):
                    st.session_state.sam3_segmenter = segmenter
                    st.session_state.sam3_checkpoint = checkpoint_path
                    # st.success("SAM3 initialized successfully!") # Done in load_model

    # Run segmentation if model is loaded and image is available
    if 'sam3_segmenter' in st.session_state and 'image_data' in st.session_state and st.session_state.image_data is not None:
        if st.button("Run SAM3 Detection"):
            run_sam3_detection(text_prompt, confidence_threshold, (min_size, max_size))

def run_sam3_detection(prompt: str, confidence_threshold: float, size_filter: Tuple[int, int]):
    """Run SAM3 detection on loaded images."""
    try:
        # Get the first loaded image
        image = st.session_state.image_data[0] if isinstance(st.session_state.image_data, list) else st.session_state.image_data

        with st.spinner(f"Running SAM3 detection for '{prompt}'..."):
            detections = st.session_state.sam3_segmenter.detect_particles(
                image, prompt, confidence_threshold, size_filter
            )

        if not detections.empty:
            st.success(f"Detected {len(detections)} particles with SAM3")
            st.session_state.sam3_detections = detections

            # Save detections as a structured format for tracking
            frame = 0  # First frame
            particle_detections = {}
            particle_detections[frame] = []

            for _, row in detections.iterrows():
                particle = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'intensity': float(row.get('confidence', 1.0)),
                    'size': float(np.sqrt(row['area'] / np.pi)),
                    'id': int(row.name) if isinstance(row.name, (int, np.integer)) else 0
                }
                particle_detections[frame].append(particle)

            # Store for use in tracking
            st.session_state.particle_detections = particle_detections

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

            # Visualize detections on the image
            vis_image = st.session_state.sam3_segmenter.visualize_detections(image, detections)
            st.subheader("Visualization")
            st.image(vis_image, caption="Detected Particles (SAM3)", use_container_width=True)

            # Allow using these detections for tracking
            st.info("These detections can now be used for tracking in the Tracking tab.")
        else:
            st.warning("No particles detected with current parameters. Try adjusting the confidence threshold or prompt.")

    except Exception as e:
        st.error(f"SAM3 detection failed: {str(e)}")


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
    if 'cellsam_segmenter' in st.session_state and 'image_data' in st.session_state and st.session_state.image_data is not None:
        if st.button("Run CellSAM Detection"):
            run_cellsam_detection(confidence_threshold, (min_size, max_size))


def run_cellsam_detection(confidence_threshold: float, size_filter: Tuple[int, int]):
    """Run CellSAM detection on loaded images."""
    try:
        # Get the first loaded image
        image = st.session_state.image_data[0] if isinstance(st.session_state.image_data, list) else st.session_state.image_data
        
        with st.spinner("Running CellSAM detection..."):
            detections = st.session_state.cellsam_segmenter.detect_particles(
                image, confidence_threshold, size_filter
            )
        
        if not detections.empty:
            st.success(f"Detected {len(detections)} particles with CellSAM")
            st.session_state.cellsam_detections = detections
            
            # Save detections as a structured format for tracking
            frame = 0  # First frame
            particle_detections = {}
            particle_detections[frame] = []
            
            for _, row in detections.iterrows():
                particle = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'intensity': float(row.get('confidence', 1.0)),
                    'size': float(np.sqrt(row['area'] / np.pi)),
                    'id': int(row.name)  # Use DataFrame index as particle ID
                }
                particle_detections[frame].append(particle)
            
            # Store for use in tracking
            st.session_state.particle_detections = particle_detections
            
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
            
            # Visualize detections on the image
            vis_image = st.session_state.cellsam_segmenter.visualize_detections(image, detections)
            st.subheader("Visualization")
            st.image(vis_image, caption="Detected Particles", use_container_width=True)
            
            # Allow using these detections for tracking
            st.info("These detections can now be used for tracking in the Tracking tab.")
        else:
            st.warning("No particles detected with current parameters")
            
    except Exception as e:
        st.error(f"CellSAM detection failed: {str(e)}")


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
    
    df1_coords = df1[['x', 'y']].values
    df2_coords = df2[['x', 'y']].values
    
    # Calculate all pairwise distances using broadcasting
    distances = np.sqrt(np.sum((df1_coords[:, np.newaxis, :] - df2_coords[np.newaxis, :, :]) ** 2, axis=2))
    
    # Count rows in df1 that have at least one point in df2 within threshold
    overlap_count = np.sum(np.any(distances < threshold, axis=1))
    
    return overlap_count


def get_available_segmenters():
    """
    Get list of available segmentation methods based on installed dependencies.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary of available segmentation methods and their status
    """
    available = {
        'SAM3': SAM3_AVAILABLE,
        'CellSAM': CELLSAM_AVAILABLE,
        'Cellpose': CELLPOSE_AVAILABLE
    }
    
    return available


# Function to integrate with app.py
def integrate_advanced_segmentation_with_app():
    """
    Integration function for app.py to call and render advanced segmentation options.
    """
    # Check for available segmentation methods
    available_methods = get_available_segmenters()
    
    if not any(available_methods.values()):
        st.warning("No advanced segmentation methods are available. Please install required packages.")
        
        with st.expander("Installation Instructions"):
            st.markdown("""
            ### Required packages:
            
            For SAM3:
            ```
            See installation instructions for facebookresearch/sam3
            ```

            For CellSAM:
            ```
            pip install torch segment-anything
            ```
            
            For Cellpose:
            ```
            pip install cellpose
            ```
            """)
        return
    
    # Create the interface
    create_advanced_segmentation_interface()


def export_segmentation_results():
    """Export segmentation results to CSV and images."""
    if 'cellsam_detections' in st.session_state:
        detections = st.session_state.cellsam_detections
        
        # Export to CSV
        csv_buffer = io.StringIO()
        detections.to_csv(csv_buffer)
        
        st.download_button(
            label="Download Detections as CSV",
            data=csv_buffer.getvalue(),
            file_name="cellsam_detections.csv",
            mime="text/csv"
        )
        
        # Export visualization
        if 'image_data' in st.session_state and st.session_state.image_data is not None:
            image = st.session_state.image_data[0] if isinstance(st.session_state.image_data, list) else st.session_state.image_data
            
            vis_image = st.session_state.cellsam_segmenter.visualize_detections(image, detections)
            
            # Convert visualization to PNG
            from io import BytesIO
            from PIL import Image as PILImage
            
            img_buffer = BytesIO()
            pil_image = PILImage.fromarray(vis_image)
            pil_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            st.download_button(
                label="Download Visualization as PNG",
                data=img_buffer.getvalue(),
                file_name="cellsam_visualization.png",
                mime="image/png"
            )
