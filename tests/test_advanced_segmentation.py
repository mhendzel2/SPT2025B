import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Try to import the advanced segmentation classes, but don't fail if the
# optional dependencies are not installed.
try:
    from advanced_segmentation import CellSAMSegmentation, CellposeSegmentation
    ADVANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ADVANCED_SEGMENTATION_AVAILABLE = False

@unittest.skipIf(not ADVANCED_SEGMENTATION_AVAILABLE, "Advanced segmentation dependencies not installed.")
class TestAdvancedSegmentation(unittest.TestCase):

    def setUp(self):
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[10:30, 10:30, :] = 150
        self.test_image[50:80, 50:80, :] = 200

    @patch('advanced_segmentation.sam_model_registry')
    @patch('advanced_segmentation.SamPredictor')
    def test_cellsam_segmentation(self, MockSamPredictor, mock_sam_registry):
        """Test CellSAM segmentation class."""
        # Mock the SAM model and predictor
        mock_sam_registry.return_value = MagicMock()
        MockSamPredictor.return_value = MagicMock()

        segmenter = CellSAMSegmentation(model_type="vit_b", device="cpu")
        segmenter.model = MagicMock()
        segmenter.predictor = MockSamPredictor()
        segmenter.loaded = True

        # Mock the detection results
        mock_masks = [{
            'segmentation': self.test_image[:,:,0] > 0,
            'stability_score': 0.9,
            'bbox': (10, 10, 20, 20)
        }]

        with patch.object(segmenter, '_create_mask_generator') as mock_mask_gen:
            mock_mask_gen.return_value.generate.return_value = mock_masks
            detections = segmenter.detect_particles(self.test_image)

        self.assertIsInstance(detections, pd.DataFrame)
        self.assertEqual(len(detections), 1)
        self.assertIn('x', detections.columns)
        self.assertIn('y', detections.columns)
        self.assertIn('area', detections.columns)

    @patch('advanced_segmentation.CellposeModel')
    def test_cellpose_segmentation(self, MockCellposeModel):
        """Test Cellpose segmentation class."""
        # Mock the Cellpose model
        mock_model_instance = MockCellposeModel.return_value
        mock_model_instance.eval.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        segmenter = CellposeSegmentation(model_type="cyto", device="cpu")
        segmenter.model = mock_model_instance
        segmenter.loaded = True

        # Mock the mask extraction to return one particle
        with patch.object(segmenter, '_extract_particles_from_masks', return_value=[{'x': 1, 'y': 1, 'area': 100}]):
            detections = segmenter.detect_particles(self.test_image)

        self.assertIsInstance(detections, pd.DataFrame)
        self.assertEqual(len(detections), 1)
        self.assertIn('x', detections.columns)
        self.assertIn('y', detections.columns)
        self.assertIn('area', detections.columns)

if __name__ == '__main__':
    unittest.main()
