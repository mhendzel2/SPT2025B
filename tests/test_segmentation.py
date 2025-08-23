import unittest
import numpy as np
import pandas as pd
from segmentation import (
    segment_image_channel_otsu,
    segment_image_channel_simple_threshold,
    segment_image_channel_adaptive_threshold,
    gaussian_mixture_segmentation,
    bayesian_gaussian_mixture_segmentation,
)

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        # Create a simple test image with two objects
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        self.test_image[10:30, 10:30] = 150  # Object 1
        self.test_image[50:80, 50:80] = 200  # Object 2
        self.pixel_size = 0.1
        # Create a ROI mask that includes both objects
        self.roi_mask = np.zeros((100, 100), dtype=bool)
        self.roi_mask[10:30, 10:30] = True
        self.roi_mask[50:80, 50:80] = True


    def test_segment_image_channel_otsu(self):
        """Test Otsu segmentation."""
        compartments = segment_image_channel_otsu(
            self.test_image, min_object_size=50, pixel_size=self.pixel_size
        )
        # Should detect 2 objects
        self.assertEqual(len(compartments), 2)
        # Check that the areas are roughly correct
        # Object 1 is 20x20 = 400 pixels. Object 2 is 30x30 = 900 pixels
        areas = sorted([comp['area_pixels'] for comp in compartments])
        self.assertAlmostEqual(areas[0], 400, delta=20)
        self.assertAlmostEqual(areas[1], 900, delta=30)

    def test_segment_image_channel_simple_threshold(self):
        """Test simple threshold segmentation."""
        compartments = segment_image_channel_simple_threshold(
            self.test_image, threshold_value=100, min_object_size=50, pixel_size=self.pixel_size
        )
        self.assertEqual(len(compartments), 2)

    def test_segment_image_channel_adaptive_threshold(self):
        """Test adaptive threshold segmentation."""
        # Adaptive thresholding might merge the objects, so we check for at least one object
        compartments = segment_image_channel_adaptive_threshold(
            self.test_image, block_size=51, min_object_size=50, pixel_size=self.pixel_size
        )
        self.assertGreater(len(compartments), 0)

    def test_gaussian_mixture_segmentation(self):
        """Test Gaussian Mixture Model segmentation."""
        # Test with 3 components (background + 2 objects)
        result = gaussian_mixture_segmentation(
            self.test_image, roi_mask=self.roi_mask, max_components=3, min_components=3
        )
        self.assertIn('classes', result)
        self.assertEqual(result['optimal_n_components'], 3)
        # The number of classified pixels should be the sum of the object areas
        self.assertAlmostEqual(np.sum(result['classes'] > 0), 1300, delta=50)

    def test_bayesian_gaussian_mixture_segmentation(self):
        """Test Bayesian Gaussian Mixture Model segmentation."""
        result = bayesian_gaussian_mixture_segmentation(
            self.test_image, roi_mask=self.roi_mask, max_components=5
        )
        self.assertIn('classes', result)
        # The number of effective components should be around 3
        self.assertIn(result['n_effective_components'], [2, 3, 4])
        self.assertAlmostEqual(np.sum(result['classes'] > 0), 1300, delta=50)

if __name__ == '__main__':
    unittest.main()
