import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation import DiffusionSimulator

class TestDiffusionSimulator(unittest.TestCase):

    def setUp(self):
        """Set up a new DiffusionSimulator instance for each test."""
        self.simulator = DiffusionSimulator()

    def test_simulation_run(self):
        """Test that a simple simulation runs without errors and produces output."""
        # Create a simple mask (e.g., a 100x100x100 box)
        mask = np.zeros((100, 100, 100), dtype=np.uint8)
        mask[10:90, 10:90, 10:90] = 1  # A smaller box inside

        # Load the mask
        self.simulator.boundary_map = (mask == 0).astype(np.uint8)
        self.simulator.region_map = mask.astype(np.uint8)

        # Run a short simulation
        results = self.simulator.run_multi_particle_simulation(
            particle_diameters=[10.0],
            mobility=1.0,
            num_steps=100,
            num_particles_per_size=5
        )

        # Check that results are produced
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self.assertEqual(len(results), 5)
        self.assertIn('diffusion_coefficient', results.columns)

    def test_no_valid_start_position(self):
        """Test that the simulation handles cases where no valid starting position is found."""
        # Create a mask that is all boundary
        mask = np.ones((10, 10, 10), dtype=np.uint8)

        # Load the mask
        self.simulator.boundary_map = mask.astype(np.uint8)
        self.simulator.region_map = np.zeros_like(mask).astype(np.uint8)

        # Run the simulation
        results = self.simulator.run_multi_particle_simulation(
            particle_diameters=[10.0],
            mobility=1.0,
            num_steps=100,
            num_particles_per_size=1
        )

        # Check that the simulation fails gracefully (results with NaN)
        self.assertTrue(results['diffusion_coefficient'].isna().all())

if __name__ == '__main__':
    unittest.main()
